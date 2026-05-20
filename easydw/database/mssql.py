"""Microsoft SQL Server connection helpers."""

import re
from typing import override
from urllib.parse import quote_plus

import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from easydw.database.database import Database
from easydw.logging import get_logger

logger = get_logger()

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str) -> None:
    """Raise ValueError if *name* is not a safe SQL identifier."""
    if not _SAFE_IDENTIFIER_RE.match(name):
        msg = f"Unsafe SQL identifier: {name!r}"
        raise ValueError(msg)


class MSSQLDatabase(Database):
    """Microsoft SQL Server-specific database implementation."""

    UPSERT_BATCH_SIZE = 10000

    @override
    def connect(self) -> None:
        """Establish a connection with the SQL Server database."""
        connection_string = self._build_mssql_conn()

        logger.info("Trying to establish connection with %s", self.name)
        self.engine = create_engine(connection_string, fast_executemany=True)

        if self.is_connected():
            logger.info("Successfully connected to %s", self.name)
        else:
            logger.error("Failed to connect to %s", self.name)
            msg = f"Could not connect to {self.name}"
            raise ConnectionError(msg)

    def _validate_params(self) -> None:
        required_params = ["user", "password", "host", "database"]
        missing_params = [
            param
            for param in required_params
            if param not in self.params or self.params[param] is None
        ]

        if missing_params:
            msg = (
                "Missing required SQL Server connection "
                f"parameters: {', '.join(missing_params)}"
            )
            raise ValueError(msg)

        if "port" in self.params:
            port = int(self.params["port"])
            if not (0 < port <= self.MAX_PORT):
                msg = f"Port must be between 1 and {self.MAX_PORT}, got {port}"
                raise ValueError(msg)

    def _build_mssql_conn(self) -> str:
        self._validate_params()

        user = self.params.get("user")
        password = self.params.get("password")
        host = self.params.get("host")
        port = self.params.get("port", 1433)
        database = self.params.get("database")
        driver = self.params.get("driver", "ODBC Driver 17 for SQL Server")

        odbc_params = (
            f"DRIVER={{{driver}}};"
            f"SERVER={host},{port};"
            f"DATABASE={database};"
            f"UID={user};"
            f"PWD={quote_plus(password)}"
        )

        return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_params)}"

    @override
    def upsert(self, df: pl.DataFrame, table_name: str, keys: list[str]) -> int | None:
        """Upsert records into SQL Server using a MERGE statement.

        :param df: DataFrame with the data to upsert
        :param table_name: Name of the target table
        :param keys: Columns that define the match condition
        :return: Number of rows affected or None on error
        """
        if not self.engine:
            msg = "Database not connected. Call connect() first."
            raise RuntimeError(msg)
        if not keys:
            logger.error("upsert() requires at least one key column")
            return None
        if df.is_empty():
            return 0

        table = self._get_table(table_name)
        all_columns = [c.name for c in table.columns]
        update_columns = [c for c in all_columns if c not in keys]

        # Validate identifiers before embedding in SQL
        _validate_identifier(table_name)
        for col in all_columns:
            _validate_identifier(col)
        if self.schema:
            _validate_identifier(self.schema)

        # Qualify the target table name with schema if present
        qualified_table = (
            f"[{self.schema}].[{table_name}]" if self.schema else f"[{table_name}]"
        )

        # Build the MERGE SQL using named parameters prefixed with "p_"
        source_cols = ", ".join(f":p_{c} AS [{c}]" for c in all_columns)
        match_conditions = " AND ".join(f"target.[{k}] = source.[{k}]" for k in keys)
        update_set = ", ".join(f"target.[{c}] = source.[{c}]" for c in update_columns)
        insert_cols = ", ".join(f"[{c}]" for c in all_columns)
        insert_vals = ", ".join(f"source.[{c}]" for c in all_columns)

        merge_sql = f"""
            MERGE INTO {qualified_table} AS target
            USING (SELECT {source_cols}) AS source
            ON {match_conditions}
            WHEN MATCHED THEN
                UPDATE SET {update_set}
            WHEN NOT MATCHED BY TARGET THEN
                INSERT ({insert_cols})
                VALUES ({insert_vals});
        """  # noqa: S608  # identifiers are validated by _validate_identifier above

        data = df.to_dicts()

        try:
            with self.engine.connect() as connection, connection.begin():
                affected_rows = 0

                for start in range(0, len(data), self.UPSERT_BATCH_SIZE):
                    batch = data[start : start + self.UPSERT_BATCH_SIZE]
                    payload = [{f"p_{k}": v for k, v in row.items()} for row in batch]

                    logger.info(
                        "Upserting batch of rows %d to %d into %s",
                        start,
                        start + self.UPSERT_BATCH_SIZE,
                        table_name,
                    )

                    for record in payload:
                        result = connection.execute(text(merge_sql), record)
                        rowcount = result.rowcount
                        if rowcount is not None and rowcount > 0:
                            affected_rows += rowcount

                return affected_rows

        except SQLAlchemyError:
            logger.exception("Exception occurred while upserting data into table.")
            return None
