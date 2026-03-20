"""PostgreSQL connection helpers."""

from typing import override
from urllib.parse import quote_plus

import polars as pl
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from easydw.database.database import Database
from easydw.logging import get_logger

logger = get_logger()


class PostgresDatabase(Database):
    """PostgreSQL-specific database implementation."""

    UPSERT_BATCH_SIZE = 10000

    @override
    def connect(self) -> None:
        """Establish a connection with the PostgreSQL database."""
        connection_string = self._build_pg_conn()

        logger.info("Trying to establish connection with %s", self.name)
        self.engine = create_engine(connection_string)

    def _validate_params(self) -> None:
        required_params = ["user", "password", "host", "database"]
        missing_params = [
            param
            for param in required_params
            if param not in self.params or self.params[param] is None
        ]

        if missing_params:
            msg = (
                "Missing required PostgreSQL connection "
                f"parameters: {', '.join(missing_params)}"
            )
            raise ValueError(msg)

        if "port" in self.params:
            port = int(self.params["port"])
            if not (0 < port <= self.MAX_PORT):
                msg = f"Port must be between 1 and {self.MAX_PORT}, got {port}"
                raise ValueError(msg)

    def _build_pg_conn(self) -> str:
        self._validate_params()

        user = self.params.get("user")
        password = self.params.get("password")
        host = self.params.get("host")
        port = self.params.get("port", 5432)
        database = self.params.get("database")

        return (
            f"postgresql+psycopg2://"
            f"{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{database}"
        )

    @override
    def upsert(self, df: pl.DataFrame, table_name: str, keys: list[str]) -> int | None:
        """Upsert data from a DataFrame into a table using PostgreSQL ON CONFLICT.

        :param df: DataFrame with the data to insert
        :param table_name: Name of the table to insert the data into
        :param keys: Columns that define conflicts (e.g., unique keys)
        :return: Number of rows affected or None on error
        """
        if not self.engine:
            msg = "Database not connected. Call connect() first."
            raise RuntimeError(msg)
        if not keys:
            logger.error("upsert() requires at least one key column")
            return None

        table = self._get_table(table_name)

        try:
            with self.engine.connect() as connection, connection.begin():
                data = df.to_dicts()

                if not data:
                    return 0

                affected_rows = 0
                for start in range(0, len(data), self.UPSERT_BATCH_SIZE):
                    batch_data = data[start : start + self.UPSERT_BATCH_SIZE]

                    stmt = pg_insert(table).values(batch_data)

                    update_dict = {
                        c.name: stmt.excluded[c.name]
                        for c in table.columns
                        if c.name not in keys and not c.primary_key
                    }

                    stmt = stmt.on_conflict_do_update(
                        index_elements=keys,
                        set_=update_dict,
                    )

                    logger.info(
                        "Upserting batch of rows %d to %d into %s",
                        start,
                        start + self.UPSERT_BATCH_SIZE,
                        table_name,
                    )

                    result = connection.execute(stmt)
                    rowcount = result.rowcount
                    if rowcount is not None and rowcount > 0:
                        affected_rows += rowcount

                return affected_rows
        except SQLAlchemyError:
            logger.exception("Exception occurred while upserting data into table.")
            return None
