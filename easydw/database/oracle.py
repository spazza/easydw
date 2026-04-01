"""Oracle connection helpers."""

from typing import override
from urllib.parse import quote_plus

import polars as pl
from sqlalchemy import create_engine

from easydw.database.database import Database
from easydw.logging import get_logger

logger = get_logger()


class OracleDatabase(Database):
    """Oracle-specific database implementation."""

    @override
    def connect(self) -> None:
        """Establish a connection with Oracle database."""
        connection_string = self._build_oracle_conn()

        logger.info("Trying to establish connection with %s", self.name)
        self.engine = create_engine(connection_string)

        if self.is_connected():
            logger.info("Successfully connected to %s", self.name)
        else:
            logger.error("Failed to connect to %s", self.name)
            msg = f"Could not connect to {self.name}"
            raise ConnectionError(msg)

    def _validate_params(self) -> None:
        required_params = ["user", "password", "host"]
        missing_params = [
            param
            for param in required_params
            if param not in self.params or self.params[param] is None
        ]

        if missing_params:
            msg = (
                "Missing required Oracle connection "
                f"parameters: {', '.join(missing_params)}"
            )
            raise ValueError(msg)

        if not self.params.get("service") and not self.params.get("sid"):
            msg = "Either 'service' or 'sid' must be provided for Oracle connection"
            raise ValueError(msg)

        if "port" in self.params:
            port = int(self.params["port"])
            if not (0 < port <= self.MAX_PORT):
                msg = f"Port must be between 1 and {self.MAX_PORT}, got {port}"
                raise ValueError(msg)

    def _build_oracle_conn(self) -> str:
        self._validate_params()

        user = self.params.get("user")
        password = self.params.get("password")
        host = self.params.get("host")
        port = self.params.get("port", 1521)
        service = self.params.get("service")
        sid = self.params.get("sid")

        base = (
            f"oracle+oracledb://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}"
        )

        if service:
            return f"{base}/{service}"

        if sid:
            return f"{base}/?sid={sid}"

        msg = "Either service or sid must be provided"
        raise ValueError(msg)

    @override
    def upsert(self, df: pl.DataFrame, table_name: str, keys: list[str]) -> int | None:
        """Upsert records for Oracle by splitting into update and insert sets."""
        if not self.engine:
            msg = "Database not connected. Call connect() first."
            raise RuntimeError(msg)
        if not keys:
            logger.error("upsert() requires at least one key column")
            return None
        if df.is_empty():
            return 0

        existing_keys_df = self.select(table_name=table_name).select(keys).unique()

        rows_to_update = df.join(existing_keys_df, on=keys, how="semi")
        rows_to_insert = df.join(existing_keys_df, on=keys, how="anti")

        updated = self.update(rows_to_update, table_name, keys)
        inserted = self.insert(rows_to_insert, table_name)

        if updated is None or inserted is None:
            logger.error("Oracle upsert failed while processing update or insert phase")
            return None

        return updated + inserted
