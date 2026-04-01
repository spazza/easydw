"""Database module for interacting a smoother way with the database."""

from abc import ABC, abstractmethod

import polars as pl
from sqlalchemy import (
    Engine,
    MetaData,
    Table,
    and_,
    select,
    text,
    update,
    values,
)
from sqlalchemy import (
    insert as insert_,
)
from sqlalchemy.exc import SQLAlchemyError

from easydw.logging import get_logger

logger = get_logger()


class Database(ABC):
    """Abstract base class for database interactions.

    Provides methods for connecting, selecting, inserting, updating, and upserting
    data in a database.
    """

    MAX_PORT: int = 65535

    def __init__(self, name: str, params: dict[str]) -> None:
        """Initialize the Database object with connection parameters.

        :param name: Name of the database
        :type name: str
        :param params: Dictionary containing the connection parameters
        :type params: dict[str]
        """
        self.name = name
        self.params = params
        self.schema: str | None = params.get("schema")
        self.engine: Engine | None = None
        self._table_cache: dict[str, Table] = {}

    def _get_table(self, table_name: str) -> Table:
        """Return cached Table objects to avoid repeated reflection."""
        cache_key = f"{self.schema}.{table_name}" if self.schema else table_name

        if cache_key not in self._table_cache:
            metadata = MetaData()
            self._table_cache[cache_key] = Table(
                table_name,
                metadata,
                schema=self.schema,
                autoload_with=self.engine,
            )
        return self._table_cache[cache_key]

    def clear_cache(self) -> None:
        """Clear the table metadata cache."""
        self._table_cache.clear()

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection with the database."""

    def is_connected(self) -> bool:
        """Check whether the database is reachable.

        This validates both that an engine exists and that the database can
        respond to a lightweight query.

        :return: True if the database is reachable, False otherwise
        :rtype: bool
        """
        if self.engine is None:
            return False

        try:
            with self.engine.connect() as connection:
                connection.execute(select(1))
        except SQLAlchemyError:
            logger.exception("Database connectivity check failed.")
            return False
        else:
            return True

    def get_engine(self) -> Engine:
        """Return the engine object.

        :return: Engine object
        :rtype: Engine
        """
        return self.engine

    def select(
        self,
        table_name: str | None = None,
        query: str | None = None,
        params: dict | None = None,
    ) -> pl.DataFrame:
        """Select data from a table or query and return it as a DataFrame.

        Exactly one of `table_name` or `query` must be provided.

        :param table_name: Name of the table to select data from
        :param query: SQL query to execute
        :param params: Parameters to pass to the query
        :return: DataFrame with the selected data
        :rtype: pl.DataFrame
        """
        if not self.is_connected():
            msg = "Database not connected. Call connect() first."
            raise RuntimeError(msg)

        if bool(table_name) == bool(query):
            msg = "Provide exactly one of table_name or query."
            raise ValueError(msg)

        try:
            with self.engine.connect() as connection:
                if query:
                    if params is not None and not isinstance(params, dict):
                        msg = "params must be a dict."
                        raise TypeError(msg)
                    stmt = text(query).bindparams(**params) if params else text(query)
                else:
                    table = self._get_table(table_name)
                    stmt = select(table)

                return pl.read_database(stmt, connection)

        except SQLAlchemyError:
            logger.exception("Exception occurred while selecting data.")
            return pl.DataFrame()

    def insert(self, df: pl.DataFrame, table_name: str) -> int | None:
        """Insert data from a DataFrame into a table in the database.

        :param df: DataFrame with the data to insert
        :type df: pl.DataFrame
        :param table_name: Name of the table to insert the data into
        :type table_name: str
        :return: Number of rows inserted or None on error
        :rtype: int | None
        """
        if not self.engine:
            msg = "Database not connected. Call connect() first."
            raise RuntimeError(msg)
        if df.is_empty():
            return 0

        table = self._get_table(table_name)

        try:
            with self.engine.connect() as connection, connection.begin():
                data = df.to_dicts()
                stmt = insert_(table)
                result = connection.execute(stmt, data)
                return result.rowcount if result.rowcount is not None else 0

        except SQLAlchemyError:
            logger.exception("Exception occurred while inserting data into table.")
            return None

    def update(self, df: pl.DataFrame, table_name: str, keys: list[str]) -> int | None:
        """Update data in a table with data from a DataFrame (batch operation).

        :param df: DataFrame with the data to update
        :param table_name: Name of the table to update
        :param keys: Columns that define the update condition
        :return: Number of rows updated or None on error
        """
        if not self.is_connected():
            msg = "Database not connected. Call connect() first."
            raise RuntimeError(msg)
        if not keys:
            logger.error("update() requires at least one key column")
            return None

        data = df.to_dicts()

        if not data:
            return 0

        try:
            with self.engine.connect() as connection, connection.begin():
                table = self._get_table(table_name)

                columns = list(data[0].keys())
                update_columns = [col for col in columns if col not in keys]

                if not update_columns:
                    logger.warning("No columns to update (all are key columns)")
                    return 0

                values_rows = [tuple(record[col] for col in columns) for record in data]
                values_table = values(*columns).data(values_rows).alias("v")

                stmt = (
                    update(table)
                    .where(and_(*[table.c[key] == values_table.c[key] for key in keys]))
                    .values({col: values_table.c[col] for col in update_columns})
                )

                result = connection.execute(stmt)
                return result.rowcount if result.rowcount is not None else 0

        except SQLAlchemyError:
            logger.exception("Exception occurred while updating data in table.")
            return None

    @abstractmethod
    def upsert(self, df: pl.DataFrame, table_name: str, keys: list[str]) -> int | None:
        """Upsert data from a DataFrame into a table in the database.

        :param df: DataFrame with the data to insert
        :param table_name: Name of the table to insert the data into
        :param keys: Columns that define conflicts (e.g., unique keys)
        :return: Number of rows affected or None on error
        """

    def dispose(self) -> None:
        """Dispose the engine object and close the connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
        self.clear_cache()
