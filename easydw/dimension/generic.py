"""Module for generic dimensions operations."""

from abc import ABC, abstractmethod

import polars as pl

from easydw.database import Database
from easydw.logging import get_logger

logger = get_logger()


class Dimension(ABC):
    """Base class for dimensions in the data warehouse.

    :param name: Name of the dimension
    :type name: str
    :param dwh: Data warehouse connection
    :type dwh: Database
    """

    def __init__(self, name: str, dwh: Database) -> None:
        """Initialize the Dimension class.

        :param name: Name of the dimension
        :type name: str
        :param dwh: Data warehouse connection
        :type dwh: Database
        """
        self.name = name
        self.dwh = dwh

    def extract(self, query: str | None = None, **kwargs: dict) -> pl.DataFrame:
        """Extract the records from the data warehouse.

        :param query: Optional query to execute
        :type query: str, optional
        :param kwargs: Additional arguments for the query
        :return: Extracted records
        :rtype: pl.DataFrame
        """
        df = self.dwh.select(self.name, query=query, params=kwargs.get("params"))

        logger.info("Extracted %d records", df.height)

        return df

    @abstractmethod
    def insert(self, df: pl.DataFrame, keys: list[str]) -> None:
        """Insert records contained in `df` in the dimension table.

        :param df: Dataframe with records to load
        :type df: pl.DataFrame
        :param keys: Key columns for identifying unique records
        :type keys: list[str]
        """

    def _identify_new_records(
        self, df: pl.DataFrame, dwh_df: pl.DataFrame, keys: list[str]
    ) -> pl.DataFrame:
        # Align types with `df` in case `dwh_df` has different types
        # (e.g., due to NULLs or type inference)

        keys_df = dwh_df.select(keys)
        keys_df = keys_df.with_columns([pl.col(k).cast(df.schema[k]) for k in keys])

        return df.join(keys_df, on=keys, how="anti")

    def _identify_existing_records(
        self, df: pl.DataFrame, dwh_df: pl.DataFrame, keys: list[str]
    ) -> pl.DataFrame:
        # Align types with `df` in case `dwh_df` has different types
        # (e.g., due to NULLs or type inference)

        keys_df = dwh_df.select(keys)
        keys_df = keys_df.with_columns([pl.col(k).cast(df.schema[k]) for k in keys])

        return df.join(keys_df, on=keys, how="semi")

    @abstractmethod
    def bind(self, df: pl.DataFrame) -> None:
        """Bind the records contained in `df` to the dimension table.

        :param df: Dataframe with records to load
        :type df: pl.DataFrame
        """
