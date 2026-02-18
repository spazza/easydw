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
    def insert(self, df: pl.DataFrame, key_columns: list[str]) -> None:
        """Insert records contained in `df` in the dimension table.

        :param df: Dataframe with records to load
        :type df: pl.DataFrame
        :param key_columns: Key columns for identifying unique records
        :type key_columns: list[str]
        """

    def _identify_new_records(
        self, df: pl.DataFrame, dwh_df: pl.DataFrame, key_columns: list[str]
    ) -> pl.DataFrame:
        return df.join(dwh_df.select(key_columns), on=key_columns, how="anti")

    def _identify_existing_records(
        self, df: pl.DataFrame, dwh_df: pl.DataFrame, key_columns: list[str]
    ) -> pl.DataFrame:
        return df.join(dwh_df.select(key_columns), on=key_columns, how="semi")
