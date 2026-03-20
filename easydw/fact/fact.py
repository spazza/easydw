"""Generic Fact class for data warehouse I/O operations."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar

import polars as pl

from easydw.database.database import Database
from easydw.logging import get_logger

logger = get_logger()


class Fact(ABC):
    """Class for interacting with fact tables in the data warehouse.

    It provides methods to insert or update records and retrieve data based
    on queries.
    :param name: Name of the fact table
    :type name: str
    :param dwh: Data warehouse connection
    :type dwh: Database
    """

    keys: ClassVar[list[str]] = []
    dimensions: ClassVar[list[object]] = []

    def __init__(self, name: str, dwh: Database) -> None:
        """Initialize the Fact class.

        :param name: Name of the fact table
        :type name: str
        :param dwh: Data warehouse connection
        :type dwh: Database
        """
        self.name = name
        self.dwh = dwh

    def insert(self, from_timestamp: datetime, to_timestamp: datetime) -> None:
        """Insert records into the fact table.

        This method prepares the records and then inserts them in the data warehouse.
        :param from_timestamp: Start timestamp for the records
        :type from_timestamp: datetime
        :param to_timestamp: End timestamp for the records
        :type to_timestamp: datetime
        """
        df = self._prepare_records(from_timestamp, to_timestamp)

        if not df.is_empty():
            if self.keys and df.select(self.keys).is_duplicated().any():
                msg = f"Duplicate keys found in records to insert into {self.name}"
                raise ValueError(msg)

            rows = self.dwh.upsert(df, self.name, keys=self.keys)

            logger.info("Inserted %d records into %s", rows, self.name)
        else:
            logger.info("No records to insert into %s", self.name)

    def get(self, query: str, params: dict) -> pl.DataFrame:
        """Retrieve records from the fact table based on a query.

        :param query: SQL query to execute
        :type query: str
        :param params: Parameters for the SQL query
        :type params: dict
        :return: DataFrame containing the retrieved records
        :rtype: pl.DataFrame
        """
        logger.info("Fetching data with query from %s", self.name)

        df = self.dwh.select(query=query, params=params)

        logger.info("Retrieved %d records from %s", len(df), self.name)

        return df

    @abstractmethod
    def _prepare_records(
        self, from_timestamp: datetime, to_timestamp: datetime
    ) -> pl.DataFrame:
        """Prepare records for insertion in the data warehouse.

        :param from_timestamp: Start timestamp for the records
        :type from_timestamp: datetime
        :param to_timestamp: End timestamp for the records
        :type to_timestamp: datetime
        """

    def _bind_dimensions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Bind dimensions to the DataFrame.

        :param df: DataFrame to bind dimensions to
        :type df: pl.DataFrame
        :return: DataFrame with dimensions bound
        :rtype: pl.DataFrame
        """
        for dimension in self.dimensions:
            dimension_object = dimension(self.dwh)
            df = dimension_object.bind(df)

        return df
