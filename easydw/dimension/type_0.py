"""Module for managing ETL operations for type 0 dimensions in a data warehouse."""

import polars as pl

from easydw.dimension.generic import Dimension
from easydw.logging import get_logger

logger = get_logger()


class DimensionType0(Dimension):
    """Dimension of type 0.

    Existing records are not updated, only new records are inserted. This is used for
    dimensions that do not change over time.
    :param name: Name of the dimension
    :type name: str
    :param dwh: Data warehouse connection
    :type dwh: Database
    """

    def insert(self, df: pl.DataFrame, key_columns: list[str]) -> None:
        """Load the records in the data warehouse.

        For type 0 dimensions,
        existing records are not updated, only new records are inserted.
        :param df: Data to load
        :type df: pl.DataFrame
        :param key_columns: Columns to identify unique records
        :type key_columns: list[str]
        """
        logger.info("Updating %s", self.name)

        dwh_df = self.extract()
        new_records = self._identify_new_records(df, dwh_df, key_columns)

        if new_records.is_empty():
            logger.info("No records to load in %s", self.name)
        else:
            self.dwh.insert(new_records, self.name)
            logger.info("%s : added %d records", self.name, new_records.height)
