"""Module for managing ETL operations for type 1 dimensions in a data warehouse."""

from datetime import datetime
from typing import override

import polars as pl

from easydw.dimension.generic import Dimension
from easydw.logging import get_logger

logger = get_logger()


class DimensionType1(Dimension):
    """Dimension of type 1.

    Existing records are updated with new values, and new records are inserted.
    This is used for dimensions that can change over time, but the history of
    changes is not needed.

    :param name: Name of the dimension
    :type name: str
    :param dwh: Data warehouse connection
    :type dwh: Database
    """

    class Constants:
        """Constants for DimensionType1."""

        UPDATE_DATE = "update_date"

    def _identify_existing_records(
        self, df: pl.DataFrame, dwh_df: pl.DataFrame, keys: list[str]
    ) -> pl.DataFrame:
        if dwh_df.is_empty():
            return pl.DataFrame()

        merged_df = df.join(dwh_df, on=keys, how="inner", suffix="_dwh")

        dimension_columns = [
            column
            for column in dwh_df.columns
            if column not in keys
            and column not in vars(self.Constants).values()
            and column != "id"
        ]

        if not dimension_columns:
            return pl.DataFrame()

        comparison_exprs = [
            pl.when(pl.col(col).is_null() & pl.col(f"{col}_dwh").is_null())
            .then(statement=False)
            .otherwise(pl.col(col) != pl.col(f"{col}_dwh"))
            for col in dimension_columns
        ]

        changed_records = merged_df.filter(pl.any_horizontal(comparison_exprs))

        return df.join(changed_records.select(keys), on=keys, how="semi")

    @override
    def insert(self, df: pl.DataFrame, keys: list[str]) -> None:
        """Load the records in the data warehouse.

        For type 1 dimensions, existing records are updated, and new records are
        inserted.
        :param df: Data to load
        :type df: pl.DataFrame
        :param keys: Columns to identify unique records
        :type keys: list[str]
        """
        logger.info("Updating %s", self.name)

        dwh_df = self.extract()

        updated_records = self._identify_existing_records(df, dwh_df, keys)

        if updated_records.is_empty():
            logger.info("No records to update in %s", self.name)
        else:
            updated_records = updated_records.with_columns(
                pl.lit(datetime.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")).alias(
                    self.Constants.UPDATE_DATE
                )
            )
            self.dwh.update(updated_records, self.name, keys)
            logger.info("%s : updated %d records", self.name, updated_records.height)

        new_records = self._identify_new_records(df, dwh_df, keys)

        if new_records.is_empty():
            logger.info("No records to load in %s", self.name)
        else:
            new_records = new_records.with_columns(
                pl.lit(None).alias(self.Constants.UPDATE_DATE)
            )
            self.dwh.insert(new_records, self.name)
            logger.info("%s : added %d records", self.name, new_records.height)
