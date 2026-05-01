"""Module for managing ETL operations for type 2 dimensions in a data warehouse."""

from typing import override

import polars as pl

from easydw.dimension.generic import Dimension
from easydw.logging import get_logger

logger = get_logger()


class DimensionType2(Dimension):
    """Dimension of type 2.

    Existing records are closed (deactivated) and new records are inserted with
    the new values. This is used for dimensions that can change over time, and the
    history of changes is needed.

    :param name: Name of the dimension
    :type name: str
    :param dwh: Data warehouse connection
    :type dwh: Database
    """

    class Constants:
        """Constants for DimensionType2."""

        CREATION_DATE = "creation_date"
        DEACTIVATION_DATE = "deactivation_date"
        CURRENT_RECORD = "current_record"

    def _get_required_scd2_columns(self) -> list[tuple[str, pl.DataType]]:
        return [
            (self.Constants.CREATION_DATE, pl.Datetime(time_zone=self.timezone)),
            (self.Constants.DEACTIVATION_DATE, pl.Datetime(time_zone=self.timezone)),
            (self.Constants.CURRENT_RECORD, pl.Boolean),
        ]

    def _normalize_datetime_expr(self, dwh_df: pl.DataFrame, column: str) -> pl.Expr:
        expected_dtype = pl.Datetime(time_zone=self.timezone)
        column_dtype = dwh_df.schema[column]

        if isinstance(column_dtype, pl.Datetime):
            if column_dtype.time_zone is None:
                return pl.col(column).dt.replace_time_zone(self.timezone).cast(
                    expected_dtype
                )

            if column_dtype.time_zone != self.timezone:
                return pl.col(column).dt.convert_time_zone(self.timezone).cast(
                    expected_dtype
                )

        return pl.col(column).cast(expected_dtype)

    def _validate_and_cast_scd2_columns(self, dwh_df: pl.DataFrame) -> pl.DataFrame:
        required_columns = self._get_required_scd2_columns()
        missing_columns = [
            column for column, _ in required_columns if column not in dwh_df.columns
        ]
        if missing_columns:
            msg = (
                "Missing required SCD type 2 columns in warehouse dataframe: "
                f"{missing_columns}. Maybe this is not a type 2 dimension."
            )
            raise ValueError(msg)

        datetime_columns = {
            self.Constants.CREATION_DATE,
            self.Constants.DEACTIVATION_DATE,
        }

        return dwh_df.with_columns(
            [
                self._normalize_datetime_expr(dwh_df, column)
                if column in datetime_columns
                else pl.col(column).cast(dtype)
                for column, dtype in required_columns
            ]
        )

    def _identify_existing_records(
        self, df: pl.DataFrame, dwh_df: pl.DataFrame, keys: list[str]
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        dwh_df = dwh_df.filter(pl.col(self.Constants.CURRENT_RECORD))

        if "id" in dwh_df.columns:
            dwh_df = dwh_df.drop("id")

        if dwh_df.is_empty():
            return pl.DataFrame(), pl.DataFrame()

        merged_df = df.join(dwh_df, on=keys, how="inner", suffix="_dwh")

        dimension_columns = [
            column
            for column in dwh_df.columns
            if column not in keys
            and column not in vars(self.Constants).values()
            and column != "id"
        ]

        if not dimension_columns:
            return pl.DataFrame(), pl.DataFrame()

        comparison_exprs = [
            pl.when(pl.col(col).is_null() & pl.col(f"{col}_dwh").is_null())
            .then(statement=True)
            .otherwise(pl.col(col) != pl.col(f"{col}_dwh"))
            for col in dimension_columns
        ]

        changed_records = merged_df.filter(pl.any_horizontal(comparison_exprs))

        old_records = dwh_df.join(changed_records.select(keys), on=keys, how="semi")

        updated_records = df.join(old_records.select(keys), on=keys, how="semi")

        return old_records, updated_records

    @override
    def insert(self, df: pl.DataFrame, keys: list[str]) -> None:
        """Load the records in the data warehouse.

        For type 2 dimensions, existing records that have been updated are closed
        (deactivated) and the new records are inserted with the new values.
        :param df: Data to load
        :type df: pl.DataFrame
        :param keys: Columns to identify unique records
        :type keys: list[str]
        """
        logger.info("Updating %s", self.name)

        dwh_df = self.extract()
        dwh_df = self._validate_and_cast_scd2_columns(dwh_df)

        new_records = self._identify_new_records(df, dwh_df, keys)
        old_records, updated_records = self._identify_existing_records(df, dwh_df, keys)

        if old_records.is_empty():
            logger.info("No old records to update in %s", self.name)
        else:
            old_records = old_records.with_columns(
                [
                    pl.lit(
                        self._get_now()
                    ).alias(self.Constants.DEACTIVATION_DATE),
                    pl.lit(value=False).alias(self.Constants.CURRENT_RECORD),
                ]
            )

            self.dwh.update(old_records, self.name, keys)
            logger.info("%s : closed %d records", self.name, old_records.height)

        if not updated_records.is_empty():
            new_records = pl.concat([new_records, updated_records], how="vertical")

        if new_records.is_empty():
            logger.info("No new records to load in %s", self.name)
        else:
            new_records = new_records.with_columns(
                [
                    pl.lit(
                        self._get_now()
                    ).alias(self.Constants.CREATION_DATE),
                    pl.lit(None).alias(self.Constants.DEACTIVATION_DATE),
                    pl.lit(value=True).alias(self.Constants.CURRENT_RECORD),
                ]
            )

            self.dwh.insert(new_records, self.name)
            logger.info("%s : added %d records", self.name, new_records.height)
