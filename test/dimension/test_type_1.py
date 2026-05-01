"""Module for testing the dimension type 1 functionalities."""

from datetime import datetime, timezone
from typing import override
from unittest.mock import Mock, patch

import polars as pl
from polars.testing import assert_frame_equal

from easydw.dimension import DimensionType1


class _TestableDimensionType1(DimensionType1):
    """Concrete test double for `DimensionType1`."""

    @override
    def bind(self, df: pl.DataFrame) -> None:
        """No-op bind implementation for abstract interface compliance in tests."""


def test_insert_empty_table() -> None:
    """Test the insertion in a empty table.

    In this case, the table is empty, so all records from the DataFrame
    should be inserted.
    """
    mock_db = Mock()
    mock_db.select.return_value = pl.DataFrame(
        schema={
            "key-column": pl.Int64,
            "column-1": pl.Int64,
            "column-2": pl.Utf8,
        }
    )
    mock_db.insert.return_value = 5

    test_df = pl.DataFrame(
        {
            "key-column": [10, 20, 30, 40, 50],
            "column-1": [1, 2, 3, 4, 5],
            "column-2": ["a", "b", "c", "d", "e"],
        }
    )

    test_dimension = _TestableDimensionType1(
        name="TestDimension",
        dwh=mock_db,
    )
    test_dimension.insert(test_df, keys=["key-column"])

    expected_df = pl.DataFrame(
        {
            "key-column": [10, 20, 30, 40, 50],
            "column-1": [1, 2, 3, 4, 5],
            "column-2": ["a", "b", "c", "d", "e"],
            "update_date": [None] * 5,
        }
    )
    result_df = mock_db.insert.call_args[0][0]

    assert mock_db.insert.call_count == 1
    assert_frame_equal(result_df, expected_df)


def test_insert_full_table_all_overlaps() -> None:
    """Test the insertion but all records in the data warehouse already exist.

    In this case, all the records have to be updated.
    """
    mock_datetime = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    with (
        patch("easydw.dimension.generic.datetime") as mock_datetime_module,
    ):
        mock_datetime_module.now.return_value = mock_datetime
        mock_db = Mock()
        mock_db.select.return_value = pl.DataFrame(
            {
                "key-column": [10, 20, 30, 40, 50],
                "column-1": [1, 2, 3, 4, 5],
                "column-2": ["a", "b", "c", "d", "e"],
                "update_date": [None] * 5,
            }
        )
        mock_db.update.return_value = 5

        test_df = pl.DataFrame(
            {
                "key-column": [10, 20, 30, 40, 50],
                "column-1": [6, 7, 8, 9, 10],
                "column-2": ["f", "g", "h", "i", "j"],
            }
        )

        test_dimension = _TestableDimensionType1(
            name="TestDimension",
            dwh=mock_db,
        )
        test_dimension.insert(test_df, keys=["key-column"])

        expected_df = pl.DataFrame(
            {
                "key-column": [10, 20, 30, 40, 50],
                "column-1": [6, 7, 8, 9, 10],
                "column-2": ["f", "g", "h", "i", "j"],
                "update_date": [mock_datetime] * 5,
            }
        )
        result_df = mock_db.update.call_args[0][0]

        assert mock_db.update.call_count == 1
        assert mock_db.insert.call_count == 0
        assert_frame_equal(result_df, expected_df)


def test_insert_full_table_with_overlaps() -> None:
    """Test the insertion with some records that are already in the data warehouse.

    In this case, existing records should be updated and new records should be inserted.
    """
    mock_datetime = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    with (
        patch("easydw.dimension.generic.datetime") as mock_datetime_module,
    ):
        mock_datetime_module.now.return_value = mock_datetime
        mock_db = Mock()
        mock_db.select.return_value = pl.DataFrame(
            {
                "key-column": [10, 20, 30, 40, 50],
                "column-1": [1, 2, 3, 4, 5],
                "column-2": ["a", "b", "c", "d", "e"],
                "update_date": [None] * 5,
            }
        )
        mock_db.update.return_value = 3
        mock_db.insert.return_value = 2

        test_df = pl.DataFrame(
            {
                "key-column": [10, 20, 30, 60, 70],
                "column-1": [6, 7, 8, 9, 10],
                "column-2": ["f", "g", "h", "i", "j"],
            }
        )

        test_dimension = _TestableDimensionType1(
            name="TestDimension",
            dwh=mock_db,
        )
        test_dimension.insert(test_df, keys=["key-column"])

        expected_update_df = pl.DataFrame(
            {
                "key-column": [10, 20, 30],
                "column-1": [6, 7, 8],
                "column-2": ["f", "g", "h"],
                "update_date": [mock_datetime] * 3,
            }
        )
        result_update_df = mock_db.update.call_args[0][0].sort("key-column")

        expected_insert_df = pl.DataFrame(
            {
                "key-column": [60, 70],
                "column-1": [9, 10],
                "column-2": ["i", "j"],
                "update_date": [None] * 2,
            }
        )
        result_insert_df = mock_db.insert.call_args[0][0].sort("key-column")

        assert mock_db.update.call_count == 1
        assert_frame_equal(result_update_df, expected_update_df.sort("key-column"))

        assert mock_db.insert.call_count == 1
        assert_frame_equal(result_insert_df, expected_insert_df.sort("key-column"))


def test_insert_full_table_no_overlaps() -> None:
    """Test the insertion with no records already in the data warehouse.

    In this case, all records should be inserted.
    """
    mock_db = Mock()
    mock_db.select.return_value = pl.DataFrame(
        {
            "key-column": [10, 20, 30, 40, 50],
            "column-1": [1, 2, 3, 4, 5],
            "column-2": ["a", "b", "c", "d", "e"],
            "update_date": [None] * 5,
        }
    )
    mock_db.insert.return_value = 3

    test_df = pl.DataFrame(
        {
            "key-column": [60, 70, 80],
            "column-1": [6, 7, 8],
            "column-2": ["f", "g", "h"],
        }
    )

    test_dimension = _TestableDimensionType1(
        name="TestDimension",
        dwh=mock_db,
    )
    test_dimension.insert(test_df, keys=["key-column"])

    expected_df = pl.DataFrame(
        {
            "key-column": [60, 70, 80],
            "column-1": [6, 7, 8],
            "column-2": ["f", "g", "h"],
            "update_date": [None] * 3,
        }
    )
    result_df = mock_db.insert.call_args[0][0]

    assert mock_db.insert.call_count == 1
    assert mock_db.update.call_count == 0
    assert_frame_equal(result_df, expected_df)


def test_insert_full_table_with_overlaps_no_changes() -> None:
    """Test the insertion with overlapping records and partial changes.

    In this case, only changed records should be updated; unchanged ones should
    not be updated.
    """
    mock_datetime = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    with (
        patch("easydw.dimension.generic.datetime") as mock_datetime_module,
    ):
        mock_datetime_module.now.return_value = mock_datetime
        mock_db = Mock()
        mock_db.select.return_value = pl.DataFrame(
            {
                "key-column": [10, 20, 30, 40, 50],
                "column-1": [1, 2, 3, 4, 5],
                "column-2": ["a", "b", "c", "d", "e"],
                "update_date": [None] * 5,
            }
        )

        test_df = pl.DataFrame(
            {
                "key-column": [10, 20, 30, 40, 50],
                "column-1": [1, 2, 3, 9, 10],
                "column-2": ["a", "b", "c", "i", "j"],
            }
        )

        test_dimension = _TestableDimensionType1(
            name="TestDimension",
            dwh=mock_db,
        )
        test_dimension.insert(test_df, keys=["key-column"])

        expected_df = pl.DataFrame(
            {
                "key-column": [40, 50],
                "column-1": [9, 10],
                "column-2": ["i", "j"],
                "update_date": [mock_datetime] * 2,
            }
        )
        result_df = mock_db.update.call_args[0][0].sort("key-column")

        assert mock_db.update.call_count == 1
        assert mock_db.insert.call_count == 0
        assert_frame_equal(result_df, expected_df.sort("key-column"))
