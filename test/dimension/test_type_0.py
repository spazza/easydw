"""Module for testing the dimension type 0 functionalities."""

from typing import override
from unittest.mock import Mock

import polars as pl
from polars.testing import assert_frame_equal

from easydw.dimension import DimensionType0


class TestableDimensionType0(DimensionType0):
    """Concrete test double for `DimensionType0`."""

    @override
    def bind(self, df: pl.DataFrame) -> None:
        """No-op bind implementation for abstract interface compliance in tests."""


def test_insert_empty_table() -> None:
    """Test the insertion of records into an empty table.

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

    test_dimension = TestableDimensionType0(
        name="TestDimension",
        dwh=mock_db,
    )
    test_dimension.insert(test_df, keys=["key-column"])

    expected_df = pl.DataFrame(
        {
            "key-column": [10, 20, 30, 40, 50],
            "column-1": [1, 2, 3, 4, 5],
            "column-2": ["a", "b", "c", "d", "e"],
        }
    )
    result_df = mock_db.insert.call_args[0][0]

    mock_db.select.assert_called_once_with("TestDimension", query=None, params=None)
    assert_frame_equal(result_df, expected_df)


def test_insert_full_table_all_overlaps() -> None:
    """Test the insertion of records into a table where all records already exist.

    In this case, all records in the DataFrame already exist in the table, so no
    records should be inserted, and the insert method should not be called.
    """
    mock_db = Mock()
    mock_db.select.return_value = pl.DataFrame(
        {
            "key-column": [10, 20, 30, 40, 50],
            "column-1": [1, 2, 3, 4, 5],
            "column-2": ["a", "b", "c", "d", "e"],
        }
    )
    mock_db.insert.return_value = 0

    test_df = pl.DataFrame(
        {
            "key-column": [10, 20, 30, 40, 50],
            "column-1": [1, 2, 3, 4, 5],
            "column-2": ["a", "b", "c", "d", "e"],
        }
    )

    test_dimension = TestableDimensionType0(
        name="TestDimension",
        dwh=mock_db,
    )
    test_dimension.insert(test_df, keys=["key-column"])

    mock_db.select.assert_called_once_with("TestDimension", query=None, params=None)
    mock_db.insert.assert_not_called()


def test_insert_full_table_with_overlaps() -> None:
    """Test the insertion into a table where a some records already exist.

    In this case, only the new records should be inserted, and the existing records
    should be ignored.
    """
    mock_db = Mock()
    mock_db.select.return_value = pl.DataFrame(
        {
            "key-column": [10, 20, 30],
            "column-1": [1, 2, 3],
            "column-2": ["a", "b", "c"],
        }
    )
    mock_db.insert.return_value = 2

    test_df = pl.DataFrame(
        {
            "key-column": [20, 30, 40, 50],
            "column-1": [2, 3, 4, 5],
            "column-2": ["b", "c", "d", "e"],
        }
    )

    test_dimension = TestableDimensionType0(
        name="TestDimension",
        dwh=mock_db,
    )
    test_dimension.insert(test_df, keys=["key-column"])

    expected_df = pl.DataFrame(
        {"key-column": [40, 50], "column-1": [4, 5], "column-2": ["d", "e"]}
    )
    result_df = mock_db.insert.call_args[0][0]

    mock_db.select.assert_called_once_with("TestDimension", query=None, params=None)
    mock_db.insert.assert_called_once()
    assert_frame_equal(result_df.sort("key-column"), expected_df.sort("key-column"))


def test_insert_full_table_no_overlaps() -> None:
    """Test the insertion where none of the records in the DataFrame already exist.

    In this case, all records from the DataFrame should be inserted.
    """
    mock_db = Mock()
    mock_db.select.return_value = pl.DataFrame(
        {
            "key-column": [1, 2, 3],
            "column-1": [10, 20, 30],
            "column-2": ["x", "y", "z"],
        }
    )
    mock_db.insert.return_value = 3

    test_df = pl.DataFrame(
        {
            "key-column": [4, 5, 6],
            "column-1": [40, 50, 60],
            "column-2": ["a", "b", "c"],
        }
    )

    test_dimension = TestableDimensionType0(
        name="TestDimension",
        dwh=mock_db,
    )
    test_dimension.insert(test_df, keys=["key-column"])

    expected_df = test_df
    result_df = mock_db.insert.call_args[0][0]

    mock_db.select.assert_called_once_with("TestDimension", query=None, params=None)
    mock_db.insert.assert_called_once()
    assert_frame_equal(result_df.sort("key-column"), expected_df.sort("key-column"))
