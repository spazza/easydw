"""Module for testing the dimension type 2 functionalities."""

from datetime import datetime, timezone
from typing import override
from unittest.mock import Mock, patch

import polars as pl
from polars.testing import assert_frame_equal

from easydw.dimension import DimensionType2


class _TestableDimensionType2(DimensionType2):
    """Concrete test double for `DimensionType2`."""

    @override
    def bind(self, df: pl.DataFrame) -> None:
        """No-op bind implementation for abstract interface compliance in tests."""


def test_insert_empty_table() -> None:
    """Test the insertion in a empty table.

    In this case, the table is empty, so all records from the DataFrame
    should be inserted.
    """
    mock_datetime = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    with patch("easydw.dimension.type_2.datetime") as mock_datetime_module:
        mock_datetime_module.now.return_value = mock_datetime
        mock_db = Mock()
        mock_db.select.return_value = pl.DataFrame(
            schema={
                "key-column": pl.Int64,
                "column-1": pl.Int64,
                "column-2": pl.Utf8,
                "creation_date": pl.Datetime(time_zone="UTC"),
                "deactivation_date": pl.Datetime(time_zone="UTC"),
                "current_record": pl.Boolean,
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

        test_dimension = _TestableDimensionType2(
            name="TestDimension",
            dwh=mock_db,
        )
        test_dimension.insert(test_df, keys=["key-column"])

        expected_df = pl.DataFrame(
            {
                "key-column": [10, 20, 30, 40, 50],
                "column-1": [1, 2, 3, 4, 5],
                "column-2": ["a", "b", "c", "d", "e"],
                "creation_date": [mock_datetime] * 5,
                "deactivation_date": [None] * 5,
                "current_record": [True] * 5,
            }
        )
        result_df = mock_db.insert.call_args[0][0]

        mock_db.select.assert_called_once_with("TestDimension", query=None, params=None)
        assert mock_db.insert.call_count == 1
        assert_frame_equal(result_df, expected_df)


def test_insert_empty_table_no_types() -> None:
    """Test the insertion in a empty table with no data types in the schema.

    Some databases might return an empty DataFrame with no schema when selecting
    from an empty table.
    In this case, the table is empty, so all records from the DataFrame
    should be inserted.
    """
    mock_datetime = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    with patch("easydw.dimension.type_2.datetime") as mock_datetime_module:
        mock_datetime_module.now.return_value = mock_datetime
        mock_db = Mock()
        mock_db.select.return_value = pl.DataFrame(
            {
                "key-column": [],
                "column-1": [],
                "column-2": [],
                "creation_date": [],
                "deactivation_date": [],
                "current_record": [],
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

        test_dimension = _TestableDimensionType2(
            name="TestDimension",
            dwh=mock_db,
        )
        test_dimension.insert(test_df, keys=["key-column"])

        expected_df = pl.DataFrame(
            {
                "key-column": [10, 20, 30, 40, 50],
                "column-1": [1, 2, 3, 4, 5],
                "column-2": ["a", "b", "c", "d", "e"],
                "creation_date": [mock_datetime] * 5,
                "deactivation_date": [None] * 5,
                "current_record": [True] * 5,
            }
        )
        result_df = mock_db.insert.call_args[0][0]

        mock_db.select.assert_called_once_with("TestDimension", query=None, params=None)
        assert mock_db.insert.call_count == 1
        assert_frame_equal(result_df, expected_df)


def test_insert_full_table_all_overlaps() -> None:
    """Test the insertion but all records in the data warehouse already exist.

    In this case, all the records have to be updated.
    """
    creation_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    mock_datetime = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    with patch("easydw.dimension.type_2.datetime") as mock_datetime_module:
        mock_datetime_module.now.return_value = mock_datetime
        mock_db = Mock()
        mock_db.select.return_value = pl.DataFrame(
            {
                "key-column": [10, 20, 30],
                "column-1": [100, 200, 300],
                "column-2": ["x", "y", "z"],
                "creation_date": [creation_date] * 3,
                "deactivation_date": [None] * 3,
                "current_record": [True] * 3,
            }
        )
        mock_db.update.return_value = 3
        mock_db.insert.return_value = 3

        test_df = pl.DataFrame(
            {
                "key-column": [10, 20, 30],
                "column-1": [101, 201, 301],
                "column-2": ["a", "b", "c"],
            }
        )

        test_dimension = _TestableDimensionType2(
            name="TestDimension",
            dwh=mock_db,
        )
        test_dimension.insert(test_df, keys=["key-column"])

        expected_update_df = pl.DataFrame(
            {
                "key-column": [10, 20, 30],
                "column-1": [100, 200, 300],
                "column-2": ["x", "y", "z"],
                "creation_date": [creation_date] * 3,
                "deactivation_date": [mock_datetime] * 3,
                "current_record": [False] * 3,
            }
        )

        result_update_df = mock_db.update.call_args[0][0]

        mock_db.select.assert_called_once_with("TestDimension", query=None, params=None)
        assert mock_db.update.call_count == 1
        assert_frame_equal(result_update_df, expected_update_df)

        expected_insert_df = pl.DataFrame(
            {
                "key-column": [10, 20, 30],
                "column-1": [101, 201, 301],
                "column-2": ["a", "b", "c"],
                "creation_date": [mock_datetime] * 3,
                "deactivation_date": [None] * 3,
                "current_record": [True] * 3,
            }
        )

        result_insert_df = mock_db.insert.call_args[0][0]

        assert mock_db.insert.call_count == 1
        assert_frame_equal(result_insert_df, expected_insert_df)


def test_insert_full_table_with_overlaps() -> None:
    """Test the insertion with some records that are already in the data warehouse.

    Existing records should be updated, and new records should be inserted.
    """
    creation_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    mock_datetime = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    with patch("easydw.dimension.type_2.datetime") as mock_datetime_module:
        mock_datetime_module.now.return_value = mock_datetime
        mock_db = Mock()
        mock_db.select.return_value = pl.DataFrame(
            {
                "key-column": [10, 20],
                "column-1": [100, 200],
                "column-2": ["x", "y"],
                "creation_date": [creation_date] * 2,
                "deactivation_date": [None] * 2,
                "current_record": [True] * 2,
            }
        )
        mock_db.update.return_value = 1
        mock_db.insert.return_value = 2

        test_df = pl.DataFrame(
            {
                "key-column": [10, 30],
                "column-1": [101, 301],
                "column-2": ["a", "c"],
            }
        )

        test_dimension = _TestableDimensionType2(
            name="TestDimension",
            dwh=mock_db,
        )
        test_dimension.insert(test_df, keys=["key-column"])

        expected_update_df = pl.DataFrame(
            {
                "key-column": [10],
                "column-1": [100],
                "column-2": ["x"],
                "creation_date": [creation_date],
                "deactivation_date": [mock_datetime],
                "current_record": [False],
            }
        )
        result_update_df = mock_db.update.call_args[0][0]
        assert mock_db.update.call_count == 1
        assert_frame_equal(result_update_df, expected_update_df)

        expected_insert_df = pl.DataFrame(
            {
                "key-column": [10, 30],
                "column-1": [101, 301],
                "column-2": ["a", "c"],
                "creation_date": [mock_datetime] * 2,
                "deactivation_date": [None] * 2,
                "current_record": [True] * 2,
            }
        )
        result_insert_df = mock_db.insert.call_args[0][0]
        mock_db.select.assert_called_once_with("TestDimension", query=None, params=None)
        assert mock_db.insert.call_count == 1
        assert_frame_equal(
            result_insert_df.sort("key-column"),
            expected_insert_df.sort("key-column"),
        )


def test_insert_full_table_no_overlaps() -> None:
    """Test the insertion with no records already in the data warehouse.

    All records should be inserted.
    """
    creation_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    mock_datetime = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    with patch("easydw.dimension.type_2.datetime") as mock_datetime_module:
        mock_datetime_module.now.return_value = mock_datetime
        mock_db = Mock()
        mock_db.select.return_value = pl.DataFrame(
            {
                "key-column": [1, 2],
                "column-1": [10, 20],
                "column-2": ["a", "b"],
                "creation_date": [creation_date] * 2,
                "deactivation_date": [None] * 2,
                "current_record": [True] * 2,
            }
        )
        mock_db.insert.return_value = 2

        test_df = pl.DataFrame(
            {
                "key-column": [3, 4],
                "column-1": [30, 40],
                "column-2": ["c", "d"],
            }
        )

        test_dimension = _TestableDimensionType2(
            name="TestDimension",
            dwh=mock_db,
        )
        test_dimension.insert(test_df, keys=["key-column"])

        expected_insert_df = pl.DataFrame(
            {
                "key-column": [3, 4],
                "column-1": [30, 40],
                "column-2": ["c", "d"],
                "creation_date": [mock_datetime] * 2,
                "deactivation_date": [None] * 2,
                "current_record": [True] * 2,
            }
        )
        result_insert_df = mock_db.insert.call_args[0][0]
        mock_db.select.assert_called_once_with("TestDimension", query=None, params=None)
        assert mock_db.insert.call_count == 1
        assert_frame_equal(result_insert_df, expected_insert_df)


def test_insert_full_table_no_change() -> None:
    """Test the insertion with no changes in the records already in the data warehouse.

    No updates or inserts should occur.
    """
    creation_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    mock_db = Mock()
    mock_db.select.return_value = pl.DataFrame(
        {
            "key-column": [10, 20, 30],
            "column-1": [100, 200, 300],
            "column-2": ["x", "y", "z"],
            "creation_date": [creation_date] * 3,
            "deactivation_date": [None] * 3,
            "current_record": [True] * 3,
        }
    )

    test_df = pl.DataFrame(
        {
            "key-column": [10, 20, 30],
            "column-1": [100, 200, 300],
            "column-2": ["x", "y", "z"],
        }
    )

    test_dimension = _TestableDimensionType2(
        name="TestDimension",
        dwh=mock_db,
    )
    test_dimension.insert(test_df, keys=["key-column"])

    mock_db.select.assert_called_once_with("TestDimension", query=None, params=None)
    assert mock_db.update.call_count == 0
    assert mock_db.insert.call_count == 0


def test_insert_with_already_closed_records() -> None:
    """Test the insertion with some records that are already closed.

    Existing records should be updated, and new records should be inserted.
    """
    mock_datetime = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    with patch("easydw.dimension.type_2.datetime") as mock_datetime_module:
        mock_datetime_module.now.return_value = mock_datetime
        mock_db = Mock()
        mock_db.select.return_value = pl.DataFrame(
            {
                "key-column": [10, 10, 20, 20],
                "column-1": [100, 101, 200, 201],
                "column-2": ["x", "x", "y", "y"],
                "creation_date": [
                    datetime(2023, 1, 1, tzinfo=timezone.utc),
                    datetime(2023, 2, 1, tzinfo=timezone.utc),
                    datetime(2023, 5, 1, tzinfo=timezone.utc),
                    datetime(2023, 7, 1, tzinfo=timezone.utc),
                ],
                "deactivation_date": [
                    datetime(2023, 2, 1, tzinfo=timezone.utc),
                    None,
                    datetime(2023, 7, 1, tzinfo=timezone.utc),
                    None,
                ],
                "current_record": [False, True, False, True],
            }
        )
        mock_db.insert.return_value = 1

        test_df = pl.DataFrame(
            {
                "key-column": [10],
                "column-1": [102],
                "column-2": ["x"],
            }
        )

        test_dimension = _TestableDimensionType2(
            name="TestDimension",
            dwh=mock_db,
        )
        test_dimension.insert(test_df, keys=["key-column"])

        expected_update_df = pl.DataFrame(
            {
                "key-column": [10],
                "column-1": [101],
                "column-2": ["x"],
                "creation_date": [datetime(2023, 2, 1, tzinfo=timezone.utc)],
                "deactivation_date": [mock_datetime],
                "current_record": [False],
            }
        )
        result_update_df = mock_db.update.call_args[0][0]
        assert mock_db.update.call_count == 1
        assert_frame_equal(result_update_df, expected_update_df)

        expected_insert_df = pl.DataFrame(
            {
                "key-column": [10],
                "column-1": [102],
                "column-2": ["x"],
                "creation_date": [mock_datetime],
                "deactivation_date": [None],
                "current_record": [True],
            }
        )
        result_insert_df = mock_db.insert.call_args[0][0]
        assert mock_db.insert.call_count == 1
        assert_frame_equal(result_insert_df, expected_insert_df)
