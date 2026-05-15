"""Tests for generic database operations."""

from typing import override

import polars as pl
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, select

from easydw.database import Database


class _TestableDatabase(Database):
    """Minimal concrete implementation for testing base `Database` logic."""

    @override
    def connect(self) -> None:
        """No-op for tests using a manually configured engine."""

    @override
    def upsert(self, df: pl.DataFrame, table_name: str, keys: list[str]) -> int | None:
        """Unused in these tests; only required by the abstract interface."""
        _ = df, table_name, keys
        return 0


def test_update_handles_null_keys_and_values() -> None:
    """Validate update semantics when both key and payload contain NULL values.

    This regression test covers a subtle SQL edge case: the update condition must
    match records whose key column is ``NULL`` in both the target table and the
    incoming payload. A plain equality predicate (``col = :param``) does not
    match ``NULL`` keys, so the implementation must use a null-safe condition.

    Scenario:
    - Seed two rows, one with a ``NULL`` business key and one with key ``1``.
    - Execute a batch update where:
        - the ``NULL`` key row gets a non-null value ("new-null")
        - the key ``1`` row gets a ``NULL`` value

    Expectations:
    - Two rows are reported as updated.
    - The row identified by ``NULL`` key is updated correctly.
    - ``NULL`` payload values are persisted as SQL NULL.
    """
    db = _TestableDatabase(name="test", params={})
    db.engine = create_engine("sqlite+pysqlite:///:memory:")

    metadata = MetaData()
    table = Table(
        "sample_table",
        metadata,
        Column("business_key", Integer, nullable=True),
        Column("value", String, nullable=True),
    )
    metadata.create_all(db.engine)

    with db.engine.connect() as connection, connection.begin():
        connection.execute(
            table.insert(),
            [
                {"business_key": None, "value": "old-null"},
                {"business_key": 1, "value": "old-one"},
            ],
        )

    update_df = pl.DataFrame(
        {
            "business_key": [None, 1],
            "value": ["new-null", None],
        }
    )

    updated_rows = db.update(update_df, "sample_table", ["business_key"])

    expected_updated_rows = 2
    assert updated_rows == expected_updated_rows

    with db.engine.connect() as connection:
        rows = connection.execute(select(table.c.business_key, table.c.value)).all()

    rows_by_key = {row.business_key: row.value for row in rows}
    assert rows_by_key[None] == "new-null"
    assert rows_by_key[1] is None
