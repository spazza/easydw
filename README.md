# easydw

easydw is a Python library for building data warehouse loading pipelines with less boilerplate and more consistency.

It provides reusable abstractions for:

- Connecting to warehouse engines (PostgreSQL and Oracle)
- Loading dimensions with Slowly Changing Dimension strategies (Type 0, Type 1, Type 2)
- Loading facts with key validation and upsert workflows
- Working with Polars DataFrames while persisting through SQLAlchemy

The goal is simple: model dimensions and facts as Python classes, then run predictable and testable load logic.

## Why easydw

Data warehouse projects often repeat the same patterns:

- Select data from source or staging
- Detect new records and changed records
- Apply SCD behavior on dimensions
- Upsert facts safely

easydw centralizes these patterns so project-specific code can focus on data rules and transformations instead of database plumbing.

## Core concepts

### Database abstraction

`Database` is the core abstraction used by dimensions and facts.

Main operations:

- `connect()`: initialize SQLAlchemy engine
- `select(table_name=..., query=..., params=...)`: return Polars DataFrame
- `insert(df, table_name)`: bulk insert rows from DataFrame
- `update(df, table_name, keys)`: batch update by key columns
- `upsert(df, table_name, keys)`: backend-specific upsert strategy
- `dispose()`: close engine and clear metadata cache

Included adapters:

- `PostgresDatabase`: native `ON CONFLICT DO UPDATE` upsert
- `OracleDatabase`: update-then-insert upsert emulation

### Dimension strategies

easydw includes 3 SCD-oriented base classes:

- `DimensionType0`: insert-only, no updates for existing keys
- `DimensionType1`: overwrite changed records, no historical versions
- `DimensionType2`: close old records and insert new active versions

You implement a concrete dimension by subclassing one of these and providing `bind(df)`, used to attach dimension keys to fact datasets.

### Fact loading

`Fact` defines a standard workflow:

1. Build rows in `_prepare_records(from_timestamp, to_timestamp)`
2. Optionally bind dimension keys through `_bind_dimensions(df)`
3. Validate uniqueness on declared `keys`
4. Upsert into warehouse table

## Quickstart

This example shows a minimal pipeline using PostgreSQL, one Type 1 dimension, and one fact table.

### 1) Install

```bash
pip install .
```

### 2) Create a database connection

```python
from easydw.database import PostgresDatabase

dwh = PostgresDatabase(
	name="warehouse",
	params={
		"user": "dw_user",
		"password": "dw_password",
		"host": "localhost",
		"port": 5432,
		"database": "analytics",
		"schema": "public",
	},
)

dwh.connect()
```

### 3) Implement a dimension

```python
import polars as pl

from easydw.dimension import DimensionType1


class CustomerDimension(DimensionType1):
	def __init__(self, dwh):
		super().__init__(name="dim_customer", dwh=dwh)

	def bind(self, df: pl.DataFrame) -> pl.DataFrame:
		# Join business key with dimension surrogate key.
		dim_df = self.extract().select(["customer_code", "id"])
		return df.join(dim_df, on="customer_code", how="left")
```

Load dimension records:

```python
incoming_customers = pl.DataFrame(
	{
		"customer_code": ["C001", "C002"],
		"customer_name": ["Alice", "Bob"],
		"country": ["IT", "FR"],
	}
)

customer_dim = CustomerDimension(dwh)
customer_dim.insert(incoming_customers, keys=["customer_code"])
```

### 4) Implement a fact

```python
from datetime import datetime

import polars as pl

from easydw.fact import Fact


class SalesFact(Fact):
	keys = ["order_id"]
	dimensions = [CustomerDimension]

	def __init__(self, dwh):
		super().__init__(name="fct_sales", dwh=dwh)

	def _prepare_records(
		self, from_timestamp: datetime, to_timestamp: datetime
	) -> pl.DataFrame:
		# Replace this with your source extraction logic.
		raw = pl.DataFrame(
			{
				"order_id": [1001, 1002],
				"customer_code": ["C001", "C002"],
				"amount": [120.0, 75.5],
			}
		)

		return self._bind_dimensions(raw)
```

Load fact rows:

```python
sales_fact = SalesFact(dwh)
sales_fact.insert(
	from_timestamp=datetime(2026, 3, 1),
	to_timestamp=datetime(2026, 3, 20),
)

dwh.dispose()
```

## Development

### Build package

```bash
python -m build
cp dist/*.whl /your/desired/location
pip install /your/desired/location/easydw-*.whl
```

### Run tests

```bash
pytest
```

## Requirements

- Python 3.10+
- SQLAlchemy 2.x
- Polars 1.x
- psycopg2 (for PostgreSQL usage)
- oracledb (for Oracle usage)