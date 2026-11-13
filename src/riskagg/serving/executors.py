"""Executors run one ``Query`` and return rows with the same columns on every backend."""

from pathlib import Path

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from riskagg.semantic.query import Query

PARTITION_SCHEMA = pa.schema([("as_of_date", pa.string()), ("run_id", pa.string())])


class PinotError(RuntimeError):
    """Pinot answered with an error."""


class PandasExecutor:
    """Runs the query on the curated Parquet files; partition filters prune as_of_date/run_id."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def execute(self, query: Query) -> pd.DataFrame:
        dataset = ds.dataset(
            self.root / "curated" / query.table,
            format="parquet",
            partitioning=ds.partitioning(PARTITION_SCHEMA, flavor="hive"),
        )
        expression = None
        for col, values in query.filters.items():
            term = pc.field(col).isin(list(values))
            expression = term if expression is None else expression & term
        source = query.aggregate[1] if query.aggregate else "value"
        needed = [
            *query.select_dims,
            *(["scenario_idx"] if query.kind == "vector_scenarios" else []),
        ]
        df = dataset.to_table(columns=[*needed, source], filter=expression).to_pandas()
        if query.aggregate:
            if needed:
                df = df.groupby(needed, sort=True, as_index=False)[source].sum()
            elif not df.empty:
                df = pd.DataFrame({source: [df[source].sum()]})
        return df.rename(columns={source: query.value_column})[query.columns].reset_index(drop=True)


class PinotExecutor:
    """POSTs the query's SQL to a Pinot broker."""

    def __init__(self, broker_url: str, client: httpx.Client | None = None):
        self.url = broker_url.rstrip("/") + "/query/sql"
        self.client = client or httpx.Client(timeout=60.0)

    def execute(self, query: Query) -> pd.DataFrame:
        response = self.client.post(self.url, json={"sql": query.to_pinot_sql()})
        if response.status_code != 200:
            raise PinotError(f"Pinot broker returned HTTP {response.status_code}: {response.text}")
        body = response.json()
        if body.get("exceptions"):
            messages = "; ".join(str(e.get("message", e)) for e in body["exceptions"])
            raise PinotError(f"Pinot query failed: {messages}")
        table = body["resultTable"]
        columns = query.columns
        names = table["dataSchema"]["columnNames"]
        if len(names) != len(columns):
            raise PinotError(f"Pinot returned columns {names}, expected {columns}")
        return pd.DataFrame(table["rows"], columns=columns)
