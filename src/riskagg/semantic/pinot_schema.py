"""Apache Pinot schema and table config, generated from the semantic model's tables."""

from typing import Any

from riskagg.semantic.model import SemanticModel, SpecError

TIME_FORMAT = "1:DAYS:SIMPLE_DATE_FORMAT:yyyy-MM-dd"
INVERTED_INDEX_COLUMNS = ("node_level", "desk_id", "measure")
PINOT_TYPES = {
    "STRING": "STRING",
    "DOUBLE": "DOUBLE",
    "INT": "INT",
    "LONG": "LONG",
    "DATE": "STRING",
}


def _columns(model: SemanticModel, table: str) -> dict[str, dict[str, Any]]:
    try:
        return model.tables[table]["columns"]
    except KeyError:
        raise SpecError(f"unknown table '{table}'") from None


def pinot_schema(model: SemanticModel, table: str) -> dict[str, Any]:
    """dimension -> dimensionFieldSpecs, metric -> metricFieldSpecs, time -> dateTimeFieldSpecs."""
    schema: dict[str, Any] = {
        "schemaName": table,
        "dimensionFieldSpecs": [],
        "metricFieldSpecs": [],
        "dateTimeFieldSpecs": [],
    }
    keys = {
        "dimension": "dimensionFieldSpecs",
        "metric": "metricFieldSpecs",
        "time": "dateTimeFieldSpecs",
    }
    for name, col in _columns(model, table).items():
        try:
            key, dtype = keys[col["role"]], PINOT_TYPES[col["type"]]
        except KeyError as e:
            raise SpecError(f"table '{table}' column '{name}': unsupported {e}") from None
        spec = {"name": name, "dataType": dtype}
        if col["role"] == "time":
            # as_of_date is stored as a 'YYYY-MM-DD' string
            spec |= {"format": TIME_FORMAT, "granularity": "1:DAYS"}
        schema[key].append(spec)
    return schema


def _star_tree(model: SemanticModel, table: str) -> dict[str, Any] | None:
    """Only for tables whose every metric-serving measure is additive (pre-aggregation is safe)."""
    serving = [m for m in model.measures.values() if m.table == table]
    if not serving or any(m.measure_class != "additive" for m in serving):
        return None
    cols = _columns(model, table)
    time = [c for c, s in cols.items() if s["role"] == "time"]
    dims = [c for c, s in cols.items() if s["role"] == "dimension"]
    metrics = [c for c, s in cols.items() if s["role"] == "metric"]
    return {
        "dimensionsSplitOrder": [*time, *dims],
        "functionColumnPairs": [f"SUM__{c}" for c in metrics],
        "maxLeafRecords": 10000,
    }


def pinot_table_config(model: SemanticModel, table: str) -> dict[str, Any]:
    cols = _columns(model, table)
    index: dict[str, Any] = {
        "invertedIndexColumns": [c for c in INVERTED_INDEX_COLUMNS if c in cols],
        "loadMode": "MMAP",
    }
    star = _star_tree(model, table)
    if star:
        index["starTreeIndexConfigs"] = [star]
    time = next(c for c, s in cols.items() if s["role"] == "time")
    return {
        "tableName": table,
        "tableType": "OFFLINE",
        "segmentsConfig": {"timeColumnName": time, "replication": "1"},
        "tenants": {},
        "tableIndexConfig": index,
        "metadata": {},
    }
