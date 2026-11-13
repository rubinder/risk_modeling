"""Semantic-model variants for tests."""

import dataclasses

from riskagg.semantic.model import SemanticModel


def prefixed(model: SemanticModel, prefix: str) -> SemanticModel:
    """The same model with every physical table renamed ``<prefix><table>``."""
    return dataclasses.replace(
        model,
        tables={prefix + t: spec for t, spec in model.tables.items()},
        measures={
            n: dataclasses.replace(m, table=prefix + m.table) for n, m in model.measures.items()
        },
        vector_table=prefix + model.vector_table,
    )
