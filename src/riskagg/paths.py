"""Default locations of the config files (repo-root relative, overridable by argument)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def semantic_model_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else CONFIG_DIR / "semantic_model.yaml"


def aggregations_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else CONFIG_DIR / "aggregations.yaml"
