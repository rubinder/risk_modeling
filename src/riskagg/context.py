"""Run context: identifies a run and locates every file it reads or writes.

Layout under ``root``::

    raw/<profile>/as_of_date=<d>/<dataset>/
    curated/<table>/as_of_date=<d>/run_id=<r>/
    runs/<run_id>/manifest.json
    runs/official.json              {as_of_date: run_id}
"""

import datetime
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from riskagg.paths import REPO_ROOT

DEFAULT_ROOT = REPO_ROOT / "data"


def raw_path(root: Path, profile: str, as_of_date: datetime.date, dataset: str) -> Path:
    return Path(root) / "raw" / profile / f"as_of_date={as_of_date.isoformat()}" / dataset


@dataclass(frozen=True)
class RunContext:
    profile: str
    as_of_date: datetime.date
    run_id: str
    spec_hash: str
    root: Path

    def raw(self, dataset: str) -> Path:
        return raw_path(self.root, self.profile, self.as_of_date, dataset)

    def curated_root(self, table: str) -> Path:
        """Table root; reading it exposes ``as_of_date`` and ``run_id`` as columns."""
        return self.root / "curated" / table

    def curated(self, table: str) -> Path:
        """This run's partition of ``table``."""
        return (
            self.curated_root(table)
            / f"as_of_date={self.as_of_date.isoformat()}"
            / f"run_id={self.run_id}"
        )

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"


def new_run(
    profile: str, as_of_date: datetime.date, root: str | Path, spec_hash: str
) -> RunContext:
    """A fresh run id (UTC timestamp + 6 hex, so ids sort chronologically to the second)."""
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{secrets.token_hex(3)}"
    return RunContext(profile, as_of_date, run_id, spec_hash, Path(root))


def _official_path(root: Path) -> Path:
    return Path(root) / "runs" / "official.json"


def _read_official(root: Path) -> dict[str, str]:
    path = _official_path(root)
    return json.loads(path.read_text()) if path.exists() else {}


def official_run(root: str | Path, as_of_date: datetime.date) -> str | None:
    return _read_official(Path(root)).get(as_of_date.isoformat())


def official_runs(root: str | Path) -> dict[datetime.date, str]:
    """Every as-of date that has an official run, mapped to that run's id."""
    return {datetime.date.fromisoformat(d): r for d, r in _read_official(Path(root)).items()}


def run_context(root: str | Path, run_id: str) -> RunContext:
    """The context of a finished run, rebuilt from its manifest."""
    path = Path(root) / "runs" / run_id / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"run '{run_id}' has no manifest at {path}")
    m = json.loads(path.read_text())
    as_of = datetime.date.fromisoformat(m["as_of_date"])
    return RunContext(m["profile"], as_of, m["run_id"], m["spec_hash"], Path(root))


def mark_official(ctx: RunContext) -> None:
    """Make ``ctx`` the official run for its as-of date (atomic replace of official.json)."""
    official = _read_official(ctx.root)
    official[ctx.as_of_date.isoformat()] = ctx.run_id
    path = _official_path(ctx.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(official, indent=2, sort_keys=True))
    os.replace(tmp, path)
