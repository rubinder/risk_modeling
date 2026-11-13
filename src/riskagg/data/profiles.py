"""Data-size profiles shared by every generator."""

from dataclasses import dataclass

_BOTH = ("CURRENT", "STRESSED")


@dataclass(frozen=True)
class Profile:
    name: str
    seed: int
    n_business_lines: int
    n_desks: int
    n_books: int
    n_trades: int
    n_scenarios: int
    skew_desk_share: float
    windows: tuple[str, ...]


PROFILES: dict[str, Profile] = {
    "tiny": Profile("tiny", 42, 2, 3, 6, 20, 20, 0.0, _BOTH),
    "demo": Profile("demo", 42, 4, 20, 200, 100_000, 250, 0.0, _BOTH),
    # bench was 1,000,000 trades. Cut to 100,000 so `riskagg bench` fits the free disk of the
    # machine the committed report was generated on (about 2 GB): its peak disk use is several
    # times the P&L vectors (100k trades are about 360 MB of Parquet), because the input, the
    # shuffle files of a run that moves whole vectors (a sort-merge join; twice under AQE's
    # forced skew split) and the previous run's not-yet-cleaned shuffle coexist.
    "bench": Profile("bench", 42, 4, 40, 400, 100_000, 250, 0.4, ("CURRENT",)),
}
