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
    "bench": Profile("bench", 42, 4, 40, 400, 1_000_000, 250, 0.4, ("CURRENT",)),
}
