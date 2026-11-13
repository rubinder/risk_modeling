"""Run inputs a measure may need beyond the reporting grain."""

from dataclasses import dataclass, field

from pyspark.sql import DataFrame


@dataclass
class RunInputs:
    """The frames of one run. ``grain`` is the P&L reporting grain (vector measures).

    Stress measures need the raw ``sensitivities``, ``trades``, ``hierarchy`` and the small
    ``shocks`` table (scenario, risk_factor_id, shock). ``stress_grain`` builds the stress P&L
    at the reporting-grain dimensions once and returns the same lazy frame afterwards, so the
    run can cache it, write it and derive the worst-scenario figures from it.
    """

    grain: DataFrame | None = None
    sensitivities: DataFrame | None = None
    trades: DataFrame | None = None
    hierarchy: DataFrame | None = None
    shocks: DataFrame | None = None
    default_risk_positions: DataFrame | None = None
    _stress_grain: DataFrame | None = field(default=None, repr=False)
    # FRTB levels by node columns. ``run_all`` copies the inputs with ``dataclasses.replace``,
    # which passes this same dict on, so a level built through a copy is visible to the original.
    _frtb_levels: dict = field(default_factory=dict, repr=False)

    def frtb_level(self, node_cols: list[str]):
        """FRTB SA frames at a node level, built and cached once per run (see ``release``)."""
        key = tuple(node_cols)
        if key not in self._frtb_levels:
            from riskagg.measures.frtb_drc import build_level

            self._frtb_levels[key] = build_level(self, list(node_cols))
        return self._frtb_levels[key]

    def release(self) -> None:
        """Unpersist every frame cached for FRTB."""
        for level in self._frtb_levels.values():
            for df in level.cached:
                df.unpersist()
        self._frtb_levels.clear()

    def stress_grain(self) -> DataFrame:
        if self._stress_grain is None:
            from riskagg.measures.stress import stress_by_grain, stress_pnl

            missing = [
                n
                for n in ("sensitivities", "trades", "hierarchy", "shocks")
                if getattr(self, n) is None
            ]
            if missing:
                raise ValueError(f"stress measures need run inputs {missing}, which are not set")
            pnl = stress_pnl(self.sensitivities, self.shocks)
            self._stress_grain = stress_by_grain(pnl, self.trades, self.hierarchy)
        return self._stress_grain
