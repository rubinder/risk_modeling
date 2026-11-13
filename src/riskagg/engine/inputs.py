"""Run inputs a measure may need beyond the reporting grain."""

from dataclasses import dataclass, field

from pyspark.sql import DataFrame


@dataclass
class RunInputs:
    """The frames of one run. ``grain`` is the P&L reporting grain (vector measures and FRTB
    IMA), with ``n_scenarios`` its vector length.

    Stress measures need the raw ``sensitivities``, ``trades``, ``hierarchy`` and the small
    ``shocks`` table (scenario, risk_factor_id, shock). ``stress_grain`` builds the stress P&L
    at the reporting-grain dimensions once and returns the same lazy frame afterwards, so the
    run can cache it, write it and derive the worst-scenario figures from it.

    CCAR needs ``banking_book``, ``macro_scenarios`` and ``capital_start``, and takes its
    trading loss from ``stress_grain``. ``ccar_projection`` builds the projection once and
    returns the same lazy frame afterwards (the run caches and writes it); ``ccar_summary``
    derives the capital measures from it.
    """

    grain: DataFrame | None = None
    sensitivities: DataFrame | None = None
    trades: DataFrame | None = None
    hierarchy: DataFrame | None = None
    shocks: DataFrame | None = None
    default_risk_positions: DataFrame | None = None
    banking_book: DataFrame | None = None
    macro_scenarios: DataFrame | None = None
    capital_start: DataFrame | None = None
    n_scenarios: int | None = None
    _stress_grain: DataFrame | None = field(default=None, repr=False)
    # Shared through ``dataclasses.replace`` copies, like the FRTB levels below.
    _ccar: dict = field(default_factory=dict, repr=False)
    # FRTB levels by node columns. ``run_all`` copies the inputs with ``dataclasses.replace``,
    # which passes this same dict on, so a level built through a copy is visible to the original.
    _frtb_levels: dict = field(default_factory=dict, repr=False)
    _ima_levels: dict = field(default_factory=dict, repr=False)

    def frtb_level(self, node_cols: list[str]):
        """FRTB SA frames at a node level, built and cached once per run (see ``release``)."""
        key = tuple(node_cols)
        if key not in self._frtb_levels:
            from riskagg.measures.frtb_drc import build_level

            self._frtb_levels[key] = build_level(self, list(node_cols))
        return self._frtb_levels[key]

    def ima_level(self, node_cols: list[str]):
        """FRTB IMA frames at a node level, built from ``grain`` and cached once per run."""
        key = tuple(node_cols)
        if key not in self._ima_levels:
            from riskagg.measures.frtb_ima import build_ima_level

            self._ima_levels[key] = build_ima_level(self, list(node_cols))
        return self._ima_levels[key]

    def release(self) -> None:
        """Unpersist every frame cached for FRTB (standardised and internal models)."""
        for levels in (self._frtb_levels, self._ima_levels):
            for level in levels.values():
                for df in level.cached:
                    df.unpersist()
            levels.clear()

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

    def ccar_projection(self) -> DataFrame:
        if "projection" not in self._ccar:
            from riskagg.measures import ccar
            from riskagg.measures.stress import load_scenarios

            missing = [
                n for n in ("banking_book", "macro_scenarios", "capital_start")
                if getattr(self, n) is None
            ]  # fmt: skip
            if missing:
                raise ValueError(f"CCAR measures need run inputs {missing}, which are not set")
            gms = ccar.gms_loss_by_entity_bl(
                self.stress_grain(), ccar.global_market_shock(load_scenarios())
            )
            self._ccar["projection"] = ccar.project(
                self.banking_book, self.macro_scenarios, self.capital_start, gms, ccar.load_params()
            )
        return self._ccar["projection"]

    def ccar_summary(self) -> DataFrame:
        if "summary" not in self._ccar:
            from riskagg.measures import ccar

            self._ccar["summary"] = ccar.summary(
                self.ccar_projection(), self.capital_start, ccar.load_params()
            )
        return self._ccar["summary"]
