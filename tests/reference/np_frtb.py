"""Plain-loop reference for the FRTB SA sensitivities-based charge (MAR21).

Reads ``config/reference/frtb_sa.yaml`` itself and computes from raw sensitivities (a pandas
frame) with dictionaries and loops. Does not import ``riskagg.measures.frtb_sa``.
Keys: ``(node, risk_class, component, corr_scenario)`` with ``node`` a tuple over node_cols.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import yaml

REF_PATH = Path(__file__).resolve().parents[2] / "config" / "reference" / "frtb_sa.yaml"
SCENARIOS = ("LOW", "MEDIUM", "HIGH")


def _ref() -> dict:
    return yaml.safe_load(REF_PATH.read_text())


def _scenario(x: float, scenario: str) -> float:
    if scenario == "HIGH":
        return min(1.25 * x, 1.0)
    if scenario == "LOW":
        return max(2.0 * x - 1.0, 0.75 * x)
    return x


def _num(v) -> float:
    return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _risk_weight(ref: dict, rc: str, comp: str, bucket: str, tenor) -> float:
    rep = ref["reporting_currency"]
    if comp == "VEGA":
        if rc == "EQ":
            return ref["EQ"]["vega"]["risk_weights"][bucket]
        return ref[rc]["vega"]["risk_weight"]
    if rc == "GIRR":
        d = ref["GIRR"]["delta"]
        rw = {float(t): w for t, w in d["risk_weights"].items()}[float(tenor)]
        specified = set(d["specified_currencies"]) | {rep}
        return rw / math.sqrt(2) if bucket in specified else rw
    if rc == "EQ":
        return ref["EQ"]["delta"]["buckets"][bucket]["risk_weight"]
    d = ref["FX"]["delta"]
    pair = f"{rep}/{bucket}"
    return d["risk_weight"] / math.sqrt(2) if pair in d["specified_pairs"] else d["risk_weight"]


def _rho(ref: dict, rc: str, comp: str, bucket: str, tk, tl) -> float:
    if rc == "GIRR" and comp == "DELTA":
        c = ref["GIRR"]["delta"]["tenor_correlation"]
        tk, tl = float(tk), float(tl)
        return max(math.exp(-c["theta"] * abs(tk - tl) / min(tk, tl)), c["floor"])
    if rc == "EQ":
        return ref["EQ"]["delta"]["buckets"][bucket]["rho"]
    return 1.0


def _gamma(ref: dict, rc: str, b: str, c: str) -> float:
    if rc != "EQ":
        return ref[rc]["delta"]["gamma"]
    g = ref["EQ"]["delta"]["gamma"]
    bi, ci = int(b), int(c)
    if 11 in (bi, ci):
        return g["with_bucket_11"]
    if 1 <= bi <= 10 and 1 <= ci <= 10:
        return g["buckets_1_to_10"]
    if {bi, ci} == {12, 13}:
        return g["between_12_and_13"]
    return g["otherwise"]


def _across(ref, rc, comp, scenario, buckets: dict[str, tuple[float, float]]) -> float:
    names = sorted(buckets)

    def total(clamp: bool) -> float:
        out = 0.0
        for b in names:
            kb, sb = buckets[b]
            out += kb * kb
            for c in names:
                if c == b:
                    continue
                kc, sc = buckets[c]
                g = _gamma(ref, rc, b, c)
                if comp == "CURVATURE":
                    g = g * g
                g = _scenario(g, scenario)
                if clamp:
                    sb_, sc_ = max(min(sb, kb), -kb), max(min(sc, kc), -kc)
                else:
                    sb_, sc_ = sb, sc
                psi = 0.0 if comp == "CURVATURE" and sb < 0 and sc < 0 else 1.0
                out += g * sb_ * sc_ * psi
        return out

    t = total(False)
    if comp != "CURVATURE" and t < 0:
        t = total(True)
    return math.sqrt(max(0.0, t))


def sbm_charges(sens: pd.DataFrame, node_cols: Sequence[str]) -> dict[tuple, float]:
    ref = _ref()
    # net per node x factor
    net: dict[tuple, float] = defaultdict(float)
    meta: dict[tuple, tuple] = {}
    cvr: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in sens.to_dict("records"):
        node = tuple(row[c] for c in node_cols)
        rc, ft, bucket = row["risk_class"], row["factor_type"], row["bucket"]
        key = (node, rc, ft, bucket, row["risk_factor_id"])
        value = _num(row["delta"]) if ft == "DELTA" else _num(row["vega"])
        net[key] += value
        meta[key] = row["tenor_years"]
        up, down = _num(row["curvature_up"]), _num(row["curvature_down"])
        if ft == "DELTA" and (up != 0 or down != 0):
            acc = cvr[(node, rc, bucket)]
            acc[0] += up
            acc[1] += down

    # weighted sensitivities grouped by node, class, component, bucket
    groups: dict[tuple, list[tuple[str, object, float]]] = defaultdict(list)
    for (node, rc, ft, bucket, rf), s in net.items():
        ws = s * _risk_weight(ref, rc, ft, bucket, meta[(node, rc, ft, bucket, rf)])
        groups[(node, rc, ft, bucket)].append((rf, meta[(node, rc, ft, bucket, rf)], ws))

    out: dict[tuple, float] = {}
    for scenario in SCENARIOS:
        per_class: dict[tuple, dict[str, tuple[float, float]]] = defaultdict(dict)
        for (node, rc, comp, bucket), factors in groups.items():
            k2, sb = 0.0, 0.0
            for rf_k, t_k, ws_k in factors:
                sb += ws_k
                for rf_l, t_l, ws_l in factors:
                    if rf_k == rf_l:
                        rho = 1.0
                    else:
                        rho = _scenario(_rho(ref, rc, comp, bucket, t_k, t_l), scenario)
                    k2 += rho * ws_k * ws_l
            per_class[(node, rc, comp)][bucket] = (math.sqrt(max(0.0, k2)), sb)
        for (node, rc, bucket), (up, down) in cvr.items():
            k_up, k_down = max(up, 0.0), max(down, 0.0)
            up_selected = k_up > k_down or (k_up == k_down and up > down)
            per_class[(node, rc, "CURVATURE")][bucket] = (
                max(k_up, k_down),
                up if up_selected else down,
            )
        for (node, rc, comp), buckets in per_class.items():
            out[(node, rc, comp, scenario)] = _across(ref, rc, comp, scenario, buckets)
    return out


def sbm_total(charges: dict[tuple, float]) -> dict[tuple, tuple[float, bool]]:
    """``{(node, scenario): (total, is_binding)}``; ties go to LOW, then MEDIUM, then HIGH."""
    totals: dict[tuple, float] = defaultdict(float)
    for (node, _rc, _comp, scenario), value in charges.items():
        totals[(node, scenario)] += value
    out = {}
    for node in {n for n, _ in totals}:
        best = max(SCENARIOS, key=lambda s: (totals[(node, s)], -SCENARIOS.index(s)))
        for s in SCENARIOS:
            out[(node, s)] = (totals[(node, s)], s == best)
    return out
