"""Dual-direction (North + West) skyline evidence for place recognition — ``EXP-SKY-011``.

Maturity: **prototype**. Additive: it consumes what :func:`hsreloc.simret.recog.retrieval_outcome`
records for each view against the *same paired reference list*, plus the raw per-reference score
vectors of both views, and combines them with a handful of declared, interpretable rules. Nothing
here is learned, nothing is a probability, and the frozen ``BoundedLagNccMatcher`` is not touched.

Rules
-----
``strict_agreement``
    each view must pass its own hard gate (top-1 score ≥ θ_s and a margin ≥ θ_m — the frozen
    ``accept-v2-margin`` shape, frame-level or region-level) **and** the two top-1 references must
    lie in the same physical region (within ``tau_agree`` of each other; adjacent captures of one
    place count as agreement, the exact same frame is not required).
``weakest_view``
    one ranking over the paired references by ``min(s_N, s_W)``: a reference is only as strong as
    the direction that supports it least; the frozen gate is then applied to that ranking.
``mean_score``
    one ranking by ``(s_N + s_W) / 2``; legal only because both views are the same matcher on the
    same NCC scale (the report checks the score distributions before reading this rule).

``temporal_confirmation`` is the single-view comparison INT's design relies on: the last ``k``
consecutive queries of one trajectory segment must each pass the gate and agree on the region.
The caller groups queries by reconstructed segment, so a window never spans a teleport.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

DUALVIEW_VERSION = "1.0.0"
RULES = ("strict_agreement", "weakest_view", "mean_score")
GATE_LEVELS = ("frame", "region")


class DualViewError(Exception):
    pass


@dataclass(frozen=True)
class Gate:
    """One view's hard gate. ``level`` picks the margin: ``frame`` = top-1 minus top-2 (the frozen
    rule on a sparse memory); ``region`` = top-1 minus the best reference outside the top-1's own
    region (the ``EXP-SKY-010`` variant that survives a dense memory)."""
    theta_s: float = 0.9
    theta_m: float = 0.15
    level: str = "frame"

    def __post_init__(self):
        if self.level not in GATE_LEVELS:
            raise DualViewError(f"gate level must be one of {GATE_LEVELS}, got {self.level!r}")


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))


def margin_of(outcome: dict, level: str) -> Optional[float]:
    m = outcome.get("frame_margin") if level == "frame" else outcome.get("region_level_margin")
    return float(m) if _finite(m) else None


def view_passes(outcome: dict, gate: Gate) -> bool:
    """The frozen acceptance shape applied to one view's outcome."""
    s1 = outcome.get("top1_score")
    if not _finite(s1):
        return False
    m = margin_of(outcome, gate.level)
    return float(s1) >= gate.theta_s and m is not None and m >= gate.theta_m


def agreement(north: dict, west: dict, ref_xy: dict, tau_agree: float) -> dict:
    """Do the two views' top-1 references name the same physical region?"""
    a, b = north.get("top1_id"), west.get("top1_id")
    if a is None or b is None or a not in ref_xy or b not in ref_xy:
        return {"agree": False, "top1_separation_m": None, "same_reference": False}
    (ax, ay), (bx, by) = ref_xy[a], ref_xy[b]
    d = math.hypot(ax - bx, ay - by)
    return {"agree": d <= tau_agree, "top1_separation_m": d, "same_reference": a == b}


def strict_agreement(north: dict, west: dict, ref_xy: dict, query_xy, gate_n: Gate, gate_w: Gate,
                     tau_agree: float, tau_region: float) -> dict:
    """The primary dual rule. The dual candidate is the North top-1 (West must sit in its region);
    it is *false* when that candidate is farther than ``tau_region`` from the true query position.
    ``dual_region_top1`` is the ungated correctness of the agreed candidate (abstention counts as
    not correct), the recall counterpart of the gated ``accepted``."""
    pass_n, pass_w = view_passes(north, gate_n), view_passes(west, gate_w)
    ag = agreement(north, west, ref_xy, tau_agree)
    cand = north.get("top1_id") if ag["agree"] else None
    if cand is not None:
        cx, cy = ref_xy[cand]
        d_cand = math.hypot(cx - float(query_xy[0]), cy - float(query_xy[1]))
    else:
        d_cand = None
    accepted = bool(pass_n and pass_w and ag["agree"])
    false_region = (d_cand is None) or (d_cand > tau_region)
    both_in_top_k = bool(north.get("region_in_top_k") and west.get("region_in_top_k"))
    return {
        "rule": "strict_agreement", "pass_north": pass_n, "pass_west": pass_w, **ag,
        "candidate_id": cand, "candidate_distance_m": d_cand,
        "accepted": accepted, "false_region": false_region,
        "accepted_false": bool(accepted and false_region),
        "dual_region_top1": bool(cand is not None and not false_region),
        "dual_region_in_top_k": both_in_top_k,
        "min_score": (min(float(north["top1_score"]), float(west["top1_score"]))
                      if _finite(north.get("top1_score")) and _finite(west.get("top1_score")) else None),
        "min_margin": (min(margin_of(north, gate_n.level) or -1.0, margin_of(west, gate_w.level) or -1.0)
                       if margin_of(north, gate_n.level) is not None and margin_of(west, gate_w.level) is not None else None),
    }


def fuse_scores(scores_n: np.ndarray, scores_w: np.ndarray, rule: str) -> np.ndarray:
    """One score vector over the paired references. ``-inf`` (no admissible alignment) in either
    view makes the reference lose under both rules — never a promotion."""
    a = np.asarray(scores_n, dtype=np.float64)
    b = np.asarray(scores_w, dtype=np.float64)
    if a.shape != b.shape:
        raise DualViewError(f"score vectors differ in shape: {a.shape} vs {b.shape}")
    if rule == "weakest_view":
        return np.minimum(a, b)
    if rule == "mean_score":
        out = 0.5 * (a + b)
        out[~(np.isfinite(a) & np.isfinite(b))] = -np.inf
        return out
    raise DualViewError(f"unknown fusion rule {rule!r}; expected weakest_view or mean_score")


def temporal_confirmation(seq: Sequence[dict], query_xy: Sequence, k: int, gate: Gate, ref_xy: dict,
                          tau_agree: float, tau_region: float) -> list:
    """Single-view temporal confirmation over ONE trajectory segment, in trajectory order.

    Query ``i`` is accepted when the last ``k`` consecutive queries (``i-k+1 .. i``, all inside
    this segment) each pass the gate and every top-1 in the window lies within ``tau_agree`` of the
    window's last top-1. The candidate is the last top-1; false when farther than ``tau_region``
    from the true position of query ``i``. Windows never cross a segment boundary because the
    caller passes one segment at a time — a teleport resets the state by construction."""
    if k < 1:
        raise DualViewError("k must be >= 1")
    out = []
    for i, o in enumerate(seq):
        window = seq[i - k + 1:i + 1] if i - k + 1 >= 0 else None
        accepted, n_pass, agree = False, 0, None
        if window is not None:
            passes = [view_passes(w, gate) for w in window]
            n_pass = sum(passes)
            last = o.get("top1_id")
            if all(passes) and last is not None and last in ref_xy:
                lx, ly = ref_xy[last]
                agree = all(w.get("top1_id") in ref_xy and
                            math.hypot(ref_xy[w["top1_id"]][0] - lx, ref_xy[w["top1_id"]][1] - ly) <= tau_agree
                            for w in window)
                accepted = bool(agree)
        cand = o.get("top1_id")
        d_cand = None
        if cand is not None and cand in ref_xy:
            cx, cy = ref_xy[cand]
            d_cand = math.hypot(cx - float(query_xy[i][0]), cy - float(query_xy[i][1]))
        false_region = d_cand is None or d_cand > tau_region
        out.append({"temporal_k": k, "window_complete": window is not None, "n_pass_in_window": n_pass,
                    "window_agrees": agree, "accepted": accepted, "candidate_id": cand,
                    "candidate_distance_m": d_cand, "false_region": false_region,
                    "accepted_false": bool(accepted and false_region)})
    return out


# --------------------------------------------------------------------------------------------------
# gate sweep and the pre-registered DEV selection
# --------------------------------------------------------------------------------------------------

def sweep(rows: Sequence[dict], thetas_s: Sequence[float], thetas_m: Sequence[float],
          evaluate: Callable[[dict, Gate], tuple], level: str = "frame") -> list:
    """``evaluate(row, gate) -> (accepted, false_region)`` over a grid; one summary per point."""
    out = []
    for ts in thetas_s:
        for tm in thetas_m:
            g = Gate(float(ts), float(tm), level)
            n_acc = n_false = 0
            for r in rows:
                acc, false = evaluate(r, g)
                n_acc += int(bool(acc))
                n_false += int(bool(acc and false))
            n = len(rows)
            out.append({"theta_s": float(ts), "theta_m": float(tm), "level": level, "n": n,
                        "n_accepted": n_acc, "n_accepted_false": n_false,
                        "coverage": (n_acc / n) if n else None,
                        "p_false_given_accepted": (n_false / n_acc) if n_acc else None})
    return out


def select_gate(sweep_rows: Sequence[dict], hard_rows: Sequence[dict] = ()) -> dict:
    """The pre-registered DEV choice: among grid points with **zero** accepted-false on DEV (the
    in-coverage sweep and every hard-negative sweep at the same point), the largest DEV coverage;
    ties go to the larger margin, then the larger score threshold (the more conservative point).
    If no point is false-free, the fewest accepted-false decides and the result says so."""
    hard_by = {}
    for h in hard_rows:
        key = (h["theta_s"], h["theta_m"])
        hard_by[key] = hard_by.get(key, 0) + int(h["n_accepted_false"])
    scored = []
    for r in sweep_rows:
        key = (r["theta_s"], r["theta_m"])
        total_false = int(r["n_accepted_false"]) + hard_by.get(key, 0)
        scored.append((total_false, -(r["coverage"] or 0.0), -r["theta_m"], -r["theta_s"], r))
    scored.sort(key=lambda t: t[:4])
    total_false, _, _, _, best = scored[0]
    return {**best, "total_accepted_false_dev": total_false, "false_free_point_exists": total_false == 0,
            "n_false_free_points": sum(1 for t in scored if t[0] == 0)}
