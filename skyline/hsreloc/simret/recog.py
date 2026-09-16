"""EXP-SKY-010 — recognition radius of the frozen C1-32 matcher versus true horizontal separation.

``prototype``. Additive: imports the frozen matcher and the bit-equal vectorised search of
``hsreloc.simret.relpose`` and edits nothing. No range map, no relative-position estimate — this
module only asks *which stored reference the frozen matcher ranks first, and how far away it is*.

Pieces, each small and tested:

* :class:`ReferenceBank` — every reference's shifted copies pre-computed once, so one query is
  scored against the whole database in one masked-NCC call. The winner per reference is the frozen
  ``BoundedLagNccMatcher`` result — lag exactly, score to 1e-12 (float summation order; asserted by
  ``tests/test_simret_recog.py``, and the study script re-checks against ``match()`` at run time).
* :func:`retrieval_outcome` — the region/exact bookkeeping for one query: nearest reference,
  ranks, frozen frame-level acceptance (``accept-v2-margin``), region margin, false region,
  ambiguity, and the region-level acceptance variant reported beside the frozen rule.
* :func:`bin_table` / :func:`radius_from_bins` — the pre-registered bins, support requirement and
  radius rule of the experiment record; :func:`spacing_implication` — the ``2R`` geometry.

Conventions: positions are ENU metres (``PROT-SKY-001``); ``d_near`` is the horizontal distance
from the query to the nearest reference *present* in the database instance; ranks are 1-based;
ties break by ``(-score, reference_id)`` exactly as ``hsreloc.simret.report.variant_retrieval``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from hsreloc.simret.relpose import (DEFAULT_MAX_LAG_SAMPLES, DEFAULT_MIN_OVERLAP_FRAC,
                                    _masked_ncc, _shift_matrix)

RECOG_VERSION = "1.0.0"


class RecogError(Exception):
    """Inputs that cannot be evaluated as asked."""


# --------------------------------------------------------------------------------------------------
# batched frozen-C1 scoring
# --------------------------------------------------------------------------------------------------

def winner_per_reference(scores: np.ndarray, lags: np.ndarray) -> tuple:
    """Per row of ``scores`` (R, L): the frozen tie-break — max score, then least |lag|, then the
    negative lag. Rows with no finite score get ``(-inf, nan)``."""
    scores = np.asarray(scores, dtype=np.float64)
    lags = np.asarray(lags, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != lags.size:
        raise RecogError(f"scores must be (R, L) with L = {lags.size}, got {scores.shape}")
    best = np.max(scores, axis=1)
    # lexicographic key over lags: (|lag|, lag) ascending — smallest key among the ties wins
    key = np.abs(lags) * (2 * lags.size) + (lags + lags.size)
    masked = np.where(scores == best[:, None], key[None, :], np.inf)
    j = np.argmin(masked, axis=1)
    lag = lags[j]
    none = ~np.isfinite(best)
    lag = np.where(none, np.nan, lag)
    return best, lag


class ReferenceBank:
    """A database of reference profiles with every bounded-lag shift pre-computed.

    ``score(query)`` returns, per reference, the frozen C1 score and winning lag (the lag exactly and
    the score within 1e-12 of ``BoundedLagNccMatcher(max_lag_samples, min_overlap_frac).match()``).
    """

    def __init__(self, reference_ids: Sequence[str], profiles: Sequence[np.ndarray],
                 max_lag: int = DEFAULT_MAX_LAG_SAMPLES,
                 min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC, chunk_references: int = 48):
        if len(reference_ids) != len(profiles):
            raise RecogError("reference_ids and profiles differ in length")
        if not profiles:
            raise RecogError("an empty reference bank cannot rank anything")
        self.reference_ids = list(reference_ids)
        self.max_lag = int(max_lag)
        self.lags = np.arange(-self.max_lag, self.max_lag + 1, dtype=np.float64)
        n = int(np.asarray(profiles[0]).size)
        self.n_samples = n
        self.floor = max(2, int(math.ceil(min_overlap_frac * n)))
        mats = []
        for p in profiles:
            p = np.asarray(p, dtype=np.float64)
            if p.ndim != 1 or p.size != n:
                raise RecogError(f"every profile must be 1-D of length {n}, got {p.shape}")
            mats.append(_shift_matrix(p, self.lags))
        self.n_references = len(profiles)
        # chunked so the masked-NCC temporaries stay small (a few tens of MB) on a whole database
        self.chunk = max(1, int(chunk_references))
        self.S_chunks = [np.concatenate(mats[i:i + self.chunk], axis=0)
                         for i in range(0, self.n_references, self.chunk)]   # each (r_i * L, n)

    def score(self, query: np.ndarray) -> tuple:
        q = np.asarray(query, dtype=np.float64)
        if q.ndim != 1 or q.size != self.n_samples:
            raise RecogError(f"query must be 1-D of length {self.n_samples}, got {q.shape}")
        L = self.lags.size
        best, lag = [], []
        for S in self.S_chunks:
            flat = _masked_ncc(q, S, self.floor)
            b, l = winner_per_reference(flat.reshape(-1, L), self.lags)
            best.append(b)
            lag.append(l)
        return np.concatenate(best), np.concatenate(lag)


# --------------------------------------------------------------------------------------------------
# one query's retrieval outcome
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Acceptance:
    theta_s: float = 0.9
    theta_m: float = 0.15


def _rank_order(scores: np.ndarray, ids: Sequence[str], keep: np.ndarray) -> np.ndarray:
    """Indices of the kept references ranked by ``(-score, reference_id)``; -inf scores last."""
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return idx
    s = scores[idx]
    order = sorted(range(idx.size), key=lambda k: (-(s[k] if np.isfinite(s[k]) else -np.inf), ids[idx[k]]))
    return idx[np.asarray(order, dtype=int)]


def retrieval_outcome(query_xy, ref_ids: Sequence[str], ref_xy: np.ndarray, scores: np.ndarray,
                      lags: np.ndarray, keep: Optional[np.ndarray] = None, tau_region: float = 125.0,
                      k: int = 5, acceptance: Acceptance = Acceptance(),
                      extra_taus: Sequence[float] = ()) -> dict:
    """Everything the study records for one query against one database instance.

    ``keep`` marks the references present in this instance (hold-out radii and the query's own
    frame are removed by the caller *after* scoring). ``tau_region`` is the frozen region rule;
    ``extra_taus`` add ``region_rank_tau<int>`` / ``false_region_tau<int>`` / ``attainable_tau<int>``
    columns for the sensitivity check.
    """
    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    lags = np.asarray(lags, dtype=np.float64)
    if ref_xy.shape != (len(ref_ids), 2) or scores.shape != (len(ref_ids),):
        raise RecogError("ref_xy must be (R, 2) and scores (R,)")
    keep = np.ones(len(ref_ids), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    qx, qy = float(query_xy[0]), float(query_xy[1])
    dist = np.hypot(ref_xy[:, 0] - qx, ref_xy[:, 1] - qy)
    order = _rank_order(scores, ref_ids, keep)
    out = {"n_references": int(keep.sum())}
    if order.size == 0:
        out.update({"d_near_m": None, "nearest_id": None, "outcome": "EMPTY_DATABASE"})
        return out
    kept = np.flatnonzero(keep)
    i_near = kept[np.argmin(dist[kept])]
    d_near = float(dist[i_near])
    ranked_d = dist[order]
    ranked_s = scores[order]
    s1 = float(ranked_s[0])
    s2 = float(ranked_s[1]) if order.size > 1 else None
    finite1 = np.isfinite(s1)
    margin = (s1 - s2) if (finite1 and s2 is not None and np.isfinite(s2)) else None
    accepted = bool(finite1 and s1 >= acceptance.theta_s and margin is not None and margin >= acceptance.theta_m)

    def region_fields(tau: float, suffix: str) -> None:
        in_reg = ranked_d <= tau
        r_rank = int(np.argmax(in_reg)) + 1 if in_reg.any() else None
        attainable = d_near <= tau
        out[f"attainable{suffix}"] = attainable
        out[f"region_rank{suffix}"] = r_rank
        out[f"region_top1{suffix}"] = r_rank == 1
        out[f"region_in_top_k{suffix}"] = r_rank is not None and r_rank <= k
        out[f"false_region{suffix}"] = not (r_rank == 1)
        best_in = float(np.max(ranked_s[in_reg])) if in_reg.any() else None
        best_out = float(np.max(ranked_s[~in_reg])) if (~in_reg).any() else None
        rm = (best_in - best_out) if (best_in is not None and best_out is not None
                                      and np.isfinite(best_in) and np.isfinite(best_out)) else None
        out[f"best_in_region_score{suffix}"] = best_in
        out[f"best_out_region_score{suffix}"] = best_out
        out[f"region_margin{suffix}"] = rm
        out[f"ambiguous{suffix}"] = bool(attainable and rm is not None and abs(rm) < acceptance.theta_m)
        # region-level acceptance: top-1 against the best reference outside the top-1's own region
        top1_xy = ref_xy[order[0]]
        far_from_top1 = np.hypot(ref_xy[order, 0] - top1_xy[0], ref_xy[order, 1] - top1_xy[1]) > tau
        best_far = float(np.max(ranked_s[far_from_top1])) if far_from_top1.any() else None
        rmargin = (s1 - best_far) if (finite1 and best_far is not None and np.isfinite(best_far)) else None
        out[f"region_level_margin{suffix}"] = rmargin
        out[f"accepted_region{suffix}"] = bool(finite1 and s1 >= acceptance.theta_s
                                               and rmargin is not None and rmargin >= acceptance.theta_m)

    exact_rank = int(np.flatnonzero(order == i_near)[0]) + 1
    out.update({
        "d_near_m": d_near, "nearest_id": ref_ids[i_near],
        "exact_rank": exact_rank, "exact_top1": exact_rank == 1, "exact_in_top_k": exact_rank <= k,
        "nearest_score": float(scores[i_near]), "nearest_lag": float(lags[i_near]),
        "top1_id": ref_ids[order[0]], "top1_score": s1, "top1_lag": float(lags[order[0]]),
        "top1_distance_m": float(ranked_d[0]), "top2_score": s2, "frame_margin": margin,
        "accepted_frame": accepted,
        "outcome": "SCORED" if finite1 else "NO_ADMISSIBLE_ALIGNMENT",
    })
    for j in range(k):
        if j < order.size:
            out[f"top{j + 1}_id"] = ref_ids[order[j]]
            out[f"top{j + 1}_score"] = float(ranked_s[j])
            out[f"top{j + 1}_distance_m"] = float(ranked_d[j])
        else:
            out[f"top{j + 1}_id"] = None
            out[f"top{j + 1}_score"] = None
            out[f"top{j + 1}_distance_m"] = None
    region_fields(float(tau_region), "")
    for t in extra_taus:
        region_fields(float(t), f"_tau{int(round(t))}")
    return out


# --------------------------------------------------------------------------------------------------
# bins, support, radius rule, spacing
# --------------------------------------------------------------------------------------------------

def wilson_low(p: Optional[float], n: int, z: float = 1.96) -> Optional[float]:
    if p is None or n <= 0:
        return None
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / denom


def bin_label(lo: float, hi: Optional[float]) -> str:
    return f"{lo:g}-{hi:g}" if hi is not None else f"{lo:g}+"


def assign_bin(d: float, edges: Sequence[float]) -> str:
    for lo, hi in zip(edges, list(edges[1:]) + [None]):
        if hi is None or d < hi:
            if d >= lo:
                return bin_label(lo, hi)
    raise RecogError(f"distance {d} below the first bin edge {edges[0]}")


def _rate(flags: list) -> Optional[float]:
    return (sum(1 for f in flags if f) / len(flags)) if flags else None


def _median(vals: list) -> Optional[float]:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def bin_table(rows: list, edges: Sequence[float], support: dict, k: int = 5,
              site_key: str = "site", d_key: str = "d_near_m") -> list:
    """Per-bin metrics over query rows (dicts as :func:`retrieval_outcome` returns, plus
    ``site``). One entry per bin in edge order; ``supported`` applies the pre-registered rule."""
    labels = [bin_label(lo, hi) for lo, hi in zip(edges, list(edges[1:]) + [None])]
    groups = {lab: [] for lab in labels}
    for r in rows:
        d = r.get(d_key)
        if d is None:
            continue
        groups[assign_bin(float(d), edges)].append(r)
    table = []
    for lab in labels:
        g = groups[lab]
        n = len(g)
        sites = sorted({r[site_key] for r in g})
        att = [r for r in g if r["attainable"]]
        acc = [r for r in g if r["accepted_frame"]]
        acc_false = [r for r in acc if r["false_region"]]
        acc_r = [r for r in g if r["accepted_region"]]
        acc_r_false = [r for r in acc_r if r["false_region"]]
        per_site = {}
        for s in sites:
            m = [r for r in g if r[site_key] == s]
            per_site[s] = {"n": len(m), "region_top1": _rate([r["region_top1"] for r in m])}
        site_rates = [v["region_top1"] for v in per_site.values() if v["region_top1"] is not None]
        entry = {
            "bin": lab, "n": n, "n_sites": len(sites), "n_attainable": len(att),
            "supported": n >= int(support["min_queries"]) and len(sites) >= int(support["min_sites"]),
            "exact_top1": _rate([r["exact_top1"] for r in g]),
            "exact_recall_k": _rate([r["exact_in_top_k"] for r in g]),
            "region_top1": _rate([r["region_top1"] for r in g]),
            "region_top1_of_attainable": _rate([r["region_top1"] for r in att]),
            "region_recall_k": _rate([r["region_in_top_k"] for r in g]),
            "region_top1_wilson_low": wilson_low(_rate([r["region_top1"] for r in g]), n),
            "top1_score_median": _median([r["top1_score"] for r in g]),
            "top1_score_p10": (float(np.percentile([r["top1_score"] for r in g if r["top1_score"] is not None
                                                     and np.isfinite(r["top1_score"])], 10)) if g else None),
            "correct_region_score_median": _median([r["best_in_region_score"] for r in att]),
            "region_margin_median": _median([r["region_margin"] for r in att]),
            "region_margin_p10": (float(np.percentile([r["region_margin"] for r in att
                                                       if r["region_margin"] is not None], 10))
                                  if any(r["region_margin"] is not None for r in att) else None),
            "false_region_rate": _rate([r["false_region"] for r in g]),
            "n_accepted_frame": len(acc),
            "coverage_frame": _rate([r["accepted_frame"] for r in g]),
            "coverage_region": _rate([r["accepted_region"] for r in g]),
            "accepted_false_region_rate": (len(acc_false) / n) if n else None,
            "p_false_given_accepted": (len(acc_false) / len(acc)) if acc else None,
            "n_accepted_false_region": len(acc_false),
            "n_accepted_region": len(acc_r),
            "p_false_given_accepted_region": (len(acc_r_false) / len(acc_r)) if acc_r else None,
            "n_accepted_region_false": len(acc_r_false),
            "ambiguity_rate": _rate([r["ambiguous"] for r in att]),
            "n_matcher_failed_with_nearby_reference": sum(1 for r in att if not r["region_top1"]),
            "n_no_suitable_reference": n - len(att),
            "site_region_top1_min": (min(site_rates) if site_rates else None),
            "site_region_top1_median": (float(np.median(site_rates)) if site_rates else None),
            "per_site": per_site,
            "k": k,
        }
        table.append(entry)
    return table


def radius_from_bins(table: list, rule: dict) -> dict:
    """The pre-registered walk: from the first bin upward, the run of passing bins; stop at the
    first failing or unsupported bin. Returns the upper edge of the last passing bin (None if the
    first bin already fails) for the conservative and the recoverable criterion separately."""
    def upper(lab: str) -> Optional[float]:
        return None if lab.endswith("+") else float(lab.split("-")[1])

    result = {"conservative_m": None, "recoverable_m": None,
              "conservative_stop": None, "recoverable_stop": None}
    for crit in ("conservative", "recoverable"):
        last = None
        stop = None
        for e in table:
            if not e["supported"]:
                stop = {"bin": e["bin"], "reason": f"unsupported (n={e['n']}, sites={e['n_sites']})"}
                break
            if crit == "conservative":
                ok = (e["region_top1"] is not None and e["region_top1"] >= float(rule["region_top1_min"])
                      and e["accepted_false_region_rate"] is not None
                      and e["accepted_false_region_rate"] <= float(rule["accepted_false_region_max"]))
                why = (f"region_top1={e['region_top1']:.3f}, accepted_false_region="
                       f"{e['accepted_false_region_rate']:.3f}")
            else:
                ok = e["region_recall_k"] is not None and e["region_recall_k"] >= float(rule["region_recall_k_min"])
                why = f"region_recall_k={e['region_recall_k']:.3f}"
            if not ok:
                stop = {"bin": e["bin"], "reason": f"fails ({why})"}
                break
            if upper(e["bin"]) is None:
                stop = {"bin": e["bin"], "reason": "open-ended bin passes; radius capped at its lower edge"}
                break
            last = upper(e["bin"])
        result[f"{crit}_m"] = last
        result[f"{crit}_stop"] = stop
    return result


def spacing_implication(radius_m: Optional[float], current_spacing_m: float = 10.0) -> dict:
    """``s <= 2R`` is the no-gap same-path bound; ``s ~ R`` keeps a 2x margin (worst case R/2)."""
    if radius_m is None:
        return {"theoretical_max_spacing_m": None, "conservative_spacing_m": None,
                "current_spacing_m": current_spacing_m,
                "assessment": "no reliable radius — spacing cannot be justified by recognition alone"}
    s_theory = 2.0 * radius_m
    s_cons = float(radius_m)
    if current_spacing_m < s_cons / 2:
        verdict = "denser than the conservative suggestion (margin > 2x)"
    elif current_spacing_m <= s_cons:
        verdict = "reasonable (within the conservative suggestion)"
    elif current_spacing_m <= s_theory:
        verdict = "sparse (inside the 2R bound but without margin)"
    else:
        verdict = "too sparse (gaps between recognition neighbourhoods)"
    return {"theoretical_max_spacing_m": s_theory, "conservative_spacing_m": s_cons,
            "current_spacing_m": current_spacing_m, "assessment": verdict}
