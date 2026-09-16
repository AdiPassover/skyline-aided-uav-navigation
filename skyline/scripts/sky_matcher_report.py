#!/usr/bin/env python
"""EXP-SKY-013 report stage — positional resolution and safety, per matcher arm.

Imported by ``sky_matcher_resolution.py --stage report``. Every number is computed from the saved
nested-bound score matrices, so all five arms are read off numerically identical per-lag scores and
every memory is a keep-mask over one scoring pass (the EXP-SKY-010/011 discipline).

Nothing here re-scores, re-ranks by a new rule, or tunes a threshold. The acceptance rule, the gate
values, the region rule and the dual rules are the frozen ones, applied unchanged.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from hsreloc.simret import dualview, lagstudy, recog

RULES = ("north", "west", "strict", "weakest")
RULE_LABEL = {"north": "North only", "west": "West only", "strict": "strict dual agreement",
              "weakest": "weakest-view fusion"}


# --------------------------------------------------------------------------------------------------
# memories — every one is a keep-mask over the same saved matrix
# --------------------------------------------------------------------------------------------------

def fig8_memories(cfg: dict, z, self_cell: bool) -> dict:
    """``name -> (n_queries, n_references) boolean eligibility``. For the figure eight the memory is
    the run's own captures: this *is* INT's online-mapping case."""
    qo, ro = np.asarray(z["query_order"]), np.asarray(z["reference_order"])
    nq, nr = len(qo), len(ro)
    mem = {}
    if self_cell:
        for k in cfg["recency_k"]:
            mem[f"self_recency_k{int(k)}"] = np.abs(qo[:, None] - ro[None, :]) > int(k)
        for stride in cfg["sparse_strides"]:
            keep = (ro % int(stride)) == 0
            m = (np.abs(qo[:, None] - ro[None, :]) > 0) & keep[None, :]
            mem[f"self_sparse{int(stride)}"] = m
    else:
        mem["cross_run"] = np.ones((nq, nr), dtype=bool)
    return mem


def terrain_memories(cfg: dict, z) -> dict:
    """The EXP-SKY-011 memory shapes: the sparse anchor memory, the dense path memory, and the
    dense memory with every reference inside tau removed (leave-site-out)."""
    qxy, rxy = np.asarray(z["query_xy"]), np.asarray(z["reference_xy"])
    anchor = np.asarray(z["reference_is_anchor"], dtype=bool)
    D = np.hypot(qxy[:, None, 0] - rxy[None, :, 0], qxy[:, None, 1] - rxy[None, :, 1])
    tau = float(cfg["tau_region_m"])
    nq, nr = len(qxy), len(rxy)
    return {
        "sparse": np.broadcast_to(anchor[None, :], (nq, nr)).copy(),
        "dense0": np.ones((nq, nr), dtype=bool),
        "dense_lso": D > tau,
    }


# --------------------------------------------------------------------------------------------------
# per-query rows
# --------------------------------------------------------------------------------------------------

def build_rows(cfg: dict, zn, zw, bound: int, memory: str, keep: np.ndarray, meta: dict) -> list:
    """One row per query: the four rules' selections, distances, lags, margins and acceptance."""
    tau = float(cfg["tau_region_m"])
    tau_agree = float(cfg["tau_agree_m"])
    k = int(cfg["recall_k"])
    acc = recog.Acceptance(float(cfg["acceptance"]["theta_s"]), float(cfg["acceptance"]["theta_m"]))
    cat = float(cfg["catastrophic_m"])
    ref_ids = [str(x) for x in zn["reference_keys"]]
    ref_xy = np.asarray(zn["reference_xy"], dtype=np.float64)
    ref_up = np.asarray(zn["reference_up"], dtype=np.float64)
    q_xy = np.asarray(zn["query_xy"], dtype=np.float64)
    q_up = np.asarray(zn["query_up"], dtype=np.float64)
    q_ids = [str(x) for x in zn["query_keys"]]
    ref_index = {rid: i for i, rid in enumerate(ref_ids)}
    ref_xy_map = {rid: (float(ref_xy[i, 0]), float(ref_xy[i, 1])) for i, rid in enumerate(ref_ids)}
    Sn, Ln = zn[f"scores_{bound}"], zn[f"lags_{bound}"]
    Sw, Lw = zw[f"scores_{bound}"], zw[f"lags_{bound}"]
    gate = dualview.Gate(acc.theta_s, acc.theta_m,
                         "frame" if memory.startswith(("sparse", "self")) else "region")
    rows = []
    for i in range(len(q_ids)):
        m = keep[i]
        if not m.any():
            rows.append({**meta, "bound": bound, "matcher": lagstudy.bound_label(bound),
                         "memory": memory, "query_id": q_ids[i], "n_eligible": 0,
                         "outcome": "EMPTY_DATABASE"})
            continue
        on = recog.retrieval_outcome(q_xy[i], ref_ids, ref_xy, Sn[i], Ln[i], keep=m,
                                     tau_region=tau, k=k, acceptance=acc)
        ow = recog.retrieval_outcome(q_xy[i], ref_ids, ref_xy, Sw[i], Lw[i], keep=m,
                                     tau_region=tau, k=k, acceptance=acc)
        fused = dualview.fuse_scores(Sn[i], Sw[i], "weakest_view")
        omin = recog.retrieval_outcome(q_xy[i], ref_ids, ref_xy, fused, Ln[i], keep=m,
                                       tau_region=tau, k=k, acceptance=acc)
        st = dualview.strict_agreement(on, ow, ref_xy_map, q_xy[i], gate, gate, tau_agree, tau)
        row = {**meta, "bound": bound, "matcher": lagstudy.bound_label(bound), "memory": memory,
               "query_id": q_ids[i], "n_eligible": int(m.sum()),
               "singleton": bool(m.sum() <= 1),
               "query_up_m": float(q_up[i]), "d_near_m": on.get("d_near_m"),
               "nearest_rank_north": on.get("exact_rank"), "nearest_rank_west": ow.get("exact_rank"),
               "outcome": on.get("outcome")}
        for rule, o in (("north", on), ("west", ow), ("weakest", omin)):
            pf = lagstudy.positional_fields(q_xy[i], float(q_up[i]), ref_xy, ref_up, o,
                                            ref_index, cat)
            row[f"{rule}_top1_id"] = o.get("top1_id")
            row[f"{rule}_top1_score"] = o.get("top1_score")
            row[f"{rule}_top1_lag"] = o.get("top1_lag")
            row[f"{rule}_selected_distance_m"] = pf["selected_distance_m"]
            row[f"{rule}_selection_excess_m"] = pf["selection_excess_m"]
            row[f"{rule}_top1_minus_nearest_score"] = pf["top1_minus_nearest_score"]
            row[f"{rule}_selected_dh_m"] = pf.get("selected_dh_m")
            row[f"{rule}_nearest_score"] = o.get("nearest_score")
            row[f"{rule}_nearest_lag"] = o.get("nearest_lag")
            row[f"{rule}_frame_margin"] = o.get("frame_margin")
            row[f"{rule}_region_margin"] = o.get("region_level_margin")
            row[f"{rule}_accepted_frame"] = o.get("accepted_frame")
            row[f"{rule}_accepted_region"] = o.get("accepted_region")
            row[f"{rule}_false_region"] = o.get("false_region")
            row[f"{rule}_attainable"] = o.get("attainable")
            row[f"{rule}_region_top1"] = o.get("region_top1")
            row[f"{rule}_catastrophic"] = pf["catastrophic"]
        row["strict_accepted"] = st["accepted"]
        row["strict_false_region"] = st["false_region"]
        row["strict_accepted_false"] = st["accepted_false"]
        row["strict_top1_id"] = st["candidate_id"]
        row["strict_selected_distance_m"] = st["candidate_distance_m"]
        row["strict_agree"] = st["agree"]
        row["nw_top1_separation_m"] = st["top1_separation_m"]
        row["strict_catastrophic"] = bool(st["candidate_distance_m"] is not None
                                          and st["candidate_distance_m"] > cat)
        rows.append(row)
    return rows


def accepted(row: dict, rule: str, level: str) -> bool:
    if rule == "strict":
        return bool(row.get("strict_accepted"))
    return bool(row.get(f"{rule}_accepted_{level}"))


def selected_distance(row: dict, rule: str):
    return row.get(f"{rule}_selected_distance_m")


# --------------------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------------------

def selection_summary(rows: list, rule: str, qs=(50, 90, 95, 99)) -> dict:
    d = [selected_distance(r, rule) for r in rows if r.get("outcome") == "SCORED"]
    out = lagstudy.quantiles(d, qs)
    excess = [r.get(f"{rule}_selection_excess_m") for r in rows] if rule != "strict" else []
    out["excess"] = lagstudy.quantiles(excess, (50, 90, 95)) if excess else None
    nr = [r.get("nearest_rank_north") if rule in ("north", "strict") else r.get("nearest_rank_west")
          for r in rows]
    nr = [x for x in nr if x is not None]
    out["nearest_rank_median"] = float(np.median(nr)) if nr else None
    out["nearest_is_top1_frac"] = (float(np.mean([x == 1 for x in nr])) if nr else None)
    cat = [r.get(f"{rule}_catastrophic") for r in rows]
    out["catastrophic_frac"] = float(np.mean([bool(x) for x in cat])) if cat else None
    return out


def acceptance_summary(rows: list, rule: str, level: str, theta_s: float, theta_m: float) -> dict:
    """Accepted count, accepted-false count and the selected-distance distribution *given accepted*
    — the conditional the reopening brief asks for in §10, on development GT only."""
    acc, acc_false, dists = 0, 0, []
    for r in rows:
        if r.get("outcome") != "SCORED":
            continue
        s = r.get(f"{rule}_top1_score") if rule != "strict" else r.get("north_top1_score")
        m = (r.get(f"{rule}_frame_margin") if level == "frame" else r.get(f"{rule}_region_margin")) \
            if rule != "strict" else r.get("north_frame_margin" if level == "frame" else "north_region_margin")
        if rule == "strict":
            ok = bool(r.get("strict_agree")) and _passes(r, "north", level, theta_s, theta_m) \
                 and _passes(r, "west", level, theta_s, theta_m)
        else:
            ok = _passes(r, rule, level, theta_s, theta_m)
        if not ok:
            continue
        acc += 1
        dists.append(selected_distance(r, rule))
        false = r.get("strict_false_region") if rule == "strict" else r.get(f"{rule}_false_region")
        acc_false += int(bool(false))
    out = {"theta_s": theta_s, "theta_m": theta_m, "level": level, "n": len(rows),
           "n_accepted": acc, "n_accepted_false_region": acc_false,
           "coverage": acc / len(rows) if rows else None}
    out["selected_distance_given_accepted"] = lagstudy.quantiles(dists)
    return out


def _passes(row: dict, view: str, level: str, theta_s: float, theta_m: float) -> bool:
    s = row.get(f"{view}_top1_score")
    m = row.get(f"{view}_frame_margin") if level == "frame" else row.get(f"{view}_region_margin")
    return bool(s is not None and np.isfinite(s) and s >= theta_s
                and m is not None and np.isfinite(m) and m >= theta_m)


def score_vs_distance(cfg: dict, z, bound: int, exclude_self: bool = True) -> list:
    """Median/quantile score in bins of true horizontal query-reference separation — the full
    distribution, never a single mean."""
    qxy, rxy = np.asarray(z["query_xy"]), np.asarray(z["reference_xy"])
    D = np.hypot(qxy[:, None, 0] - rxy[None, :, 0], qxy[:, None, 1] - rxy[None, :, 1])
    S = np.asarray(z[f"scores_{bound}"], dtype=np.float64)
    mask = np.ones_like(S, dtype=bool)
    if exclude_self and S.shape[0] == S.shape[1]:
        np.fill_diagonal(mask, False)
    return lagstudy.binned(S[mask].ravel(), D[mask].ravel(), cfg["bins_m"])


def lag_vs_distance(cfg: dict, z, bound: int, score_floor=None) -> list:
    """|winning lag| in the same bins — the H3 discriminator. With ``score_floor`` the table is
    restricted to high-scoring pairs, which is the question that actually matters: do the *distant
    matches that would be believed* use more alignment freedom than nearby ones?"""
    qxy, rxy = np.asarray(z["query_xy"]), np.asarray(z["reference_xy"])
    D = np.hypot(qxy[:, None, 0] - rxy[None, :, 0], qxy[:, None, 1] - rxy[None, :, 1])
    S = np.asarray(z[f"scores_{bound}"], dtype=np.float64)
    L = np.abs(np.asarray(z[f"lags_{bound}"], dtype=np.float64))
    mask = np.ones_like(S, dtype=bool)
    if S.shape[0] == S.shape[1]:
        np.fill_diagonal(mask, False)
    if score_floor is not None:
        mask &= S >= float(score_floor)
    rows = lagstudy.binned(L[mask].ravel(), D[mask].ravel(), cfg["bins_m"])
    for row, (lo, hi) in zip(rows, zip(cfg["bins_m"][:-1], cfg["bins_m"][1:])):
        m = mask & (D >= lo) & (D < hi)
        row["frac_nonzero_lag"] = float(np.mean(L[m] > 0)) if m.any() else None
    return rows


def calibration(cfg: dict, z, bound: int, near_m: float = 20.0, far_m: float = 150.0) -> dict:
    """Score distributions for near / far-same-scene pairs, and their separation — 'is 0.95 still
    the same threshold under this matcher?'"""
    qxy, rxy = np.asarray(z["query_xy"]), np.asarray(z["reference_xy"])
    D = np.hypot(qxy[:, None, 0] - rxy[None, :, 0], qxy[:, None, 1] - rxy[None, :, 1])
    S = np.asarray(z[f"scores_{bound}"], dtype=np.float64)
    mask = np.ones_like(S, dtype=bool)
    if S.shape[0] == S.shape[1]:
        np.fill_diagonal(mask, False)
    near = S[mask & (D <= near_m)]
    far = S[mask & (D >= far_m)]
    out = {"near_m": near_m, "far_m": far_m}
    for name, v in (("near", near), ("far", far)):
        v = v[np.isfinite(v)]
        out[name] = ({"n": int(v.size), "p05": float(np.percentile(v, 5)),
                      "p50": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                      "frac_ge_0.95": float(np.mean(v >= 0.95)),
                      "frac_ge_0.90": float(np.mean(v >= 0.90))} if v.size else None)
    if out["near"] and out["far"]:
        out["separation_p05near_minus_p95far"] = out["near"]["p05"] - out["far"]["p95"]
        out["overlap_frac_far_above_median_near"] = float(
            np.mean(far[np.isfinite(far)] >= out["near"]["p50"]))
    return out


def selection_table(cfg: dict, zn, zw, keep: np.ndarray, bound: int) -> dict:
    """Selected-reference distance for each rule, on one memory — the INT-facing quantity."""
    qxy, rxy = np.asarray(zn["query_xy"]), np.asarray(zn["reference_xy"])
    ref_ids = [str(x) for x in zn["reference_keys"]]
    ref_xy_map = {rid: (float(rxy[i, 0]), float(rxy[i, 1])) for i, rid in enumerate(ref_ids)}
    tau = float(cfg["tau_region_m"])
    acc = recog.Acceptance(float(cfg["acceptance"]["theta_s"]), float(cfg["acceptance"]["theta_m"]))
    gate = dualview.Gate(acc.theta_s, acc.theta_m, "region")
    Sn, Ln = zn[f"scores_{bound}"], zn[f"lags_{bound}"]
    Sw, Lw = zw[f"scores_{bound}"], zw[f"lags_{bound}"]
    got = {r: [] for r in RULES}
    ranks, lags_top = [], []
    for i in range(len(qxy)):
        m = keep[i]
        if not m.any():
            continue
        on = recog.retrieval_outcome(qxy[i], ref_ids, rxy, Sn[i], Ln[i], keep=m,
                                     tau_region=tau, acceptance=acc)
        if on.get("outcome") != "SCORED":
            continue
        ow = recog.retrieval_outcome(qxy[i], ref_ids, rxy, Sw[i], Lw[i], keep=m,
                                     tau_region=tau, acceptance=acc)
        om = recog.retrieval_outcome(qxy[i], ref_ids, rxy,
                                     dualview.fuse_scores(Sn[i], Sw[i], "weakest_view"), Ln[i],
                                     keep=m, tau_region=tau, acceptance=acc)
        st = dualview.strict_agreement(on, ow, ref_xy_map, qxy[i], gate, gate,
                                       float(cfg["tau_agree_m"]), tau)
        got["north"].append(on["top1_distance_m"])
        got["west"].append(ow["top1_distance_m"])
        got["weakest"].append(om["top1_distance_m"])
        got["strict"].append(st["candidate_distance_m"])
        ranks.append(on["exact_rank"])
        lags_top.append(abs(on["top1_lag"]) if on.get("top1_lag") is not None else None)
    out = {}
    for rule, vals in got.items():
        q = lagstudy.quantiles(vals, cfg["selected_distance_quantiles"])
        v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
        q["frac_beyond_tau"] = float(np.mean(v > tau)) if v.size else None
        q["frac_catastrophic"] = float(np.mean(v > float(cfg["catastrophic_m"]))) if v.size else None
        out[rule] = q
    out["nearest_rank_median"] = float(np.median(ranks)) if ranks else None
    out["nearest_is_top1_frac"] = float(np.mean([r == 1 for r in ranks])) if ranks else None
    lt = [x for x in lags_top if x is not None]
    out["top1_abs_lag_median"] = float(np.median(lt)) if lt else None
    return out


def height_analysis(cfg: dict, z, bound: int) -> dict:
    """Score, winning lag and selected distance versus |height difference| — the characterization
    that must come before any altitude-aware variant is even considered."""
    qxy, rxy = np.asarray(z["query_xy"]), np.asarray(z["reference_xy"])
    qup, rup = np.asarray(z["query_up"]), np.asarray(z["reference_up"])
    D = np.hypot(qxy[:, None, 0] - rxy[None, :, 0], qxy[:, None, 1] - rxy[None, :, 1])
    dH = np.abs(qup[:, None] - rup[None, :])
    S = np.asarray(z[f"scores_{bound}"], dtype=np.float64)
    L = np.abs(np.asarray(z[f"lags_{bound}"], dtype=np.float64))
    mask = np.ones_like(S, dtype=bool)
    if S.shape[0] == S.shape[1]:
        np.fill_diagonal(mask, False)
    # hold horizontal separation roughly fixed so height is not confounded by distance
    close = mask & (D <= 40.0)
    edges = [0, 2, 5, 10, 15, 20, 30]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = close & (dH >= lo) & (dH < hi)
        rows.append({"dh_lo_m": lo, "dh_hi_m": hi, "n": int(m.sum()),
                     "score_p50": float(np.median(S[m])) if m.any() else None,
                     "score_p05": float(np.percentile(S[m], 5)) if m.any() else None,
                     "lag_p50": float(np.median(L[m])) if m.any() else None,
                     "lag_p95": float(np.percentile(L[m], 95)) if m.any() else None})
    return {"horizontal_separation_max_m": 40.0, "bins": rows}


# --------------------------------------------------------------------------------------------------
# safety — the frozen hard negatives, every rule, every arm, no tuning
# --------------------------------------------------------------------------------------------------

def hard_negative_table(cfg: dict, zn, zw, keep: np.ndarray, bound: int, level: str,
                        theta_s: float, theta_m: float, query_mask=None) -> dict:
    """Accepted counts on a population where the queried place is ABSENT: every accept is false."""
    qxy, rxy = np.asarray(zn["query_xy"]), np.asarray(zn["reference_xy"])
    ref_ids = [str(x) for x in zn["reference_keys"]]
    ref_xy_map = {rid: (float(rxy[i, 0]), float(rxy[i, 1])) for i, rid in enumerate(ref_ids)}
    tau = float(cfg["tau_region_m"])
    acc = recog.Acceptance(theta_s, theta_m)
    gate = dualview.Gate(theta_s, theta_m, level)
    Sn, Ln = zn[f"scores_{bound}"], zn[f"lags_{bound}"]
    Sw, Lw = zw[f"scores_{bound}"], zw[f"lags_{bound}"]
    key = "accepted_frame" if level == "frame" else "accepted_region"
    counts = {r: 0 for r in RULES}
    dists, n = [], 0
    for i in range(len(qxy)):
        if query_mask is not None and not query_mask[i]:
            continue
        m = keep[i]
        if not m.any():
            continue
        n += 1
        on = recog.retrieval_outcome(qxy[i], ref_ids, rxy, Sn[i], Ln[i], keep=m,
                                     tau_region=tau, acceptance=acc)
        ow = recog.retrieval_outcome(qxy[i], ref_ids, rxy, Sw[i], Lw[i], keep=m,
                                     tau_region=tau, acceptance=acc)
        om = recog.retrieval_outcome(qxy[i], ref_ids, rxy,
                                     dualview.fuse_scores(Sn[i], Sw[i], "weakest_view"), Ln[i],
                                     keep=m, tau_region=tau, acceptance=acc)
        st = dualview.strict_agreement(on, ow, ref_xy_map, qxy[i], gate, gate,
                                       float(cfg["tau_agree_m"]), tau)
        if on.get(key):
            counts["north"] += 1
            dists.append(on["top1_distance_m"])
        counts["west"] += int(bool(ow.get(key)))
        counts["weakest"] += int(bool(om.get(key)))
        counts["strict"] += int(bool(st["accepted"]))
    out = {"n_queries": n, "theta_s": theta_s, "theta_m": theta_m, "gate_level": level,
           "accepted_false": counts}
    out["north_false_distance"] = lagstudy.quantiles(dists, (50, 95))
    return out


# --------------------------------------------------------------------------------------------------
# report driver
# --------------------------------------------------------------------------------------------------

def revisit_coverage(cfg: dict, zn, zw, keep: np.ndarray, bound: int, level: str) -> dict:
    """The cost side of the trade: how many genuine revisits each arm actually accepts, and how many
    of those accepts name the wrong region. Frozen gate, no tuning."""
    qxy, rxy = np.asarray(zn["query_xy"]), np.asarray(zn["reference_xy"])
    ref_ids = [str(x) for x in zn["reference_keys"]]
    ref_xy_map = {rid: (float(rxy[i, 0]), float(rxy[i, 1])) for i, rid in enumerate(ref_ids)}
    tau = float(cfg["tau_region_m"])
    acc = recog.Acceptance(float(cfg["acceptance"]["theta_s"]), float(cfg["acceptance"]["theta_m"]))
    gate = dualview.Gate(acc.theta_s, acc.theta_m, level)
    Sn, Ln = zn[f"scores_{bound}"], zn[f"lags_{bound}"]
    Sw, Lw = zw[f"scores_{bound}"], zw[f"lags_{bound}"]
    key = "accepted_frame" if level == "frame" else "accepted_region"
    acc_n = {r: 0 for r in RULES}
    false_n = {r: 0 for r in RULES}
    dists = {r: [] for r in RULES}
    n_attainable = n = 0
    region_top1 = 0
    for i in range(len(qxy)):
        m = keep[i]
        if not m.any():
            continue
        on = recog.retrieval_outcome(qxy[i], ref_ids, rxy, Sn[i], Ln[i], keep=m,
                                     tau_region=tau, acceptance=acc)
        if on.get("outcome") != "SCORED":
            continue
        n += 1
        n_attainable += int(bool(on["attainable"]))
        region_top1 += int(bool(on["region_top1"]))
        ow = recog.retrieval_outcome(qxy[i], ref_ids, rxy, Sw[i], Lw[i], keep=m,
                                     tau_region=tau, acceptance=acc)
        om = recog.retrieval_outcome(qxy[i], ref_ids, rxy,
                                     dualview.fuse_scores(Sn[i], Sw[i], "weakest_view"), Ln[i],
                                     keep=m, tau_region=tau, acceptance=acc)
        st = dualview.strict_agreement(on, ow, ref_xy_map, qxy[i], gate, gate,
                                       float(cfg["tau_agree_m"]), tau)
        for rule, o in (("north", on), ("west", ow), ("weakest", om)):
            if o.get(key):
                acc_n[rule] += 1
                false_n[rule] += int(bool(o["false_region"]))
                dists[rule].append(o["top1_distance_m"])
        if st["accepted"]:
            acc_n["strict"] += 1
            false_n["strict"] += int(bool(st["false_region"]))
            dists["strict"].append(st["candidate_distance_m"])
    return {"n_queries": n, "n_attainable": n_attainable,
            "region_top1_frac": (region_top1 / n if n else None),
            "n_accepted": acc_n, "n_accepted_false_region": false_n,
            "accepted_distance": {r: lagstudy.quantiles(v, (50, 95)) for r, v in dists.items()}}


def _cells(out_dir):
    return {p.stem: p for p in sorted((out_dir / "cells").glob("*.npz"))}


def stage_report(cfg: dict, base, out_dir, json_writer) -> dict:
    """Every table the experiment record quotes. Reads only the saved matrices."""
    cells = _cells(out_dir)
    bounds = [int(b) for b in cfg["bounds"]]
    M = {"study_version": "1.0.0", "bounds": bounds,
         "degrees_per_sample": lagstudy.degrees_per_sample(float(cfg["fov_deg"]),
                                                           int(cfg["profile_n_samples"])),
         "matcher_labels": {str(b): lagstudy.bound_label(b) for b in bounds},
         "figure_eight": {}, "terrain": {}, "hard_city": {}, "calibration": {}, "height": {}}

    def load(name):
        return np.load(cells[name], allow_pickle=True) if name in cells else None

    for run in ("140138", "140549"):
        for src in cfg["sources"]:
            zn = load(f"fig8_{run}_vs_{run}_{src}_north")
            zw = load(f"fig8_{run}_vs_{run}_{src}_west")
            if zn is None or zw is None:
                continue
            entry = {"n_queries": int(len(zn["query_xy"])),
                     "n_references": int(len(zn["reference_xy"])),
                     "height_mode": "constant" if run == "140138" else "varying",
                     "score_vs_distance": {}, "lag_vs_distance": {}, "calibration": {},
                     "recency": {}}
            qo, ro = np.asarray(zn["query_order"]), np.asarray(zn["reference_order"])
            for view, z in (("north", zn), ("west", zw)):
                entry["score_vs_distance"][view] = {
                    lagstudy.bound_label(b): score_vs_distance(cfg, z, b) for b in bounds}
                entry["lag_vs_distance"][view] = {
                    lagstudy.bound_label(b): lag_vs_distance(cfg, z, b) for b in bounds}
                entry["calibration"][view] = {
                    lagstudy.bound_label(b): calibration(cfg, z, b) for b in bounds}
            for k in cfg["recency_k"]:
                keep = np.abs(qo[:, None] - ro[None, :]) > int(k)
                entry["recency"][f"k{int(k)}"] = {
                    "mean_eligible": float(keep.sum(1).mean()),
                    "singleton_frac": float(np.mean(keep.sum(1) <= 1)),
                    "arms": {lagstudy.bound_label(b): selection_table(cfg, zn, zw, keep, b)
                             for b in bounds}}
            for stride in cfg["sparse_strides"]:
                keep = (np.abs(qo[:, None] - ro[None, :]) > 0) & ((ro % int(stride)) == 0)[None, :]
                entry["recency"][f"sparse{int(stride)}"] = {
                    "mean_eligible": float(keep.sum(1).mean()),
                    "singleton_frac": float(np.mean(keep.sum(1) <= 1)),
                    "arms": {lagstudy.bound_label(b): selection_table(cfg, zn, zw, keep, b)
                             for b in bounds}}
            entry["selection"] = entry["recency"]["k0"]["arms"]
            M["figure_eight"][f"{run}/{src}"] = entry

    for src in cfg["sources"]:
        zn = load(f"fig8_140549_vs_140138_{src}_north")
        zw = load(f"fig8_140549_vs_140138_{src}_west")
        if zn is None or zw is None:
            continue
        keep = np.ones((len(zn["query_xy"]), len(zn["reference_xy"])), dtype=bool)
        M["height"][f"cross_run/{src}"] = {
            "description": "query = varying-height run, reference = constant-height run, same XY path",
            "arms": {lagstudy.bound_label(b): selection_table(cfg, zn, zw, keep, b) for b in bounds},
            "north_vs_dh": height_analysis(cfg, zn, 32),
            "west_vs_dh": height_analysis(cfg, zw, 32)}

    for level in ("village", "mountains", "city"):
        for src in cfg["sources"]:
            zn = load(f"terrain_{level}_{src}_north")
            zw = load(f"terrain_{level}_{src}_west")
            if zn is None or zw is None:
                continue
            qk = [str(x) for x in zn["query_keys"]]
            rk = [str(x) for x in zn["reference_keys"]]
            rindex = {k: i for i, k in enumerate(rk)}
            keep = np.ones((len(qk), len(rk)), dtype=bool)
            for i, k in enumerate(qk):
                if k in rindex:
                    keep[i, rindex[k]] = False
            anchor = np.asarray(zn["reference_is_anchor"], dtype=bool)
            sparse = np.broadcast_to(anchor[None, :], keep.shape) & keep
            qxy, rxy = np.asarray(zn["query_xy"]), np.asarray(zn["reference_xy"])
            D = np.hypot(qxy[:, None, 0] - rxy[None, :, 0], qxy[:, None, 1] - rxy[None, :, 1])
            revisit = D > 25.0      # the query must be recognised from a genuinely different pass
            M["terrain"][f"{level}/{src}"] = {
                "n_queries": len(qk), "n_references": len(rk),
                "dense_online": {lagstudy.bound_label(b): selection_table(cfg, zn, zw, keep, b)
                                 for b in bounds},
                "sparse_anchor": {lagstudy.bound_label(b): selection_table(cfg, zn, zw, sparse, b)
                                  for b in bounds},
                "revisit_coverage_lso25": {
                    lagstudy.bound_label(b): revisit_coverage(cfg, zn, zw, revisit, b, "region")
                    for b in bounds}}
            M["calibration"][f"{level}/{src}"] = {
                view: {lagstudy.bound_label(b): calibration(cfg, z, b, 20.0, 500.0) for b in bounds}
                for view, z in (("north", zn), ("west", zw))}

    for src in cfg["sources"]:
        zn = load(f"hardcity_{src}_north")
        zw = load(f"hardcity_{src}_west")
        if zn is None or zw is None:
            continue
        qxy, rxy = np.asarray(zn["query_xy"]), np.asarray(zn["reference_xy"])
        qup = np.asarray(zn["query_up"])
        horiz = qup <= float(np.percentile(qup, 10)) + 5.0
        anchor = np.asarray(zn["reference_is_anchor"], dtype=bool)
        D = np.hypot(qxy[:, None, 0] - rxy[None, :, 0], qxy[:, None, 1] - rxy[None, :, 1])
        mems = {"sparse": (np.broadcast_to(anchor[None, :], D.shape), "frame"),
                "dense_lso": (D > float(cfg["tau_region_m"]), "region")}
        entry = {"n_queries": int(len(qxy)), "n_horizontal": int(horiz.sum()),
                 "n_references": int(len(rxy)), "n_anchors": int(anchor.sum()),
                 "note": "the queried place is absent from these memories: every accept is false"}
        dev = tuple(cfg["dev_gates"]["sparse_frame"])
        gates = {"frozen_0.90_0.15": (0.90, 0.15), f"dev_{dev[0]}_{dev[1]}": dev}
        for mem, (keep, level) in mems.items():
            entry[mem] = {}
            for gname, (ts, tm) in gates.items():
                entry[mem][gname] = {
                    lagstudy.bound_label(b): hard_negative_table(cfg, zn, zw, keep, b, level,
                                                                 float(ts), float(tm), horiz)
                    for b in bounds}
        M["hard_city"][src] = entry
    json_writer(out_dir / "metrics.json", M)
    return M


# --------------------------------------------------------------------------------------------------
# readable report
# --------------------------------------------------------------------------------------------------

def _q(d, key):
    v = d.get(key) if isinstance(d, dict) else None
    return "-" if v is None else f"{v:.1f}"


def _pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def write_report(cfg: dict, out_dir, M: dict, json_writer) -> None:
    """report.md — the tables the experiment record quotes, in reading order."""
    L = [lagstudy.bound_label(b) for b in M["bounds"]]
    w = []
    w.append("# EXP-SKY-013 — matcher lag bound versus positional resolution\n")
    w.append(f"Profile 256 samples over {cfg['fov_deg']:.0f} deg FOV = "
             f"{M['degrees_per_sample']:.4f} deg/sample, so the arms allow "
             + ", ".join(f"{lab} +-{int(b) * M['degrees_per_sample']:.2f} deg"
                         for lab, b in zip(L, M["bounds"])) + ".\n")
    w.append("All arms are read off ONE bound-32 scoring pass; the C1-32 column reproduces the "
             "frozen EXP-SKY-011/012 cells bit-exactly (see score_manifest.json).\n")

    w.append("\n## A. Figure eight — selected-reference distance (self memory, leave-self-out)\n")
    w.append("| run | source | view-rule | " + " | ".join(L) + " |")
    w.append("|---|---|---|" + "---|" * len(L))
    for key, e in sorted(M["figure_eight"].items()):
        run, src = key.split("/")
        for rule in ("north", "west", "weakest", "strict"):
            cells = []
            for lab in L:
                s = e["selection"][lab][rule]
                cells.append(f"p95 {_q(s,'p95')} / max {_q(s,'max')}")
            w.append(f"| {run} ({e['height_mode']}) | {src} | {rule} | " + " | ".join(cells) + " |")

    w.append("\n## B. Figure eight — median score by true separation (the flatness question)\n")
    for key, e in sorted(M["figure_eight"].items()):
        for view in ("north", "west"):
            w.append(f"\n**{key} / {view}**\n")
            bins = e["score_vs_distance"][view][L[0]]
            w.append("| arm | " + " | ".join(f"{int(b['lo_m'])}-{int(b['hi_m'])} m" for b in bins) + " |")
            w.append("|---|" + "---|" * len(bins))
            for lab in L:
                rows = e["score_vs_distance"][view][lab]
                w.append(f"| {lab} | " + " | ".join(
                    ("-" if r["p50"] is None else f"{r['p50']:.4f}") for r in rows) + " |")

    w.append("\n## C. Lag behaviour — median |winning lag| by separation (H3)\n")
    for key, e in sorted(M["figure_eight"].items()):
        for view in ("north", "west"):
            rows = e["lag_vs_distance"][view]["C1-32"]
            w.append(f"\n**{key} / {view}, C1-32**\n")
            w.append("| quantity | " + " | ".join(f"{int(b['lo_m'])}-{int(b['hi_m'])} m" for b in rows) + " |")
            w.append("|---|" + "---|" * len(rows))
            w.append("| median &#124;lag&#124; | " + " | ".join(
                ("-" if r["p50"] is None else f"{r['p50']:.1f}") for r in rows) + " |")
            w.append("| fraction lag != 0 | " + " | ".join(_pct(r.get("frac_nonzero_lag")) for r in rows) + " |")

    w.append("\n## D. Recent-reference exclusion (declared sweep; INT's policy is not visible here)\n")
    w.append("| run | source | k | mean eligible | singleton | " +
             " | ".join(f"{lab} p95 / max" for lab in L) + " |")
    w.append("|---|---|---|---|---|" + "---|" * len(L))
    for key, e in sorted(M["figure_eight"].items()):
        run, src = key.split("/")
        for mem, d in e["recency"].items():
            cells = [f"{_q(d['arms'][lab]['weakest'],'p95')} / {_q(d['arms'][lab]['weakest'],'max')}"
                     for lab in L]
            w.append(f"| {run} | {src} | {mem} | {d['mean_eligible']:.1f} | "
                     f"{_pct(d['singleton_frac'])} | " + " | ".join(cells) + " |")

    w.append("\n## E. Cross-terrain — dense online memory, near references present\n")
    w.append("| level | source | rule | " + " | ".join(f"{lab} p95 / >tau" for lab in L) + " |")
    w.append("|---|---|---|" + "---|" * len(L))
    for key, e in sorted(M["terrain"].items()):
        level, src = key.split("/")
        for rule in ("north", "weakest", "strict"):
            cells = []
            for lab in L:
                s = e["dense_online"][lab][rule]
                cells.append(f"{_q(s,'p95')} / {_pct(s.get('frac_beyond_tau'))}")
            w.append(f"| {level} | {src} | {rule} | " + " | ".join(cells) + " |")

    w.append("\n## F. Hard city — accepted FALSE (the place is absent; every accept is wrong)\n")
    for src, e in sorted(M["hard_city"].items()):
        w.append(f"\n**{src}** — {e['n_horizontal']} horizontal queries, {e['n_references']} "
                 f"references ({e['n_anchors']} anchors)\n")
        w.append("| memory | gate | rule | " + " | ".join(L) + " |")
        w.append("|---|---|---|" + "---|" * len(L))
        for mem in ("sparse", "dense_lso"):
            for gname in sorted(e[mem]):
                for rule in ("north", "west", "strict", "weakest"):
                    cells = [str(e[mem][gname][lab]["accepted_false"][rule]) for lab in L]
                    w.append(f"| {mem} | {gname} | {rule} | " + " | ".join(cells) + " |")

    w.append("\n## G. Calibration — near (true) vs far (wrong place) score separation\n")
    w.append("Separation = near p05 - far p95. Positive means the distributions do not overlap "
             "at those quantiles.\n")
    w.append("| population | view | " + " | ".join(L) + " |")
    w.append("|---|---|" + "---|" * len(L))
    for key, e in sorted(M["calibration"].items()):
        for view in ("north", "west"):
            cells = []
            for lab in L:
                c = e[view][lab]
                sep = c.get("separation_p05near_minus_p95far")
                far = c.get("far") or {}
                cells.append("-" if sep is None else
                             f"{sep:+.3f} (far>=0.90 {far.get('frac_ge_0.90', 0) * 100:.2f}%)")
            w.append(f"| {key} | {view} | " + " | ".join(cells) + " |")

    w.append("\n## H. Height\n")
    for key, e in sorted(M["height"].items()):
        w.append(f"\n**{key}** — {e['description']}\n")
        w.append("| arm | " + " | ".join(f"{r} p95 / max" for r in ("north", "west", "weakest")) + " |")
        w.append("|---|---|---|---|")
        for lab in L:
            a = e["arms"][lab]
            w.append(f"| {lab} | " + " | ".join(
                f"{_q(a[r],'p95')} / {_q(a[r],'max')}" for r in ("north", "west", "weakest")) + " |")

    (out_dir / "report.md").write_text("\n".join(w) + "\n", encoding="utf-8")
