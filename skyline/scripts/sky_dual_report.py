#!/usr/bin/env python
"""EXP-SKY-011 report stage: every rule, computed from the saved paired score matrices.

Imported by ``scripts/sky_dual_study.py --stage report``. Reads ``<out_dir>/cells/scores_*.npz``
(one exact exhaustive C1-32 pass per level × source × view, same query and reference order for
both views) and ``<out_dir>/index.csv``; writes ``queries.csv.gz``, ``metrics.json``, ``report.md``
and the figures. Rules and thresholds are the pre-registered ones in ``configs/sky-dual.json``
and the ``EXP-SKY-011`` record; the DEV (village) sweep selects one gate per rule, source and
memory kind, which is then applied unchanged to the held-out levels.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from hsreloc.simret import dualview as dv                                       # noqa: E402
from hsreloc.simret import recog                                                # noqa: E402
from sky_relpose_study import (LEVEL_COLOR, LEVEL_ORDER, _json, _open, _plt,    # noqa: E402
                               _read_csv, _resolve, _save)
from sky_dual_study import STUDY_VERSION, _index_rows, _key, _write_csv         # noqa: E402

CHANNELS = ("north", "west", "min", "mean")
RULES = ("north", "west", "strict", "min", "mean")
RULE_LABEL = {"north": "North only", "west": "West only", "strict": "North+West strict agreement",
              "min": "North+West weakest view", "mean": "North+West mean score"}
KEEP_FIELDS = ("top1_id", "top1_score", "top2_score", "frame_margin", "region_level_margin",
               "region_top1", "region_in_top_k", "exact_top1", "exact_in_top_k", "false_region",
               "best_in_region_score", "best_out_region_score", "region_margin", "ambiguous",
               "top1_distance_m", "top1_lag", "nearest_lag", "nearest_score", "accepted_frame",
               "accepted_region", "outcome", "region_rank")
SRC_LABEL = {"sim_exact": "GT", "segformer": "SegFormer"}


# --------------------------------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------------------------------

def _trim(o: dict) -> dict:
    return {k: o.get(k) for k in KEEP_FIELDS}


def _memories(cfg: dict) -> list:
    mems = [("sparse", None), ("sparse_lso", None), ("dense_lso", None)]
    mems += [("dense", float(h)) for h in cfg["dense_holdout_radii_m"]]
    return mems


def _gate_level(memory: str) -> str:
    return "frame" if memory.startswith("sparse") else "region"


def build_rows(cfg: dict, level: str, source: str, zN, zW, meta: dict) -> tuple:
    """One row per (query, memory instance) with the four channels' trimmed outcomes and the
    gate-independent strict-agreement facts (agreement, candidate distance)."""
    tau = float(cfg["tau_region_m"])
    tau_agree = float(cfg["tau_agree_m"])
    tau_agree2 = float(cfg["tau_agree_sensitivity_m"][0]) if cfg.get("tau_agree_sensitivity_m") else None
    k = int(cfg["recall_k"])
    acc = recog.Acceptance(float(cfg["acceptance"]["theta_s"]), float(cfg["acceptance"]["theta_m"]))
    qk = [str(x) for x in zN["query_keys"]]
    rk = [str(x) for x in zN["reference_keys"]]
    if qk != [str(x) for x in zW["query_keys"]] or rk != [str(x) for x in zW["reference_keys"]]:
        raise SystemExit(f"{level}/{source}: North and West matrices are not paired in the same order")
    SN, LN, SW, LW = zN["scores"], zN["lags"], zW["scores"], zW["lags"]
    rxy = np.asarray(zN["reference_xy"], dtype=np.float64)
    qxy = np.asarray(zN["query_xy"], dtype=np.float64)
    is_anchor = np.asarray(zN["reference_is_anchor"], dtype=bool)
    ref_xy = {key: (float(rxy[i, 0]), float(rxy[i, 1])) for i, key in enumerate(rk)}
    rows = []
    frozen = dv.Gate(acc.theta_s, acc.theta_m, "frame")
    for qi, key in enumerate(qk):
        m = meta[key]
        dist = np.hypot(rxy[:, 0] - qxy[qi, 0], rxy[:, 1] - qxy[qi, 1])
        not_self = np.array([r != key for r in rk])
        fused = {"min": dv.fuse_scores(SN[qi], SW[qi], "weakest_view"),
                 "mean": dv.fuse_scores(SN[qi], SW[qi], "mean_score")}
        for memory, h in _memories(cfg):
            if memory == "sparse":
                keep = is_anchor
            elif memory == "sparse_lso":
                keep = is_anchor & (dist > tau)
            elif memory == "dense_lso":
                keep = not_self & (dist > tau)
            else:
                keep = not_self & (dist > h)
            if not keep.any():
                continue
            outs = {
                "north": recog.retrieval_outcome(qxy[qi], rk, rxy, SN[qi], LN[qi], keep=keep, tau_region=tau, k=k, acceptance=acc),
                "west": recog.retrieval_outcome(qxy[qi], rk, rxy, SW[qi], LW[qi], keep=keep, tau_region=tau, k=k, acceptance=acc),
                "min": recog.retrieval_outcome(qxy[qi], rk, rxy, fused["min"], LN[qi], keep=keep, tau_region=tau, k=k, acceptance=acc),
                "mean": recog.retrieval_outcome(qxy[qi], rk, rxy, fused["mean"], LN[qi], keep=keep, tau_region=tau, k=k, acceptance=acc),
            }
            if outs["north"].get("outcome") == "EMPTY_DATABASE":
                continue
            glev = _gate_level(memory)
            g = dv.Gate(acc.theta_s, acc.theta_m, glev)
            sa = dv.strict_agreement(outs["north"], outs["west"], ref_xy, qxy[qi], g, g, tau_agree, tau)
            sa2 = dv.agreement(outs["north"], outs["west"], ref_xy, tau_agree2) if tau_agree2 else {"agree": None}
            primary = (m["segment_kind"] == "star_swipe" and not m["hard_tag"] and m["leg_axis"] != "vertical"
                       and (m["vertical_up_m"] is None or abs(m["vertical_up_m"]) < 1.0))
            row = {
                "level": level, "split": m["split"], "source": source, "memory": memory, "holdout_m": h,
                "query_key": key, "segment_id": m["segment_id"], "segment_kind": m["segment_kind"],
                "site": m["site_id"], "hard_tag": m["hard_tag"], "phase": m["phase"],
                "leg_axis": m["leg_axis"], "leg_direction": m["leg_direction"],
                "vertical_up_m": m["vertical_up_m"], "total_displacement_m": m["total_displacement_m"],
                "index_in_segment": m["index_in_segment"], "east_m": m["east_m"], "north_m": m["north_m"],
                "up_m": m["up_m"], "primary": primary, "horizontal": m["leg_axis"] != "vertical",
                "n_references": int(keep.sum()), "d_near_m": outs["north"]["d_near_m"],
                "nearest_key": outs["north"]["nearest_id"], "attainable": outs["north"]["attainable"],
                "gate_level": glev,
                "sa_agree": sa["agree"], "sa_agree_tau2": sa2["agree"], "sa_same_reference": sa["same_reference"],
                "sa_top1_separation_m": sa["top1_separation_m"], "sa_candidate_id": sa["candidate_id"],
                "sa_candidate_distance_m": sa["candidate_distance_m"], "sa_false_region": sa["false_region"],
                "sa_dual_region_top1": sa["dual_region_top1"], "sa_dual_region_in_top_k": sa["dual_region_in_top_k"],
                "sa_min_score": sa["min_score"], "sa_accepted_frozen": sa["accepted"],
                "sa_accepted_false_frozen": sa["accepted_false"],
                "_out": {c: _trim(outs[c]) for c in CHANNELS},
            }
            rows.append(row)
    return rows, ref_xy


def flat(row: dict) -> dict:
    d = {k: v for k, v in row.items() if k != "_out"}
    for c in CHANNELS:
        for k, v in row["_out"][c].items():
            d[f"{c}_{k}"] = v
    return d


# --------------------------------------------------------------------------------------------------
# rules at a gate
# --------------------------------------------------------------------------------------------------

def accepted_at(row: dict, rule: str, gate: dv.Gate) -> bool:
    if rule == "strict":
        return bool(dv.view_passes(row["_out"]["north"], gate) and dv.view_passes(row["_out"]["west"], gate)
                    and row["sa_agree"])
    return bool(dv.view_passes(row["_out"][rule], gate))


def false_of(row: dict, rule: str) -> bool:
    return bool(row["sa_false_region"]) if rule == "strict" else bool(row["_out"][rule]["false_region"])


def top1_of(row: dict, rule: str) -> bool:
    return bool(row["sa_dual_region_top1"]) if rule == "strict" else bool(row["_out"][rule]["region_top1"])


def topk_of(row: dict, rule: str) -> bool:
    return bool(row["sa_dual_region_in_top_k"]) if rule == "strict" else bool(row["_out"][rule]["region_in_top_k"])


def exact_of(row: dict, rule: str) -> bool:
    if rule == "strict":
        return bool(row["_out"]["north"]["exact_top1"] and row["_out"]["west"]["exact_top1"] and row["sa_agree"])
    return bool(row["_out"][rule]["exact_top1"])


def exactk_of(row: dict, rule: str) -> bool:
    if rule == "strict":
        return bool(row["_out"]["north"]["exact_in_top_k"] and row["_out"]["west"]["exact_in_top_k"])
    return bool(row["_out"][rule]["exact_in_top_k"])


def score_of(row: dict, rule: str):
    return row["sa_min_score"] if rule == "strict" else row["_out"][rule]["top1_score"]


def margin_of(row: dict, rule: str):
    if rule == "strict":
        a, b = row["_out"]["north"]["region_margin"], row["_out"]["west"]["region_margin"]
        return None if (a is None or b is None) else min(a, b)
    return row["_out"][rule]["region_margin"]


def ambiguous_of(row: dict, rule: str) -> bool:
    if rule == "strict":
        return bool(row["_out"]["north"]["ambiguous"] or row["_out"]["west"]["ambiguous"])
    return bool(row["_out"][rule]["ambiguous"])


def _rate(flags):
    flags = list(flags)
    return (sum(1 for f in flags if f) / len(flags)) if flags else None


def _median(vals):
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def summarize(rows: list, rule: str, gate: dv.Gate) -> dict:
    n = len(rows)
    acc = [r for r in rows if accepted_at(r, rule, gate)]
    acc_false = [r for r in acc if false_of(r, rule)]
    att = [r for r in rows if r["attainable"]]
    sites = {r["site"] for r in rows}
    return {"rule": rule, "theta_s": gate.theta_s, "theta_m": gate.theta_m, "gate_level": gate.level,
            "n": n, "n_sites": len(sites), "n_segments": len({r["segment_id"] for r in rows}),
            "n_attainable": len(att),
            "exact_top1": _rate(exact_of(r, rule) for r in rows),
            "exact_recall_k": _rate(exactk_of(r, rule) for r in rows),
            "region_top1": _rate(top1_of(r, rule) for r in rows),
            "region_recall_k": _rate(topk_of(r, rule) for r in rows),
            "top1_score_median": _median(score_of(r, rule) for r in rows),
            "region_margin_median": _median(margin_of(r, rule) for r in att),
            "ambiguity_rate": _rate(ambiguous_of(r, rule) for r in att),
            "false_region_rate": _rate(false_of(r, rule) for r in rows),
            "n_accepted": len(acc), "coverage": (len(acc) / n) if n else None,
            "n_accepted_false": len(acc_false),
            "accepted_false_rate": (len(acc_false) / n) if n else None,
            "p_false_given_accepted": (len(acc_false) / len(acc)) if acc else None,
            "accepted_precision": (1.0 - len(acc_false) / len(acc)) if acc else None,
            "site_min_region_top1": (min(_rate(top1_of(r, rule) for r in rows if r["site"] == s) for s in sites)
                                     if sites else None),
            "accepted_false_sites": sorted({r["site"] for r in acc_false})}


# --------------------------------------------------------------------------------------------------
# populations
# --------------------------------------------------------------------------------------------------

def population(rows: list, name: str, attainable_sparse: dict, attainable_dense: dict) -> list:
    if name == "in_coverage_sparse":
        return [r for r in rows if r["memory"] == "sparse" and r["primary"] and r["attainable"]]
    if name == "ooc_sparse":
        return [r for r in rows if r["memory"] == "sparse" and r["horizontal"] and not r["attainable"]]
    if name == "lso_sparse":
        return [r for r in rows if r["memory"] == "sparse_lso" and r["primary"] and attainable_sparse.get(r["query_key"])]
    if name == "hard_sparse":
        return [r for r in rows if r["memory"] == "sparse" and r["hard_tag"] and r["horizontal"]]
    if name == "vertical_sparse":
        return [r for r in rows if r["memory"] == "sparse" and r["leg_axis"] == "vertical" and not r["hard_tag"]]
    if name == "transit_sparse":
        return [r for r in rows if r["memory"] == "sparse" and r["segment_kind"] == "transit"]
    if name == "in_coverage_dense":
        return [r for r in rows if r["memory"] == "dense" and r["holdout_m"] == 0.0 and r["primary"] and r["attainable"]]
    if name == "lso_dense":
        return [r for r in rows if r["memory"] == "dense_lso" and r["primary"] and attainable_dense.get(r["query_key"])]
    if name == "transit_dense":
        return [r for r in rows if r["memory"] == "dense" and r["holdout_m"] == 0.0 and r["segment_kind"] == "transit"]
    raise KeyError(name)


POPS_SPARSE = ("in_coverage_sparse", "ooc_sparse", "lso_sparse", "hard_sparse", "vertical_sparse", "transit_sparse")
POPS_DENSE = ("in_coverage_dense", "lso_dense", "transit_dense")
HARD_OF = {"sparse": ("ooc_sparse", "lso_sparse"), "dense": ("lso_dense",)}
IN_OF = {"sparse": "in_coverage_sparse", "dense": "in_coverage_dense"}


# --------------------------------------------------------------------------------------------------
# bins / radii
# --------------------------------------------------------------------------------------------------

def to_bin_row(row: dict, rule: str, gate: dv.Gate) -> dict:
    acc = accepted_at(row, rule, gate)
    return {"site": row["site"], "d_near_m": row["d_near_m"], "attainable": row["attainable"],
            "accepted_frame": acc, "accepted_region": acc, "false_region": false_of(row, rule),
            "region_top1": top1_of(row, rule), "region_in_top_k": topk_of(row, rule),
            "exact_top1": exact_of(row, rule), "exact_in_top_k": exactk_of(row, rule),
            "top1_score": score_of(row, rule), "best_in_region_score": (row["_out"]["north"]["best_in_region_score"]
                                                                        if rule == "strict" else row["_out"][rule]["best_in_region_score"]),
            "region_margin": margin_of(row, rule), "ambiguous": ambiguous_of(row, rule)}


def dedupe_dense(rows: list) -> list:
    seen, out = set(), []
    for r in rows:
        key = (r["query_key"], r["nearest_key"], r["n_references"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# --------------------------------------------------------------------------------------------------
# temporal confirmation
# --------------------------------------------------------------------------------------------------

def temporal_rows(rows: list, k: int, gate: dv.Gate, ref_xy: dict, tau_agree: float, tau: float) -> dict:
    """North-only temporal confirmation per reconstructed segment (never across a teleport)."""
    by_seg = defaultdict(list)
    for r in rows:
        by_seg[r["segment_id"]].append(r)
    out = {}
    for seg, g in by_seg.items():
        g.sort(key=lambda r: r["index_in_segment"])
        # consecutive means consecutive captures: a gap in index_in_segment breaks the window
        chunks, cur = [], [g[0]]
        for a, b in zip(g[:-1], g[1:]):
            if b["index_in_segment"] == a["index_in_segment"] + 1:
                cur.append(b)
            else:
                chunks.append(cur)
                cur = [b]
        chunks.append(cur)
        for chunk in chunks:
            seq = [r["_out"]["north"] for r in chunk]
            xy = [(r["east_m"], r["north_m"]) for r in chunk]
            res = dv.temporal_confirmation(seq, xy, k, gate, ref_xy, tau_agree, tau)
            for r, t in zip(chunk, res):
                out[r["query_key"]] = t
    return out


def dual_temporal(rows: list, gate: dv.Gate, tau_agree: float) -> dict:
    """Strict dual agreement at two consecutive captures of one segment, candidates agreeing."""
    by_seg = defaultdict(list)
    for r in rows:
        by_seg[r["segment_id"]].append(r)
    out = {}
    for seg, g in by_seg.items():
        g.sort(key=lambda r: r["index_in_segment"])
        prev = None
        for r in g:
            acc_now = accepted_at(r, "strict", gate)
            ok = False
            if prev is not None and r["index_in_segment"] == prev["index_in_segment"] + 1:
                if acc_now and accepted_at(prev, "strict", gate):
                    a, b = r["sa_candidate_id"], prev["sa_candidate_id"]
                    ok = a is not None and b is not None and a == b
                    if not ok and a is not None and b is not None:
                        # candidates within tau_agree count as the same place
                        ok = r["_cand_xy"] is not None and prev["_cand_xy"] is not None and \
                            math.hypot(r["_cand_xy"][0] - prev["_cand_xy"][0], r["_cand_xy"][1] - prev["_cand_xy"][1]) <= tau_agree
            out[r["query_key"]] = {"accepted": ok, "false_region": bool(r["sa_false_region"]),
                                   "accepted_false": bool(ok and r["sa_false_region"])}
            prev = r
    return out


# --------------------------------------------------------------------------------------------------
# relative position (secondary)
# --------------------------------------------------------------------------------------------------

def relative_position(cfg: dict, level: str, source: str, zN, zW, meta: dict) -> dict:
    """Same-swipe horizontal pairs on the dense memory: what the two lags carry about Δp."""
    qk = [str(x) for x in zN["query_keys"]]
    rk = [str(x) for x in zN["reference_keys"]]
    rseg = [str(x) for x in zN["reference_segments"]]
    rxy, qxy = np.asarray(zN["reference_xy"]), np.asarray(zN["query_xy"])
    LN, LW, SN, SW = zN["lags"], zW["lags"], zN["scores"], zW["scores"]
    max_h = 100.0
    recs = []
    for qi, key in enumerate(qk):
        m = meta[key]
        if m["segment_kind"] != "star_swipe" or m["hard_tag"] or m["leg_axis"] == "vertical":
            continue
        if m["vertical_up_m"] is not None and abs(m["vertical_up_m"]) >= 1.0:
            continue
        for ri, rkey in enumerate(rk):
            if rseg[ri] != m["segment_id"] or rkey == key:
                continue
            rm = meta[rkey]
            if rm["vertical_up_m"] is not None and abs(rm["vertical_up_m"]) >= 1.0:
                continue
            dE, dN = float(qxy[qi, 0] - rxy[ri, 0]), float(qxy[qi, 1] - rxy[ri, 1])
            d = math.hypot(dE, dN)
            if d == 0.0 or d > max_h:
                continue
            if not (np.isfinite(SN[qi, ri]) and np.isfinite(SW[qi, ri]) and SN[qi, ri] >= 0.5 and SW[qi, ri] >= 0.5):
                continue
            recs.append({"site": m["site_id"], "dE": dE, "dN": dN, "d": d, "lagN": float(LN[qi, ri]),
                         "lagW": float(LW[qi, ri]), "sN": float(SN[qi, ri]), "sW": float(SW[qi, ri])})
    if len(recs) < 20:
        return {"n_pairs": len(recs), "note": "too few matched same-swipe pairs"}
    X = np.array([[r["dE"], r["dN"]] for r in recs])
    lagN = np.array([r["lagN"] for r in recs])
    lagW = np.array([r["lagW"] for r in recs])

    def ols(y, X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else None
        return beta.tolist(), r2

    bN, r2N = ols(lagN, X)
    bW, r2W = ols(lagW, X)
    # single-axis gains per site (samples per metre): lag_N vs ΔE, lag_W vs ΔN
    per_site = {}
    for s in sorted({r["site"] for r in recs}):
        sub = [r for r in recs if r["site"] == s]
        if len(sub) < 8:
            continue
        e = np.array([r["dE"] for r in sub]); n_ = np.array([r["dN"] for r in sub])
        ln = np.array([r["lagN"] for r in sub]); lw = np.array([r["lagW"] for r in sub])
        gN = float((e * ln).sum() / (e * e).sum()) if (e * e).sum() > 0 else None
        gW = float((n_ * lw).sum() / (n_ * n_).sum()) if (n_ * n_).sum() > 0 else None
        per_site[s] = {"n": len(sub), "gain_lagN_per_m_east": gN, "gain_lagW_per_m_north": gW}
    gains_N = [v["gain_lagN_per_m_east"] for v in per_site.values() if v["gain_lagN_per_m_east"]]
    gains_W = [v["gain_lagW_per_m_north"] for v in per_site.values() if v["gain_lagW_per_m_north"]]
    # direction observability with level-pooled single-axis gains: (ΔE, ΔN) ∝ (lagN/gN, lagW/gW)
    gN = float((X[:, 0] * lagN).sum() / (X[:, 0] ** 2).sum())
    gW = float((X[:, 1] * lagW).sum() / (X[:, 1] ** 2).sum())
    ang_err_pair, ang_err_single, len_ratio = [], [], []
    for r in recs:
        true_ang = math.atan2(r["dN"], r["dE"])
        if gN and gW and (r["lagN"] != 0 or r["lagW"] != 0):
            pe, pn = r["lagN"] / gN, r["lagW"] / gW
            ang_err_pair.append(abs(((math.atan2(pn, pe) - true_ang + math.pi) % (2 * math.pi)) - math.pi))
            len_ratio.append(math.hypot(pe, pn) / r["d"])
        # a single North lag only says "east or west of the reference": direction error of the best
        # single-lag guess (pure east/west) against the true direction
        guess = 0.0 if r["lagN"] / (gN or 1.0) >= 0 else math.pi
        ang_err_single.append(abs(((guess - true_ang + math.pi) % (2 * math.pi)) - math.pi))
    return {"n_pairs": len(recs), "n_sites": len(per_site),
            "ols_lagN_on_dE_dN": {"beta": bN, "r2": r2N}, "ols_lagW_on_dE_dN": {"beta": bW, "r2": r2W},
            "pooled_gain_lagN_per_m_east": gN, "pooled_gain_lagW_per_m_north": gW,
            "per_site": per_site,
            "site_gain_spread_lagN": ([min(gains_N), max(gains_N)] if gains_N else None),
            "site_gain_spread_lagW": ([min(gains_W), max(gains_W)] if gains_W else None),
            "direction_error_deg_median_pair": (float(np.degrees(np.median(ang_err_pair))) if ang_err_pair else None),
            "direction_error_deg_p90_pair": (float(np.degrees(np.percentile(ang_err_pair, 90))) if ang_err_pair else None),
            "direction_error_deg_median_single_north": (float(np.degrees(np.median(ang_err_single))) if ang_err_single else None),
            "length_ratio_pooled_gain_median": (float(np.median(len_ratio)) if len_ratio else None),
            "length_ratio_pooled_gain_p10_p90": ([float(np.percentile(len_ratio, 10)), float(np.percentile(len_ratio, 90))] if len_ratio else None),
            "_recs": recs}


# --------------------------------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------------------------------

def cost(cfg: dict, base: Path, out: Path) -> dict:
    res = {"profile_bytes_per_reference_per_view": int(cfg["profile_n_samples"]) * 8,
           "profile_bytes_per_reference_dual": int(cfg["profile_n_samples"]) * 8 * 2}
    idx = Path(cfg["segformer"]["mask_root"]) / "inference_index.csv"
    if idx.exists():
        by = defaultdict(list)
        for r in _read_csv(idx):
            if r.get("seconds") not in ("", None):
                view = "west" if r["condition"].endswith("-west") else "north"
                by[view].append(float(r["seconds"]))
        res["segformer_seconds_per_frame"] = {v: {"n": len(t), "median": float(np.median(t)), "mean": float(np.mean(t))}
                                              for v, t in by.items()}
        if "north" in by and "west" in by:
            res["segformer_dual_seconds_per_observation_median"] = float(np.median(by["north"]) + np.median(by["west"]))
            res["segformer_dual_over_single_ratio"] = float((np.median(by["north"]) + np.median(by["west"])) / np.median(by["north"]))
    mp = out / "score_manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        t = defaultdict(list)
        for cell, info in m.get("cells", {}).items():
            for view, v in info.get("views", {}).items():
                if "score_s_per_query" in v:
                    t[view].append((v["score_s_per_query"], v["score_s_per_query_per_reference"], v["n_references"]))
        res["c1_bank_scoring"] = {v: {"ms_per_query_median": float(np.median([a for a, _, _ in x]) * 1e3),
                                      "us_per_query_per_reference_median": float(np.median([b for _, b, _ in x]) * 1e6),
                                      "n_references": [c for _, _, c in x]} for v, x in t.items()}
        res["c1_dual_over_single_ratio"] = 2.0
    # an idle-CPU micro-benchmark on synthetic profiles of the study's size (the bounded-lag NCC
    # cost is content-independent): the in-run timings above were taken while SegFormer inference
    # shared the CPU
    import time as _time
    rng = np.random.default_rng(0)
    n = int(cfg["profile_n_samples"])
    bench = {}
    for R in (30, 340, 540):
        profs = [rng.standard_normal(n) for _ in range(R)]
        bank = recog.ReferenceBank([f"r{i}" for i in range(R)], profs, max_lag=int(cfg["max_lag_samples"]),
                                   min_overlap_frac=float(cfg["min_overlap_frac"]))
        q = rng.standard_normal(n)
        bank.score(q)
        t0 = _time.perf_counter()
        for _ in range(10):
            bank.score(q)
        bench[f"R{R}"] = {"ms_per_query_single_view": (_time.perf_counter() - t0) / 10 * 1e3,
                          "ms_per_query_dual": (_time.perf_counter() - t0) / 10 * 1e3 * 2}
    res["c1_bank_scoring_idle_benchmark"] = bench
    return res


# --------------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------------

def _fig_ablation(out: Path, abl: dict):
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex="col")
    rules = list(RULES)
    x = np.arange(len(rules))
    for j, level in enumerate(LEVEL_ORDER):
        for i, key in enumerate(("p_false_given_accepted", "coverage")):
            ax = axes[i, j]
            for s, (src, off, col) in enumerate((("sim_exact", -0.18, "#1f77b4"), ("segformer", 0.18, "#ff7f0e"))):
                vals = [abl.get(f"{level}/{src}/{r}", {}).get(key) for r in rules]
                vals = [np.nan if v is None else v for v in vals]
                ax.bar(x + off, vals, width=0.34, color=col, label=SRC_LABEL[src])
                if key == "p_false_given_accepted":
                    for xi, r in zip(x, rules):
                        e = abl.get(f"{level}/{src}/{r}", {})
                        ax.text(xi + off, (e.get("p_false_given_accepted") or 0) + 0.01,
                                f"{e.get('n_accepted_false', '')}/{e.get('n_accepted', '')}", ha="center", fontsize=6, rotation=90)
            ax.set_xticks(x)
            ax.set_xticklabels(["N", "W", "N+W strict", "N+W min", "N+W mean"], fontsize=8)
            ax.set_title(f"{level}: {'P(false region | accepted)' if i == 0 else 'accepted coverage'}", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
            ax.set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("In-coverage primary population, sparse anchor memory, DEV-selected gates (labels: accepted-false / accepted)", fontsize=9)
    return _save(fig, out / "figures" / "fig1_ablation_sparse.jpg")


def _fig_bins(out: Path, name: str, groups: dict, key: str, ylabel: str, title: str, rules: tuple, memory: str):
    plt = _plt()
    labels = None
    fig, axes = plt.subplots(1, len(rules), figsize=(5.2 * len(rules), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, rule in zip(axes, rules):
        for level in LEVEL_ORDER:
            for src, ls in (("sim_exact", "-"), ("segformer", "--")):
                g = groups.get(f"{memory}/{level}/{src}/{rule}")
                if g is None:
                    continue
                labels = [e["bin"] for e in g["bins"]]
                ys = [e[key] if (e["n"] > 0 and e[key] is not None) else np.nan for e in g["bins"]]
                ax.plot(range(len(labels)), ys, ls, marker="o", color=LEVEL_COLOR[level],
                        label=f"{level} {SRC_LABEL[src]}", ms=4)
                for i, e in enumerate(g["bins"]):
                    if e["n"] and not e["supported"]:
                        ax.plot(i, ys[i], "x", color="k", ms=8)
        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_xlabel("d_near to the nearest stored reference (m)")
        ax.set_title(RULE_LABEL[rule], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(f"{title} — x marks an unsupported bin (< 30 queries or < 3 sites)", fontsize=9)
    return _save(fig, out / "figures" / name)


def _fig_scatter(out: Path, rows_by: dict):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, level in zip(axes, LEVEL_ORDER):
        rows = [r for r in rows_by.get((level, "sim_exact"), []) if r["memory"] == "sparse" and r["horizontal"]]
        cats = {"both correct": ("#2ca02c", "o"), "North false only": ("#ff7f0e", "^"),
                "West false only": ("#1f77b4", "v"), "both false": ("#d62728", "x")}
        for cat, (col, mk) in cats.items():
            xs, ys = [], []
            for r in rows:
                fn, fw = r["_out"]["north"]["false_region"], r["_out"]["west"]["false_region"]
                c = "both correct" if not fn and not fw else "North false only" if fn and not fw else \
                    "West false only" if fw and not fn else "both false"
                if c != cat:
                    continue
                a, b = r["_out"]["north"]["top1_score"], r["_out"]["west"]["top1_score"]
                if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
                    continue
                xs.append(a); ys.append(b)
            ax.scatter(xs, ys, s=10, c=col, marker=mk, label=f"{cat} ({len(xs)})", alpha=0.7)
        ax.set_xlabel("North top-1 score")
        ax.set_ylabel("West top-1 score")
        ax.set_title(f"{level}: sparse memory, GT curves, all horizontal queries", fontsize=9)
        ax.axvline(0.9, color="grey", ls=":", lw=0.8); ax.axhline(0.9, color="grey", ls=":", lw=0.8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    return _save(fig, out / "figures" / "fig5_north_vs_west_top1_score.jpg")


def _fig_sweep(out: Path, sweeps: dict, cfg: dict):
    plt = _plt()
    ts, tm = cfg["gate_grid"]["theta_s"], cfg["gate_grid"]["theta_m"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for j, rule in enumerate(("north", "west", "strict")):
        for i, src in enumerate(("sim_exact", "segformer")):
            ax = axes[i, j]
            sw = sweeps.get(f"sparse/{src}/{rule}")
            if not sw:
                ax.set_visible(False)
                continue
            cov = np.full((len(tm), len(ts)), np.nan)
            fal = np.zeros((len(tm), len(ts)))
            for e in sw["grid"]:
                a, b = tm.index(e["theta_m"]), ts.index(e["theta_s"])
                cov[a, b] = e["coverage"] or 0.0
                fal[a, b] = e["n_accepted_false"] + e.get("hard_false", 0)
            im = ax.imshow(cov, vmin=0, vmax=1, cmap="Blues", origin="lower")
            for a in range(len(tm)):
                for b in range(len(ts)):
                    ax.text(b, a, f"{cov[a, b]:.2f}\n{int(fal[a, b])}F", ha="center", va="center", fontsize=7,
                            color="red" if fal[a, b] > 0 else "black")
            ax.set_xticks(range(len(ts))); ax.set_xticklabels(ts)
            ax.set_yticks(range(len(tm))); ax.set_yticklabels(tm)
            ax.set_xlabel("theta_s"); ax.set_ylabel("theta_m")
            sel = sw["selected"]
            ax.set_title(f"{RULE_LABEL[rule]} — {SRC_LABEL[src]} (DEV village, sparse); selected {sel['theta_s']}/{sel['theta_m']}", fontsize=8)
    fig.suptitle("DEV sweep: in-coverage coverage (colour, number) and total accepted-false incl. hard negatives (nF)", fontsize=9)
    return _save(fig, out / "figures" / "fig6_dev_sweep.jpg")


def _fig_temporal(out: Path, temp: dict):
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    keys = ["north_k1", "north_k2", "north_k3", "strict", "strict_k2"]
    labels = ["N k=1", "N k=2", "N k=3", "N+W", "N+W k=2"]
    for j, level in enumerate(LEVEL_ORDER):
        for i, pop in enumerate(("in_coverage_sparse", "hard_sparse_all")):
            ax = axes[i, j]
            for src, off, col in (("sim_exact", -0.18, "#1f77b4"), ("segformer", 0.18, "#ff7f0e")):
                e = temp.get(f"{level}/{src}", {}).get(pop, {})
                val = [e.get(k, {}).get("coverage" if pop == "in_coverage_sparse" else "accepted_rate") for k in keys]
                val = [np.nan if v is None else v for v in val]
                ax.bar(np.arange(len(keys)) + off, val, width=0.34, color=col, label=SRC_LABEL[src])
                for xi, k in enumerate(keys):
                    ee = e.get(k, {})
                    ax.text(xi + off, (val[xi] if np.isfinite(val[xi]) else 0) + 0.01,
                            f"{ee.get('n_accepted_false', '')}F", ha="center", fontsize=6)
            ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(f"{level}: {'in-coverage coverage' if i == 0 else 'hard-negative acceptance (OOC + LSO)'}", fontsize=9)
            ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Temporal confirmation (North, k consecutive 10 m captures) vs single-location dual agreement, sparse memory, DEV gates (nF = accepted-false)", fontsize=9)
    return _save(fig, out / "figures" / "fig7_temporal_vs_dual.jpg")


def _fig_lags(out: Path, rel: dict):
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, level in enumerate(LEVEL_ORDER):
        r = rel.get(f"{level}/sim_exact")
        if not r or "_recs" not in r:
            axes[0, j].set_visible(False); axes[1, j].set_visible(False)
            continue
        recs = r["_recs"]
        axes[0, j].scatter([x["dE"] for x in recs], [x["lagN"] for x in recs], s=6, alpha=0.5, label="lag_N vs ΔE")
        axes[0, j].scatter([x["dN"] for x in recs], [x["lagW"] for x in recs], s=6, alpha=0.5, label="lag_W vs ΔN")
        axes[0, j].set_xlabel("true displacement component (m)"); axes[0, j].set_ylabel("C1 lag (samples)")
        axes[0, j].set_title(f"{level}: same-swipe pairs ≤ 100 m, GT (n={len(recs)})", fontsize=9)
        axes[0, j].legend(fontsize=7); axes[0, j].grid(alpha=0.3)
        gN, gW = r["pooled_gain_lagN_per_m_east"], r["pooled_gain_lagW_per_m_north"]
        pe = [x["lagN"] / gN for x in recs]; pn = [x["lagW"] / gW for x in recs]
        axes[1, j].scatter([x["dE"] for x in recs], [x["dN"] for x in recs], s=6, c="grey", alpha=0.4, label="true Δp")
        axes[1, j].scatter(pe, pn, s=6, c="#d62728", alpha=0.4, label="from (lag_N, lag_W), pooled gains")
        axes[1, j].set_xlabel("ΔE (m)"); axes[1, j].set_ylabel("ΔN (m)"); axes[1, j].set_aspect("equal")
        axes[1, j].set_title(f"direction error median {r['direction_error_deg_median_pair']:.0f}°, length ratio p10–p90 "
                             f"{r['length_ratio_pooled_gain_p10_p90'][0]:.2f}–{r['length_ratio_pooled_gain_p10_p90'][1]:.2f}", fontsize=8)
        axes[1, j].legend(fontsize=7); axes[1, j].grid(alpha=0.3)
    return _save(fig, out / "figures" / "fig8_lag_pair_vs_displacement.jpg")


# --------------------------------------------------------------------------------------------------
# markdown helpers
# --------------------------------------------------------------------------------------------------

def _pct(v):
    return "—" if v is None else f"{100 * v:.0f} %"


def _num(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def _summary_md(entries: list, cols: list) -> list:
    head = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for e in entries:
        lines.append("| " + " | ".join(fmt(e) for _, fmt in cols) + " |")
    return lines


# --------------------------------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------------------------------

def stage_report(cfg: dict, base: Path) -> None:
    out = _resolve(base, cfg["out_dir"])
    cells = out / "cells"
    rows_idx = _index_rows(cfg, base)
    meta = {_key(r): r for r in rows_idx}
    tau = float(cfg["tau_region_m"])
    tau_agree = float(cfg["tau_agree_m"])
    k = int(cfg["recall_k"])
    frozen_s, frozen_m = float(cfg["acceptance"]["theta_s"]), float(cfg["acceptance"]["theta_m"])
    grid_s, grid_m = cfg["gate_grid"]["theta_s"], cfg["gate_grid"]["theta_m"]
    dev = cfg["dev_level"]

    rows_by, ref_xy_by, rel = {}, {}, {}
    sources_present = []
    for source in cfg["sources"]:
        for level in LEVEL_ORDER:
            pn, pw = cells / f"scores_{level}_{source}_north.npz", cells / f"scores_{level}_{source}_west.npz"
            if not (pn.exists() and pw.exists()):
                print(f"[report] {level}/{source}: matrices absent, skipped")
                continue
            with np.load(pn, allow_pickle=False) as zN, np.load(pw, allow_pickle=False) as zW:
                rows, ref_xy = build_rows(cfg, level, source, zN, zW, meta)
                rel[f"{level}/{source}"] = relative_position(cfg, level, source, zN, zW, meta)
            rows_by[(level, source)] = rows
            ref_xy_by[(level, source)] = ref_xy
            if source not in sources_present:
                sources_present.append(source)
            print(f"[report] {level}/{source}: {len(rows)} query x memory rows")
    if not rows_by:
        raise SystemExit("no score matrices found — run the score stage")
    # candidate xy for the dual-temporal rule
    for (level, source), rows in rows_by.items():
        rx = ref_xy_by[(level, source)]
        for r in rows:
            r["_cand_xy"] = rx.get(r["sa_candidate_id"]) if r["sa_candidate_id"] else None

    # attainability per query on the full sparse / dense memories (for the LSO populations)
    att_sparse, att_dense = {}, {}
    for rows in rows_by.values():
        for r in rows:
            if r["memory"] == "sparse":
                att_sparse[r["query_key"]] = bool(r["attainable"])
            if r["memory"] == "dense" and r["holdout_m"] == 0.0:
                att_dense[r["query_key"]] = bool(r["attainable"])

    def pop(level, source, name):
        return population(rows_by.get((level, source), []), name, att_sparse, att_dense)

    metrics = {"study_version": STUDY_VERSION, "config": {k_: v for k_, v in cfg.items() if not k_.startswith("_")},
               "sources": sources_present, "population_sizes": {}, "score_distributions": {},
               "sweeps": {}, "selected_gates": {}, "baseline_frozen": {}, "ablation": {}, "populations": {},
               "bins": {}, "radii": {}, "temporal": {}, "aliasing": {}, "relative_position": {}, "cost": {}}

    # population sizes and score-scale check (N vs W on the same queries)
    for (level, source), rows in rows_by.items():
        for name in POPS_SPARSE + POPS_DENSE:
            p = pop(level, source, name)
            metrics["population_sizes"][f"{level}/{source}/{name}"] = {
                "n": len(p), "n_sites": len({r["site"] for r in p}), "n_segments": len({r["segment_id"] for r in p})}
        sp = [r for r in rows if r["memory"] == "sparse"]
        sn = [r["_out"]["north"]["top1_score"] for r in sp if r["_out"]["north"]["top1_score"] is not None and np.isfinite(r["_out"]["north"]["top1_score"])]
        sw = [r["_out"]["west"]["top1_score"] for r in sp if r["_out"]["west"]["top1_score"] is not None and np.isfinite(r["_out"]["west"]["top1_score"])]
        metrics["score_distributions"][f"{level}/{source}"] = {
            "north_top1_median": _median(sn), "west_top1_median": _median(sw),
            "north_top1_p10": float(np.percentile(sn, 10)) if sn else None,
            "west_top1_p10": float(np.percentile(sw, 10)) if sw else None,
            "n": len(sp)}

    # ---- DEV sweep → selected gate per (memory kind, source, rule) ------------------------------
    for source in sources_present:
        for mem in ("sparse", "dense"):
            level_gate = "frame" if mem == "sparse" else "region"
            for rule in RULES:
                inc = pop(dev, source, IN_OF[mem])
                hard_rows = [r for name in HARD_OF[mem] for r in pop(dev, source, name)]
                ev = (lambda r, g, rule=rule: (accepted_at(r, rule, g), false_of(r, rule)))
                sw_in = dv.sweep(inc, grid_s, grid_m, ev, level_gate)
                sw_hard = dv.sweep(hard_rows, grid_s, grid_m, ev, level_gate)
                sel = dv.select_gate(sw_in, sw_hard)
                hard_by = {(h["theta_s"], h["theta_m"]): h["n_accepted_false"] for h in sw_hard}
                for e in sw_in:
                    e["hard_false"] = hard_by.get((e["theta_s"], e["theta_m"]), 0)
                    e["hard_n"] = len(hard_rows)
                metrics["sweeps"][f"{mem}/{source}/{rule}"] = {"grid": sw_in, "selected": sel,
                                                                "dev_level": dev, "hard_populations": list(HARD_OF[mem]),
                                                                "n_in_coverage": len(inc), "n_hard": len(hard_rows)}
                metrics["selected_gates"][f"{mem}/{source}/{rule}"] = {
                    "theta_s": sel["theta_s"], "theta_m": sel["theta_m"], "level": level_gate,
                    "false_free_point_exists": sel["false_free_point_exists"],
                    "dev_coverage": sel["coverage"], "dev_total_false": sel["total_accepted_false_dev"]}

    def gate_for(mem, source, rule):
        g = metrics["selected_gates"][f"{mem}/{source}/{rule}"]
        return dv.Gate(g["theta_s"], g["theta_m"], g["level"])

    # ---- populations at the selected gate and at the frozen baseline ---------------------------
    for (level, source) in rows_by:
        for name in POPS_SPARSE + POPS_DENSE:
            mem = "sparse" if name.endswith("sparse") else "dense"
            p = pop(level, source, name)
            for rule in RULES:
                metrics["populations"][f"{level}/{source}/{name}/{rule}"] = summarize(p, rule, gate_for(mem, source, rule))
                metrics["baseline_frozen"][f"{level}/{source}/{name}/{rule}"] = summarize(
                    p, rule, dv.Gate(frozen_s, frozen_m, "frame" if mem == "sparse" else "region"))
        for rule in RULES:
            metrics["baseline_frozen"][f"{level}/{source}/in_coverage_dense/{rule}/frame"] = summarize(
                pop(level, source, "in_coverage_dense"), rule, dv.Gate(frozen_s, frozen_m, "frame"))
        for rule in RULES:
            e = dict(metrics["populations"][f"{level}/{source}/in_coverage_sparse/{rule}"])
            ooc = metrics["populations"][f"{level}/{source}/ooc_sparse/{rule}"]
            lso = metrics["populations"][f"{level}/{source}/lso_sparse/{rule}"]
            e["ooc_accepted_rate"] = ooc["coverage"]; e["ooc_n"] = ooc["n"]; e["ooc_n_accepted"] = ooc["n_accepted"]
            e["lso_accepted_rate"] = lso["coverage"]; e["lso_n"] = lso["n"]; e["lso_n_accepted"] = lso["n_accepted"]
            e["hard_negative_accepted_rate"] = ((ooc["n_accepted"] + lso["n_accepted"]) / (ooc["n"] + lso["n"])) if (ooc["n"] + lso["n"]) else None
            metrics["ablation"][f"{level}/{source}/{rule}"] = e

    # ---- per-level grid sensitivity (on-level oracle gate; reported as sensitivity only) ----------
    metrics["grid_by_level"] = {}
    for (level, source) in rows_by:
        for mem in ("sparse", "dense"):
            level_gate = "frame" if mem == "sparse" else "region"
            inc = pop(level, source, IN_OF[mem])
            hard_rows = [r for name in HARD_OF[mem] for r in pop(level, source, name)]
            for rule in RULES:
                ev = (lambda r, g, rule=rule: (accepted_at(r, rule, g), false_of(r, rule)))
                sw_in = dv.sweep(inc, grid_s, grid_m, ev, level_gate)
                sw_hard = dv.sweep(hard_rows, grid_s, grid_m, ev, level_gate)
                hard_by = {(h["theta_s"], h["theta_m"]): h["n_accepted_false"] for h in sw_hard}
                pts = []
                for e in sw_in:
                    tf = int(e["n_accepted_false"]) + hard_by.get((e["theta_s"], e["theta_m"]), 0)
                    pts.append({"theta_s": e["theta_s"], "theta_m": e["theta_m"], "coverage": e["coverage"],
                                "n_accepted": e["n_accepted"], "n_accepted_false_in_coverage": e["n_accepted_false"],
                                "n_hard_accepted": hard_by.get((e["theta_s"], e["theta_m"]), 0), "total_false": tf})
                zero = [x for x in pts if x["total_false"] == 0]
                best_zero = max(zero, key=lambda x: (x["coverage"] or 0.0, x["theta_m"], x["theta_s"])) if zero else None
                # the frontier: for each distinct total_false count, the largest coverage
                front = {}
                for x in pts:
                    if x["total_false"] not in front or (x["coverage"] or 0) > (front[x["total_false"]]["coverage"] or 0):
                        front[x["total_false"]] = x
                metrics["grid_by_level"][f"{mem}/{level}/{source}/{rule}"] = {
                    "n_in_coverage": len(inc), "n_hard": len(hard_rows), "grid": pts,
                    "max_coverage_at_zero_false": best_zero,
                    "frontier": [front[k_] for k_ in sorted(front)][:4]}

    # ---- bins and radii (sparse at the selected gate; dense deduped over hold-outs) --------------
    edges = [float(b) for b in cfg["bins_m"]]
    for (level, source), rows in rows_by.items():
        for mem, sel_rows in (("sparse", [r for r in rows if r["memory"] == "sparse" and r["primary"]]),
                              ("dense", dedupe_dense([r for r in rows if r["memory"] == "dense" and r["primary"]]))):
            for rule in ("north", "west", "strict", "min"):
                g = gate_for(mem, source, rule)
                brows = [to_bin_row(r, rule, g) for r in sel_rows]
                table = recog.bin_table(brows, edges, cfg["support"], k=k)
                radius = recog.radius_from_bins(table, cfg["radius_rule"])
                metrics["bins"][f"{mem}/{level}/{source}/{rule}"] = {"bins": table, "gate": {"theta_s": g.theta_s, "theta_m": g.theta_m, "level": g.level}}
                metrics["radii"][f"{mem}/{level}/{source}/{rule}"] = {**radius, "spacing": recog.spacing_implication(radius["conservative_m"])}

    # ---- temporal confirmation vs dual --------------------------------------------------------
    for (level, source), rows in rows_by.items():
      rx = ref_xy_by[(level, source)]
      entry = {}
      for mem in ("sparse", "dense"):
        gN = gate_for(mem, source, "north")
        gS = gate_for(mem, source, "strict")
        if mem == "sparse":
            sparse_rows = [r for r in rows if r["memory"] == "sparse"]
            lso_rows = [r for r in rows if r["memory"] == "sparse_lso"]
            pops_ = (("in_coverage_sparse", pop(level, source, "in_coverage_sparse")),
                     ("hard_sparse_all", pop(level, source, "ooc_sparse") + pop(level, source, "lso_sparse")))
            lso_name = "sparse_lso"
        else:
            sparse_rows = [r for r in rows if r["memory"] == "dense" and r["holdout_m"] == 0.0]
            lso_rows = [r for r in rows if r["memory"] == "dense_lso"]
            pops_ = (("in_coverage_dense", pop(level, source, "in_coverage_dense")),
                     ("hard_dense_lso", pop(level, source, "lso_dense")))
            lso_name = "dense_lso"
        for popname, prows in pops_:
            res = {}
            for kk in (1, 2, 3):
                tr_s = temporal_rows(sparse_rows, kk, gN, rx, tau_agree, tau)
                tr_l = temporal_rows(lso_rows, kk, gN, rx, tau_agree, tau)
                acc = fal = 0
                for r in prows:
                    t = (tr_l if r["memory"] == lso_name else tr_s).get(r["query_key"])
                    if t is None:
                        continue
                    acc += int(t["accepted"]); fal += int(t["accepted_false"])
                res[f"north_k{kk}"] = {"n": len(prows), "n_accepted": acc, "n_accepted_false": fal,
                                       "coverage": acc / len(prows) if prows else None,
                                       "accepted_rate": acc / len(prows) if prows else None,
                                       "p_false_given_accepted": fal / acc if acc else None}
            acc = fal = 0
            for r in prows:
                a = accepted_at(r, "strict", gS)
                acc += int(a); fal += int(a and r["sa_false_region"])
            res["strict"] = {"n": len(prows), "n_accepted": acc, "n_accepted_false": fal,
                             "coverage": acc / len(prows) if prows else None, "accepted_rate": acc / len(prows) if prows else None,
                             "p_false_given_accepted": fal / acc if acc else None}
            dt_s = dual_temporal(sparse_rows, gS, tau_agree)
            dt_l = dual_temporal(lso_rows, gS, tau_agree)
            acc = fal = 0
            for r in prows:
                t = (dt_l if r["memory"] == lso_name else dt_s).get(r["query_key"])
                if t is None:
                    continue
                acc += int(t["accepted"]); fal += int(t["accepted_false"])
            res["strict_k2"] = {"n": len(prows), "n_accepted": acc, "n_accepted_false": fal,
                                "coverage": acc / len(prows) if prows else None, "accepted_rate": acc / len(prows) if prows else None,
                                "p_false_given_accepted": fal / acc if acc else None}
            entry[popname] = res
      metrics["temporal"][f"{level}/{source}"] = entry

    # ---- aliasing: confident-false North cases and what West does with them ----------------------
    for (level, source), rows in rows_by.items():
        gS = gate_for("sparse", source, "strict")
        cases = [r for r in rows if r["memory"] == "sparse" and r["horizontal"]
                 and r["_out"]["north"]["false_region"] and r["_out"]["north"]["top1_score"] is not None
                 and r["_out"]["north"]["top1_score"] >= 0.99]
        n_w_false = sum(1 for r in cases if r["_out"]["west"]["false_region"])
        n_agree = sum(1 for r in cases if r["sa_agree"])
        n_dual_acc = sum(1 for r in cases if accepted_at(r, "strict", gS))
        n_north_acc = sum(1 for r in cases if accepted_at(r, "north", gate_for("sparse", source, "north")))
        examples = []
        for r in sorted(cases, key=lambda r: -(r["_out"]["north"]["top1_score"] or 0))[:6]:
            examples.append({"query": r["query_key"], "site": r["site"], "d_near_m": r["d_near_m"],
                             "north_top1": r["_out"]["north"]["top1_id"], "north_top1_distance_m": r["_out"]["north"]["top1_distance_m"],
                             "north_score": r["_out"]["north"]["top1_score"], "north_frame_margin": r["_out"]["north"]["frame_margin"],
                             "west_top1": r["_out"]["west"]["top1_id"], "west_top1_distance_m": r["_out"]["west"]["top1_distance_m"],
                             "west_score": r["_out"]["west"]["top1_score"], "west_false": r["_out"]["west"]["false_region"],
                             "agree": r["sa_agree"], "dual_accepted": accepted_at(r, "strict", gS),
                             "north_accepted": accepted_at(r, "north", gate_for("sparse", source, "north"))})
        metrics["aliasing"][f"{level}/{source}"] = {
            "n_north_confident_false": len(cases), "n_of_which_west_also_false": n_w_false,
            "n_of_which_views_agree_on_the_wrong_region": n_agree,
            "n_accepted_by_north_alone_at_its_gate": n_north_acc, "n_accepted_by_dual_at_its_gate": n_dual_acc,
            "examples": examples}

    # ---- relative position, cost ---------------------------------------------------------------
    for key_, r in rel.items():
        metrics["relative_position"][key_] = {kk: v for kk, v in r.items() if kk != "_recs"}
    metrics["cost"] = cost(cfg, base, out)

    # ---- outcome classification per level (declared criteria) -----------------------------------
    metrics["classification"] = classify(metrics, cfg)

    # ---- tables to disk -------------------------------------------------------------------------
    all_rows = [flat(r) for rows in rows_by.values() for r in rows]
    fields = list(all_rows[0].keys())
    _write_csv(out / "queries.csv.gz", all_rows, fields)
    _json(out / "metrics.json", metrics)
    # figures
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _fig_ablation(out, metrics["ablation"])
    _fig_bins(out, "fig2_region_top1_vs_distance_sparse.jpg", metrics["bins"], "region_top1",
              "correct-region top-1", "Correct-region top-1 vs separation, sparse anchor memory", ("north", "west", "strict"), "sparse")
    _fig_bins(out, "fig3_accepted_false_vs_distance_sparse.jpg", metrics["bins"], "accepted_false_region_rate",
              "accepted false-region rate", "Accepted false-region rate vs separation, sparse anchor memory", ("north", "west", "strict"), "sparse")
    _fig_bins(out, "fig4_coverage_vs_distance.jpg", metrics["bins"], "coverage_frame",
              "accepted coverage", "Accepted coverage vs separation, sparse (top) rules at the DEV gates", ("north", "west", "strict"), "sparse")
    _fig_scatter(out, rows_by)
    _fig_sweep(out, metrics["sweeps"], cfg)
    _fig_temporal(out, metrics["temporal"])
    _fig_lags(out, rel)
    _fig_bins(out, "fig9_region_top1_vs_distance_dense.jpg", metrics["bins"], "region_top1",
              "correct-region top-1", "Correct-region top-1 vs separation, dense 10 m memory (hold-out radii)", ("north", "west", "strict"), "dense")
    write_report(out, cfg, metrics)
    print(f"[report] -> {out / 'report.md'}")


def classify(metrics: dict, cfg: dict) -> dict:
    """The declared A/B/C/D criteria per level, strict rule vs the single views at the DEV-selected
    gates. "Better single view" is read in the way that is hardest for the dual rule: the lower
    P(false | accepted) of the two views for the precision comparisons, the higher accepted coverage
    of the two for the coverage comparisons, the fewer accepted-false for the count comparison."""
    out = {}
    for level in LEVEL_ORDER:
        for source in metrics["sources"]:
            key = f"{level}/{source}"
            abl = metrics["ablation"]
            d = abl.get(f"{key}/strict"); n = abl.get(f"{key}/north"); w = abl.get(f"{key}/west")
            if not (d and n and w):
                continue
            pfs = [e["p_false_given_accepted"] for e in (n, w) if e["p_false_given_accepted"] is not None]
            pf_b = min(pfs) if pfs else None
            cov_b = max(n["coverage"] or 0.0, w["coverage"] or 0.0)
            nf_b = min(n["n_accepted_false"], w["n_accepted_false"])
            hards = [x for x in (n["hard_negative_accepted_rate"], w["hard_negative_accepted_rate"]) if x is not None]
            hard_b = min(hards) if hards else None
            lso = metrics["populations"].get(f"{key}/lso_sparse/strict", {})
            ooc = metrics["populations"].get(f"{key}/ooc_sparse/strict", {})
            temporal = metrics["temporal"].get(key, {}).get("in_coverage_sparse", {})
            t2_false = temporal.get("north_k2", {}).get("n_accepted_false")
            hard_rate = d.get("hard_negative_accepted_rate")
            zero_all = d["n_accepted_false"] == 0 and lso.get("n_accepted", 0) == 0 and ooc.get("n_accepted", 0) == 0
            cov_ok_half = (d["coverage"] or 0) >= 0.5 * cov_b
            pf_d = d["p_false_given_accepted"]
            reduces_third = (pf_d is not None and pf_b is not None and ((pf_b == 0 and pf_d == 0) or (pf_b > 0 and pf_d <= pf_b / 3.0)))
            reduces_half = (d["n_accepted_false"] <= 0.5 * nf_b) if nf_b else (d["n_accepted_false"] == 0)
            has_acc = (d["n_accepted"] or 0) > 0
            if zero_all and cov_ok_half and has_acc:
                label = "D"
            elif (reduces_third and hard_rate is not None and hard_rate <= 0.02 and (d["coverage"] or 0) >= 0.25
                  and (t2_false is None or d["n_accepted_false"] <= t2_false)):
                label = "C"
            elif reduces_half and has_acc:
                label = "B"
            else:
                label = "A"
            pick = ("coverage", "n_accepted", "n_accepted_false", "p_false_given_accepted", "hard_negative_accepted_rate")
            single_false = (n["n_accepted_false"] + w["n_accepted_false"] + n["ooc_n_accepted"] + w["ooc_n_accepted"]
                            + n["lso_n_accepted"] + w["lso_n_accepted"])
            out[key] = {"label": label, "single_views_accepted_false_total": int(single_false),
                        "informative": bool(single_false > 0),
                        "single_reference": {"p_false_given_accepted": pf_b, "coverage": cov_b,
                                             "n_accepted_false": nf_b, "hard_negative_accepted_rate": hard_b},
                        "dual": {k_: d.get(k_) for k_ in pick}, "north": {k_: n.get(k_) for k_ in pick},
                        "west": {k_: w.get(k_) for k_ in pick},
                        "ooc_dual_accepted": ooc.get("n_accepted"), "lso_dual_accepted": lso.get("n_accepted"),
                        "north_temporal_k2_accepted_false": t2_false,
                        "split": "dev" if level == cfg["dev_level"] else "held-out"}
    return out


# --------------------------------------------------------------------------------------------------
# report.md
# --------------------------------------------------------------------------------------------------

def write_report(out: Path, cfg: dict, m: dict) -> None:
    L = []
    L.append("# EXP-SKY-011 — dual-direction (North + West) skyline place recognition: report\n")
    L.append(f"Generated by `scripts/sky_dual_study.py --stage report` (study {STUDY_VERSION}). Frozen C1-32; "
             f"τ_region = {cfg['tau_region_m']} m, τ_agree = {cfg['tau_agree_m']} m; DEV = {cfg['dev_level']}; "
             f"sources {m['sources']}. Evidence tier T2.\n")
    L.append("## Population sizes (queries / sites / segments)\n")
    L.append("| level/source | in-cov sparse | OOC sparse | LSO sparse | hard | vertical | transit | in-cov dense | LSO dense |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for level in LEVEL_ORDER:
        for src in m["sources"]:
            cells_ = []
            for name in ("in_coverage_sparse", "ooc_sparse", "lso_sparse", "hard_sparse", "vertical_sparse", "transit_sparse", "in_coverage_dense", "lso_dense"):
                p = m["population_sizes"].get(f"{level}/{src}/{name}")
                cells_.append("—" if p is None else f"{p['n']} / {p['n_sites']} / {p['n_segments']}")
            L.append(f"| {level}/{SRC_LABEL[src]} | " + " | ".join(cells_) + " |")
    L.append("\n## Score scale check (sparse memory, all queries)\n")
    L.append("| level/source | N top-1 median | W top-1 median | N p10 | W p10 |")
    L.append("|---|---|---|---|---|")
    for key_, e in sorted(m["score_distributions"].items()):
        L.append(f"| {key_} | {_num(e['north_top1_median'], 3)} | {_num(e['west_top1_median'], 3)} | {_num(e['north_top1_p10'], 3)} | {_num(e['west_top1_p10'], 3)} |")
    L.append("\n## DEV-selected gates (village; zero accepted-false on in-coverage + OOC + LSO, then max coverage)\n")
    L.append("| memory/source/rule | θ_s | θ_m | level | false-free point exists | DEV coverage | DEV total false |")
    L.append("|---|---|---|---|---|---|---|")
    for key_, g in sorted(m["selected_gates"].items()):
        L.append(f"| {key_} | {g['theta_s']} | {g['theta_m']} | {g['level']} | {g['false_free_point_exists']} | {_pct(g['dev_coverage'])} | {g['dev_total_false']} |")
    L.append("\n## Primary ablation — sparse anchor memory, in-coverage primary population, DEV-selected gates\n")
    cols = [("level/source", lambda e: e["_k"]), ("rule", lambda e: RULE_LABEL[e["rule"]]), ("gate", lambda e: f"{e['theta_s']}/{e['theta_m']}"),
            ("n", lambda e: str(e["n"])), ("sites", lambda e: str(e["n_sites"])),
            ("region top-1", lambda e: _pct(e["region_top1"])), ("R@5", lambda e: _pct(e["region_recall_k"])),
            ("coverage", lambda e: _pct(e["coverage"])), ("precision", lambda e: _pct(e["accepted_precision"])),
            ("acc. false", lambda e: str(e["n_accepted_false"])), ("P(false|acc)", lambda e: _pct(e["p_false_given_accepted"])),
            ("OOC acc.", lambda e: f"{_pct(e['ooc_accepted_rate'])} ({e['ooc_n_accepted']}/{e['ooc_n']})"),
            ("LSO acc.", lambda e: f"{_pct(e['lso_accepted_rate'])} ({e['lso_n_accepted']}/{e['lso_n']})"),
            ("cons. radius", lambda e: e["_radius"])]
    entries = []
    for level in LEVEL_ORDER:
        for src in m["sources"]:
            for rule in RULES:
                e = m["ablation"].get(f"{level}/{src}/{rule}")
                if not e:
                    continue
                rad = m["radii"].get(f"sparse/{level}/{src}/{rule}", {})
                rc = rad.get("conservative_m")
                entries.append({**e, "_k": f"{level}/{SRC_LABEL[src]}", "_radius": ("none" if rc is None else f"{rc:.0f} m") if rule != "mean" else "n/a"})
    L += _summary_md(entries, cols)
    L.append("\n## Same table at the frozen baseline gate (0.90 / 0.15, frame level)\n")
    entries = []
    for level in LEVEL_ORDER:
        for src in m["sources"]:
            for rule in RULES:
                e = m["baseline_frozen"].get(f"{level}/{src}/in_coverage_sparse/{rule}")
                if not e:
                    continue
                ooc = m["baseline_frozen"][f"{level}/{src}/ooc_sparse/{rule}"]; lso = m["baseline_frozen"][f"{level}/{src}/lso_sparse/{rule}"]
                entries.append({**e, "_k": f"{level}/{SRC_LABEL[src]}", "_radius": "",
                                "ooc_accepted_rate": ooc["coverage"], "ooc_n": ooc["n"], "ooc_n_accepted": ooc["n_accepted"],
                                "lso_accepted_rate": lso["coverage"], "lso_n": lso["n"], "lso_n_accepted": lso["n_accepted"]})
    L += _summary_md(entries, cols[:-1])
    L.append("\n## Sensitivity — coverage attainable at ZERO accepted-false on each level's own populations (in-coverage + OOC + LSO), best point of the 4×4 grid per rule — an on-level oracle, not a held-out number\n")
    L.append("| memory | level/source | North | West | N+W strict | N+W min | N+W mean |")
    L.append("|---|---|---|---|---|---|---|")
    for mem in ("sparse", "dense"):
        for level in LEVEL_ORDER:
            for src in m["sources"]:
                cells_ = []
                for rule in RULES:
                    g = m["grid_by_level"].get(f"{mem}/{level}/{src}/{rule}")
                    if not g:
                        cells_.append("—"); continue
                    bz = g["max_coverage_at_zero_false"]
                    if bz is None:
                        worst = min(g["grid"], key=lambda x: (x["total_false"], -(x["coverage"] or 0)))
                        cells_.append(f"no zero-false point (best {worst['total_false']} false at {_pct(worst['coverage'])})")
                    else:
                        cells_.append(f"{_pct(bz['coverage'])} @ {bz['theta_s']}/{bz['theta_m']}")
                L.append(f"| {mem} | {level}/{SRC_LABEL[src]} | " + " | ".join(cells_) + " |")
    L.append("\n## Hard negatives — accepted count / n (DEV gates): natural out-of-coverage, leave-site-out, hard sites, transit\n")
    L.append("| level/source | rule | OOC | LSO | hard sites | transit (sparse) | vertical legs |")
    L.append("|---|---|---|---|---|---|---|")
    for level in LEVEL_ORDER:
        for src in m["sources"]:
            for rule in RULES:
                cells_ = []
                for name in ("ooc_sparse", "lso_sparse", "hard_sparse", "transit_sparse", "vertical_sparse"):
                    e = m["populations"].get(f"{level}/{src}/{name}/{rule}")
                    cells_.append("—" if not e else f"{e['n_accepted']} / {e['n']} (false {e['n_accepted_false']}; top-1 {_pct(e['region_top1'])})")
                L.append(f"| {level}/{SRC_LABEL[src]} | {RULE_LABEL[rule]} | " + " | ".join(cells_) + " |")
    L.append("\n## Dense 10 m memory (hold-out 0, region-level gates, DEV-selected)\n")
    L.append("| level/source | rule | gate | n | region top-1 | R@5 | coverage | acc. false | P(false|acc) | LSO dense acc. | transit acc. (false) | frame-level frozen gate coverage (false) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for level in LEVEL_ORDER:
        for src in m["sources"]:
            for rule in RULES:
                e = m["populations"].get(f"{level}/{src}/in_coverage_dense/{rule}")
                if not e:
                    continue
                lso = m["populations"][f"{level}/{src}/lso_dense/{rule}"]; tr = m["populations"][f"{level}/{src}/transit_dense/{rule}"]
                fr = m["baseline_frozen"].get(f"{level}/{src}/in_coverage_dense/{rule}/frame", {})
                L.append(f"| {level}/{SRC_LABEL[src]} | {RULE_LABEL[rule]} | {e['theta_s']}/{e['theta_m']} | {e['n']} | {_pct(e['region_top1'])} | {_pct(e['region_recall_k'])} | "
                         f"{_pct(e['coverage'])} | {e['n_accepted_false']} | {_pct(e['p_false_given_accepted'])} | {lso['n_accepted']}/{lso['n']} | {tr['n_accepted']}/{tr['n']} ({tr['n_accepted_false']}) | "
                         f"{_pct(fr.get('coverage'))} ({fr.get('n_accepted_false', '—')}) |")
    L.append("\n## Recognition radii (m) — conservative / recoverable, DEV gates\n")
    L.append("| memory | level/source | North | West | N+W strict | N+W min |")
    L.append("|---|---|---|---|---|---|")
    for mem in ("sparse", "dense"):
        for level in LEVEL_ORDER:
            for src in m["sources"]:
                cells_ = []
                for rule in ("north", "west", "strict", "min"):
                    r = m["radii"].get(f"{mem}/{level}/{src}/{rule}")
                    if not r:
                        cells_.append("—"); continue
                    c, rr = r["conservative_m"], r["recoverable_m"]
                    cells_.append(f"{'none' if c is None else f'{c:.0f}'} / {'none' if rr is None else f'{rr:.0f}'}")
                L.append(f"| {mem} | {level}/{SRC_LABEL[src]} | " + " | ".join(cells_) + " |")
    L.append("\n## Per-bin detail — sparse memory, region top-1 / accepted-false rate / coverage (n; sites)\n")
    for level in LEVEL_ORDER:
        for src in m["sources"]:
            for rule in ("north", "west", "strict"):
                b = m["bins"].get(f"sparse/{level}/{src}/{rule}")
                if not b:
                    continue
                L.append(f"\n**{level} / {SRC_LABEL[src]} / {RULE_LABEL[rule]}** (gate {b['gate']['theta_s']}/{b['gate']['theta_m']})\n")
                L.append("| bin | n | sites | supported | region top-1 | R@5 | site min top-1 | score med | margin med | acc. false rate (n) | coverage | ambiguity |")
                L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
                for e in b["bins"]:
                    if e["n"] == 0:
                        continue
                    L.append(f"| {e['bin']} | {e['n']} | {e['n_sites']} | {'yes' if e['supported'] else 'no'} | {_pct(e['region_top1'])} | {_pct(e['region_recall_k'])} | "
                             f"{_pct(e['site_region_top1_min'])} | {_num(e['top1_score_median'], 3)} | {_num(e['region_margin_median'], 3)} | "
                             f"{_pct(e['accepted_false_region_rate'])} ({e['n_accepted_false_region']}) | {_pct(e['coverage_frame'])} | {_pct(e['ambiguity_rate'])} |")
    L.append("\n## Temporal confirmation (North, k consecutive captures of one segment) vs dual single-location agreement — sparse, DEV gates\n")
    L.append("| level/source | population | N k=1 | N k=2 | N k=3 | N+W strict | N+W k=2 |")
    L.append("|---|---|---|---|---|---|---|")
    for key_, e in sorted(m["temporal"].items()):
        for popname, res in e.items():
            cells_ = []
            for rk in ("north_k1", "north_k2", "north_k3", "strict", "strict_k2"):
                r = res.get(rk, {})
                cells_.append(f"{_pct(r.get('coverage'))} acc, {r.get('n_accepted_false')} false")
            L.append(f"| {key_} | {popname} | " + " | ".join(cells_) + " |")
    L.append("\n## Aliasing: North confident-false cases (top-1 ≥ 0.99 and wrong region, sparse, horizontal queries)\n")
    L.append("| level/source | N confident-false | West also false | views agree on the wrong region | accepted by N alone | accepted by N+W |")
    L.append("|---|---|---|---|---|---|")
    for key_, e in sorted(m["aliasing"].items()):
        L.append(f"| {key_} | {e['n_north_confident_false']} | {e['n_of_which_west_also_false']} | {e['n_of_which_views_agree_on_the_wrong_region']} | "
                 f"{e['n_accepted_by_north_alone_at_its_gate']} | {e['n_accepted_by_dual_at_its_gate']} |")
    for key_, e in sorted(m["aliasing"].items()):
        if e["examples"]:
            L.append(f"\n**{key_} examples**\n")
            L.append("| query | site | d_near | N top-1 (dist) | N score / margin | W top-1 (dist) | W score | W false | agree | N accepted | N+W accepted |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for x in e["examples"]:
                L.append(f"| {x['query']} | {x['site']} | {_num(x['d_near_m'], 0)} | {x['north_top1']} ({_num(x['north_top1_distance_m'], 0)}) | "
                         f"{_num(x['north_score'], 3)} / {_num(x['north_frame_margin'], 3)} | {x['west_top1']} ({_num(x['west_top1_distance_m'], 0)}) | "
                         f"{_num(x['west_score'], 3)} | {x['west_false']} | {x['agree']} | {x['north_accepted']} | {x['dual_accepted']} |")
    L.append("\n## Relative position (secondary): same-swipe horizontal pairs ≤ 100 m, dense memory\n")
    L.append("| level/source | pairs | sites | R² lag_N~(ΔE,ΔN) | R² lag_W~(ΔE,ΔN) | gain lag_N/m East (site min–max) | gain lag_W/m North (site min–max) | direction err median pair / single-N | length ratio p10–p90 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for key_, r in sorted(m["relative_position"].items()):
        if "ols_lagN_on_dE_dN" not in r:
            L.append(f"| {key_} | {r.get('n_pairs')} | — | {r.get('note', '')} | | | | | |"); continue
        gn, gw = r["site_gain_spread_lagN"], r["site_gain_spread_lagW"]
        L.append(f"| {key_} | {r['n_pairs']} | {r['n_sites']} | {_num(r['ols_lagN_on_dE_dN']['r2'])} | {_num(r['ols_lagW_on_dE_dN']['r2'])} | "
                 f"{_num(r['pooled_gain_lagN_per_m_east'], 3)} ({_num(gn[0], 3) if gn else '—'}–{_num(gn[1], 3) if gn else '—'}) | "
                 f"{_num(r['pooled_gain_lagW_per_m_north'], 3)} ({_num(gw[0], 3) if gw else '—'}–{_num(gw[1], 3) if gw else '—'}) | "
                 f"{_num(r['direction_error_deg_median_pair'], 0)}° / {_num(r['direction_error_deg_median_single_north'], 0)}° | "
                 f"{_num(r['length_ratio_pooled_gain_p10_p90'][0])}–{_num(r['length_ratio_pooled_gain_p10_p90'][1])} |")
    L.append("\n## Cost\n")
    L.append("```json\n" + json.dumps(m["cost"], indent=2) + "\n```\n")
    L.append("\n## Declared outcome criteria applied\n")
    L.append("| level/source | split | label | dual: coverage / acc.false / P(false|acc) / hard-neg acc. | North: same | West: same | N temporal k=2 acc.false |")
    L.append("|---|---|---|---|---|---|---|")

    def _q(e):
        return f"{_pct(e['coverage'])} / {e['n_accepted_false']} / {_pct(e['p_false_given_accepted'])} / {_pct(e['hard_negative_accepted_rate'])}"
    for key_, c in sorted(m["classification"].items()):
        lab = f"**{c['label']}**" + ("" if c["informative"] else " (n/i: single views false-free at their DEV gates)")
        L.append(f"| {key_} | {c['split']} | {lab} | {_q(c['dual'])} | {_q(c['north'])} | {_q(c['west'])} | {c['north_temporal_k2_accepted_false']} |")
    L.append("\n## Figures\n")
    for f in sorted((out / "figures").glob("*.jpg")):
        L.append(f"- `figures/{f.name}`")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
