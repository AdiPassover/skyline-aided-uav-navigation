"""EXP-SKY-014 Q-B: azimuth-error tolerance of the adopted C0 skyline evidence, one view vs two.

Run from ``skyline/`` with ``PYTHONPATH=.``::

    PYTHONPATH=. python ../evaluation/tools/claim_closure/sky_azimuth_tolerance.py --out ../evaluations/claim-closure-2026-09/sky

Reads the frozen EXP-SKY-011 index/store (both views, sim_exact + segformer) and the EXP-SKY-012
hard-city store; scores every query against the dense online memory ONCE at bound 32 with the
EXP-SKY-013 instrument (``NestedBoundBank``), verifies it against the real frozen matcher objects,
and reads the score at a fixed lag ℓ(δ) off the same per-lag matrix for every pre-specified
azimuth offset δ. Acceptance: the frozen SKY rule (0.90 / 0.15 region-level, τ 125 m) and the
frozen INT gate (0.90 / 0.30, ambiguity 50 m, dual agreement 15 m). Arms: view A, view B,
weakest-view fusion, strict agreement. Secondary exact check: the rotated profile synthesised from
the two GT curves (adjacent-view content enters the frame) scored pointwise.

Nothing frozen is modified. No threshold is chosen.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
HSR = REPO / "skyline"
sys.path.insert(0, str(HSR))
sys.path.insert(0, str(HSR / "scripts"))

from hsreloc.observation import read_session                                    # noqa: E402
from hsreloc.retrieval.skyline_curve import SkylineCurve                        # noqa: E402
from hsreloc.simret import recog, relpose                                       # noqa: E402
from hsreloc.simret.lagstudy import NestedBoundBank, verify_bounds              # noqa: E402
from hsreloc.simret.sources import SessionSilverSource, SimExactSource, session_map  # noqa: E402
from sky_relpose_study import _Curves, _camera                                  # noqa: E402

DEG_PER_SAMPLE = 90.0 / 256.0                     # EXP-SKY-013 degrees_per_sample = 0.3515625
OFFSETS_DEG = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 11.0)
MAX_LAG = 32
MIN_OVERLAP = 0.6
LEVELS = ("village", "mountains", "city")
SOURCES = ("sim_exact", "segformer")
VIEWS = ("north", "west")
RULES = {
    "R-SKY": {"theta_s": 0.90, "theta_m": 0.15, "tau_region": 125.0, "tau_agree": 125.0, "wrong_place": 125.0},
    "R-INT": {"theta_s": 0.90, "theta_m": 0.30, "tau_region": 50.0, "tau_agree": 15.0, "wrong_place": 50.0},
}
LSO_RADIUS = 100.0
HARDCITY_LSO = 125.0
TRUE_MATCH_MAX_M = 15.0


def lag_for(delta_deg: float) -> int:
    return int(round(delta_deg / DEG_PER_SAMPLE))


#: The only lag columns of the ±32 grid the pre-specified offsets read (both signs), and where
#: each sits in the reduced per-query matrix.
NEEDED_LAGS = sorted({s * lag_for(d) for d in OFFSETS_DEG for s in (1, -1)})
NEEDED_COLS = [MAX_LAG + ell for ell in NEEDED_LAGS]
NEEDED_INDEX = {ell: j for j, ell in enumerate(NEEDED_LAGS)}


# ----------------------------------------------------------------------------- inputs

def load_index(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for r in rows:
        for k in ("east_m", "north_m", "up_m", "vertical_up_m"):
            v = r.get(k, "")
            r[k] = float(v) if v not in ("", None, "nan") else None
        r["pair_ok"] = r.get("pair_ok", "True") == "True"
    return rows


def is_dense_reference(r: dict) -> bool:
    if r["segment_kind"] == "anchor_capture":
        return True
    if r.get("leg_axis") == "vertical":
        return False
    return r["vertical_up_m"] is None or abs(r["vertical_up_m"]) < 1.0


def key_of(r: dict) -> str:
    return f"{r['combined_run_id']}/{r['observation_raw_id']}"


def make_sources(cfg: dict, base: Path, store: Path, sessions: dict, cam, mask_root: Path) -> dict:
    out = {"sim_exact": SimExactSource(store, sessions, cam.width_px, cam.height_px)}
    spec = cfg["segformer"]
    out["segformer"] = SessionSilverSource(
        mask_root, sessions, cam.width_px, cam.height_px,
        scale_short_side=spec.get("scale_short_side", 512), closing_px=spec.get("closing_px", 0),
        min_valid_frac=spec.get("min_valid_frac", 0.5),
        expected_model_revision=spec.get("expected_model_revision"))
    return out


# ----------------------------------------------------------------------------- exact rotation

def rotate_toward_adjacent(primary: SkylineCurve, adjacent: SkylineCurve, delta_deg: float, cam,
                           adjacent_on_left: bool) -> SkylineCurve:
    """The exact image-plane effect of turning the camera by ``delta_deg`` toward the adjacent view.

    Pinhole, horizontal optical axis: a direction (azimuth a from the axis, elevation e) projects to
    ``x = cx + f tan a``, ``row = cy − f tan e / cos a``. The rotated frame's column x' sees world
    azimuth ``a' ∓ δ``; content beyond the primary's ±45° edge is taken from the adjacent view, whose
    axis is 90° away. Rows are converted through the elevation angle. δ = 0 is the identity.
    """
    W, f, cx, cy = cam.width_px, cam.fx, cam.cx, cam.cy
    d = math.radians(delta_deg)
    cols = np.arange(W, dtype=np.float64)
    a_new = np.arctan((cols - cx) / f)
    a_world = a_new - d if adjacent_on_left else a_new + d
    if adjacent_on_left:
        from_primary = a_world >= -math.pi / 4 - 1e-12
        a_adj = a_world + math.pi / 2
    else:
        from_primary = a_world <= math.pi / 4 + 1e-12
        a_adj = a_world - math.pi / 2
    x_p = cx + f * np.tan(a_world)
    x_a = cx + f * np.tan(a_adj)
    r_p = np.interp(np.clip(x_p, 0, W - 1), cols, np.asarray(primary.row_per_col, dtype=np.float64))
    r_a = np.interp(np.clip(x_a, 0, W - 1), cols, np.asarray(adjacent.row_per_col, dtype=np.float64))
    a_src = np.where(from_primary, a_world, a_adj)
    r_src = np.where(from_primary, r_p, r_a)
    x_src = np.where(from_primary, x_p, x_a)
    # The last sampled column (W-1) sits 0.11° inside the 45° field edge, so the seam between the
    # two views falls in a sliver narrower than one column; that sliver is filled by the edge value
    # (np.interp clamps). Anything further than one column outside the sampled field is a genuine
    # departure from the two-view coverage and is refused.
    if np.any(x_src < -1.0) or np.any(x_src > W):
        raise ValueError(f"rotation {delta_deg} deg leaves the two-view field")
    r_new = cy - (cy - r_src) * np.cos(a_src) / np.cos(a_new)
    return SkylineCurve(f"{primary.observation_id}_rot{delta_deg:g}", r_new, W, primary.image_height_px,
                        primary.provenance, primary.source_ref, "synthetic-rotation")


# ----------------------------------------------------------------------------- evaluation

def evaluate_arm(query_xy, ref_ids, ref_xy, scores, keep, rule: dict) -> dict:
    acc = recog.Acceptance(theta_s=rule["theta_s"], theta_m=rule["theta_m"])
    o = recog.retrieval_outcome(query_xy, ref_ids, ref_xy, scores, np.zeros(len(ref_ids)), keep=keep,
                                tau_region=rule["tau_region"], k=5, acceptance=acc)
    if o.get("outcome") == "EMPTY_DATABASE":
        return {"empty": True}
    return {"top1_id": o["top1_id"], "top1_score": o["top1_score"], "top1_distance_m": o["top1_distance_m"],
            "exact_top1": bool(o["exact_top1"]), "region_top1": bool(o["region_top1"]),
            "accepted": bool(o["accepted_region"]), "region_level_margin": o["region_level_margin"],
            "d_near_m": o["d_near_m"], "nearest_score": o["nearest_score"], "attainable": bool(o["attainable"])}


def strict_arm(a: dict, b: dict, ref_xy_by_id: dict, rule: dict) -> dict:
    if a.get("empty") or b.get("empty"):
        return {"empty": True}
    ax, ay = ref_xy_by_id[a["top1_id"]]
    bx, by = ref_xy_by_id[b["top1_id"]]
    agree = math.hypot(ax - bx, ay - by) <= rule["tau_agree"]
    return {"top1_id": a["top1_id"], "top1_distance_m": a["top1_distance_m"],
            "exact_top1": a["exact_top1"] and b["exact_top1"], "region_top1": a["region_top1"] and agree,
            "accepted": bool(a["accepted"] and b["accepted"] and agree), "agree": agree,
            "d_near_m": a["d_near_m"], "nearest_score": min(a["nearest_score"], b["nearest_score"]),
            "attainable": a["attainable"]}


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float | None:
    """One-sided exact (Clopper–Pearson) 95 % upper bound on a binomial proportion. Only the
    zero-count case is needed here (k = 0 → 1 − alpha^(1/n), the exact 'rule of three' form);
    a positive count is reported as the count itself."""
    if n <= 0 or k != 0:
        return None
    return 1.0 - alpha ** (1.0 / n)


# ----------------------------------------------------------------------------- main study

def run_level(level: str, source: str, rows: list[dict], cache: _Curves, hc_rows: list[dict] | None,
              hc_cache: _Curves | None, cam, out_dir: Path, verify_every: int = 25) -> dict:
    refs_all = [r for r in rows if r["level_key"] == level and is_dense_reference(r)]
    queries_all = [r for r in rows if r["level_key"] == level and r["segment_kind"] != "anchor_capture"]

    def both(cache_, r):
        return all(cache_.get(r[f"{v}_observation_id"]) is not None for v in VIEWS)
    refs = [r for r in refs_all if both(cache, r)]
    queries = [r for r in queries_all if both(cache, r)]
    ref_ids = [key_of(r) for r in refs]
    ref_xy = np.asarray([(r["east_m"], r["north_m"]) for r in refs], dtype=np.float64)
    ref_xy_by_id = {k: (float(x), float(y)) for k, (x, y) in zip(ref_ids, ref_xy)}
    info = {"level": level, "source": source, "n_references": len(refs), "n_queries": len(queries),
            "n_reference_candidates": len(refs_all), "n_query_candidates": len(queries_all)}
    print(f"[{level}/{source}] {len(queries)} queries x {len(refs)} references")

    # per-view bank; every query scored ONCE over the full ±32 grid, but only the 13 lag columns the
    # pre-specified offsets need are kept (the full Q×R×65 matrices tripped the host's memory guard);
    # the bound-32 winner is read off the full row before it is discarded, for the EXP-SKY-011 gate.
    lags = np.arange(-MAX_LAG, MAX_LAG + 1, dtype=np.float64)
    full = {}
    win = {}
    checks = 0
    t0 = time.time()
    for v in VIEWS:
        profs = [cache.profiles[r[f"{v}_observation_id"]] for r in refs]
        bank = NestedBoundBank(ref_ids, profs, bounds=(0, MAX_LAG), max_lag=MAX_LAG, min_overlap_frac=MIN_OVERLAP)
        M = np.empty((len(queries), len(refs), len(NEEDED_LAGS)), dtype=np.float64)
        W = np.empty((len(queries), len(refs)), dtype=np.float64)
        for qi, q in enumerate(queries):
            qp = cache.profiles[q[f"{v}_observation_id"]]
            row = bank.score_full(qp)
            W[qi] = recog.winner_per_reference(row, lags)[0]
            M[qi] = row[:, NEEDED_COLS]
            if qi % verify_every == 0:
                verify_bounds(bank, qp, profs, indices=[qi % len(profs), (qi * 7) % len(profs)])
                checks += 1
        full[v] = M
        win[v] = W
        del bank
    info["frozen_cross_checks"] = checks
    info["score_seconds"] = time.time() - t0

    # gate against the committed EXP-SKY-011 C1-32 winners (same pairs)
    gate = {}
    for v in VIEWS:
        p = REPO / "evaluations/sky-dual/cells" / f"scores_{level}_{source}_{v}.npz"
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                qk = {k: i for i, k in enumerate(z["query_keys"].tolist())}
                rk = {k: i for i, k in enumerate(z["reference_keys"].tolist())}
                S_old = z["scores"]
            worst = 0.0
            n = 0
            for qi, q in enumerate(queries):
                if key_of(q) not in qk:
                    continue
                wq = win[v][qi]
                for ri, rid in enumerate(ref_ids):
                    if rid in rk:
                        worst = max(worst, abs(float(wq[ri]) - float(S_old[qk[key_of(q)], rk[rid]])))
                        n += 1
            gate[v] = {"pairs_compared": n, "max_abs_delta_vs_EXP-SKY-011": worst, "passed": worst <= 1e-12}
            if worst > 1e-12:
                raise SystemExit(f"{level}/{source}/{v}: rebuilt C1-32 winners differ from EXP-SKY-011 ({worst})")
    info["gate_vs_exp_sky_011"] = gate

    q_xy = np.asarray([(q["east_m"], q["north_m"]) for q in queries], dtype=np.float64)
    q_keys = [key_of(q) for q in queries]
    self_idx = [ref_ids.index(k) if k in ref_ids else -1 for k in q_keys]
    dist_qr = np.hypot(q_xy[:, None, 0] - ref_xy[None, :, 0], q_xy[:, None, 1] - ref_xy[None, :, 1])

    memories = {"dense_self": [np.array([j != self_idx[qi] for j in range(len(refs))]) for qi in range(len(queries))],
                "dense_lso100": [(dist_qr[qi] > LSO_RADIUS) & np.array([j != self_idx[qi] for j in range(len(refs))])
                                 for qi in range(len(queries))]}

    def score_at(v, lag):
        return full[v][:, :, NEEDED_INDEX[lag]]

    results = {}
    for delta in OFFSETS_DEG:
        lag = lag_for(delta)
        signs = (0,) if lag == 0 else (+1, -1)
        for sgn in signs:
            ell = sgn * lag
            S = {v: score_at(v, ell) for v in VIEWS}
            S["weakest"] = np.minimum(S["north"], S["west"])
            for mem_name, keeps in memories.items():
                for rule_name, rule in RULES.items():
                    agg = defaultdict(lambda: {"n": 0, "attainable": 0, "exact_top1": 0, "region_top1": 0,
                                               "accepted": 0, "accepted_false": 0, "true_match_scores": []})
                    for qi in range(len(queries)):
                        keep = keeps[qi]
                        if keep.sum() < 2:
                            continue
                        outs = {}
                        for arm in ("north", "west", "weakest"):
                            outs[arm] = evaluate_arm(q_xy[qi], ref_ids, ref_xy, S[arm][qi], keep, rule)
                        outs["strict"] = strict_arm(outs["north"], outs["west"], ref_xy_by_id, rule)
                        for arm, o in outs.items():
                            if o.get("empty"):
                                continue
                            a = agg[arm]
                            a["n"] += 1
                            a["attainable"] += int(o["attainable"])
                            a["exact_top1"] += int(o["exact_top1"])
                            a["region_top1"] += int(o["region_top1"])
                            a["accepted"] += int(o["accepted"])
                            a["accepted_false"] += int(o["accepted"] and o["top1_distance_m"] > rule["wrong_place"])
                            if mem_name == "dense_self" and o["d_near_m"] is not None and o["d_near_m"] <= TRUE_MATCH_MAX_M:
                                a["true_match_scores"].append(float(o["nearest_score"]))
                    for arm, a in agg.items():
                        tm = np.asarray(a.pop("true_match_scores"), dtype=np.float64)
                        rec = {**a, "coverage": a["accepted"] / max(1, a["n"]),
                               "exact_top1_rate": a["exact_top1"] / max(1, a["n"]),
                               "region_top1_rate": a["region_top1"] / max(1, a["n"]),
                               "accepted_false_upper95": clopper_pearson_upper(a["accepted_false"], a["n"]) if a["accepted_false"] == 0 else None}
                        if tm.size:
                            rec["true_match_score_median"] = float(np.median(tm))
                            rec["true_match_score_p10"] = float(np.percentile(tm, 10))
                            rec["true_match_n"] = int(tm.size)
                        results[f"{mem_name}|{rule_name}|{arm}|{delta:g}|{'+' if sgn >= 0 else '-'}"] = rec

    # hard city (city level only): frozen city memory, leave-site-out 125 m, every accept false
    hard = None
    if level == "city" and hc_rows:
        hq = [r for r in hc_rows if both(hc_cache, r)]
        hq_xy = np.asarray([(q["east_m"], q["north_m"]) for q in hq], dtype=np.float64)
        horiz = np.array([q.get("leg_axis") != "vertical" for q in hq])
        fullh = {}
        for v in VIEWS:
            profs = [cache.profiles[r[f"{v}_observation_id"]] for r in refs]
            bank = NestedBoundBank(ref_ids, profs, bounds=(0, MAX_LAG), max_lag=MAX_LAG, min_overlap_frac=MIN_OVERLAP)
            M = np.empty((len(hq), len(refs), len(NEEDED_LAGS)), dtype=np.float64)
            for qi, q in enumerate(hq):
                M[qi] = bank.score_full(hc_cache.profiles[q[f"{v}_observation_id"]])[:, NEEDED_COLS]
            fullh[v] = M
            del bank
        dist_h = np.hypot(hq_xy[:, None, 0] - ref_xy[None, :, 0], hq_xy[:, None, 1] - ref_xy[None, :, 1])
        hard = {"n_queries": len(hq), "n_horizontal": int(horiz.sum()), "results": {}}
        for delta in OFFSETS_DEG:
            lag = lag_for(delta)
            for sgn in ((0,) if lag == 0 else (+1, -1)):
                ell = sgn * lag
                S = {v: fullh[v][:, :, NEEDED_INDEX[ell]] for v in VIEWS}
                S["weakest"] = np.minimum(S["north"], S["west"])
                for rule_name, rule in RULES.items():
                    cnt = defaultdict(lambda: {"n": 0, "accepted": 0, "accepted_horizontal": 0, "n_horizontal": 0})
                    for qi in range(len(hq)):
                        keep = dist_h[qi] > HARDCITY_LSO
                        if keep.sum() < 2:
                            continue
                        outs = {arm: evaluate_arm(hq_xy[qi], ref_ids, ref_xy, S[arm][qi], keep, rule)
                                for arm in ("north", "west", "weakest")}
                        outs["strict"] = strict_arm(outs["north"], outs["west"], ref_xy_by_id, rule)
                        for arm, o in outs.items():
                            if o.get("empty"):
                                continue
                            c = cnt[arm]
                            c["n"] += 1
                            c["n_horizontal"] += int(horiz[qi])
                            c["accepted"] += int(o["accepted"])
                            c["accepted_horizontal"] += int(o["accepted"] and horiz[qi])
                    for arm, c in cnt.items():
                        hard["results"][f"{rule_name}|{arm}|{delta:g}|{'+' if sgn >= 0 else '-'}"] = {
                            **c, "accepted_false_upper95": clopper_pearson_upper(c["accepted"], c["n"]) if c["accepted"] == 0 else None}
    info["hard_city"] = hard

    # secondary exact check (sim_exact only): rotated profile scored POINTWISE vs unrotated nearest reference
    exact = None
    if source == "sim_exact":
        exact = {}
        idx_near = [int(np.argmin(np.where(memories["dense_self"][qi], dist_qr[qi], np.inf))) for qi in range(len(queries))]
        for v, adj, adj_left in (("north", "west", True), ("west", "north", False)):
            per_delta = {}
            for delta in OFFSETS_DEG:
                pw, tr = [], []
                for qi, q in enumerate(queries):
                    if dist_qr[qi, idx_near[qi]] > TRUE_MATCH_MAX_M:
                        continue
                    cp = cache.curves[q[f"{v}_observation_id"]]
                    ca = cache.curves[q[f"{adj}_observation_id"]]
                    rc = rotate_toward_adjacent(cp, ca, delta, cam, adjacent_on_left=adj_left)
                    prof = relpose.profile_of(rc)
                    if delta == 0.0:
                        assert np.allclose(prof, cache.profiles[q[f"{v}_observation_id"]], atol=1e-12), "rotation gate"
                    ref_prof = cache.profiles[refs[idx_near[qi]][f"{v}_observation_id"]]
                    pw.append(float(recog.winner_per_reference(
                        NestedBoundBank([ref_ids[idx_near[qi]]], [ref_prof], bounds=(0,), max_lag=0,
                                        min_overlap_frac=MIN_OVERLAP).score_full(prof), np.array([0.0]))[0][0]))
                    lag = lag_for(delta)
                    tr.append(float(np.mean([full[v][qi, idx_near[qi], NEEDED_INDEX[s * lag]] for s in ((0,) if lag == 0 else (1, -1))])))
                pw, tr = np.asarray(pw), np.asarray(tr)
                per_delta[f"{delta:g}"] = {"n": int(pw.size), "pointwise_exact_median": float(np.median(pw)) if pw.size else None,
                                           "pointwise_exact_p10": float(np.percentile(pw, 10)) if pw.size else None,
                                           "truncated_model_median": float(np.median(tr)) if tr.size else None,
                                           "median_model_minus_exact": float(np.median(tr - pw)) if pw.size else None}
            exact[v] = per_delta
    info["exact_rotation_check"] = exact
    info["results"] = results
    return info


def tolerance_table(res: dict, hard: dict | None) -> dict:
    """The pre-frozen rule: largest δ with 0 false (lso100 and hard city) and coverage ≥ 80 % of δ=0."""
    out = {}
    for rule in RULES:
        for arm in ("north", "west", "weakest", "strict"):
            base = res.get(f"dense_self|{rule}|{arm}|0|+", {}).get("coverage", 0.0)
            tol, graceful = None, True
            prev_cov = None
            for delta in OFFSETS_DEG:
                signs = ("+",) if delta == 0 else ("+", "-")
                cov = float(np.mean([res[f"dense_self|{rule}|{arm}|{delta:g}|{s}"]["coverage"] for s in signs]))
                false_lso = sum(res[f"dense_lso100|{rule}|{arm}|{delta:g}|{s}"]["accepted_false"] for s in signs)
                false_self = sum(res[f"dense_self|{rule}|{arm}|{delta:g}|{s}"]["accepted_false"] for s in signs)
                false_hard = sum(hard["results"][f"{rule}|{arm}|{delta:g}|{s}"]["accepted"] for s in signs) if hard else 0
                ok = (false_lso + false_self + false_hard) == 0 and (base > 0 and cov >= 0.8 * base)
                if prev_cov is not None and cov > prev_cov + 1e-9:
                    graceful = False
                prev_cov = cov
                if ok:
                    tol = delta
                else:
                    break
            out[f"{rule}|{arm}"] = {"tolerance_deg": tol, "coverage_at_0": base, "monotone_coverage": graceful}
    return out


def write_summary(all_info: list[dict], out: Path) -> None:
    lines = ["# EXP-SKY-014 Q-B — azimuth-error tolerance (generated by sky_azimuth_tolerance.py)", "",
             f"Offsets {OFFSETS_DEG} deg → lags {[lag_for(d) for d in OFFSETS_DEG]} samples at {DEG_PER_SAMPLE:.7f} deg/sample. "
             "Perturbation = fixed-lag frozen NCC on the overlapping samples (optimistic bound for pointwise C0; see the exact check). "
             "Values are the mean over the two signs; the worse sign is in parentheses where it differs.", ""]
    for info in all_info:
        lines.append(f"## {info['level']} / {info['source']} — {info['n_queries']} queries × {info['n_references']} references; "
                     f"frozen cross-checks {info['frozen_cross_checks']}; gate vs EXP-SKY-011: "
                     + ", ".join(f"{v}: Δ≤{g['max_abs_delta_vs_EXP-SKY-011']:.1e} over {g['pairs_compared']} pairs" for v, g in info['gate_vs_exp_sky_011'].items()))
        res = info["results"]
        for rule in RULES:
            lines.append(f"\n### {rule} (θ_s {RULES[rule]['theta_s']}, θ_m {RULES[rule]['theta_m']}, region {RULES[rule]['tau_region']:g} m, agree {RULES[rule]['tau_agree']:g} m, wrong-place > {RULES[rule]['wrong_place']:g} m)\n")
            lines.append("| δ (deg) | arm | coverage (dense_self) | region-top-1 | true-match score med / p10 | false (dense_self) | false (lso100) | false (hard city) |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for delta in OFFSETS_DEG:
                signs = ("+",) if delta == 0 else ("+", "-")
                for arm in ("north", "west", "weakest", "strict"):
                    rs = [res[f"dense_self|{rule}|{arm}|{delta:g}|{s}"] for s in signs]
                    rl = [res[f"dense_lso100|{rule}|{arm}|{delta:g}|{s}"] for s in signs]
                    cov = np.mean([r["coverage"] for r in rs]); cov_w = min(r["coverage"] for r in rs)
                    rt1 = np.mean([r["region_top1_rate"] for r in rs])
                    tm = [r.get("true_match_score_median") for r in rs if r.get("true_match_score_median") is not None]
                    tp = [r.get("true_match_score_p10") for r in rs if r.get("true_match_score_p10") is not None]
                    fs = sum(r["accepted_false"] for r in rs); fl = sum(r["accepted_false"] for r in rl)
                    fh = sum(info["hard_city"]["results"][f"{rule}|{arm}|{delta:g}|{s}"]["accepted"] for s in signs) if info.get("hard_city") else "—"
                    covs = f"{cov:.3f}" + (f" ({cov_w:.3f})" if abs(cov_w - cov) > 1e-9 else "")
                    tms = f"{np.mean(tm):.3f} / {np.mean(tp):.3f}" if tm else "—"
                    lines.append(f"| {delta:g} | {arm} | {covs} | {rt1:.3f} | {tms} | {fs} | {fl} | {fh} |")
        tol = tolerance_table(res, info.get("hard_city"))
        lines.append("\n**Tolerance (pre-frozen rule: 0 false anywhere and coverage ≥ 80 % of δ=0):** " +
                     "; ".join(f"{k}: {v['tolerance_deg'] if v['tolerance_deg'] is not None else 'none'}° (cov₀ {v['coverage_at_0']:.2f}{'' if v['monotone_coverage'] else ', non-monotone'})" for k, v in tol.items()))
        if info.get("exact_rotation_check"):
            lines.append("\n**Exact rotation check (sim_exact, pointwise C0 on the synthesised rotated profile vs the truncated-lag model; true-match pairs ≤ 15 m):**\n")
            lines.append("| view | δ | n | exact pointwise median / p10 | truncated model median | model − exact (median) |")
            lines.append("|---|---|---|---|---|---|")
            for v, pd in info["exact_rotation_check"].items():
                for d, r in pd.items():
                    if r["n"]:
                        lines.append(f"| {v} | {d} | {r['n']} | {r['pointwise_exact_median']:.3f} / {r['pointwise_exact_p10']:.3f} | {r['truncated_model_median']:.3f} | {r['median_model_minus_exact']:+.3f} |")
        lines.append("")
    (out / "azimuth_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(all_info: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for j, level in enumerate(LEVELS):
        for i, source in enumerate(SOURCES):
            ax = axes[i, j]
            info = next((x for x in all_info if x["level"] == level and x["source"] == source), None)
            if info is None:
                ax.set_axis_off(); continue
            res = info["results"]
            for arm, ls in (("north", "-"), ("west", "--"), ("weakest", "-."), ("strict", ":")):
                cov = [np.mean([res[f"dense_self|R-INT|{arm}|{d:g}|{s}"]["coverage"] for s in (("+",) if d == 0 else ("+", "-"))]) for d in OFFSETS_DEG]
                ax.plot(OFFSETS_DEG, cov, ls, marker="o", ms=3, label=f"{arm} coverage")
                fal = [sum(res[f"dense_lso100|R-INT|{arm}|{d:g}|{s}"]["accepted_false"] for s in (("+",) if d == 0 else ("+", "-"))) for d in OFFSETS_DEG]
                if info.get("hard_city"):
                    fal = [f + sum(info["hard_city"]["results"][f"R-INT|{arm}|{d:g}|{s}"]["accepted"] for s in (("+",) if d == 0 else ("+", "-"))) for f, d in zip(fal, OFFSETS_DEG)]
                ax2 = ax.twinx() if arm == "north" else ax2
                ax2.plot(OFFSETS_DEG, fal, ls, marker="x", ms=4, color="tab:red", alpha=0.7, label=f"{arm} false accepts")
            ax.set_ylim(0, 1.02); ax.set_xlabel("azimuth error δ (deg)"); ax.set_ylabel("coverage, dense_self (R-INT gate)")
            ax2.set_ylabel("false accepts (lso100 + hard city)", color="tab:red")
            ax.set_title(f"{level} / {source}")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("EXP-SKY-014: coverage (left axis) and false accepts (right axis) versus assumed-azimuth error, per arm — frozen INT gate 0.90/0.30/50 m")
    fig.tight_layout(); fig.savefig(out / "fig_azimuth_tolerance.png", dpi=140); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "evaluations/claim-closure-2026-09/sky")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--sources", default=",".join(SOURCES))
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = HSR / "configs/sky-dual.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent
    hc_cfg = json.loads((HSR / "configs/sky-dual-hardcity.json").read_text(encoding="utf-8"))

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (base / p).resolve()
    store = resolve(cfg["store_root"])
    hc_store = resolve(hc_cfg["store_root"])
    rows = [r for r in load_index(REPO / "evaluations/sky-dual/index.csv") if r["pair_ok"]]
    hc_rows = [r for r in load_index(REPO / "evaluations/sky-dual-hardcity/index.csv") if r["pair_ok"]]
    session_ids = sorted({r[f"{v}_session_id"] for r in rows for v in VIEWS})
    sessions = session_map(store, session_ids)
    cam = _camera(store, session_ids, int(cfg["profile_n_samples"]))
    hc_session_ids = sorted({r[f"{v}_session_id"] for r in hc_rows for v in VIEWS})
    hc_sessions = session_map(hc_store, hc_session_ids)
    sources = make_sources(cfg, base, store, sessions, cam, Path(cfg["segformer"]["mask_root"]))
    hc_sources = make_sources(cfg, base, hc_store, hc_sessions, cam, Path(hc_cfg["segformer_mask_root"]))

    all_info = []
    for source in args.sources.split(","):
        cache = _Curves(sources[source])
        hc_cache = _Curves(hc_sources[source])
        for level in args.levels.split(","):
            ckpt = out / f"cell_{level}_{source}.json"
            if ckpt.exists():                       # a finished cell survives a killed run
                info = json.loads(ckpt.read_text(encoding="utf-8"))
                print(f"[{level}/{source}] reused checkpoint {ckpt.name}")
            else:
                info = run_level(level, source, rows, cache, hc_rows if level == "city" else None,
                                 hc_cache if level == "city" else None, cam, out)
                info["tolerance"] = tolerance_table(info["results"], info.get("hard_city"))
                ckpt.write_text(json.dumps(info, indent=1), encoding="utf-8")
            all_info.append(info)
            print(f"  tolerance: {json.dumps(info['tolerance'])}")
    payload = {"experiment": "EXP-SKY-014 Q-B", "offsets_deg": OFFSETS_DEG, "lags": [lag_for(d) for d in OFFSETS_DEG],
               "degrees_per_sample": DEG_PER_SAMPLE, "rules": RULES, "lso_radius_m": LSO_RADIUS,
               "hardcity_lso_m": HARDCITY_LSO, "true_match_max_m": TRUE_MATCH_MAX_M,
               "camera": cam.as_dict(), "levels": all_info}
    (out / "azimuth_tolerance.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    write_summary(all_info, out)
    write_figure(all_info, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
