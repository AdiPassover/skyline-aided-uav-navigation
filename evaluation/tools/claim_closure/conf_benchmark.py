"""EXP-CONF-006: does ``delta_rot_refit`` carry information beyond conventional estimator-internal
diagnostics? (thesis claim-closure pass 2026-09, Phase 1)

Reads committed run artifacts only (no VO re-run, no imagery): the frozen capture-off / conf runs
(conventional signals), the diagnostic-refit runs (``delta_rot_refit``), the EXP-CONF-005 /
EXP-CONF-003 independent references (T-B) and the fused-attitude streams (T-C). Implements exactly
the front half of the EXP-CONF-006 pre-registration: fixed
signal set with a-priori directions, common frame masks, AP / AUROC / Spearman / review-budget
recall, the EXP-CONF-005 block bootstrap, the three complementarity analyses, the restart lead
tables, and the sanity gate that reproduces EXP-CONF-005's numbers before anything is written.

No fitting, no thresholds, no probability.

    $PY evaluation/tools/claim_closure/conf_benchmark.py --out evaluations/claim-closure-2026-09/conf
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
REPO = TOOL_DIR.parents[2]
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_001"))
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_002"))
from p4_analysis import average_precision, spearman                             # noqa: E402
from panel_analysis import tail_flags                                            # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "exp_conf_003_analysis", TOOL_DIR.parent / "exp_conf_003" / "analysis.py")
_a3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a3)
read_cols, unwrap_interp_deg = _a3.read_cols, _a3.unwrap_interp_deg

HELDOUT = ("hkairport03", "amtown03")
DEV = ("hkairport01-b", "amtown01-d")
FROZEN_RUN = {"hkairport03": "hkairport03-homography-rigid-conf-off-v1",
              "amtown03": "amtown03-homography-rigid-conf-off-v1",
              "hkairport01-b": "hkairport01-b-homography-rigid-conf-v1",
              "amtown01-d": "amtown01-d-homography-rigid-conf-v1"}
DIAG_RUN = {s: f"{s}-homography-rigid-diagrefit-v1" for s in HELDOUT + DEV}
REF_CSV = {"hkairport03": REPO / "evaluations/exp-conf-005/reference/hkairport03.csv",
           "amtown03": REPO / "evaluations/exp-conf-005/reference/amtown03.csv",
           "hkairport01-b": REPO / "evaluations/exp-conf-003/phase_a/hkairport01-b.csv",
           "amtown01-d": REPO / "evaluations/exp-conf-003/phase_a/amtown01-d.csv"}
ATT_CSV = {"hkairport03": REPO / "datasets/hkairport03/attitude.csv",
           "amtown03": REPO / "datasets/amtown03/attitude.csv",
           "hkairport01-b": REPO / "datasets/hkairport01-b/attitude.csv",
           # locally ingested attitude sidecar (not part of the repository)
           "amtown01-d": REPO / "datasets/amtown01-d/attitude.csv"}
DEV_SIGN_JSON = {s: REPO / f"evaluations/exp-conf-001-dev/labels/{s}.json" for s in DEV}
RESTARTS_LEAD = 15
BOOT_L, BOOT_B, BOOT_SEED = 100, 500, 20260831
BUDGETS = (0.02, 0.05, 0.10)
GATE_TOL = 1e-9

# (name, source, column, direction) — direction +1: higher = more suspect; -1: lower = more suspect
HELDOUT_SIGNALS = [
    ("S1_delta_rot_refit", "diag", "delta_rotation_deg", +1),
    ("S2_inlier_count", "frames", "inlier_count", -1),
    ("S3_track_count", "frames", "track_count", -1),
    ("S4_inlier_ratio", "derived", "inlier_ratio", -1),
    ("S5_inc_flow_px", "logical", "inc_flow_px", +1),
    ("S6_inc_anisotropy", "logical", "inc_anisotropy", +1),
    ("S7_inc_deformation", "logical", "inc_deformation", +1),
    ("S8_inc_perspective", "logical", "inc_perspective", +1),
    ("S9_abs_inc_log_scale", "logical_abs", "inc_log_scale", +1),
    ("S10_abs_inc_rotation_deg", "logical_abs", "inc_rotation_deg", +1),
]
DEV_EXTRA_SIGNALS = [
    ("S11_residual_rms_px", "frames", "residual_rms_px", +1),
    ("S12_residual_max_sq_px", "frames", "residual_max_sq_px", +1),
    ("S13_inlier_coverage", "frames", "inlier_coverage", -1),
    ("S14_relative_support", "frames", "relative_support", -1),
    ("S15_inc_log_scale_dispersion", "frames", "inc_log_scale_dispersion", +1),
]
CONVENTIONAL_PREFIX = ("S2_", "S3_", "S4_", "S5_", "S6_", "S7_", "S8_", "S9_", "S10_",
                       "S11_", "S12_", "S13_", "S14_", "S15_")


# ----------------------------------------------------------------------------- loading

def load_csv_cols(path: Path, cols: list[str], n: int, key: str = "frame_index") -> dict[str, np.ndarray]:
    out = {c: np.full(n, np.nan) for c in cols}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i = int(r[key])
            if i >= n:
                continue
            for c in cols:
                v = r.get(c, "")
                if v not in ("", None):
                    out[c][i] = float(v)
    return out


def load_sequence(seq: str) -> dict:
    frozen = REPO / "runs" / FROZEN_RUN[seq]
    diag = REPO / "runs" / DIAG_RUN[seq]
    if (frozen / "logical_transform.csv").read_bytes() != (diag / "logical_transform.csv").read_bytes():
        raise SystemExit(f"{seq}: frozen and diagnostic runs are not the same VO track")
    with (frozen / "frames.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    events = np.array([r["event"] for r in rows])
    t_cam = np.array([float(r["timestamp_s"]) for r in rows])
    frame_cols = ["track_count", "inlier_count", "residual_rms_px", "residual_max_sq_px",
                  "inlier_coverage", "relative_support", "inc_log_scale_dispersion"]
    frames = {c: np.full(n, np.nan) for c in frame_cols}
    for i, r in enumerate(rows):
        for c in frame_cols:
            v = r.get(c, "")
            if v not in ("", None):
                frames[c][i] = float(v)
    logical = load_csv_cols(frozen / "logical_transform.csv",
                            ["inc_flow_px", "inc_anisotropy", "inc_deformation", "inc_perspective",
                             "inc_log_scale", "inc_rotation_deg"], n)
    d = load_csv_cols(diag / "diagnostic_refit.csv",
                      ["delta_rotation_deg", "refit_ok", "inliers"], n)
    ref = load_csv_cols(REF_CSV[seq], ["rot_dis_deg", "n_fb_valid"], n)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = frames["inlier_count"] / frames["track_count"]
    accepted = (events == "none") | (events == "recenter")
    restarts = [int(i) for i in np.where(events == "restart")[0]]

    # T-C: fused attitude per-increment rotation error, exactly the EXP-CONF-005 / -004 construction
    att_path = ATT_CSV[seq]
    err_att = np.full(n, np.nan)
    att_info: dict = {"path": str(att_path), "available": att_path.exists()}
    if att_path.exists():
        att = read_cols(att_path, ["timestamp_s", "yaw_compass_deg"])
        ta = att["timestamp_s"]
        gaps = np.diff(ta)
        gate = {"rate_hz": float(1.0 / np.median(gaps)), "max_gap_s": float(np.max(gaps)),
                "covers": bool(ta[0] <= t_cam[0] and ta[-1] >= t_cam[-1])}
        gate["passed"] = (90.0 <= gate["rate_hz"] <= 110.0 and gate["max_gap_s"] <= 0.5
                          and gate["covers"])
        att_info["continuity"] = gate
        if gate["passed"]:
            att_cam = unwrap_interp_deg(t_cam, ta, att["yaw_compass_deg"])
            d_att = np.full(n, np.nan)
            d_att[1:] = np.diff(att_cam)
            inc_rot = logical["inc_rotation_deg"]
            has_inc = (events != "init") & (events != "restart")
            mm = has_inc & np.isfinite(d_att)
            if seq in DEV:
                sgn = float(json.load(DEV_SIGN_JSON[seq].open(encoding="utf-8"))["t_a_sign_convention"])
                att_info["sign_convention"] = {"chosen": sgn, "source": "EXP-CONF-001 label file"}
            else:
                meds = {sgn_: float(np.median(np.abs(sgn_ * inc_rot[mm] - d_att[mm])))
                        for sgn_ in (1.0, -1.0)}
                sgn = 1.0 if meds[1.0] <= meds[-1.0] else -1.0
                att_info["sign_convention"] = {"chosen": sgn, "median_abs_err": meds,
                                               "source": "EXP-CONF-005 rule"}
            err_att[mm] = np.abs(sgn * inc_rot[mm] - d_att[mm])

    signals: dict[str, np.ndarray] = {}
    spec = list(HELDOUT_SIGNALS) + (list(DEV_EXTRA_SIGNALS) if seq in DEV else [])
    for name, src, col, direction in spec:
        if src == "diag":
            v = d[col]
        elif src == "frames":
            v = frames[col]
        elif src == "derived":
            v = ratio
        elif src == "logical":
            v = logical[col]
        elif src == "logical_abs":
            v = np.abs(logical[col])
        else:
            raise ValueError(src)
        signals[name] = direction * v          # "score": higher = more suspect for every signal
    return {"seq": seq, "n": n, "events": events, "accepted": accepted, "restarts": restarts,
            "signals": signals, "spec": spec, "rot_dis": ref["rot_dis_deg"],
            "n_fb_valid": ref["n_fb_valid"], "err_att": err_att, "att_info": att_info,
            "refit_ok": d["refit_ok"], "raw_drot": d["delta_rotation_deg"],
            "inlier_count": frames["inlier_count"]}


# ----------------------------------------------------------------------------- metrics

def avg_rank(x: np.ndarray) -> np.ndarray:
    """1-based average ranks with ties averaged (finite input only). Vectorised; equal to the
    EXP-CONF-001 ``rankdata`` loop (``0.5 * (i + j) + 1`` over each tie block ``i..j``)."""
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    n = len(s)
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    avg = 0.5 * (starts + ends - 1) + 1.0
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def fast_ap(score: np.ndarray, label: np.ndarray) -> float:
    """Vectorised tie-blocked average precision, equal to the EXP-CONF-001 ``average_precision``
    loop (precision/recall evaluated at the end of every equal-score block)."""
    m = np.isfinite(score)
    s, y = score[m], label[m].astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0 or len(s) == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = np.cumsum(~y)[ends]
    recall = tp / n_pos
    precision = tp / (tp + fp)
    prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))


def fast_spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = avg_rank(x[m]), avg_rank(y[m])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def check_fast_equals_reference(score: np.ndarray, target: np.ndarray, flags: np.ndarray, mask: np.ndarray) -> None:
    """The vectorised implementations must equal the reference ones on real data (1e-12)."""
    a_ref, _, _ = average_precision(score[mask], flags[mask])
    a_fast = fast_ap(score[mask], flags[mask])
    r_ref, _ = spearman(score[mask], target[mask])
    r_fast = fast_spearman(score[mask], target[mask])
    if not (abs(a_ref - a_fast) < 1e-12 and abs(r_ref - r_fast) < 1e-12):
        raise SystemExit(f"fast/reference disagreement: AP {a_ref} vs {a_fast}, rho {r_ref} vs {r_fast}")


def pct_rank(x: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] among finite values (ties averaged); NaN elsewhere."""
    out = np.full(len(x), np.nan)
    m = np.isfinite(x)
    if m.sum() > 1:
        out[m] = (avg_rank(x[m]) - 1.0) / (m.sum() - 1.0)
    return out


def auroc(score: np.ndarray, flags: np.ndarray) -> float:
    m = np.isfinite(score)
    s, y = score[m], flags[m].astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = avg_rank(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ap_of(score: np.ndarray, flags: np.ndarray, mask: np.ndarray) -> float:
    ap, _, _ = average_precision(score[mask], flags[mask])
    return float(ap)


def budget_table(score: np.ndarray, flags: np.ndarray, mask: np.ndarray) -> dict:
    s, y = score[mask], flags[mask].astype(bool)
    n_pos = int(y.sum())
    out = {}
    for b in BUDGETS:
        thr = np.quantile(s, 1.0 - b)
        flagged = s > thr
        caught = int(np.sum(flagged & y))
        out[f"top{int(b * 100)}pct"] = {"flagged": int(flagged.sum()), "caught": caught,
                                        "recall": caught / max(1, n_pos),
                                        "precision": caught / max(1, int(flagged.sum()))}
    return out


def block_bootstrap(score: np.ndarray, target: np.ndarray, flags: np.ndarray, mask: np.ndarray,
                    rng: np.random.Generator) -> dict:
    idx = np.where(mask)[0]
    n = len(idx)
    if n < 500:
        return {"n": n, "note": "too few samples for bootstrap"}
    check_fast_equals_reference(score, target, flags, mask)
    sp, ap = [], []
    n_blocks = int(np.ceil(n / BOOT_L))
    for _ in range(BOOT_B):
        starts = rng.integers(0, n, size=n_blocks)
        take = (starts[:, None] + np.arange(BOOT_L)[None, :]).ravel() % n
        ii = idx[take[:n]]
        sp.append(fast_spearman(score[ii], target[ii]))
        if flags[ii].sum() >= 3:
            ap.append(fast_ap(score[ii], flags[ii]))

    def ci(v):
        v = np.array([x for x in v if np.isfinite(x)])
        return ({"lo2.5": float(np.percentile(v, 2.5)), "hi97.5": float(np.percentile(v, 97.5)),
                 "n_resamples": int(len(v))} if len(v) else {"n_resamples": 0})
    return {"n": n, "block_len": BOOT_L, "B": BOOT_B, "spearman_ci": ci(sp), "ap_ci": ci(ap)}


def score_signals(seq_data: dict, target: np.ndarray, target_name: str, extra: dict[str, np.ndarray]) -> dict:
    """All metrics for every signal (+ combinations) on one (sequence, target) common mask."""
    sig = dict(seq_data["signals"])
    sig.update(extra)
    mask = seq_data["accepted"] & np.isfinite(target)
    for v in sig.values():
        mask &= np.isfinite(v)
    flags = tail_flags(np.where(mask, target, np.nan))
    out: dict = {"target": target_name, "n_frames": int(mask.sum()), "n_pos": int(flags[mask].sum()),
                 "base_rate": float(flags[mask].mean()) if mask.sum() else float("nan"), "signals": {}}
    for name, v in sig.items():
        rng = np.random.default_rng(BOOT_SEED)
        rho, nn = spearman(v[mask], target[mask])
        out["signals"][name] = {
            "ap": ap_of(v, flags, mask), "auroc": auroc(v[mask], flags[mask]),
            "spearman": {"rho": rho, "n": nn}, "budgets": budget_table(v, flags, mask),
            "bootstrap": block_bootstrap(v, target, flags, mask, rng)}
    # ---- complementarity ---------------------------------------------------------------
    s1 = sig["S1_delta_rot_refit"]
    conv = [k for k in sig if k.startswith(CONVENTIONAL_PREFIX)]

    def flag_top(v, b):
        f = np.zeros(len(v), dtype=bool)
        thr = np.quantile(v[mask], 1.0 - b)
        f[mask] = v[mask] > thr
        return f
    tail = flags & mask
    s1_top = flag_top(s1, 0.05)
    any_conv = np.zeros(len(s1), dtype=bool)
    for k in conv:
        any_conv |= flag_top(sig[k], 0.05)
    s2_top = flag_top(sig["S2_inlier_count"], 0.05)
    out["complementarity"] = {
        "budget": 0.05, "n_tail": int(tail.sum()),
        "tail_caught_by_S1": int(np.sum(tail & s1_top)),
        "tail_caught_by_S2": int(np.sum(tail & s2_top)),
        "tail_caught_by_any_conventional": int(np.sum(tail & any_conv)),
        "tail_caught_by_S1_not_S2": int(np.sum(tail & s1_top & ~s2_top)),
        "tail_caught_by_S2_not_S1": int(np.sum(tail & s2_top & ~s1_top)),
        "tail_caught_by_S1_not_any_conventional": int(np.sum(tail & s1_top & ~any_conv)),
        "tail_caught_by_any_conventional_not_S1": int(np.sum(tail & any_conv & ~s1_top)),
        "tail_caught_by_neither": int(np.sum(tail & ~s1_top & ~any_conv)),
    }
    # S1 within terciles of inlier count (raw count, low = suspect)
    ic = seq_data["inlier_count"]
    q = np.quantile(ic[mask], [1 / 3, 2 / 3])
    strata = {"low_inliers": mask & (ic <= q[0]), "mid_inliers": mask & (ic > q[0]) & (ic <= q[1]),
              "high_inliers": mask & (ic > q[1])}
    strat = {}
    for k, sm in strata.items():
        if sm.sum() < 50 or flags[sm].sum() == 0:
            strat[k] = {"n": int(sm.sum()), "n_pos": int(flags[sm].sum()), "note": "too few positives"}
            continue
        rho, nn = spearman(s1[sm], target[sm])
        strat[k] = {"n": int(sm.sum()), "n_pos": int(flags[sm].sum()),
                    "base_rate": float(flags[sm].mean()), "ap_S1": ap_of(s1, flags, sm),
                    "auroc_S1": auroc(s1[sm], flags[sm]), "spearman_S1": rho,
                    "inlier_count_range": [float(ic[sm].min()), float(ic[sm].max())]}
    out["stratified_S1_by_inlier_tercile"] = strat
    out["incremental"] = {"ap_K1": out["signals"]["K1_rankmean_S1_S2"]["ap"],
                          "ap_S1": out["signals"]["S1_delta_rot_refit"]["ap"],
                          "ap_S2": out["signals"]["S2_inlier_count"]["ap"]}
    return out


def restart_tables(seq_data: dict, extra: dict[str, np.ndarray]) -> list[dict]:
    sig = dict(seq_data["signals"])
    sig.update(extra)
    acc = seq_data["accepted"]
    pct = {k: pct_rank(np.where(acc, v, np.nan)) for k, v in sig.items()}
    rows = []
    for r in seq_data["restarts"]:
        lead = np.arange(max(1, r - RESTARTS_LEAD), r)
        row: dict = {"restart": int(r), "n_lead": int(len(lead)), "signals": {}}
        for k in sig:
            p = pct[k][lead]
            fin = np.isfinite(p)
            if fin.sum() == 0:
                row["signals"][k] = {"note": "no finite lead frames"}
                continue
            above = np.where(fin & (p >= 0.95))[0]
            row["signals"][k] = {
                "max_pct": float(np.nanmax(p)),
                "n_lead_ge_p95": int(len(above)),
                "warning_lead_frames": int(r - lead[above[0]]) if len(above) else None,
                "last3_max_pct": float(np.nanmax(p[-3:])) if fin[-3:].any() else None}
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- gate

def sanity_gate(seq_data: dict) -> dict:
    """Reproduce EXP-CONF-005's primary Spearman and AP for delta_rot_refit on its own mask."""
    ref = json.load((REPO / "evaluations/exp-conf-005/analysis.json").open(encoding="utf-8"))
    p = ref["sequences"][seq_data["seq"]]["primary"]
    drot = seq_data["raw_drot"]
    rot_dis = seq_data["rot_dis"]
    m = np.isfinite(drot) & np.isfinite(rot_dis)
    flags = tail_flags(np.where(m, rot_dis, np.nan))
    rho, nn = spearman(drot, rot_dis)
    ap = ap_of(drot, flags, m)
    g = {"expected": {"rho": p["spearman"]["rho"], "ap": p["ap_vs_ref_tail"]["ap"]},
         "recomputed": {"rho": rho, "ap": ap},
         "passed": abs(rho - p["spearman"]["rho"]) < GATE_TOL and abs(ap - p["ap_vs_ref_tail"]["ap"]) < GATE_TOL}
    if not g["passed"]:
        raise SystemExit(f"sanity gate failed on {seq_data['seq']}: {g}")
    return g


# ----------------------------------------------------------------------------- report

def md_table(res: dict, names: list[str]) -> str:
    lines = ["| signal | AP | AP 95% CI | AUROC | Spearman | recall@2% | recall@5% | prec@5% | recall@10% |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k in names:
        s = res["signals"][k]
        ci = s["bootstrap"].get("ap_ci", {})
        cis = f"[{ci['lo2.5']:.3f}, {ci['hi97.5']:.3f}]" if "lo2.5" in ci else "—"
        b = s["budgets"]
        lines.append(f"| {k} | **{s['ap']:.3f}** | {cis} | {s['auroc']:.3f} | {s['spearman']['rho']:+.3f} | "
                     f"{b['top2pct']['recall']:.3f} | {b['top5pct']['recall']:.3f} | {b['top5pct']['precision']:.3f} | "
                     f"{b['top10pct']['recall']:.3f} |")
    return "\n".join(lines)


def verdict(results: dict) -> dict:
    """The pre-frozen rule, mechanically, on the held-out T-B (and separately T-C) results."""
    def one(target_key: str) -> dict:
        per = {}
        for seq in HELDOUT:
            r = results["sequences"][seq][target_key]
            s1 = r["signals"]["S1_delta_rot_refit"]
            conv = {k: v for k, v in r["signals"].items() if k.startswith(CONVENTIONAL_PREFIX)}
            best_name = max(conv, key=lambda k: conv[k]["ap"])
            best_ap = conv[best_name]["ap"]
            lo = s1["bootstrap"].get("ap_ci", {}).get("lo2.5", float("nan"))
            best_ci = conv[best_name]["bootstrap"].get("ap_ci", {})
            per[seq] = {"ap_S1": s1["ap"], "best_conventional": best_name, "ap_best_conventional": best_ap,
                        "S1_beats_all": s1["ap"] > best_ap,
                        "S1_ci_lo_above_best_point": bool(lo > best_ap),
                        "S1_within_best_ci": bool(best_ci and best_ci.get("lo2.5", 1) <= s1["ap"] <= best_ci.get("hi97.5", 0)),
                        "unique_catches_S1": r["complementarity"]["tail_caught_by_S1_not_any_conventional"],
                        "ap_K1": r["incremental"]["ap_K1"], "ap_S2": r["incremental"]["ap_S2"],
                        "K1_beats_both": r["incremental"]["ap_K1"] > max(s1["ap"], r["incremental"]["ap_S2"])}
        beats_both = all(p["S1_beats_all"] for p in per.values())
        ci_some = any(p["S1_ci_lo_above_best_point"] for p in per.values())
        unique_both = all(p["unique_catches_S1"] > 0 for p in per.values())
        best_or_within_some = any(p["S1_beats_all"] or p["S1_within_best_ci"] for p in per.values())
        k1_some = any(p["K1_beats_both"] for p in per.values())
        dominated_both = all((not p["S1_beats_all"]) for p in per.values())
        no_unique = all(p["unique_catches_S1"] == 0 for p in per.values())
        if beats_both and ci_some and unique_both:
            label = "STRONG"
        elif dominated_both and no_unique:
            label = "NOT SUPPORTED"
        elif best_or_within_some or k1_some:
            label = "PARTIAL"
        else:
            label = "NOT SUPPORTED"
        return {"label": label, "per_sequence": per}
    return {"T-B_primary": one("T-B"), "T-C_secondary": one("T-C")}


def write_figures(results: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from p4_analysis import pr_curve
    data = results["_data"]
    # fig 1: PR curves, held-out, T-B
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, seq in zip(axes, HELDOUT):
        sd = data[seq]
        r = results["sequences"][seq]["T-B"]
        target = sd["rot_dis"]
        mask = sd["accepted"] & np.isfinite(target)
        for v in sd["signals_all"].values():
            mask &= np.isfinite(v)
        flags = tail_flags(np.where(mask, target, np.nan))
        for k in ["S1_delta_rot_refit", "S2_inlier_count", "S4_inlier_ratio", "S8_inc_perspective",
                  "S5_inc_flow_px", "K1_rankmean_S1_S2"]:
            pts = pr_curve(sd["signals_all"][k][mask], flags[mask])
            ax.plot([p["recall"] for p in pts], [p["precision"] for p in pts],
                    label=f"{k} (AP {r['signals'][k]['ap']:.3f})", lw=1.4)
        ax.axhline(r["base_rate"], color="k", ls=":", lw=1, label=f"base rate {r['base_rate']:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("recall (worst-2 % rot_dis_independent tail)"); ax.set_ylabel("precision")
        ax.set_title(f"{seq} — T-B (held-out, T3), n = {r['n_frames']}, positives {r['n_pos']}")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("EXP-CONF-006: precision–recall of estimator-internal diagnostics vs the independent instability tail")
    fig.tight_layout(); fig.savefig(out / "fig_pr_heldout.png", dpi=150); plt.close(fig)
    # fig 2: AP bars, all signals, both targets, four sequences
    seqs = list(HELDOUT) + list(DEV)
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharey="row")
    for j, seq in enumerate(seqs):
        for i, tk in enumerate(("T-B", "T-C")):
            ax = axes[i, j]
            r = results["sequences"][seq].get(tk)
            if not r or "signals" not in r:
                ax.set_axis_off(); continue
            names = list(r["signals"].keys())
            aps = [r["signals"][k]["ap"] for k in names]
            los = [r["signals"][k]["bootstrap"].get("ap_ci", {}).get("lo2.5", np.nan) for k in names]
            his = [r["signals"][k]["bootstrap"].get("ap_ci", {}).get("hi97.5", np.nan) for k in names]
            colors = ["tab:red" if k.startswith("S1_") else ("tab:purple" if k.startswith("K") else "tab:gray") for k in names]
            y = np.arange(len(names))
            ax.barh(y, aps, color=colors)
            err = np.array([[a - lo if np.isfinite(lo) else 0 for a, lo in zip(aps, los)],
                            [hi - a if np.isfinite(hi) else 0 for a, hi in zip(aps, his)]])
            ax.errorbar(aps, y, xerr=err, fmt="none", ecolor="k", elinewidth=0.8, capsize=2)
            ax.axvline(r["base_rate"], color="k", ls=":", lw=1)
            ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7); ax.invert_yaxis()
            role = "held-out" if seq in HELDOUT else "development"
            ax.set_title(f"{seq} ({role}) — {tk}, base {r['base_rate']:.3f}", fontsize=9)
            ax.set_xlabel("AP (worst-2 % tail)")
    fig.suptitle("EXP-CONF-006: AP of every diagnostic — T-B independent-reference tail (top), T-C fused-attitude tail (bottom); 95 % block-bootstrap CIs")
    fig.tight_layout(); fig.savefig(out / "fig_ap_bars.png", dpi=140); plt.close(fig)
    # fig 3: S1 within inlier-count terciles (T-B), all sequences
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.2
    for j, seq in enumerate(seqs):
        st = results["sequences"][seq]["T-B"]["stratified_S1_by_inlier_tercile"]
        keys = ["low_inliers", "mid_inliers", "high_inliers"]
        vals = [st[k].get("ap_S1", np.nan) for k in keys]
        bases = [st[k].get("base_rate", np.nan) for k in keys]
        x = np.arange(3) + (j - 1.5) * width
        ax.bar(x, vals, width, label=f"{seq} AP(S1)")
        ax.plot(x, bases, "k_", ms=10)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(["low inlier count", "mid", "high"])
    ax.set_ylabel("AP of delta_rot_refit vs T-B tail (within stratum)")
    ax.set_title("EXP-CONF-006: does the association survive holding support roughly constant? (black ticks = stratum base rate)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_stratified.png", dpi=150); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "evaluations/claim-closure-2026-09/conf")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    results: dict = {"experiment": "EXP-CONF-006", "tail_frac": 0.02,
                     "bootstrap": {"block_len": BOOT_L, "B": BOOT_B, "seed": BOOT_SEED},
                     "roles": {"held_out": list(HELDOUT), "development": list(DEV)},
                     "sequences": {}, "_data": {}}
    for seq in HELDOUT + DEV:
        sd = load_sequence(seq)
        entry: dict = {"role": "held-out" if seq in HELDOUT else "development",
                       "n_frames": sd["n"], "restarts": sd["restarts"],
                       "attitude": sd["att_info"],
                       "refit_failures": int(np.nansum(sd["refit_ok"] == 0))}
        if seq in HELDOUT:
            entry["sanity_gate_vs_EXP-CONF-005"] = sanity_gate(sd)
        # combinations (rank means over the accepted frames)
        acc = sd["accepted"]
        s1p = pct_rank(np.where(acc, sd["signals"]["S1_delta_rot_refit"], np.nan))
        s2p = pct_rank(np.where(acc, sd["signals"]["S2_inlier_count"], np.nan))
        extra = {"K1_rankmean_S1_S2": (s1p + s2p) / 2.0}
        if seq in DEV:
            s11p = pct_rank(np.where(acc, sd["signals"]["S11_residual_rms_px"], np.nan))
            extra["K2_rankmean_S1_S11"] = (s1p + s11p) / 2.0
        entry["T-B"] = score_signals(sd, sd["rot_dis"], "worst-2% tail of rot_dis_independent (B-type)", extra)
        if np.isfinite(sd["err_att"]).sum() > 500:
            entry["T-C"] = score_signals(sd, sd["err_att"], "worst-2% tail of fused-attitude rotation error (C-type)", extra)
        else:
            entry["T-C"] = {"note": "attitude unavailable or continuity gate failed"}
        entry["T-E_restarts"] = restart_tables(sd, extra)
        results["sequences"][seq] = entry
        sd["signals_all"] = dict(sd["signals"]); sd["signals_all"].update(extra)
        results["_data"][seq] = sd
        print(f"{seq}: T-B n={entry['T-B']['n_frames']} pos={entry['T-B']['n_pos']} "
              f"AP(S1)={entry['T-B']['signals']['S1_delta_rot_refit']['ap']:.3f} "
              f"AP(S2)={entry['T-B']['signals']['S2_inlier_count']['ap']:.3f}")
    results["verdict"] = verdict(results)
    write_figures(results, out)
    data = results.pop("_data")

    # ---- summary.md ------------------------------------------------------------------------
    lines = ["# EXP-CONF-006 — diagnostic comparison benchmark (generated by conf_benchmark.py)", "",
             f"Tail = worst 2 % per sequence (rank-based); bootstrap L={BOOT_L}, B={BOOT_B}, seed {BOOT_SEED}. "
             "Every signal is oriented so that higher = more suspect. Frames: accepted increments on which every "
             "signal and the target are finite (one common mask per sequence and target).", ""]
    for seq in HELDOUT + DEV:
        e = results["sequences"][seq]
        lines.append(f"## {seq} ({e['role']}) — {e['n_frames']} frames, restarts {e['restarts']}")
        for tk in ("T-B", "T-C"):
            r = e[tk]
            if "signals" not in r:
                lines.append(f"\n### {tk}: {r['note']}\n"); continue
            lines.append(f"\n### {tk}: {r['target']} — n = {r['n_frames']}, positives {r['n_pos']}, base rate {r['base_rate']:.4f}\n")
            names = sorted(r["signals"], key=lambda k: -r["signals"][k]["ap"])
            lines.append(md_table(r, names))
            c = r["complementarity"]
            lines.append(f"\nComplementarity at the 5 % budget ({c['n_tail']} tail frames): caught by S1 {c['tail_caught_by_S1']}, "
                         f"by S2 {c['tail_caught_by_S2']}, by any conventional {c['tail_caught_by_any_conventional']}; "
                         f"**S1-only (vs any conventional) {c['tail_caught_by_S1_not_any_conventional']}**, "
                         f"conventional-only {c['tail_caught_by_any_conventional_not_S1']}, neither {c['tail_caught_by_neither']}; "
                         f"S1-not-S2 {c['tail_caught_by_S1_not_S2']}, S2-not-S1 {c['tail_caught_by_S2_not_S1']}.")
            st = r["stratified_S1_by_inlier_tercile"]
            lines.append("\nS1 within inlier-count terciles: " + "; ".join(
                f"{k}: n={v['n']}, pos={v['n_pos']}" + (f", AP {v['ap_S1']:.3f} (base {v['base_rate']:.3f}), AUROC {v['auroc_S1']:.3f}, ρ {v['spearman_S1']:+.3f}" if "ap_S1" in v else " (too few positives)")
                for k, v in st.items()))
            inc = r["incremental"]
            lines.append(f"\nIncremental: AP(K1) {inc['ap_K1']:.3f} vs AP(S1) {inc['ap_S1']:.3f}, AP(S2) {inc['ap_S2']:.3f}.\n")
        if e["T-E_restarts"]:
            lines.append("\n### T-E restarts — max lead-window percentile (frames ≥ p95 / warning lead in frames)\n")
            sig_names = list(e["T-E_restarts"][0]["signals"].keys())
            short = [s.split("_")[0] for s in sig_names]
            lines.append("| restart | " + " | ".join(short) + " |")
            lines.append("|---|" + "---|" * len(short))
            for row in e["T-E_restarts"]:
                cells = []
                for k in sig_names:
                    s = row["signals"][k]
                    cells.append("—" if "note" in s else f"{s['max_pct']:.2f} ({s['n_lead_ge_p95']}/{s['warning_lead_frames'] if s['warning_lead_frames'] is not None else '-'})")
                lines.append(f"| {row['restart']} | " + " | ".join(cells) + " |")
            lines.append("")
    v = results["verdict"]
    lines.append("## Verdict (pre-frozen rule, mechanical)\n")
    for tk in ("T-B_primary", "T-C_secondary"):
        lines.append(f"- **{tk}: {v[tk]['label']}**")
        for seq, p in v[tk]["per_sequence"].items():
            lines.append(f"  - {seq}: AP(S1) {p['ap_S1']:.3f} vs best conventional {p['best_conventional']} {p['ap_best_conventional']:.3f}; "
                         f"S1 beats all: {p['S1_beats_all']}; S1 CI-lo above best: {p['S1_ci_lo_above_best_point']}; "
                         f"S1 within best's CI: {p['S1_within_best_ci']}; unique catches {p['unique_catches_S1']}; "
                         f"AP(K1) {p['ap_K1']:.3f} beats both: {p['K1_beats_both']}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else str(o))),
                                      encoding="utf-8")
    print(json.dumps(results["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
