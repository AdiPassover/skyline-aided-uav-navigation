"""`EXP-VO-014`: score three height arms on two paired UE figure-8 runs, fitting no scale anywhere.

Two VO runs, three arms each. The arms differ **only** in which height array is multiplied into the
per-frame increments the estimator already produced, so no arm can perturb the estimator and the six
cells are exactly comparable — `EXP-VO-012`'s architecture, reused unmodified via
`exp_vo_012/metric_readout.py` and `exp_vo_012/metric_metrics.py`.

    A  FIXED    h_used(t) = h0                          the null: no height sensor
    B  BARO     h_used(t) = h0 + baro_relative_alt(t)    the design under test (DEC-VO-007 D3)
    C  ORACLE   h_used(t) = true_agl(t)                  evaluation-only; no sensor supplies it
    C' ORACLE-SMOOTH  median(true_agl, 5)                pre-registered SECONDARY sensitivity check

**Zero fitted scale parameters** in the primary family. `reference_initialised` fits nothing at all;
`align_se2` fits translation and rotation only and holds scale at exactly 1. Sim(2) — which fits
scale — is computed for continuity with every prior VO result and is labelled scale-fitted wherever
it appears. `EXP-VO-012` measured Sim(2) to be *exactly* blind to a constant scale error, which is
the whole reason the primary family exists.

`h0` comes from the simulator (`dataset.json → metadata.h0_agl_m`) and is never fitted to a
trajectory. Visual scale never enters the computation (`DEC-VO-007` D4).

    PY=.venv/Scripts/python.exe
    $PY evaluation/tools/exp_vo_014/analyse.py --out evaluations/exp-vo-014
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVALUATION_DIR = HERE.parents[1]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(HERE.parent / "exp_vo_012"))

import metric_metrics as mm                                              # noqa: E402
import metric_readout as mro                                             # noqa: E402

#: Pre-registered in EXP-VO-014 (Variables), inherited from EXP-VO-012/013 and NOT chosen here.
#: LIT-VO-003 section 10.6 is why the stride is not 1 and why it must be declared.
PATH_STEP = 5
RPE_LENGTHS = (10, 25, 50, 100)
#: Pre-registered secondary sensitivity check: 5 frames at 10 Hz and 7.5 m/s spans ~3.7 m of ground,
#: the scale of one building edge. Chosen from the capture geometry, before any result was read.
ORACLE_SMOOTH_WINDOW = 5

CASES = [
    ("A", "uevo-fig8-const-v1", "uevo-fig8-const-homography-rigid-v1", "constant_height"),
    ("B", "uevo-fig8-vary-v1", "uevo-fig8-vary-homography-rigid-v1", "varying_height"),
]
ARMS = ("fixed", "baro", "oracle", "oracle_smooth")


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def col(rows: list[dict], k: str) -> np.ndarray:
    return np.array([float(r[k]) for r in rows], dtype=float)


def median_filter(v: np.ndarray, w: int) -> np.ndarray:
    pad = w // 2
    ext = np.pad(v, pad, mode="edge")
    return np.array([np.median(ext[i:i + w]) for i in range(len(v))])


# --------------------------------------------------------------------------- loading


def load_case(label: str, dataset_id: str, run_id: str, role: str) -> dict:
    ds = REPO / "datasets" / dataset_id
    run = REPO / "runs" / run_id
    meta = json.loads((ds / "dataset.json").read_text())
    intr = meta["metadata"]["camera_intrinsics"]
    inc = mro.load_increments(run, intr["width"], intr["height"])

    # Ground truth by EXACT timestamp, not interpolation: the ingest verified that every image
    # timestamp is bit-identical to a telemetry sample time, so a lookup is exact and an
    # interpolation would only hide a future desynchronisation.
    gt = read_csv(ds / "groundtruth.csv")
    t_gt = col(gt, "timestamp_s")
    terr_rows = read_csv(ds / "terrain.csv")
    by_index = {int(r["frame_index"]): r for r in terr_rows}
    idx = inc.frame_index
    t_frame = np.array([float(by_index[int(i)]["timestamp_s"]) for i in idx])
    j = np.searchsorted(t_gt, t_frame)
    j = np.clip(j, 0, len(t_gt) - 1)
    if np.abs(t_gt[j] - t_frame).max() > 1e-6:
        raise RuntimeError("frame timestamps are not present in groundtruth.csv")

    agl = np.array([float(by_index[int(i)]["agl_m"]) for i in idx])
    baro = np.array([float(by_index[int(i)]["baro_relative_m"]) for i in idx])
    terrain = np.array([float(by_index[int(i)]["terrain_m"]) for i in idx])

    health = read_csv(run / "frames.csv")
    success = np.array([r["success"].strip().lower() == "true" for r in health])
    events = [r["event"] for r in health]

    return {
        "label": label, "role": role, "dataset_id": dataset_id, "run_id": run_id,
        "n_frames": len(inc),
        "east": col(gt, "east_m")[j], "north": col(gt, "north_m")[j],
        "up": col(gt, "up_m")[j], "heading": col(gt, "heading_deg")[j], "t": t_frame,
        "agl": agl, "baro": baro, "terrain": terrain,
        "h0": float(meta["metadata"]["h0_agl_m"]),
        "f_working": mro.f_working(intr["fx"], 1),
        "inc": inc,
        "health": {
            "n_frames": len(health),
            "frame_success_rate": float(success.mean()),
            "n_failed": int((~success).sum()),
            # VoRunner.java distinguishes them: `restart` is a FAILURE-driven re-initialisation,
            # `recenter` a routine mosaic reference change. Both are reference changes, only the
            # first is a health signal, and conflating them would make H5 unreadable.
            "n_restart": int(sum(1 for e in events if e == "restart")),
            "n_recenter": int(sum(1 for e in events if e == "recenter")),
            "n_init": int(sum(1 for e in events if e == "init")),
            "track_count_mean": float(np.mean(col(health, "track_count"))),
            "track_count_median": float(np.median(col(health, "track_count"))),
            "track_count_min": float(np.min(col(health, "track_count"))),
            "inlier_count_mean": float(np.mean(col(health, "inlier_count"))),
            "inlier_fraction_mean": float(np.mean(
                col(health, "inlier_count") / np.maximum(col(health, "track_count"), 1))),
            "process_time_ms_median": float(np.median(col(health, "process_time_ns")) / 1e6),
        },
    }


def gt_xy(case: dict) -> np.ndarray:
    return np.stack([case["east"], case["north"]], axis=1)


def arm_heights(case: dict) -> dict:
    n = case["n_frames"]
    return {
        "fixed": np.full(n, case["h0"]),
        "baro": mro.h_agl_from_baro(case["h0"], case["baro"]),
        "oracle": case["agl"].copy(),
        "oracle_smooth": median_filter(case["agl"], ORACLE_SMOOTH_WINDOW),
    }


# --------------------------------------------------------------------------- closed form


def path_weighted_ratio(h_used: np.ndarray, h_true: np.ndarray, gt: np.ndarray) -> float:
    """`EXP-VO-012` R7's closed form: the path-weighted mean of `h_used / h_true`.

    Weights are each step's TRUE ground distance and heights are taken at the step's REFERENCE
    frame, because the increment into frame k is expressed in frame k−1's pixels (`LIT-VO-003`
    §10.1). Both choices are load-bearing: `test_exp_vo_014.py` pins that an unweighted mean gives a
    different, wrong answer.
    """
    d = np.linalg.norm(np.diff(np.asarray(gt, float), axis=0), axis=1)
    r = np.asarray(h_used, float)[:-1] / np.asarray(h_true, float)[:-1]
    return float(np.sum(d * r) / np.sum(d))


# --------------------------------------------------------------------------- scoring


def score_case(case: dict) -> dict:
    gt = gt_xy(case)
    heights = arm_heights(case)
    heading0 = float(case["heading"][0])
    out = {
        "label": case["label"], "role": case["role"], "dataset_id": case["dataset_id"],
        "run_id": case["run_id"], "n_frames": case["n_frames"], "h0_m": case["h0"],
        "h0_source": "simulator ground truth; never fitted from any trajectory",
        "f_working_px": case["f_working"], "heading0_deg": heading0,
        "heading_frame": "CAMERA image-up azimuth (NOT airframe yaw)",
        "gt_path_length_m": float(mm.path_length(gt, 1)),
        "gt_path_length_step5_m": float(mm.path_length(gt, PATH_STEP)),
        "gt_start_end_m": float(np.linalg.norm(gt[-1] - gt[0])),
        "gt_agl_range_m": [float(case["agl"].min()), float(case["agl"].max())],
        "gt_terrain_range_m": [float(case["terrain"].min()), float(case["terrain"].max())],
        "gt_baro_range_m": [float(case["baro"].min()), float(case["baro"].max())],
        "fitted_scale_parameters_in_primary_family": 0,
        "health": case["health"],
        "arms": {},
    }
    for arm in ARMS:
        h = heights[arm]
        track = mro.integrate_metric(case["inc"], h, case["f_working"], arm=arm)
        s = mm.score(track.xy(), gt, heading0, path_step=PATH_STEP, rpe_lengths=RPE_LENGTHS)
        est = track.xy()
        s["h_used_range_m"] = [float(h.min()), float(h.max())]
        s["est_path_length_m"] = float(mm.path_length(est, 1))
        s["est_path_length_step5_m"] = float(mm.path_length(est, PATH_STEP))
        s["est_start_end_m"] = float(np.linalg.norm(est[-1] - est[0]))
        s["closed_form_path_weighted_h_ratio"] = path_weighted_ratio(h, case["agl"], gt)
        out["arms"][arm] = s

    # H2 is tested as a ratio against the same run's oracle arm, so the estimator's own path-length
    # ratio — common to all arms — cancels. EXP-VO-013 H3's construction, unchanged.
    o1 = out["arms"]["oracle"]["path_length_ratio"]
    o5 = out["arms"]["oracle"]["path_length_ratio_step1"]
    for arm in ARMS:
        a = out["arms"][arm]
        pred = a["closed_form_path_weighted_h_ratio"]
        meas5 = a["path_length_ratio"] / o1 if o1 else float("nan")
        meas1 = a["path_length_ratio_step1"] / o5 if o5 else float("nan")
        a["closed_form_test"] = {
            "predicted": pred,
            "measured_ratio_vs_oracle_step5": meas5,
            "measured_ratio_vs_oracle_step1": meas1,
            "relative_error_step5_pct": 100.0 * (meas5 / pred - 1.0) if pred else float("nan"),
            "relative_error_step1_pct": 100.0 * (meas1 / pred - 1.0) if pred else float("nan"),
        }
    return out


def error_curves(case: dict) -> dict:
    gt = gt_xy(case)
    heights = arm_heights(case)
    h0 = float(case["heading"][0])
    curves = {}
    for arm in ARMS:
        tr = mro.integrate_metric(case["inc"], heights[arm], case["f_working"], arm=arm)
        d, e = mm.error_vs_distance(tr.xy(), gt, h0)
        curves[arm] = {"distance_m": d.tolist(), "error_m": e.tolist(),
                       "xy": tr.xy().tolist(), "h_used_m": heights[arm].tolist()}
    curves["_gt_xy"] = gt.tolist()
    curves["_agl_m"] = case["agl"].tolist()
    curves["_terrain_m"] = case["terrain"].tolist()
    curves["_t_s"] = case["t"].tolist()
    return curves


# --------------------------------------------------------------------------- paired analysis


def paired(a: dict, b: dict) -> dict:
    """The comparison the design exists for: does the HEIGHT CHANGE create the predicted difference,
    and does the barometer remove it? Everything here is a difference between two runs whose
    horizontal paths agree to 3.4 cm."""
    def scale_err(case, arm, key="path_length_ratio"):
        return 100.0 * (case["arms"][arm][key] - 1.0)

    out = {
        "design": "same figure-8 spline, same terrain, same tracker; only flight height differs",
        "gt_path_length_diff_m": b["gt_path_length_m"] - a["gt_path_length_m"],
        "gt_path_length_diff_pct": 100.0 * (b["gt_path_length_m"] / a["gt_path_length_m"] - 1.0),
    }
    for arm in ARMS:
        ea, eb = scale_err(a, arm), scale_err(b, arm)
        out[arm] = {
            "scale_error_pct_const": ea,
            "scale_error_pct_vary": eb,
            "added_by_height_change_pp": eb - ea,
            "ate_ref_init_const_m": a["arms"][arm]["ref_init"]["ate_rmse_m"],
            "ate_ref_init_vary_m": b["arms"][arm]["ref_init"]["ate_rmse_m"],
        }
    fixed_added = out["fixed"]["added_by_height_change_pp"]
    baro_added = out["baro"]["added_by_height_change_pp"]
    out["height_induced_error_removed_by_baro_pct"] = (
        100.0 * (1.0 - abs(baro_added) / abs(fixed_added)) if fixed_added else float("nan"))

    # The raw path-length ratio carries the estimator's OWN run-to-run difference, which is common
    # to every arm and has nothing to do with height — the oracle arm moves +0.5 pp between the two
    # runs by itself. Attributing that to the barometer would understate it. Dividing each arm by
    # its own run's oracle removes the common term, which is the same construction H2 uses, and
    # gives the height-induced change actually attributable to the height readout.
    def rel(case, arm):
        return 100.0 * (case["arms"][arm]["closed_form_test"]["measured_ratio_vs_oracle_step5"] - 1.0)

    fa, fb = rel(a, "fixed"), rel(b, "fixed")
    ba, bb = rel(a, "baro"), rel(b, "baro")
    out["oracle_relative"] = {
        "note": "each arm divided by its own run's oracle arm; removes the estimator's own "
                "run-to-run difference, which is common to all arms and is not a height effect",
        "fixed_scale_error_pct": [fa, fb],
        "baro_scale_error_pct": [ba, bb],
        "fixed_added_by_height_change_pp": fb - fa,
        "baro_added_by_height_change_pp": bb - ba,
        "height_induced_error_removed_by_baro_pct":
            100.0 * (1.0 - abs(bb - ba) / abs(fb - fa)) if (fb - fa) else float("nan"),
    }

    # RPE is alignment-free and therefore the least gameable scale-sensitive statistic available.
    out["rpe_reduction_baro_vs_fixed_vary_pct"] = {
        k: 100.0 * (1.0 - b["arms"]["baro"]["rpe"][k] / b["arms"]["fixed"]["rpe"][k])
        for k in b["arms"]["fixed"]["rpe"]}
    out["rpe_reduction_oracle_vs_fixed_vary_pct"] = {
        k: 100.0 * (1.0 - b["arms"]["oracle"]["rpe"][k] / b["arms"]["fixed"]["rpe"][k])
        for k in b["arms"]["fixed"]["rpe"]}
    # The single sentence the paired design earns: does the barometer arm on the VARYING-height run
    # match the fixed arm on the run where there is no height variation to get wrong?
    out["baro_vary_vs_fixed_const_ref_init_nate_pct"] = [
        100.0 * a["arms"]["fixed"]["ref_init"]["ate_rmse_normalised"],
        100.0 * b["arms"]["baro"]["ref_init"]["ate_rmse_normalised"]]
    out["baro_vs_oracle_gap_pct_const"] = (
        scale_err(a, "baro") - scale_err(a, "oracle"))
    out["baro_vs_oracle_gap_pct_vary"] = (
        scale_err(b, "baro") - scale_err(b, "oracle"))
    out["baro_vs_oracle_gap_consistency_pp"] = abs(
        out["baro_vs_oracle_gap_pct_vary"] - out["baro_vs_oracle_gap_pct_const"])

    # H1 — on the constant-height run the two arms must be bit-identical, not merely close.
    out["h1_fixed_equals_baro_on_const_run"] = {
        k: a["arms"]["fixed"][k] == a["arms"]["baro"][k]
        for k in ("path_length_ratio", "path_length_ratio_step1", "est_path_length_m",
                  "est_start_end_m")}
    out["h1_all_bit_identical"] = all(out["h1_fixed_equals_baro_on_const_run"].values())

    # H5 — did the imagery itself get harder, or only the height readout get wrong?
    ha, hb = a["health"], b["health"]
    out["tracking"] = {
        "frame_success_pp_diff": 100.0 * (hb["frame_success_rate"] - ha["frame_success_rate"]),
        "restart_ratio": (hb["n_restart"] / ha["n_restart"]) if ha["n_restart"] else
                         (float("inf") if hb["n_restart"] else 1.0),
        "restarts": [ha["n_restart"], hb["n_restart"]],
        "recenters": [ha["n_recenter"], hb["n_recenter"]],
        "track_count_mean": [ha["track_count_mean"], hb["track_count_mean"]],
        "inlier_fraction_mean": [ha["inlier_fraction_mean"], hb["inlier_fraction_mean"]],
        "failed_motion_estimates": [ha["n_failed"], hb["n_failed"]],
    }

    # H6 — Sim(2) fits the scale away; measure how much of the effect it hides.
    def spread(case, key, sub=None):
        vals = [case["arms"][x][key][sub] if sub else case["arms"][x][key]
                for x in ("fixed", "baro", "oracle")]
        return max(vals) - min(vals)
    out["h6_sim2_blindness"] = {
        "metric_path_ratio_spread_vary": spread(b, "path_length_ratio"),
        "sim2_nate_spread_vary": spread(b, "sim2", "ate_rmse_normalised"),
        "ref_init_nate_spread_vary": spread(b, "ref_init", "ate_rmse_normalised"),
        "sim2_fitted_scale_by_arm_vary": {
            x: b["arms"][x]["sim2"]["alignment_scale"] for x in ARMS},
    }
    return out


# --------------------------------------------------------------------------- reporting


def markdown(results: dict) -> str:
    L = []
    L.append("# EXP-VO-014 results — paired UE figure-8, three height arms\n")
    L.append("**Zero fitted scale parameters** in every number below except the explicitly "
             "labelled Sim(2) column.\n")
    for c in results["cases"]:
        L.append(f"\n## Run {c['label']} — `{c['dataset_id']}` ({c['role']})\n")
        L.append(f"- frames {c['n_frames']}, GT horizontal path {c['gt_path_length_m']:.2f} m, "
                 f"GT start→end {c['gt_start_end_m']:.2f} m")
        L.append(f"- h0 = {c['h0_m']:.6f} m ({c['h0_source']}), f_working = {c['f_working_px']:.1f} px")
        L.append(f"- true AGL {c['gt_agl_range_m'][0]:.2f}–{c['gt_agl_range_m'][1]:.2f} m; "
                 f"terrain {c['gt_terrain_range_m'][0]:.2f}–{c['gt_terrain_range_m'][1]:.2f} m; "
                 f"baro {c['gt_baro_range_m'][0]:.2f}–{c['gt_baro_range_m'][1]:.2f} m\n")
        L.append("| arm | h_used (m) | est path (m) | **path ratio** (step 5) | endpoint (m) | "
                 "endpoint (% path) | **ATE scale=1** (m) | SE(2) ATE (m) | RPE 25 m | RPE 100 m | "
                 "est ‖end−start‖ (m) | *Sim(2) nATE* |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            a = c["arms"][arm]
            L.append(
                f"| {arm} | {a['h_used_range_m'][0]:.2f}–{a['h_used_range_m'][1]:.2f} | "
                f"{a['est_path_length_m']:.1f} | **{a['path_length_ratio']:.4f}** | "
                f"{a['ref_init']['endpoint_error_m']:.2f} | "
                f"{a['ref_init']['endpoint_error_pct_of_path']:.2f} | "
                f"**{a['ref_init']['ate_rmse_m']:.2f}** | {a['se2']['ate_rmse_m']:.2f} | "
                f"{a['rpe']['rpe_25m']:.3f} | {a['rpe']['rpe_100m']:.3f} | "
                f"{a['est_start_end_m']:.2f} | *{a['sim2']['ate_rmse_normalised'] * 100:.3f} %* |")
        L.append("\n**Closed-form test (H2)** — predicted vs measured, each arm relative to this "
                 "run's own oracle arm:\n")
        L.append("| arm | predicted | measured (step 5) | rel. error | measured (step 1) | rel. error |")
        L.append("|---|---|---|---|---|---|")
        for arm in ARMS:
            t = c["arms"][arm]["closed_form_test"]
            L.append(f"| {arm} | {t['predicted']:.6f} | {t['measured_ratio_vs_oracle_step5']:.6f} | "
                     f"{t['relative_error_step5_pct']:+.2f} % | "
                     f"{t['measured_ratio_vs_oracle_step1']:.6f} | "
                     f"{t['relative_error_step1_pct']:+.2f} % |")
        h = c["health"]
        L.append(f"\n**Front-end health** — success {100 * h['frame_success_rate']:.2f} %, "
                 f"restarts (failure-driven) {h['n_restart']}, recenters {h['n_recenter']}, "
                 f"failed {h['n_failed']}, "
                 f"tracks mean {h['track_count_mean']:.0f} (min {h['track_count_min']:.0f}), "
                 f"inlier fraction {h['inlier_fraction_mean']:.4f}, "
                 f"median {h['process_time_ms_median']:.1f} ms/frame\n")

    p = results["paired"]
    L.append("\n## Paired comparison — what the height change alone did\n")
    L.append("| arm | scale error, const | scale error, varying | **added by height change** |")
    L.append("|---|---|---|---|")
    for arm in ARMS:
        d = p[arm]
        L.append(f"| {arm} | {d['scale_error_pct_const']:+.2f} % | "
                 f"{d['scale_error_pct_vary']:+.2f} % | **{d['added_by_height_change_pp']:+.2f} pp** |")
    o = p["oracle_relative"]
    L.append("\n**Oracle-relative** (the correct attribution — removes the estimator's own "
             "run-to-run difference, which is common to all arms):\n")
    L.append("| arm | const | varying | **added by height change** |")
    L.append("|---|---|---|---|")
    L.append(f"| fixed | {o['fixed_scale_error_pct'][0]:+.3f} % | "
             f"{o['fixed_scale_error_pct'][1]:+.3f} % | "
             f"**{o['fixed_added_by_height_change_pp']:+.2f} pp** |")
    L.append(f"| baro | {o['baro_scale_error_pct'][0]:+.3f} % | "
             f"{o['baro_scale_error_pct'][1]:+.3f} % | "
             f"**{o['baro_added_by_height_change_pp']:+.2f} pp** |")
    L.append(f"\n- Barometer removes **{o['height_induced_error_removed_by_baro_pct']:.2f} %** of the "
             f"height-induced scale error (oracle-relative); "
             f"**{p['height_induced_error_removed_by_baro_pct']:.1f} %** on the raw ratio.")
    L.append(f"- Metric RPE on the varying run, BARO vs FIXED: " + ", ".join(
        f"{k.replace('rpe_', '')} −{v:.1f} %" for k, v in
        p["rpe_reduction_baro_vs_fixed_vary_pct"].items()) + ".")
    nb = p["baro_vary_vs_fixed_const_ref_init_nate_pct"]
    L.append(f"- BARO on the **varying**-height run scores {nb[1]:.3f} % zero-parameter normalised "
             f"ATE, against FIXED's {nb[0]:.3f} % on the run with **no height variation at all**.")
    L.append(f"- BARO−ORACLE gap: **{p['baro_vs_oracle_gap_pct_const']:+.2f} pp** (const) vs "
             f"**{p['baro_vs_oracle_gap_pct_vary']:+.2f} pp** (varying); "
             f"consistency {p['baro_vs_oracle_gap_consistency_pp']:.3f} pp.")
    L.append(f"- H1 (FIXED ≡ BARO on the constant-height run, bit-identical): "
             f"**{p['h1_all_bit_identical']}**")
    t = p["tracking"]
    L.append(f"- H5 tracking: success Δ {t['frame_success_pp_diff']:+.2f} pp, restarts "
             f"{t['restarts'][0]} → {t['restarts'][1]}, recenters "
             f"{t['recenters'][0]} → {t['recenters'][1]}, tracks "
             f"{t['track_count_mean'][0]:.0f} → {t['track_count_mean'][1]:.0f}, inlier fraction "
             f"{t['inlier_fraction_mean'][0]:.4f} → {t['inlier_fraction_mean'][1]:.4f}")
    s = p["h6_sim2_blindness"]
    L.append(f"- H6 Sim(2) blindness: metric path-ratio spread "
             f"{s['metric_path_ratio_spread_vary']:.4f} vs Sim(2) nATE spread "
             f"{s['sim2_nate_spread_vary'] * 100:.4f} % of path")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(REPO / "evaluations" / "exp-vo-014"))
    p.add_argument("--curves", action="store_true", default=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = [load_case(*c) for c in CASES]
    scored = [score_case(c) for c in cases]
    results = {
        "experiment": "EXP-VO-014",
        "configuration": {"navigation_source": "rigid_motion", "motion_model": "homography",
                          "refineEstimate": False, "downsampleFactor": 1,
                          "path_length_step_frames": PATH_STEP, "rpe_lengths_m": list(RPE_LENGTHS),
                          "oracle_smooth_window_frames": ORACLE_SMOOTH_WINDOW},
        "fitted_scale_parameters": {"ref_init": 0, "se2": 3, "sim2": 4},
        "primary_family": "reference_initialised and SE(2); scale is NEVER fitted in either",
        "cases": scored,
        "paired": paired(scored[0], scored[1]),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "results.md").write_text(markdown(results), encoding="utf-8")
    if a.curves:
        for c in cases:
            (out / f"curves_{c['label']}.json").write_text(json.dumps(error_curves(c)) + "\n")
    print(markdown(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
