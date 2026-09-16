"""EXP-VO-006: score the pre-registered hypotheses for `NavigationSource.RIGID_MOTION`
(`DEC-VO-004` Alternative E) on HKairport01, against the committed legacy and full-logical
baselines and against `EXP-VO-004` R6's offline prediction.

`naveval` is imported read-only (shared file); the VO is not re-run here. Scale statistics come from
the run's own per-frame diagnostics (`logical_transform.csv`), never from `scale_series.csv` —
`EXP-VO-003` established that the per-segment fitted scale is valid only for `l/sigma >~ 20` and is
unbiased-but-noisy under correlated error.

Usage::

    python evaluation/tools/exp_vo_006/rigid_readout.py --out evaluations/exp-vo-006
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parents[2]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))

from naveval.alignment import fit_sim2                                  # noqa: E402
from naveval.dataset import load_dataset                                # noqa: E402
from naveval.metrics import (absolute_trajectory_error, endpoint_error,  # noqa: E402
                             path_length, relative_pose_error, runtime_statistics,
                             translational_drift_rate, yaw_error_metric)
from naveval.runrecord import load_run_record                           # noqa: E402
from naveval.sync import synchronize                                    # noqa: E402

CASES = [("a", "homography"), ("a", "affine"), ("b", "homography"), ("b", "affine")]
DATASETS = {"a": "datasets/hkairport01-a", "b": "datasets/hkairport01-b"}
RPE_LENGTHS = [10, 25, 50, 100, 200, 400]
PHYSICAL_BAND = (0.90, 1.10)

# Run records per (execution, model, readout).
RUNS = {
    ("a", "homography", "legacy"): "runs/hkairport01-a-run-v1",
    ("b", "homography", "legacy"): "runs/hkairport01-b-run-v1",
    ("a", "affine", "legacy"): "runs/hkairport01-a-affine-v1",
    ("b", "affine", "legacy"): "runs/hkairport01-b-affine-v1",
    ("a", "homography", "full_logical"): "runs/hkairport01-a-homography-logical-v1",
    ("a", "affine", "full_logical"): "runs/hkairport01-a-affine-logical-v1",
    ("b", "homography", "full_logical"): "runs/hkairport01-b-homography-logical-v1",
    ("b", "affine", "full_logical"): "runs/hkairport01-b-affine-logical-v1",
    ("a", "homography", "rigid"): "runs/hkairport01-a-homography-rigid-v1",
    ("a", "affine", "rigid"): "runs/hkairport01-a-affine-rigid-v1",
    ("b", "homography", "rigid"): "runs/hkairport01-b-homography-rigid-v1",
    ("b", "affine", "rigid"): "runs/hkairport01-b-affine-rigid-v1",
}
BORDER60 = {
    ("a", "homography"): "runs/hkairport01-a-homography-rigid-border60-v1",
    ("a", "affine"): "runs/hkairport01-a-affine-rigid-border60-v1",
}
# EXP-VO-004 R6, offline SE(2)-every-frame re-composition: ATE m, normalised %.
R6_PREDICTION = {
    ("a", "homography"): (8.22, 1.52), ("a", "affine"): (8.98, 1.66),
    ("b", "homography"): (21.21, 1.15), ("b", "affine"): (18.26, 0.99),
}


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def evaluate(run_dir: Path, dataset) -> dict:
    """Primary metrics exactly as naveval.evaluate computes them: one global Sim(2) fit."""
    rr = load_run_record(run_dir)
    gt_heading = dataset.gt_heading if dataset.has_heading else None
    sync = synchronize(rr.timestamps_s, dataset.gt_timestamps, dataset.gt_east, dataset.gt_north,
                       gt_heading, dataset.gt_valid, clock_offset_s=dataset.clock_offset_s,
                       clock_drift_s_per_s=dataset.clock_drift_s_per_s)
    idx = sync.frame_indices
    est = np.stack([rr.est_x[idx], rr.est_y[idx]], axis=1)
    gt = np.stack([sync.east, sync.north], axis=1)
    al = fit_sim2(est, gt)
    aligned = al.apply(est)
    ate = absolute_trajectory_error(aligned, gt)
    plen = path_length(gt)
    out = {
        "ate_rmse_m": ate["rmse"],
        "ate_rmse_normalised": ate["rmse"] / plen,
        "endpoint_error_m": endpoint_error(aligned, gt),
        "drift_rate": translational_drift_rate(aligned, gt)["drift_rate"],
        "path_length_m": plen,
        "alignment_scale": al.scale,
        "alignment_rotation_deg": al.rotation_deg,
        "n_points": int(idx.size),
    }
    for L in RPE_LENGTHS:
        r = relative_pose_error(aligned, gt, float(L))
        out[f"rpe_{L}m"] = r["rmse"] if r else None
    if gt_heading is not None and sync.has_heading:
        out["yaw_rmse_deg"] = yaw_error_metric(al.transform_yaw(rr.est_yaw_deg[idx]), sync.heading)["rmse"]
    ev = [fe.event for fe in rr.frame_estimates]
    out["n_recenters"] = sum(1 for e in ev if e == "recenter")
    out["n_restarts"] = sum(1 for e in ev if e == "restart")
    out["success_rate"] = float(np.mean([fe.success for fe in rr.frame_estimates]))
    tracks = np.array([fe.track_count for fe in rr.frame_estimates if fe.track_count is not None])
    out["track_count_median"] = float(np.median(tracks)) if tracks.size else None
    out["runtime"] = runtime_statistics(
        np.array([fe.process_time_ns for fe in rr.frame_estimates if fe.process_time_ns is not None]))
    # Pose discontinuities: per-frame aligned step above 5x the median (a run-local definition,
    # reported for comparison across readouts of the SAME run, not as an absolute count).
    steps = np.linalg.norm(np.diff(aligned, axis=0), axis=1)
    med = float(np.median(steps))
    out["discontinuities_5x_median_step"] = int((steps > 5 * med).sum())
    out["_aligned"] = aligned
    out["_gt"] = gt
    out["_idx"] = idx
    return out


def diagnostics(run_dir: Path) -> dict:
    """Per-frame geometric state from the run's own sidecar (never from scale_series.csv)."""
    p = run_dir / "logical_transform.csv"
    if not p.exists():
        return {}
    rows = read_csv(p)
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    acc = col("rigid_accum_scale")
    dlog = col("inc_log_scale")[1:]
    aniso = col("inc_anisotropy")[1:]
    persp = col("inc_perspective")[1:]
    t = dlog.mean() / (dlog.std(ddof=1) / math.sqrt(dlog.size))
    outside = np.where((acc < PHYSICAL_BAND[0]) | (acc > PHYSICAL_BAND[1]))[0]
    return {
        "accumulated_scale_end": float(acc[-1]),
        "accumulated_scale_min": float(acc.min()),
        "accumulated_scale_max": float(acc.max()),
        "leaves_physical_band": bool(outside.size > 0),
        "first_exit_frame": int(outside[0]) if outside.size else None,
        "mean_dlog_scale": float(dlog.mean()),
        "sd_dlog_scale": float(dlog.std(ddof=1)),
        "t_dlog_scale": float(t),
        "anisotropy_p50": float(np.median(aniso)),
        "anisotropy_p95": float(np.percentile(aniso, 95)),
        "anisotropy_max": float(aniso.max()),
        "perspective_p50": float(np.median(persp)),
        "perspective_p95": float(np.percentile(persp, 95)),
        "perspective_max": float(persp.max()),
        "improper_increments": int(float(rows[-1]["improper_count"])),
        "_aniso": aniso, "_persp": persp,
    }


def h5_test(full_eval: dict, rigid_diag: dict) -> dict:
    """H5: are anisotropy/perspective higher where FULL_LOGICAL's error grows fastest?"""
    aligned, gt, idx = full_eval["_aligned"], full_eval["_gt"], full_eval["_idx"]
    err = np.linalg.norm(aligned - gt, axis=1)
    derr = np.abs(np.diff(err))
    aniso, persp = rigid_diag["_aniso"], rigid_diag["_persp"]
    n = min(derr.size, aniso.size)
    derr, aniso, persp = derr[:n], aniso[:n], persp[:n]
    hot = derr >= np.percentile(derr, 90)
    return {
        "n_frames": int(n),
        "anisotropy_mean_unstable": float(aniso[hot].mean()),
        "anisotropy_mean_rest": float(aniso[~hot].mean()),
        "perspective_mean_unstable": float(persp[hot].mean()),
        "perspective_mean_rest": float(persp[~hot].mean()),
        "anisotropy_higher_when_unstable": bool(aniso[hot].mean() > aniso[~hot].mean()),
        "perspective_higher_when_unstable": bool(persp[hot].mean() > persp[~hot].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    datasets = {k: load_dataset(REPO / v) for k, v in DATASETS.items()}
    results: dict[str, dict] = {}
    for (exe, model, readout), rel in RUNS.items():
        d = REPO / rel
        if not (d / "frames.csv").exists():
            print("MISSING", rel)
            continue
        key = f"{exe}-{model}-{readout}"
        results[key] = evaluate(d, datasets[exe])
        if readout == "rigid":
            results[key]["diagnostics"] = diagnostics(d)

    # H1: rigid vs full logical
    h1 = {}
    for exe, model in CASES:
        f = results.get(f"{exe}-{model}-full_logical")
        r = results.get(f"{exe}-{model}-rigid")
        if f and r:
            rel_impr = 1 - r["ate_rmse_normalised"] / f["ate_rmse_normalised"]
            h1[f"{exe}-{model}"] = {
                "full_norm_ate": f["ate_rmse_normalised"], "rigid_norm_ate": r["ate_rmse_normalised"],
                "relative_improvement": rel_impr, "improved": rel_impr > 0,
                "improved_30pc": rel_impr > 0.30,
            }
    # vs legacy, and vs the EXP-VO-004 R6 prediction
    versus = {}
    for exe, model in CASES:
        leg = results.get(f"{exe}-{model}-legacy")
        r = results.get(f"{exe}-{model}-rigid")
        if leg and r:
            pred_ate, pred_norm = R6_PREDICTION[(exe, model)]
            versus[f"{exe}-{model}"] = {
                "legacy_ate": leg["ate_rmse_m"], "legacy_norm": leg["ate_rmse_normalised"],
                "rigid_ate": r["ate_rmse_m"], "rigid_norm": r["ate_rmse_normalised"],
                "vs_legacy_relative": 1 - r["ate_rmse_normalised"] / leg["ate_rmse_normalised"],
                "r6_offline_ate": pred_ate, "r6_offline_norm": pred_norm / 100.0,
                "gap_vs_r6_relative": r["ate_rmse_m"] / pred_ate - 1.0,
                "within_25pc_of_r6": abs(r["ate_rmse_m"] / pred_ate - 1.0) <= 0.25,
            }
    # H2: canvas border threshold
    h2 = {}
    for (exe, model), rel in BORDER60.items():
        d = REPO / rel
        base = results.get(f"{exe}-{model}-rigid")
        if not (d / "frames.csv").exists() or not base:
            continue
        alt = evaluate(d, datasets[exe])
        rr_base = load_run_record(REPO / RUNS[(exe, model, "rigid")])
        rr_alt = load_run_record(d)
        n = min(len(rr_base.frame_estimates), len(rr_alt.frame_estimates))
        dx = np.array([rr_alt.est_x[i] - rr_base.est_x[i] for i in range(n)])
        dy = np.array([rr_alt.est_y[i] - rr_base.est_y[i] for i in range(n)])
        disp = np.hypot(dx, dy)
        travelled = float(np.sum(np.hypot(np.diff(rr_base.est_x[:n]), np.diff(rr_base.est_y[:n]))))
        h2[f"{exe}-{model}"] = {
            "recenters_10": base["n_recenters"], "recenters_60": alt["n_recenters"],
            "endpoint_diff_px": float(disp[-1]), "max_diff_px": float(disp.max()),
            "estimator_path_px": travelled,
            "endpoint_diff_frac_path": float(disp[-1] / travelled) if travelled else None,
            "ate_10": base["ate_rmse_m"], "ate_60": alt["ate_rmse_m"],
            "norm_ate_10": base["ate_rmse_normalised"], "norm_ate_60": alt["ate_rmse_normalised"],
        }
    # H3: model gap
    h3 = {}
    for exe in ("a", "b"):
        for readout in ("full_logical", "rigid", "legacy"):
            h = results.get(f"{exe}-homography-{readout}")
            af = results.get(f"{exe}-affine-{readout}")
            if h and af:
                h3[f"{exe}-{readout}"] = abs(h["ate_rmse_normalised"] - af["ate_rmse_normalised"])
    h3["narrowed_a"] = h3.get("a-rigid", 9e9) < h3.get("a-full_logical", 0)
    h3["narrowed_b"] = h3.get("b-rigid", 9e9) < h3.get("b-full_logical", 0)
    # H4: scale still drifts
    h4 = {k: {kk: vv for kk, vv in v["diagnostics"].items() if not kk.startswith("_")}
          for k, v in results.items() if k.endswith("-rigid") and "diagnostics" in v}
    # H5
    h5 = {}
    for exe, model in CASES:
        f = results.get(f"{exe}-{model}-full_logical")
        r = results.get(f"{exe}-{model}-rigid")
        if f and r and "diagnostics" in r and r["diagnostics"]:
            h5[f"{exe}-{model}"] = h5_test(f, r["diagnostics"])

    clean = {}
    for k, v in results.items():
        clean[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        if "diagnostics" in clean[k]:
            clean[k]["diagnostics"] = {kk: vv for kk, vv in clean[k]["diagnostics"].items()
                                       if not kk.startswith("_")}
    summary = {"metrics": clean, "H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5,
               "versus_baselines": versus}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n=== normalised ATE (%) by readout ===")
    print(f"{'case':16s} {'legacy':>9s} {'full':>9s} {'rigid':>9s}   {'R6 offline':>10s}")
    for exe, model in CASES:
        row = [results.get(f"{exe}-{model}-{r}") for r in ("legacy", "full_logical", "rigid")]
        vals = [f"{100*x['ate_rmse_normalised']:9.3f}" if x else "        -" for x in row]
        print(f"{exe+'-'+model:16s} {' '.join(vals)}   {R6_PREDICTION[(exe,model)][1]:10.2f}")
    print("\n=== yaw RMSE (deg) ===")
    for exe, model in CASES:
        row = [results.get(f"{exe}-{model}-{r}") for r in ("legacy", "full_logical", "rigid")]
        vals = [f"{x.get('yaw_rmse_deg', float('nan')):9.2f}" if x else "        -" for x in row]
        print(f"{exe+'-'+model:16s} {' '.join(vals)}")
    print("\nH1", json.dumps(h1, indent=2, default=float))
    print("H2", json.dumps(h2, indent=2, default=float))
    print("H3", json.dumps(h3, indent=2, default=float))
    print("H4", json.dumps(h4, indent=2, default=float))
    print("H5", json.dumps(h5, indent=2, default=float))
    print("VS", json.dumps(versus, indent=2, default=float))


if __name__ == "__main__":
    main()
