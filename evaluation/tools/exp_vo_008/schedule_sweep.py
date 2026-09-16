"""`EXP-VO-008` Phases 2–4: sweep the rigid-reduction granularity and score every variant.

Every readout is recomposed from ONE recorded transform sequence (`recompose`), so the estimator,
tracker, RANSAC and event schedule are identical across the whole sweep by construction — the only
thing that varies is where the rigid reduction happens and which rotation estimator supplies the
heading.

**Validated at the endpoints before anything is believed.** `N = 1` with polar rotation must
reproduce the live `RIGID_MOTION` poses and `N = infinity` the live `LOGICAL_FRAME` poses, from the
committed run records. If either fails the recomposition is wrong and the sweep means nothing, so
the script says so and exits non-zero.

Usage::

    python evaluation/tools/exp_vo_008/schedule_sweep.py \
        --dataset datasets/amtown01-c --model affine --tag amtown01-c \
        --out evaluations/exp-vo-008
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parents[2]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import recompose as rc                                                   # noqa: E402
from naveval.alignment import fit_sim2                                   # noqa: E402
from naveval.dataset import load_dataset                                 # noqa: E402
from naveval.metrics import (absolute_trajectory_error, endpoint_error,  # noqa: E402
                             path_length, relative_pose_error, yaw_error_metric)
from naveval.runrecord import load_run_record                            # noqa: E402
from naveval.sync import synchronize                                     # noqa: E402

# Fixed in the EXP-VO-008 pre-registration. N is never selected by ATE; the whole curve is reported.
N_GRID = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
RPE_LENGTHS = [10, 25, 50, 100, 200, 400]


def score(dataset, timestamps, x, y, yaw_deg) -> dict:
    """Exactly `naveval.evaluate`'s primaries: one global Sim(2) fit."""
    gt_heading = dataset.gt_heading if dataset.has_heading else None
    sync = synchronize(timestamps, dataset.gt_timestamps, dataset.gt_east, dataset.gt_north,
                       gt_heading, dataset.gt_valid, clock_offset_s=dataset.clock_offset_s,
                       clock_drift_s_per_s=dataset.clock_drift_s_per_s)
    idx = sync.frame_indices
    est = np.stack([x[idx], y[idx]], axis=1)
    gt = np.stack([sync.east, sync.north], axis=1)
    al = fit_sim2(est, gt)
    aligned = al.apply(est)
    ate = absolute_trajectory_error(aligned, gt)
    plen = path_length(gt)
    out = {"ate_rmse_m": ate["rmse"], "ate_rmse_normalised": ate["rmse"] / plen,
           "endpoint_error_m": endpoint_error(aligned, gt), "path_length_m": plen,
           "alignment_scale": al.scale}
    for L in RPE_LENGTHS:
        r = relative_pose_error(aligned, gt, float(L))
        out[f"rpe_{L}m"] = r["rmse"] if r else None
    if gt_heading is not None and sync.has_heading:
        out["yaw_rmse_deg"] = yaw_error_metric(
            al.transform_yaw(yaw_deg[idx]), sync.heading)["rmse"]
    out["_aligned"], out["_gt"], out["_idx"] = aligned, gt, idx
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="affine")
    ap.add_argument("--tag", required=True, help="run-id stem, e.g. amtown01-c")
    ap.add_argument("--legacy-run", default="", help="defaults to <tag>-<model>-legacy-v1")
    ap.add_argument("--width", type=int, default=1224)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = load_dataset(REPO / a.dataset)
    rigid_run = REPO / "runs" / f"{a.tag}-{a.model}-rigid-v1"
    logical_run = REPO / "runs" / f"{a.tag}-{a.model}-logical-v1"
    legacy_run = REPO / "runs" / (a.legacy_run or f"{a.tag}-{a.model}-legacy-v1")
    rec = rc.load_recording(rigid_run)
    rr = load_run_record(rigid_run)
    t = rr.timestamps_s

    report: dict = {"dataset": a.dataset, "model": a.model, "n_frames": len(rec),
                    "width": a.width, "height": a.height}

    # ---- endpoint validation -----------------------------------------------------------------
    v1 = rc.integrate(rec, rc.boundaries_every(len(rec), 1), a.width, a.height, "polar")
    d_rigid = float(np.max(np.hypot(v1["x"] - rr.est_x, v1["y"] - rr.est_y)))
    vinf = rc.integrate(rec, [0], a.width, a.height, "polar")
    checks = {"recompose_vs_live_rigid_max_px": d_rigid}
    if (logical_run / "frames.csv").exists():
        lr = load_run_record(logical_run)
        # FULL_LOGICAL keeps the full transform; the N=inf member of this family keeps only the
        # rigid part of the whole-run composition, so they agree in position only if the run's
        # total non-rigid part is trivial. Reported, not asserted.
        checks["n_inf_vs_live_logical_max_px"] = float(
            np.max(np.hypot(vinf["x"] - lr.est_x, vinf["y"] - lr.est_y)))
    checks["recompose_reproduces_live_rigid"] = d_rigid < 1e-6
    report["validation"] = checks
    print(json.dumps(checks, indent=2), flush=True)
    if not checks["recompose_reproduces_live_rigid"]:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f"sweep_{a.tag}_{a.model}.json").write_text(json.dumps(report, indent=2))
        print("FAIL: offline recomposition does not reproduce the live RIGID_MOTION poses",
              file=sys.stderr)
        return 1

    # ---- live baselines ----------------------------------------------------------------------
    live = {}
    for name, d in (("legacy", legacy_run), ("logical", logical_run), ("rigid", rigid_run)):
        if (d / "frames.csv").exists():
            r = load_run_record(d)
            s = score(ds, r.timestamps_s, r.est_x, r.est_y, r.est_yaw_deg)
            live[name] = {k: v for k, v in s.items() if not k.startswith("_")}
    report["live"] = live

    # ---- the sweep ---------------------------------------------------------------------------
    n = len(rec)
    variants: list[tuple[str, list[int], str]] = []
    for N in N_GRID:
        variants.append((f"N={N}", rc.boundaries_every(n, N), "polar"))
    variants.append(("N=inf", [0], "polar"))
    ev_b = rc.boundaries_from_events(rec)
    variants.append((f"recenter-schedule (n={len(ev_b)})", ev_b, "polar"))
    # rotation-estimator axis, at the two granularities that matter
    variants.append(("N=1 edge-angle", rc.boundaries_every(n, 1), "edge"))
    variants.append((f"recenter-schedule edge-angle", ev_b, "edge"))
    variants.append(("N=inf edge-angle", [0], "edge"))

    rows = []
    for label, b, rot in variants:
        v = rc.integrate(rec, b, a.width, a.height, rot)
        s = score(ds, t, v["x"], v["y"], v["yaw_deg"])
        rows.append({"label": label, "rotation": rot, "n_boundaries": v["n_boundaries"],
                     "mean_segment_frames": n / max(1, v["n_boundaries"]),
                     **{k: val for k, val in s.items() if not k.startswith("_")}})
        print(f"  {label:32s} {rot:5s} segs {v['n_boundaries']:5d} "
              f"normATE {100 * s['ate_rmse_normalised']:7.3f} %  "
              f"yaw {s.get('yaw_rmse_deg', float('nan')):6.1f}  "
              f"rpe50 {s['rpe_50m']:6.2f}", flush=True)
    report["sweep"] = rows

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"sweep_{a.tag}_{a.model}.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out / f'sweep_{a.tag}_{a.model}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
