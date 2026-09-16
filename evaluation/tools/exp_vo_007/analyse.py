"""EXP-VO-007: score the pre-registered hypotheses for `RIGID_MOTION` on the independent
`AMtown01` sequence, against legacy and full-logical arms computed on the *same* sequence.

Deliberately parallel to `evaluation/tools/exp_vo_006/rigid_readout.py`, so the two records'
numbers are produced by the same metric code paths and are directly comparable. Differences, all
forced by this being an out-of-sample test rather than a development one:

- **Baselines are computed here, not carried over.** `EXP-VO-006` compared against committed
  `HKairport01` runs; this record runs legacy and full-logical arms on `AMtown01` itself, because
  "better than legacy" has to mean better on *this* flight.
- **H2 has a pre-declared materiality band** (3x the `hkairport01-b` rigid value) with a second
  threshold at the degradation the one external method benchmarked on both sequences shows (9.7x).
- **H4 is a partial correlation**, controlling for distance travelled: cumulative |log scale| and
  cumulative position error both grow with path length, so a raw correlation would be near 1 for
  reasons that say nothing about coupling.
- **H5's physical band comes from the sequence's own measured effective height**, not from its
  commanded altitude (`LIT-VO-004` section 3: p5-p95 height ratio 1.66 -> band [0.60, 1.66]).
- **Results are split by effective-height regime**, from the LiDAR profile frozen in `DEC-VO-005`.

`naveval` is imported read-only. Scale statistics come from each run's own per-frame diagnostic
sidecar, never from `scale_series.csv` (`EXP-VO-003`: the per-segment fitted scale is valid only for
`l/sigma >~ 20` and is unbiased-but-noisy under correlated error).

Usage::

    python evaluation/tools/exp_vo_007/analyse.py --out evaluations/exp-vo-007 \
        [--height-profile <candidate_probe.json>]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from height_profile import load_height_profile                           # noqa: E402
from naveval.alignment import fit_sim2                                   # noqa: E402
from naveval.dataset import load_dataset                                 # noqa: E402
from naveval.metrics import (absolute_trajectory_error, endpoint_error,  # noqa: E402
                             path_length, relative_pose_error, runtime_statistics,
                             translational_drift_rate, yaw_error_metric)
from naveval.runrecord import load_run_record                            # noqa: E402
from naveval.sync import synchronize                                     # noqa: E402

DATASET = "datasets/amtown01-c"
MODELS = ("homography", "affine")
READOUTS = ("legacy", "logical", "rigid")
RPE_LENGTHS = [10, 25, 50, 100, 200, 400]
PHYSICAL_BAND = (0.90, 1.10)

def run_dir(model: str, readout: str) -> str:
    return f"runs/amtown01-c-{model}-{readout}-v1"

# EXP-VO-006 R1, hkairport01-b: normalised ATE as a fraction.
HKB_RIGID = {"homography": 0.01148, "affine": 0.00988}
HKB_LEGACY = {"homography": 0.02649, "affine": 0.01099}
# DEC-VO-005 / EXP-VO-007 H2 materiality band, fixed before execution.
GENERALISATION_FACTOR = 3.0
EXTERNAL_DEGRADATION_FACTOR = 9.7      # ORB-SLAM3, 0.189 % -> 1.830 % (LIT-006 Table 5)
# EXP-VO-007 H5: physically explicable accumulated-scale band, from the MEASURED effective-height
# p5-p95 ratio of this sequence (LIT-VO-004 section 3), not from its commanded altitude.
MEASURED_HEIGHT_RATIO = 1.66


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def evaluate(rd: Path, dataset) -> dict:
    """Primary metrics exactly as `naveval.evaluate` computes them: one global Sim(2) fit."""
    rr = load_run_record(rd)
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
        out["yaw_rmse_deg"] = yaw_error_metric(
            al.transform_yaw(rr.est_yaw_deg[idx]), sync.heading)["rmse"]
    ev = [fe.event for fe in rr.frame_estimates]
    out["n_recenters"] = sum(1 for e in ev if e == "recenter")
    out["n_restarts"] = sum(1 for e in ev if e == "restart")
    out["success_rate"] = float(np.mean([fe.success for fe in rr.frame_estimates]))
    tracks = np.array([fe.track_count for fe in rr.frame_estimates if fe.track_count is not None])
    out["track_count_median"] = float(np.median(tracks)) if tracks.size else None
    out["runtime"] = runtime_statistics(np.array(
        [fe.process_time_ns for fe in rr.frame_estimates if fe.process_time_ns is not None]))
    steps = np.linalg.norm(np.diff(aligned, axis=0), axis=1)
    med = float(np.median(steps))
    out["discontinuities_5x_median_step"] = int((steps > 5 * med).sum())
    out["_aligned"], out["_gt"], out["_idx"] = aligned, gt, idx
    out["_frame_t"] = rr.timestamps_s[idx]
    return out


def diagnostics(rd: Path) -> dict:
    """Per-frame geometric state from the run's own sidecar (never from `scale_series.csv`)."""
    p = rd / "logical_transform.csv"
    if not p.exists():
        return {}
    rows = read_csv(p)
    col = lambda k: np.array([float(r[k]) for r in rows])                # noqa: E731
    acc = col("rigid_accum_scale")
    dlog, aniso, persp = col("inc_log_scale")[1:], col("inc_anisotropy")[1:], col("inc_perspective")[1:]
    t = dlog.mean() / (dlog.std(ddof=1) / math.sqrt(dlog.size))
    outside = np.where((acc < PHYSICAL_BAND[0]) | (acc > PHYSICAL_BAND[1]))[0]
    lo, hi = 1.0 / MEASURED_HEIGHT_RATIO, MEASURED_HEIGHT_RATIO
    out_meas = np.where((acc < lo) | (acc > hi))[0]
    return {
        "accumulated_scale_end": float(acc[-1]),
        "accumulated_scale_min": float(acc.min()),
        "accumulated_scale_max": float(acc.max()),
        "leaves_physical_band": bool(outside.size > 0),
        "first_exit_frame": int(outside[0]) if outside.size else None,
        "measured_height_band": [lo, hi],
        "leaves_measured_height_band": bool(out_meas.size > 0),
        "first_exit_measured_band_frame": int(out_meas[0]) if out_meas.size else None,
        "mean_dlog_scale": float(dlog.mean()),
        "sd_dlog_scale": float(dlog.std(ddof=1)),
        "t_dlog_scale": float(t),
        "anisotropy_p50": float(np.median(aniso)),
        "anisotropy_p95": float(np.percentile(aniso, 95)),
        "anisotropy_p99": float(np.percentile(aniso, 99)),
        "anisotropy_max": float(aniso.max()),
        "perspective_p50": float(np.median(persp)),
        "perspective_p95": float(np.percentile(persp, 95)),
        "perspective_p99": float(np.percentile(persp, 99)),
        "perspective_max": float(persp.max()),
        "improper_increments": int(float(rows[-1]["improper_count"])),
        "_acc": acc, "_aniso": aniso, "_persp": persp,
    }


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    return r


def partial_spearman(a: np.ndarray, b: np.ndarray, ctrl: np.ndarray) -> float:
    """Spearman correlation of `a` and `b` with `ctrl` partialled out (H4).

    Written here rather than pulled in from scipy: `research.md` R4 keeps this project's Python
    stack at numpy + matplotlib, and a partial rank correlation is three linear fits.
    """
    ra, rb, rc = _rank(a), _rank(b), _rank(ctrl)
    def resid(y):
        A = np.stack([rc, np.ones_like(rc)], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ coef
    ea, eb = resid(ra), resid(rb)
    denom = np.linalg.norm(ea) * np.linalg.norm(eb)
    return float(ea @ eb / denom) if denom else float("nan")


def h4_coupling(ev: dict, diag: dict) -> dict:
    """Is the rigid trajectory's error explained by accumulated scale, beyond shared growth
    with distance travelled?"""
    aligned, gt = ev["_aligned"], ev["_gt"]
    err = np.linalg.norm(aligned - gt, axis=1)
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
    acc = diag["_acc"]
    idx = ev["_idx"]
    acc_at = acc[np.clip(idx, 0, acc.size - 1)]
    logscale = np.abs(np.log(np.clip(acc_at, 1e-12, None)))
    n = min(err.size, logscale.size, dist.size)
    raw = float(np.corrcoef(_rank(logscale[:n]), _rank(err[:n]))[0, 1])
    return {
        "n": int(n),
        "spearman_raw": raw,
        "spearman_partial_controlling_distance":
            partial_spearman(logscale[:n], err[:n], dist[:n]),
    }


def height_regime_split(ev: dict, profile: dict | None) -> dict:
    """Split the rigid trajectory's error by effective-height regime.

    The regime boundaries are the LiDAR height profile's terciles, computed from dataset metadata
    and frozen in `DEC-VO-005` -- never from trajectory quality.
    """
    if not profile:
        return {"available": False,
                "reason": "no LiDAR height profile supplied (--height-profile)"}
    t = np.array(profile["t"], dtype=float)
    h = np.array(profile["height_m"], dtype=float)
    aligned, gt, ft = ev["_aligned"], ev["_gt"], ev["_frame_t"]
    hf = np.interp(ft, t, h)
    err = np.linalg.norm(aligned - gt, axis=1)
    lo, hi = float(np.percentile(h, 33.3)), float(np.percentile(h, 66.7))
    bands = {"low": hf < lo, "mid": (hf >= lo) & (hf < hi), "high": hf >= hi}
    out = {"available": True, "tercile_bounds_m": [round(lo, 1), round(hi, 1)], "bands": {}}
    for name, m in bands.items():
        if m.sum() < 10:
            continue
        out["bands"][name] = {
            "n_frames": int(m.sum()),
            "height_mean_m": float(hf[m].mean()),
            "error_rmse_m": float(np.sqrt(np.mean(err[m] ** 2))),
            "error_median_m": float(np.median(err[m])),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--height-profile", default="",
                    help="probe_candidates.py report, or a bare {t, height_m} object")
    ap.add_argument("--scene", default="AMtown01",
                    help="which sequence to read from a multi-sequence probe report")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(REPO / a.dataset)
    profile = load_height_profile(a.height_profile, a.scene) if a.height_profile else None

    results: dict[str, dict] = {}
    for model in MODELS:
        for readout in READOUTS:
            d = REPO / run_dir(model, readout)
            if not (d / "frames.csv").exists():
                print("MISSING", run_dir(model, readout))
                continue
            key = f"{model}-{readout}"
            results[key] = evaluate(d, ds)
            dg = diagnostics(d)
            if dg:
                results[key]["diagnostics"] = dg

    # H1: rigid vs full logical, both models
    h1 = {}
    for model in MODELS:
        f, r = results.get(f"{model}-logical"), results.get(f"{model}-rigid")
        if f and r:
            impr = 1 - r["ate_rmse_normalised"] / f["ate_rmse_normalised"]
            h1[model] = {"logical_norm_ate": f["ate_rmse_normalised"],
                         "rigid_norm_ate": r["ate_rmse_normalised"],
                         "relative_improvement": impr, "improved": impr > 0}
    h1["supported"] = all(v.get("improved") for k, v in h1.items() if isinstance(v, dict))

    # H2: generalisation band, fixed before execution
    h2 = {}
    for model in MODELS:
        r, leg = results.get(f"{model}-rigid"), results.get(f"{model}-legacy")
        if not r:
            continue
        ratio = r["ate_rmse_normalised"] / HKB_RIGID[model]
        h2[model] = {
            "rigid_norm_ate": r["ate_rmse_normalised"],
            "hkairport01_b_rigid_norm_ate": HKB_RIGID[model],
            "ratio_to_development_set": ratio,
            "within_3x_band": ratio <= GENERALISATION_FACTOR,
            "within_external_degradation_envelope": ratio <= EXTERNAL_DEGRADATION_FACTOR,
            "legacy_norm_ate": leg["ate_rmse_normalised"] if leg else None,
            "beats_legacy": (r["ate_rmse_normalised"] <= leg["ate_rmse_normalised"]) if leg else None,
        }
    h2["supported"] = any(v.get("within_3x_band") for k, v in h2.items() if isinstance(v, dict))

    # H3: model convergence
    h3 = {}
    for readout in READOUTS:
        hh, af = results.get(f"homography-{readout}"), results.get(f"affine-{readout}")
        if hh and af:
            h3[readout] = abs(hh["ate_rmse_normalised"] - af["ate_rmse_normalised"])
    h3["narrowed_vs_logical"] = h3.get("rigid", 9e9) < h3.get("logical", 0.0)

    # H4 / H5 / H6
    h4, h5, h6 = {}, {}, {}
    for model in MODELS:
        r = results.get(f"{model}-rigid")
        if not r or "diagnostics" not in r:
            continue
        d = r["diagnostics"]
        h4[model] = h4_coupling(r, d)
        h4[model]["isolated"] = (
            abs(h4[model]["spearman_partial_controlling_distance"]) <= 0.5)
        h5[model] = {k: v for k, v in d.items()
                     if k in ("accumulated_scale_end", "accumulated_scale_min",
                              "accumulated_scale_max", "leaves_physical_band",
                              "measured_height_band", "leaves_measured_height_band",
                              "first_exit_measured_band_frame", "mean_dlog_scale",
                              "t_dlog_scale", "improper_increments")}
        h6[model] = {k: v for k, v in d.items() if k.startswith(("anisotropy", "perspective"))}
        h6[model]["height_regime_split"] = height_regime_split(r, profile)

    clean = {}
    for k, v in results.items():
        clean[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        if "diagnostics" in clean[k]:
            clean[k]["diagnostics"] = {kk: vv for kk, vv in clean[k]["diagnostics"].items()
                                       if not kk.startswith("_")}
    summary = {"dataset": a.dataset, "metrics": clean,
               "H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n=== normalised ATE (%) on AMtown01, by readout ===")
    print(f"{'model':14s} {'legacy':>9s} {'logical':>9s} {'rigid':>9s}   "
          f"{'hkb rigid':>10s} {'ratio':>7s}")
    for model in MODELS:
        row = [results.get(f"{model}-{r}") for r in READOUTS]
        vals = [f"{100 * x['ate_rmse_normalised']:9.3f}" if x else "        -" for x in row]
        ratio = h2.get(model, {}).get("ratio_to_development_set")
        print(f"{model:14s} {' '.join(vals)}   {100 * HKB_RIGID[model]:10.3f} "
              f"{ratio:7.2f}" if ratio else f"{model:14s} {' '.join(vals)}")
    for name, obj in (("H1", h1), ("H2", h2), ("H3", h3),
                      ("H4", h4), ("H5", h5), ("H6", h6)):
        print(f"\n{name}", json.dumps(obj, indent=2, default=float))


if __name__ == "__main__":
    main()
