"""`EXP-VO-010` Phase 5: heading, split into the three things "yaw RMSE" has been conflating.

`EXP-VO-009` R2 established that the aligned yaw RMSE this lane has reported since `EXP-002` — 43° to
157° across records — is **≈ 95 % a constant frame offset**, and that the part which actually measures
heading *tracking* is 3.1–9.4°. The cause is a frame mismatch, not an estimator defect: the estimator's
yaw datum is the first frame, ground-truth heading there is some arbitrary compass bearing, and
`naveval.alignment.transform_yaw` corrects by the Sim(2) rotation fitted to **positions**, which is a
different quantity.

This module reports all of it, separately, so a reader can see which component a number is:

===========================  ==================================================================
``raw_yaw_rmse_deg``         exactly what ``naveval`` computes, for continuity with every prior
                             record. **Not evidence of heading quality on its own.**
``constant_offset_deg``      the circular mean of the aligned yaw error — the frame mismatch
``aligned_heading_rms_deg``  circular RMS about that offset. **This is the heading-tracking
                             number.**
``cumulative_drift_deg``     the accumulated per-frame rotation error at the end of the run
``per_frame_*``              mean, SD, robust MAD-sigma of the per-frame rotation error
===========================  ==================================================================

**`naveval` is not modified and no historical metric is rewritten.** This is VO-owned tooling that
computes an additional decomposition of the same quantity; `raw_yaw_rmse_deg` is reproduced here and
cross-checked against the evaluator's own figure.

**Circular-safe throughout.** Every mean is a circular mean (``atan2`` of summed unit vectors) and
every difference is wrapped into (−180, 180]. A naive arithmetic mean of headings straddling 0°/360°
is wrong by up to 180°, and the offsets here are large enough (43–77°) to sit anywhere on the circle.

The decomposition is validated two ways before any real number is trusted — see
``test_exp_vo_010.py``: against a second, independently written implementation (a least-squares
minimisation over the offset rather than a closed-form circular mean), and against known-answer
synthetic series whose offset and spread are constructed.
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
sys.path.insert(0, str(EVALUATION_DIR / "tools" / "exp_vo_008"))

import recompose as rc                                                   # noqa: E402
from naveval.alignment import fit_sim2                                   # noqa: E402
from naveval.dataset import load_dataset                                 # noqa: E402
from naveval.runrecord import load_run_record                            # noqa: E402
from naveval.sync import synchronize                                     # noqa: E402
from rotation_diagnostic import gt_heading_increments                    # noqa: E402


# --------------------------------------------------------------------------- circular primitives

def wrap180(deg):
    """Wrap into (-180, 180]. Array-safe."""
    w = (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0
    return np.where(w == -180.0, 180.0, w)


def circular_mean_deg(deg) -> float:
    """Circular mean: atan2 of the summed unit vectors. Undefined for a uniform spread, which is
    not a regime this metric is used in (the offsets here are tightly concentrated)."""
    r = np.radians(np.asarray(deg, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def circular_rms_about_deg(deg, centre_deg: float) -> float:
    """RMS of the wrapped deviations from `centre_deg`."""
    return float(np.sqrt((wrap180(np.asarray(deg, dtype=float) - centre_deg) ** 2).mean()))


# --------------------------------------------------------------------------- the decomposition

def decompose_heading(aligned_yaw_deg: np.ndarray, gt_heading_deg: np.ndarray) -> dict:
    """Split the aligned yaw error into its constant part and its tracking part.

    `aligned_yaw_deg` is the estimator's yaw after `naveval`'s alignment rotation has been applied —
    i.e. exactly the series the evaluator scores.
    """
    err = wrap180(np.asarray(aligned_yaw_deg) - np.asarray(gt_heading_deg))
    offset = circular_mean_deg(err)
    return {
        "n": int(err.size),
        # Reproduces naveval's own metric: RMS of the wrapped error about zero.
        "raw_yaw_rmse_deg": float(np.sqrt((err ** 2).mean())),
        "constant_offset_deg": offset,
        "aligned_heading_rms_deg": circular_rms_about_deg(err, offset),
        "aligned_heading_p95_deg": float(np.percentile(np.abs(wrap180(err - offset)), 95)),
        "aligned_heading_max_deg": float(np.abs(wrap180(err - offset)).max()),
        # How much of the raw figure is the constant. 1.0 would mean the raw metric is pure offset.
        # Guarded: a perfect run has zero raw RMSE, where the ratio is 0/0 rather than large.
        "offset_fraction_of_raw": (float(abs(offset) / np.sqrt((err ** 2).mean()))
                                   if err.size and np.sqrt((err ** 2).mean()) > 0 else 0.0),
    }


def per_frame_rotation_stats(inc_deg: np.ndarray, gt_inc_deg: np.ndarray) -> dict:
    """Per-frame rotation error against RTK heading increments, and its accumulation."""
    err = wrap180(np.asarray(inc_deg) - np.asarray(gt_inc_deg))
    n = err.size
    sd = float(err.std(ddof=1)) if n > 1 else float("nan")
    mean = float(err.mean())
    return {
        "n": int(n),
        "per_frame_mean_deg": mean,
        "per_frame_sd_deg": sd,
        "per_frame_mad_sigma_deg": float(1.4826 * np.median(np.abs(err - np.median(err)))),
        "per_frame_rms_deg": float(np.sqrt((err ** 2).mean())),
        "per_frame_t_of_mean": float(mean / (sd / np.sqrt(n))) if sd > 0 else float("nan"),
        "cumulative_drift_deg": float(err.sum()),
        "random_walk_prediction_deg": float(sd * np.sqrt(n)),
        "excess_over_random_walk": float(abs(err.sum()) / (sd * np.sqrt(n))) if sd > 0 else float("nan"),
        "p99_abs_error_deg": float(np.percentile(np.abs(err), 99)),
        "max_abs_error_deg": float(np.abs(err).max()),
    }


# --------------------------------------------------------------------------- driver

def analyse_run(dataset_rel: str, run_rel: str, width: int = 1224, height: int = 1024) -> dict:
    """Every heading quantity for one run, from the run record and its logical-transform sidecar."""
    ds = load_dataset(REPO / dataset_rel)
    run = REPO / run_rel
    rr = load_run_record(run)

    sync = synchronize(rr.timestamps_s, ds.gt_timestamps, ds.gt_east, ds.gt_north,
                       ds.gt_heading if ds.has_heading else None, ds.gt_valid,
                       clock_offset_s=ds.clock_offset_s,
                       clock_drift_s_per_s=ds.clock_drift_s_per_s)
    idx = sync.frame_indices
    est = np.stack([rr.est_x[idx], rr.est_y[idx]], axis=1)
    gt = np.stack([sync.east, sync.north], axis=1)
    al = fit_sim2(est, gt)

    out = {"run": run_rel, "dataset": dataset_rel, "alignment_rotation_deg": float(al.rotation_deg)}
    out.update(decompose_heading(al.transform_yaw(rr.est_yaw_deg[idx]), sync.heading))

    # Per-frame rotation, recomposed from the sidecar exactly as RigidNavigationState reads it.
    sidecar = run / "logical_transform.csv"
    if sidecar.exists():
        rec = rc.load_recording(run)
        cx, cy = width / 2.0, height / 2.0
        Ginv = np.linalg.inv(rec.G)
        inc = np.zeros(len(rec))
        for k in range(1, len(rec)):
            inc[k] = rc.polar_rotation(rc._jacobian(rec.G[k - 1] @ Ginv[k], cx, cy))
        dgt = gt_heading_increments(ds, rr.timestamps_s)
        # Sign convention fixed from the correlation and reported, as EXP-VO-008/009 do.
        sign = 1.0 if np.corrcoef(inc[1:], dgt[1:])[0, 1] > 0 else -1.0
        out["gt_sign_convention"] = float(sign)
        out.update(per_frame_rotation_stats(np.degrees(sign * inc[1:]), np.degrees(dgt[1:])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    r = analyse_run(a.dataset, a.run)
    r["label"] = a.label or a.run
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
