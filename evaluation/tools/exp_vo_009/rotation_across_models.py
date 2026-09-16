"""`EXP-VO-009` Phase 6: per-frame rotation error of every motion model, against RTK heading.

This is the **direct test of H1**, and it is decided before any trajectory is looked at. For each
flight window it loads the `*-rigid-v1` run of every model, recovers the per-frame rotation
increment exactly as `RigidNavigationState` does — the Jacobian of `D_k⁻¹ = G_{k-1} G_k⁻¹` at the
image centre, then its polar rotation — and scores it against ground-truth heading interpolated to
the frame timestamps.

Three things make the comparison meaningful rather than merely available:

* **The recomposition is the estimator's own arithmetic**, reused from `EXP-VO-008`'s `recompose`
  module, which was validated at both endpoints against the live runs (2.8 x 10^-11 px). Nothing is
  re-derived here.
* **For a similarity the polar rotation IS the fitted parameter** (`LIT-VO-005` section 1), so this
  reads the similarity arm's estimate directly, with no projection in between — which is the whole
  point of the experiment.
* **Ground truth is used to measure, never to choose.** No parameter is fitted anywhere in this
  script, and the sign convention between image rotation and compass heading is determined once from
  the correlation and reported, so a convention slip cannot hide.

Also reported, because `EXP-VO-008` left the bias/random-walk split unresolved: the lag-1
autocorrelation of the per-frame error, and the accumulated error as a multiple of the random-walk
prediction `sd * sqrt(n)`. A ratio near 1 is consistent with pure accumulation of independent
per-frame noise; a ratio well above 1 means a systematic component the walk cannot account for.
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
from naveval.dataset import load_dataset                                 # noqa: E402
from naveval.runrecord import load_run_record                            # noqa: E402
from rotation_diagnostic import gt_heading_increments                    # noqa: E402


def per_frame_rotation(rec: rc.Recording, width: int, height: int) -> np.ndarray:
    """Polar rotation of each frame's incremental transform, radians. Index 0 is 0 by definition."""
    n = len(rec)
    cx, cy = width / 2.0, height / 2.0
    Ginv = np.linalg.inv(rec.G)
    out = np.zeros(n)
    for k in range(1, n):
        S = rec.G[k - 1] @ Ginv[k]
        out[k] = rc.polar_rotation(rc._jacobian(S, cx, cy))
    return out


def per_frame_anisotropy(rec: rc.Recording, width: int, height: int) -> np.ndarray:
    """`sigma_1/sigma_2` of each frame's incremental Jacobian — exactly 1 for a similarity."""
    n = len(rec)
    cx, cy = width / 2.0, height / 2.0
    Ginv = np.linalg.inv(rec.G)
    out = np.ones(n)
    for k in range(1, n):
        J = rc._jacobian(rec.G[k - 1] @ Ginv[k], cx, cy)
        s = np.linalg.svd(J, compute_uv=False)
        out[k] = s[0] / s[1] if s[1] > 0 else np.inf
    return out


def summarise(inc: np.ndarray, dgt: np.ndarray) -> dict:
    """Per-frame rotation-error statistics, in degrees. `inc` and `dgt` are radians."""
    err = np.degrees(inc[1:] - dgt[1:])
    n = err.size
    sd = float(err.std(ddof=1))
    mean = float(err.mean())
    absolute = np.abs(err)
    lag1 = float(np.corrcoef(err[:-1], err[1:])[0, 1]) if n > 2 else float("nan")
    return {
        "n": int(n),
        "mean_error_deg": mean,
        "median_error_deg": float(np.median(err)),
        "sd_error_deg": sd,
        # Robust spread: 1.4826 * MAD, so a few gross frames cannot drive the headline.
        "mad_sigma_deg": float(1.4826 * np.median(np.abs(err - np.median(err)))),
        "rms_error_deg": float(np.sqrt((err ** 2).mean())),
        "t_of_mean": float(mean / (sd / np.sqrt(n))) if sd > 0 else float("nan"),
        "accumulated_error_deg": float(err.sum()),
        # A pure random walk of per-frame errors accumulates to about sd*sqrt(n). Anything beyond
        # that is a systematic component. (`mean * n` is NOT reported: it is identically the sum
        # above, so it would be a tautology dressed as a prediction.)
        "random_walk_prediction_deg": float(sd * np.sqrt(n)),
        "excess_over_random_walk": float(abs(err.sum()) / (sd * np.sqrt(n))) if sd > 0 else float("nan"),
        "lag1_autocorrelation": lag1,
        "p95_abs_error_deg": float(np.percentile(absolute, 95)),
        "p99_abs_error_deg": float(np.percentile(absolute, 99)),
        "max_abs_error_deg": float(absolute.max()),
        # Guarded: a constant GT series (only reachable in a synthetic test) has zero variance,
        # and numpy would return NaN with a divide warning rather than saying so.
        "corr_with_gt_increment": (float(np.corrcoef(inc[1:], dgt[1:])[0, 1])
                                   if dgt[1:].std() > 0 and inc[1:].std() > 0 else None),
        "total_estimated_turn_deg": float(np.degrees(inc.sum())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="repo-relative dataset dir")
    ap.add_argument("--tag", required=True, help="run-id stem, e.g. amtown01-c")
    ap.add_argument("--models", default="homography,affine,similarity")
    ap.add_argument("--width", type=int, default=1224)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = load_dataset(REPO / a.dataset)
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    out: dict = {"tag": a.tag, "dataset": a.dataset, "models": {}}
    series: dict[str, np.ndarray] = {}
    sign = None
    dgt = None

    for model in models:
        run = REPO / "runs" / f"{a.tag}-{model}-rigid-v1"
        if not (run / "logical_transform.csv").exists():
            print(f"  {model:12s} SKIPPED (no sidecar at {run})")
            continue
        rec = rc.load_recording(run)
        rr = load_run_record(run)
        inc = per_frame_rotation(rec, a.width, a.height)
        d = gt_heading_increments(ds, rr.timestamps_s)

        if sign is None:
            # Fixed once, from the first model, and applied to all — so the arms cannot silently
            # disagree about the convention.
            sign = 1.0 if np.corrcoef(inc[1:], d[1:])[0, 1] > 0 else -1.0
            dgt = d
            out["gt_sign_convention"] = float(sign)
            out["total_gt_turn_deg"] = float(np.degrees(d.sum()))
        signed = sign * inc
        stats = summarise(signed, d)

        aniso = per_frame_anisotropy(rec, a.width, a.height)
        stats["anisotropy_median"] = float(np.median(aniso[1:]))
        stats["anisotropy_p99"] = float(np.percentile(aniso[1:], 99))
        stats["track_count_median"] = float(np.median(
            [f.track_count for f in rr.frame_estimates if f.track_count is not None]))
        stats["inlier_count_median"] = float(np.median(
            [f.inlier_count for f in rr.frame_estimates if f.inlier_count is not None]))
        stats["events"] = {e: int(sum(1 for f in rr.frame_estimates if f.event == e))
                           for e in ("recenter", "restart")}
        out["models"][model] = stats
        series[model] = signed
        print(f"  {model:12s} sd {stats['sd_error_deg']:.4f} deg/frame   "
              f"mad {stats['mad_sigma_deg']:.4f}   mean {stats['mean_error_deg']:+.5f} "
              f"(t={stats['t_of_mean']:+.2f})   accum {stats['accumulated_error_deg']:+.2f} deg",
              flush=True)

    # Pairwise, against the affine baseline, since that is what H1's materiality band is set from.
    if "affine" in out["models"]:
        base = out["models"]["affine"]["sd_error_deg"]
        out["sd_relative_to_affine"] = {
            m: out["models"][m]["sd_error_deg"] / base for m in out["models"]}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    if series:
        np.savez(str(Path(a.out).with_suffix(".npz")), dgt=dgt, **series)
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
