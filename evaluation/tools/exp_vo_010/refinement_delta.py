"""`EXP-VO-010` Phase 4: how far refinement actually moves the model, per frame, on real imagery.

The experiment must establish not only that the trajectory changed but **how much the model
parameters changed** — otherwise a null trajectory result is ambiguous between "refinement barely
moves the model" and "it moves the model but the trajectory does not care". Those are outcome D and
outcome C respectively, and they have different consequences.

Reads the `refinement.csv` sidecar (`VoRunner`, `EXP-VO-010`), which records for every accepted
frame RANSAC's winning **minimal-sample** model, the model actually shipped, and the size of the
inlier set the refiner was given. On a `refineEstimate = false` run the two are identical by
construction, so such a run is a free correctness check rather than a measurement.

Reports the distributions of

    d_rot_deg    = rotation(shipped) - rotation(minimal)        [wrapped]
    d_trans_px   = |centre(shipped) - centre(minimal)|
    d_log_scale  = log scale(shipped) - log scale(minimal)

and how each relates to the inlier count, since `LIT-VO-005` eq. (8) predicts the benefit of fitting
more points depends on how many there are and how they are spread.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def load(run_rel: str) -> dict[str, np.ndarray]:
    p = REPO / run_rel / "refinement.csv"
    if not p.exists():
        raise FileNotFoundError(f"no refinement sidecar at {p}; the run needs "
                                f'"refinement_sidecar": true')
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) for r in rows])                # noqa: E731
    return {
        "frame": np.array([int(r["frame_index"]) for r in rows]),
        "event": np.array([r["event"] for r in rows]),
        "refine_enabled": np.array([int(r["refine_enabled"]) for r in rows]),
        "inliers": np.array([int(r["inliers"]) for r in rows]),
        "d_rot": col("d_rot_deg"),
        "d_trans": col("d_trans_px"),
        "d_log_scale": col("d_log_scale"),
        "min_rot": col("min_rot_deg"),
        "shp_rot": col("shp_rot_deg"),
        "min_aniso": col("min_aniso"),
        "shp_aniso": col("shp_aniso"),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, ties broken by order — enough for a monotone-association statement."""
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        return r
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def summarise(d: dict, label: str) -> dict:
    n = d["frame"].size
    enabled = bool(d["refine_enabled"].max() > 0)
    inl = d["inliers"].astype(float)
    out = {
        "label": label,
        "frames": int(n),
        "refine_enabled": enabled,
        "inliers_median": float(np.median(inl)),
        "inliers_p05": float(np.percentile(inl, 5)),
        "inliers_min": int(inl.min()),
        "inliers_max": int(inl.max()),
    }
    for key, name in (("d_rot", "d_rot_deg"), ("d_trans", "d_trans_px"),
                      ("d_log_scale", "d_log_scale")):
        v = d[key]
        av = np.abs(v)
        out[name] = {
            "mean": float(v.mean()),
            "median_abs": float(np.median(av)),
            "sd": float(v.std(ddof=1)) if n > 1 else float("nan"),
            "rms": float(np.sqrt((v ** 2).mean())),
            "p95_abs": float(np.percentile(av, 95)),
            "max_abs": float(av.max()),
            # Does refinement move the model more when it has fewer points to work with?
            "spearman_with_inliers": spearman(inl, av),
        }
    # A magnitude comparison the record needs: is the per-frame change comparable to the per-frame
    # rotation itself, or negligible beside it?
    out["d_rot_deg"]["rms_over_rms_of_minimal_rotation"] = float(
        np.sqrt((d["d_rot"] ** 2).mean()) / np.sqrt((d["min_rot"] ** 2).mean())) \
        if np.sqrt((d["min_rot"] ** 2).mean()) > 0 else float("nan")
    # Refinement fits an unconstrained model to many points; does the fitted anisotropy change?
    out["anisotropy"] = {
        "minimal_median": float(np.median(d["min_aniso"])),
        "shipped_median": float(np.median(d["shp_aniso"])),
        "minimal_p99": float(np.percentile(d["min_aniso"], 99)),
        "shipped_p99": float(np.percentile(d["shp_aniso"], 99)),
    }
    # Low-support frames are where LIT-VO-005 eq. (8) predicts the largest effect.
    lo = inl <= np.percentile(inl, 10)
    hi = inl >= np.percentile(inl, 90)
    out["d_rot_by_support"] = {
        "low_decile_inliers_median": float(np.median(inl[lo])),
        "low_decile_d_rot_rms": float(np.sqrt((d["d_rot"][lo] ** 2).mean())),
        "high_decile_inliers_median": float(np.median(inl[hi])),
        "high_decile_d_rot_rms": float(np.sqrt((d["d_rot"][hi] ** 2).mean())),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True,
                    help="comma-separated repo-relative run dirs with a refinement sidecar")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    results = []
    for run in [r.strip() for r in a.runs.split(",") if r.strip()]:
        d = load(run)
        s = summarise(d, run)
        results.append(s)
        print(f"  {run:52s} inliers med {s['inliers_median']:6.0f}  "
              f"|d_rot| med {s['d_rot_deg']['median_abs']:.5f} deg  rms {s['d_rot_deg']['rms']:.5f}  "
              f"|d_trans| med {s['d_trans_px']['median_abs']:.4f} px  "
              f"rho(inliers,|d_rot|) {s['d_rot_deg']['spearman_with_inliers']:+.3f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
