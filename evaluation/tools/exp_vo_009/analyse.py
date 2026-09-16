"""`EXP-VO-009` Phases 7 and 9: score every motion model on every window, in one table.

Five arms per window — legacy homography, legacy affine, rigid homography, rigid affine and the new
**rigid similarity** — using the committed baselines for the first four and re-scoring all of them
through the *same* code so no number in the comparison comes from a different computation than any
other. `FULL_LOGICAL` is not re-run; its failure is established (`EXP-VO-006`, `EXP-VO-007`).

The primary metrics and the per-frame geometric diagnostics are imported unchanged from
`EXP-VO-007`'s `analyse` module rather than reimplemented, so this experiment's numbers are directly
comparable with that record's and with the 2026-08-26 synthesis report: one global Sim(2) alignment
(`DEC-003`), `naveval` untouched, the same RPE lengths.

**Runtime is reported with `EXP-VO-002`'s confound attached**, not bare: a model that tracks better
does more work per frame, so wall-clock cost is only interpretable alongside track and inlier counts.
Development laptop only — no target hardware is named anywhere in this repository, so no onboard or
real-time claim is available (Principle IX).
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
sys.path.insert(0, str(EVALUATION_DIR / "tools" / "exp_vo_007"))

from analyse import diagnostics, evaluate                                # noqa: E402
from naveval.dataset import load_dataset                                 # noqa: E402
from naveval.runrecord import load_run_record                            # noqa: E402

# (model, readout) -> run-id suffix. Legacy homography on hkairport01 predates the naming
# convention, so it is resolved per window below.
ARMS = [
    ("homography", "legacy"),
    ("affine", "legacy"),
    ("homography", "rigid"),
    ("affine", "rigid"),
    ("similarity", "rigid"),
]

WINDOWS = {
    "hkairport01-a": {
        "dataset": "datasets/hkairport01-a",
        "overrides": {("homography", "legacy"): "runs/hkairport01-a-run-v1",
                      ("affine", "legacy"): "runs/hkairport01-a-affine-v1"},
    },
    "hkairport01-b": {
        "dataset": "datasets/hkairport01-b",
        "overrides": {("homography", "legacy"): "runs/hkairport01-b-run-v1",
                      ("affine", "legacy"): "runs/hkairport01-b-affine-v1"},
    },
    "amtown01-c": {
        "dataset": "datasets/amtown01-c",
        "overrides": {},
    },
}


def resolve(window: str, model: str, readout: str) -> Path:
    spec = WINDOWS[window]
    rel = spec["overrides"].get((model, readout), f"runs/{window}-{model}-{readout}-v1")
    return REPO / rel


def inlier_stats(rd: Path) -> dict:
    rr = load_run_record(rd)
    tracks = np.array([f.track_count for f in rr.frame_estimates if f.track_count is not None],
                      dtype=float)
    inliers = np.array([f.inlier_count for f in rr.frame_estimates if f.inlier_count is not None],
                       dtype=float)
    out = {}
    if tracks.size:
        out["track_median"] = float(np.median(tracks))
        out["track_p05"] = float(np.percentile(tracks, 5))
    if inliers.size:
        out["inlier_median"] = float(np.median(inliers))
        out["inlier_p05"] = float(np.percentile(inliers, 5))
    if tracks.size and inliers.size:
        n = min(tracks.size, inliers.size)
        ratio = np.divide(inliers[:n], tracks[:n], out=np.full(n, np.nan), where=tracks[:n] > 0)
        out["inlier_ratio_median"] = float(np.nanmedian(ratio))
        out["inlier_ratio_p05"] = float(np.nanpercentile(ratio, 5))
    times = np.array([f.process_time_ns for f in rr.frame_estimates
                      if f.process_time_ns is not None], dtype=float)
    if times.size:
        out["ms_per_frame_median"] = float(np.median(times) / 1e6)
        out["ms_per_frame_p95"] = float(np.percentile(times, 95) / 1e6)
        out["effective_fps"] = float(1e9 / np.median(times))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default="hkairport01-a,hkairport01-b,amtown01-c")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    result: dict = {"windows": {}}
    for window in [w.strip() for w in a.windows.split(",") if w.strip()]:
        ds = load_dataset(REPO / WINDOWS[window]["dataset"])
        rows = {}
        for model, readout in ARMS:
            rd = resolve(window, model, readout)
            if not (rd / "frames.csv").exists():
                print(f"  {window:14s} {model:11s} {readout:7s} MISSING {rd}")
                continue
            ev = evaluate(rd, ds)
            row = {k: v for k, v in ev.items() if not k.startswith("_")}
            row["run_dir"] = str(rd.relative_to(REPO)).replace("\\", "/")
            row.update(inlier_stats(rd))
            diag = diagnostics(rd)
            row["diagnostics"] = {k: v for k, v in diag.items() if not k.startswith("_")}
            rows[f"{model}-{readout}"] = row
            print(f"  {window:14s} {model:11s} {readout:7s} "
                  f"normATE {100 * row['ate_rmse_normalised']:7.3f} %  "
                  f"yaw {row.get('yaw_rmse_deg', float('nan')):7.2f} deg  "
                  f"tracks {row.get('track_median', float('nan')):7.0f}  "
                  f"inliers {row.get('inlier_median', float('nan')):7.0f}  "
                  f"{row.get('ms_per_frame_median', float('nan')):6.2f} ms", flush=True)
        result["windows"][window] = rows

    # H4: does similarity sit inside the homography/affine rigid interval?
    conv = {}
    for window, rows in result["windows"].items():
        h = rows.get("homography-rigid", {}).get("ate_rmse_normalised")
        f = rows.get("affine-rigid", {}).get("ate_rmse_normalised")
        s = rows.get("similarity-rigid", {}).get("ate_rmse_normalised")
        if None in (h, f, s):
            continue
        lo, hi = min(h, f), max(h, f)
        conv[window] = {
            "rigid_homography": h, "rigid_affine": f, "rigid_similarity": s,
            "model_spread_homography_minus_affine": h - f,
            "similarity_inside_interval": bool(lo <= s <= hi),
            "similarity_beats_both": bool(s < lo),
            "similarity_worse_than_both": bool(s > hi),
        }
    result["h4_model_convergence"] = conv

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
