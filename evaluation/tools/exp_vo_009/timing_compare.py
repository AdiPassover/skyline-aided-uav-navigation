"""`EXP-VO-009` Phase 9: motion-stage runtime of the three models, measured back to back.

**Why this exists rather than reading the run records.** The homography and affine run records were
produced by `EXP-VO-006` and `EXP-VO-007` on other days; the similarity records were produced by
this experiment. Comparing their `process_time_ns` across sessions measures machine state as much as
model cost — which is precisely the caveat the 2026-08-26 synthesis report §10.2 attached to the
0–51 % spread it found in the readout overhead. This driver instead runs all three models over the
**same** window, in one process invocation each, in immediate succession, with
`VoRunnerApp --motion-timing` isolating the motion stage (KLT + RANSAC) from mosaic rendering.

It still is not a clean benchmark and does not claim to be:

* The JVM is fresh per run, so JIT warm-up is included in each arm equally rather than removed.
* `EXP-VO-002`'s confound applies unchanged and is the reason track and inlier counts are reported
  beside every timing figure: **a model that tracks better does more work per frame**, so a slower
  arm is not necessarily a more expensive one. `EXP-VO-002` found affine 22–34 % slower than
  homography for exactly this reason, and within ±11 % at matched track counts.
* Development laptop only. No target hardware is named anywhere in this repository, so no onboard or
  real-time claim is available (Principle IX).

The run records it writes are throwaway (`timing-*` ids) and are deleted afterwards; only the timing
CSVs and this script's JSON summary are kept.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parents[2]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))

from naveval.runrecord import load_run_record                            # noqa: E402

MODELS = ("homography", "affine", "similarity")


def summarise_ns(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    ms = values / 1e6
    return {
        "n": int(ms.size),
        "median_ms": float(np.median(ms)),
        "mean_ms": float(ms.mean()),
        "p05_ms": float(np.percentile(ms, 5)),
        "p95_ms": float(np.percentile(ms, 95)),
        "effective_fps_median": float(1000.0 / np.median(ms)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", default="hkairport01-a",
                    help="the window to time; the short one by default, since this is a cost "
                         "measurement and not an accuracy one")
    ap.add_argument("--hkairport-root",
                    default="datasets")
    ap.add_argument("--java-home", default="C:/Program Files/Java/jdk-19")
    ap.add_argument("--keep-runs", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dataset_dir = (f"{a.hkairport_root}/{a.window}" if a.window.startswith("hkairport01")
                   else f"datasets/{a.window}")
    timing_dir = REPO / "evaluations" / "exp-vo-009" / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)

    out: dict = {"window": a.window, "dataset_dir": dataset_dir, "models": {},
                 "method": "one VoRunnerApp invocation per model, back to back, same window; "
                           "--motion-timing isolates KLT+RANSAC from mosaic rendering",
                 "caveats": [
                     "development laptop only; no target hardware is named in this repository, so "
                     "no onboard or real-time claim is available (Principle IX)",
                     "JIT warm-up is included in every arm rather than removed",
                     "EXP-VO-002's confound applies: a model that tracks better does more work, so "
                     "cost must be read at matched support",
                 ]}

    for model in MODELS:
        run_id = f"timing-{a.window}-{model}"
        cfg = {
            "dataset_dir": dataset_dir,
            "output_dir": "runs",
            "run_id": run_id,
            "estimator_id": "stitching-vo",
            "estimator_version": "exp-vo-009-timing",
            "is_target_hardware": False,
            "downsampleFactor": 2,
            "motion_model": model,
            "navigation_source": "rigid_motion",
        }
        cfg_path = timing_dir / f"config-{model}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        timing_csv = timing_dir / f"motion-{model}.csv"

        print(f"### timing {model} ...", flush=True)
        t0 = time.time()
        subprocess.run(
            [str(REPO / "gradlew.bat"), "run",
             "-PmainClass=org.boofcv.evaluation.VoRunnerApp",
             f'--args=--config {cfg_path.relative_to(REPO)} '
             f'--motion-timing {timing_csv.relative_to(REPO)}', "-q"],
            cwd=REPO, check=True,
            env={**dict(__import__("os").environ), "JAVA_HOME": a.java_home})
        wall = time.time() - t0

        with timing_csv.open(newline="") as f:
            motion_ns = np.array([float(r["motion_time_ns"]) for r in csv.DictReader(f)])
        rr = load_run_record(REPO / "runs" / run_id)
        end_ns = np.array([f.process_time_ns for f in rr.frame_estimates
                           if f.process_time_ns is not None], dtype=float)
        tracks = np.array([f.track_count for f in rr.frame_estimates
                           if f.track_count is not None], dtype=float)
        inliers = np.array([f.inlier_count for f in rr.frame_estimates
                            if f.inlier_count is not None], dtype=float)

        out["models"][model] = {
            "motion_stage": summarise_ns(motion_ns),
            "end_to_end": summarise_ns(end_ns),
            "wall_clock_s": wall,
            "track_median": float(np.median(tracks)) if tracks.size else None,
            "inlier_median": float(np.median(inliers)) if inliers.size else None,
            "us_per_inlier_median": (float(np.median(motion_ns) / 1e3 / np.median(inliers))
                                     if inliers.size and np.median(inliers) > 0 else None),
        }
        m = out["models"][model]
        print(f"    motion {m['motion_stage']['median_ms']:.2f} ms   "
              f"end-to-end {m['end_to_end']['median_ms']:.2f} ms   "
              f"tracks {m['track_median']:.0f}   inliers {m['inlier_median']:.0f}   "
              f"{m['us_per_inlier_median']:.3f} us/inlier", flush=True)

        if not a.keep_runs:
            shutil.rmtree(REPO / "runs" / run_id, ignore_errors=True)

    if "affine" in out["models"]:
        base = out["models"]["affine"]["motion_stage"]["median_ms"]
        out["motion_stage_relative_to_affine"] = {
            m: out["models"][m]["motion_stage"]["median_ms"] / base for m in out["models"]}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
