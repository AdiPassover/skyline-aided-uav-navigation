"""`EXP-VO-010` Phase 9: what refinement costs, measured back to back rather than across sessions.

Four arms — {affine, homography} × {refine false, true} — over the same window, one process
invocation each in immediate succession, with `VoRunnerApp --motion-timing` isolating the motion
stage (KLT + RANSAC + refinement) from mosaic rendering. Comparing run records produced on different
days would measure machine state as much as refinement cost, which is the caveat the 2026-08-26
synthesis report attached to the 0–51 % spread it found in readout overhead.

What this still is not, and does not claim to be:

* a clean benchmark — the JVM is fresh per run, so JIT warm-up is included in each arm equally
  rather than removed;
* independent of support — `EXP-VO-002`'s confound applies unchanged, which is why track and inlier
  counts are reported beside every timing figure. Refinement's cost is a function of the inlier
  count (`O(N)` for affine, an `SVD` of a `2N × 9` matrix for homography), so the two must be read
  together;
* a hardware claim — development laptop only. No target hardware is named anywhere in this
  repository, so no onboard or real-time claim is available (Principle IX).

The run records it writes are throwaway (`timing-*` ids) and are deleted afterwards; only the timing
CSVs and this script's JSON summary are kept.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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

ARMS = [(m, r) for m in ("affine", "homography") for r in (False, True)]


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
                    help="the window to time; the short one by default, and the same one "
                         "EXP-VO-009 R9 used, so the two are comparable")
    ap.add_argument("--hkairport-root",
                    default="datasets")
    ap.add_argument("--java-home", default="C:/Program Files/Java/jdk-19")
    ap.add_argument("--keep-runs", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dataset_dir = (f"{a.hkairport_root}/{a.window}" if a.window.startswith("hkairport01")
                   else f"datasets/{a.window}")
    timing_dir = REPO / "evaluations" / "exp-vo-010" / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)

    out: dict = {
        "window": a.window, "dataset_dir": dataset_dir, "arms": {},
        "method": "one VoRunnerApp invocation per arm, back to back, same window; "
                  "--motion-timing isolates KLT+RANSAC+refinement from mosaic rendering",
        "caveats": [
            "development laptop only; no target hardware is named in this repository, so no "
            "onboard or real-time claim is available (Principle IX)",
            "JIT warm-up is included in every arm rather than removed",
            "EXP-VO-002's confound applies, and refinement's cost is itself a function of the "
            "inlier count, so cost must be read beside the support figures",
        ],
    }

    for model, refine in ARMS:
        key = f"{model}-refine{'On' if refine else 'Off'}"
        run_id = f"timing-{a.window}-{key}"
        cfg = {
            "dataset_dir": dataset_dir,
            "output_dir": "runs",
            "run_id": run_id,
            "estimator_id": "stitching-vo",
            "estimator_version": "exp-vo-010-timing",
            "is_target_hardware": False,
            "downsampleFactor": 2,
            "motion_model": model,
            "navigation_source": "rigid_motion",
            "refineEstimate": refine,
        }
        cfg_path = timing_dir / f"config-{key}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        timing_csv = timing_dir / f"motion-{key}.csv"

        print(f"### timing {key} ...", flush=True)
        t0 = time.time()
        subprocess.run(
            [str(REPO / "gradlew.bat"), "run",
             "-PmainClass=org.boofcv.evaluation.VoRunnerApp",
             f'--args=--config {cfg_path.relative_to(REPO)} '
             f'--motion-timing {timing_csv.relative_to(REPO)}', "-q"],
            cwd=REPO, check=True, env={**dict(os.environ), "JAVA_HOME": a.java_home})
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

        out["arms"][key] = {
            "model": model, "refine": refine,
            "motion_stage": summarise_ns(motion_ns),
            "end_to_end": summarise_ns(end_ns),
            "wall_clock_s": wall,
            "track_median": float(np.median(tracks)) if tracks.size else None,
            "inlier_median": float(np.median(inliers)) if inliers.size else None,
        }
        m = out["arms"][key]
        print(f"    motion {m['motion_stage']['median_ms']:6.2f} ms   "
              f"end-to-end {m['end_to_end']['median_ms']:6.2f} ms   "
              f"tracks {m['track_median']:.0f}   inliers {m['inlier_median']:.0f}", flush=True)

        if not a.keep_runs:
            shutil.rmtree(REPO / "runs" / run_id, ignore_errors=True)

    # The number the record needs: refinement's overhead, per model, at matched conditions.
    overhead = {}
    for model in ("affine", "homography"):
        off = out["arms"].get(f"{model}-refineOff")
        on = out["arms"].get(f"{model}-refineOn")
        if not off or not on:
            continue
        overhead[model] = {
            "motion_ms_off": off["motion_stage"]["median_ms"],
            "motion_ms_on": on["motion_stage"]["median_ms"],
            "motion_overhead_rel": (on["motion_stage"]["median_ms"]
                                    / off["motion_stage"]["median_ms"] - 1.0),
            "motion_overhead_ms": (on["motion_stage"]["median_ms"]
                                   - off["motion_stage"]["median_ms"]),
            "end_to_end_overhead_rel": (on["end_to_end"]["median_ms"]
                                        / off["end_to_end"]["median_ms"] - 1.0),
            "inlier_median_off": off["inlier_median"],
            "inlier_median_on": on["inlier_median"],
        }
    out["refinement_overhead"] = overhead

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("\n" + json.dumps(overhead, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
