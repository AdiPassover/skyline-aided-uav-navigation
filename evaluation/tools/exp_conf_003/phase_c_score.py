"""EXP-CONF-003 Phase C: score a counterfactual (refineEstimate = true) run's pair
transforms against the SAME per-pair independent references Phase A persisted.

The reference H_ref for frame i is derived from imagery alone, so it is valid for any run
over the same dataset. The counterfactual run's trajectory diverges from the frozen run's
(different recenters/keyframes), so this is a per-frame-index comparison of two systems on
identical imagery, not a paired comparison of identical match sets — the exact paired
minimal-vs-refined comparison on production match sets is the run's own refinement.csv
(EXP-VO-010's sidecar), analysed separately.

Output: evaluations/exp-conf-003/phase_c/<run_id>.csv with the same decomposition columns
as Phase A (transform under test = the counterfactual run's pair homography, conjugated to
full resolution).

Usage: python phase_c_score.py --run runs/amtown01-d-homography-rigid-refine-v1 \
           --phase-a evaluations/exp-conf-003/phase_a/amtown01-d.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
REPO = TOOL_DIR.parents[2]
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_002"))
from panel import load_run, vo_pair_homography                                  # noqa: E402
from motion_decomp import decompose                                             # noqa: E402
from phase_a import DECOMP_KEYS, H_KEYS, conjugate_to_full, downsample_factor   # noqa: E402

COLUMNS = ["frame_index", "event"] + DECOMP_KEYS


def load_refs(phase_a_csv: Path) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    refs: dict[int, np.ndarray] = {}
    with phase_a_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("r00", "") != "":
                refs[int(r["frame_index"])] = np.array(
                    [float(r[k]) for k in H_KEYS]).reshape(3, 3)
    meta = json.load(phase_a_csv.with_suffix(".json").open(encoding="utf-8"))
    return refs, tuple(meta["image_size"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--phase-a", required=True)
    a = ap.parse_args()

    run_dir = REPO / a.run
    frames, sidecar = load_run(run_dir)
    dfac = downsample_factor(run_dir)
    refs, (w, h) = load_refs(REPO / a.phase_a)

    out_path = REPO / f"evaluations/exp-conf-003/phase_c/{run_dir.name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        wtr.writeheader()
        for i in range(1, len(frames)):
            row: dict = {"frame_index": i, "event": frames[i]["event"]}
            H_cf, _ = vo_pair_homography(frames, sidecar, i)
            H_ref = refs.get(i)
            if H_cf is not None and H_ref is not None:
                d = decompose(conjugate_to_full(H_cf, dfac), H_ref, w, h)
                for k in DECOMP_KEYS:
                    row[k] = d[k]
                n += 1
            wtr.writerow({k: row.get(k, "") for k in COLUMNS})
    print(f"[phase_c] {run_dir.name}: {n} scored pairs -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
