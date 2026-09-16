"""`EXP-VO-008` Phase 6e: can a longer composition baseline reduce the heading deficit?

The polar rotation accumulates a heading error of -32.2 deg on `AMtown01` and -18.2 deg on
`hkairport01-b`. If that were dominated by per-frame *noise*, composing the transform over longer
baselines before extracting the rotation should reduce it, and would suggest an easy improvement:
decouple the rotation baseline from the translation baseline, keeping per-frame increments for
position (which the granularity sweep shows is best) while estimating heading over longer spans.

This script tests that, and the answer is the reason it is committed: **the deficit is invariant to
the baseline.** It is a property of the estimator's transform sequence, not of how the readout
chunks it, so no re-chunking can fix it.

**Wraparound guard.** `atan2` returns an angle in (-pi, pi], so once a segment's own rotation exceeds
half a turn the extracted angle wraps and the sum is meaningless. Segments whose GT turn exceeds
`WRAP_GUARD_DEG` are flagged and the row is marked invalid rather than reported as a number — the
raw sweep showed a -381 deg "result" at N = 500 on `hkairport01-b` that is entirely this artefact.
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
from naveval.dataset import load_dataset                                 # noqa: E402
from naveval.runrecord import load_run_record                            # noqa: E402
from rotation_diagnostic import gt_heading_increments                    # noqa: E402

N_GRID = [1, 2, 5, 10, 25, 50, 100, 200, 500]
WRAP_GUARD_DEG = 150.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True,
                    help="label:dataset:run triples")
    ap.add_argument("--width", type=int, default=1224)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cx, cy = a.width / 2.0, a.height / 2.0

    report = {}
    for case in a.cases:
        label, dsn, run = case.split(":")
        ds = load_dataset(REPO / dsn)
        rec = rc.load_recording(REPO / run)
        rr = load_run_record(REPO / run)
        Ginv = np.linalg.inv(rec.G)
        n = len(rec)
        dgt = gt_heading_increments(ds, rr.timestamps_s)
        gt_tot = float(np.degrees(dgt.sum()))
        rows = []
        print(f"=== {label}   GT total turn {gt_tot:+.1f} deg ===")
        for N in N_GRID:
            tp = te = 0.0
            worst = 0.0
            for i in range(0, n - 1, N):
                j = min(i + N, n - 1)
                S = rec.G[i] @ Ginv[j]
                tp += rc.polar_rotation(rc._jacobian(S, cx, cy))
                te += rc.edge_angle(S, a.width, a.height)
                worst = max(worst, abs(np.degrees(dgt[i + 1:j + 1].sum())))
            valid = worst < WRAP_GUARD_DEG
            row = {"N": N, "segment_max_gt_turn_deg": worst, "valid": bool(valid),
                   "polar_accumulated_error_deg": float(np.degrees(tp) - gt_tot),
                   "edge_accumulated_error_deg": float(np.degrees(te) - gt_tot)}
            rows.append(row)
            flag = "" if valid else "   INVALID (segment rotation wraps past +/-180 deg)"
            print(f"  N={N:4d}  polar {row['polar_accumulated_error_deg']:+8.1f}   "
                  f"edge {row['edge_accumulated_error_deg']:+8.1f}{flag}")
        v = [r["polar_accumulated_error_deg"] for r in rows if r["valid"]]
        report[label] = {"gt_total_turn_deg": gt_tot, "rows": rows,
                         "polar_spread_over_valid_N_deg": float(max(v) - min(v))}
        print(f"  polar deficit spread across valid N: "
              f"{report[label]['polar_spread_over_valid_N_deg']:.2f} deg\n")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
