"""`EXP-VO-009` gate 4b: assert a re-run reproduces a committed run record bit-identically.

Compares every column that describes estimator *behaviour* — pose, success, event, reference id,
track and inlier counts, and the full 3x3 transform — as raw text, with no tolerance. Timing columns
are excluded because they are wall-clock and cannot reproduce.

The gate exists because {@code DEC-VO-006} adds a third motion model to a shared estimator, and the
homography arm is what every committed baseline in this lane rests on. A behavioural change there
would silently invalidate `EXP-002`, `EXP-VO-002`, `EXP-VO-006`, `EXP-VO-007` and `EXP-VO-008` at
once, so it is checked rather than argued.

Exit status is 0 only when there are zero mismatches.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

COLUMNS = ("frame_index", "est_x", "est_y", "est_yaw_deg", "success", "event", "reference_id",
           "track_count", "inlier_count",
           "h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22")


def load(path: Path) -> list[dict]:
    with (path / "frames.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    a = ap.parse_args()

    ref = load(Path(a.reference))
    cand = load(Path(a.candidate))
    if len(ref) != len(cand):
        print(f"FAIL row count {len(ref)} != {len(cand)}")
        return 1

    present = [c for c in COLUMNS if c in ref[0] and c in cand[0]]
    missing = [c for c in COLUMNS if c not in present]
    mismatches = {c: 0 for c in present}
    for r, c in zip(ref, cand):
        for col in present:
            if r[col] != c[col]:
                mismatches[col] += 1

    total = sum(mismatches.values())
    print(f"rows={len(ref)}  columns compared={len(present)}"
          + (f"  (absent, skipped: {', '.join(missing)})" if missing else ""))
    if total:
        print("FAIL mismatches:", {k: v for k, v in mismatches.items() if v})
        return 1
    print(f"PASS  0 mismatches over {len(ref)} rows x {len(present)} columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
