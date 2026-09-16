"""EXP-CONF-001 P3: real-data extension of the FR-013/T027 bit-identity invariant.

Compares a confidence-instrumented capture (run-record v1.1.0) against the committed
baseline capture of the same sequence (v1.0.0), asserting that every persisted
*estimation* field is string-identical frame by frame:

- ``frames.csv``: all v1.0.0 columns except ``process_time_ns`` (wall-clock, excluded
  exactly as T027 excludes it);
- ``logical_transform.csv``: every column (the sidecar carries no timing).

Any difference is a mismatch; the pre-registered expectation is zero. Usage:

    python verify_bit_identity.py --baseline runs/<id>-rigid-v1 --candidate runs/<id>-rigid-conf-v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ESTIMATION_COLUMNS = [
    "frame_index", "timestamp_s", "est_x", "est_y", "est_z", "est_yaw_deg",
    "success", "event", "reference_id",
    "h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22",
    "track_count", "inlier_count",
]  # process_time_ns deliberately excluded (wall-clock)


def read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        return header, list(reader)


def compare(baseline: Path, candidate: Path) -> dict:
    result: dict = {"baseline": str(baseline), "candidate": str(candidate)}
    mismatches = 0
    details: list[str] = []

    bh, brows = read_rows(baseline / "frames.csv")
    ch, crows = read_rows(candidate / "frames.csv")
    if len(brows) != len(crows):
        raise SystemExit(f"row-count mismatch: {len(brows)} vs {len(crows)}")
    bidx = {c: bh.index(c) for c in ESTIMATION_COLUMNS}
    cidx = {c: ch.index(c) for c in ESTIMATION_COLUMNS}
    for i, (br, cr) in enumerate(zip(brows, crows)):
        for col in ESTIMATION_COLUMNS:
            if br[bidx[col]] != cr[cidx[col]]:
                mismatches += 1
                if len(details) < 20:
                    details.append(
                        f"frames.csv row {i} col {col}: {br[bidx[col]]!r} != {cr[cidx[col]]!r}")
    result["frames_rows"] = len(brows)
    result["frames_columns_compared"] = len(ESTIMATION_COLUMNS)

    blt = baseline / "logical_transform.csv"
    clt = candidate / "logical_transform.csv"
    if blt.exists() and clt.exists():
        bh2, brows2 = read_rows(blt)
        ch2, crows2 = read_rows(clt)
        if bh2 != ch2:
            raise SystemExit("logical_transform.csv header mismatch")
        if len(brows2) != len(crows2):
            raise SystemExit(
                f"logical_transform.csv row-count mismatch: {len(brows2)} vs {len(crows2)}")
        for i, (br, cr) in enumerate(zip(brows2, crows2)):
            if br != cr:
                mismatches += 1
                if len(details) < 20:
                    details.append(f"logical_transform.csv row {i}: differs")
        result["logical_transform_rows"] = len(brows2)
        result["logical_transform_columns_compared"] = len(bh2)
    else:
        result["logical_transform_rows"] = None

    result["mismatches"] = mismatches
    result["mismatch_examples"] = details
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    args = ap.parse_args()
    result = compare(args.baseline, args.candidate)
    print(json.dumps(result, indent=2))
    if result["mismatches"] != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
