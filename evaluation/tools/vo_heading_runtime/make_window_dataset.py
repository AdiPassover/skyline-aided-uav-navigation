"""Cut a time window out of an existing dataset, referencing its imagery in place (`DEC-VO-010`).

Why this exists: the runtime heading verification needs a real-imagery replay **through a real hard
visual loss**, and the only sequence in this repository with one is `amtown01-d`'s full 11 894-frame
cruise — far more compute than the machine this was run on could host. Its two hard-loss events sit at
frames 10698 and 11793, so a window around them carries the whole of what the verification needs at a
fraction of the cost.

**What this is and is not.** It is a *view* of an existing dataset: no imagery is copied, decoded or
re-encoded — `frames.csv` points back at the parent's `images/` with a relative path. It is **not a
new flight**, and a run over it is **not comparable to the parent's committed run record**: the
estimator starts at a different frame, so its keyframe schedule, canvas schedule and accumulated state
all differ, and its restart events may not fall in the same places. Any number from it characterises
the *runtime plumbing*, never the algorithm's accuracy on the parent sequence.

Usage:
  python evaluation/tools/vo_heading_runtime/make_window_dataset.py \
      --source amtown01-d --start 10500 --end 11894 --id amtown01-d-w10500
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]


def rows(path: pathlib.Path) -> list[dict]:
    with io.open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, header: list[str], out_rows: list[list[str]]) -> None:
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in out_rows:
            f.write(",".join(r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True, help="exclusive")
    ap.add_argument("--id", required=True)
    a = ap.parse_args()

    src = ROOT / "datasets" / a.source
    dst = ROOT / "datasets" / a.id
    dst.mkdir(parents=True, exist_ok=True)

    frames = rows(src / "frames.csv")
    window = [r for r in frames if a.start <= int(r["frame_index"]) < a.end]
    if not window:
        print("no frames in [%d, %d)" % (a.start, a.end))
        return 1
    t0, t1 = float(window[0]["timestamp_s"]), float(window[-1]["timestamp_s"])

    write_csv(dst / "frames.csv", ["frame_index", "timestamp_s", "image_path"],
              [[str(i), r["timestamp_s"], "../%s/%s" % (a.source, r["image_path"])]
               for i, r in enumerate(window)])

    for name in ("groundtruth.csv", "attitude.csv"):
        p = src / name
        if not p.exists():
            continue
        allr = rows(p)
        hdr = list(allr[0].keys())
        # One sample of margin on each side, so a zero-order hold at the first frame has something
        # to hold and the window is not silently short of its own edges.
        keep = [r for r in allr if t0 - 1.0 <= float(r["timestamp_s"]) <= t1 + 1.0]
        write_csv(dst / name, hdr, [[r[h] for h in hdr] for r in keep])

    meta = json.load(io.open(src / "dataset.json", encoding="utf-8"))
    meta["dataset_id"] = a.id
    meta["dataset_revision"] = "v1"
    meta["evidence_caveat"] = (
        "WINDOW OF %s, NOT A FLIGHT. Frames %d..%d of that sequence, re-indexed from 0, with the "
        "imagery referenced in place (no copy, no re-encode). It exists for ONE purpose: a runtime "
        "replay through the two real hard visual losses that sequence contains, at a fraction of "
        "the full cruise's compute. A run over this window is NOT comparable to %s's committed run "
        "record -- the estimator starts at a different frame, so its keyframe and canvas schedules "
        "and all accumulated state differ, and its restart events need not fall in the same places. "
        "No trajectory-accuracy number from it characterises the algorithm or the flight. || %s"
        % (a.source, a.start, a.end - 1, a.source, meta.get("evidence_caveat", "")))
    meta.setdefault("metadata", {})["window_of"] = {
        "source_dataset": a.source,
        "frame_range": [a.start, a.end - 1],
        "n_frames": len(window),
        "time_span_s": [t0, t1],
        "imagery": "referenced in place via a relative image_path; nothing was copied",
    }
    io.open(dst / "dataset.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps(meta, indent=2) + "\n")

    print("%s: %d frames, t=%.3f..%.3f (%.1f s)" % (a.id, len(window), t0, t1, t1 - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
