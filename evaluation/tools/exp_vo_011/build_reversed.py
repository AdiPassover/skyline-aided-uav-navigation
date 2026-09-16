"""EXP-VO-011 Phase 6b: build a time-reversed replay of a real window, as a known-answer test on
REAL imagery.

**Why this is the sharpest real-data test available here.** The estimator reports a persistent
positive per-frame log-scale increment -- under `LIT-VO-003` eq. (2) that means "the camera is
receding from the surface", a phantom climb of 0.21-0.44 m/s on flights whose height above the
imaged surface is flat to a few percent. Play the same frames backwards and the physics inverts
exactly: every true height change, every true parallax, every true distortion-induced apparent
scale reverses sign, because the camera trajectory is the same path traversed the other way.

So the two outcomes separate cleanly, with no ground truth and no calibration:

- **bias flips sign** -> it tracks something about the scene-and-motion geometry (relief, distortion,
  a real height trend). A geometric mechanism cannot prefer a direction of time.
- **bias keeps its sign** -> it is a property of the estimator's interaction with the imagery
  (tracking asymmetry, appearance change, the fit itself), because only an arrow-of-time-carrying
  process can report "climbing" in both directions at once.

The reversed dataset re-uses the original JPEGs untouched -- only the order and the timestamps
change -- so nothing about image content, compression or exposure differs between the arms.

Usage::

    python evaluation/tools/exp_vo_011/build_reversed.py --window hkairport01-a \
        --source datasets/hkairport01-a
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

CAVEAT = (
    "DIAGNOSTIC REPLAY, NOT A FLIGHT. This dataset is {src} with its frame order reversed in time "
    "and its timestamps re-synthesised to stay increasing; the JPEGs are the originals, referenced "
    "in place and not copied or re-encoded. It exists for exactly one measurement (EXP-VO-011 "
    "Phase 6b): whether the estimator's per-frame log-scale bias flips sign when the arrow of time "
    "does. Ground truth is the original trajectory reversed, which is the correct physical reading "
    "of a reversed replay, but NO trajectory-accuracy number from this dataset should be reported "
    "as characterising the flight or the algorithm -- the aircraft never flew this. Original "
    "caveat follows. || {orig}"
)


def build(window: str, source: Path, out_root: Path) -> Path:
    src_desc = json.loads((source / "dataset.json").read_text())

    frames = list(csv.DictReader((source / "frames.csv").open(newline="")))
    gt = list(csv.DictReader((source / "groundtruth.csv").open(newline="")))
    att_path = source / "attitude.csv"
    att = list(csv.DictReader(att_path.open(newline=""))) if att_path.exists() else []

    t = [float(f["timestamp_s"]) for f in frames]
    t0, t_end = t[0], t[-1]

    dest = out_root / f"{window}-reversed"
    dest.mkdir(parents=True, exist_ok=True)

    # Where the images actually live, expressed relative to the new dataset root so the descriptor
    # carries no absolute path. `DirectoryFrameSource` does `root.resolve(image_path)`, so a
    # relative path that walks out of `dest` resolves correctly.
    def rel_image(p: str) -> str:
        target = (source / p).resolve()
        try:
            return Path(target).relative_to(dest.resolve(), walk_up=True).as_posix()
        except (ValueError, TypeError):
            import os
            return Path(os.path.relpath(target, dest.resolve())).as_posix()

    with (dest / "frames.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["frame_index", "timestamp_s", "image_path"])
        for i, f in enumerate(reversed(frames)):
            # Mirror the original sampling instants about the window so that inter-frame intervals
            # are the original ones in reverse -- not a uniform re-grid, which would silently change
            # the per-frame baseline distribution the bias is measured over.
            w.writerow([i, f"{t0 + (t_end - float(f['timestamp_s'])):.6f}", rel_image(f["image_path"])])

    def mirror_rows(rows: list[dict], out: Path) -> None:
        if not rows:
            return
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            for r in reversed(rows):
                r = dict(r)
                r["timestamp_s"] = f"{t0 + (t_end - float(r['timestamp_s'])):.6f}"
                w.writerow(r)

    mirror_rows(gt, dest / "groundtruth.csv")
    mirror_rows(att, dest / "attitude.csv")

    desc = dict(src_desc)
    desc["dataset_id"] = f"{window}-reversed"
    desc["evidence_caveat"] = CAVEAT.format(src=window, orig=src_desc.get("evidence_caveat", ""))
    md = dict(desc.get("metadata", {}))
    md["notes"] = (f"EXP-VO-011 Phase 6b time-reversed replay of {window}. "
                   f"Built by evaluation/tools/exp_vo_011/build_reversed.py.")
    md["provenance"] = f"derived from {window} by frame-order reversal; images referenced in place"
    md["reversed_from"] = window
    desc["metadata"] = md
    (dest / "dataset.json").write_text(json.dumps(desc, indent=2))

    print(f"{dest}: {len(frames)} frames, "
          f"t {t0:.3f}..{t0 + (t_end - t0):.3f}, images referenced at {rel_image(frames[0]['image_path'])[:60]}...")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default=str(REPO / "datasets"))
    a = ap.parse_args()
    build(a.window, Path(a.source), Path(a.out))


if __name__ == "__main__":
    main()
