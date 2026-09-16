"""Read/write the canonical benchmark GT store (``contracts/benchmark-gt-store.md``).

Tracked side only: ``manifest.json``, ``samples.csv``, ``gt_curves/<sample_id>.csv``. Heavy imagery
stays under the untracked benchmark root; ``samples.csv`` refers to it by *relative* path and the
benchmark root arrives from the run config — a tracked file containing an absolute path is a
recorded defect class in this lane (EXP-SKY-001 found one), so writers here refuse them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path, PureWindowsPath

import numpy as np

from hsreloc.extraction.curves import (
    CONVERSION_KIND_GT,
    CanonicalCurveError,
    REASON_NONE,
    CanonicalCurve,
    make_curve,
)

SPLITS = ("eval_only", "dev", "eval")


class GtStoreError(Exception):
    """The GT store is malformed, incomplete, or self-inconsistent."""


def _is_absolute(p: str) -> bool:
    return Path(p).is_absolute() or PureWindowsPath(p).is_absolute()


def write_curve_csv(path: Path, curve: CanonicalCurve) -> None:
    curve.validate()
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["col", "row", "reason"])
        for c in range(curve.width_px):
            v = curve.rows[c]
            if np.isfinite(v):
                w.writerow([c, f"{float(v):.3f}", REASON_NONE])
            else:
                w.writerow([c, "", curve.invalid_reasons[c]])


def read_curve_csv(path: Path, width_px: int, height_px: int, producer: str) -> CanonicalCurve:
    p = Path(path)
    if not p.exists():
        raise GtStoreError(f"missing GT curve file {p}")
    rows = np.full(width_px, np.nan, dtype=np.float64)
    reasons = [None] * width_px
    seen = 0
    with p.open("r", encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            c = int(rec["col"])
            if not (0 <= c < width_px):
                raise GtStoreError(f"{p}: column {c} outside 0..{width_px - 1}")
            reasons[c] = rec["reason"]
            if rec["row"] != "":
                rows[c] = float(rec["row"])
            seen += 1
    if seen != width_px or any(r is None for r in reasons):
        raise GtStoreError(f"{p}: covers {seen} of {width_px} columns — GT must be complete")
    try:
        return make_curve(width_px, height_px, rows, reasons, CONVERSION_KIND_GT, producer)
    except CanonicalCurveError as e:
        raise GtStoreError(f"{p}: {e}") from e


def write_store(out_dir: Path, manifest: dict, samples: list) -> None:
    """``samples``: dicts with sample_id, camera_id, split, width_px, height_px, image_relpath,
    curve (CanonicalCurve), and optionally ``gt_ref`` — samples sharing one GT curve (a static
    camera's frames) name the same ref so the curve is stored once, not per frame."""
    out = Path(out_dir)
    (out / "gt_curves").mkdir(parents=True, exist_ok=True)
    written = {}
    for s in samples:
        if _is_absolute(s["image_relpath"]):
            raise GtStoreError(
                f"{s['sample_id']}: image_relpath {s['image_relpath']!r} is absolute — tracked "
                f"store files must stay machine-independent")
        if s["split"] not in SPLITS:
            raise GtStoreError(f"{s['sample_id']}: unknown split {s['split']!r}")
        ref = s.get("gt_ref", s["sample_id"])
        if ref in written:
            if written[ref] is not s["curve"] and written[ref].rows.tolist() != s["curve"].rows.tolist():
                raise GtStoreError(f"gt_ref {ref!r} bound to two different curves")
        else:
            write_curve_csv(out / "gt_curves" / f"{_flat(ref)}.csv", s["curve"])
            written[ref] = s["curve"]
    with (out / "samples.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "camera_id", "split", "width_px", "height_px",
                    "image_relpath", "gt_ref", "gt_invalid_count"])
        for s in samples:
            w.writerow([s["sample_id"], s.get("camera_id", ""), s["split"],
                        s["width_px"], s["height_px"], s["image_relpath"],
                        s.get("gt_ref", s["sample_id"]),
                        s["curve"].width_px - s["curve"].n_valid])
    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def _flat(sample_id: str) -> str:
    return sample_id.replace("/", "__")


def load_store(store_dir: Path):
    """Returns ``(manifest, samples)``; each sample dict carries its loaded GT curve."""
    store = Path(store_dir)
    mf = store / "manifest.json"
    if not mf.exists():
        raise GtStoreError(f"no manifest.json under {store}")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    samples = []
    cache = {}
    with (store / "samples.csv").open("r", encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            width = int(rec["width_px"])
            height = int(rec["height_px"])
            ref = rec.get("gt_ref") or rec["sample_id"]
            if ref not in cache:
                cache[ref] = read_curve_csv(store / "gt_curves" / f"{_flat(ref)}.csv",
                                            width, height,
                                            producer=f"gt_store:{store.name}")
            curve = cache[ref]
            if curve.n_valid == 0:
                raise GtStoreError(
                    f"{rec['sample_id']}: GT has zero valid columns — such a sample must be "
                    f"excluded at ingest, not evaluated")
            samples.append({"sample_id": rec["sample_id"], "camera_id": rec["camera_id"],
                            "split": rec["split"], "width_px": width, "height_px": height,
                            "image_relpath": rec["image_relpath"], "curve": curve})
    if not samples:
        raise GtStoreError(f"{store}: empty samples.csv")
    return manifest, samples
