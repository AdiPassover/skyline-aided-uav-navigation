"""Semantic-segmentation inference for the SKY silver-label DEV trial (EXP-SKY-005).

**Runs in an isolated virtual environment**, not the project venv: it imports torch/transformers/PIL
only and deliberately does *not* import ``hsreloc``, so the deep-learning runtime never becomes a
project dependency (see DEC-SKY-005). It reads the tracked ECL index with the stdlib ``csv`` module
and writes one compressed ADE20K label map per image; every downstream step (mask -> curve,
metrics, renders) runs in the ordinary project environment.

Preprocessing reproduces the NVlabs SegFormer ADE20K *test* pipeline
(``local_configs/_base_/datasets/ade20k_repeat.py``): keep-ratio resize to ``img_scale=(2048, 512)``,
align each side up to a multiple of 32, ImageNet normalisation, whole-image forward, then bilinear
upsampling of the logits to the original image size before ``argmax``.

The output is a **silver label**: a second opinion from an independent published model. It is never
ground truth, and nothing here is compared against, or tuned towards, our own extractor.

Two input modes, one identical model path
-----------------------------------------
``mode: "ecl_index"`` (the default, and what ``EXP-SKY-005``/``006`` ran) walks the tracked ECL
condition index and writes ``<condition>/s<scale>/<group_id>.npz``.

``mode: "sessions"`` walks a **frame list** produced by
``hsreloc.simret.sources.write_silver_frame_list`` — a plain CSV of
``session_id,observation_id,image_path`` — and writes ``<session_id>/s<scale>/<observation_id>.npz``.
That is the only difference: the pinned revision, the NVlabs preprocessing, the sky class and the
conversion downstream are untouched, so a simulator curve and an ECL curve are produced by the same
method and remain comparable. The list is written on the *other* side of the venv boundary precisely
so this script still imports no ``hsreloc`` module.

⚠️ Simulator imagery is **out of ADE20K's photographic domain**. Nothing here adapts to that, and
nothing may: ``PROT-SKY-001`` §5 requires a DEV inspection of a declared handful of rendered frames
before any freeze names SegFormer the primary source.

Usage (isolated venv):

    <venv>/Scripts/python.exe scripts/silver_infer.py --config configs/silver-infer-dev.json
    <venv>/Scripts/python.exe scripts/silver_infer.py --config configs/silver-infer-sim.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

# NVlabs ADE20K test pipeline constants (verified from the repository config, 2026-08-25).
IMG_SCALE_LONG = 2048
IMG_SCALE_SHORT = 512
SIZE_DIVISOR = 32
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SKY_CLASS_INDEX = 2          # ADE20K 150-class id2label[2] == "sky" (model config, verified)
CONDITIONS = ("original", "summer", "winter", "evening")
INFER_VERSION = "1.0.0"


def aligned_size(w: int, h: int, long_side: int, short_side: int, divisor: int) -> tuple:
    """mmseg keep-ratio resize to (long, short) followed by AlignedResize's round-up to `divisor`."""
    scale = min(long_side / max(w, h), short_side / min(w, h))
    rw, rh = int(w * scale + 0.5), int(h * scale + 0.5)
    up = lambda v: int(np.ceil(v / divisor)) * divisor  # noqa: E731
    return up(rw), up(rh)


def preprocess(img: Image.Image, long_side: int, short_side: int) -> torch.Tensor:
    w, h = img.size
    tw, th = aligned_size(w, h, long_side, short_side, SIZE_DIVISOR)
    resized = img.resize((tw, th), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def label_map(model, img: Image.Image, long_side: int, short_side: int) -> np.ndarray:
    """Full-resolution ADE20K label map (uint8) for one image."""
    w, h = img.size
    x = preprocess(img, long_side, short_side)
    with torch.no_grad():
        logits = model(pixel_values=x).logits           # (1, 150, H/4, W/4)
    up = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return up.argmax(dim=1)[0].to(torch.uint8).numpy()


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def _plan_ecl(cfg, cfg_path, args) -> tuple:
    """(work items, out_root, extra manifest fields) for the ECL condition index."""
    root = Path(cfg["benchmark_root"])
    resolve = lambda v: (Path(v) if Path(v).is_absolute() else (cfg_path.parent / v).resolve())  # noqa: E731
    groups_csv = resolve(cfg["index_groups_csv"])
    out_root = root / cfg["out_root"] if not Path(cfg["out_root"]).is_absolute() else Path(cfg["out_root"])
    population = cfg.get("population", "dev")
    if population not in ("dev", "all"):
        raise SystemExit(f"population must be 'dev' or 'all', got {population!r}")
    rows = [r for r in csv.DictReader(groups_csv.open(encoding="utf-8"))
            if population == "all" or r["dev_subset"] == "True"]
    rows.sort(key=lambda r: r["group_id"])
    if args.groups:
        wanted = {g.strip() for g in args.groups.split(",") if g.strip()}
        rows = [r for r in rows if r["group_id"] in wanted]
    groups_file = cfg.get("groups_file")
    if groups_file:
        gf = resolve(groups_file)
        wanted = {ln.strip() for ln in gf.read_text(encoding="utf-8").splitlines() if ln.strip()}
        rows = [r for r in rows if r["group_id"] in wanted]
        print(f"[silver-infer] groups_file {gf.name}: {len(rows)} of the population selected")
    print(f"[silver-infer] population {population}: {len(rows)} groups")

    items = []
    for r in rows:
        for cond in CONDITIONS:
            rel = r["original_relpath"] if cond == "original" else r[f"{cond}_relpath"]
            if rel:
                items.append({"bucket": cond, "stem": r["group_id"], "path": root / rel,
                              "relpath": rel, "unit": r["group_id"]})
    return items, out_root, {"population": population, "groups_file": groups_file,
                             "keying": "condition", "n_units": len(rows)}


def _plan_sessions(cfg, cfg_path, args) -> tuple:
    """(work items, out_root, extra manifest fields) for an observation-session frame list.

    The list is a plain CSV written by ``hsreloc.simret.sources.write_silver_frame_list`` so that this
    script still imports no project module. Output is keyed ``<session_id>/s<scale>/<observation_id>``.
    """
    resolve = lambda v: (Path(v) if Path(v).is_absolute() else (cfg_path.parent / v).resolve())  # noqa: E731
    frame_list = resolve(cfg["frame_list"])
    out_root = Path(cfg["out_root"]) if Path(cfg["out_root"]).is_absolute() \
        else resolve(cfg["out_root"])
    rows = list(csv.DictReader(frame_list.open(encoding="utf-8")))
    missing = [c for c in ("session_id", "observation_id", "image_path") if c not in (rows[0] if rows else {})]
    if not rows or missing:
        raise SystemExit(f"{frame_list}: needs columns session_id,observation_id,image_path "
                         f"(missing {missing}, {len(rows)} rows)")
    only = {s.strip() for s in (cfg.get("sessions") or []) if s.strip()}
    if only:
        rows = [r for r in rows if r["session_id"] in only]
    if args.groups:
        wanted = {g.strip() for g in args.groups.split(",") if g.strip()}
        rows = [r for r in rows if r["observation_id"] in wanted]
    rows.sort(key=lambda r: (r["session_id"], r["observation_id"]))
    sessions = sorted({r["session_id"] for r in rows})
    print(f"[silver-infer] sessions {sessions}: {len(rows)} frames")
    items = [{"bucket": r["session_id"], "stem": r["observation_id"], "path": Path(r["image_path"]),
              "relpath": r["image_path"], "unit": r["observation_id"]} for r in rows]
    return items, out_root, {"population": "sessions", "groups_file": None, "keying": "session",
                             "sessions": sessions, "n_units": len(rows),
                             "frame_list": str(frame_list),
                             "domain_note": ("simulator/rendered imagery is OUT OF ADE20K's "
                                             "photographic training domain; the output remains a "
                                             "silver label and PROT-SKY-001 §5 requires a DEV "
                                             "inspection before any freeze names it primary")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N images")
    ap.add_argument("--groups", default="",
                    help="inspection: comma-separated group_ids (ecl_index) or observation_ids (sessions)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse label maps already on disk (identical by construction: same model "
                         "revision, same deterministic preprocessing) instead of recomputing them")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    scales = [int(s) for s in cfg["scales_short_side"]]
    repo_id, revision = cfg["model"]["repo_id"], cfg["model"]["revision"]
    threads = int(cfg.get("torch_threads", 4))

    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)

    mode = cfg.get("mode", "ecl_index")
    if mode == "ecl_index":
        items, out_root, extra = _plan_ecl(cfg, cfg_path, args)
    elif mode == "sessions":
        items, out_root, extra = _plan_sessions(cfg, cfg_path, args)
    else:
        raise SystemExit(f"mode must be 'ecl_index' or 'sessions', got {mode!r}")

    model = SegformerForSemanticSegmentation.from_pretrained(repo_id, revision=revision)
    model.eval()
    id2label = model.config.id2label
    if id2label[SKY_CLASS_INDEX] != "sky":
        raise SystemExit(f"class {SKY_CLASS_INDEX} is {id2label[SKY_CLASS_INDEX]!r}, not 'sky' — refusing")
    n_params = sum(p.numel() for p in model.parameters())

    out_root.mkdir(parents=True, exist_ok=True)
    records, times, done, seen = [], [], 0, set()
    for item in items:
        img = Image.open(item["path"]).convert("RGB")
        for short in scales:
            out = out_root / item["bucket"] / f"s{short}" / f"{item['stem']}.npz"
            if args.resume and out.exists():
                with np.load(out) as z:
                    lab = z["label"]
                dt, reused = None, True
            else:
                t0 = time.time()
                lab = label_map(model, img, IMG_SCALE_LONG, short)
                dt, reused = time.time() - t0, False
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, label=lab)
                times.append((short, dt))
            records.append({"group_id": item["stem"], "condition": item["bucket"],
                            "scale_short_side": short, "image_relpath": item["relpath"],
                            "width_px": img.size[0], "height_px": img.size[1],
                            "sky_fraction": float((lab == SKY_CLASS_INDEX).mean()),
                            "seconds": None if dt is None else round(dt, 3), "reused": reused})
        seen.add(item["unit"])
        done += 1
        if args.limit and done >= args.limit:
            break

    digest = hashlib.sha256()
    for rec in sorted(records, key=lambda x: (x["group_id"], x["condition"], x["scale_short_side"])):
        p = out_root / rec["condition"] / f"s{rec['scale_short_side']}" / f"{rec['group_id']}.npz"
        digest.update(np.load(p)["label"].tobytes())
    manifest = {
        "mode": mode,
        "infer_version": INFER_VERSION,
        "model": {"repo_id": repo_id, "revision": revision, "n_parameters": int(n_params),
                  "num_labels": len(id2label), "sky_class_index": SKY_CLASS_INDEX,
                  "sky_class_name": id2label[SKY_CLASS_INDEX]},
        "preprocessing": {"pipeline": "NVlabs SegFormer ADE20K test pipeline",
                          "img_scale_long": IMG_SCALE_LONG, "scales_short_side": scales,
                          "size_divisor": SIZE_DIVISOR, "resample": "PIL BILINEAR",
                          "mean": MEAN.tolist(), "std": STD.tolist(),
                          "logit_upsample": "bilinear to original size, align_corners=False, then argmax"},
        "runtime": {"torch": torch.__version__, "threads": threads, "device": "cpu",
                    "platform": platform.platform(),
                    "median_seconds_per_image": float(np.median([t for _, t in times])) if times else float("nan"),
                    "median_seconds_per_image_by_scale": {
                        f"s{sc}": float(np.median([t for s_, t in times if s_ == sc]))
                        for sc in scales if any(s_ == sc for s_, _ in times)},
                    "n_timed": len(times), "n_reused": sum(1 for r in records if r["reused"])},
        **extra,
        # ``groups`` is kept verbatim for the ECL path so a re-run's manifest stays comparable with
        # the EXP-SKY-005/006 ones already on disk.
        "counts": {"units": len(seen), "label_maps": len(records),
                   **({"groups": extra["n_units"]} if extra.get("keying") == "condition" else {})},
        "output_digest": digest.hexdigest(),
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "licence": ("Weights: NVIDIA Source Code License-NC (non-commercial research/evaluation only) — "
                    "not redistributed. Code: transformers, Apache-2.0. Training data: ADE20K, "
                    "non-commercial research/education. Derived label maps are not committed."),
    }
    (out_root / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (out_root / "inference_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"[silver-infer] {len(records)} label maps -> {out_root}")
    print(f"[silver-infer] median {manifest['runtime']['median_seconds_per_image']:.2f} s/image "
          f"over {manifest['runtime']['n_timed']} timed ({manifest['runtime']['n_reused']} reused); "
          f"per scale {manifest['runtime']['median_seconds_per_image_by_scale']}; "
          f"digest {manifest['output_digest'][:16]}")


if __name__ == "__main__":
    main()
