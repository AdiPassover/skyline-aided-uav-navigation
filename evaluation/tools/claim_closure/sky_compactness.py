"""EXP-SKY-014 Q-D: measured compactness of the adopted skyline reference representation, against
the source imagery it is derived from (claim-closure Phase 2D).

    python evaluation/tools/claim_closure/sky_compactness.py --out evaluations/claim-closure-2026-09/sky

Measures: PNG sizes of every skyline view image and GT sky mask in the dual-direction batch (read
only), the adopted in-memory descriptor (256 float64 per view), its float32 form, the
on-disk INT exchange CSV, the trusted-memory sizes reached in EXP-INT-001/003, and the EXP-SKY-011
dense memories as database totals. No generalisation beyond "relative to the source imagery".
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DUAL_BATCH = REPO / "simulator_skyline_data_both_directions"
DESCRIPTOR_SAMPLES = 256


def file_sizes(pattern: str) -> np.ndarray:
    return np.asarray([os.path.getsize(p) for p in glob.glob(pattern, recursive=True)], dtype=np.float64)


def qt(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)), "min": float(x.min()),
            "max": float(x.max()), "total": float(x.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "evaluations/claim-closure-2026-09/sky")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"experiment": "EXP-SKY-014 Q-D", "source_batch": str(DUAL_BATCH), "measured": {}}

    # --- source imagery and GT masks (dual-direction batch, 512x512, both views) --------------------
    imgs = file_sizes(str(DUAL_BATCH / "**" / "skyline" / "images" / "*.png"))
    masks = file_sizes(str(DUAL_BATCH / "**" / "skyline" / "sim" / "*_sky.png"))
    res["measured"]["view_image_png_bytes"] = qt(imgs)
    res["measured"]["gt_sky_mask_png_bytes"] = qt(masks)
    per_level = {}
    for level_dir in sorted(p for p in DUAL_BATCH.iterdir() if p.is_dir()):
        per_level[level_dir.name] = {"view_image_png": qt(file_sizes(str(level_dir / "**" / "skyline" / "images" / "*.png"))),
                                     "gt_mask_png": qt(file_sizes(str(level_dir / "**" / "skyline" / "sim" / "*_sky.png")))}
    res["measured"]["per_level"] = per_level
    # image geometry from one settings.json
    w = h = None
    for s in glob.glob(str(DUAL_BATCH / "**" / "settings.json"), recursive=True)[:1]:
        d = json.loads(Path(s).read_text(encoding="utf-8"))
        cam = d.get("north_camera") or {}
        w, h = cam.get("width_px"), cam.get("height_px")
    res["measured"]["image_px"] = {"width": w, "height": h, "raw_rgb_bytes": (w * h * 3) if w and h else None}

    # --- the adopted representation ------------------------------------------------------------------
    desc64 = DESCRIPTOR_SAMPLES * 8
    res["representation"] = {
        "descriptor_float64_bytes_per_view": desc64, "descriptor_float64_bytes_dual": 2 * desc64,
        "descriptor_float32_bytes_per_view": DESCRIPTOR_SAMPLES * 4, "descriptor_float32_bytes_dual": 2 * DESCRIPTOR_SAMPLES * 4,
        "stored_pose_bytes_dual_reference": 2 * desc64 + 3 * 8 + 8 + 4,   # two profiles + East/North/heading + timestamp + id
        "note": "SkylineDescriptor holds 256 float64 per view (org.boofcv.relocalization.SkylineDescriptor); a TrustedReference adds its stored pose",
    }
    # --- INT exchange CSV (text, both views, full precision) -----------------------------------------
    csv_rows = []
    for p in sorted(glob.glob(str(REPO / "datasets" / "*" / "skyline_profiles.csv"))):
        n = sum(1 for _ in open(p, encoding="utf-8")) - 1
        if n > 0:
            csv_rows.append({"dataset": Path(p).parent.name, "bytes": os.path.getsize(p), "captures": n,
                             "bytes_per_capture": os.path.getsize(p) / n})
    res["int_exchange_csv"] = csv_rows
    # --- memory sizes actually reached in the INT experiments (run manifests) ------------------------
    mem = []
    for p in sorted(glob.glob(str(REPO / "runs" / "exp-int-00*-int-c0*" / "relocalization_manifest.json"))):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        size = None
        for k in ("memory", "trusted_memory", "reference_memory"):
            v = d.get(k)
            if isinstance(v, dict):
                size = v.get("size") or v.get("references") or v.get("n_references")
        if size is None:
            # fall back: count reference_inserted rows in the sidecar
            ev = Path(p).parent / "alignment_events.csv"
            if ev.exists():
                with ev.open(newline="", encoding="utf-8") as f:
                    size = sum(1 for r in csv.DictReader(f) if r.get("kind") == "reference_inserted")
        mem.append({"run": Path(p).parent.name, "references": size,
                    "bytes_float64_dual": (size or 0) * 2 * desc64, "bytes_float32_dual": (size or 0) * 2 * DESCRIPTOR_SAMPLES * 4})
    res["int_trusted_memories"] = mem
    # --- EXP-SKY-011 dense memories as database totals ------------------------------------------------
    idx = REPO / "evaluations/sky-dual/index.csv"
    dense = {}
    if idx.exists():
        rows = list(csv.DictReader(idx.open(newline="", encoding="utf-8")))
        for lv in ("village", "mountains", "city"):
            n = 0
            for r in rows:
                if r["level_key"] != lv:
                    continue
                vu = r.get("vertical_up_m", "")
                vert = r.get("leg_axis") == "vertical"
                if r["segment_kind"] == "anchor_capture" or (not vert and (vu in ("", "nan") or abs(float(vu)) < 1.0)):
                    n += 1
            dense[lv] = {"references": n, "bytes_float64_dual": n * 2 * desc64, "bytes_float32_dual": n * 2 * DESCRIPTOR_SAMPLES * 4,
                         "source_images_bytes_dual_estimate": n * 2 * float(np.median(imgs)) if imgs.size else None}
    res["exp_sky_011_dense_memories"] = dense
    # --- ratios ---------------------------------------------------------------------------------------
    if imgs.size:
        med = float(np.median(imgs))
        res["ratios"] = {"view_png_median_over_descriptor_float64": med / desc64,
                         "view_png_median_over_descriptor_float32": med / (DESCRIPTOR_SAMPLES * 4),
                         "raw_rgb_over_descriptor_float64": (w * h * 3) / desc64 if w and h else None,
                         "gt_mask_png_median_over_descriptor_float64": float(np.median(masks)) / desc64 if masks.size else None,
                         "int_csv_bytes_per_capture_over_descriptor_float64_dual": (np.median([r["bytes_per_capture"] for r in csv_rows]) / (2 * desc64)) if csv_rows else None}
    res["scaling"] = {"storage": "linear in N by construction (a list of fixed-size descriptors; no index)",
                      "retrieval": "exact exhaustive per view: O(N · 256) multiply-adds per query per view; timing measured in Phase 4"}
    (out / "compactness.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    lines = ["# EXP-SKY-014 Q-D — compactness (generated by sky_compactness.py)", "",
             f"Source batch: `{DUAL_BATCH}` (read-only). Image {w}×{h} RGB.", "",
             "| quantity | bytes | n |", "|---|---|---|"]
    m = res["measured"]
    lines.append(f"| view image PNG, median (mean) | {m['view_image_png_bytes'].get('median', 0):,.0f} ({m['view_image_png_bytes'].get('mean', 0):,.0f}) | {m['view_image_png_bytes'].get('n', 0)} |")
    lines.append(f"| GT sky-mask PNG, median | {m['gt_sky_mask_png_bytes'].get('median', 0):,.0f} | {m['gt_sky_mask_png_bytes'].get('n', 0)} |")
    lines.append(f"| raw RGB pixels | {m['image_px']['raw_rgb_bytes']:,} | — |")
    lines.append(f"| descriptor, 256 float64, one view | {desc64:,} | — |")
    lines.append(f"| descriptor, two views | {2 * desc64:,} | — |")
    lines.append(f"| descriptor, 256 float32, two views | {2 * DESCRIPTOR_SAMPLES * 4:,} | — |")
    if csv_rows:
        lines.append(f"| INT exchange CSV per capture (text, two views), median over datasets | {np.median([r['bytes_per_capture'] for r in csv_rows]):,.0f} | {len(csv_rows)} datasets |")
    lines.append("")
    if "ratios" in res:
        r = res["ratios"]
        lines.append(f"Ratios: view PNG / descriptor(float64) = **{r['view_png_median_over_descriptor_float64']:.0f}×**; "
                     f"view PNG / descriptor(float32) = {r['view_png_median_over_descriptor_float32']:.0f}×; raw RGB / descriptor = {r['raw_rgb_over_descriptor_float64']:.0f}×; "
                     f"GT mask PNG / descriptor = {r['gt_mask_png_median_over_descriptor_float64']:.1f}×; INT CSV per capture / dual descriptor = {r['int_csv_bytes_per_capture_over_descriptor_float64_dual']:.1f}×.")
    lines.append("\nPer level (view image PNG median bytes): " + "; ".join(f"{k}: {v['view_image_png'].get('median', 0):,.0f} (n={v['view_image_png'].get('n', 0)})" for k, v in per_level.items()))
    lines.append("\nTrusted memories reached in the INT experiments (references → dual float64 bytes): " +
                 "; ".join(f"{x['run']}: {x['references']} → {x['bytes_float64_dual']:,}" for x in mem))
    lines.append("\nEXP-SKY-011 dense memories: " + "; ".join(f"{k}: {v['references']} refs → {v['bytes_float64_dual']:,} B (float64 dual) vs ≈ {v['source_images_bytes_dual_estimate']:,.0f} B of source PNGs" for k, v in dense.items()))
    (out / "compactness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
