"""Measure MARS-LVIG candidate sequences *before* choosing one (`LIT-VO-004` §3, `DEC-VO-005`).

Two cheap probes, ~250 MB total, no imagery evaluated and no VO run:

- **RTK track** -> duration, path length, and height above the takeoff datum through the cruise.
  The cruise window comes from `naveval.ingest_mars_lvig.find_cruise_window`, the same function with
  the same rule that produced the committed `hkairport01-b` window, which it reproduces exactly when
  run on `HKairport01` (a built-in check: `--control HKairport01`).
- **LiDAR** -> height above *the imaged surface*, and the depth spread inside the camera footprint.

The second probe is the point of this script. `LIT-005` and `LIT-006` judged per-sequence altitude
constancy and terrain flatness from documentation and left both as open questions; MARS-LVIG ships a
ground-facing Livox Avia alongside the camera, so both are directly measurable. The distinction it
exposes -- **height above takeoff** (what the telemetry reports, constant to a few centimetres on
every sequence) versus **height above what is actually below the aircraft** (what sets the
pixel-to-metre factor, and varies by 1.26x to 8.8x depending on the sequence) -- is what decides
`DEC-VO-005`.

Both probes read the UAVScenes republication (`sijieaaa/UAVScenes`, CC BY-NC-SA 4.0) by HTTP range
request through the ZIP central directory, reusing `evaluation/tools/stage1/ziprange.py` unmodified.

    python evaluation/tools/exp_vo_007/probe_candidates.py \
        --scenes HKairport01 AMtown01 HKisland01 AMvalley01 \
        --target-agl 80 80 90 130 --cache-dir <scratch> --out <report.json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1"))

import ziprange as zr                                                # noqa: E402
from naveval.ingest_mars_lvig import (                               # noqa: E402
    find_cruise_window,
    ground_datum_m,
    load_rtk_track,
)

UAVSCENES_URL = (
    "https://huggingface.co/datasets/sijieaaa/UAVScenes/resolve/main/interval5_CAM_LIDAR.zip"
)
UAVSCENES_SIZE = 28_682_115_865
TAIL_LEN = 65_536

# Calibrated camera cone (LIT-006 "Camera calibration"): fx = fy = 1471.0653 on 2448 x 2048.
HALF_HFOV_RAD = np.arctan(1224.0 / 1471.0653)
HALF_VFOV_RAD = np.arctan(1024.0 / 1471.0653)

# `hkairport01-b`'s committed window, for the control check.
HKAIRPORT01_B_WINDOW = (1671606510.406, 1671607126.188)


def _entries(cache: Path) -> list[dict]:
    tail = cache / "tail.bin"
    if not tail.exists() or tail.stat().st_size != TAIL_LEN:
        zr.fetch_range(UAVSCENES_URL, UAVSCENES_SIZE - TAIL_LEN, UAVSCENES_SIZE - 1, tail)
    cd_off, cd_size = zr.parse_eocd(tail.read_bytes(), UAVSCENES_SIZE - TAIL_LEN)
    cd = cache / "cd.bin"
    if not cd.exists() or cd.stat().st_size != cd_size:
        zr.fetch_range(UAVSCENES_URL, cd_off, cd_off + cd_size - 1, cd)
    return zr.parse_central_directory(cd.read_bytes())


def rtk_probe(cache: Path, entries: list[dict], scene: str, target_agl_m: float) -> dict:
    name = f"interval5_CAM_LIDAR/interval5_{scene}/rtk_positions_raw.csv"
    entry = next(e for e in entries if e["name"] == name)
    csv_path = cache / f"{scene}_rtk.csv"
    if not csv_path.exists():
        zr.extract_entry(UAVSCENES_URL, entry, csv_path, cache / "tmp.bin")
    track = load_rtk_track(csv_path, expect_scene=scene)
    datum = ground_datum_m(track)
    t0, t1 = find_cruise_window(track, target_agl_m)
    sel = [s for s in track if t0 <= s.timestamp_s <= t1]
    e = np.array([s.easting for s in sel])
    n = np.array([s.northing for s in sel])
    agl = np.array([s.alt_m for s in sel]) - datum
    whole_t = np.array([s.timestamp_s for s in track])
    return {
        "scene": scene,
        "rtk_csv": str(csv_path),
        "n_rtk_whole": len(track),
        "duration_whole_s": round(float(whole_t[-1] - whole_t[0]), 1),
        "ground_datum_m": round(datum, 3),
        "cruise_window_utc": [round(t0, 3), round(t1, 3)],
        "cruise_duration_s": round(t1 - t0, 1),
        "cruise_path_m": round(float(np.hypot(np.diff(e), np.diff(n)).sum()), 1),
        "frames_at_10hz": int((t1 - t0) * 10),
        "agl_above_takeoff_mean_m": round(float(agl.mean()), 3),
        "agl_above_takeoff_sd_m": round(float(agl.std()), 3),
        "agl_above_takeoff_ptp_m": round(float(agl.max() - agl.min()), 3),
    }


def lidar_probe(cache: Path, entries: list[dict], scene: str, n_clouds: int,
                cruise: tuple[float, float]) -> dict:
    """Height above the imaged surface, and in-footprint depth spread.

    Livox Avia clouds are in the sensor frame with `+x` along the ground-facing boresight, so for a
    near-nadir mount `x` *is* the vertical drop to each return. Points outside the *camera's* cone
    are discarded: the LiDAR's field of view is wider, and what matters is the depth structure the
    planar-homography model actually has to absorb.
    """
    prefix = f"interval5_CAM_LIDAR/interval5_{scene}/interval5_LIDAR/"
    lid = sorted((e for e in entries if e["name"].startswith(prefix) and e["name"].endswith(".txt")),
                 key=lambda e: e["name"])
    if not lid:
        raise SystemExit(f"No LiDAR entries for {scene}")
    ts = np.array([float(re.search(r"image(\d+\.\d+)_lidar", e["name"]).group(1)) for e in lid])
    out_dir = cache / f"lidar_{scene}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in np.linspace(0, len(lid) - 1, n_clouds).astype(int):
        e = lid[int(i)]
        p = out_dir / Path(e["name"]).name
        if not p.exists():
            zr.extract_entry(UAVSCENES_URL, e, p, cache / "tmp.bin")
        a = np.loadtxt(p)
        x, y, z = a[:, 0], a[:, 1], a[:, 2]
        m = x > 1.0
        x, y, z = x[m], y[m], z[m]
        inside = ((np.abs(np.arctan2(y, x)) < HALF_HFOV_RAD)
                  & (np.abs(np.arctan2(z, x)) < HALF_VFOV_RAD))
        xi = x[inside]
        if len(xi) < 500:
            continue
        rows.append((float(ts[int(i)]), float(np.median(xi)),
                     float(np.percentile(xi, 95) - np.percentile(xi, 5))))
    r = np.array(sorted(rows))
    t, height, spread = r[:, 0], r[:, 1], r[:, 2]
    c = (t >= cruise[0]) & (t <= cruise[1])
    if c.sum() < 10:
        c = np.ones_like(t, dtype=bool)
    h, s = height[c], spread[c]
    p5, p95 = float(np.percentile(h, 5)), float(np.percentile(h, 95))
    return {
        "n_clouds_sampled": int(len(r)), "n_in_cruise": int(c.sum()),
        "height_above_surface_median_m": round(float(np.median(h)), 1),
        "height_above_surface_min_m": round(float(h.min()), 1),
        "height_above_surface_max_m": round(float(h.max()), 1),
        "height_above_surface_sd_m": round(float(h.std()), 2),
        "height_p5_m": round(p5, 1), "height_p95_m": round(p95, 1),
        "height_p95_over_p5": round(p95 / p5, 2),
        "height_max_over_min": round(float(h.max() / h.min()), 2),
        "footprint_depth_spread_p95_p5_m": round(float(np.median(s)), 1),
        "footprint_depth_spread_frac_of_height": round(float(np.median(s) / np.median(h)), 3),
        # Raw series, because `EXP-VO-007` splits its results by effective-height regime and the
        # regime boundaries must come from dataset metadata frozen before any VO run.
        "height_profile": {
            "t": [round(float(v), 3) for v in t],
            "height_m": [round(float(v), 2) for v in height],
            "footprint_spread_m": [round(float(v), 2) for v in spread],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe MARS-LVIG candidates before selecting one.")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--target-agl", nargs="+", type=float, required=True)
    ap.add_argument("--n-clouds", type=int, default=70)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", default="HKairport01",
                    help="sequence whose committed cruise window must be reproduced exactly")
    args = ap.parse_args()
    if len(args.scenes) != len(args.target_agl):
        raise SystemExit("--scenes and --target-agl must have the same length")

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    entries = _entries(cache)
    report = {"uavscenes_url": UAVSCENES_URL, "sequences": {}}
    for scene, agl in zip(args.scenes, args.target_agl):
        rec = rtk_probe(cache, entries, scene, agl)
        rec["lidar"] = lidar_probe(cache, entries, scene, args.n_clouds,
                                   tuple(rec["cruise_window_utc"]))
        report["sequences"][scene] = rec
        print(json.dumps(rec, indent=2), flush=True)

    ctrl = report["sequences"].get(args.control)
    if ctrl is not None:
        got = tuple(ctrl["cruise_window_utc"])
        ok = all(abs(a - b) < 1e-3 for a, b in zip(got, HKAIRPORT01_B_WINDOW))
        report["control_check"] = {
            "scene": args.control, "expected": list(HKAIRPORT01_B_WINDOW),
            "got": list(got), "reproduces_committed_window": ok,
        }
        print(f"\ncontrol {args.control}: committed window reproduced = {ok}", flush=True)
        if not ok:
            Path(args.out).write_text(json.dumps(report, indent=2))
            return 1
    Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
