"""How closed is each possible prefix of a frozen acquisition window? (`EXP-VO-007`)

Written because a truncated acquisition raised a question the pre-registration had not anticipated,
and the answer is available from ground-truth metadata alone — no imagery, no VO, so consulting it
cannot be a result-driven choice.

**Why closure and not just length.** `MARS-LVIG` survey flights are lawn-mower patterns: they pass
near their own start every few minutes. So a prefix is not simply "a shorter version of the flight"
— depending on where it stops it is either a near-closed loop or an open path ending several hundred
metres from its origin, and those are different *shapes*, not different *lengths*:

- **endpoint error** is a pre-registered metric, and on a closed path it measures drift back to a
  known place while on an open path it measures something else entirely;
- a **single global Sim(2) alignment** takes more leverage from the ends of an open path than from a
  loop, where errors partly average out.

Comparing a truncated open prefix against `hkairport01-b` — itself a near-closed pattern, 19 m over
1,848 m, 1.0 % of path — would therefore introduce a confound that has nothing to do with the
readout under test. This script makes that visible before the choice is made rather than after.

Usage::

    python evaluation/tools/exp_vo_007/prefix_closure.py \
        --rtk-csv <UAVScenes rtk_positions_raw.csv> --scene AMtown01 \
        --t-start 1658137128.011 --t-end 1658138317.359 --subwindow-s 45 \
        --compare datasets/hkairport01-b --out evaluations/exp-vo-007/prefix_closure.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parents[2]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcap_source import iter_subwindows                                  # noqa: E402
from naveval.ingest_mars_lvig import load_rtk_track                      # noqa: E402


def _shape(east: np.ndarray, north: np.ndarray) -> dict:
    e, n = east - east[0], north - north[0]
    path = float(np.hypot(np.diff(e), np.diff(n)).sum())
    off = float(np.hypot(e[-1], n[-1]))
    return {
        "path_m": round(path, 1),
        "endpoint_offset_m": round(off, 1),
        "endpoint_offset_pct_of_path": round(100 * off / path, 2) if path else None,
        "extent_east_m": round(float(e.max() - e.min()), 1),
        "extent_north_m": round(float(n.max() - n.min()), 1),
    }


def reference_shape(dataset_dir: Path) -> dict:
    with (dataset_dir / "groundtruth.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    return _shape(np.array([float(r["east_m"]) for r in rows]),
                  np.array([float(r["north_m"]) for r in rows]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtk-csv", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--t-start", type=float, required=True)
    ap.add_argument("--t-end", type=float, required=True)
    ap.add_argument("--subwindow-s", type=float, default=45.0)
    ap.add_argument("--compare", default="", help="dataset dir whose shape to report alongside")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    track = load_rtk_track(a.rtk_csv, expect_scene=a.scene)
    t = np.array([s.timestamp_s for s in track])
    e = np.array([s.easting for s in track])
    n = np.array([s.northing for s in track])
    windows = list(iter_subwindows(a.t_start, a.t_end, a.subwindow_s))

    prefixes = []
    for k, (_lo, hi) in enumerate(windows, 1):
        t_end = hi / 1e9
        m = (t >= a.t_start) & (t <= t_end)
        if m.sum() < 3:
            continue
        rec = {"n_subwindows": k, "t_end": round(t_end, 3),
               "flight_s": round(t_end - a.t_start, 1), **_shape(e[m], n[m])}
        prefixes.append(rec)

    report = {"scene": a.scene, "frozen_window": [a.t_start, a.t_end],
              "subwindow_s": a.subwindow_s, "n_subwindows_frozen": len(windows),
              "prefixes": prefixes}
    if a.compare:
        d = Path(a.compare) if Path(a.compare).is_absolute() else REPO / a.compare
        report["comparison"] = {"dataset": a.compare, **reference_shape(d)}

    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"{'N':>3} {'flight s':>9} {'path m':>9} {'endpoint m':>11} {'% of path':>10}")
    for p in prefixes:
        print(f"{p['n_subwindows']:3d} {p['flight_s']:9.0f} {p['path_m']:9.0f} "
              f"{p['endpoint_offset_m']:11.0f} {p['endpoint_offset_pct_of_path']:10.1f}")
    if "comparison" in report:
        c = report["comparison"]
        print(f"\ncomparison {c['dataset']}: path {c['path_m']:.0f} m, endpoint "
              f"{c['endpoint_offset_m']:.0f} m ({c['endpoint_offset_pct_of_path']:.1f} % of path)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
