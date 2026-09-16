"""Compare a Java runtime `metric_track.csv` against the offline Python reconstruction of the SAME run.

This is the end-to-end half of the Python/Java agreement evidence. Where
`MetricReadoutPythonParityTest` feeds both implementations one frozen set of increments, this script
takes a run the Java estimator actually produced — real imagery, real tracking, real RANSAC, the
runtime height channel fed sample by sample — and re-derives the metric trajectory offline from that
run's own `logical_transform.csv` and the dataset's height column. If the two disagree, the runtime
port is not doing what `EXP-VO-012`/`EXP-VO-014` validated.

It also reports the three things the demonstration has to show, from the run's own files rather than
from an assertion:

  * the raw pose stayed in image units and is unchanged from the committed pixel baseline;
  * the metric pose is in metres and was produced online;
  * the conversion scale followed the aircraft's altitude, causally.

    PY=.venv/Scripts/python.exe
    $PY evaluation/tools/vo_metric_runtime/compare_runtime_metric.py \
        --run runs/uevo-fig8-vary-homography-metric-v1 \
        --dataset datasets/uevo-fig8-vary-v1 \
        --baseline runs/uevo-fig8-vary-homography-rigid-v1
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVALUATION_DIR = HERE.parents[1]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP012 = EVALUATION_DIR / "tools" / "exp_vo_012"
mr = _load("vo_metric_runtime_cmp_readout", EXP012 / "metric_readout.py")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def col(rows, key, cast=float):
    return np.array([cast(r[key]) for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory holding metric_track.csv")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--baseline", default=None,
                    help="committed pixel-valued run to check the raw pose against")
    ap.add_argument("--pos-tol-m", type=float, default=1e-6)
    args = ap.parse_args()

    run = REPO / args.run if not Path(args.run).is_absolute() else Path(args.run)
    ds = REPO / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)

    manifest = json.loads((run / "manifest.json").read_text())
    cfg = manifest["estimator_config"]["metric_readout"]
    f_working = float(cfg["f_working_px"])
    h0 = float(cfg["h0_agl_m"])
    meta = json.loads((ds / "dataset.json").read_text())
    intr = meta["metadata"]["camera_intrinsics"]

    print(f"run            {manifest['run_id']}")
    print(f"dataset        {meta['dataset_id']}  ({meta['source_type']}, {meta['evidence_tier']})")
    print(f"h0             {h0} m   (source: {meta['metadata'].get('h0_source', 'n/a')})")
    print(f"f_working      {f_working} px  = fx_native {cfg['fx_native_px']} / "
          f"downsampleFactor {manifest['estimator_config']['downsampleFactor']}")
    print(f"published unit {cfg['navigation_units']}  "
          f"(navigation_source {manifest['estimator_config']['navigation_source']})")

    # ---------------------------------------------------------------- offline reconstruction
    inc = mr.load_increments(run, int(intr["width"]), int(intr["height"]))
    terrain = read_csv(ds / "terrain.csv")
    by_index = {int(r["frame_index"]): r for r in terrain}
    rel = np.array([float(by_index[int(i)][cfg["height_relative_column"]]) for i in inc.frame_index])
    h_agl = mr.h_agl_from_baro(h0, rel)
    offline = mr.integrate_metric(inc, h_agl, f_working, arm="baro")

    # ---------------------------------------------------------------- the runtime's own output
    track = read_csv(run / "metric_track.csv")
    if len(track) != len(inc):
        raise SystemExit(f"metric_track.csv has {len(track)} rows, increments have {len(inc)}")
    east = col(track, "metric_east_m")
    north = col(track, "metric_north_m")
    yaw = col(track, "metric_yaw_deg")
    gsd = np.array([float(r["gsd_m_per_px"]) if r["gsd_m_per_px"] not in ("", "NaN") else np.nan
                    for r in track])
    h_used = np.array([float(r["h_used_m"]) if r["h_used_m"] else np.nan for r in track])
    raw_x = col(track, "raw_x_px")
    raw_y = col(track, "raw_y_px")
    status = [r["h_status"] for r in track]

    d_east = np.abs(east - offline.east_m)
    d_north = np.abs(north - offline.north_m)
    d_yaw = np.abs((yaw - offline.yaw_deg + 180.0) % 360.0 - 180.0)
    # The runtime logs the height/gsd the LAST increment used, i.e. the reference frame's.
    h_ref = np.concatenate([[np.nan], offline.h_used_m[:-1]])
    gsd_ref = np.concatenate([[np.nan], offline.gsd_m_per_px[:-1]])
    ok = np.isfinite(h_used) & np.isfinite(h_ref)
    d_h = np.abs(h_used[ok] - h_ref[ok])
    d_gsd = np.abs(gsd[ok] - gsd_ref[ok])

    print("\n--- Python <-> Java, end to end -------------------------------------------------")
    print(f"frames                         {len(track)}")
    print(f"max |d east|                   {d_east.max():.3e} m")
    print(f"max |d north|                  {d_north.max():.3e} m")
    print(f"max |d yaw|                    {d_yaw.max():.3e} deg")
    print(f"max |d height used|            {d_h.max():.3e} m")
    print(f"max |d gsd used|               {d_gsd.max():.3e} m/px")
    passed = (max(d_east.max(), d_north.max()) <= args.pos_tol_m
              and d_h.max() <= 1e-9 and d_gsd.max() <= 1e-12)
    print(f"PARITY                         {'PASS' if passed else 'FAIL'} "
          f"(position tolerance {args.pos_tol_m:g} m)")

    # ---------------------------------------------------------------- units and causality
    print("\n--- units ------------------------------------------------------------------------")
    print(f"raw pose      end (x, y)       ({raw_x[-1]:.3f}, {raw_y[-1]:.3f}) px")
    print(f"metric pose   end (E, N)       ({east[-1]:.3f}, {north[-1]:.3f}) m")
    print(f"ratio |metric| / |raw|         {np.hypot(east[-1], north[-1]) / np.hypot(raw_x[-1], raw_y[-1]):.6f} "
          f"m/px   (mean gsd {np.nanmean(gsd):.6f} m/px)")

    print("\n--- the conversion scale follows altitude, causally --------------------------------")
    print(f"h_AGL used     min / max        {np.nanmin(h_used):.3f} / {np.nanmax(h_used):.3f} m "
          f"(x{np.nanmax(h_used) / np.nanmin(h_used):.3f})")
    print(f"gsd used       min / max        {np.nanmin(gsd):.6f} / {np.nanmax(gsd):.6f} m/px "
          f"(x{np.nanmax(gsd) / np.nanmin(gsd):.3f})")
    # gsd_k must be exactly h_{k-1}/f -- a per-frame identity, not a correlation.
    resid = np.abs(gsd[ok] - h_used[ok] / f_working)
    print(f"max |gsd - h_used/f_working|   {resid.max():.3e} m/px   (an identity, not a fit)")
    counts = {s: status.count(s) for s in sorted(set(status)) if s}
    print(f"height status counts            {counts}")
    print(f"segments                        {int(col(track, 'segment_index', int).max()) + 1}, "
          f"{int(col(track, 'gap_before_segment', int).sum())} frame(s) marked as following an "
          f"UNKNOWN gap")

    # ---------------------------------------------------------------- the raw path is unchanged
    if args.baseline:
        base = REPO / args.baseline if not Path(args.baseline).is_absolute() else Path(args.baseline)
        b = read_csv(base / "logical_transform.csv")
        bx = col(b, "rigid_x")
        by = col(b, "rigid_y")
        n = min(len(bx), len(raw_x))
        dr = np.hypot(raw_x[:n] - bx[:n], raw_y[:n] - by[:n])
        print("\n--- the raw visual path is unchanged ----------------------------------------------")
        print(f"baseline                       {base.name}")
        print(f"max |d raw pose|               {dr.max():.3e} px over {n} frames")
        print(f"RAW BIT-IDENTITY               {'PASS' if dr.max() == 0.0 else 'FAIL'}")
        passed = passed and dr.max() == 0.0

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
