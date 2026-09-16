"""Inventory and structure recovery for a directory of raw simulator runs (read-only).

The first real simulator batch (2026-09-02) was collected manually: a handful of intentionally
chosen anchor positions rendered under several TIME_OF_DAY x CLOUD_STATE conditions, plus
translation sweeps flown away from some anchors. Nothing in the export records which runs belong
together — and nothing here guesses from folder names or folder order. Everything below is derived
from ``settings.json`` and the logged poses, through the same adapter that will later ingest the
runs, so a run this inventory calls valid is an ingestible run.

Read-only by construction: the raw directory is opened for reading only; all output goes to
``--out``. Outputs:

* ``inventory.csv`` / ``inventory.json`` — one row per run: identity, environment, pose extents,
  spacing, GT-mask status, validation status;
* ``clusters.json`` — position clusters over the runs' *first* poses (single-linkage at a declared
  radius), with the member runs and the conditions each cluster carries. Clusters are *candidate*
  anchors: the human confirms them by authoring ``anchors.csv`` (hsreloc.simret.groups), which is
  the only thing that makes them experimental groups;
* ``classification.json`` — per multi-observation run: net/path displacement, its East (lateral),
  North (longitudinal) and Up components w.r.t. a world-locked-North camera, attitude variation,
  the dominant axis and a conservative class (lateral / longitudinal / vertical / mixed), and the
  nearest cluster to the run's start. Single-observation runs are classified as condition captures
  of their cluster. Duplicates (same cluster, same condition, > 1 run) are listed, never resolved.

Usage (from ``skyline/``, ``PYTHONPATH=.``)::

    python scripts/sim_inventory.py --root <raw runs dir> --out <output dir> \
        [--cluster-radius-m 2.0] [--start-anchor-radius-m 25.0] [--dominance 0.9]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from hsreloc.simret import adapter, simgt
from hsreloc.simret.conventions import SimConventions, ue_attitude_to_enu, ue_position_to_enu

INVENTORY_VERSION = "1.0.0"

INVENTORY_COLUMNS = [
    "run_folder", "run_id", "level", "time_of_day", "hour", "clouds", "time_speed",
    "automatic_capture", "capture_distance_m", "width_px", "height_px", "hfov_deg", "vfov_deg",
    "heading_mode", "n_observations",
    "first_east_m", "first_north_m", "first_up_m", "last_east_m", "last_north_m", "last_up_m",
    "east_span_m", "north_span_m", "up_span_m", "net_displacement_m", "path_length_m",
    "yaw_span_deg", "pitch_span_deg", "roll_span_deg",
    "spacing_median_m", "spacing_min_m", "spacing_max_m",
    "gt_ok", "gt_holes", "gt_worst_intermediate_frac", "sky_frac_min", "sky_frac_max",
    "validation_ok", "n_errors", "n_warnings",
]


def _pose_rows(run: adapter.SimRun, conv: SimConventions) -> list:
    out = []
    for r in run.rows:
        e, n, u = ue_position_to_enu(float(r["north_ue_x_cm"]), float(r["north_ue_y_cm"]),
                                     float(r["north_ue_z_cm"]), conv)
        yaw, pitch, roll = ue_attitude_to_enu(float(r["north_ue_yaw_deg"]),
                                              float(r["north_ue_pitch_deg"]),
                                              float(r["north_ue_roll_deg"]), conv)
        out.append({"observation_id": r["observation_id"], "east_m": e, "north_m": n, "up_m": u,
                    "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
                    "sim_time_s": float(r["sim_time_s"])})
    return out


def _gt_status(run: adapter.SimRun) -> dict:
    """Decode every GT mask through the adapter's own conversion; counts only, nothing written."""
    ok = holes = 0
    worst_intermediate = 0.0
    sky_fracs = []
    for r in run.rows:
        p = run.run_dir / r["sim_sky_mask_path"]
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            holes += 1
            continue
        try:
            sky, rep = simgt.sky_mask_from_image(raw)
        except simgt.SimGtError:
            holes += 1
            worst_intermediate = max(worst_intermediate, simgt.binarity(raw)["intermediate_frac"])
            continue
        worst_intermediate = max(worst_intermediate, rep["intermediate_frac"])
        gt = simgt.sim_gt_curve(sky)
        sky_fracs.append(gt["sky_frac_raw"])
        if gt["status"] == simgt.STATUS_OK:
            ok += 1
        else:
            holes += 1
    return {"gt_ok": ok, "gt_holes": holes, "gt_worst_intermediate_frac": worst_intermediate,
            "sky_frac_min": min(sky_fracs) if sky_fracs else None,
            "sky_frac_max": max(sky_fracs) if sky_fracs else None}


def _span(values: list) -> float:
    return float(max(values) - min(values)) if values else 0.0


def inventory_run(run_dir: Path, conv: SimConventions) -> dict:
    run = adapter.read_run(run_dir)
    report = adapter.validate_run(run, conv)
    poses = _pose_rows(run, conv)
    e = [p["east_m"] for p in poses]
    n = [p["north_m"] for p in poses]
    u = [p["up_m"] for p in poses]
    steps = [math.hypot(e[i + 1] - e[i], n[i + 1] - n[i]) for i in range(len(poses) - 1)]
    steps3 = [math.hypot(steps[i], u[i + 1] - u[i]) for i in range(len(steps))]
    env = run.settings.get("environment", {})
    cam = run.camera
    sky = run.settings.get("skyline", {}) or {}
    row = {
        "run_folder": run_dir.name, "run_id": run.run_id, "level": run.settings.get("level"),
        "time_of_day": env.get("time_of_day"), "hour": env.get("hour"),
        "clouds": env.get("clouds"), "time_speed": env.get("time_speed"),
        "automatic_capture": sky.get("automatic_capture"),
        "capture_distance_m": sky.get("capture_distance_m"),
        "width_px": cam.get("width_px"), "height_px": cam.get("height_px"),
        "hfov_deg": cam.get("horizontal_fov_deg"), "vfov_deg": cam.get("vertical_fov_deg"),
        "heading_mode": cam.get("heading_mode"), "n_observations": len(poses),
        "first_east_m": e[0], "first_north_m": n[0], "first_up_m": u[0],
        "last_east_m": e[-1], "last_north_m": n[-1], "last_up_m": u[-1],
        "east_span_m": _span(e), "north_span_m": _span(n), "up_span_m": _span(u),
        "net_displacement_m": math.hypot(math.hypot(e[-1] - e[0], n[-1] - n[0]), u[-1] - u[0]),
        "path_length_m": float(sum(steps3)),
        "yaw_span_deg": _span([p["yaw_deg"] for p in poses]),
        "pitch_span_deg": _span([p["pitch_deg"] for p in poses]),
        "roll_span_deg": _span([p["roll_deg"] for p in poses]),
        "spacing_median_m": float(np.median(steps3)) if steps3 else None,
        "spacing_min_m": min(steps3) if steps3 else None,
        "spacing_max_m": max(steps3) if steps3 else None,
        "validation_ok": report.ok, "n_errors": len(report.errors),
        "n_warnings": len(report.warnings),
    }
    row.update(_gt_status(run))
    return {"row": row, "poses": poses, "errors": list(report.errors),
            "warnings": list(report.warnings)}


def cluster_first_poses(items: list, radius_m: float) -> list:
    """Single-linkage clusters over each run's FIRST pose (horizontal). Deliberately simple: the
    same-pose condition captures should coincide to well under a metre, so any sane radius finds
    the same clusters; the radius is declared in the output either way."""
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]
            if math.hypot(a["first_east_m"] - b["first_east_m"],
                          a["first_north_m"] - b["first_north_m"]) <= radius_m:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    clusters = []
    for members in groups.values():
        rows = [items[i] for i in members]
        e = [r["first_east_m"] for r in rows]
        nn = [r["first_north_m"] for r in rows]
        uu = [r["first_up_m"] for r in rows]
        conds = sorted({f"{r['time_of_day']}+{r['clouds']}" for r in rows})
        clusters.append({
            "center_east_m": float(np.mean(e)), "center_north_m": float(np.mean(nn)),
            "center_up_m": float(np.mean(uu)),
            "spread_m": float(max(math.hypot(x - np.mean(e), y - np.mean(nn))
                                  for x, y in zip(e, nn))),
            "n_runs": len(rows),
            "n_single_observation_runs": sum(1 for r in rows if r["n_observations"] == 1),
            "runs": sorted(r["run_folder"] for r in rows),
            "conditions": conds,
        })
    clusters.sort(key=lambda c: (-c["n_single_observation_runs"], c["center_east_m"]))
    for k, c in enumerate(clusters):
        c["cluster_id"] = f"cluster_{k:02d}"
    return clusters


def classify_runs(items: list, poses_by_run: dict, clusters: list, start_radius_m: float,
                  dominance: float) -> dict:
    """Geometry-only classification. A run is lateral / longitudinal / vertical when one axis holds
    at least ``dominance`` of the net 3-D displacement; otherwise mixed_or_ambiguous. Single-
    observation runs are condition captures of their cluster (or unanchored)."""
    by_cluster = {c["cluster_id"]: c for c in clusters}
    # A cluster is an anchor CANDIDATE only if it carries at least one single-observation
    # (condition) capture; a cluster formed solely by a translation run's own start pose is not
    # a place anyone declared, merely where a sweep happened to begin.
    candidates = [c for c in clusters if c["n_single_observation_runs"] > 0]
    condition_captures, translation_runs, ambiguous, unanchored = [], [], [], []
    for it in items:
        start_cluster, start_dist = None, None
        for c in candidates:
            d = math.hypot(it["first_east_m"] - c["center_east_m"],
                           it["first_north_m"] - c["center_north_m"])
            if d <= start_radius_m and (start_dist is None or d < start_dist):
                start_cluster, start_dist = c["cluster_id"], d
        entry = {"run_folder": it["run_folder"], "time_of_day": it["time_of_day"],
                 "clouds": it["clouds"], "n_observations": it["n_observations"],
                 "start_cluster": start_cluster,
                 "start_distance_m": None if start_dist is None else round(start_dist, 3)}
        if it["n_observations"] == 1:
            (condition_captures if start_cluster else unanchored).append(entry)
            continue
        poses = poses_by_run[it["run_folder"]]
        de = poses[-1]["east_m"] - poses[0]["east_m"]
        dn = poses[-1]["north_m"] - poses[0]["north_m"]
        du = poses[-1]["up_m"] - poses[0]["up_m"]
        net = math.sqrt(de * de + dn * dn + du * du)
        comp = {"lateral_east_m": de, "longitudinal_north_m": dn, "vertical_up_m": du}
        entry.update({k: round(v, 3) for k, v in comp.items()})
        entry["net_displacement_m"] = round(net, 3)
        entry["attitude_span_deg"] = {
            "yaw": round(it["yaw_span_deg"], 4), "pitch": round(it["pitch_span_deg"], 4),
            "roll": round(it["roll_span_deg"], 4)}
        if net <= 0:
            entry["class"] = "mixed_or_ambiguous"
            entry["reason"] = "zero net displacement over multiple observations"
            ambiguous.append(entry)
            continue
        fracs = {"lateral": abs(de) / net, "longitudinal": abs(dn) / net, "vertical": abs(du) / net}
        axis = max(fracs, key=fracs.get)
        entry["dominance_frac"] = round(fracs[axis], 4)
        if fracs[axis] >= dominance:
            entry["class"] = axis
            sign = {"lateral": de, "longitudinal": dn, "vertical": du}[axis]
            entry["direction"] = {("lateral", True): "east", ("lateral", False): "west",
                                  ("longitudinal", True): "north", ("longitudinal", False): "south",
                                  ("vertical", True): "up", ("vertical", False): "down"}[(axis, sign >= 0)]
            translation_runs.append(entry)
        else:
            entry["class"] = "mixed_or_ambiguous"
            entry["reason"] = (f"dominant axis {axis} holds only {fracs[axis]:.2%} of the net "
                               f"displacement (threshold {dominance:.0%})")
            ambiguous.append(entry)

    # duplicates: > 1 single-observation run with the same cluster AND the same condition
    seen: dict = {}
    for e in condition_captures:
        seen.setdefault((e["start_cluster"], e["time_of_day"], e["clouds"]), []).append(e["run_folder"])
    duplicates = [{"cluster": k[0], "time_of_day": k[1], "clouds": k[2], "runs": v}
                  for k, v in sorted(seen.items()) if len(v) > 1]

    # the recovered condition matrix per cluster
    matrix: dict = {}
    for e in condition_captures:
        matrix.setdefault(e["start_cluster"], {}).setdefault(f"{e['time_of_day']}+{e['clouds']}", 0)
        matrix[e["start_cluster"]][f"{e['time_of_day']}+{e['clouds']}"] += 1
    return {
        "rule": {"start_anchor_radius_m": start_radius_m, "dominance": dominance,
                 "classes": "single obs -> condition capture; multi obs -> lateral(E/W) / "
                            "longitudinal(N/S) / vertical(U/D) when one axis >= dominance of the "
                            "net 3-D displacement, else mixed_or_ambiguous; folder names and order "
                            "are never used"},
        "condition_captures": condition_captures,
        "translation_runs": translation_runs,
        "ambiguous_runs": ambiguous,
        "unanchored_runs": unanchored,
        "duplicate_condition_runs": duplicates,
        "condition_matrix_by_cluster": {k: dict(sorted(v.items())) for k, v in sorted(matrix.items())},
        "n_clusters": len(by_cluster),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="directory containing raw simulator run folders")
    ap.add_argument("--out", required=True, help="output directory for the inventory records")
    ap.add_argument("--cluster-radius-m", type=float, default=2.0)
    ap.add_argument("--start-anchor-radius-m", type=float, default=25.0)
    ap.add_argument("--dominance", type=float, default=0.9)
    args = ap.parse_args(argv)

    root = Path(args.root)
    run_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "settings.json").exists())
    if not run_dirs:
        print(f"no run folders (with settings.json) under {root}", file=sys.stderr)
        return 1

    conv = SimConventions()
    items, poses_by_run, problems = [], {}, []
    for rd in run_dirs:
        try:
            res = inventory_run(rd, conv)
        except adapter.SimRunError as exc:
            problems.append({"run_folder": rd.name, "error": str(exc)})
            continue
        items.append(res["row"])
        poses_by_run[rd.name] = res["poses"]
        if res["errors"]:
            problems.append({"run_folder": rd.name, "errors": res["errors"]})

    clusters = cluster_first_poses(items, args.cluster_radius_m)
    classification = classify_runs(items, poses_by_run, clusters,
                                   args.start_anchor_radius_m, args.dominance)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "inventory.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INVENTORY_COLUMNS)
        w.writeheader()
        for r in items:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in INVENTORY_COLUMNS})
    (out / "inventory.json").write_text(json.dumps({
        "inventory_version": INVENTORY_VERSION, "root": str(root),
        "n_run_folders": len(run_dirs), "n_valid": sum(1 for r in items if r["validation_ok"]),
        "cluster_radius_m": args.cluster_radius_m, "runs": items,
        "unreadable_or_invalid": problems,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "clusters.json").write_text(json.dumps({
        "rule": f"single-linkage over each run's FIRST pose, horizontal ENU distance <= "
                f"{args.cluster_radius_m} m; clusters are CANDIDATE anchors until a human confirms "
                f"them in anchors.csv",
        "clusters": clusters}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "classification.json").write_text(json.dumps(classification, indent=2, sort_keys=True)
                                             + "\n", encoding="utf-8")

    print(f"[sim_inventory] {len(run_dirs)} run folders; "
          f"{sum(1 for r in items if r['validation_ok'])} valid, {len(problems)} with problems")
    print(f"[sim_inventory] {len(clusters)} position clusters "
          f"({sum(1 for c in clusters if c['n_single_observation_runs'] > 0)} with condition captures)")
    print(f"[sim_inventory] {len(classification['translation_runs'])} translation runs, "
          f"{len(classification['ambiguous_runs'])} ambiguous, "
          f"{len(classification['duplicate_condition_runs'])} duplicate condition cells")
    print(f"[sim_inventory] outputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
