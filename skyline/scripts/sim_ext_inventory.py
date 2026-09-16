"""Inventory and structure recovery for the EXTENDED simulator batch (read-only).

The extended batch (2026-09-03) is organised as ``<root>/<level>/Run_*``, and differs from the
first batch structurally: a single ``NoPath`` run holds ONE observation per nominal anchor position
(all anchors of the level, teleported captures under one TIME x CLOUD condition), and translation
experiments are star-swipe runs whose ``settings.json`` carries a ``path.path_id``. Nothing here
trusts folder names, folder order, or the author's verbal description; every recovered structure is
derived from ``settings.json`` and the logged poses through the same adapter that will later ingest
the runs.

Read-only by construction. Outputs (to ``--out``):

* ``inventory.csv`` / ``inventory.json`` — one row per run (per-run validation, environment,
  ``path_id``, pose extents, spacing, GT-mask status), plus the level-folder vs settings-level
  names, recorded verbatim, never repaired;
* ``anchors.json`` — per level: single-linkage clusters over the NoPath runs' *observations*
  (horizontal ENU). Each cluster lists its members (run, observation id, condition, pose), the
  recovered TIME x CLOUD condition matrix, missing cells, duplicate cells, and the worst pairwise
  position mismatch between conditions. Clusters are CANDIDATE anchors until a human confirms them
  in an ``anchors.csv`` sidecar;
* ``swipes.json`` — per multi-observation (path) run: the recovered center, per-observation
  displacement components (lateral=East / longitudinal=North / vertical=Up for the world-locked-
  North camera), the mechanical leg decomposition (center-crossing segmentation, dominant-axis
  labelling, outbound/inbound from the leg's own extremum), attitude differences, and explicit
  ambiguity flags where the geometry does not decompose cleanly — preserved, never forced;
* ``raw_digest.txt`` — a content digest over every file under the root (order-independent).

Usage (from ``skyline/``, ``PYTHONPATH=.``)::

    python scripts/sim_ext_inventory.py --root <extended raw root> --out <output dir> \
        [--anchor-radius-m 2.0] [--center-radius-m 15.0] [--dominance 0.9]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from hsreloc.simret import adapter
from hsreloc.simret.conventions import SimConventions
from scripts.sim_inventory import INVENTORY_COLUMNS, inventory_run

EXT_INVENTORY_VERSION = "1.0.0"
EXT_COLUMNS = ["level_folder", "path_id"] + INVENTORY_COLUMNS


def _cluster(points: list, radius_m: float) -> list:
    """Single-linkage over horizontal ENU positions; returns lists of member indices."""
    n = len(points)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]) <= radius_m:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda m: min(m))


def recover_anchors(level: str, nopath_obs: list, radius_m: float) -> dict:
    """Cluster every NoPath observation of a level; report the condition matrix per cluster."""
    pts = [(o["east_m"], o["north_m"]) for o in nopath_obs]
    conditions_present = sorted({f"{o['time_of_day']}+{o['clouds']}" for o in nopath_obs})
    clusters = []
    for members in _cluster(pts, radius_m):
        rows = [nopath_obs[i] for i in members]
        e = [r["east_m"] for r in rows]
        nn = [r["north_m"] for r in rows]
        uu = [r["up_m"] for r in rows]
        ce, cn = float(np.mean(e)), float(np.mean(nn))
        worst = max((math.hypot(a["east_m"] - b["east_m"], a["north_m"] - b["north_m"])
                     for i, a in enumerate(rows) for b in rows[i + 1:]), default=0.0)
        cell: dict = {}
        for r in rows:
            cell.setdefault(f"{r['time_of_day']}+{r['clouds']}", []).append(
                f"{r['run_folder']}/{r['observation_id']}")
        clusters.append({
            "center_east_m": ce, "center_north_m": cn, "center_up_m": float(np.mean(uu)),
            "up_span_m": float(max(uu) - min(uu)),
            "worst_pairwise_horizontal_m": worst,
            "n_observations": len(rows),
            "cells": {k: sorted(v) for k, v in sorted(cell.items())},
            "missing_cells": sorted(set(conditions_present) - set(cell)),
            "duplicate_cells": sorted(k for k, v in cell.items() if len(v) > 1),
        })
    clusters.sort(key=lambda c: (c["center_east_m"], c["center_north_m"]))
    for k, c in enumerate(clusters):
        c["anchor_id"] = f"{level}_a{k:02d}"
    seps = [math.hypot(a["center_east_m"] - b["center_east_m"],
                       a["center_north_m"] - b["center_north_m"])
            for i, a in enumerate(clusters) for b in clusters[i + 1:]]
    return {"level": level, "conditions_present": conditions_present,
            "n_anchors": len(clusters), "min_anchor_separation_m": min(seps) if seps else None,
            "anchors": clusters}


def decompose_swipe(poses: list, center_radius_m: float, dominance: float) -> dict:
    """Mechanical star-leg decomposition from pose geometry alone.

    The center is the run's first pose (the star swipes begin at their center; this is *verified*,
    not assumed, by requiring every recovered leg to end back inside ``center_radius_m``). An
    observation is 'at center' when its horizontal+vertical distance to the center is below the
    radius; maximal stretches of off-center observations are legs. A leg's axis comes from the
    dominant displacement component at its extremum (>= ``dominance`` of the 3-D displacement);
    outbound = up to and including the extremum, inbound after. Anything that does not fit is
    flagged, never forced.
    """
    c = poses[0]
    obs = []
    for i, p in enumerate(poses):
        de = p["east_m"] - c["east_m"]
        dn = p["north_m"] - c["north_m"]
        du = p["up_m"] - c["up_m"]
        obs.append({
            "observation_id": p["observation_id"], "index": i,
            "lateral_east_m": round(de, 3), "longitudinal_north_m": round(dn, 3),
            "vertical_up_m": round(du, 3),
            "total_displacement_m": round(math.sqrt(de * de + dn * dn + du * du), 3),
            "yaw_diff_deg": round(p["yaw_deg"] - c["yaw_deg"], 4),
            "pitch_diff_deg": round(p["pitch_deg"] - c["pitch_deg"], 4),
            "roll_diff_deg": round(p["roll_deg"] - c["roll_deg"], 4),
        })
    at_center = [o["total_displacement_m"] <= center_radius_m for o in obs]
    legs, flags = [], []
    i = 0
    while i < len(obs):
        if at_center[i]:
            i += 1
            continue
        j = i
        while j < len(obs) and not at_center[j]:
            j += 1
        seg = obs[i:j]
        ext = max(range(len(seg)), key=lambda k: seg[k]["total_displacement_m"])
        ep = seg[ext]
        net = ep["total_displacement_m"]
        fr = {"lateral": abs(ep["lateral_east_m"]) / net,
              "longitudinal": abs(ep["longitudinal_north_m"]) / net,
              "vertical": abs(ep["vertical_up_m"]) / net}
        axis = max(fr, key=fr.get)
        sign = {"lateral": ep["lateral_east_m"], "longitudinal": ep["longitudinal_north_m"],
                "vertical": ep["vertical_up_m"]}[axis]
        direction = {("lateral", True): "east", ("lateral", False): "west",
                     ("longitudinal", True): "north", ("longitudinal", False): "south",
                     ("vertical", True): "up", ("vertical", False): "down"}[(axis, sign >= 0)]
        clean = fr[axis] >= dominance
        returned = j < len(obs)  # the leg ends because the trajectory re-entered the center
        leg = {"leg_id": len(legs), "first_index": i, "last_index": j - 1,
               "extremum_index": i + ext, "extremum_displacement_m": net,
               "axis": axis if clean else "mixed_or_ambiguous", "direction": direction if clean else None,
               "dominance_frac": round(fr[axis], 4), "returns_to_center": returned,
               "n_outbound": ext + 1, "n_inbound": len(seg) - ext - 1}
        if not clean:
            flags.append(f"leg {leg['leg_id']}: dominant axis {axis} holds only {fr[axis]:.2%} "
                         f"of the extremum displacement (threshold {dominance:.0%})")
        if not returned:
            flags.append(f"leg {leg['leg_id']}: run ends without returning to center")
        legs.append(leg)
        for k, o in enumerate(seg):
            o["leg_id"] = leg["leg_id"]
            o["phase"] = "outbound" if k <= ext else "inbound"
        i = j
    for o in obs:
        o.setdefault("leg_id", None)
        o.setdefault("phase", "center")
    axes_seen = sorted({(l["axis"], l["direction"]) for l in legs if l["direction"]})
    return {"center": {k: round(c[k], 3) for k in ("east_m", "north_m", "up_m")},
            "center_radius_m": center_radius_m, "n_legs": len(legs),
            "leg_directions": [f"{a}:{d}" for a, d in axes_seen],
            "legs": legs, "flags": flags, "observations": obs}


def _digest(root: Path) -> str:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return f"{h.hexdigest()}  {len(files)} files"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchor-radius-m", type=float, default=2.0)
    ap.add_argument("--center-radius-m", type=float, default=15.0)
    ap.add_argument("--dominance", type=float, default=0.9)
    ap.add_argument("--skip-digest", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root)
    level_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    conv = SimConventions()
    items, problems = [], []
    nopath_by_level: dict = {}
    swipes = {}
    for lvl in level_dirs:
        for rd in sorted(p for p in lvl.iterdir() if p.is_dir() and (p / "settings.json").exists()):
            try:
                res = inventory_run(rd, conv)
            except adapter.SimRunError as exc:
                problems.append({"run_folder": f"{lvl.name}/{rd.name}", "error": str(exc)})
                continue
            row = res["row"]
            settings = json.loads((rd / "settings.json").read_text(encoding="utf-8"))
            row["level_folder"] = lvl.name
            row["path_id"] = settings.get("path", {}).get("path_id")
            row["run_folder"] = f"{lvl.name}/{rd.name}"
            items.append(row)
            if res["errors"]:
                problems.append({"run_folder": row["run_folder"], "errors": res["errors"]})
            if row["path_id"] == "NoPath":
                for p in res["poses"]:
                    nopath_by_level.setdefault(lvl.name, []).append({
                        **{k: p[k] for k in ("observation_id", "east_m", "north_m", "up_m")},
                        "run_folder": row["run_folder"], "time_of_day": row["time_of_day"],
                        "clouds": row["clouds"]})
            else:
                swipes[row["run_folder"]] = {
                    "path_id": row["path_id"], "time_of_day": row["time_of_day"],
                    "clouds": row["clouds"],
                    **decompose_swipe(res["poses"], args.center_radius_m, args.dominance)}

    anchors = {lvl: recover_anchors(lvl, obs, args.anchor_radius_m)
               for lvl, obs in sorted(nopath_by_level.items())}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "inventory.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXT_COLUMNS)
        w.writeheader()
        for r in items:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in EXT_COLUMNS})
    (out / "inventory.json").write_text(json.dumps({
        "inventory_version": EXT_INVENTORY_VERSION, "root": str(root),
        "n_runs": len(items), "n_valid": sum(1 for r in items if r["validation_ok"]),
        "runs": items, "unreadable_or_invalid": problems,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "anchors.json").write_text(json.dumps({
        "rule": f"single-linkage over every NoPath observation's horizontal ENU position, "
                f"radius {args.anchor_radius_m} m; clusters are CANDIDATE anchors until confirmed "
                f"in an anchors.csv sidecar; folder names and order are never used",
        "levels": anchors}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "swipes.json").write_text(json.dumps({
        "rule": f"center = first pose, verified by every leg re-entering radius "
                f"{args.center_radius_m} m; legs = maximal off-center stretches; axis = dominant "
                f"component at the leg extremum (>= {args.dominance:.0%}), outbound/inbound split "
                f"at the extremum; ambiguity flagged, never forced",
        "swipes": swipes}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.skip_digest:
        (out / "raw_digest.txt").write_text(_digest(root) + "\n", encoding="utf-8")

    print(f"[sim_ext_inventory] {len(items)} runs, "
          f"{sum(1 for r in items if r['validation_ok'])} valid, {len(problems)} with problems")
    for lvl, a in anchors.items():
        full = sum(1 for c in a["anchors"] if not c["missing_cells"] and not c["duplicate_cells"])
        print(f"[sim_ext_inventory] {lvl}: {a['n_anchors']} anchors "
              f"({full} complete 9-condition), min separation "
              f"{a['min_anchor_separation_m'] and round(a['min_anchor_separation_m'], 1)} m")
    flagged = {k: v["flags"] for k, v in swipes.items() if v["flags"]}
    print(f"[sim_ext_inventory] {len(swipes)} swipe runs, {len(flagged)} with flags")
    for k, fl in flagged.items():
        for f_ in fl:
            print(f"  {k}: {f_}")
    print(f"[sim_ext_inventory] outputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
