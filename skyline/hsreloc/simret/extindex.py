"""The derived index of the EXTENDED simulator batch (research R2 of the final-validation feature).

One row per observation, joining what the store knows (session metadata, poses, GT seam status)
with what the committed inventory recovered (path identity, swipe leg decomposition) and what the
human-confirmable sidecars declare (anchors, reference roles). The DEV/held-out split is **by
level** and nothing else — a test asserts exactly that — so the discipline boundary is readable
off any row.

Nothing here opens an image or a curve: the index is pose/metadata only, which is why building it
for the held-out levels before the freeze is legitimate (dataset construction), while every
experiment that *reads curves* goes through the population guard instead.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from hsreloc.observation import read_session
from hsreloc.simret import groups

INDEX_VERSION = "1.0.0"

INDEX_COLUMNS = [
    "level", "settings_level", "split", "session_id", "observation_id", "path_id", "kind",
    "anchor_id", "time_of_day", "clouds", "hour", "gt_status",
    "east_m", "north_m", "up_m", "yaw_deg", "pitch_deg", "roll_deg",
    "leg_id", "leg_axis", "leg_direction", "phase",
    "lateral_east_m", "longitudinal_north_m", "vertical_up_m", "total_displacement_m",
    "swipe_center_anchor_id", "swipe_center_anchor_distance_m", "hard_tag",
]

_FLOATS = ("hour", "east_m", "north_m", "up_m", "yaw_deg", "pitch_deg", "roll_deg",
           "lateral_east_m", "longitudinal_north_m", "vertical_up_m", "total_displacement_m",
           "swipe_center_anchor_distance_m")

ANCHOR_ASSIGN_RADIUS_M = 2.0


class ExtIndexError(Exception):
    """The extended index cannot be built as declared."""


def _resolve(base: Path, value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def build_index(cfg: dict, base: Path) -> dict:
    """Assemble the index from the store + committed inventory + sidecars. Read-only everywhere."""
    store = _resolve(base, cfg["store_root"])
    inv_dir = _resolve(base, cfg["inventory_dir"])
    swipes_inv = json.loads((inv_dir / "swipes.json").read_text(encoding="utf-8"))["swipes"]

    rows = []
    for level, lvl in cfg["levels"].items():
        split, task_key = lvl["split"], lvl["task_key"]
        anchors = groups.read_anchors(_resolve(base, cfg["anchors_csv"][task_key]))
        for sid in lvl["sessions"]:
            meta, observations = read_session(store / sid)
            sim = meta.get("simulator_run", {})
            path_id = (sim.get("raw_settings", {}).get("path", {}) or {}).get("path_id")
            if path_id is None:
                raise ExtIndexError(f"{sid}: session records no path_id — re-ingest the run")
            cond = meta.get("condition", {})
            is_swipe = path_id != "NoPath"
            sw = None
            if is_swipe:
                run_key = f"{level}/{sid}"
                if run_key not in swipes_inv:
                    raise ExtIndexError(f"{run_key}: swipe session absent from the committed "
                                        f"inventory decomposition")
                sw = swipes_inv[run_key]
                by_raw = {o["observation_id"]: o for o in sw["observations"]}
                c = sw["center"]
                center_anchor, center_dist = None, None
                for a in anchors:
                    d = math.hypot(c["east_m"] - a.east_m, c["north_m"] - a.north_m)
                    if center_dist is None or d < center_dist:
                        center_anchor, center_dist = a.anchor_id, d
            for obs in observations:
                gt_ok = (store / sid / "skylines_oracle" / f"{obs.observation_id}.csv").exists()
                row = {
                    "level": level, "settings_level": sim.get("level"), "split": split,
                    "session_id": sid, "observation_id": obs.observation_id,
                    "path_id": path_id, "kind": "swipe" if is_swipe else "anchor_capture",
                    "anchor_id": "", "time_of_day": cond.get("time_of_day"),
                    "clouds": cond.get("clouds"), "hour": cond.get("hour"),
                    "gt_status": "ok" if gt_ok else "insufficient_sky",
                    "east_m": obs.pos_east_m, "north_m": obs.pos_north_m, "up_m": obs.up_m,
                    "yaw_deg": obs.yaw_deg, "pitch_deg": obs.pitch_deg, "roll_deg": obs.roll_deg,
                    "leg_id": "", "leg_axis": "", "leg_direction": "", "phase": "",
                    "lateral_east_m": None, "longitudinal_north_m": None, "vertical_up_m": None,
                    "total_displacement_m": None, "swipe_center_anchor_id": "",
                    "swipe_center_anchor_distance_m": None,
                    "hard_tag": "hard" in path_id.lower(),
                }
                if is_swipe:
                    raw = obs.extra.get("sim_observation_id_raw") if obs.extra else None
                    o = by_raw.get(raw or "")
                    if o is None:
                        raise ExtIndexError(f"{sid}/{obs.observation_id}: raw id {raw!r} not in "
                                            f"the inventory decomposition")
                    row.update({
                        "leg_id": "" if o.get("leg_id") is None else o["leg_id"],
                        "phase": o["phase"],
                        "lateral_east_m": o["lateral_east_m"],
                        "longitudinal_north_m": o["longitudinal_north_m"],
                        "vertical_up_m": o["vertical_up_m"],
                        "total_displacement_m": o["total_displacement_m"],
                        "swipe_center_anchor_id": center_anchor,
                        "swipe_center_anchor_distance_m": center_dist,
                    })
                    if o.get("leg_id") is not None:
                        leg = sw["legs"][o["leg_id"]]
                        row["leg_axis"] = leg["axis"]
                        row["leg_direction"] = leg["direction"] or ""
                else:
                    aid, dist = groups.assign_anchor(obs, anchors)
                    if aid is None or dist > ANCHOR_ASSIGN_RADIUS_M:
                        raise ExtIndexError(
                            f"{sid}/{obs.observation_id}: no anchor within "
                            f"{ANCHOR_ASSIGN_RADIUS_M} m (nearest {aid!r} at {dist}) — the "
                            f"sidecar and the store disagree; fix the sidecar, never the data")
                    row["anchor_id"] = aid
                rows.append(row)

    _validate(cfg, rows)
    manifest = {
        "index_version": INDEX_VERSION,
        "split_rule": "by level and nothing else: " + json.dumps(
            {k: v["split"] for k, v in sorted(cfg["levels"].items())}, sort_keys=True),
        "anchor_assign_radius_m": ANCHOR_ASSIGN_RADIUS_M,
        "n_rows": len(rows),
        "n_anchor_captures": sum(1 for r in rows if r["kind"] == "anchor_capture"),
        "n_swipe_rows": sum(1 for r in rows if r["kind"] == "swipe"),
        "n_gt_holes": sum(1 for r in rows if r["gt_status"] != "ok"),
        "per_level": {lvl: {"rows": sum(1 for r in rows if r["level"] == lvl),
                            "anchors": len({r["anchor_id"] for r in rows
                                            if r["level"] == lvl and r["anchor_id"]}),
                            "split": cfg["levels"][lvl]["split"]}
                      for lvl in cfg["levels"]},
        "reads": "poses and metadata only — no image or curve is opened building this index",
    }
    return {"rows": rows, "manifest": manifest}


def _validate(cfg: dict, rows: list) -> None:
    for r in rows:
        want = cfg["levels"][r["level"]]["split"]
        if r["split"] != want:
            raise ExtIndexError(f"{r['session_id']}: split {r['split']!r} != level rule {want!r}")
    per_anchor: dict = {}
    for r in rows:
        if r["kind"] == "anchor_capture":
            per_anchor.setdefault((r["level"], r["anchor_id"]), []).append(r)
    bad = {k: len(v) for k, v in per_anchor.items() if len(v) != 9}
    if bad:
        raise ExtIndexError(f"anchors without exactly 9 condition captures: {bad}")
    for (lvl, aid), members in per_anchor.items():
        cells = {f"{m['time_of_day']}+{m['clouds']}" for m in members}
        if len(cells) != 9:
            raise ExtIndexError(f"{lvl}/{aid}: duplicate condition cells: {sorted(cells)}")


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return str(v)
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def write_index(index: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        for r in index["rows"]:
            w.writerow({k: _fmt(r.get(k)) for k in INDEX_COLUMNS})
    digest = hashlib.sha256((out_dir / "index.csv").read_bytes()).hexdigest()
    manifest = dict(index["manifest"], content_digest=digest)
    (out_dir / "index.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    return manifest


def load_index(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    if manifest.get("index_version") != INDEX_VERSION:
        raise ExtIndexError(f"index version {manifest.get('index_version')!r} != {INDEX_VERSION}")
    if hashlib.sha256((out_dir / "index.csv").read_bytes()).hexdigest() != manifest["content_digest"]:
        raise ExtIndexError(f"{out_dir}/index.csv does not match its recorded digest")
    rows = []
    with (out_dir / "index.csv").open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for k in _FLOATS:
                r[k] = None if r[k] == "" else float(r[k])
            r["leg_id"] = None if r["leg_id"] == "" else int(r["leg_id"])
            r["hard_tag"] = r["hard_tag"] == "True"
            rows.append(r)
    return {"rows": rows, "manifest": manifest}
