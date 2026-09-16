#!/usr/bin/env python
"""EXP-SKY-011 — dual-direction (North + West) skyline place recognition with the frozen C1-32.

Stages (each reads the previous one's outputs; every stage is re-runnable)::

    audit    raw batch -> reconstructed star-swipe segments, teleport boundaries (60 Hz confirmed),
             site identities from the EXP-SKY-008 catalogue, deterministic North/West pairing plan,
             the plan-view figure; writes <out_dir>/index.csv + audit.json. Reads poses only.
    ingest   one spec-006 session per (run, view) into store_root through the unchanged adapter
             (view option), then verifies every physical observation has both views at the same
             position and time; refreshes index.csv with GT-curve status per view.
    score    North-only and West-only retrieval outcomes per query against the sparse anchor memory
             (leave-site-out variant included) and the dense path memory (hold-out radii), per curve
             source; the paired raw score matrices are saved so every fusion rule is computed from
             the same numbers.
    report   single-view baselines, the dual rules, hard negatives, radii, temporal comparison,
             density, the relative-position secondary, compute cost; metrics.json + report.md +
             figures.

The matcher is the frozen ``BoundedLagNccMatcher(max_lag_samples=32)`` through
``hsreloc.simret.recog.ReferenceBank`` (lag exact, score to 1e-12 of ``match()``; cross-checked in
run). Nothing in ``hsreloc/retrieval``, ``hsreloc/matchers`` or INT is modified. The run-level
``path_id`` is raw provenance only: star-swipe identity comes from the trajectory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from hsreloc.observation import read_session                                    # noqa: E402
from hsreloc.simret import adapter, recog                                       # noqa: E402
from hsreloc.simret.conventions import SimConventions, ue_position_to_enu       # noqa: E402
from sim_ext_inventory import decompose_swipe                                   # noqa: E402
from sky_relpose_study import (LEVEL_COLOR, LEVEL_ORDER, _f, _json, _open,      # noqa: E402
                               _plt, _read_csv, _resolve, _save)

STUDY_VERSION = "1.0.0"
VIEWS = ("north", "west")


def _load_cfg(path: Path) -> tuple:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return cfg, path.parent


def _write_csv(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def session_id_for(run_id: str, view: str) -> str:
    return f"{run_id}-{view}"


def observation_id_for(run_id: str, view: str, raw_id: str) -> str:
    return f"{session_id_for(run_id, view)}__{raw_id}"


# ==================================================================================================
# audit
# ==================================================================================================

def _raw_runs(cfg: dict, base: Path) -> list:
    root = _resolve(base, cfg["raw_root"])
    out = []
    for level_dir, lvl in cfg["levels"].items():
        for rd in sorted((root / level_dir).glob("Run_*")):
            if (rd / "settings.json").exists():
                out.append((level_dir, lvl, rd))
    if not out:
        raise SystemExit(f"no runs under {root}")
    return out


def _read_raw(run_dir: Path) -> tuple:
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    obs_path = next(run_dir / rel for rel in adapter.OBSERVATIONS_LOCATIONS if (run_dir / rel).exists())
    with obs_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    trace_path = run_dir / "vo" / "groundtruth.csv"
    trace = []
    if trace_path.exists():
        with trace_path.open("r", encoding="utf-8", newline="") as f:
            trace = [(float(r["sim_time_s"]), float(r["east_m"]), float(r["north_m"]), float(r["up_m"]))
                     for r in csv.DictReader(f)]
    return settings, rows, trace


def _positions(rows: list, conv: SimConventions) -> tuple:
    """ENU positions from the North columns; the West columns must be the same physical point."""
    P, T, worst = [], [], 0.0
    for r in rows:
        e, n, u = ue_position_to_enu(float(r["north_ue_x_cm"]), float(r["north_ue_y_cm"]),
                                     float(r["north_ue_z_cm"]), conv)
        if "west_ue_x_cm" in r:
            for a, b in (("north_ue_x_cm", "west_ue_x_cm"), ("north_ue_y_cm", "west_ue_y_cm"),
                         ("north_ue_z_cm", "west_ue_z_cm")):
                worst = max(worst, abs(float(r[a]) - float(r[b])) / 100.0)
        P.append((e, n, u))
        T.append(float(r["sim_time_s"]))
    return np.asarray(P, dtype=np.float64), np.asarray(T, dtype=np.float64), worst


def segment_by_discontinuity(P: np.ndarray, threshold_m: float) -> list:
    """``[start, end)`` index ranges split where a consecutive step exceeds ``threshold_m`` in the
    horizontal or the vertical. Deterministic; no other information is used."""
    segs, start = [], 0
    for i in range(1, len(P)):
        dxy = math.hypot(P[i, 0] - P[i - 1, 0], P[i, 1] - P[i - 1, 1])
        dz = abs(P[i, 2] - P[i - 1, 2])
        if dxy > threshold_m or dz > threshold_m:
            segs.append((start, i))
            start = i
    segs.append((start, len(P)))
    return segs


def confirm_with_trace(trace: list, T: np.ndarray, segs: list, threshold_m: float) -> dict:
    """Every reconstructed boundary must coincide with a tick-level jump on the 60 Hz trace inside
    the window between the two observations, and no tick-level jump may fall inside a segment."""
    if not trace:
        return {"available": False}
    if len(set(np.round(T, 6))) == 1:
        return {"available": True, "applicable": False,
                "reason": "every capture shares one sim_time_s (manual captures with the clock frozen); "
                          "the trace cannot bracket the boundaries — each capture is its own segment"}
    eps = 0.01                                    # half a 60 Hz tick, for the equal-endpoint case
    jumps = []
    for k in range(1, len(trace)):
        d = math.hypot(trace[k][1] - trace[k - 1][1], trace[k][2] - trace[k - 1][2])
        dz = abs(trace[k][3] - trace[k - 1][3])
        if d > threshold_m or dz > threshold_m:
            jumps.append({"tick": k, "t": trace[k][0], "dxy_m": d, "dz_m": dz})
    boundaries = [s[0] for s in segs[1:]]
    confirmed, unconfirmed = [], []
    for i in boundaries:
        lo, hi = T[i - 1], T[i]
        inside = [j for j in jumps if lo - eps < j["t"] <= hi + eps]
        (confirmed if inside else unconfirmed).append({"obs_index": int(i), "t_lo": lo, "t_hi": hi,
                                                      "trace_jumps": inside})
    # a trace jump strictly inside a segment (between its first and last observation) is a defect
    inside_segment = []
    for s0, s1 in segs:
        if s1 - s0 < 2:
            continue
        lo, hi = T[s0], T[s1 - 1]
        inside_segment += [j for j in jumps if lo + eps < j["t"] < hi - eps]
    tick_d = np.array([math.hypot(trace[k][1] - trace[k - 1][1], trace[k][2] - trace[k - 1][2])
                       for k in range(1, len(trace))]) if len(trace) > 1 else np.zeros(0)
    return {"available": True, "applicable": True, "n_ticks": len(trace), "tick_step_median_m": float(np.median(tick_d)) if tick_d.size else None,
            "tick_step_p999_m": float(np.percentile(tick_d, 99.9)) if tick_d.size else None,
            "n_trace_jumps": len(jumps), "n_boundaries": len(boundaries),
            "n_confirmed": len(confirmed), "unconfirmed": unconfirmed,
            "trace_jumps_inside_a_segment": inside_segment,
            "before_first_observation": [j for j in jumps if j["t"] <= T[0]]}


def _star_center(P: np.ndarray, radius: float, min_revisits: int = 3) -> int | None:
    """Index of the star centre: the pose with the most *other* observations inside the centre
    radius (a star passes its centre once per leg), earliest on ties. ``None`` when no pose is
    revisited at least ``min_revisits`` times — the segment never comes back to anywhere."""
    n = len(P)
    if n < 2:
        return None
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
    counts = (d <= radius).sum(axis=1) - 1
    best = int(np.argmax(counts))                   # argmax returns the first maximum
    return best if counts[best] >= min_revisits else None


def _decompose_from_hub(poses: list, hub: int, center_radius_m: float, dominance: float) -> dict:
    """``decompose_swipe`` measures every displacement from ``poses[0]``; the star's hub is the pose
    with the most revisits, which need not be the first frame (the village stars begin 10 m north
    of their hub, the city star is entered from a transit). The hub is prepended as the reference
    pose and dropped again from the result, so leg bookkeeping stays in observation order."""
    dec = decompose_swipe([poses[hub]] + poses, center_radius_m, dominance)
    dec["observations"] = dec["observations"][1:]
    for o in dec["observations"]:
        o["index"] -= 1
    for leg in dec["legs"]:
        for key in ("first_index", "last_index", "extremum_index"):
            leg[key] -= 1
    dec["hub_index"] = hub
    dec["hub_observation_id"] = poses[hub]["observation_id"]
    return dec


def classify_segment(P: np.ndarray, ids: list, seg: tuple, cfg_seg: dict) -> list:
    """One reconstructed segment -> one or two typed pieces: ``anchor_capture`` (a single
    observation), ``star_swipe`` (the EXP-SKY-008 star decomposition succeeds around the hub),
    ``transit`` (a stretch that never returns to anywhere). A run that flies to a hub before
    starting the star is split into ``transit`` + ``star_swipe`` at the first observation inside
    the centre radius of the hub."""
    s0, s1 = seg
    n = s1 - s0
    if n == 1:
        return [{"start": s0, "end": s1, "kind": "anchor_capture", "decomposition": None}]
    sub = P[s0:s1]
    cr = float(cfg_seg["center_radius_m"])
    hub = _star_center(sub, cr)
    if hub is None:
        return [{"start": s0, "end": s1, "kind": "transit", "decomposition": None}]
    near_hub = np.hypot(sub[:, 0] - sub[hub, 0], sub[:, 1] - sub[hub, 1]) <= cr
    star_at = int(np.argmax(near_hub))               # first observation inside the hub's radius
    pieces = []
    if star_at > 0:                                  # the approach to the hub is a transit
        pieces.append({"start": s0, "end": s0 + star_at, "kind": "transit", "decomposition": None})
    poses = [{"observation_id": ids[k], "east_m": float(P[k, 0]), "north_m": float(P[k, 1]),
              "up_m": float(P[k, 2]), "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
             for k in range(s0 + star_at, s1)]
    dec = _decompose_from_hub(poses, hub - star_at, cr, float(cfg_seg["dominance"]))
    legs = dec["legs"]
    returning = [l for l in legs if l["returns_to_center"]]
    ok = len(legs) >= 4 and len(returning) >= len(legs) - 1 and all(l["direction"] for l in legs)
    pieces.append({"start": s0 + star_at, "end": s1, "kind": "star_swipe" if ok else "unclassified",
                   "decomposition": dec})
    return pieces


def _catalogue(cfg: dict, base: Path) -> dict:
    """The EXP-SKY-008 sites: anchors per level key, and every swipe's centre / anchor / hard tag."""
    anchors = {}
    for key, rel in cfg["old_anchors_csv"].items():
        anchors[key] = [{"anchor_id": r["anchor_id"], "east_m": float(r["east_m"]),
                         "north_m": float(r["north_m"]), "up_m": float(r["up_m"])}
                        for r in _read_csv(_resolve(base, rel))]
    old = _read_csv(_resolve(base, cfg["old_index_csv"]))
    key_of = {lvl["old_level"]: lvl["key"] for lvl in cfg["levels"].values()}
    swipes = defaultdict(list)
    for r in old:
        if r["kind"] != "swipe":
            continue
        swipes[(key_of[r["level"]], r["session_id"])].append(r)
    sites = []
    for (key, sid), rows in swipes.items():
        c = min(rows, key=lambda r: (float(r["total_displacement_m"] or 0.0), r["observation_id"]))
        # every centre-phase pass of the old swipe: the new hub is matched against the closest one
        # (the old decomposition measured from the run-start frame, 10 m north of the hub)
        passes = [(float(r["east_m"]), float(r["north_m"])) for r in rows if r["phase"] == "center"]
        sites.append({"level_key": key, "old_session": sid, "path_id": c["path_id"],
                      "anchor_id": c["swipe_center_anchor_id"], "hard_tag": c["hard_tag"] == "True",
                      "east_m": float(c["east_m"]), "north_m": float(c["north_m"]),
                      "up_m": float(c["up_m"]), "center_passes": passes})
    return {"anchors": anchors, "swipe_sites": sites}


def _nearest_site(sites: list, e: float, n: float) -> tuple:
    best, bd = None, None
    for s in sites:
        d = min(math.hypot(pe - e, pn - n) for pe, pn in s["center_passes"]) if s["center_passes"] else             math.hypot(s["east_m"] - e, s["north_m"] - n)
        if bd is None or d < bd:
            best, bd = s, d
    return best, bd


def _nearest(items: list, e: float, n: float) -> tuple:
    best, bd = None, None
    for a in items:
        d = math.hypot(a["east_m"] - e, a["north_m"] - n)
        if bd is None or d < bd:
            best, bd = a, d
    return best, bd


def stage_audit(cfg: dict, base: Path) -> None:
    out = _resolve(base, cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    conv = SimConventions.from_dict(cfg.get("conventions"))
    seg_cfg = cfg["segmentation"]
    cat = _catalogue(cfg, base)
    index_rows, audit = [], {"study_version": STUDY_VERSION, "runs": [], "catalogue": {
        k: len(v) for k, v in cat["anchors"].items()}, "n_old_swipe_sites": len(cat["swipe_sites"])}
    n_seg_total = 0
    for level_dir, lvl, rd in _raw_runs(cfg, base):
        settings, rows, trace = _read_raw(rd)
        run_id = settings["run_id"]
        key = lvl["key"]
        cap = float(settings.get("skyline", {}).get("capture_distance_m", 10.0))
        thr = float(seg_cfg["teleport_factor"]) * cap
        P, T, worst_pair = _positions(rows, conv)
        ids = [r["observation_id"] for r in rows]
        segs = segment_by_discontinuity(P, thr)
        steps = np.array([math.hypot(P[i, 0] - P[i - 1, 0], P[i, 1] - P[i - 1, 1]) for i in range(1, len(P))])
        within = [steps[i - 1] for s0, s1 in segs for i in range(s0 + 1, s1)]
        across = [steps[s0 - 1] for s0, _ in segs[1:]]
        trace_check = confirm_with_trace(trace, T, segs, thr)
        pieces = []
        for seg in segs:
            pieces += classify_segment(P, ids, seg, seg_cfg)
        run_rec = {"run_id": run_id, "level_dir": level_dir, "level_key": key, "split": lvl["split"],
                   "raw_path_id": settings.get("path", {}).get("path_id"),
                   "environment": settings.get("environment"), "n_observations": len(rows),
                   "capture_distance_m": cap, "teleport_threshold_m": thr,
                   "north_west_position_max_diff_m": worst_pair,
                   "n_segments_by_discontinuity": len(segs),
                   "max_within_segment_step_m": float(max(within)) if within else None,
                   "min_across_boundary_step_m": float(min(across)) if across else None,
                   "boundaries": [{"obs_index": int(s0), "from": ids[s0 - 1], "to": ids[s0],
                                   "dxy_m": float(steps[s0 - 1]), "dz_m": float(P[s0, 2] - P[s0 - 1, 2]),
                                   "dt_s": float(T[s0] - T[s0 - 1])} for s0, _ in segs[1:]],
                   "trace_confirmation": trace_check,
                   "pieces": []}
        for k, pc in enumerate(pieces):
            n_seg_total += 1
            seg_id = f"{run_id}/seg{k:02d}"
            s0, s1 = pc["start"], pc["end"]
            e0, n0, u0 = (float(v) for v in P[s0])
            rec = {"segment_id": seg_id, "kind": pc["kind"], "start_obs": ids[s0], "end_obs": ids[s1 - 1],
                   "n": s1 - s0, "start_east_m": e0, "start_north_m": n0, "start_up_m": u0}
            site_id, hard, center_anchor, center_dist, flags = None, False, None, None, []
            if pc["kind"] == "anchor_capture":
                a, d = _nearest(cat["anchors"][key], e0, n0)
                if d is None or d > float(seg_cfg["anchor_match_max_m"]):
                    raise SystemExit(f"{seg_id}: anchor capture {d} m from the nearest catalogue anchor "
                                     f"{a and a['anchor_id']} — the catalogue and the batch disagree")
                site_id, center_anchor, center_dist = f"anchor:{a['anchor_id']}", a["anchor_id"], d
                rec.update({"anchor_id": a["anchor_id"], "anchor_distance_m": d})
            elif pc["kind"] in ("star_swipe", "unclassified"):
                dec = pc["decomposition"]
                c = dec["center"]
                site, d = _nearest_site([s for s in cat["swipe_sites"] if s["level_key"] == key], c["east_m"], c["north_m"])
                a, da = _nearest(cat["anchors"][key], c["east_m"], c["north_m"])
                center_anchor, center_dist = a["anchor_id"], da
                if site is not None and d <= float(seg_cfg["site_match_max_m"]):
                    site_id, hard = f"{site['path_id']}@{site['anchor_id']}", site["hard_tag"]
                    rec.update({"old_session": site["old_session"], "old_site_distance_m": d})
                else:
                    site_id = f"newswipe@{a['anchor_id']}"
                    flags.append(f"no EXP-SKY-008 swipe centre within {seg_cfg['site_match_max_m']} m (nearest {d})")
                rec.update({"n_legs": dec["n_legs"], "leg_directions": dec["leg_directions"],
                            "decomposition_flags": dec["flags"], "hub_observation": dec["hub_observation_id"],
                            "hub_east_m": c["east_m"], "hub_north_m": c["north_m"], "hub_up_m": c["up_m"]})
            else:                                                   # transit
                a0, d0 = _nearest(cat["anchors"][key], e0, n0)
                e1, n1 = float(P[s1 - 1, 0]), float(P[s1 - 1, 1])
                a1, d1 = _nearest(cat["anchors"][key], e1, n1)
                site_id = "transit"
                length = float(sum(steps[i - 1] for i in range(s0 + 1, s1)))
                rec.update({"length_m": length, "from_anchor": a0["anchor_id"], "from_anchor_distance_m": d0,
                            "to_anchor": a1["anchor_id"], "to_anchor_distance_m": d1})
            rec.update({"site_id": site_id, "hard_tag": hard, "center_anchor_id": center_anchor,
                        "center_anchor_distance_m": center_dist, "flags": flags})
            run_rec["pieces"].append(rec)
            by_raw = {}
            if pc.get("decomposition"):
                by_raw = {o["observation_id"]: o for o in pc["decomposition"]["observations"]}
                legs = pc["decomposition"]["legs"]
            for j in range(s0, s1):
                r = rows[j]
                a, da = _nearest(cat["anchors"][key], float(P[j, 0]), float(P[j, 1]))
                o = by_raw.get(ids[j])
                leg = legs[o["leg_id"]] if (o and o.get("leg_id") is not None) else None
                index_rows.append({
                    "level": settings.get("level"), "level_dir": level_dir, "level_key": key,
                    "split": lvl["split"], "combined_run_id": run_id,
                    "raw_path_id": settings.get("path", {}).get("path_id"),
                    "segment_id": seg_id, "segment_kind": pc["kind"],
                    "site_id": (f"transit:{a['anchor_id']}" if pc["kind"] == "transit" else site_id),
                    "segment_site_id": site_id, "hard_tag": hard,
                    "observation_raw_id": ids[j], "obs_index_in_run": j, "index_in_segment": j - s0,
                    "sim_time_s": float(T[j]), "east_m": float(P[j, 0]), "north_m": float(P[j, 1]),
                    "up_m": float(P[j, 2]),
                    "north_session_id": session_id_for(run_id, "north"),
                    "west_session_id": session_id_for(run_id, "west"),
                    "north_observation_id": observation_id_for(run_id, "north", ids[j]),
                    "west_observation_id": observation_id_for(run_id, "west", ids[j]),
                    "north_image": r.get("image_path"), "west_image": r.get("west_image_path"),
                    "time_of_day": settings.get("environment", {}).get("time_of_day"),
                    "clouds": settings.get("environment", {}).get("clouds"),
                    "leg_id": "" if not o or o.get("leg_id") is None else o["leg_id"],
                    "leg_axis": leg["axis"] if leg else "",
                    "leg_direction": (leg["direction"] or "") if leg else "",
                    "phase": o["phase"] if o else ("anchor" if pc["kind"] == "anchor_capture" else "transit"),
                    "lateral_east_m": o["lateral_east_m"] if o else None,
                    "longitudinal_north_m": o["longitudinal_north_m"] if o else None,
                    "vertical_up_m": o["vertical_up_m"] if o else None,
                    "total_displacement_m": o["total_displacement_m"] if o else None,
                    "swipe_center_anchor_id": center_anchor or "",
                    "swipe_center_anchor_distance_m": center_dist,
                    "nearest_anchor_id": a["anchor_id"], "nearest_anchor_distance_m": da,
                    "pair_ok": "", "gt_status_north": "", "gt_status_west": "",
                })
        audit["runs"].append(run_rec)
        print(f"[audit] {level_dir}/{run_id} path_id={run_rec['raw_path_id']}: {len(rows)} obs, "
              f"{len(segs)} discontinuity segments -> {[p['kind'] for p in pieces]}; trace "
              f"{trace_check.get('n_confirmed')}/{trace_check.get('n_boundaries')} boundaries confirmed, "
              f"{len(trace_check.get('trace_jumps_inside_a_segment', []))} jumps inside a segment")
    audit["n_segments"] = n_seg_total
    audit["n_index_rows"] = len(index_rows)
    audit["segmentation"] = seg_cfg
    audit["kinds"] = dict(sorted(defaultdict(int, {k: sum(1 for r in index_rows if r["segment_kind"] == k)
                                                 for k in {r["segment_kind"] for r in index_rows}}).items()))
    fields = list(index_rows[0].keys())
    _write_csv(out / "index.csv", index_rows, fields)
    _json(out / "audit.json", audit)
    _fig_plan(out, cfg, index_rows, cat)
    print(f"[audit] {len(index_rows)} observations, {n_seg_total} segments -> {out / 'index.csv'}")


def _fig_plan(out: Path, cfg: dict, rows: list, cat: dict) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, key in zip(axes, LEVEL_ORDER):
        lv = [r for r in rows if r["level_key"] == key]
        for a in cat["anchors"][key]:
            ax.plot(a["east_m"], a["north_m"], marker="^", color="#444444", ms=7, ls="none")
            ax.annotate(a["anchor_id"], (a["east_m"], a["north_m"]), fontsize=6, color="#444444")
        segs = defaultdict(list)
        for r in lv:
            segs[r["segment_id"]].append(r)
        prev_end = {}
        for sid in sorted(segs):
            g = sorted(segs[sid], key=lambda r: r["obs_index_in_run"])
            e = [r["east_m"] for r in g]
            n = [r["north_m"] for r in g]
            kind = g[0]["segment_kind"]
            color = {"star_swipe": "#d62728" if g[0]["hard_tag"] else "#1f77b4", "transit": "#ff7f0e",
                     "anchor_capture": "#2ca02c"}.get(kind, "#9467bd")
            if kind == "anchor_capture":
                ax.plot(e, n, marker="o", color=color, ms=4, ls="none")
            else:
                ax.plot(e, n, "-", color=color, lw=1.2)
                ax.plot(e[0], n[0], marker="s", color=color, ms=5, ls="none")
                ax.annotate(g[0]["segment_site_id"], (e[0], n[0]), fontsize=7, color=color)
            run = g[0]["combined_run_id"]
            if run in prev_end and kind != "anchor_capture":
                pe, pn = prev_end[run]
                ax.plot([pe, e[0]], [pn, n[0]], ls="--", color="#999999", lw=0.8)
            prev_end[run] = (e[-1], n[-1])
        ax.set_title(f"{key}: reconstructed segments (dashed = teleport)")
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    _save(fig, out / "figures" / "fig0_plan_view_segments.jpg")


# ==================================================================================================
# ingest
# ==================================================================================================

def stage_ingest(cfg: dict, base: Path) -> None:
    out = _resolve(base, cfg["out_dir"])
    store = _resolve(base, cfg["store_root"])
    store.mkdir(parents=True, exist_ok=True)
    conv = SimConventions.from_dict(cfg.get("conventions"))
    summaries = {}
    for level_dir, lvl, rd in _raw_runs(cfg, base):
        run_id = json.loads((rd / "settings.json").read_text(encoding="utf-8"))["run_id"]
        for view in cfg["views"]:
            sid = session_id_for(run_id, view)
            if (store / sid / "session.json").exists():
                print(f"[ingest] {sid}: present, kept")
                summaries[sid] = {"kept": True}
                continue
            t0 = time.time()
            s = adapter.ingest_run(rd, store, session_id=sid, conventions=conv,
                                   options={**cfg["ingest_options"], "view": view})
            s["seconds"] = time.time() - t0
            summaries[sid] = s
            print(f"[ingest] {sid}: {s['n_observations']} observations, {s['n_seam_curves']} GT seam "
                  f"curves, {s['n_database_holes']} holes ({s['seconds']:.0f} s)")
    _json(out / "ingest_summary.json", summaries)
    _verify_pairs(cfg, base)
    _write_frame_list(cfg, base)


def _verify_pairs(cfg: dict, base: Path) -> None:
    """Every physical observation must exist in both views at the same position and time."""
    out = _resolve(base, cfg["out_dir"])
    store = _resolve(base, cfg["store_root"])
    rows = _read_csv(out / "index.csv")
    obs_by_session = {}
    report = {"n_rows": len(rows), "n_pairs_ok": 0, "problems": []}
    for r in rows:
        ok = True
        ref = None
        for view in cfg["views"]:
            sid = r[f"{view}_session_id"]
            if sid not in obs_by_session:
                _, obs = read_session(store / sid)
                obs_by_session[sid] = {o.observation_id: o for o in obs}
            o = obs_by_session[sid].get(r[f"{view}_observation_id"])
            if o is None:
                ok = False
                report["problems"].append(f"{r['segment_id']}/{r['observation_raw_id']}: no {view} observation")
                continue
            gt = "ok" if (store / sid / "skylines_oracle" / f"{o.observation_id}.csv").exists() else "insufficient_sky"
            r[f"gt_status_{view}"] = gt
            here = (o.pos_east_m, o.pos_north_m, o.up_m, o.timestamp_s)
            if ref is None:
                ref = here
            elif max(abs(a - b) for a, b in zip(ref, here)) > 1e-6:
                ok = False
                report["problems"].append(f"{r['segment_id']}/{r['observation_raw_id']}: views disagree {ref} vs {here}")
            if abs(o.pos_east_m - float(r["east_m"])) > 1e-6 or abs(o.pos_north_m - float(r["north_m"])) > 1e-6:
                ok = False
                report["problems"].append(f"{r['segment_id']}/{r['observation_raw_id']}: store position differs from the audit index")
        r["pair_ok"] = ok
        report["n_pairs_ok"] += int(ok)
    report["gt_holes"] = {v: sum(1 for r in rows if r[f"gt_status_{v}"] != "ok") for v in cfg["views"]}
    fields = list(rows[0].keys())
    _write_csv(out / "index.csv", rows, fields)
    _json(out / "pairs.json", report)
    print(f"[ingest] pairs: {report['n_pairs_ok']}/{report['n_rows']} ok, GT holes {report['gt_holes']}, "
          f"{len(report['problems'])} problems")
    if report["problems"]:
        raise SystemExit(f"pairing problems: {report['problems'][:5]}")


def _write_frame_list(cfg: dict, base: Path) -> None:
    """The SegFormer frame list (isolated venv) from the store: same ids the sources will key by."""
    store = _resolve(base, cfg["store_root"])
    rows = []
    for sid in sorted(p.name for p in store.iterdir() if (p / "observations.csv").exists()):
        meta, obs = read_session(store / sid)
        run_dir = Path(meta["simulator_run"]["run_dir"])
        for o in obs:
            rows.append({"session_id": sid, "observation_id": o.observation_id,
                         "image_path": str((run_dir / o.extra["sim_image_source"]).resolve()),
                         "width_px": o.image_width_px, "height_px": o.image_height_px})
    _write_csv(store / "silver_frames_dual.csv", rows, ["session_id", "observation_id", "image_path",
                                                        "width_px", "height_px"])
    print(f"[ingest] frame list: {len(rows)} frames -> {store / 'silver_frames_dual.csv'}")


# ==================================================================================================
# score — one exact exhaustive C1-32 pass per (level, source, view); memories are keep-masks
# ==================================================================================================

def _index_rows(cfg: dict, base: Path) -> list:
    out = _resolve(base, cfg["out_dir"])
    rows = _read_csv(out / "index.csv")
    for r in rows:
        for k in ("east_m", "north_m", "up_m", "sim_time_s", "lateral_east_m", "longitudinal_north_m",
                  "vertical_up_m", "total_displacement_m", "swipe_center_anchor_distance_m",
                  "nearest_anchor_distance_m"):
            v = _f(r.get(k))
            r[k] = None if (isinstance(v, float) and math.isnan(v)) else v
        r["obs_index_in_run"] = int(r["obs_index_in_run"])
        r["index_in_segment"] = int(r["index_in_segment"])
        r["hard_tag"] = r["hard_tag"] == "True"
        r["pair_ok"] = r["pair_ok"] == "True"
    return rows


def _sources(cfg: dict, base: Path, sessions: dict, cam) -> dict:
    from hsreloc.simret.sources import SessionSilverSource, SimExactSource
    store = _resolve(base, cfg["store_root"])
    out = {}
    for key in cfg["sources"]:
        if key == "sim_exact":
            out[key] = SimExactSource(store, sessions, cam.width_px, cam.height_px)
        elif key == "segformer":
            spec = cfg["segformer"]
            out[key] = SessionSilverSource(
                Path(spec["mask_root"]), sessions, cam.width_px, cam.height_px,
                scale_short_side=spec.get("scale_short_side", 512), closing_px=spec.get("closing_px", 0),
                min_valid_frac=spec.get("min_valid_frac", 0.5),
                expected_model_revision=spec.get("expected_model_revision"))
        else:
            raise SystemExit(f"unsupported source {key!r}")
    return out


def _key(r: dict) -> str:
    """The physical observation key (raw ids repeat across runs)."""
    return f"{r['combined_run_id']}/{r['observation_raw_id']}"


def _is_dense_reference(r: dict) -> bool:
    """Anchors plus every *horizontal* path frame (star legs on the lateral/longitudinal axes,
    centre passes, transit frames); vertical-leg frames are queries only."""
    if r["segment_kind"] == "anchor_capture":
        return True
    if r["leg_axis"] == "vertical":
        return False
    return r["vertical_up_m"] is None or abs(r["vertical_up_m"]) < 1.0


def stage_score(cfg: dict, base: Path, sources_only=None) -> None:
    from hsreloc.simret.relpose import frozen_c1
    from hsreloc.simret.sources import session_map
    from sky_relpose_study import _Curves, _camera
    out = _resolve(base, cfg["out_dir"])
    store = _resolve(base, cfg["store_root"])
    rows = _index_rows(cfg, base)
    if not all(r["pair_ok"] for r in rows):
        raise SystemExit("index.csv has unpaired observations — run the ingest stage first")
    session_ids = sorted({r[f"{v}_session_id"] for r in rows for v in cfg["views"]})
    sessions = session_map(store, session_ids)
    cam = _camera(store, session_ids, int(cfg["profile_n_samples"]))
    if sources_only:                                  # score one source while another's inputs finish
        cfg = {**cfg, "sources": [k for k in cfg["sources"] if k in sources_only]}
    sources = _sources(cfg, base, sessions, cam)
    max_lag = int(cfg["max_lag_samples"])
    every = int(cfg["frozen_check_every"])
    cells = out / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    manifest = {"study_version": STUDY_VERSION, "matcher": {"max_lag_samples": max_lag,
                "min_overlap_frac": cfg["min_overlap_frac"], "n_samples": cfg["profile_n_samples"]},
                "camera": cam.as_dict(), "cells": {}, "sources": {k: s.describe() for k, s in sources.items()}}
    by_level = defaultdict(list)
    for r in rows:
        by_level[r["level_key"]].append(r)
    for level in LEVEL_ORDER:
        lv = by_level[level]
        refs_all = [r for r in lv if _is_dense_reference(r)]
        queries_all = [r for r in lv if r["segment_kind"] != "anchor_capture"]
        for source, src in sources.items():
            cache = _Curves(src)
            # a physical reference / query enters only when BOTH views have a curve
            def both(r):
                return all(cache.get(r[f"{v}_observation_id"]) is not None for v in cfg["views"])
            refs = [r for r in refs_all if both(r)]
            queries = [r for r in queries_all if both(r)]
            holes = {v: [r["observation_raw_id"] for r in refs_all + queries_all
                         if cache.get(r[f"{v}_observation_id"]) is None] for v in cfg["views"]}
            info = {"level": level, "source": source, "n_references_paired": len(refs),
                    "n_anchor_references": sum(1 for r in refs if r["segment_kind"] == "anchor_capture"),
                    "n_queries_paired": len(queries), "n_reference_candidates": len(refs_all),
                    "n_query_candidates": len(queries_all),
                    "holes_by_view": {v: len(h) for v, h in holes.items()}, "views": {}}
            for view in cfg["views"]:
                name = f"scores_{level}_{source}_{view}"
                target = cells / f"{name}.npz"
                if target.exists():
                    print(f"[score] {name}: reused checkpoint")
                    with np.load(target, allow_pickle=False) as z:
                        info["views"][view] = {"reused": True, "n_queries": int(z["scores"].shape[0]),
                                               "n_references": int(z["scores"].shape[1])}
                    continue
                ids = [r[f"{view}_observation_id"] for r in refs]
                profs = [cache.profiles[i] for i in ids]
                t0 = time.time()
                bank = recog.ReferenceBank(ids, profs, max_lag=max_lag,
                                           min_overlap_frac=float(cfg["min_overlap_frac"]))
                t_build = time.time() - t0
                S = np.empty((len(queries), len(refs)), dtype=np.float64)
                L = np.empty_like(S)
                checks, t_score = 0, 0.0
                for qi, q in enumerate(queries):
                    qp = cache.profiles[q[f"{view}_observation_id"]]
                    t1 = time.time()
                    scores, lags = bank.score(qp)
                    t_score += time.time() - t1
                    if qi % every == 0:                  # the batched search must equal the frozen matcher
                        j = qi % len(ids)
                        fr = frozen_c1(qp, profs[j], max_lag=max_lag)
                        if not (abs(fr.score - scores[j]) <= 1e-12 and fr.shift == lags[j]):
                            raise SystemExit(f"frozen cross-check FAILED {name} {_key(q)} vs "
                                             f"{ids[j]}: {fr.score}/{fr.shift} vs {scores[j]}/{lags[j]}")
                        checks += 1
                    S[qi] = scores
                    L[qi] = lags
                np.savez_compressed(
                    target, scores=S, lags=L,
                    query_keys=np.asarray([_key(q) for q in queries]),
                    query_segments=np.asarray([q["segment_id"] for q in queries]),
                    reference_keys=np.asarray([_key(r) for r in refs]),
                    reference_segments=np.asarray([r["segment_id"] for r in refs]),
                    reference_xy=np.asarray([(r["east_m"], r["north_m"]) for r in refs], dtype=np.float64),
                    query_xy=np.asarray([(q["east_m"], q["north_m"]) for q in queries], dtype=np.float64),
                    reference_is_anchor=np.asarray([r["segment_kind"] == "anchor_capture" for r in refs]))
                info["views"][view] = {"reused": False, "n_queries": len(queries), "n_references": len(refs),
                                       "frozen_cross_checks": checks, "bank_build_s": t_build,
                                       "score_s_per_query": t_score / max(1, len(queries)),
                                       "score_s_per_query_per_reference": t_score / max(1, len(queries) * len(refs)),
                                       "c1_check": "bank_lag_exact_score_1e-12"}
                print(f"[score] {name}: {len(queries)} queries x {len(refs)} references, {checks} frozen "
                      f"cross-checks, {t_score / max(1, len(queries)) * 1e3:.1f} ms/query")
            manifest["cells"][f"{level}/{source}"] = info
    mpath = out / "score_manifest.json"
    if mpath.exists():                                # merge: cells of other sources scored earlier
        old = json.loads(mpath.read_text(encoding="utf-8"))
        manifest["cells"] = {**old.get("cells", {}), **manifest["cells"]}
        manifest["sources"] = {**old.get("sources", {}), **manifest["sources"]}
    _json(mpath, manifest)
    print(f"[score] manifest -> {mpath}")


# ==================================================================================================
# main
# ==================================================================================================

def stage_report(cfg: dict, base: Path) -> None:
    from sky_dual_report import stage_report as _report
    _report(cfg, base)


STAGES = {"audit": stage_audit, "ingest": stage_ingest, "score": stage_score, "report": stage_report}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--sources", default="", help="score: comma-separated subset of the config's sources")
    a = ap.parse_args(argv)
    cfg, base = _load_cfg(Path(a.config).resolve())
    if a.stage == "score" and a.sources:
        stage_score(cfg, base, sources_only=[s.strip() for s in a.sources.split(",") if s.strip()])
    else:
        STAGES[a.stage](cfg, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
