#!/usr/bin/env python
"""EXP-SKY-012 — dual-view (North + West) validation on the previously missing city HardSwipe.

A targeted follow-up to ``EXP-SKY-011``, not a new experiment: ONE new run
(``large_flat_city/Run_20260907_125137``, the ``EXP-SKY-008``/``010`` city HardSwipe flown again with
both world-locked views) is scored against the **identical** city reference memory ``EXP-SKY-011``
used, with every rule, threshold and gate frozen. Stages::

    audit    the new run alone: segmentation (same discontinuity rule, 60 Hz trace), site identity
             against the EXP-SKY-008 catalogue (must be the HardSwipe@a16 site), North/West pairing,
             camera/settings equality with the frozen city run, geometry against the old HardSwipe
             frames and against the frozen sparse/dense memories; index.csv + audit.json + plan view.
    ingest   two spec-006 sessions (north, west) into a SEPARATE store through the unchanged adapter
             (view option); pairing verified; the SegFormer frame list for the isolated venv.
    score    the frozen city reference list (evaluations/sky-dual/cells/scores_city_*.npz) is rebuilt
             from the frozen store + frozen mask root and must reproduce frozen score rows to 1e-12
             (lag exact) before any new query is scored; then every new query, both views, both
             sources, with frozen ``match()`` cross-checks; matrices saved under the new out_dir.
             Diagnostic: North against the unpaired full anchor memory (the EXP-SKY-010 memory shape).
    report   North-only, West-only, the three EXP-SKY-011 dual rules and the declared
             North-primary/West-veto rule at the DEV-frozen gates and at the frozen 0.90/0.15 gate;
             what West does on every North false accept; the zero-false stress test with the combined
             EXP-SKY-011 + this-run hard-negative count; temporal (free) and the 4x4 grid as
             sensitivity only; the declared A-D classification.

Nothing under ``evaluations/sky-dual``, ``observations_sim_dual`` or the EXP-SKY-011 mask root is
written. ``hsreloc/retrieval``, ``hsreloc/matchers``, ``hsreloc/simret/dualview.py`` and INT are
not modified. No threshold is chosen on this run.
"""
from __future__ import annotations

import argparse
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

from hsreloc.simret import adapter, recog                                       # noqa: E402
from hsreloc.simret import dualview as dv                                       # noqa: E402
from hsreloc.simret.conventions import SimConventions                           # noqa: E402
from sky_relpose_study import _f, _json, _plt, _read_csv, _resolve, _save       # noqa: E402
from sky_dual_study import (STUDY_VERSION as DUAL_STUDY_VERSION, _catalogue,    # noqa: E402
                            _index_rows, _key, _nearest, _nearest_site, _positions, _read_raw,
                            _verify_pairs, _write_csv, _write_frame_list, classify_segment,
                            confirm_with_trace, observation_id_for, segment_by_discontinuity,
                            session_id_for)
from sky_dual_report import (CHANNELS, KEEP_FIELDS, RULE_LABEL, RULES, SRC_LABEL,   # noqa: E402
                             accepted_at, build_rows, false_of, flat, summarize, temporal_rows,
                             top1_of, topk_of)

STUDY_VERSION = "1.0.0"
VIEWS = ("north", "west")
LEVEL = "city"
FROZEN_GATE = (0.90, 0.15)
SETTINGS_BLOCKS = ("north_camera", "west_camera", "skyline", "environment", "world_frame",
                   "quaternion_convention", "capture")


def _load_cfg(path: Path) -> tuple:
    hard = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    frozen = json.loads((base / hard["frozen_config"]).read_text(encoding="utf-8"))
    cfg = {k: v for k, v in frozen.items() if not k.startswith("_")}
    cfg["frozen"] = {"config": hard["frozen_config"], "store_root": frozen["store_root"],
                     "out_dir": frozen["out_dir"], "segformer_mask_root": frozen["segformer"]["mask_root"]}
    cfg.update({k: v for k, v in hard.items() if not k.startswith("_")})
    cfg["segformer"] = {**frozen["segformer"], "mask_root": hard["segformer_mask_root"]}
    return cfg, base


# ==================================================================================================
# audit
# ==================================================================================================

def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def stage_audit(cfg: dict, base: Path) -> None:
    out = _resolve(base, cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    conv = SimConventions.from_dict(cfg.get("conventions"))
    seg_cfg = cfg["segmentation"]
    cat = _catalogue(cfg, base)
    rd = _resolve(base, cfg["run_dir"])
    level_dir = cfg["level_dir"]
    lvl = cfg["levels"][level_dir]
    key = lvl["key"]
    settings, rows, trace = _read_raw(rd)
    run_id = settings["run_id"]
    cap = float(settings.get("skyline", {}).get("capture_distance_m", 10.0))
    thr = float(seg_cfg["teleport_factor"]) * cap
    P, T, worst_pair = _positions(rows, conv)
    ids = [r["observation_id"] for r in rows]
    segs = segment_by_discontinuity(P, thr)
    steps = np.array([_dist(P[i], P[i - 1]) for i in range(1, len(P))])
    trace_check = confirm_with_trace(trace, T, segs, thr)
    pieces = [pc for seg in segs for pc in classify_segment(P, ids, seg, seg_cfg)]
    stars = [pc for pc in pieces if pc["kind"] == "star_swipe"]
    if len(segs) != 1 or len(pieces) != 1 or len(stars) != 1:
        raise SystemExit(f"{run_id}: expected exactly one star swipe and no teleport, got "
                         f"{len(segs)} discontinuity segments -> {[p['kind'] for p in pieces]}")
    exp = cfg["expected_site"]
    # --- settings equality with the frozen city run ------------------------------------------
    frozen_settings = json.loads((_resolve(base, cfg["raw_root"]) / level_dir / cfg["frozen_city_settings_run"]
                                  / "settings.json").read_text(encoding="utf-8"))
    settings_cmp = {}
    for blk in SETTINGS_BLOCKS:
        a, b = settings.get(blk), frozen_settings.get(blk)
        settings_cmp[blk] = {"equal": a == b}
        if a != b:
            settings_cmp[blk]["new"] = a
            settings_cmp[blk]["frozen"] = b
    headings = {"north_yaw_deg": sorted({r["north_ue_yaw_deg"] for r in rows}),
                "west_yaw_deg": sorted({r["west_ue_yaw_deg"] for r in rows}),
                "north_pitch_roll": sorted({(r["north_ue_pitch_deg"], r["north_ue_roll_deg"]) for r in rows}),
                "west_pitch_roll": sorted({(r["west_ue_pitch_deg"], r["west_ue_roll_deg"]) for r in rows}),
                "north_heading_mode": settings["north_camera"].get("heading_mode"),
                "west_heading_mode": settings["west_camera"].get("heading_mode")}
    # --- pieces and index rows (same fields as EXP-SKY-011's index.csv) -----------------------
    index_rows, run_pieces = [], []
    for k, pc in enumerate(pieces):
        seg_id = f"{run_id}/seg{k:02d}"
        s0, s1 = pc["start"], pc["end"]
        dec = pc["decomposition"]
        c = dec["center"]
        site, d_site = _nearest_site([s for s in cat["swipe_sites"] if s["level_key"] == key], c["east_m"], c["north_m"])
        a, da = _nearest(cat["anchors"][key], c["east_m"], c["north_m"])
        if site is None or d_site > float(seg_cfg["site_match_max_m"]):
            raise SystemExit(f"{seg_id}: hub {d_site} m from the nearest EXP-SKY-008 swipe centre — not a catalogue site")
        site_id = f"{site['path_id']}@{site['anchor_id']}"
        if site["path_id"] != exp["path_id"] or site["anchor_id"] != exp["anchor_id"] or site["hard_tag"] != exp["hard_tag"]:
            raise SystemExit(f"{seg_id}: matched site {site_id} (hard={site['hard_tag']}) is not the expected "
                             f"{exp['path_id']}@{exp['anchor_id']} (hard={exp['hard_tag']})")
        rec = {"segment_id": seg_id, "kind": pc["kind"], "start_obs": ids[s0], "end_obs": ids[s1 - 1], "n": s1 - s0,
               "site_id": site_id, "hard_tag": site["hard_tag"], "old_session": site["old_session"],
               "old_site_distance_m": d_site, "center_anchor_id": a["anchor_id"], "center_anchor_distance_m": da,
               "n_legs": dec["n_legs"], "leg_directions": dec["leg_directions"], "decomposition_flags": dec["flags"],
               "hub_observation": dec["hub_observation_id"], "hub_east_m": c["east_m"], "hub_north_m": c["north_m"],
               "hub_up_m": c["up_m"]}
        run_pieces.append(rec)
        by_raw = {o["observation_id"]: o for o in dec["observations"]}
        legs = dec["legs"]
        for j in range(s0, s1):
            r = rows[j]
            an, dan = _nearest(cat["anchors"][key], float(P[j, 0]), float(P[j, 1]))
            o = by_raw.get(ids[j])
            leg = legs[o["leg_id"]] if (o and o.get("leg_id") is not None) else None
            index_rows.append({
                "level": settings.get("level"), "level_dir": level_dir, "level_key": key, "split": lvl["split"],
                "combined_run_id": run_id, "raw_path_id": settings.get("path", {}).get("path_id"),
                "segment_id": seg_id, "segment_kind": pc["kind"], "site_id": site_id, "segment_site_id": site_id,
                "hard_tag": site["hard_tag"], "observation_raw_id": ids[j], "obs_index_in_run": j,
                "index_in_segment": j - s0, "sim_time_s": float(T[j]), "east_m": float(P[j, 0]),
                "north_m": float(P[j, 1]), "up_m": float(P[j, 2]),
                "north_session_id": session_id_for(run_id, "north"), "west_session_id": session_id_for(run_id, "west"),
                "north_observation_id": observation_id_for(run_id, "north", ids[j]),
                "west_observation_id": observation_id_for(run_id, "west", ids[j]),
                "north_image": r.get("image_path"), "west_image": r.get("west_image_path"),
                "time_of_day": settings.get("environment", {}).get("time_of_day"),
                "clouds": settings.get("environment", {}).get("clouds"),
                "leg_id": "" if not o or o.get("leg_id") is None else o["leg_id"],
                "leg_axis": leg["axis"] if leg else "", "leg_direction": (leg["direction"] or "") if leg else "",
                "phase": o["phase"] if o else "", "lateral_east_m": o["lateral_east_m"] if o else None,
                "longitudinal_north_m": o["longitudinal_north_m"] if o else None,
                "vertical_up_m": o["vertical_up_m"] if o else None,
                "total_displacement_m": o["total_displacement_m"] if o else None,
                "swipe_center_anchor_id": a["anchor_id"], "swipe_center_anchor_distance_m": da,
                "nearest_anchor_id": an["anchor_id"], "nearest_anchor_distance_m": dan,
                "pair_ok": "", "gt_status_north": "", "gt_status_west": ""})
    # --- geometry against the old HardSwipe frames, the anchors and the frozen memories -------
    site = run_pieces[0]
    old = [r for r in _read_csv(_resolve(base, cfg["old_index_csv"])) if r["session_id"] == site["old_session"]]
    O = np.array([(float(r["east_m"]), float(r["north_m"]), float(r["up_m"])) for r in old])
    d_new_old = np.hypot(P[:, None, 0] - O[None, :, 0], P[:, None, 1] - O[None, :, 1])
    old_centre = min(old, key=lambda r: float(r["total_displacement_m"] or 0.0))
    tau = float(cfg["tau_region_m"])
    anchors_d = np.array([r["nearest_anchor_distance_m"] for r in index_rows])
    frozen_cells = _resolve(base, cfg["frozen_out_dir"]) / "cells"
    dense_geom = {}
    for source in cfg["sources"]:
        with np.load(frozen_cells / f"scores_{LEVEL}_{source}_north.npz", allow_pickle=False) as z:
            rxy = np.asarray(z["reference_xy"])
            rseg = [str(s) for s in z["reference_segments"]]
            is_anchor = np.asarray(z["reference_is_anchor"], dtype=bool)
        dd = np.hypot(P[:, None, 0] - rxy[None, :, 0], P[:, None, 1] - rxy[None, :, 1])
        near = dd.min(axis=1)
        near_seg = [rseg[int(j)] for j in dd.argmin(axis=1)]
        dense_geom[source] = {
            "n_references_paired": int(len(rxy)), "n_anchor_references_paired": int(is_anchor.sum()),
            "n_queries_with_a_paired_anchor_within_tau": int((dd[:, is_anchor].min(axis=1) <= tau).sum()),
            "n_queries_with_a_dense_reference_within_tau": int((near <= tau).sum()),
            "nearest_dense_reference_m": {"min": float(near.min()), "median": float(np.median(near)), "max": float(near.max())},
            "nearest_dense_reference_segments": sorted(set(near_seg)),
            "n_references_within_tau_of_any_query": int((dd <= tau).any(axis=0).sum())}
    audit = {
        "study_version": STUDY_VERSION, "dual_study_version": DUAL_STUDY_VERSION, "run_id": run_id,
        "run_dir": str(rd), "level_dir": level_dir, "level_key": key, "split": lvl["split"],
        "raw_path_id": settings.get("path", {}).get("path_id"), "environment": settings.get("environment"),
        "n_observations": len(rows), "capture_distance_m": cap, "teleport_threshold_m": thr,
        "north_west_position_max_diff_m": worst_pair, "headings": headings,
        "settings_vs_frozen_city_run": {"frozen_run": cfg["frozen_city_settings_run"], "blocks": settings_cmp,
                                        "all_equal": all(v["equal"] for v in settings_cmp.values())},
        "n_segments_by_discontinuity": len(segs), "max_step_m": float(steps.max()), "median_step_m": float(np.median(steps)),
        "max_dz_step_m": float(np.abs(np.diff(P[:, 2])).max()), "boundaries": [],
        "trace_confirmation": trace_check, "pieces": run_pieces,
        "expected_site": exp,
        "site_match": {"site_id": site["site_id"], "hard_tag": site["hard_tag"], "old_session": site["old_session"],
                       "hub_to_old_centre_pass_m": site["old_site_distance_m"],
                       "hub_to_old_run_start_frame_m": _dist((site["hub_east_m"], site["hub_north_m"]),
                                                             (float(old_centre["east_m"]), float(old_centre["north_m"]))),
                       "start_frame_to_old_run_start_frame_m": _dist(P[0], (float(old_centre["east_m"]), float(old_centre["north_m"])))},
        "old_hard_swipe": {"session": site["old_session"], "n": len(old),
                           "east_range_m": [float(O[:, 0].min()), float(O[:, 0].max())],
                           "north_range_m": [float(O[:, 1].min()), float(O[:, 1].max())],
                           "up_range_m": [float(O[:, 2].min()), float(O[:, 2].max())]},
        "new_hard_swipe": {"n": len(rows), "n_legs": site["n_legs"], "leg_directions": site["leg_directions"],
                           "east_range_m": [float(P[:, 0].min()), float(P[:, 0].max())],
                           "north_range_m": [float(P[:, 1].min()), float(P[:, 1].max())],
                           "up_range_m": [float(P[:, 2].min()), float(P[:, 2].max())],
                           "n_horizontal": sum(1 for r in index_rows if r["leg_axis"] != "vertical"),
                           "n_vertical": sum(1 for r in index_rows if r["leg_axis"] == "vertical")},
        "new_to_old_nearest_frame_m": {"median": float(np.median(d_new_old.min(axis=1))),
                                       "p90": float(np.percentile(d_new_old.min(axis=1), 90)),
                                       "max": float(d_new_old.min(axis=1).max())},
        "old_to_new_nearest_frame_m": {"median": float(np.median(d_new_old.min(axis=0))),
                                       "max": float(d_new_old.min(axis=0).max())},
        "sparse_memory_geometry": {"tau_region_m": tau, "nearest_anchor_id": sorted({r["nearest_anchor_id"] for r in index_rows}),
                                   "nearest_anchor_m": {"min": float(anchors_d.min()), "median": float(np.median(anchors_d)),
                                                        "max": float(anchors_d.max())},
                                   "n_queries_with_an_anchor_within_tau": int((anchors_d <= tau).sum()),
                                   "out_of_coverage_by_construction": bool((anchors_d > tau).all())},
        "dense_memory_geometry": dense_geom,
        "segmentation": seg_cfg,
    }
    _write_csv(out / "index.csv", index_rows, list(index_rows[0].keys()))
    _json(out / "audit.json", audit)
    _fig_plan(out, cfg, index_rows, cat, key, O, frozen_cells)
    print(f"[audit] {run_id} path_id={audit['raw_path_id']}: {len(rows)} obs, {len(segs)} segment(s) -> "
          f"{[p['kind'] for p in pieces]}; site {site['site_id']} (hard={site['hard_tag']}), hub {site['old_site_distance_m']:.2f} m "
          f"from the old centre pass; settings equal to frozen city run: {audit['settings_vs_frozen_city_run']['all_equal']}; "
          f"nearest anchor {anchors_d.min():.1f} m (OOC by construction: {audit['sparse_memory_geometry']['out_of_coverage_by_construction']})")


def _fig_plan(out, cfg, rows, cat, key, O, frozen_cells):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    with np.load(frozen_cells / f"scores_{LEVEL}_{cfg['sources'][0]}_north.npz", allow_pickle=False) as z:
        rxy = np.asarray(z["reference_xy"])
        is_anchor = np.asarray(z["reference_is_anchor"], dtype=bool)
    e_all = [r["east_m"] for r in rows]
    n_all = [r["north_m"] for r in rows]
    for ax, zoom in zip(axes, (False, True)):
        for a in cat["anchors"][key]:
            ax.plot(a["east_m"], a["north_m"], marker="^", color="#444444", ms=7, ls="none")
            ax.annotate(a["anchor_id"], (a["east_m"], a["north_m"]), fontsize=6, color="#444444")
        ax.plot(rxy[~is_anchor, 0], rxy[~is_anchor, 1], ".", color="#ff7f0e", ms=3,
                label="frozen dense references (EXP-SKY-011 transit + star)")
        ax.plot(O[:, 0], O[:, 1], "o", color="#bbbbbb", ms=4, label="old HardSwipe frames (EXP-SKY-008/010)")
        cols = {"lateral": "#1f77b4", "longitudinal": "#2ca02c", "vertical": "#d62728", "": "#9467bd"}
        for axis_, col in cols.items():
            g = [r for r in rows if r["leg_axis"] == axis_]
            if g:
                ax.plot([r["east_m"] for r in g], [r["north_m"] for r in g], "x", color=col, ms=5,
                        label=f"new run, leg axis {axis_ or 'centre'} ({len(g)})")
        circ = plt.Circle((float(np.mean(e_all)), float(np.mean(n_all))), float(cfg["tau_region_m"]),
                          fill=False, ls="--", color="grey")
        ax.add_patch(circ)
        if zoom:
            ax.set_xlim(min(e_all) - 200, max(e_all) + 200)
            ax.set_ylim(min(n_all) - 200, max(n_all) + 200)
            ax.set_title("zoom: the HardSwipe site, τ_region circle, nearest frozen transit frames", fontsize=9)
        else:
            ax.set_title("city: anchors, frozen dense references, old and new HardSwipe", fontsize=9)
            ax.legend(fontsize=7, loc="lower left")
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    _save(fig, out / "figures" / "fig0_plan_view_hard_city.jpg")


# ==================================================================================================
# ingest
# ==================================================================================================

def stage_ingest(cfg: dict, base: Path) -> None:
    out = _resolve(base, cfg["out_dir"])
    store = _resolve(base, cfg["store_root"])
    store.mkdir(parents=True, exist_ok=True)
    conv = SimConventions.from_dict(cfg.get("conventions"))
    rd = _resolve(base, cfg["run_dir"])
    run_id = json.loads((rd / "settings.json").read_text(encoding="utf-8"))["run_id"]
    summaries = {}
    for view in cfg["views"]:
        sid = session_id_for(run_id, view)
        if (store / sid / "session.json").exists():
            print(f"[ingest] {sid}: present, kept")
            summaries[sid] = {"kept": True}
            continue
        t0 = time.time()
        s = adapter.ingest_run(rd, store, session_id=sid, conventions=conv, options={**cfg["ingest_options"], "view": view})
        s["seconds"] = time.time() - t0
        summaries[sid] = s
        print(f"[ingest] {sid}: {s['n_observations']} observations, {s['n_seam_curves']} GT seam curves, "
              f"{s['n_database_holes']} holes ({s['seconds']:.0f} s)")
    _json(out / "ingest_summary.json", summaries)
    _verify_pairs(cfg, base)          # resolves cfg["out_dir"] / cfg["store_root"] -> the new paths only
    _write_frame_list(cfg, base)      # <store>/silver_frames_dual.csv, the new store only


# ==================================================================================================
# score
# ==================================================================================================

class _Routed:
    """One curve source over two stores: the new run's sessions and the frozen EXP-SKY-011 sessions."""

    def __init__(self, new_src, frozen_src, new_ids: set):
        self.new, self.frozen, self.new_ids = new_src, frozen_src, set(new_ids)

    def get(self, oid: str):
        return (self.new if oid in self.new_ids else self.frozen).get(oid)

    def describe(self) -> dict:
        return {"new_run": self.new.describe(), "frozen_references": self.frozen.describe()}


def _sources(cfg, base, sess_new, sess_old, cam) -> dict:
    from hsreloc.simret.sources import SessionSilverSource, SimExactSource
    store = _resolve(base, cfg["store_root"])
    frozen_store = _resolve(base, cfg["frozen"]["store_root"])
    out = {}
    for key in cfg["sources"]:
        if key == "sim_exact":
            new = SimExactSource(store, sess_new, cam.width_px, cam.height_px)
            old = SimExactSource(frozen_store, sess_old, cam.width_px, cam.height_px)
        elif key == "segformer":
            spec = cfg["segformer"]
            kw = dict(scale_short_side=spec.get("scale_short_side", 512), closing_px=spec.get("closing_px", 0),
                      min_valid_frac=spec.get("min_valid_frac", 0.5),
                      expected_model_revision=spec.get("expected_model_revision"))
            new = SessionSilverSource(Path(spec["mask_root"]), sess_new, cam.width_px, cam.height_px, **kw)
            old = SessionSilverSource(Path(cfg["frozen"]["segformer_mask_root"]), sess_old, cam.width_px, cam.height_px, **kw)
        else:
            raise SystemExit(f"unsupported source {key!r}")
        out[key] = _Routed(new, old, set(sess_new))
    return out


def _oid(key: str, view: str) -> str:
    run, raw = key.split("/", 1)
    return observation_id_for(run, view, raw)


def stage_score(cfg: dict, base: Path, sources_only=None) -> None:
    from hsreloc.simret.relpose import frozen_c1
    from hsreloc.simret.sources import session_map
    from sky_relpose_study import _Curves, _camera
    out = _resolve(base, cfg["out_dir"])
    store = _resolve(base, cfg["store_root"])
    frozen_store = _resolve(base, cfg["frozen"]["store_root"])
    frozen_cells = _resolve(base, cfg["frozen_out_dir"]) / "cells"
    rows = _index_rows(cfg, base)
    if not all(r["pair_ok"] for r in rows):
        raise SystemExit("index.csv has unpaired observations — run the ingest stage first")
    new_sessions = sorted({r[f"{v}_session_id"] for r in rows for v in cfg["views"]})
    old_sessions = [session_id_for(rid, v) for rid in cfg["frozen_city_run_ids"] for v in cfg["views"]]
    sess_new = session_map(store, new_sessions)
    sess_old = session_map(frozen_store, old_sessions)
    cam = _camera(store, new_sessions, int(cfg["profile_n_samples"]))
    cam_old = _camera(frozen_store, old_sessions, int(cfg["profile_n_samples"]))
    if cam.as_dict() != cam_old.as_dict():
        raise SystemExit(f"camera model differs from the frozen city sessions: {cam.as_dict()} vs {cam_old.as_dict()}")
    if sources_only:
        cfg = {**cfg, "sources": [k for k in cfg["sources"] if k in sources_only]}
    sources = _sources(cfg, base, sess_new, sess_old, cam)
    max_lag = int(cfg["max_lag_samples"])
    every = int(cfg["frozen_check_every"])
    n_check = int(cfg["memory_check_queries"])
    cells = out / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    frozen_index = _read_csv(_resolve(base, cfg["frozen_out_dir"]) / "index.csv")
    frozen_anchor_rows = [r for r in frozen_index if r["level_key"] == LEVEL and r["segment_kind"] == "anchor_capture"]
    manifest = {"study_version": STUDY_VERSION,
                "matcher": {"max_lag_samples": max_lag, "min_overlap_frac": cfg["min_overlap_frac"],
                            "n_samples": cfg["profile_n_samples"]},
                "camera": cam.as_dict(), "frozen_reference_source": str(frozen_cells), "cells": {},
                "sources": {k: s.describe() for k, s in sources.items()}}
    queries_all = [r for r in rows if r["segment_kind"] != "anchor_capture"]
    for source, src in sources.items():
        cache = _Curves(src)
        with np.load(frozen_cells / f"scores_{LEVEL}_{source}_north.npz", allow_pickle=False) as z:
            ref_keys = [str(x) for x in z["reference_keys"]]
            ref_xy = np.asarray(z["reference_xy"], dtype=np.float64)
            ref_seg = [str(x) for x in z["reference_segments"]]
            ref_anchor = np.asarray(z["reference_is_anchor"], dtype=bool)
            frozen_qkeys = [str(x) for x in z["query_keys"]]
        # the SAME physical references, both views (paired list, frozen order)
        for view in cfg["views"]:
            for k in ref_keys:
                if cache.get(_oid(k, view)) is None:
                    raise SystemExit(f"{source}/{view}: frozen reference {k} has no curve now "
                                     f"({cache.refused.get(_oid(k, view))}) — the frozen memory cannot be rebuilt")

        def both(r):
            return all(cache.get(r[f"{v}_observation_id"]) is not None for v in cfg["views"])
        queries = [r for r in queries_all if both(r)]
        holes = {v: [r["observation_raw_id"] for r in queries_all if cache.get(r[f"{v}_observation_id"]) is None]
                 for v in cfg["views"]}
        info = {"level": LEVEL, "source": source, "n_references_paired": len(ref_keys),
                "n_anchor_references": int(ref_anchor.sum()), "n_queries_paired": len(queries),
                "n_query_candidates": len(queries_all), "holes_by_view": {v: len(h) for v, h in holes.items()},
                "query_holes": holes, "views": {}}
        for view in cfg["views"]:
            ids = [_oid(k, view) for k in ref_keys]
            profs = [cache.profiles[i] for i in ids]
            bank = recog.ReferenceBank(ids, profs, max_lag=max_lag, min_overlap_frac=float(cfg["min_overlap_frac"]))
            # ---- memory check: the rebuilt bank must reproduce frozen rows exactly ---------------
            with np.load(frozen_cells / f"scores_{LEVEL}_{source}_{view}.npz", allow_pickle=False) as zf:
                if [str(x) for x in zf["reference_keys"]] != ref_keys:
                    raise SystemExit(f"{source}/{view}: frozen reference order differs between views")
                Sf, Lf = zf["scores"], zf["lags"]
            pick = sorted(set(np.linspace(0, len(frozen_qkeys) - 1, n_check).astype(int).tolist()))
            worst_s, lag_mismatch = 0.0, 0
            for qi in pick:
                qp_id = _oid(frozen_qkeys[qi], view)
                if cache.get(qp_id) is None:
                    raise SystemExit(f"{source}/{view}: frozen query {frozen_qkeys[qi]} has no curve now")
                s, l = bank.score(cache.profiles[qp_id])
                fin = np.isfinite(Sf[qi]) & np.isfinite(s)
                if not np.array_equal(np.isfinite(Sf[qi]), np.isfinite(s)):
                    raise SystemExit(f"{source}/{view}: finite pattern differs for frozen query {frozen_qkeys[qi]}")
                worst_s = max(worst_s, float(np.max(np.abs(Sf[qi][fin] - s[fin]))) if fin.any() else 0.0)
                lag_mismatch += int((Lf[qi][fin] != l[fin]).sum())
            if worst_s > 1e-12 or lag_mismatch:
                raise SystemExit(f"{source}/{view}: rebuilt memory does NOT reproduce the frozen matrix "
                                 f"(max |dscore| {worst_s}, {lag_mismatch} lag mismatches over {len(pick)} frozen queries)")
            # ---- the new queries -----------------------------------------------------------
            S = np.empty((len(queries), len(ref_keys)), dtype=np.float64)
            L = np.empty_like(S)
            checks, t_score = 0, 0.0
            for qi, q in enumerate(queries):
                qp = cache.profiles[q[f"{view}_observation_id"]]
                t1 = time.time()
                scores, lags = bank.score(qp)
                t_score += time.time() - t1
                if qi % every == 0:
                    j = (qi * 7) % len(ids)
                    fr = frozen_c1(qp, profs[j], max_lag=max_lag)
                    if not (abs(fr.score - scores[j]) <= 1e-12 and fr.shift == lags[j]):
                        raise SystemExit(f"frozen cross-check FAILED {source}/{view} {_key(q)} vs {ids[j]}")
                    checks += 1
                S[qi], L[qi] = scores, lags
            np.savez_compressed(cells / f"scores_{LEVEL}_{source}_{view}.npz", scores=S, lags=L,
                                query_keys=np.asarray([_key(q) for q in queries]),
                                query_segments=np.asarray([q["segment_id"] for q in queries]),
                                reference_keys=np.asarray(ref_keys), reference_segments=np.asarray(ref_seg),
                                reference_xy=ref_xy,
                                query_xy=np.asarray([(q["east_m"], q["north_m"]) for q in queries], dtype=np.float64),
                                reference_is_anchor=ref_anchor)
            info["views"][view] = {"n_queries": len(queries), "n_references": len(ref_keys), "frozen_cross_checks": checks,
                                   "memory_check": {"n_frozen_queries_rescored": len(pick), "max_abs_score_diff": worst_s,
                                                    "lag_mismatches": lag_mismatch, "verdict": "identical_memory"},
                                   "score_s_per_query": t_score / max(1, len(queries)),
                                   "c1_check": "bank_lag_exact_score_1e-12"}
            print(f"[score] {source}/{view}: memory check {len(pick)} frozen queries max|ds| {worst_s:.1e}, "
                  f"{lag_mismatch} lag mismatches; {len(queries)} new queries x {len(ref_keys)} references, "
                  f"{checks} frozen cross-checks")
            # ---- diagnostic: North against the UNPAIRED full anchor memory (EXP-SKY-010 shape) --
            if view == "north":
                a_ids, a_keys, a_xy, a_holes = [], [], [], []
                for r in frozen_anchor_rows:
                    oid = r["north_observation_id"]
                    if cache.get(oid) is None:
                        a_holes.append(r["observation_raw_id"])
                        continue
                    a_ids.append(oid)
                    a_keys.append(_key(r))
                    a_xy.append((float(r["east_m"]), float(r["north_m"])))
                bank_a = recog.ReferenceBank(a_ids, [cache.profiles[i] for i in a_ids], max_lag=max_lag,
                                             min_overlap_frac=float(cfg["min_overlap_frac"]))
                q_n = [r for r in queries_all if cache.get(r["north_observation_id"]) is not None]
                Sa = np.empty((len(q_n), len(a_ids)))
                La = np.empty_like(Sa)
                for qi, q in enumerate(q_n):
                    Sa[qi], La[qi] = bank_a.score(cache.profiles[q["north_observation_id"]])
                np.savez_compressed(cells / f"scores_{LEVEL}_{source}_north_allanchors.npz", scores=Sa, lags=La,
                                    query_keys=np.asarray([_key(q) for q in q_n]),
                                    reference_keys=np.asarray(a_keys), reference_xy=np.asarray(a_xy, dtype=np.float64),
                                    query_xy=np.asarray([(q["east_m"], q["north_m"]) for q in q_n], dtype=np.float64),
                                    reference_is_anchor=np.ones(len(a_ids), dtype=bool))
                info["north_all_anchors"] = {"n_anchor_references": len(a_ids), "anchor_holes": a_holes, "n_queries": len(q_n)}
        manifest["cells"][f"{LEVEL}/{source}"] = info
    mpath = out / "score_manifest.json"
    if mpath.exists():
        old = json.loads(mpath.read_text(encoding="utf-8"))
        manifest["cells"] = {**old.get("cells", {}), **manifest["cells"]}
        manifest["sources"] = {**old.get("sources", {}), **manifest["sources"]}
    _json(mpath, manifest)
    print(f"[score] manifest -> {mpath}")


# ==================================================================================================
# report
# ==================================================================================================

def _gate_from(sel: dict) -> dv.Gate:
    return dv.Gate(float(sel["theta_s"]), float(sel["theta_m"]), sel["level"])


def veto_accepted(row: dict, gate: dv.Gate) -> bool:
    """North-primary / West-veto: North passes; West may only contradict (pass its gate with a top-1
    outside North's region). A West refusal leaves North's decision standing."""
    n, w = row["_out"]["north"], row["_out"]["west"]
    return bool(dv.view_passes(n, gate) and not (dv.view_passes(w, gate) and not row["sa_agree"]))


def _acc(row, rule, gate):
    return veto_accepted(row, gate) if rule == "veto" else accepted_at(row, rule, gate)


def _false(row, rule):
    return bool(row["_out"]["north"]["false_region"]) if rule == "veto" else false_of(row, rule)


def _top1(row, rule):
    return bool(row["_out"]["north"]["region_top1"]) if rule == "veto" else top1_of(row, rule)


def _cand(row, rule):
    if rule == "strict":
        return row["sa_candidate_id"]
    ch = "north" if rule == "veto" else rule
    return row["_out"][ch]["top1_id"]


def _cand_d(row, rule):
    if rule == "strict":
        return row["sa_candidate_distance_m"]
    ch = "north" if rule == "veto" else rule
    return row["_out"][ch]["top1_distance_m"]


ALL_RULES = tuple(RULES) + ("veto",)
LABEL = {**RULE_LABEL, "veto": "North-primary / West-veto"}


def summ(rows: list, rule: str, gate: dv.Gate) -> dict:
    if rule != "veto":
        e = summarize(rows, rule, gate)
    else:
        n = len(rows)
        acc = [r for r in rows if veto_accepted(r, gate)]
        fal = [r for r in acc if _false(r, "veto")]
        e = {"rule": rule, "theta_s": gate.theta_s, "theta_m": gate.theta_m, "gate_level": gate.level, "n": n,
             "region_top1": (sum(1 for r in rows if _top1(r, "veto")) / n) if n else None,
             "region_recall_k": (sum(1 for r in rows if topk_of(r, "north")) / n) if n else None,
             "n_accepted": len(acc), "coverage": (len(acc) / n) if n else None, "n_accepted_false": len(fal),
             "accepted_false_rate": (len(fal) / n) if n else None,
             "p_false_given_accepted": (len(fal) / len(acc)) if acc else None}
    acc = [r for r in rows if _acc(r, rule, gate)]
    cands = defaultdict(int)
    for r in acc:
        cands[str(_cand(r, rule))] += 1
    e["accepted_candidates"] = dict(sorted(cands.items(), key=lambda kv: -kv[1]))
    e["accepted_candidate_distance_m"] = sorted({round(float(_cand_d(r, rule)), 1) for r in acc
                                                 if _cand_d(r, rule) is not None})
    return e


def _pct(v):
    return "—" if v is None else f"{100 * v:.0f} %"


def _num(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def _unflatten(fr: dict) -> dict:
    """A frozen EXP-SKY-011 queries.csv.gz row back into the shape build_rows produces (read-only use)."""
    def b(v):
        return v == "True"

    def f(v):
        x = _f(v)
        return None if (isinstance(x, float) and math.isnan(x)) else x
    row = {k: fr[k] for k in ("level", "split", "source", "memory", "query_key", "segment_id", "segment_kind", "site",
                              "phase", "leg_axis", "leg_direction", "gate_level")}
    for k in ("sa_candidate_id", "nearest_key"):          # an absent candidate is written as ""
        row[k] = fr[k] or None
    row["holdout_m"] = f(fr["holdout_m"])
    for k in ("hard_tag", "primary", "horizontal", "attainable", "sa_agree", "sa_false_region",
              "sa_dual_region_top1", "sa_dual_region_in_top_k"):
        row[k] = b(fr[k])
    row["sa_candidate_distance_m"] = f(fr["sa_candidate_distance_m"])
    row["sa_min_score"] = f(fr["sa_min_score"])
    row["index_in_segment"] = int(fr["index_in_segment"])
    row["east_m"], row["north_m"] = f(fr["east_m"]), f(fr["north_m"])
    row["_out"] = {}
    for c in CHANNELS:
        o = {}
        for k in KEEP_FIELDS:
            v = fr.get(f"{c}_{k}")
            if k in ("region_top1", "region_in_top_k", "exact_top1", "exact_in_top_k", "false_region", "ambiguous",
                     "accepted_frame", "accepted_region"):
                o[k] = b(v)
            elif k in ("top1_id", "outcome"):
                o[k] = v or None
            else:
                o[k] = f(v)
        row["_out"][c] = o
    return row


POP_NAMES = ("hard_sparse_h", "hard_sparse_v", "hard_dense_lso_h", "hard_dense_lso_v", "hard_dense0_h",
             "hard_dense0_h_attainable", "hard_dense0_v")
MEM_OF = {"hard_sparse_h": "sparse", "hard_sparse_v": "sparse", "hard_dense_lso_h": "dense", "hard_dense_lso_v": "dense",
          "hard_dense0_h": "dense", "hard_dense0_h_attainable": "dense", "hard_dense0_v": "dense"}
HARD_POPS = ("hard_sparse_h", "hard_dense_lso_h")


def _pops(rows):
    return {
        "hard_sparse_h": [r for r in rows if r["memory"] == "sparse" and r["horizontal"]],
        "hard_sparse_v": [r for r in rows if r["memory"] == "sparse" and not r["horizontal"]],
        "hard_dense_lso_h": [r for r in rows if r["memory"] == "dense_lso" and r["horizontal"]],
        "hard_dense_lso_v": [r for r in rows if r["memory"] == "dense_lso" and not r["horizontal"]],
        "hard_dense0_h": [r for r in rows if r["memory"] == "dense" and r["holdout_m"] == 0.0 and r["horizontal"]],
        "hard_dense0_h_attainable": [r for r in rows if r["memory"] == "dense" and r["holdout_m"] == 0.0
                                     and r["horizontal"] and r["attainable"]],
        "hard_dense0_v": [r for r in rows if r["memory"] == "dense" and r["holdout_m"] == 0.0 and not r["horizontal"]],
    }


def stage_report(cfg: dict, base: Path) -> None:
    out = _resolve(base, cfg["out_dir"])
    cells = out / "cells"
    frozen_out = _resolve(base, cfg["frozen_out_dir"])
    fm = json.loads((frozen_out / "metrics.json").read_text(encoding="utf-8"))
    sel = fm["selected_gates"]
    meta = {_key(r): r for r in _index_rows(cfg, base)}
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    tau, tau_agree = float(cfg["tau_region_m"]), float(cfg["tau_agree_m"])
    grid_s, grid_m = cfg["gate_grid"]["theta_s"], cfg["gate_grid"]["theta_m"]
    frozen_frame, frozen_region = dv.Gate(*FROZEN_GATE, "frame"), dv.Gate(*FROZEN_GATE, "region")

    def gate(mem, source, rule):
        if rule == "veto":
            rule = "north"                       # the veto rule runs both views at North's DEV gate
        return _gate_from(sel[f"{mem}/{source}/{rule}"])

    metrics = {"study_version": STUDY_VERSION, "dual_study_version": DUAL_STUDY_VERSION, "level": LEVEL,
               "frozen_gates_source": str(frozen_out / "metrics.json"),
               "frozen_baseline_gate": {"theta_s": FROZEN_GATE[0], "theta_m": FROZEN_GATE[1]},
               "selected_gates": dict(sel),
               "audit": {k: audit[k] for k in ("run_id", "raw_path_id", "n_observations", "site_match", "sparse_memory_geometry",
                                               "dense_memory_geometry", "north_west_position_max_diff_m",
                                               "settings_vs_frozen_city_run")},
               "sources": [], "population_sizes": {}, "populations": {}, "north_false_accepts": {}, "elimination": {},
               "temporal": {}, "sensitivity_grid": {}, "north_all_anchors": {}, "combined_hard_negatives": {},
               "veto_on_frozen_exp011": {}, "score_distributions": {}}
    rows_by, ref_xy_by = {}, {}
    for source in cfg["sources"]:
        pn, pw = cells / f"scores_{LEVEL}_{source}_north.npz", cells / f"scores_{LEVEL}_{source}_west.npz"
        if not (pn.exists() and pw.exists()):
            print(f"[report] {source}: matrices absent, skipped")
            continue
        with np.load(pn, allow_pickle=False) as zN, np.load(pw, allow_pickle=False) as zW:
            rows, ref_xy = build_rows(cfg, LEVEL, source, zN, zW, meta)
        rows_by[source] = rows
        ref_xy_by[source] = ref_xy
        metrics["sources"].append(source)
        print(f"[report] {source}: {len(rows)} query x memory rows")
    if not rows_by:
        raise SystemExit("no score matrices — run the score stage")

    for source, rows in rows_by.items():
        P = _pops(rows)
        for name, p in P.items():
            metrics["population_sizes"][f"{source}/{name}"] = {
                "n": len(p), "n_attainable": sum(1 for r in p if r["attainable"]),
                "d_near_m_median": (float(np.median([r["d_near_m"] for r in p])) if p else None)}
            mem = MEM_OF[name]
            for rule in ALL_RULES:
                g_dev = gate(mem, source, rule)
                g_fr = frozen_frame if mem == "sparse" else frozen_region
                metrics["populations"][f"{source}/{name}/{rule}/dev"] = summ(p, rule, g_dev)
                metrics["populations"][f"{source}/{name}/{rule}/frozen"] = summ(p, rule, g_fr)
        sp = P["hard_sparse_h"]
        metrics["score_distributions"][source] = {
            "north_top1_median": float(np.median([r["_out"]["north"]["top1_score"] for r in sp])),
            "west_top1_median": float(np.median([r["_out"]["west"]["top1_score"] for r in sp])),
            "north_frame_margin_median": float(np.median([r["_out"]["north"]["frame_margin"] for r in sp
                                                          if r["_out"]["north"]["frame_margin"] is not None])),
            "west_frame_margin_median": float(np.median([r["_out"]["west"]["frame_margin"] for r in sp
                                                         if r["_out"]["west"]["frame_margin"] is not None])),
            "north_top1_ge_0.99": int(sum(1 for r in sp if (r["_out"]["north"]["top1_score"] or 0) >= 0.99)),
            "north_top1_ge_0.95": int(sum(1 for r in sp if (r["_out"]["north"]["top1_score"] or 0) >= 0.95)),
            "north_top1_ge_0.90": int(sum(1 for r in sp if (r["_out"]["north"]["top1_score"] or 0) >= 0.90)),
            "n": len(sp)}

        # ---- what West does on every North false accept ------------------------------------
        for name in HARD_POPS + ("hard_sparse_v", "hard_dense_lso_v"):
            mem = MEM_OF[name]
            for tag, g in (("frozen", frozen_frame if mem == "sparse" else frozen_region), ("dev", gate(mem, source, "north"))):
                g_min = gate(mem, source, "min")
                g_mean = gate(mem, source, "mean")
                cases = [r for r in P[name] if dv.view_passes(r["_out"]["north"], g) and r["_out"]["north"]["false_region"]]
                detail, cat = [], defaultdict(int)
                for r in cases:
                    n, w = r["_out"]["north"], r["_out"]["west"]
                    w_pass = dv.view_passes(w, g)
                    if w_pass and r["sa_agree"]:
                        c = "west_agrees_same_wrong_region"
                    elif w_pass:
                        c = "west_disagrees_different_region"
                    else:
                        c = "west_refuses_fails_gate"
                    cat[c] += 1
                    detail.append({"query": r["query_key"], "index_in_segment": r["index_in_segment"], "leg_axis": r["leg_axis"],
                                   "leg_direction": r["leg_direction"], "d_near_m": r["d_near_m"],
                                   "north_top1": n["top1_id"], "north_top1_distance_m": n["top1_distance_m"],
                                   "north_score": n["top1_score"], "north_frame_margin": n["frame_margin"],
                                   "north_region_level_margin": n["region_level_margin"],
                                   "west_top1": w["top1_id"], "west_top1_distance_m": w["top1_distance_m"],
                                   "west_score": w["top1_score"], "west_frame_margin": w["frame_margin"],
                                   "west_region_level_margin": w["region_level_margin"],
                                   "west_passes_gate": w_pass, "west_false_region": w["false_region"], "agree": r["sa_agree"],
                                   "top1_separation_m": r["sa_top1_separation_m"], "category": c,
                                   "strict_accepted": accepted_at(r, "strict", g),
                                   "min_accepted_at_dev_gate": accepted_at(r, "min", g_min),
                                   "mean_accepted_at_dev_gate": accepted_at(r, "mean", g_mean),
                                   "veto_accepted": veto_accepted(r, g)})
                wrong = defaultdict(int)
                for d in detail:
                    wrong[str(d["north_top1"])] += 1
                metrics["north_false_accepts"][f"{source}/{name}/{tag}"] = {
                    "gate": {"theta_s": g.theta_s, "theta_m": g.theta_m, "level": g.level}, "n_population": len(P[name]),
                    "n_north_accepted_false": len(cases), "west_categories": dict(cat),
                    "n_survive_strict": sum(1 for d in detail if d["strict_accepted"]),
                    "n_survive_min": sum(1 for d in detail if d["min_accepted_at_dev_gate"]),
                    "n_survive_mean": sum(1 for d in detail if d["mean_accepted_at_dev_gate"]),
                    "n_survive_veto": sum(1 for d in detail if d["veto_accepted"]),
                    "north_wrong_candidates": dict(sorted(wrong.items(), key=lambda kv: -kv[1])),
                    "detail": detail}
                if tag == "frozen" and name in HARD_POPS:
                    metrics["elimination"][f"{source}/{name}"] = {
                        "north_accepted_false_at_frozen_gate": len(cases),
                        "eliminated_by_west_refusal": cat.get("west_refuses_fails_gate", 0),
                        "eliminated_by_west_disagreement": cat.get("west_disagrees_different_region", 0),
                        "survive_strict_agreement": cat.get("west_agrees_same_wrong_region", 0)}

        # ---- temporal (North k=1..3 on the single segment; free) ------------------------------
        rx = ref_xy_by[source]
        for name in HARD_POPS:
            mem = MEM_OF[name]
            for tag, gN in (("frozen", frozen_frame if mem == "sparse" else frozen_region), ("dev", gate(mem, source, "north"))):
                res = {}
                for kk in (1, 2, 3):
                    tr = temporal_rows(P[name], kk, gN, rx, tau_agree, tau)
                    acc = sum(int(t["accepted"]) for t in tr.values())
                    fal = sum(int(t["accepted_false"]) for t in tr.values())
                    res[f"north_k{kk}"] = {"n": len(P[name]), "n_accepted": acc, "n_accepted_false": fal}
                metrics["temporal"][f"{source}/{name}/{tag}"] = res

        # ---- the 4x4 grid: sensitivity only, never a selection ---------------------------------
        for name in HARD_POPS:
            lev = "frame" if MEM_OF[name] == "sparse" else "region"
            for rule in ALL_RULES:
                ev = (lambda r, g, rule=rule: (_acc(r, rule, g), _false(r, rule)))
                sw = dv.sweep(P[name], grid_s, grid_m, ev, lev)
                metrics["sensitivity_grid"][f"{source}/{name}/{rule}"] = [
                    {"theta_s": e["theta_s"], "theta_m": e["theta_m"], "n_accepted": e["n_accepted"],
                     "n_accepted_false": e["n_accepted_false"]} for e in sw]

        # ---- North against the unpaired full anchor memory (EXP-SKY-010 memory shape) ----------
        pa = cells / f"scores_{LEVEL}_{source}_north_allanchors.npz"
        if pa.exists():
            with np.load(pa, allow_pickle=False) as z:
                qk = [str(x) for x in z["query_keys"]]
                rk = [str(x) for x in z["reference_keys"]]
                rxy, qxy, S, Lg = np.asarray(z["reference_xy"]), np.asarray(z["query_xy"]), z["scores"], z["lags"]
            acc = recog.Acceptance(*FROZEN_GATE)
            outs = [(k, recog.retrieval_outcome(qxy[i], rk, rxy, S[i], Lg[i], tau_region=tau, k=int(cfg["recall_k"]),
                                                acceptance=acc)) for i, k in enumerate(qk)]
            hz = [(k, o) for k, o in outs if meta[k]["leg_axis"] != "vertical"]
            metrics["north_all_anchors"][source] = {
                "n_anchor_references": len(rk), "n_queries_horizontal": len(hz), "n_queries_all": len(outs),
                "frozen_gate_accepted_horizontal": sum(1 for _, o in hz if o["accepted_frame"]),
                "frozen_gate_accepted_false_horizontal": sum(1 for _, o in hz if o["accepted_frame"] and o["false_region"]),
                "frozen_gate_accepted_all": sum(1 for _, o in outs if o["accepted_frame"]),
                "top1_score_median_horizontal": float(np.median([o["top1_score"] for _, o in hz])),
                "wrong_candidates": {c: sum(1 for _, o in hz if o["accepted_frame"] and o["top1_id"] == c)
                                     for c in sorted({o["top1_id"] for _, o in hz if o["accepted_frame"]})}}

    # ---- combined hard-negative evidence (EXP-SKY-011 at DEV gates + this run at DEV gates) -----
    exp011 = {}
    for rule in RULES:
        n_tot = f_tot = 0
        for k_, e in fm["populations"].items():
            level, source, name, r_ = k_.split("/")
            if r_ == rule and name in ("ooc_sparse", "lso_sparse", "lso_dense"):
                n_tot += e["n"]
                f_tot += e["n_accepted_false"]
        exp011[rule] = {"n": n_tot, "n_accepted_false": f_tot}
    for rule in ALL_RULES:
        new_n = new_f = 0
        for source in rows_by:
            for name in HARD_POPS:
                e = metrics["populations"][f"{source}/{name}/{rule}/dev"]
                new_n += e["n"]
                new_f += e["n_accepted_false"]
        base_ = exp011.get(rule)
        metrics["combined_hard_negatives"][rule] = {
            "exp011_n": base_["n"] if base_ else None, "exp011_accepted_false": base_["n_accepted_false"] if base_ else None,
            "this_run_n": new_n, "this_run_accepted_false": new_f,
            "combined_n": (base_["n"] + new_n) if base_ else None,
            "combined_accepted_false": (base_["n_accepted_false"] + new_f) if base_ else None}

    # ---- the veto rule on the frozen EXP-SKY-011 populations (read-only) ------------------------
    fq = frozen_out / "queries.csv.gz"
    if fq.exists():
        frows = [_unflatten(r) for r in _read_csv(fq)]
        att_s, att_d = {}, {}
        for r in frows:
            if r["memory"] == "sparse":
                att_s[(r["level"], r["source"], r["query_key"])] = r["attainable"]
            if r["memory"] == "dense" and r["holdout_m"] == 0.0:
                att_d[(r["level"], r["source"], r["query_key"])] = r["attainable"]

        def fpop(level, source, name):
            rs = [r for r in frows if r["level"] == level and r["source"] == source]
            if name == "in_coverage_sparse":
                return [r for r in rs if r["memory"] == "sparse" and r["primary"] and r["attainable"]]
            if name == "ooc_sparse":
                return [r for r in rs if r["memory"] == "sparse" and r["horizontal"] and not r["attainable"]]
            if name == "lso_sparse":
                return [r for r in rs if r["memory"] == "sparse_lso" and r["primary"] and att_s.get((level, source, r["query_key"]))]
            if name == "in_coverage_dense":
                return [r for r in rs if r["memory"] == "dense" and r["holdout_m"] == 0.0 and r["primary"] and r["attainable"]]
            if name == "lso_dense":
                return [r for r in rs if r["memory"] == "dense_lso" and r["primary"] and att_d.get((level, source, r["query_key"]))]
            raise KeyError(name)
        for level in ("village", "mountains", "city"):
            for source in cfg["sources"]:
                for name in ("in_coverage_sparse", "ooc_sparse", "lso_sparse", "in_coverage_dense", "lso_dense"):
                    mem = "sparse" if name.endswith("sparse") else "dense"
                    p = fpop(level, source, name)
                    if not p:
                        continue
                    entry = {}
                    for rule in ("north", "strict", "min", "veto"):
                        g = gate(mem, source, rule)
                        e = summ(p, rule, g)
                        entry[rule] = {"n": e["n"], "n_accepted": e["n_accepted"], "n_accepted_false": e["n_accepted_false"],
                                       "coverage": e["coverage"]}
                        ref = fm["populations"].get(f"{level}/{source}/{name}/{rule}")
                        if ref is not None:
                            entry[rule]["reproduces_frozen_metrics"] = (ref["n_accepted"] == e["n_accepted"]
                                                                        and ref["n_accepted_false"] == e["n_accepted_false"])
                    metrics["veto_on_frozen_exp011"][f"{level}/{source}/{name}"] = entry
        vt = {"n": 0, "n_accepted_false": 0, "n_in_cov": 0, "n_in_cov_accepted": 0}
        for k_, e in metrics["veto_on_frozen_exp011"].items():
            name = k_.split("/")[-1]
            if name in ("ooc_sparse", "lso_sparse", "lso_dense"):
                vt["n"] += e["veto"]["n"]
                vt["n_accepted_false"] += e["veto"]["n_accepted_false"]
            elif name == "in_coverage_sparse":
                vt["n_in_cov"] += e["veto"]["n"]
                vt["n_in_cov_accepted"] += e["veto"]["n_accepted"]
        cv = metrics["combined_hard_negatives"]["veto"]
        cv.update({"exp011_n": vt["n"], "exp011_accepted_false": vt["n_accepted_false"],
                   "combined_n": vt["n"] + cv["this_run_n"],
                   "combined_accepted_false": vt["n_accepted_false"] + cv["this_run_accepted_false"]})
        metrics["veto_exp011_in_coverage_sparse"] = vt

    metrics["classification"] = classify(metrics)
    all_rows = [flat(r) for rows in rows_by.values() for r in rows]
    _write_csv(out / "queries.csv.gz", all_rows, list(all_rows[0].keys()))
    _json(out / "metrics.json", metrics)
    nfa = []
    for k_, e in metrics["north_false_accepts"].items():
        for d in e["detail"]:
            nfa.append({"case": k_, **d})
    if nfa:
        _write_csv(out / "north_false_accepts.csv", nfa, list(nfa[0].keys()))
    _fig_scores(out, rows_by, sel)
    write_report(out, cfg, metrics)
    print(f"[report] -> {out / 'report.md'}")


def classify(m: dict) -> dict:
    """Declared in the EXP-SKY-012 record before scoring. Horizontal hard-swipe queries, DEV-frozen
    gates, both sources, sparse and dense-LSO memories. Primary rule strict agreement; weakest view
    is the secondary rule that must also be false-free for C/D."""
    srcs = m["sources"]

    def tot(rule, tag="dev"):
        n = f = 0
        for s in srcs:
            for name in HARD_POPS:
                e = m["populations"][f"{s}/{name}/{rule}/{tag}"]
                n += e["n"]
                f += e["n_accepted_false"]
        return n, f
    n_hard, strict_f = tot("strict")
    _, min_f = tot("min")
    _, north_f_frozen = tot("north", "frozen")
    _, north_f_dev = tot("north")
    north_sparse_frozen_segf = m["populations"].get("segformer/hard_sparse_h/north/frozen", {})
    informative = north_f_frozen > 0
    reproduced = (north_sparse_frozen_segf.get("accepted_false_rate") or 0.0) >= 0.20
    cov_ok = False
    for s in srcs:
        for rule in ("strict", "min"):
            e = m["populations"].get(f"{s}/hard_dense0_h_attainable/{rule}/dev")
            if e and e["n"] >= 10 and e["n_accepted_false"] == 0 and (e["coverage"] or 0) >= 0.25:
                cov_ok = True
    if strict_f == 0 and min_f == 0:
        label = "D" if cov_ok else "C"
    elif strict_f <= 0.02 * n_hard and strict_f <= 0.5 * max(north_f_frozen, north_f_dev, 1):
        label = "B"
    else:
        label = "A"
    return {"label": label, "n_hard_negative_queries_this_run": n_hard, "strict_accepted_false": strict_f,
            "weakest_view_accepted_false": min_f, "north_accepted_false_frozen_gate": north_f_frozen,
            "north_accepted_false_dev_gate": north_f_dev, "informative_single_view_false_present": informative,
            "exp010_behaviour_reproduced_ge_20pct_segformer_frozen_gate": reproduced,
            "city_coverage_condition_for_D": cov_ok,
            "combined_strict": m["combined_hard_negatives"]["strict"], "combined_min": m["combined_hard_negatives"]["min"]}


def _fig_scores(out, rows_by, sel):
    plt = _plt()
    fig, axes = plt.subplots(2, len(rows_by), figsize=(7.5 * len(rows_by), 8), squeeze=False)
    for j, (source, rows) in enumerate(rows_by.items()):
        sp = sorted([r for r in rows if r["memory"] == "sparse"], key=lambda r: r["index_in_segment"])
        x = [r["index_in_segment"] for r in sp]
        for i, (field, lab) in enumerate((("top1_score", "top-1 score"), ("frame_margin", "frame margin (top-1 − top-2)"))):
            ax = axes[i, j]
            for ch, col in (("north", "#1f77b4"), ("west", "#d62728")):
                y = [r["_out"][ch][field] if r["_out"][ch][field] is not None else np.nan for r in sp]
                ax.plot(x, y, "-", color=col, lw=1, label=ch)
            for v in [r["index_in_segment"] for r in sp if not r["horizontal"]]:
                ax.axvspan(v - 0.5, v + 0.5, color="#eeeeee", zorder=0)
            if field == "top1_score":
                ax.axhline(0.90, color="grey", ls=":", lw=0.8)
                ax.axhline(0.95, color="grey", ls="--", lw=0.8)
            else:
                ax.axhline(0.15, color="grey", ls=":", lw=0.8)
                ax.axhline(float(sel[f"sparse/{source}/north"]["theta_m"]), color="#1f77b4", ls="--", lw=0.8)
            ax.set_title(f"{SRC_LABEL[source]}: {lab} vs capture index (sparse memory; grey = vertical legs)", fontsize=9)
            ax.set_xlabel("capture index in the HardSwipe")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    _save(fig, out / "figures" / "fig1_scores_along_hard_swipe.jpg")


def write_report(out: Path, cfg: dict, m: dict) -> None:
    L = [f"# EXP-SKY-012 — dual-view validation on the missing city HardSwipe: report\n",
         f"Generated by `scripts/sky_dual_hardcity.py --stage report` (study {STUDY_VERSION}; EXP-SKY-011 machinery "
         f"{DUAL_STUDY_VERSION}). Frozen C1-32; τ_region = {cfg['tau_region_m']} m, τ_agree = {cfg['tau_agree_m']} m; gates read from "
         f"`{Path(m['frozen_gates_source']).as_posix()}` (DEV-frozen) and the frozen baseline {FROZEN_GATE[0]}/{FROZEN_GATE[1]}; "
         f"sources {m['sources']}. Evidence tier T2. No threshold chosen on this run.\n"]
    a = m["audit"]
    L.append("## Data audit\n")
    L.append(f"- run `{a['run_id']}` (raw path_id `{a['raw_path_id']}`), {a['n_observations']} observations; site `{a['site_match']['site_id']}` "
             f"(hard = {a['site_match']['hard_tag']}); hub {a['site_match']['hub_to_old_centre_pass_m']:.2f} m from the old centre pass, "
             f"start frame {a['site_match']['start_frame_to_old_run_start_frame_m']:.2f} m from the old run's start frame; "
             f"North/West position difference max {a['north_west_position_max_diff_m']} m; settings equal to the frozen city run: "
             f"{a['settings_vs_frozen_city_run']['all_equal']}.")
    sg = a["sparse_memory_geometry"]
    L.append(f"- sparse memory: nearest anchor {sg['nearest_anchor_m']['min']:.1f} / {sg['nearest_anchor_m']['median']:.1f} / "
             f"{sg['nearest_anchor_m']['max']:.1f} m (min/median/max), anchors within τ: {sg['n_queries_with_an_anchor_within_tau']} queries — "
             f"out of coverage by construction: {sg['out_of_coverage_by_construction']}.")
    for s, dg in a["dense_memory_geometry"].items():
        L.append(f"- dense memory ({SRC_LABEL[s]}): {dg['n_references_paired']} paired references ({dg['n_anchor_references_paired']} anchors); "
                 f"{dg['n_queries_with_a_dense_reference_within_tau']} queries have a dense reference within τ (nearest "
                 f"{dg['nearest_dense_reference_m']['min']:.1f}–{dg['nearest_dense_reference_m']['max']:.1f} m, segments "
                 f"{dg['nearest_dense_reference_segments']}); {dg['n_references_within_tau_of_any_query']} references lie within τ of some "
                 f"query (removed in dense-LSO).")
    L.append("\n## Population sizes\n")
    L.append("| source | population | n | attainable | d_near median |")
    L.append("|---|---|---|---|---|")
    for k_, e in sorted(m["population_sizes"].items()):
        L.append(f"| {k_.split('/')[0]} | {k_.split('/')[1]} | {e['n']} | {e['n_attainable']} | {_num(e['d_near_m_median'], 1)} |")
    L.append("\n## Score scale on the hard swipe (sparse memory, horizontal queries)\n")
    L.append("| source | N top-1 median | W top-1 median | N frame margin median | W frame margin median | N ≥ 0.90 / 0.95 / 0.99 | n |")
    L.append("|---|---|---|---|---|---|---|")
    for s, e in m["score_distributions"].items():
        L.append(f"| {SRC_LABEL[s]} | {_num(e['north_top1_median'])} | {_num(e['west_top1_median'])} | {_num(e['north_frame_margin_median'])} | "
                 f"{_num(e['west_frame_margin_median'])} | {e['north_top1_ge_0.90']} / {e['north_top1_ge_0.95']} / {e['north_top1_ge_0.99']} | {e['n']} |")
    for tag, title in (("frozen", "Frozen baseline gate 0.90 / 0.15 (the EXP-SKY-010 rule; frame level on sparse, region level on dense)"),
                       ("dev", "DEV-frozen gates of EXP-SKY-011 (applied unchanged)")):
        L.append(f"\n## {title}\n")
        L.append("| source | population | rule | gate | n | region top-1 | R@5 | accepted | acc. false | P(false ∣ acc) | coverage | accepted candidates (distance m) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in m["sources"]:
            for name in POP_NAMES:
                for rule in ALL_RULES:
                    e = m["populations"].get(f"{s}/{name}/{rule}/{tag}")
                    if not e or e["n"] == 0:
                        continue
                    cands = ", ".join(f"{k_}×{v}" for k_, v in list(e["accepted_candidates"].items())[:4])
                    L.append(f"| {SRC_LABEL[s]} | {name} | {LABEL[rule]} | {e['theta_s']}/{e['theta_m']} {e['gate_level']} | {e['n']} | "
                             f"{_pct(e['region_top1'])} | {_pct(e.get('region_recall_k'))} | {e['n_accepted']} | **{e['n_accepted_false']}** | "
                             f"{_pct(e['p_false_given_accepted'])} | {_pct(e['coverage'])} | {cands} ({e['accepted_candidate_distance_m'][:4]}) |")
    L.append("\n## North against the UNPAIRED full anchor memory (the EXP-SKY-010 memory shape), frozen gate 0.90/0.15\n")
    L.append("| source | anchors | horizontal queries | accepted | accepted false | top-1 score median | wrong candidates |")
    L.append("|---|---|---|---|---|---|---|")
    for s, e in m["north_all_anchors"].items():
        L.append(f"| {SRC_LABEL[s]} | {e['n_anchor_references']} | {e['n_queries_horizontal']} | {e['frozen_gate_accepted_horizontal']} | "
                 f"{e['frozen_gate_accepted_false_horizontal']} | {_num(e['top1_score_median_horizontal'])} | {e['wrong_candidates']} |")
    L.append("\n## What West does on every North false accept\n")
    L.append("| source | population | gate | N accepted-false | W agrees (same wrong region) | W disagrees (other region, passes) | W refuses (fails gate) | survive strict | survive weakest (DEV) | survive mean (DEV) | survive veto |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k_, e in sorted(m["north_false_accepts"].items()):
        s, name, tag = k_.split("/")
        c = e["west_categories"]
        L.append(f"| {SRC_LABEL[s]} | {name} | {tag} {e['gate']['theta_s']}/{e['gate']['theta_m']} {e['gate']['level']} | "
                 f"{e['n_north_accepted_false']} / {e['n_population']} | {c.get('west_agrees_same_wrong_region', 0)} | "
                 f"{c.get('west_disagrees_different_region', 0)} | {c.get('west_refuses_fails_gate', 0)} | "
                 f"**{e['n_survive_strict']}** | {e['n_survive_min']} | {e['n_survive_mean']} | {e['n_survive_veto']} |")
    for k_, e in sorted(m["north_false_accepts"].items()):
        if e["detail"] and k_.endswith("/frozen") and "_h/" in k_:
            L.append(f"\n**{k_}** — first 12 of {len(e['detail'])}\n")
            L.append("| query | idx | leg | N top-1 (dist) | N score / fm / rm | W top-1 (dist) | W score / fm / rm | W passes | agree | sep m | strict | veto |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for d in e["detail"][:12]:
                L.append(f"| {d['query'].split('/')[-1]} | {d['index_in_segment']} | {d['leg_axis']}/{d['leg_direction']} | "
                         f"{d['north_top1']} ({_num(d['north_top1_distance_m'], 0)}) | "
                         f"{_num(d['north_score'])} / {_num(d['north_frame_margin'])} / {_num(d['north_region_level_margin'])} | "
                         f"{d['west_top1']} ({_num(d['west_top1_distance_m'], 0)}) | "
                         f"{_num(d['west_score'])} / {_num(d['west_frame_margin'])} / {_num(d['west_region_level_margin'])} | "
                         f"{d['west_passes_gate']} | {d['agree']} | {_num(d['top1_separation_m'], 0)} | {d['strict_accepted']} | {d['veto_accepted']} |")
    L.append("\n## Temporal confirmation (North, k consecutive captures of the one segment) — free comparison\n")
    L.append("| source | population | gate | k=1 acc (false) | k=2 acc (false) | k=3 acc (false) |")
    L.append("|---|---|---|---|---|---|")
    for k_, e in sorted(m["temporal"].items()):
        s, name, tag = k_.split("/")
        L.append(f"| {SRC_LABEL[s]} | {name} | {tag} | " + " | ".join(
            f"{e[f'north_k{kk}']['n_accepted']} ({e[f'north_k{kk}']['n_accepted_false']})" for kk in (1, 2, 3)) + " |")
    L.append("\n## Combined hard-negative evidence (DEV-frozen gates): EXP-SKY-011 (OOC + LSO sparse + LSO dense, all levels and sources) + this run (sparse + dense-LSO, horizontal, both sources)\n")
    L.append("| rule | EXP-SKY-011 n | EXP-SKY-011 accepted false | this run n | this run accepted false | combined n | combined accepted false |")
    L.append("|---|---|---|---|---|---|---|")
    for rule in ALL_RULES:
        e = m["combined_hard_negatives"][rule]
        L.append(f"| {LABEL[rule]} | {e['exp011_n']} | {e['exp011_accepted_false']} | {e['this_run_n']} | {e['this_run_accepted_false']} | "
                 f"{e['combined_n']} | **{e['combined_accepted_false']}** |")
    if m["veto_on_frozen_exp011"]:
        L.append("\n## The North-primary / West-veto rule on the frozen EXP-SKY-011 populations (read-only re-evaluation; the North / strict / weakest columns must reproduce the frozen metrics)\n")
        L.append("| level/source/population | North acc (false) [repro] | strict acc (false) [repro] | weakest acc (false) [repro] | veto acc (false) |")
        L.append("|---|---|---|---|---|")
        for k_, e in sorted(m["veto_on_frozen_exp011"].items()):
            def cell(r):
                x = e[r]
                rep = x.get("reproduces_frozen_metrics")
                return f"{x['n_accepted']} ({x['n_accepted_false']})" + ("" if rep is None else f" [{'ok' if rep else 'MISMATCH'}]")
            L.append(f"| {k_} | {cell('north')} | {cell('strict')} | {cell('min')} | {cell('veto')} |")
    L.append("\n## Sensitivity — accepted / accepted-false over the 4×4 gate grid (NOT a selection; horizontal hard-swipe queries)\n")
    for k_, g in sorted(m["sensitivity_grid"].items()):
        s, name, rule = k_.split("/")
        cells_ = "; ".join(f"{e['theta_s']}/{e['theta_m']}: {e['n_accepted']}/{e['n_accepted_false']}F" for e in g)
        L.append(f"- **{SRC_LABEL[s]} {name} {LABEL[rule]}**: {cells_}")
    L.append("\n## Declared classification\n")
    L.append("```json\n" + json.dumps(m["classification"], indent=2) + "\n```\n")
    L.append("## Figures\n")
    for f in sorted((out / "figures").glob("*.jpg")):
        L.append(f"- `figures/{f.name}`")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


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
