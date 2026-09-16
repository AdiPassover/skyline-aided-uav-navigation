#!/usr/bin/env python
"""EXP-SKY-013 — matcher lag bound versus positional resolution, for INT's DISCRETE_REFERENCE.

Stages (each reads the previous one's outputs; every stage is re-runnable)::

    audit          the copied figure-eight pair -> integrity report + index.csv. Poses and masks
                   only; writes nothing into the raw batch. Establishes segment continuity from the
                   skyline trace because these runs carry no vo/ directory.
    ingest         one spec-006 session per (run, view) through the unchanged adapter's view option,
                   into this study's own store; verifies both views at the same position and time.
    score          nested-bound scoring: one bound-32 pass per cell, C0/C1-4/8/16/32 read off it.
                   Cells: each figure-eight run against itself (INT's online-mapping case), each
                   against the other (the constant-vs-varying-height probe), and each against the
                   mountains and city memories (absent-place hard negatives).
    score-terrain  the same nested-bound pass over the frozen EXP-SKY-011 village/mountains/city
                   dual batch, so cross-terrain positional resolution is measured on identical data.
    score-hardcity the EXP-SKY-012 HardSwipe against the IDENTICAL frozen city memory, every rule
                   and gate frozen — the safety validation, never tuned on.
    report         score-versus-distance, selected-reference distance, lag behaviour, dual view,
                   height, calibration, recency; metrics.json + report.md + figures.

The matcher arms are the frozen objects: all five bounds come from one
``hsreloc.simret.lagstudy.NestedBoundBank`` pass and are cross-checked in run against
``BoundedLagNccMatcher`` (and ``FrozenNccMatcher`` for C0) — winning lag exactly, score to 1e-12,
aborting on disagreement. ``hsreloc/matchers``, ``hsreloc/retrieval`` and
``hsreloc/simret/{recog,relpose,dualview,acceptance}.py`` are dependencies, never edit targets.
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

from hsreloc.observation import read_session                                    # noqa: E402
from hsreloc.retrieval.skyline_curve import CurveError                          # noqa: E402
from hsreloc.simret import adapter, dualview, lagstudy, recog, relpose          # noqa: E402
from hsreloc.simret.conventions import SimConventions                           # noqa: E402
from hsreloc.simret.sources import (SessionSilverSource, SimExactSource,        # noqa: E402
                                    session_map)
from sky_dual_study import (_read_raw, _verify_pairs, _write_frame_list,        # noqa: E402
                            observation_id_for, session_id_for)
from sky_relpose_study import _f, _json, _open, _plt, _read_csv, _resolve, _save  # noqa: E402

STUDY_VERSION = "1.0.0"
VIEWS = ("north", "west")


# ==================================================================================================
# config / io
# ==================================================================================================

def _load_cfg(path: Path) -> tuple:
    cfg = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items()
           if not k.startswith("_")}
    return cfg, path.parent


def _write_csv(path: Path, rows: list, fields=None) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with _open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fig8_runs(cfg: dict, base: Path) -> list:
    root = _resolve(base, cfg["raw_root"]) / cfg["fig8_dir"]
    out = []
    for run_id, spec in cfg["fig8_runs"].items():
        d = root / run_id
        if not d.is_dir():
            raise SystemExit(f"figure-eight run directory missing: {d}")
        out.append((run_id, spec, d))
    return sorted(out, key=lambda t: t[0])


# ==================================================================================================
# stage: audit
# ==================================================================================================

def stage_audit(cfg: dict, base: Path) -> None:
    out = _resolve(base, cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    report = {"study_version": STUDY_VERSION, "runs": {}, "problems": []}
    rows = []
    for run_id, spec, d in _fig8_runs(cfg, base):
        settings, raw, trace = _read_raw(d)
        lvl = settings.get("level")
        if lvl != cfg["fig8_level"]:
            report["problems"].append(f"{run_id}: level {lvl!r} != declared {cfg['fig8_level']!r}")
        east = np.array([float(r["north_ue_y_cm"]) for r in raw]) / 100.0
        north = np.array([float(r["north_ue_x_cm"]) for r in raw]) / 100.0
        up = np.array([float(r["north_ue_z_cm"]) for r in raw]) / 100.0
        t = np.array([float(r["sim_time_s"]) for r in raw])
        we = np.array([float(r["west_ue_y_cm"]) for r in raw]) / 100.0
        wn = np.array([float(r["west_ue_x_cm"]) for r in raw]) / 100.0
        wu = np.array([float(r["west_ue_z_cm"]) for r in raw]) / 100.0
        pair_dxy = float(np.max(np.hypot(east - we, north - wn)))
        pair_dz = float(np.max(np.abs(up - wu)))
        step = np.hypot(np.diff(east), np.diff(north))
        cap = float(settings["skyline"]["capture_distance_m"])
        # these runs have no vo/ trace: continuity is established from the skyline trace itself
        teleports = int(np.sum(step > 3.0 * cap))
        missing = 0
        for r in raw:
            for k in ("image_path", "sim_sky_mask_path", "west_image_path", "west_sim_sky_mask_path"):
                if not (d / r[k].replace("/", "\\").replace("\\", "/")).exists():
                    missing += 1
        info = {
            "run_id": run_id, "key": spec["key"], "height_mode": spec["height_mode"],
            "path_id_declared": spec["path_id"], "path_id_settings": settings["path"]["path_id"],
            "level": lvl, "n_observations": len(raw),
            "vo_dir_present": (d / "vo").is_dir(), "vo_trace_rows": len(trace),
            "capture_distance_m": cap,
            "nw_max_position_delta_m": pair_dxy, "nw_max_height_delta_m": pair_dz,
            "north_yaw_unique": sorted(set(float(r["north_ue_yaw_deg"]) for r in raw)),
            "west_yaw_unique": sorted(set(float(r["west_ue_yaw_deg"]) for r in raw)),
            "spacing_min_m": float(step.min()), "spacing_median_m": float(np.median(step)),
            "spacing_max_m": float(step.max()), "path_length_m": float(step.sum()),
            "n_teleport_boundaries": teleports,
            "segments_declared": 1,
            "height_min_m": float(up.min()), "height_max_m": float(up.max()),
            "height_std_m": float(up.std()), "height_range_m": float(up.max() - up.min()),
            "extent_e_m": float(east.max() - east.min()), "extent_n_m": float(north.max() - north.min()),
            "n_missing_files": missing,
            "duration_s": float(t.max() - t.min()),
        }
        if teleports:
            report["problems"].append(f"{run_id}: {teleports} trajectory discontinuities > 3x capture spacing")
        if missing:
            report["problems"].append(f"{run_id}: {missing} referenced image/mask files missing")
        if pair_dxy > 1e-6 or pair_dz > 1e-6:
            report["problems"].append(f"{run_id}: North/West poses differ (max {pair_dxy:g} m)")
        report["runs"][run_id] = info
        for i, r in enumerate(raw):
            rows.append({
                "run_id": run_id, "run_key": spec["key"], "height_mode": spec["height_mode"],
                "level_key": cfg["fig8_level_key"], "order": i,
                "observation_raw_id": r["observation_id"], "sim_time_s": float(r["sim_time_s"]),
                "east_m": float(east[i]), "north_m": float(north[i]), "up_m": float(up[i]),
                "north_session_id": session_id_for(run_id, "north"),
                "west_session_id": session_id_for(run_id, "west"),
                "north_observation_id": observation_id_for(run_id, "north", r["observation_id"]),
                "west_observation_id": observation_id_for(run_id, "west", r["observation_id"]),
                "segment_id": f"{spec['key']}-seg0",
                "gt_status_north": "", "gt_status_west": "", "pair_ok": "",
            })
    # cross-run correspondence of the shared intended XY path
    by_key = defaultdict(list)
    for r in rows:
        by_key[r["run_key"]].append(r)
    keys = sorted(by_key)
    if len(keys) == 2:
        a, b = by_key[keys[0]], by_key[keys[1]]
        m = min(len(a), len(b))
        dxy = [math.hypot(a[i]["east_m"] - b[i]["east_m"], a[i]["north_m"] - b[i]["north_m"]) for i in range(m)]
        dh = [b[i]["up_m"] - a[i]["up_m"] for i in range(m)]
        report["cross_run"] = {
            "pair": keys, "n": m,
            "xy_deviation_median_m": float(np.median(dxy)), "xy_deviation_max_m": float(np.max(dxy)),
            "height_difference_min_m": float(np.min(dh)), "height_difference_max_m": float(np.max(dh)),
            "height_difference_median_m": float(np.median(dh)),
        }
    # self-revisit structure: what makes a figure eight the right probe
    for key, rs in by_key.items():
        P = np.array([[r["east_m"], r["north_m"]] for r in rs])
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        idx = np.arange(len(rs))
        far = np.abs(idx[:, None] - idx[None, :]) >= 5
        Dm = np.where(far, D, np.inf)
        report["runs"][rs[0]["run_id"]]["revisit"] = {
            "min_separation_temporally_distant_m": float(Dm.min()),
            "n_pairs_within_25m": int(np.sum(np.triu(Dm < 25.0, 1))),
        }
    _write_csv(out / "index.csv", rows)
    _json(out / "audit.json", report)
    print(f"[audit] {len(rows)} observations across {len(report['runs'])} runs -> {out/'index.csv'}")
    for p in report["problems"]:
        print(f"[audit] PROBLEM: {p}")
    if not report["problems"]:
        print("[audit] no integrity problems")


# ==================================================================================================
# stage: ingest
# ==================================================================================================

def stage_ingest(cfg: dict, base: Path) -> None:
    out = _resolve(base, cfg["out_dir"])
    store = _resolve(base, cfg["store_root"])
    store.mkdir(parents=True, exist_ok=True)
    conv = SimConventions.from_dict(cfg.get("conventions"))
    summaries = {}
    for run_id, spec, d in _fig8_runs(cfg, base):
        for view in cfg["views"]:
            sid = session_id_for(run_id, view)
            if (store / sid / "session.json").exists():
                print(f"[ingest] {sid}: present, kept")
                summaries[sid] = {"kept": True}
                continue
            t0 = time.time()
            s = adapter.ingest_run(d, store, session_id=sid, conventions=conv,
                                   options={**cfg["ingest_options"], "view": view})
            s["seconds"] = time.time() - t0
            summaries[sid] = s
            print(f"[ingest] {sid}: {s['n_observations']} observations, {s['n_seam_curves']} GT seam "
                  f"curves, {s['n_database_holes']} holes ({s['seconds']:.0f} s)")
    _json(out / "ingest_summary.json", summaries)
    _verify_pairs(cfg, base)
    _write_frame_list(cfg, base)


# ==================================================================================================
# curve sources and profiles
# ==================================================================================================

class _Curves:
    """Per-source curve + frozen profile cache with refusal bookkeeping (the EXP-SKY-009 shape)."""

    def __init__(self, source):
        self.source = source
        self.curves, self.profiles, self.refused = {}, {}, {}

    def get(self, oid: str):
        if oid in self.refused:
            return None
        if oid not in self.curves:
            try:
                c = self.source.get(oid)
            except CurveError as exc:
                self.refused[oid] = str(exc)[:160]
                return None
            self.curves[oid] = c
            self.profiles[oid] = relpose.profile_of(c)
        return self.curves[oid]


def _camera(store: Path, session_ids, n_samples: int):
    cams, cam = set(), None
    for sid in session_ids:
        meta, _ = read_session(store / sid)
        cam = relpose.ProfileCamera.from_session_meta(meta, n_samples)
        cams.add(tuple(sorted(cam.as_dict().items())))
    if len(cams) != 1:
        raise SystemExit(f"sessions disagree on the camera model: {cams}")
    return cam


def _sources(cfg: dict, base: Path, store: Path, sessions: dict, cam, which=None) -> dict:
    out = {}
    for key in (which or cfg["sources"]):
        if key == "sim_exact":
            out[key] = SimExactSource(store, sessions, cam.width_px, cam.height_px)
        elif key == "segformer":
            spec = cfg["segformer"]
            root = Path(spec["mask_root"])
            if not root.is_dir():
                print(f"[score] segformer mask root absent ({root}); source skipped")
                continue
            out[key] = SessionSilverSource(
                root, sessions, cam.width_px, cam.height_px,
                scale_short_side=spec.get("scale_short_side", 512),
                closing_px=spec.get("closing_px", 0), min_valid_frac=spec.get("min_valid_frac", 0.5),
                expected_model_revision=spec.get("expected_model_revision"))
        else:
            raise SystemExit(f"unsupported source {key!r}")
    return out


# ==================================================================================================
# stage: score (nested bounds, one pass)
# ==================================================================================================

def _score_cell(cfg: dict, bank_profiles: list, bank_ids: list, query_profiles: list,
                query_ids: list, log: str = "") -> dict:
    """One cell: every query against every reference, all bounds from one bound-32 pass."""
    bounds = [int(b) for b in cfg["bounds"]]
    bank = lagstudy.NestedBoundBank(bank_ids, bank_profiles, bounds=bounds,
                                    max_lag=int(cfg["max_lag_samples"]),
                                    min_overlap_frac=float(cfg["min_overlap_frac"]),
                                    chunk_references=int(cfg.get("chunk_references", 48)))
    nq, nr = len(query_ids), len(bank_ids)
    S = {b: np.empty((nq, nr), dtype=np.float64) for b in bounds}
    L = {b: np.empty((nq, nr), dtype=np.float64) for b in bounds}
    every = max(1, int(cfg.get("frozen_check_every", 10)))
    checks = {"n_queries_checked": 0, "max_abs_score_delta": 0.0, "n_lag_mismatch": 0}
    t0 = time.time()
    for qi, qp in enumerate(query_profiles):
        got = bank.score(qp)
        for b in bounds:
            S[b][qi], L[b][qi] = got[b]
        if qi % every == 0:
            probe = sorted({0, nr // 3, nr // 2, max(0, nr - 1)})
            v = lagstudy.verify_bounds(bank, qp, bank_profiles, indices=probe)
            checks["n_queries_checked"] += 1
            checks["max_abs_score_delta"] = max(checks["max_abs_score_delta"], v["max_abs_score_delta"])
            checks["n_lag_mismatch"] += v["n_lag_mismatch"]
        if log and (qi + 1) % 50 == 0:
            print(f"    {log}: {qi+1}/{nq} queries ({time.time()-t0:.0f} s)", flush=True)
    checks["seconds"] = time.time() - t0
    return {"scores": S, "lags": L, "checks": checks}


def _save_cell(path: Path, cell: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"scores_{b}": cell["scores"][b] for b in cell["scores"]}
    payload.update({f"lags_{b}": cell["lags"][b] for b in cell["lags"]})
    for k, v in meta.items():
        payload[k] = np.asarray(v)
    np.savez_compressed(path, **payload)


def _load_store_rows(store: Path, session_id: str) -> list:
    meta, obs = read_session(store / session_id)
    return [{"observation_id": o.observation_id, "east_m": o.pos_east_m, "north_m": o.pos_north_m,
             "up_m": o.up_m, "timestamp_s": o.timestamp_s} for o in obs]


def _paired_rows(store: Path, run_id: str, curves: dict) -> list:
    """Physical observations whose BOTH views have a curve in this source — so single-view
    baselines and the dual rules always share query and reference sets."""
    north = {r["observation_id"]: r for r in _load_store_rows(store, session_id_for(run_id, "north"))}
    west = {r["observation_id"]: r for r in _load_store_rows(store, session_id_for(run_id, "west"))}
    rows = []
    for i, (oid_n, r) in enumerate(sorted(north.items())):
        raw = oid_n.split("__")[-1]
        oid_w = observation_id_for(run_id, "west", raw)
        if oid_w not in west:
            continue
        if curves["north"].get(oid_n) is None or curves["west"].get(oid_w) is None:
            continue
        rows.append({**r, "raw_id": raw, "order": i, "run_id": run_id,
                     "north_observation_id": oid_n, "west_observation_id": oid_w})
    return rows


def stage_score(cfg: dict, base: Path, sources_wanted=None) -> None:
    out = _resolve(base, cfg["out_dir"])
    store = _resolve(base, cfg["store_root"])
    cells_dir = out / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    run_ids = [r for r, _, _ in _fig8_runs(cfg, base)]
    session_ids = [session_id_for(rid, v) for rid in run_ids for v in cfg["views"]]
    sessions = session_map(store, session_ids)          # observation_id -> session_id
    cam = _camera(store, session_ids, int(cfg["profile_n_samples"]))
    srcs = _sources(cfg, base, store, sessions, cam, which=sources_wanted)
    manifest_path = out / "score_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("cells", {})
    manifest["camera"] = cam.as_dict()
    manifest["bounds"] = list(cfg["bounds"])
    manifest["degrees_per_sample"] = lagstudy.degrees_per_sample(
        float(cfg["fov_deg"]), int(cfg["profile_n_samples"]))

    for src_key, src in srcs.items():
        curves = {v: _Curves(src) for v in cfg["views"]}
        rows_by_run = {rid: _paired_rows(store, rid, curves) for rid in run_ids}
        for rid, rows in rows_by_run.items():
            print(f"[score] {src_key} {rid}: {len(rows)} paired observations")
        # self cells (INT's online-mapping case) and cross-run cells (the height probe)
        for q_run in run_ids:
            for r_run in run_ids:
                for view in cfg["views"]:
                    name = f"fig8_{q_run[-6:]}_vs_{r_run[-6:]}_{src_key}_{view}"
                    path = cells_dir / f"{name}.npz"
                    if path.exists():
                        print(f"[score] {name}: present, kept")
                        continue
                    qrows, rrows = rows_by_run[q_run], rows_by_run[r_run]
                    if not qrows or not rrows:
                        continue
                    key = f"{view}_observation_id"
                    qp = [curves[view].profiles[r[key]] for r in qrows]
                    rp = [curves[view].profiles[r[key]] for r in rrows]
                    cell = _score_cell(cfg, rp, [r[key] for r in rrows], qp, [r[key] for r in qrows],
                                       log=name)
                    _save_cell(path, cell, {
                        "query_keys": [r[key] for r in qrows],
                        "reference_keys": [r[key] for r in rrows],
                        "query_xy": [[r["east_m"], r["north_m"]] for r in qrows],
                        "reference_xy": [[r["east_m"], r["north_m"]] for r in rrows],
                        "query_up": [r["up_m"] for r in qrows],
                        "reference_up": [r["up_m"] for r in rrows],
                        "query_t": [r["timestamp_s"] for r in qrows],
                        "reference_t": [r["timestamp_s"] for r in rrows],
                        "query_order": [r["order"] for r in qrows],
                        "reference_order": [r["order"] for r in rrows],
                    })
                    manifest["cells"][name] = {
                        "kind": "self" if q_run == r_run else "cross_run",
                        "query_run": q_run, "reference_run": r_run, "source": src_key, "view": view,
                        "n_queries": len(qrows), "n_references": len(rrows), **cell["checks"]}
                    print(f"[score] {name}: {cell['checks']['n_queries_checked']} cross-checks, "
                          f"max |dscore| {cell['checks']['max_abs_score_delta']:.2e}, "
                          f"{cell['checks']['n_lag_mismatch']} lag mismatches "
                          f"({cell['checks']['seconds']:.0f} s)")
        _json(manifest_path, manifest)
    _json(manifest_path, manifest)
    print(f"[score] manifest -> {manifest_path}")


# ==================================================================================================
# stage: score-terrain / score-hardcity — the same nested pass over frozen populations
# ==================================================================================================

def _obs_id(frozen_key: str, view: str) -> str:
    """Frozen cell key ``<run_id>/<raw_id>`` -> store observation id ``<run_id>-<view>__<raw_id>``."""
    run_id, raw = str(frozen_key).split("/")
    return observation_id_for(run_id, view, raw)


class _Routed:
    """A curve source that reads queries from the new store and references from the frozen one.

    The EXP-SKY-012 device, reused verbatim in intent: the frozen references must come from the
    frozen store and frozen mask root so the rebuilt memory is provably the same memory.
    """

    def __init__(self, new_src, old_src, new_ids: set):
        self.new, self.old, self.new_ids = new_src, old_src, set(new_ids)

    def get(self, observation_id: str):
        return (self.new if observation_id in self.new_ids else self.old).get(observation_id)

    def describe(self) -> dict:
        return {"kind": "routed", "new": self.new.describe(), "frozen": self.old.describe()}


def _frozen_cfg(cfg: dict, base: Path, key: str) -> tuple:
    p = _resolve(base, cfg[key]) if str(cfg[key]).endswith(".json") else base / cfg[key]
    d = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}, p.parent


def stage_score_terrain(cfg: dict, base: Path, sources_wanted=None) -> None:
    """Re-score the frozen EXP-SKY-011 village/mountains/city cells at every bound.

    The query and reference lists, their order and their XY are taken **from the frozen cell**, so
    the populations are identical to EXP-SKY-011 by construction. The C1-32 column is then required
    to reproduce the frozen scores to 1e-12 with identical lags — a whole-cell validation of this
    study's instrument against a committed artifact.
    """
    dual_cfg, dual_base = _frozen_cfg(cfg, base, "frozen_dual_config")
    out = _resolve(base, cfg["out_dir"])
    store = _resolve(dual_base, dual_cfg["store_root"])
    frozen_cells = _resolve(base, cfg["frozen_dual_out"]) / "cells"
    cells_dir = out / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "score_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("cells", {})
    session_ids = sorted(p.name for p in store.iterdir() if (p / "observations.csv").exists())
    sessions = session_map(store, session_ids)
    cam = _camera(store, session_ids, int(cfg["profile_n_samples"]))
    seg = dict(dual_cfg["segformer"])
    srcs = _sources({**cfg, "segformer": seg}, base, store, sessions, cam, which=sources_wanted)
    up_by_obs = {}
    for sid in session_ids:
        for r in _load_store_rows(store, sid):
            up_by_obs[r["observation_id"]] = r["up_m"]

    for level in ("village", "mountains", "city"):
        for src_key, src in srcs.items():
            curves = _Curves(src)
            for view in cfg["views"]:
                name = f"terrain_{level}_{src_key}_{view}"
                path = cells_dir / f"{name}.npz"
                if path.exists():
                    print(f"[score-terrain] {name}: present, kept")
                    continue
                fz = frozen_cells / f"scores_{level}_{src_key}_{view}.npz"
                if not fz.exists():
                    print(f"[score-terrain] {name}: frozen cell absent, skipped")
                    continue
                with np.load(fz, allow_pickle=False) as z:
                    qkeys = [str(x) for x in z["query_keys"]]
                    rkeys = [str(x) for x in z["reference_keys"]]
                    qxy = np.asarray(z["query_xy"], dtype=np.float64)
                    rxy = np.asarray(z["reference_xy"], dtype=np.float64)
                    qseg = [str(x) for x in z["query_segments"]]
                    rseg = [str(x) for x in z["reference_segments"]]
                    r_anchor = np.asarray(z["reference_is_anchor"], dtype=bool)
                    frozen_scores = np.asarray(z["scores"], dtype=np.float64)
                    frozen_lags = np.asarray(z["lags"], dtype=np.float64)
                qo = [_obs_id(k, view) for k in qkeys]
                ro = [_obs_id(k, view) for k in rkeys]
                miss = [o for o in set(qo) | set(ro) if curves.get(o) is None]
                if miss:
                    raise SystemExit(f"{name}: {len(miss)} frozen observations have no curve "
                                     f"(first: {miss[0]}) — the frozen population must be reproducible")
                cell = _score_cell(cfg, [curves.profiles[o] for o in ro], ro,
                                   [curves.profiles[o] for o in qo], qo, log=name)
                mx = float(np.max(np.abs(cell["scores"][32] - frozen_scores)))
                nl = int(np.sum(cell["lags"][32] != frozen_lags))
                if mx > 1e-12 or nl:
                    raise SystemExit(f"{name}: C1-32 does not reproduce the frozen EXP-SKY-011 cell "
                                     f"(max |dscore| {mx:.3e}, {nl} lag mismatches) — aborting")
                print(f"[score-terrain] {name}: reproduces frozen cell (max |dscore| {mx:.2e}, "
                      f"{nl} lag mismatches) over {frozen_scores.size} pairs")
                _save_cell(path, cell, {
                    "query_keys": qo, "reference_keys": ro,
                    "query_xy": qxy, "reference_xy": rxy,
                    "query_up": [up_by_obs[o] for o in qo],
                    "reference_up": [up_by_obs[o] for o in ro],
                    "query_segments": qseg, "reference_segments": rseg,
                    "reference_is_anchor": r_anchor,
                })
                manifest["cells"][name] = {
                    "kind": "terrain", "level": level, "source": src_key, "view": view,
                    "n_queries": len(qo), "n_references": len(ro),
                    "frozen_reproduction_max_abs_dscore": mx, "frozen_lag_mismatches": nl,
                    **cell["checks"]}
                _json(manifest_path, manifest)
    _json(manifest_path, manifest)
    print(f"[score-terrain] manifest -> {manifest_path}")


def stage_score_hardcity(cfg: dict, base: Path, sources_wanted=None) -> None:
    """The EXP-SKY-012 HardSwipe against the IDENTICAL frozen city memory, at every bound.

    Safety validation only. Nothing here is tuned; the frozen memory is rebuilt from the frozen
    store and frozen mask root through a routed source, and the C1-32 column must reproduce the
    frozen EXP-SKY-012 cell exactly before any other bound is reported.
    """
    hard_cfg, hard_base = _frozen_cfg(cfg, base, "frozen_hardcity_config")
    dual_cfg, dual_base = _frozen_cfg(cfg, base, "frozen_dual_config")
    out = _resolve(base, cfg["out_dir"])
    new_store = _resolve(hard_base, hard_cfg["store_root"])
    old_store = _resolve(dual_base, dual_cfg["store_root"])
    frozen_cells = _resolve(base, cfg["frozen_hardcity_out"]) / "cells"
    cells_dir = out / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "score_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("cells", {})
    new_sessions = sorted(p.name for p in new_store.iterdir() if (p / "observations.csv").exists())
    old_sessions = sorted(p.name for p in old_store.iterdir() if (p / "observations.csv").exists())
    sess_new, sess_old = session_map(new_store, new_sessions), session_map(old_store, old_sessions)
    cam = _camera(new_store, new_sessions, int(cfg["profile_n_samples"]))
    up_by_obs = {}
    for st, sids in ((new_store, new_sessions), (old_store, old_sessions)):
        for sid in sids:
            for r in _load_store_rows(st, sid):
                up_by_obs[r["observation_id"]] = r["up_m"]
    for src_key in (sources_wanted or cfg["sources"]):
        new_src = _sources({**cfg, "segformer": {**cfg["segformer"],
                                                 "mask_root": hard_cfg["segformer_mask_root"]}},
                           base, new_store, sess_new, cam, which=[src_key]).get(src_key)
        old_src = _sources({**cfg, "segformer": dual_cfg["segformer"]},
                           base, old_store, sess_old, cam, which=[src_key]).get(src_key)
        if new_src is None or old_src is None:
            print(f"[score-hardcity] {src_key}: a curve source is unavailable, skipped")
            continue
        curves = _Curves(_Routed(new_src, old_src, set(sess_new)))
        for view in cfg["views"]:
            name = f"hardcity_{src_key}_{view}"
            path = cells_dir / f"{name}.npz"
            if path.exists():
                print(f"[score-hardcity] {name}: present, kept")
                continue
            fz = frozen_cells / f"scores_city_{src_key}_{view}.npz"
            if not fz.exists():
                print(f"[score-hardcity] {name}: frozen cell absent, skipped")
                continue
            with np.load(fz, allow_pickle=False) as z:
                qkeys = [str(x) for x in z["query_keys"]]
                rkeys = [str(x) for x in z["reference_keys"]]
                qxy = np.asarray(z["query_xy"], dtype=np.float64)
                rxy = np.asarray(z["reference_xy"], dtype=np.float64)
                qseg = [str(x) for x in z["query_segments"]]
                rseg = [str(x) for x in z["reference_segments"]]
                r_anchor = np.asarray(z["reference_is_anchor"], dtype=bool)
                frozen_scores = np.asarray(z["scores"], dtype=np.float64)
                frozen_lags = np.asarray(z["lags"], dtype=np.float64)
            qo = [_obs_id(k, view) for k in qkeys]
            ro = [_obs_id(k, view) for k in rkeys]
            miss = [o for o in set(qo) | set(ro) if curves.get(o) is None]
            if miss:
                raise SystemExit(f"{name}: {len(miss)} frozen observations have no curve "
                                 f"(first: {miss[0]})")
            cell = _score_cell(cfg, [curves.profiles[o] for o in ro], ro,
                               [curves.profiles[o] for o in qo], qo, log=name)
            mx = float(np.max(np.abs(cell["scores"][32] - frozen_scores)))
            nl = int(np.sum(cell["lags"][32] != frozen_lags))
            if mx > 1e-12 or nl:
                raise SystemExit(f"{name}: C1-32 does not reproduce the frozen EXP-SKY-012 cell "
                                 f"(max |dscore| {mx:.3e}, {nl} lag mismatches) — aborting")
            print(f"[score-hardcity] {name}: reproduces frozen cell (max |dscore| {mx:.2e}, "
                  f"{nl} lag mismatches) over {frozen_scores.size} pairs")
            _save_cell(path, cell, {
                "query_keys": qo, "reference_keys": ro, "query_xy": qxy, "reference_xy": rxy,
                "query_up": [up_by_obs[o] for o in qo],
                "reference_up": [up_by_obs[o] for o in ro],
                "query_segments": qseg, "reference_segments": rseg,
                "reference_is_anchor": r_anchor,
            })
            manifest["cells"][name] = {
                "kind": "hardcity", "level": "city", "source": src_key, "view": view,
                "n_queries": len(qo), "n_references": len(ro),
                "frozen_reproduction_max_abs_dscore": mx, "frozen_lag_mismatches": nl,
                **cell["checks"]}
            _json(manifest_path, manifest)
    _json(manifest_path, manifest)
    print(f"[score-hardcity] manifest -> {manifest_path}")


# ==================================================================================================
# entry point
# ==================================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--stage", required=True,
                    choices=["audit", "ingest", "score", "score-terrain", "score-hardcity", "report"])
    ap.add_argument("--sources", default=None,
                    help="comma-separated subset of the configured curve sources")
    args = ap.parse_args(argv)
    cfg, base = _load_cfg(args.config)
    which = args.sources.split(",") if args.sources else None
    if args.stage == "audit":
        stage_audit(cfg, base)
    elif args.stage == "ingest":
        stage_ingest(cfg, base)
    elif args.stage == "score":
        stage_score(cfg, base, which)
    elif args.stage == "score-terrain":
        stage_score_terrain(cfg, base, which)
    elif args.stage == "score-hardcity":
        stage_score_hardcity(cfg, base, which)
    elif args.stage == "report":
        import sky_matcher_report as rep
        out = _resolve(base, cfg["out_dir"])
        m = rep.stage_report(cfg, base, out, _json)
        rep.write_report(cfg, out, m, _json)
        print(f"[report] metrics -> {out/'metrics.json'}   report -> {out/'report.md'}")
    else:
        raise SystemExit(f"stage {args.stage!r} is not implemented yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
