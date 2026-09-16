"""Orchestration for the FINAL simulator validation feature (spec 20260903-225416).

The pilot modules do the work; this module wires them across three levels and two populations and
holds the **discipline boundary**: ``asian_village_hills_background`` is DEV, ``mountains`` and
``large_city_flat`` are held-out, and every command that reads curve or retrieval performance goes
through :func:`guard_population` — ``dev`` refuses any non-DEV level, ``final`` refuses unless the
committed ``configs/frozen/sim-final.prereg`` matches the built artifacts verbatim
(``prereg.check_final_prereg``). Building poses/metadata (index, task sets, query/reference sets)
is dataset construction and deliberately unguarded, per the owner brief.

Task naming: ``final-<kind>-<task_key>`` (e.g. ``final-exactpose-mountains``), giving set ids
``sim-final-exactpose-mountains-s250-<source>`` and run ids
``sim-final-exactpose-mountains-s250-ncc-<source>``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from hsreloc.observation import read_session
from hsreloc.placeret.sources import REFUSAL_SENTINEL
from hsreloc.retrieval.run import execute
from hsreloc.retrieval.runconfig import ConfigError, parse_config
from hsreloc.simret import geometry, groups, prereg, sets
from hsreloc.simret.runner import AXIS_SLICES, _cell_check
from hsreloc.simret.sources import PROVENANCE_BY_KEY, build_source, session_map

POPULATIONS = ("dev", "final")
TASK_KINDS = ("exactpose", "swipes")


class ExtError(Exception):
    """The extended feature cannot proceed as asked."""


def _resolve(base: Path, value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


# --------------------------------------------------------------------------------------------------
# levels and populations
# --------------------------------------------------------------------------------------------------

def levels_for(cfg: dict, population: str) -> dict:
    """``level_folder -> level block`` for one population; refuses an unknown population."""
    if population not in POPULATIONS:
        raise ExtError(f"population must be one of {POPULATIONS}, got {population!r}")
    out = {lvl: b for lvl, b in cfg["levels"].items() if b["split"] == population}
    if not out:
        raise ExtError(f"no level has split {population!r}")
    return out


def assert_dev_only(cfg: dict, session_ids) -> None:
    """The DEV guard: every session must belong to a DEV level."""
    dev_sessions = {s for b in levels_for(cfg, "dev").values() for s in b["sessions"]}
    bad = sorted(set(session_ids) - dev_sessions)
    if bad:
        raise ExtError(f"population 'dev' refuses non-DEV session(s) {bad[:5]} — the held-out "
                       f"levels are behind the freeze (contracts/freeze.md)")


def guard_population(cfg: dict, base: Path, population: str) -> None:
    """dev: free (levels filtered elsewhere; sessions asserted by callers). final: the freeze."""
    if population not in POPULATIONS:
        raise ExtError(f"population must be one of {POPULATIONS}, got {population!r}")
    if population == "final":
        path = _resolve(base, cfg["preregistration_ref"])
        if not path.exists():
            # fail with the canonical message before any assembly work
            raise prereg.PreregError(
                f"no freeze pre-registration at {path} — run `ext-prereg` and COMMIT it before "
                f"any `--population final` command (contracts/freeze.md)")
        built = assemble_final_prereg(cfg, base)
        prereg.check_final_prereg(path, built)


# --------------------------------------------------------------------------------------------------
# tasks and sets (poses only — construction, both splits allowed)
# --------------------------------------------------------------------------------------------------

def task_name(kind: str, task_key: str) -> str:
    return f"final-{kind}-{task_key}"


def _reference_ids(cfg: dict, base: Path, task_key: str) -> list:
    assigns = groups.read_assignments(_resolve(base, cfg["assignments_csv"][task_key]))
    refs = [oid for oid, a in assigns.items() if a["role"] == "reference"]
    if not refs:
        raise ExtError(f"{task_key}: no reference rows in the assignments sidecar")
    return sorted(refs)


def _level_block(cfg: dict, task_key: str) -> tuple:
    for lvl, b in cfg["levels"].items():
        if b["task_key"] == task_key:
            return lvl, b
    raise ExtError(f"no level has task_key {task_key!r}")


def task_query_sessions(cfg: dict, task_key: str, kind: str) -> list:
    _, b = _level_block(cfg, task_key)
    if kind == "exactpose":
        ref_session = b["nopath_sessions"]["DAY+CLEAR"]
        return sorted(s for c, s in b["nopath_sessions"].items() if s != ref_session)
    if kind == "swipes":
        return sorted(b["swipe_sessions"])
    raise ExtError(f"unknown task kind {kind!r}")


def build_all_tasks(cfg: dict, base: Path) -> dict:
    """The six task sets (3 levels x exactpose/swipes), spacing 250, explicit anchor references."""
    store = _resolve(base, cfg["store_root"])
    out_root = _resolve(base, cfg["tasks"]["tasks_out"])
    manifests = {}
    for lvl, b in sorted(cfg["levels"].items()):
        key = b["task_key"]
        refs = _reference_ids(cfg, base, key)
        ref_sessions = {b["nopath_sessions"]["DAY+CLEAR"]}
        for kind in TASK_KINDS:
            name = task_name(kind, key)
            queries = task_query_sessions(cfg, key, kind)
            sessions = {sid: read_session(store / sid)[1]
                        for sid in sorted(set(queries) | ref_sessions)}
            built = geometry.build_tasks(sessions, {
                "reference_observation_ids": refs,
                "query_sessions": queries,
                "spacings_m": cfg["tasks"]["spacings_m"],
                "stage_rule": cfg["tasks"].get("stage_rule", {}),
            })
            manifests[name] = geometry.write_tasks(built, out_root / name)
            print(f"[simret] ext-tasks {name}: grids {manifests[name]['grid_counts']}, "
                  f"{manifests[name]['n_query_rows']} query rows")
    return manifests


def load_all_tasks(cfg: dict, base: Path) -> dict:
    out_root = _resolve(base, cfg["tasks"]["tasks_out"])
    return {task_name(kind, b["task_key"]): geometry.load_tasks(out_root / task_name(kind, b["task_key"]))
            for b in cfg["levels"].values() for kind in TASK_KINDS}


def _camera_dims(store: Path, session_ids: list) -> tuple:
    dims = set()
    for sid in session_ids:
        meta = json.loads((store / sid / "session.json").read_text(encoding="utf-8"))
        dims.add(tuple(meta["camera"]["resolution_px"]))
    if len(dims) != 1:
        raise ExtError(f"sessions disagree on resolution: {sorted(dims)}")
    (w, h), = dims
    return int(w), int(h)


def context(cfg: dict, base: Path) -> dict:
    """Store, session map and camera dims over the config's 44 sessions."""
    store = _resolve(base, cfg["store_root"])
    session_ids = list(cfg["sessions"])
    width, height = _camera_dims(store, session_ids)
    return {"store": store, "session_ids": session_ids, "width": width, "height": height,
            "smap": session_map(store, session_ids)}


def build_sources(cfg: dict, base: Path, ctx: dict, keys=None, refusal_mode=None) -> dict:
    kw = {} if refusal_mode is None else {"refusal_mode": refusal_mode}
    return {key: build_source(key, spec, ctx["smap"], ctx["width"], ctx["height"],
                              store_root=ctx["store"], config_dir=base, **kw)
            for key, spec in cfg["sources"].items() if keys is None or key in keys}


def _anchor_map_for_sets(cfg: dict, base: Path) -> dict:
    """observation_id -> anchor_id over every anchor capture (from the extended index)."""
    from hsreloc.simret import extindex
    idx = extindex.load_index(_resolve(base, cfg["index_out"]))
    return {r["observation_id"]: r["anchor_id"] for r in idx["rows"] if r["anchor_id"]}


def build_all_sets(cfg: dict, base: Path) -> dict:
    """Query/reference sets for every cell of the run matrix (poses + curves materialisation)."""
    ctx = context(cfg, base)
    tasks = load_all_tasks(cfg, base)
    anchors = _anchor_map_for_sets(cfg, base)
    written = {}
    for kind, spec in cfg["run_matrix"].items():
        for key in spec["levels"]:
            lvl, b = _level_block(cfg, key)
            name = task_name(kind, key)
            t = tasks[name]
            ref_session = b["nopath_sessions"]["DAY+CLEAR"]
            session_meta = json.loads(
                (ctx["store"] / ref_session / "session.json").read_text(encoding="utf-8"))
            obs_by_id = {}
            for sid in set(task_query_sessions(cfg, key, kind)) | {ref_session}:
                for o in read_session(ctx["store"] / sid)[1]:
                    obs_by_id[o.observation_id] = o
            source_keys = list(spec["sources"])
            if key in (spec.get("dp_levels") or []):
                source_keys = sorted(set(source_keys) | {"dp"})
            for spacing in cfg["tasks"]["spacings_m"]:
                for skey in source_keys:
                    src = build_source(skey, cfg["sources"][skey], ctx["smap"], ctx["width"],
                                       ctx["height"], store_root=ctx["store"], config_dir=base)
                    rman = sets.write_reference_set(t, name, spacing, skey, src,
                                                    _resolve(base, cfg["refdb_root"]))
                    sets.write_query_set(t, name, spacing, skey,
                                         _resolve(base, cfg["datasets_root"]), obs_by_id,
                                         ctx["width"], ctx["height"], session_meta=session_meta,
                                         anchors=anchors)
                    written[sets.query_set_id(name, spacing, skey)] = {
                        "n_references": rman["reference_source"]["n_references"],
                        "n_holes": rman["reference_source"]["n_holes"]}
                if len(source_keys) > 1:
                    sets.assert_source_materialisations_identical(
                        _resolve(base, cfg["datasets_root"]), name, spacing, source_keys)
    for sid, info in sorted(written.items()):
        print(f"  {sid}: {info['n_references']} references, {info['n_holes']} holes")
    return written


# --------------------------------------------------------------------------------------------------
# the freeze prereg, assembled from BUILT artifacts
# --------------------------------------------------------------------------------------------------

def assemble_final_prereg(cfg: dict, base: Path) -> dict:
    from hsreloc.simret import extindex
    raw_digest = (_resolve(base, cfg["inventory_dir"]) / "raw_digest.txt").read_text(
        encoding="utf-8").strip()
    index_manifest = extindex.load_index(_resolve(base, cfg["index_out"]))["manifest"]
    sidecars = {}
    for group in ("anchors_csv", "assignments_csv"):
        for key, rel in cfg[group].items():
            p = _resolve(base, rel)
            sidecars[f"{group}:{key}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    task_manifests = {name: t["manifest"] for name, t in load_all_tasks(cfg, base).items()}
    ctx = context(cfg, base)
    descriptions = {k: {kk: vv for kk, vv in s.describe().items() if kk != "refusal_mode"}
                    for k, s in build_sources(cfg, base, ctx).items()}
    return prereg.build_final_prereg(cfg, raw_digest, index_manifest, sidecars,
                                     task_manifests, descriptions)


# --------------------------------------------------------------------------------------------------
# the run matrix and its evaluation (population-guarded)
# --------------------------------------------------------------------------------------------------

def cells_for(cfg: dict, population: str) -> list:
    """(kind, task_key, source_key) cells of the run matrix restricted to one population."""
    wanted = {b["task_key"] for b in levels_for(cfg, population).values()}
    cells = []
    for kind, spec in cfg["run_matrix"].items():
        for key in spec["levels"]:
            if key not in wanted:
                continue
            source_keys = list(spec["sources"])
            if key in (spec.get("dp_levels") or []):
                source_keys = sorted(set(source_keys) | {"dp"})
            for skey in source_keys:
                cells.append((kind, key, skey))
    if not cells:
        raise ExtError(f"the run matrix has no {population!r} cells")
    return cells


def run_id_for(cfg: dict, name: str, spacing: float, baseline: str, source_key: str) -> str:
    return cfg["run_id_pattern"].format(task=name, spacing=f"{spacing:g}", baseline=baseline,
                                        source=source_key)


def matcher_run_config(cfg: dict, base: Path, name: str, spacing: float, baseline: str,
                       source_key: str) -> dict:
    datasets_root = _resolve(base, cfg["datasets_root"])
    refdb_root = _resolve(base, cfg["refdb_root"])
    qs_dir = datasets_root / sets.query_set_id(name, spacing, source_key)
    rs_dir = refdb_root / sets.reference_set_id(name, spacing, source_key)
    descriptor = json.loads((qs_dir / "dataset.json").read_text(encoding="utf-8"))
    origin = {k: cfg["geodetic_origin"][k] for k in ("lat_deg", "lon_deg", "alt_m")}
    # query_curves.root doubles as the frozen loader's reference cross-check root — right for the
    # oracle source, wrong for the automatic ones (EXP-SKY-007 Deviation 1); route per source.
    if source_key == "sim_exact":
        curves_root = _resolve(base, cfg["store_root"])
    elif source_key == "segformer":
        curves_root = _resolve(base, cfg["sources"]["segformer"]["mask_root"])
    else:
        curves_root = rs_dir
    return {
        "schema_version": "1.1.0",
        "run_id": run_id_for(cfg, name, spacing, baseline, source_key),
        "inputs": {
            "reference_set": str(rs_dir),
            "query_set": str(qs_dir),
            "query_curves": {"kind": "observation_store", "root": str(curves_root),
                             "expect_provenance": PROVENANCE_BY_KEY[source_key]},
        },
        "geodetic_origin": origin,
        "profile": cfg["matcher"]["profile"],
        "match": {"baseline": baseline, "recall_k": cfg["matcher"]["recall_k"],
                  "lag_search": cfg["matcher"].get("lag_search", False)},
        "acceptance": cfg["matcher"]["acceptance"],
        "output": {"root": str(_resolve(base, cfg["runs_root"]))},
        "evidence": {"tier": descriptor["evidence_tier"], "caveat": descriptor["evidence_caveat"]},
        "environment": {"is_target_hardware": False},
    }


def run_cells(cfg: dict, base: Path, population: str) -> dict:
    guard_population(cfg, base, population)
    ctx = context(cfg, base)
    srcs = build_sources(cfg, base, ctx, refusal_mode=REFUSAL_SENTINEL)
    tasks = load_all_tasks(cfg, base)
    by_task: dict = {}
    for kind, key, skey in cells_for(cfg, population):
        by_task.setdefault((kind, key), []).append(skey)
    runs, cell_reports = [], []
    for (kind, key), source_keys in sorted(by_task.items()):
        name = task_name(kind, key)
        for spacing in [float(s) for s in cfg["tasks"]["spacings_m"]]:
            for baseline in cfg["baselines"]:
                cell = {}
                for skey in source_keys:
                    raw = matcher_run_config(cfg, base, name, spacing, baseline, skey)
                    record_dir = _resolve(base, cfg["runs_root"]) / raw["run_id"]
                    existing = record_dir / "manifest.json"
                    if existing.exists():
                        m = json.loads(existing.read_text(encoding="utf-8"))
                        if m.get("completed") and m.get("processed_count") == m.get("query_count"):
                            summary = {"record_dir": str(record_dir)}
                            print(f"[simret] {raw['run_id']}: complete record present, kept")
                        else:
                            raise ExtError(f"{raw['run_id']}: an INCOMPLETE record exists at "
                                           f"{record_dir}; delete it explicitly before resuming")
                    else:
                        try:
                            rc = parse_config(raw, base_dir=base)
                            summary = execute(rc, curve_source=srcs[skey])
                            print(f"[simret] {raw['run_id']}: done")
                        except ConfigError as exc:
                            raise ExtError(f"{raw['run_id']}: {exc}") from exc
                    manifest = json.loads((Path(summary["record_dir"]) / "manifest.json")
                                          .read_text(encoding="utf-8"))
                    cell[skey] = {"run_id": raw["run_id"],
                                  "record_dir": str(summary["record_dir"]),
                                  "query_set_revision": manifest.get("query_set_revision"),
                                  "query_set_id": manifest.get("query_set_id"),
                                  "query_curves_digest": manifest.get("relocalizer_config", {})
                                  .get("query_curves_digest"),
                                  "provenance": PROVENANCE_BY_KEY[skey],
                                  "grid_digest": tasks[name]["grids"][f"{spacing:g}"]["digest"]}
                    runs.append(cell[skey])
                if len(cell) > 1:
                    cell_reports.append(_cell_check(spacing, f"{name}/{baseline}", cell))
    return {"population": population, "runs": runs, "cells": cell_reports}


def evaluate_cells(cfg: dict, base: Path, population: str) -> list:
    guard_population(cfg, base, population)
    repo = Path(__file__).resolve().parents[3]
    if str(repo / "evaluation") not in sys.path:
        sys.path.append(str(repo / "evaluation"))
    from naveval.evaluate_skyline import _load_config, run_skyline_evaluation

    tasks = load_all_tasks(cfg, base)
    datasets_root = _resolve(base, cfg["datasets_root"])
    runs_root = _resolve(base, cfg["runs_root"])
    eval_root = _resolve(base, cfg["evaluations_root"])
    eval_cfg_dir = _resolve(base, cfg["evaluation_configs_root"])
    eval_cfg_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for kind, key, skey in cells_for(cfg, population):
        name = task_name(kind, key)
        for spacing in [float(s) for s in cfg["tasks"]["spacings_m"]]:
            grid = tasks[name]["grids"][f"{spacing:g}"]
            tau = tasks[name]["manifest"]["tolerances"][f"{spacing:g}"]
            for baseline in cfg["baselines"]:
                run_id = run_id_for(cfg, name, spacing, baseline, skey)
                cfg_path = eval_cfg_dir / f"eval-{run_id}.json"
                out_dir = eval_root / f"sim-retrieval-{run_id}"
                ev = {
                    "schema_version": "1.0.0",
                    "evaluation_id": f"sim-retrieval-{run_id}",
                    "run_record": str(runs_root / run_id),
                    "query_set": str(datasets_root / sets.query_set_id(name, spacing, skey)),
                    "operational_tolerance_m": tau["tau_pos_m"],
                    "near_tolerance_m": tau["tau_near_m"],
                    "recall_k": int(cfg["matcher"]["recall_k"]),
                    "confidence_reject_threshold": float(
                        cfg["matcher"]["acceptance"]["reject_threshold"]),
                    "operating_points": [0.3, 0.5, 0.7],
                    "generic_axis_slices": AXIS_SLICES,
                    "tier_pooling": "forbidden",
                }
                cfg_path.write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
                metrics = run_skyline_evaluation(_load_config(cfg_path), out_dir)
                rate = metrics["retrieval"]["topological_success"]["success_rate"]
                done.append({"run_id": run_id, "metrics": str(out_dir / "metrics.json"),
                             "success_rate": rate, "n_references": grid["n_references"]})
                print(f"[simret] evaluated {run_id}: topological success {rate}")
    return done
