"""Config-driven entry points for the simulator skyline work (Working rule 10).

Every command takes ``--config <path>``; paths inside a config resolve relative to the config file,
as everywhere else in this repository.

    python -m hsreloc.simret.cli validate        --config configs/sim-pilot.json
    python -m hsreloc.simret.cli ingest          --config configs/sim-pilot.json
    python -m hsreloc.simret.cli extract-dp      --config configs/sim-pilot.json
    python -m hsreloc.simret.cli silver-list     --config configs/sim-pilot.json
    python -m hsreloc.simret.cli groups          --config configs/sim-pilot.json
    python -m hsreloc.simret.cli tasks           --config configs/sim-pilot.json
    python -m hsreloc.simret.cli sets            --config configs/sim-pilot.json
    python -m hsreloc.simret.cli exp-extraction  --config configs/sim-pilot.json
    python -m hsreloc.simret.cli exp-consistency --config configs/sim-pilot.json
    python -m hsreloc.simret.cli exp-deformation --config configs/sim-pilot.json
    python -m hsreloc.simret.cli prereg          --config configs/sim-pilot.json
    python -m hsreloc.simret.cli run             --config configs/sim-pilot.json
    python -m hsreloc.simret.cli evaluate        --config configs/sim-pilot.json
    python -m hsreloc.simret.cli exp-variant     --config configs/sim-pilot.json
    python -m hsreloc.simret.cli report          --config configs/sim-pilot.json

``validate`` writes nothing and is the command to run first on a fresh run directory (or, with
``raw_root`` configured, on a whole directory of them): it prints every check, every measurement
and every refusal reason. ``ingest`` runs the same checks and refuses on any error, so a validated
run is an ingestible run.

``run`` and ``evaluate`` exist **behind the pre-registration guard** (research R3 of the
2026-09-02 viewpoint feature): ``run`` refuses unless ``configs/frozen/<prereg>`` matches the built
artifacts verbatim and refuses to overwrite any record, and a re-run is a new pre-registration.
This supersedes the earlier "no run command exists" stance recorded in ``COMP-SKY-006`` — the
guard is now the placeret-proven mechanism rather than absence.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hsreloc.observation import read_session
from hsreloc.simret import adapter, geometry, groups, prereg, report, runner, sets, sources
from hsreloc.simret.conventions import SimConventions


def _load(path: str) -> tuple:
    p = Path(path).resolve()
    return json.loads(p.read_text(encoding="utf-8")), p.parent


def _resolve(base: Path, value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def _run_dirs(cfg: dict, base: Path) -> list:
    """The run directories a command operates on: one (``run_dir``) or all under ``raw_root``."""
    if cfg.get("run_dir"):
        return [_resolve(base, cfg["run_dir"])]
    if cfg.get("raw_root"):
        root = _resolve(base, cfg["raw_root"])
        dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "settings.json").exists())
        if not dirs:
            # the extended batch nests runs one level deeper (<raw_root>/<level>/Run_*)
            dirs = sorted(p for lvl in root.iterdir() if lvl.is_dir()
                          for p in lvl.iterdir() if p.is_dir() and (p / "settings.json").exists())
        if not dirs:
            raise adapter.SimRunError(f"no run folders (with settings.json) under {root}")
        return dirs
    raise adapter.SimRunError("config declares neither run_dir nor raw_root")


def cmd_validate(cfg: dict, base: Path) -> dict:
    conv = SimConventions.from_dict(cfg.get("conventions"))
    out, n_bad = {}, 0
    for rd in _run_dirs(cfg, base):
        run = adapter.read_run(rd)
        rep = adapter.validate_run(run, conv, cfg.get("options"))
        status = "PASS" if rep.ok else "REFUSED"
        print(f"[simret] validate {rd.name} (run_id {run.run_id}, {len(run.rows)} rows): {status} "
              f"— {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
        for e in rep.errors:
            print(f"  ERR  {e}")
        n_bad += 0 if rep.ok else 1
        out[rd.name] = rep.as_dict()
    print(f"[simret] validate: {len(out) - n_bad}/{len(out)} runs pass")
    return out


def cmd_ingest(cfg: dict, base: Path) -> dict:
    store = _resolve(base, cfg["store_root"])
    conv = SimConventions.from_dict(cfg.get("conventions"))
    out = {}
    for rd in _run_dirs(cfg, base):
        run_id = adapter.read_run(rd).run_id
        session_id = cfg.get("session_id") or run_id
        if (store / session_id / "session.json").exists():
            print(f"[simret] ingest {rd.name}: session {session_id} already present, kept")
            out[session_id] = {"kept": True}
            continue
        summary = adapter.ingest_run(rd, store, session_id=session_id, conventions=conv,
                                     options=cfg.get("options"))
        print(f"[simret] ingest {rd.name} -> {summary['session_dir']}: "
              f"{summary['n_observations']} observations, {summary['n_seam_curves']} GT seam "
              f"curves, {summary['n_database_holes']} holes, GT {summary['gt_status_counts']}")
        out[session_id] = summary
    return out


def _store_sessions(cfg: dict, base: Path) -> list:
    store = _resolve(base, cfg["store_root"])
    listed = cfg.get("sessions")
    if listed:
        return [s for s in listed if (store / s / "observations.csv").exists()]
    return sorted(p.name for p in store.iterdir() if (p / "observations.csv").exists())


def cmd_extract_dp(cfg: dict, base: Path) -> dict:
    from hsreloc.extraction.evaluate import method_config_digest
    from hsreloc.placeret.sources import DP_METHOD

    store = _resolve(base, cfg["store_root"])
    params = (cfg.get("dp") or {}).get("params", {})
    method_block = {"id": DP_METHOD, "params": dict(params)}
    digest = method_config_digest(method_block)
    per_session = {}
    for sid in _store_sessions(cfg, base):
        summary = sources.extract_dp_session(
            store, sid, params=params,
            run_root=_resolve(base, cfg["run_dir"]) if cfg.get("run_dir") else None)
        print(f"[simret] extract-dp {sid}: {len(summary['stored'])}/{summary['n_images']} stored, "
              f"{len(summary['skipped'])} refused")
        per_session[sid] = {"n_images": summary["n_images"], "n_ok": len(summary["stored"]),
                            "n_refused": len(summary["skipped"]),
                            "refusals": summary["skipped"]}
    # Merge into an existing manifest rather than rewriting it: the pilot batch's entries are
    # evidence inputs of a closed experiment and must survive a later batch's extraction. A digest
    # mismatch would mean a different method configuration and is refused, never merged over.
    existing_path = store / "dp_extraction_manifest.json"
    if existing_path.exists():
        prev = json.loads(existing_path.read_text(encoding="utf-8"))
        if prev.get("method_config_digest") != digest:
            raise sources.SourceError(
                f"existing DP manifest digest {str(prev.get('method_config_digest'))[:16]}… != "
                f"{digest[:16]}… — refusing to merge extractions of different method configs")
        merged = dict(prev.get("sessions", {}))
        merged.update(per_session)
        per_session = merged
    manifest = {
        "method_id": DP_METHOD, "method_params": dict(params),
        "method_config_digest": digest,
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sessions": per_session,
        "n_ok_total": sum(s["n_ok"] for s in per_session.values()),
        "n_refused_total": sum(s["n_refused"] for s in per_session.values()),
    }
    (store / "dp_extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[simret] extract-dp: manifest -> {store / 'dp_extraction_manifest.json'} "
          f"(digest {digest[:12]}…)")
    return manifest


def cmd_silver_list(cfg: dict, base: Path) -> dict:
    store = _resolve(base, cfg["store_root"])
    session_ids = _store_sessions(cfg, base)
    out_path = sources.write_silver_frame_list(store, session_ids,
                                               _resolve(base, cfg.get("silver_frame_list",
                                                                      "silver_frames.csv")))
    print(f"[simret] silver frame list -> {out_path} ({len(session_ids)} sessions; feed it to "
          f"scripts/silver_infer.py with mode 'sessions', in the ISOLATED venv)")
    return {"frame_list": str(out_path), "sessions": session_ids}


def cmd_groups(cfg: dict, base: Path) -> dict:
    out_dir = cfg.get("groups_out") or cfg["out_dir"]
    index = groups.build_index(
        _resolve(base, cfg["store_root"]), session_ids=cfg.get("sessions"),
        anchors_csv=_resolve(base, cfg["anchors_csv"]) if cfg.get("anchors_csv") else None,
        assignments_csv=_resolve(base, cfg["assignments_csv"]) if cfg.get("assignments_csv") else None,
        translation_bins_m=cfg.get("translation_bins_m"))
    manifest = groups.write_index(index, _resolve(base, out_dir))
    print(f"[simret] groups -> {out_dir}: {manifest['n_observations']} observations, "
          f"{manifest['n_anchors']} anchors, {manifest['n_unassigned']} unassigned, "
          f"{manifest['n_pose_groups']} pose groups")
    return manifest


def _sessions_by_id(store: Path, wanted: set) -> dict:
    return {sid: read_session(store / sid)[1] for sid in sorted(wanted)}


def _reference_sessions(cfg: dict, base: Path) -> set:
    """Sessions containing the declared reference observations (never parsed from the id)."""
    t = cfg["tasks"]
    if t.get("reference_session"):
        return {t["reference_session"]}
    store = _resolve(base, cfg["store_root"])
    smap = sources.session_map(store)
    refs = t["reference_observation_ids"]
    missing = [r for r in refs if r not in smap]
    if missing:
        raise geometry.GeometryError(f"reference observation(s) not in the store: {missing}")
    return {smap[r] for r in refs}


def cmd_tasks(cfg: dict, base: Path) -> dict:
    store = _resolve(base, cfg["store_root"])
    t = cfg["tasks"]
    ref_sessions = _reference_sessions(cfg, base)
    out = {}
    for name, query_key, out_key in (("pilot", "query_sessions", "tasks_dir"),
                                     ("deform", "deform_query_sessions", "deform_tasks_dir")):
        query_sessions = list(t.get(query_key) or [])
        if not query_sessions:
            print(f"[simret] tasks: no {query_key} declared, skipping the {name} task")
            continue
        sessions = _sessions_by_id(store, set(query_sessions) | ref_sessions)
        task_cfg = {
            "query_sessions": query_sessions,
            "spacings_m": t["spacings_m"],
            "stage_rule": t.get("stage_rule", {}),
        }
        if t.get("reference_session"):
            task_cfg["reference_session"] = t["reference_session"]
        else:
            task_cfg["reference_observation_ids"] = t["reference_observation_ids"]
        if t.get("translation_bins_m"):
            task_cfg["translation_bins_m"] = t["translation_bins_m"]
        built = geometry.build_tasks(sessions, task_cfg)
        manifest = geometry.write_tasks(built, _resolve(base, cfg[out_key]))
        print(f"[simret] tasks[{name}] -> {cfg[out_key]}: grids {manifest['grid_counts']}, "
              f"{manifest['n_query_rows']} query rows, stages {manifest['stage_counts']}")
        out[name] = manifest
    return out


def _camera_dims(store: Path, session_ids: list) -> tuple:
    dims = set()
    for sid in session_ids:
        meta = json.loads((store / sid / "session.json").read_text(encoding="utf-8"))
        dims.add(tuple(meta["camera"]["resolution_px"]))
    if len(dims) != 1:
        raise sets.SetError(f"sessions disagree on resolution: {sorted(dims)} — one experiment, "
                            f"one camera geometry")
    (w, h), = dims
    return int(w), int(h)


def _anchor_map(cfg: dict, base: Path) -> dict:
    out_dir = cfg.get("groups_out") or cfg.get("out_dir")
    if not out_dir or not (_resolve(base, out_dir) / "groups_manifest.json").exists():
        return {}
    index = groups.load_index(_resolve(base, out_dir))
    return {r["observation_id"]: r["anchor_id"] for r in index["rows"] if r.get("anchor_id")}


def cmd_sets(cfg: dict, base: Path) -> dict:
    store = _resolve(base, cfg["store_root"])
    tasks = geometry.load_tasks(_resolve(base, cfg["tasks_dir"]))
    all_sessions = _store_sessions(cfg, base)
    width, height = _camera_dims(store, all_sessions)
    session_objs = _sessions_by_id(store, set(cfg["tasks"]["query_sessions"])
                                   | _reference_sessions(cfg, base))
    obs_by_id = {o.observation_id: o for s in session_objs.values() for o in s}
    ref_session = sorted(_reference_sessions(cfg, base))[0]
    session_meta = json.loads((store / ref_session / "session.json").read_text(encoding="utf-8"))
    smap = sources.session_map(store, all_sessions)
    anchors = _anchor_map(cfg, base)
    written = {"query_sets": {}, "reference_sets": {}}
    for spacing in cfg["tasks"]["spacings_m"]:
        for key, spec in cfg["sources"].items():
            src = sources.build_source(key, spec, smap, width, height, store_root=store,
                                       config_dir=base)
            manifest = sets.write_reference_set(tasks, cfg["task_name"], spacing, key, src,
                                                _resolve(base, cfg["refdb_root"]))
            written["reference_sets"][manifest["reference_source"]["ref_set_id"]] = {
                "n_references": manifest["reference_source"]["n_references"],
                "n_holes": manifest["reference_source"]["n_holes"]}
            out = sets.write_query_set(tasks, cfg["task_name"], spacing, key,
                                       _resolve(base, cfg["datasets_root"]), obs_by_id,
                                       width, height, session_meta=session_meta, anchors=anchors)
            written["query_sets"][out.name] = sum(1 for q in tasks["queries"]
                                                 if float(q["spacing_m"]) == float(spacing))
        if len(cfg["sources"]) > 1:
            sets.assert_source_materialisations_identical(
                _resolve(base, cfg["datasets_root"]), cfg["task_name"], spacing,
                list(cfg["sources"]))
    print(f"[simret] sets: {len(written['query_sets'])} query sets, "
          f"{len(written['reference_sets'])} reference sets")
    for rid, info in sorted(written["reference_sets"].items()):
        print(f"  {rid}: {info['n_references']} references, {info['n_holes']} database holes")
    return written


# --------------------------------------------------------------------------------------------------
# experiments A / B / C (no retrieval; research R6/R7)
# --------------------------------------------------------------------------------------------------

def _built_sources(cfg: dict, base: Path, keys=None) -> tuple:
    store = _resolve(base, cfg["store_root"])
    all_sessions = _store_sessions(cfg, base)
    width, height = _camera_dims(store, all_sessions)
    smap = sources.session_map(store, all_sessions)
    built = {key: sources.build_source(key, spec, smap, width, height, store_root=store,
                                       config_dir=base)
             for key, spec in cfg["sources"].items() if keys is None or key in keys}
    return built, width, height


def _condition_rows(cfg: dict, base: Path) -> list:
    """Experiment-index rows of the single-observation (condition-capture) sessions."""
    index = groups.load_index(_resolve(base, cfg["groups_out"]))
    per_session: dict = {}
    for r in index["rows"]:
        per_session.setdefault(r["session_id"], []).append(r)
    return [rows[0] for rows in per_session.values() if len(rows) == 1]


def cmd_exp_extraction(cfg: dict, base: Path) -> dict:
    built, width, height = _built_sources(cfg, base)
    gt = built["sim_exact"]
    rows = sorted(_condition_rows(cfg, base), key=lambda r: r["observation_id"])
    ids = [r["observation_id"] for r in rows]
    labels = {r["observation_id"]: f"{r['anchor_id']}|{r['condition_time_of_day']}|{r['condition_clouds']}"
              for r in rows}
    meta = {r["observation_id"]: r for r in rows}
    out_dir = _resolve(base, cfg["experiments"]["extraction_out"])
    result = {}
    for key in ("segformer", "dp"):
        acc = report.extraction_accuracy(gt, built[key], ids, height, conditions=labels)
        for row in acc["rows"]:                       # add the single-axis slices
            m = meta[row["observation_id"]]
            row["anchor_id"] = m["anchor_id"]
            row["time_of_day"] = m["condition_time_of_day"]
            row["clouds"] = m["condition_clouds"]
            row["median_abs_frac_height"] = row["median_abs_px"] / float(height)
        acc["by_anchor"] = report.summarise_by(acc["rows"], "anchor_id",
                                               ["median_abs_px", "p90_abs_px", "catastrophic_frac"])
        acc["by_time_of_day"] = report.summarise_by(acc["rows"], "time_of_day",
                                                    ["median_abs_px", "p90_abs_px", "catastrophic_frac"])
        acc["by_clouds"] = report.summarise_by(acc["rows"], "clouds",
                                               ["median_abs_px", "p90_abs_px", "catastrophic_frac"])
        report.write_table(acc, out_dir, f"accuracy_{key}")
        result[key] = {"n_evaluated": acc["n_evaluated"], "n_refused": acc["n_refused"],
                       "refusal_rate": acc["refusal_rate"]}
        print(f"[simret] exp-extraction {key}: {acc['n_evaluated']} evaluated, "
              f"{acc['n_refused']} refused")
    return result


def cmd_exp_consistency(cfg: dict, base: Path) -> dict:
    built, width, height = _built_sources(cfg, base)
    rows = _condition_rows(cfg, base)
    res = report.condition_consistency(rows, built, height,
                                       catastrophic_frac=float(
                                           cfg["experiments"].get("catastrophic_frac_height", 0.05)))
    out_dir = _resolve(base, cfg["experiments"]["consistency_out"])
    report.write_table(res, out_dir, "consistency")
    print(f"[simret] exp-consistency: {res['n_pose_groups']} pose groups, pairs per source "
          f"{res['n_pairs_per_source']}, catastrophic {res['catastrophic_counts']}, "
          f"{len(res['refusals'])} refusals")
    return {"n_pose_groups": res["n_pose_groups"],
            "catastrophic_counts": res["catastrophic_counts"]}


def _diagnostic_matchers(cfg: dict) -> dict:
    from hsreloc.matchers import build_matcher
    out = {"c0_frozen_ncc": build_matcher("c0_frozen_ncc", {})}
    for variant, spec in (cfg["experiments"].get("diagnostic_matchers") or {}).items():
        out[variant] = build_matcher(variant, spec)
    return out


def cmd_exp_deformation(cfg: dict, base: Path) -> dict:
    store = _resolve(base, cfg["store_root"])
    tasks = geometry.load_tasks(_resolve(base, cfg["deform_tasks_dir"]))
    built, width, height = _built_sources(cfg, base, keys=("sim_exact",))
    session_objs = _sessions_by_id(store, set(cfg["tasks"]["deform_query_sessions"])
                                   | _reference_sessions(cfg, base))
    observations = {o.observation_id: o for s in session_objs.values() for o in s}
    res = report.deformation_vs_translation(tasks, observations, built["sim_exact"], height,
                                            matchers=_diagnostic_matchers(cfg))

    # direction classes from the inventory's geometric classification (never folder names)
    classification = json.loads(
        _resolve(base, cfg["experiments"]["classification_json"]).read_text(encoding="utf-8"))
    cls_by_session = {t["run_folder"]: (t["class"], t["direction"])
                      for t in classification["translation_runs"]}
    smap = sources.session_map(store, sorted(session_objs))
    bins = [0.0, 1.0, 7.5, 17.5, 37.5, 75.0, 150.0]
    for row in res["rows"]:
        sid = smap[row["query_id"]]
        klass, direction = cls_by_session.get(sid, ("unclassified", ""))
        row["direction_class"] = klass
        row["direction"] = direction
        axis_value = {"lateral": row["lateral_m"], "longitudinal": row["along_m"],
                      "vertical": row["up_m"]}.get(klass)
        row["axis_displacement_m"] = None if axis_value is None else abs(float(axis_value))
        row["axis_displacement_bin"] = (None if axis_value is None
                                        else geometry.assign_bin(abs(float(axis_value)), bins))
    for klass in ("lateral", "longitudinal", "vertical"):
        members = [r for r in res["rows"] if r["direction_class"] == klass]
        res[f"summary_{klass}"] = report.summarise_by(
            members, "axis_displacement_bin",
            ["curve_median_abs_px", "curve_median_abs_offset_removed_px", "curve_offset_px"]
            + [f"{n}_score" for n in _diagnostic_matchers(cfg)]
            + [f"{n}_shift" for n in _diagnostic_matchers(cfg)]) if members else {}
    vertical = [r for r in res["rows"] if r["direction_class"] == "vertical"]
    res["vertical_diagnostic"] = {
        "n": len(vertical),
        "note": ("offset = what a constant vertical shift (and therefore mean removal) absorbs; "
                 "offset-removed = the shape change no vertical shift explains"),
        "per_bin": report.summarise_by(vertical, "axis_displacement_bin",
                                       ["curve_offset_px", "curve_median_abs_px",
                                        "curve_median_abs_offset_removed_px"]) if vertical else {},
    }
    out_dir = _resolve(base, cfg["experiments"]["deformation_out"])
    report.write_table(res, out_dir, "deformation_gt")
    (out_dir / "vertical_diagnostic.json").write_text(
        json.dumps(res["vertical_diagnostic"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = {k: sum(1 for r in res["rows"] if r["direction_class"] == k)
              for k in ("lateral", "longitudinal", "vertical", "unclassified")}
    print(f"[simret] exp-deformation: {len(res['rows'])} pairs ({counts}), "
          f"{len(res['refused'])} refused")
    return {"n_rows": len(res["rows"]), "direction_counts": counts}


def cmd_exp_variant(cfg: dict, base: Path) -> dict:
    from hsreloc.matchers import build_matcher

    spec_block = cfg["experiments"].get("variant_matchers")
    if not spec_block:
        raise report.ReportError(
            "experiments.variant_matchers is not declared — the successor comparison runs only "
            "after the R9 decision has named a justified rung and its bounds (with provenance); "
            "declare it in the config at decision time, never before")
    matchers = {"c0_frozen_ncc": build_matcher("c0_frozen_ncc", {})}
    for variant, spec in spec_block.items():
        matchers[variant] = build_matcher(variant, spec)
    tasks = geometry.load_tasks(_resolve(base, cfg["tasks_dir"]))
    out_dir = _resolve(base, cfg["experiments"]["variant_out"])
    built, width, height = _built_sources(cfg, base)
    result = {}
    for key in cfg["experiments"].get("variant_sources", ["sim_exact", "segformer"]):
        res = report.variant_retrieval(tasks, built[key], matchers)
        report.write_table(res, out_dir, f"variant_retrieval_{key}")
        result[key] = res["top1_counts"]
        print(f"[simret] exp-variant {key}: top-1 counts {res['top1_counts']} of "
              f"{res['n_queries']} queries")
    return result


# --------------------------------------------------------------------------------------------------
# the pre-registered event: prereg -> run -> evaluate (research R3)
# --------------------------------------------------------------------------------------------------

def cmd_prereg(cfg: dict, base: Path) -> dict:
    built, width, height = _built_sources(cfg, base)
    descriptions = {k: {kk: vv for kk, vv in s.describe().items() if kk != "refusal_mode"}
                    for k, s in built.items()}
    task_manifest = geometry.load_tasks(_resolve(base, cfg["tasks_dir"]))["manifest"]
    deform_manifest = geometry.load_tasks(_resolve(base, cfg["deform_tasks_dir"]))["manifest"]
    path = _resolve(base, cfg["preregistration_ref"])
    pre = prereg.write_prereg(path, cfg, task_manifest, deform_manifest, descriptions)
    print(f"[simret] prereg -> {path} (task {pre['task']['task_digest'][:12]}…). COMMIT it before "
          f"`run` — the run refuses without a committed, matching pre-registration.")
    return pre


def cmd_run(cfg: dict, base: Path) -> dict:
    store = _resolve(base, cfg["store_root"])
    all_sessions = _store_sessions(cfg, base)
    width, height = _camera_dims(store, all_sessions)
    smap = sources.session_map(store, all_sessions)
    tasks = geometry.load_tasks(_resolve(base, cfg["tasks_dir"]))
    result = runner.run_matrix(cfg, base, smap, width, height, tasks)
    print(f"[simret] run: {len(result['runs'])} records, invariants "
          f"{'OK' if all(c['invariant_ok'] for c in result['cells']) else 'VIOLATED'}")
    return result


def cmd_evaluate(cfg: dict, base: Path) -> dict:
    tasks = geometry.load_tasks(_resolve(base, cfg["tasks_dir"]))
    done = runner.evaluate_all(cfg, base, tasks)
    return {"evaluations": done}


def cmd_report(cfg: dict, base: Path) -> dict:
    from hsreloc.simret import final_report
    return final_report.build(cfg, base)


# ==================================================================================================
# ext-* commands — the FINAL-validation feature (spec 20260903-225416; hsreloc.simret.ext).
# Population discipline: `--population dev` touches the DEV level only; `--population final`
# refuses without the committed, verbatim-matching freeze prereg (contracts/freeze.md).
# ==================================================================================================

def cmd_ext_index(cfg: dict, base: Path, population: str = None) -> dict:
    from hsreloc.simret import extindex
    index = extindex.build_index(cfg, base)
    manifest = extindex.write_index(index, _resolve(base, cfg["index_out"]))
    print(f"[simret] ext-index -> {cfg['index_out']}: {manifest['n_rows']} rows, "
          f"{manifest['n_anchor_captures']} anchor captures, {manifest['n_swipe_rows']} swipe "
          f"rows, {manifest['n_gt_holes']} GT holes; per level "
          f"{ {k: v['anchors'] for k, v in manifest['per_level'].items()} }")
    return manifest


def cmd_ext_sets(cfg: dict, base: Path, population: str = None) -> dict:
    from hsreloc.simret import ext
    manifests = ext.build_all_tasks(cfg, base)
    written = ext.build_all_sets(cfg, base)
    return {"tasks": {k: m["content_digest"] for k, m in manifests.items()}, "sets": written}


def _ext_guarded(fn):
    def wrapped(cfg: dict, base: Path, population: str = None) -> dict:
        from hsreloc.simret import ext
        if population is None:
            raise ext.ExtError("this command needs --population dev|final")
        ext.guard_population(cfg, base, population)
        return fn(cfg, base, population)
    return wrapped


@_ext_guarded
def cmd_ext_exp_extraction(cfg, base, population):
    from hsreloc.simret import extreport
    return extreport.exp_extraction(cfg, base, population)


@_ext_guarded
def cmd_ext_exp_consistency(cfg, base, population):
    from hsreloc.simret import extreport
    return extreport.exp_consistency(cfg, base, population)


@_ext_guarded
def cmd_ext_exp_deformation(cfg, base, population):
    from hsreloc.simret import extreport
    return extreport.exp_deformation(cfg, base, population)


def cmd_ext_run(cfg: dict, base: Path, population: str = None) -> dict:
    from hsreloc.simret import ext
    result = ext.run_cells(cfg, base, population or "dev")
    print(f"[simret] ext-run[{result['population']}]: {len(result['runs'])} records")
    return result


def cmd_ext_evaluate(cfg: dict, base: Path, population: str = None) -> dict:
    from hsreloc.simret import ext
    return {"evaluations": ext.evaluate_cells(cfg, base, population or "dev")}


@_ext_guarded
def cmd_ext_exp_exactpose(cfg, base, population):
    from hsreloc.simret import extreport
    return extreport.exp_exactpose(cfg, base, population)


@_ext_guarded
def cmd_ext_exp_swipes(cfg, base, population):
    from hsreloc.simret import extreport
    return extreport.exp_swipes(cfg, base, population)


def _pilot_rank_rows(cfg: dict, base: Path) -> list:
    """The prior pilot's records as DEV calibration evidence (EXP-SKY-007, village level)."""
    from hsreloc.simret import extreport
    rows = []
    for skey in ("sim_exact", "segformer"):
        record = _resolve(base, cfg["runs_root"]) / f"sim-pilot-s250-ncc-{skey}"
        qs = _resolve(base, cfg["datasets_root"]) / f"sim-pilot-s250-{skey}"
        rs = _resolve(base, cfg["refdb_root"]) / f"sim-refs-pilot-s250-{skey}"
        if record.exists():
            rows.extend(extreport.rank_rows_from_dirs(record, qs, rs, task="pilot", source=skey))
    return rows


def cmd_ext_accept(cfg: dict, base: Path, population: str = None) -> dict:
    from hsreloc.simret import acceptance, ext, extreport
    out_root = _resolve(base, cfg["experiments"]["acceptance_out"])
    if population in (None, "dev"):
        # Calibration evidence (declared, research R8): the pilot's village records + this
        # feature's DEV records, GT + SegFormer rows (the primary comparison — DP is a baseline
        # chain, reported under the rule, not calibrated on).
        rows = _pilot_rank_rows(cfg, base)
        for kind in ("exactpose", "swipes"):
            for skey in ("sim_exact", "segformer"):
                rows.extend(extreport.rank_rows(cfg, base, kind, "village", skey))
        res = acceptance.calibrate(rows, cfg["acceptance_variant"]["grid"])
        res["calibration_rows"] = {"n": len(rows),
                                   "population": "pilot(sim_exact+segformer) + DEV village "
                                                 "exactpose/swipes (sim_exact+segformer)"}
        acceptance.write_result(out_root / "dev" / "calibration.json", res)
        c = res["chosen"]
        print(f"[simret] ext-accept[dev]: chose theta_s={c['theta_s']} theta_m={c['theta_m']} "
              f"({res['rule_path']}); precision {c['accepted_precision']}, coverage "
              f"{c['coverage_of_attainable']}, confident-false {c['n_confident_false_strict']} "
              f"of {c['n_rows']} rows -> record it in acceptance_variant.frozen before ext-prereg")
        return res
    from hsreloc.simret import ext as _ext
    _ext.guard_population(cfg, base, "final")
    frozen = cfg["acceptance_variant"]["frozen"]
    result = {}
    for kind, key, skey in _ext.cells_for(cfg, "final"):
        rows = extreport.rank_rows(cfg, base, kind, key, skey)
        result[f"{kind}/{key}/{skey}"] = acceptance.side_by_side(rows, frozen)
    acceptance.write_result(out_root / "final" / "side_by_side.json",
                            {"frozen": frozen, "results": result})
    print(f"[simret] ext-accept[final]: {len(result)} cells, frozen rule vs "
          f"{acceptance.VARIANT_NAME} -> {out_root / 'final' / 'side_by_side.json'}")
    return result


def cmd_ext_prereg(cfg: dict, base: Path, population: str = None) -> dict:
    from hsreloc.simret import ext, prereg as _prereg
    built = ext.assemble_final_prereg(cfg, base)
    path = _resolve(base, cfg["preregistration_ref"])
    pre = _prereg.write_final_prereg(path, built)
    print(f"[simret] ext-prereg -> {path}. COMMIT it (with the config and decision records) — "
          f"that commit IS the freeze; every `--population final` command checks it verbatim.")
    return pre


@_ext_guarded
def cmd_ext_exp_variant(cfg, base, population):
    """The pre-registered successor comparison (matcher-decision.md): C0 + the frozen successor on
    identical pairs, whole-database ranking, per task x level of the population — rank-1 AND
    false-match behaviour, never rank-1 alone."""
    from hsreloc.matchers import build_matcher
    from hsreloc.simret import ext, report
    succ = cfg.get("matcher_successor")
    if not succ:
        raise report.ReportError("matcher_successor is not declared — the variant comparison runs "
                                 "only after the R7 decision (matcher-decision.md) is recorded")
    matchers = {"c0_frozen_ncc": build_matcher("c0_frozen_ncc", {}),
                succ["variant"]: build_matcher(succ["variant"], succ["spec"])}
    tasks = ext.load_all_tasks(cfg, base)
    ctx = ext.context(cfg, base)
    out_root = _resolve(base, cfg["experiments"]["variant_out"]) / population
    result = {}
    for kind, key, skey in ext.cells_for(cfg, population):
        if skey not in cfg["experiments"].get("variant_sources", []):
            continue
        name = ext.task_name(kind, key)
        src = ext.build_sources(cfg, base, ctx, keys=(skey,))[skey]
        res = report.variant_retrieval(tasks[name], src, matchers)
        report.write_table(res, out_root, f"variant_{name}_{skey}")
        result[f"{name}/{skey}"] = res["top1_counts"]
        print(f"[simret] ext-exp-variant[{population}] {name}/{skey}: top-1 {res['top1_counts']} "
              f"of {res['n_queries']}")
    return result


def cmd_ext_exp_hard(cfg: dict, base: Path, population: str = None) -> dict:
    from hsreloc.simret import extreport
    return extreport.exp_hard(cfg, base)


def cmd_ext_report(cfg: dict, base: Path, population: str = None) -> dict:
    from hsreloc.simret import extfinal
    return extfinal.build(cfg, base)


EXT_COMMANDS = {
    "ext-index": cmd_ext_index, "ext-sets": cmd_ext_sets,
    "ext-exp-extraction": cmd_ext_exp_extraction, "ext-exp-consistency": cmd_ext_exp_consistency,
    "ext-exp-deformation": cmd_ext_exp_deformation,
    "ext-run": cmd_ext_run, "ext-evaluate": cmd_ext_evaluate,
    "ext-exp-exactpose": cmd_ext_exp_exactpose, "ext-exp-swipes": cmd_ext_exp_swipes,
    "ext-accept": cmd_ext_accept, "ext-prereg": cmd_ext_prereg,
    "ext-exp-variant": cmd_ext_exp_variant,
    "ext-exp-hard": cmd_ext_exp_hard, "ext-report": cmd_ext_report,
}

COMMANDS = {"validate": cmd_validate, "ingest": cmd_ingest, "extract-dp": cmd_extract_dp,
            "silver-list": cmd_silver_list, "groups": cmd_groups, "tasks": cmd_tasks,
            "sets": cmd_sets,
            "exp-extraction": cmd_exp_extraction, "exp-consistency": cmd_exp_consistency,
            "exp-deformation": cmd_exp_deformation, "exp-variant": cmd_exp_variant,
            "prereg": cmd_prereg, "run": cmd_run, "evaluate": cmd_evaluate,
            "report": cmd_report}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hsreloc.simret.cli", description=__doc__)
    ap.add_argument("command", choices=sorted(COMMANDS) + sorted(EXT_COMMANDS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--population", choices=("dev", "final"), default=None,
                    help="ext-* commands: dev = the DEV level only; final = behind the freeze")
    args = ap.parse_args(argv)
    cfg, base = _load(args.config)
    if args.command in EXT_COMMANDS:
        result = EXT_COMMANDS[args.command](cfg, base, population=args.population)
    else:
        result = COMMANDS[args.command](cfg, base)
    return 0 if result is not None else 1


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
