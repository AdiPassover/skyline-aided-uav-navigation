"""The pre-registered run matrix and its evaluation (research R3 of the viewpoint feature).

Mirrors ``hsreloc.placeret.runner``: for each (source, baseline) a matcher ``RunConfig`` is built in
memory — identical profile / match / acceptance / origin blocks, the evidence tier read from the
query set — and the **frozen** ``hsreloc.retrieval.run.execute`` runs it with the source injected.
Afterwards the cross-source invariants are asserted (same query-set revision and grid digest,
different provenance and curve digests), so a difference between two sources' results can only come
from their curves.

Guards, in order: ``run`` refuses without a ``configs/frozen`` pre-registration matching the built
artifacts verbatim (``hsreloc.simret.prereg``); refuses if any record already exists complete
(kept, never redone) or incomplete (hard error — delete deliberately); and the frozen matcher
itself refuses overwrites. This replaces the earlier "no run command exists" stance recorded in
``COMP-SKY-006`` — the guard is now the placeret-proven mechanism rather than absence.

``evaluate_all`` scores every record with the UNMODIFIED spec-006 evaluator, one generated config
per record, exactly as the naveval CLI would.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hsreloc.simret import sets as sets_mod
from hsreloc.simret.prereg import check_prereg
from hsreloc.simret.sources import PROVENANCE_BY_KEY, build_source
from hsreloc.placeret.sources import REFUSAL_RAISE, REFUSAL_SENTINEL
from hsreloc.retrieval.run import execute
from hsreloc.retrieval.runconfig import ConfigError, parse_config

#: Numeric axis slices the unmodified evaluator can compute from the query sidecar. The categorical
#: axes (cloud state, direction class) are sliced by the report layer from the same sidecar.
AXIS_SLICES = {
    "translation": {"field": "condition_translation_m", "bins": [0.0, 1.0, 7.5, 17.5, 37.5, 75.0, 150.0]},
    "vertical": {"field": "condition_up_diff_m", "bins": [-0.5, 0.5, 7.5, 17.5, 37.5, 75.0, 150.0]},
    "hour": {"field": "condition_hour", "bins": [0.0, 9.0, 15.0, 24.0]},
}


class RunnerError(Exception):
    """The run matrix cannot proceed as configured."""


def _resolve(config_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (config_dir / p).resolve()


def run_id_for(config: dict, spacing: float, baseline: str, source_key: str) -> str:
    pattern = config.get("run_id_pattern", "sim-pilot-s{spacing}-{baseline}-{source}")
    return pattern.format(spacing=f"{spacing:g}", baseline=baseline, source=source_key)


def load_sources(config: dict, config_dir: Path, sessions: dict, width: int, height: int,
                 refusal_mode: str = REFUSAL_RAISE) -> dict:
    store = _resolve(config_dir, config["store_root"])
    return {key: build_source(key, spec, sessions, width, height, store_root=store,
                              config_dir=config_dir, refusal_mode=refusal_mode)
            for key, spec in config["sources"].items()}


def matcher_run_config(config: dict, config_dir: Path, spacing: float, baseline: str,
                       source_key: str) -> dict:
    datasets_root = _resolve(config_dir, config["datasets_root"])
    refdb_root = _resolve(config_dir, config["refdb_root"])
    runs_root = _resolve(config_dir, config["runs_root"])
    qs_dir = datasets_root / sets_mod.query_set_id(config["task_name"], spacing, source_key)
    rs_dir = refdb_root / sets_mod.reference_set_id(config["task_name"], spacing, source_key)
    descriptor = json.loads((qs_dir / "dataset.json").read_text(encoding="utf-8"))
    origin = {k: config["geodetic_origin"][k] for k in ("lat_deg", "lon_deg", "alt_m")}
    # query_curves.root doubles as the frozen loader's reference cross-check root
    # (<root>/<session>/skylines_oracle/<rid>.csv). That cross-check is exactly right for the
    # oracle source — the npz must equal the store's GT curve — and exactly wrong for the
    # automatic sources, whose reference curves legitimately differ from GT; those point at their
    # own curve root instead (no skylines_oracle/ layout there, so the npz is authoritative).
    # The actual query curves always come from the injected curve_source, never from this root.
    if source_key == "sim_exact":
        curves_root = _resolve(config_dir, config["store_root"])
    elif source_key == "segformer":
        curves_root = _resolve(config_dir, config["sources"]["segformer"]["mask_root"])
    else:
        curves_root = rs_dir
    return {
        "schema_version": "1.1.0",
        "run_id": run_id_for(config, spacing, baseline, source_key),
        "inputs": {
            "reference_set": str(rs_dir),
            "query_set": str(qs_dir),
            "query_curves": {"kind": "observation_store",
                             "root": str(curves_root),
                             "expect_provenance": PROVENANCE_BY_KEY[source_key]},
        },
        "geodetic_origin": origin,
        "profile": config["matcher"]["profile"],
        "match": {"baseline": baseline, "recall_k": config["matcher"]["recall_k"],
                  "lag_search": config["matcher"].get("lag_search", False)},
        "acceptance": config["matcher"]["acceptance"],
        "output": {"root": str(runs_root)},
        "evidence": {"tier": descriptor["evidence_tier"], "caveat": descriptor["evidence_caveat"]},
        "environment": {"is_target_hardware": False},
    }


def run_matrix(config: dict, config_dir: Path, sessions: dict, width: int, height: int,
               tasks: dict) -> dict:
    """Every (source, baseline) cell through the frozen matcher, prereg-guarded."""
    sources = load_sources(config, config_dir, sessions, width, height,
                           refusal_mode=REFUSAL_SENTINEL)
    descriptions = {k: {kk: vv for kk, vv in s.describe().items() if kk != "refusal_mode"}
                    for k, s in sources.items()}
    pre = check_prereg(_resolve(config_dir, config["preregistration_ref"]), config,
                       tasks["manifest"], descriptions)

    runs, cells = [], []
    for spacing in [float(s) for s in config["tasks"]["spacings_m"]]:
        for baseline in config["baselines"]:
            cell = {}
            for key in config["sources"]:
                raw = matcher_run_config(config, config_dir, spacing, baseline, key)
                record_dir = _resolve(config_dir, config["runs_root"]) / raw["run_id"]
                existing = record_dir / "manifest.json"
                if existing.exists():
                    m = json.loads(existing.read_text(encoding="utf-8"))
                    if m.get("completed") and m.get("processed_count") == m.get("query_count"):
                        summary = {"record_dir": str(record_dir)}
                        print(f"[simret] {raw['run_id']}: complete record present, kept")
                    else:
                        raise RunnerError(f"{raw['run_id']}: an INCOMPLETE record exists at "
                                          f"{record_dir}; delete it explicitly before resuming")
                else:
                    try:
                        rc = parse_config(raw, base_dir=config_dir)
                        summary = execute(rc, curve_source=sources[key])
                        print(f"[simret] {raw['run_id']}: done")
                    except ConfigError as exc:
                        raise RunnerError(f"{raw['run_id']}: {exc}") from exc
                manifest = json.loads((Path(summary["record_dir"]) / "manifest.json")
                                      .read_text(encoding="utf-8"))
                grid_key = f"{spacing:g}"
                cell[key] = {"run_id": raw["run_id"], "record_dir": str(summary["record_dir"]),
                             "query_set_revision": manifest.get("query_set_revision"),
                             "query_set_id": manifest.get("query_set_id"),
                             "query_curves_digest": manifest.get("relocalizer_config", {})
                             .get("query_curves_digest"),
                             "provenance": PROVENANCE_BY_KEY[key],
                             "grid_digest": tasks["grids"][grid_key]["digest"]}
                runs.append(cell[key])
            cells.append(_cell_check(spacing, baseline, cell))
    return {"prereg": pre["created_utc"], "runs": runs, "cells": cells}


def _cell_check(spacing: float, baseline: str, cell: dict) -> dict:
    keys = sorted(cell)
    ok, reasons = True, []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = cell[keys[i]], cell[keys[j]]
            if a["query_set_revision"] != b["query_set_revision"]:
                ok = False; reasons.append(f"{keys[i]}/{keys[j]}: query_set_revision differs")
            if a["grid_digest"] != b["grid_digest"]:
                ok = False; reasons.append(f"{keys[i]}/{keys[j]}: grid digest differs")
            if a["provenance"] == b["provenance"]:
                ok = False; reasons.append(f"{keys[i]}/{keys[j]}: same provenance")
            if a["query_curves_digest"] == b["query_curves_digest"]:
                ok = False; reasons.append(f"{keys[i]}/{keys[j]}: identical query-curve digests")
    if not ok:
        raise RunnerError(f"s{spacing:g}/{baseline}: cross-source invariant violated: {reasons}")
    return {"spacing_m": spacing, "baseline": baseline, "records": cell, "invariant_ok": ok}


def evaluate_all(config: dict, config_dir: Path, tasks: dict) -> list:
    """One generated evaluator config per record; the unmodified naveval evaluator does the rest."""
    repo = Path(__file__).resolve().parents[3]
    if str(repo / "evaluation") not in sys.path:
        sys.path.append(str(repo / "evaluation"))
    from naveval.evaluate_skyline import _load_config, run_skyline_evaluation

    datasets_root = _resolve(config_dir, config["datasets_root"])
    runs_root = _resolve(config_dir, config["runs_root"])
    eval_root = _resolve(config_dir, config["evaluations_root"])
    eval_cfg_dir = _resolve(config_dir, config["evaluation_configs_root"])
    eval_cfg_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for spacing in [float(s) for s in config["tasks"]["spacings_m"]]:
        grid = tasks["grids"][f"{spacing:g}"]
        for baseline in config["baselines"]:
            for key in config["sources"]:
                run_id = run_id_for(config, spacing, baseline, key)
                cfg_path = eval_cfg_dir / f"eval-{run_id}.json"
                out_dir = eval_root / f"sim-retrieval-{run_id}"
                cfg = {
                    "schema_version": "1.0.0",
                    "evaluation_id": f"sim-retrieval-{run_id}",
                    "run_record": str(runs_root / run_id),
                    "query_set": str(datasets_root
                                     / sets_mod.query_set_id(config["task_name"], spacing, key)),
                    "operational_tolerance_m": grid["tau_pos_m"],
                    "near_tolerance_m": grid["tau_near_m"],
                    "recall_k": int(config["matcher"]["recall_k"]),
                    "confidence_reject_threshold": float(
                        config["matcher"]["acceptance"]["reject_threshold"]),
                    "operating_points": [0.3, 0.5, 0.7],
                    "generic_axis_slices": AXIS_SLICES,
                    "tier_pooling": "forbidden",
                }
                cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
                metrics = run_skyline_evaluation(_load_config(cfg_path), out_dir)
                done.append({"run_id": run_id, "metrics": str(out_dir / "metrics.json"),
                             "success_rate": metrics["retrieval"]["topological_success"]
                             ["success_rate"]})
                print(f"[simret] evaluated {run_id}: topological success "
                      f"{metrics['retrieval']['topological_success']['success_rate']}")
    return done
