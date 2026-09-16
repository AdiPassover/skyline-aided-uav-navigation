"""Skyline relocalization evaluator CLI (spec 004).

Joins a skyline result record + a skyline query set + an evaluation config, computes the
004 core metrics with the relocalizer absent (DEC-002), and writes ``metrics.json`` (in the
same honest shape as ``naveval.report``: support-gated, evidence tier/caveat, omitted-with-
reason) plus the figures. Reuses ``report._metric_entry`` for per-entry shaping so the
honesty rules (value=None => omitted) are identical to the VO evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

from naveval import skyline
from naveval.errors import ContractViolationError
from naveval.report import _metric_entry
from naveval.skyline_record import (
    load_skyline_query_set,
    load_skyline_record,
    verify_record_matches_query_set,
)
from naveval.support import classify_heading, classify_position

SCHEMA_VERSION = "1.0.0"


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    digest_input = {k: v for k, v in cfg.items() if k != "config_digest"}
    cfg["config_digest"] = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True).encode("utf-8")
    ).hexdigest()
    base = config_path.parent
    # `run_record` is the canonical key (spec 004); `result_record` is accepted as an alias for
    # the generic/matcher-agnostic record naming used by spec 006's contract.
    record_rel = cfg.get("run_record", cfg.get("result_record"))
    if record_rel is None:
        raise KeyError("evaluation config must set 'run_record' (or the alias 'result_record')")
    cfg["_run_record_path"] = (base / record_rel).resolve()
    cfg["_query_set_path"] = (base / cfg["query_set"]).resolve()
    return cfg


def _pos_entries(name_prefix: str, stats: Optional[dict], support_level: str, units: str = "m") -> list:
    """Median / p90 / p95 / max entries from a distribution-stats dict (or omitted)."""
    if stats is None:
        return [_metric_entry(f"{name_prefix}_median", None, units, support_level,
                              omission_reason="no evaluable queries with ground truth")]
    return [
        _metric_entry(f"{name_prefix}_median", stats["median"], units, support_level),
        _metric_entry(f"{name_prefix}_p90", stats["p90"], units, support_level),
        _metric_entry(f"{name_prefix}_p95", stats["p95"], units, support_level),
        _metric_entry(f"{name_prefix}_max", stats["max"], units, support_level),
    ]


def run_skyline_evaluation(config: dict, out_dir: Path | str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record = load_skyline_record(config["_run_record_path"])
    query_set = load_skyline_query_set(config["_query_set_path"])
    verify_record_matches_query_set(record, query_set)

    tol = float(config["operational_tolerance_m"])
    reject = float(config.get("confidence_reject_threshold", 0.5))
    recall_k = int(config.get("recall_k", 1))
    n_bins = int(config.get("calibration_bins", 5))
    slice_cfg = config.get("condition_slices", {})
    # --- spec 006 additive config (all optional; absent => that metric is simply omitted) ---
    near = config.get("near_tolerance_m")
    near = float(near) if near is not None else None
    operating_points = config.get("operating_points")
    axis_cfg = config.get("generic_axis_slices", {})
    tier_pooling = config.get("tier_pooling", "forbidden")
    if tier_pooling != "forbidden":
        raise ContractViolationError(
            "evaluation config tier_pooling must be 'forbidden'; the evaluator never pools "
            "evidence tiers (spec 006 FR-M9)"
        )

    ds = query_set.dataset
    pos_qc = ds.position_quality.quality_class if ds.position_quality else None
    head_qc = ds.heading_quality.quality_class if ds.heading_quality else None
    pos_support = classify_position("skyline_position_error", pos_qc)
    head_support = classify_heading("skyline_heading_error", head_qc)

    # --- compute ---
    pos = skyline.position_metrics(record, query_set, tol)
    head = skyline.heading_metrics(record, query_set)
    region = skyline.region_in_candidates(record, query_set, tol, recall_k)
    rel = skyline.reliability_metrics(record, query_set, tol, reject)
    calib = skyline.confidence_calibration(record, query_set, tol, n_bins)
    latency = skyline.latency_metrics(record)
    footprint = skyline.database_footprint(record)
    slices = skyline.condition_slices(record, query_set, tol, slice_cfg)
    # spec 006 additive metrics (matcher-agnostic; VPR-Bench protocol, LIT-013)
    topological = skyline.topological_success(record, query_set, tol, recall_k)
    separation = skyline.score_separation(record, query_set, tol, near)
    prroc = skyline.pr_roc(record, query_set, tol, operating_points)
    by_distance = skyline.aliasing_by_distance(record, query_set, tol, near) if near is not None else None
    axis_slices = skyline.generic_axis_slices(record, query_set, tol, axis_cfg)

    # --- assemble metrics.json (report._metric_entry shape) ---
    entries: list = []
    # E1 position
    if pos_support == "unsupported":
        entries.append(_metric_entry("position_error_median_m", None, "m", pos_support,
                                     omission_reason="position ground-truth quality is unsupported"))
    else:
        entries += _pos_entries("position_error", pos["stats"], pos_support)
    entries.append(_metric_entry("region_in_candidates_top1_rate", region["top1_rate"], "fraction",
                                 pos_support if pos_support != "unsupported" else "unsupported",
                                 omission_reason="no in-coverage queries with ground truth" if region["top1_rate"] is None else None))
    if region["recall_at_k_rate"] is not None:
        entries.append(_metric_entry(f"recall_at_{recall_k}_rate", region["recall_at_k_rate"], "fraction", pos_support))
    # E1 heading (prior vs result)
    for label, stats in (("heading_error_result", head["result_stats"]),
                         ("heading_error_prior", head["prior_stats"]),
                         ("heading_improvement", head["improvement_stats"])):
        if head_support == "unsupported" or stats is None:
            entries.append(_metric_entry(f"{label}_median", None, "deg",
                                         "unsupported" if head_support == "unsupported" else head_support,
                                         omission_reason="no heading ground truth" if head_support == "unsupported" else "no queries with this heading"))
        else:
            entries.append(_metric_entry(f"{label}_median", stats["median"], "deg", head_support))
            entries.append(_metric_entry(f"{label}_p90", stats["p90"], "deg", head_support))
    # E2 reliability (outcome-only rates are always supported)
    for name in ("success_rate", "rejection_rate", "ambiguous_rate", "out_of_coverage_rate",
                 "extraction_failure_rate"):
        entries.append(_metric_entry(name, rel[name], "fraction", "supported"))
    for name in ("correct_localization_rate", "false_relocalization_rate",
                 "confident_false_relocalization_rate"):
        entries.append(_metric_entry(name, rel[name], "fraction",
                                     pos_support if pos_support != "unsupported" else "unsupported",
                                     omission_reason="position ground truth unsupported" if pos_support == "unsupported" else None))

    metrics_json = {
        "schema_version": SCHEMA_VERSION,
        "config_digest": config["config_digest"],
        "run_id": record.manifest.run_id,
        "query_set_id": query_set.query_set_id,
        "evidence_tier": ds.evidence_tier,
        "evidence_caveat": ds.evidence_caveat,
        "ground_truth_class": {"position": pos_qc, "heading": head_qc},
        "queries": {"total": rel["total"], "evaluable_success": rel["evaluable_success"]},
        "metrics": entries,
        "reliability_counts": rel["counts"],
        "runtime": {
            **latency["latency"],
            "peak_rss_mb": latency["peak_rss_mb"],
            "is_target_hardware": latency["is_target_hardware"],
            "hardware": latency["hardware"],
            "disclaimer": (None if latency["is_target_hardware"]
                           else "development hardware; NOT onboard performance (Principle IX)"),
        },
        "database": footprint,
        "confidence_calibration": (calib if calib is not None
                                   else {"state": "omitted",
                                         "omission_reason": "too few evaluable successes for calibration"}),
        "condition_slices": slices,
        # --- spec 006 additive blocks (matcher-agnostic retrieval + aliasing) ---
        "retrieval": {
            "topological_success": topological,
            "pr_roc": prroc,
        },
        "aliasing": {
            "score_separation": separation,
            "by_distance": by_distance,
        },
        "generic_axis_slices": axis_slices,
        "tier_pooling": tier_pooling,
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, sort_keys=True)

    # --- figures (best-effort; a plotting failure must not lose the numbers) ---
    try:
        from naveval import plots_skyline
        plots_skyline.render_all(record, query_set, config, out_dir, pos, head, rel, calib, footprint)
    except Exception as exc:  # pragma: no cover - plotting is non-critical to the metrics
        (out_dir / "figures").mkdir(exist_ok=True)
        (out_dir / "figures" / "PLOTTING_FAILED.txt").write_text(str(exc), encoding="utf-8")

    return metrics_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DEM-based skyline relocalization (spec 004)")
    parser.add_argument("--config", required=True, help="Path to a skyline EvaluationConfig JSON file")
    parser.add_argument("--out", required=True, help="Output directory for metrics.json and figures")
    args = parser.parse_args()
    config = _load_config(Path(args.config))
    run_skyline_evaluation(config, args.out)
    print(f"[skyline-eval] wrote {Path(args.out) / 'metrics.json'}")


if __name__ == "__main__":
    main()
