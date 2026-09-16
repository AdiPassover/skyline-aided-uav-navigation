"""Orchestration: curve -> profile -> rank -> acceptance -> fix -> record.

One function, deliberately linear and readable, because this is the code whose correctness the whole
attribution argument rests on. If a Stage-2 number is poor, it must be the representation's fault and
not this file's -- so this file does as little as possible and does it in one place.

The same code runs Stage 1 and Stage 2. Only the curve source, the data and the declared tier differ.
That is the point: a synthetic proof of this chain is a proof of the chain the real run uses.
"""

from __future__ import annotations

import time
from pathlib import Path

from hsreloc.retrieval import acceptance as acceptance_mod
from hsreloc.retrieval import baselines, profile as profile_mod, rank, record
from hsreloc.retrieval.fix import derive_fix
from hsreloc.retrieval.runconfig import PreflightResult, RunConfig, preflight


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _portable(path) -> str:
    """Repo-relative where possible, absolute otherwise.

    A result record is evidence and gets committed, so a path inside the repository must not be
    written as ``C:/Users/<somebody>/...`` -- that identifies one machine rather than one input, and
    the next reader cannot resolve it. External data (the main checkout's heavy artifacts) genuinely
    lives outside the repo and stays absolute, because there the machine-specific location *is* the
    honest answer.
    """
    if path is None:
        return None
    path = Path(path)
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def execute(config: RunConfig, curve_source=None, overwrite: bool = False) -> dict:
    """Run one configured matcher pass and write its spec-006 result record.

    Returns a summary dict; the record itself is the deliverable.
    """
    pre: PreflightResult = preflight(config, curve_source=curve_source)
    return _execute_prechecked(config, pre, overwrite=overwrite)


def _execute_prechecked(config: RunConfig, pre: PreflightResult, overwrite: bool = False) -> dict:
    qs = pre.query_set
    refs = pre.reference_set.references
    use_mock = config.baseline == baselines.MOCK
    scorer = None if use_mock else baselines.get_scorer(config.baseline)
    mock_scorer = baselines.MockScorer(config.mock_seed) if use_mock else None

    rows = []
    outcome_counts: dict = {}
    for query in qs.queries:
        started = time.perf_counter_ns()

        curve = pre.curve_source.get(query.observation_id)
        query_profile = profile_mod.normalize(curve, config.profile)
        degenerate = profile_mod.is_degenerate(query_profile,
                                               config.acceptance.degenerate_profile_std)

        if degenerate:
            candidates = []
        elif use_mock:
            candidates = rank.rank_by_identity(query.query_id, refs, mock_scorer)
        else:
            candidates = rank.rank_candidates(query_profile, refs, scorer)

        decision = acceptance_mod.decide(
            [c.score for c in candidates], config.acceptance,
            query_profile_degenerate=degenerate,
        )

        est_east = est_north = matched_east = matched_north = None
        if decision.is_success:
            best = candidates[0]
            fix = derive_fix(best.reference_id, best.east_m, best.north_m, pre.origin)
            est_east, est_north = fix.east_m, fix.north_m
            matched_east, matched_north = best.east_m, best.north_m

        elapsed = time.perf_counter_ns() - started
        outcome_counts[decision.outcome] = outcome_counts.get(decision.outcome, 0) + 1
        rows.append(record.QueryRow(
            query_id=query.query_id,
            timestamp_s=query.timestamp_s,
            outcome=decision.outcome,
            process_time_ns=elapsed,
            confidence=decision.confidence,
            best_score=decision.best_score,
            second_best_score=decision.second_best_score,
            score_margin=decision.score_margin,
            n_viable_candidates=decision.n_viable_candidates,
            est_east_m=est_east,
            est_north_m=est_north,
            matched_east_m=matched_east,
            matched_north_m=matched_north,
            compass_prior_deg=query.compass_prior_deg,
            candidates=[(c.east_m, c.north_m, c.score)
                        for c in rank.shortlist(candidates, config.recall_k)],
        ))

    relocalizer_config = {
        "config_digest": config.config_digest,
        "query_curves_digest": pre.query_curves_digest,
        "baseline": config.baseline,
        "recall_k": config.recall_k,
        "lag_search": config.lag_search,
        "profile": config.profile.as_dict(),
        "acceptance": config.acceptance.as_dict(),
        "geodetic_origin": pre.origin.as_dict(),
        "curve_source": pre.curve_source.describe(),
        "reference_curve_source": pre.reference_set.curve_source_used,
        "paths": {
            "reference_set": _portable(config.reference_set),
            "query_set": _portable(config.query_set),
            "query_curves_root": _portable(config.query_curves_root),
            "config": _portable(config.config_path),
        },
        # Stated in the record rather than left to be inferred: this matcher cannot know a query is
        # outside coverage without reading ground truth, so it never claims so (research R7).
        "outcomes_never_emitted": ["OUT_OF_COVERAGE"],
        "heading": "not estimated -- dataset heading quality is unsupported (spec FR-013)",
        "evidence_tier": config.evidence_tier,
        "evidence_caveat": config.evidence_caveat,
    }

    record_dir = record.write_record(
        config.output_root, config.run_id, rows,
        query_set_id=qs.dataset_id,
        query_set_revision=qs.dataset_revision,
        reference_source=pre.reference_set.reference_source,
        relocalizer_config=relocalizer_config,
        n_candidate_slots=config.recall_k,
        compass_prior={"mode": "absent"},
        overwrite=overwrite,
    )
    _write_consumed_manifest(record_dir, pre.consumed)

    return {
        "record_dir": record_dir,
        "n_queries": len(rows),
        "outcomes": outcome_counts,
        "query_curves_digest": pre.query_curves_digest,
        "config_digest": config.config_digest,
        "reference_curve_source": pre.reference_set.curve_source_used,
    }


def _write_consumed_manifest(record_dir: Path, consumed: list) -> None:
    """Exactly which curve files were read, and their digests.

    *Correction, 2026-08-23:* this was first documented as untracked. It is not -- ``.gitignore``'s
    ``skyline_runs/*`` pattern matches only one level, so ``<run_id>/sky/`` is re-included by the
    ``!skyline_runs/*/`` negation and this file is committed alongside the record. That is fine, and
    arguably better (it is genuine per-curve provenance), but it means the paths in it must be
    portable like the manifest's, which they now are.

    Nothing load-bearing lives here regardless: the tracked manifest's ``query_curves_digest``
    summarises the whole listing in one value.
    """
    import csv as _csv

    sky_dir = record_dir / "sky"
    sky_dir.mkdir(parents=True, exist_ok=True)
    with (sky_dir / "consumed_curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=["query_index", "observation_id", "session_id",
                                                "source_ref", "digest", "n_columns"])
        writer.writeheader()
        writer.writerows([{**row, "source_ref": _portable(row["source_ref"])} for row in consumed])
