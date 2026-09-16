"""Emit the spec-006 generic result record -- the sole matcher -> evaluator interface.

Contract: ``006/contracts/result-record.md`` (which is the spec-004 record plus the additive
``reference_source`` block). Consumer: ``naveval.skyline_record.load_skyline_record``, unmodified.

``DEC-002``: this file boundary is the *entire* interface. The evaluator reads the record with the
matcher absent and must be indifferent to how the ranking was produced -- so nothing here may encode
a matcher, a representation, or a language. NCC, L1, L2 and the mock differ only in the numbers.

The per-outcome field rules below are not this module's invention; they are what
``naveval.skyline_record`` actually enforces, and writing a record that violates them raises
``ContractViolationError`` at load rather than being silently scored:

* ``SUCCESS`` -> position **and** matched coordinates **and** confidence, all present
* ``REJECTED`` / ``AMBIGUOUS`` -> confidence present, position **absent**
* ``EXTRACTION_FAILURE`` / ``OUT_OF_COVERAGE`` -> position absent
* ``process_time_ns`` never empty; NaN/Inf never written anywhere
* ``processed_count`` equals the row count, or the record is read as truncated

Every load-bearing fact about a run lives in ``manifest.json`` -- the resolved config, the paths used,
the geodetic origin, the config digest and the query-curve digest -- because ``manifest.json`` and
``queries.csv`` are the two files ``.gitignore`` keeps tracked under ``skyline_runs/``. Anything
written beside them is a convenience derivable from them.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = "1.0.0"
RELOCALIZER_ID = "hsreloc.retrieval"
RELOCALIZER_VERSION = "1.0.0"

_BASE_COLUMNS = [
    "query_id", "timestamp_s", "outcome",
    "est_east_m", "est_north_m", "est_heading_deg", "heading_source", "compass_prior_deg",
    "confidence", "best_score", "second_best_score", "score_margin", "n_viable_candidates",
    "alignment_residual", "matched_east_m", "matched_north_m", "process_time_ns",
]


class RecordError(Exception):
    """A result record could not be written as the contract requires."""


def code_revision() -> Optional[str]:
    """The git commit this run was produced by, or ``None`` when it cannot be determined.

    Spec FR-005 requires the record to identify the code that produced it. ``None`` is returned
    honestly -- outside a checkout, or without git -- rather than substituting a placeholder that
    would read like a real revision. A ``-dirty`` suffix marks **modified tracked files**, because a
    run produced from an edited tree is not reproducible from the commit alone.

    Untracked files are deliberately excluded from the dirty check. A run writes its own record into
    the repository, so counting untracked files would mark *every* run dirty and the flag would carry
    no information at all. The cost is that a brand-new, never-committed source file would not be
    detected; that is the smaller error, and it is recorded here rather than left as a surprise.
    """
    try:
        root = Path(__file__).resolve().parents[3]
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, timeout=10)
        if sha.returncode != 0:
            return None
        revision = sha.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=root, capture_output=True, text=True, timeout=10)
        if dirty.returncode == 0 and dirty.stdout.strip():
            revision += "-dirty"
        return revision
    except (OSError, subprocess.SubprocessError):
        return None


def _num(value: Optional[float]) -> str:
    """Format a number for the record, or empty. NaN/Inf are refused, never written."""
    if value is None:
        return ""
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise RecordError(f"refusing to write a non-finite value ({v}) into a result record")
    return repr(v)


def _int(value: Optional[int]) -> str:
    return "" if value is None else str(int(value))


class QueryRow:
    """One relocalization attempt, in the shape the record needs."""

    def __init__(self, query_id: str, timestamp_s: float, outcome: str, process_time_ns: int,
                 confidence: Optional[float] = None, best_score: Optional[float] = None,
                 second_best_score: Optional[float] = None, score_margin: Optional[float] = None,
                 n_viable_candidates: Optional[int] = None,
                 est_east_m: Optional[float] = None, est_north_m: Optional[float] = None,
                 matched_east_m: Optional[float] = None, matched_north_m: Optional[float] = None,
                 compass_prior_deg: Optional[float] = None, candidates: Optional[list] = None) -> None:
        self.query_id = str(query_id)
        self.timestamp_s = float(timestamp_s)
        self.outcome = outcome
        self.process_time_ns = int(process_time_ns)
        self.confidence = confidence
        self.best_score = best_score
        self.second_best_score = second_best_score
        self.score_margin = score_margin
        self.n_viable_candidates = n_viable_candidates
        self.est_east_m = est_east_m
        self.est_north_m = est_north_m
        self.matched_east_m = matched_east_m
        self.matched_north_m = matched_north_m
        self.compass_prior_deg = compass_prior_deg
        self.candidates = list(candidates or [])   # [(east, north, score), ...]

    def as_row(self, n_candidate_slots: int) -> dict:
        row = {
            "query_id": self.query_id,
            "timestamp_s": _num(self.timestamp_s),
            "outcome": self.outcome,
            "est_east_m": _num(self.est_east_m),
            "est_north_m": _num(self.est_north_m),
            # Heading is never estimated by this feature (spec FR-013): the datasets it consumes
            # declare heading quality "unknown" and carry no compass prior, so any value here would
            # be fabricated. Both fields stay empty rather than being omitted, so the absence is
            # visible in the record itself.
            "est_heading_deg": "",
            "heading_source": "",
            "compass_prior_deg": _num(self.compass_prior_deg),
            "confidence": _num(self.confidence),
            "best_score": _num(self.best_score),
            "second_best_score": _num(self.second_best_score),
            "score_margin": _num(self.score_margin),
            "n_viable_candidates": _int(self.n_viable_candidates),
            "alignment_residual": "",
            "matched_east_m": _num(self.matched_east_m),
            "matched_north_m": _num(self.matched_north_m),
            "process_time_ns": str(self.process_time_ns),
        }
        for k in range(n_candidate_slots):
            if k < len(self.candidates):
                east, north, score = self.candidates[k]
                row[f"cand{k}_east_m"] = _num(east)
                row[f"cand{k}_north_m"] = _num(north)
                row[f"cand{k}_score"] = _num(score)
            else:
                row[f"cand{k}_east_m"] = ""
                row[f"cand{k}_north_m"] = ""
                row[f"cand{k}_score"] = ""
        return row


def write_record(out_root: Path | str, run_id: str, rows: list, *, query_set_id: str,
                 query_set_revision: str, reference_source: dict, relocalizer_config: dict,
                 n_candidate_slots: int, compass_prior: Optional[dict] = None,
                 environment: Optional[dict] = None, overwrite: bool = False) -> Path:
    """Write ``<out_root>/<run_id>/{manifest.json,queries.csv}``; return the record directory."""
    out_root = Path(out_root)
    record_dir = out_root / run_id
    if record_dir.exists() and not overwrite:
        raise RecordError(
            f"run_id {run_id!r} already exists at {record_dir} -- refusing to overwrite a recorded "
            f"run. Choose a new run_id; a re-run with different settings is a different run."
        )
    record_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RecordError("refusing to write a record with zero query results")

    columns = list(_BASE_COLUMNS)
    for k in range(n_candidate_slots):
        columns += [f"cand{k}_east_m", f"cand{k}_north_m", f"cand{k}_score"]

    with (record_dir / "queries.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row(n_candidate_slots))

    env = {"is_target_hardware": False}
    env.update(environment or {})
    if env.get("is_target_hardware"):
        raise RecordError(
            "is_target_hardware must be false: no target UAV hardware is specified repo-wide, so no "
            "onboard or real-time claim is available (Principle IX)"
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "query_set_id": query_set_id,
        "query_set_revision": query_set_revision,
        "relocalizer_id": RELOCALIZER_ID,
        "relocalizer_version": RELOCALIZER_VERSION,
        "code_revision": code_revision(),
        "relocalizer_config": relocalizer_config,
        # Exactly one non-empty source block. `reference_db` is the DEM line's and is omitted
        # entirely; a record carrying both would be rejected by the reader.
        "reference_source": reference_source,
        "compass_prior": compass_prior or {"mode": "absent"},
        "environment": env,
        "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query_count": len(rows),
        "processed_count": len(rows),
        "completed": True,
    }
    with (record_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return record_dir
