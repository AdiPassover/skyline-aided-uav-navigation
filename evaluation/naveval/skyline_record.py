"""Skyline result-record loading + skyline query-set loading (spec 004).

Two file-based inputs to the skyline evaluator, kept in one small module:

- the **skyline result record** (``contracts/skyline-result-record.md``) -- a per-query CSV
  plus a manifest, the sole interface between the relocalizer and its evaluation (DEC-002),
  a sibling of ``runrecord.py`` for a per-query (not per-frame-trajectory) product;
- the **skyline query set** (``contracts/skyline-query-set.md``) -- a ``naveval`` dataset
  (reused loader, geographic + heading ground truth) plus an additive per-query sidecar
  (roll/pitch, compass prior, in-coverage flag).

Malformed input is an error, not a warning: a truncated, version-mismatched, or
outcome-inconsistent record must not be silently analysed (same discipline as runrecord.py).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from naveval.dataset import Dataset, load_dataset
from naveval.errors import ContractViolationError, DatasetMismatchError, SchemaVersionError

SUPPORTED_SCHEMA_MAJOR = 1

# NOT_ATTEMPTED is a trigger-time (D3) outcome; a standalone D2 record attempts every
# query, so it must not appear here (spec 004, contracts/skyline-result-record.md).
VALID_OUTCOMES = {"SUCCESS", "REJECTED", "AMBIGUOUS", "OUT_OF_COVERAGE", "EXTRACTION_FAILURE"}
VALID_HEADING_SOURCES = {"compass_prior", "skyline_estimated", "compass_refined"}

# Required-present / required-empty position fields per outcome (the contract table).
_POSITION_FIELDS = ("est_east_m", "est_north_m")
_OUTCOME_POSITION_REQUIRED = {"SUCCESS"}
_OUTCOME_POSITION_FORBIDDEN = {"REJECTED", "OUT_OF_COVERAGE", "EXTRACTION_FAILURE"}
# AMBIGUOUS: position forbidden (no single consumable fix), confidence required.
_OUTCOME_POSITION_FORBIDDEN |= {"AMBIGUOUS"}


def _major_version(schema_version: str) -> int:
    try:
        return int(schema_version.split(".", 1)[0])
    except (ValueError, IndexError, AttributeError) as exc:
        raise SchemaVersionError(f"Malformed schema_version: {schema_version!r}") from exc


def _float_or_none(s: Optional[str]) -> Optional[float]:
    if s is None or s == "":
        return None
    low = s.strip().lower()
    if low in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
        raise ContractViolationError(f"NaN/Inf is not a valid field value: {s!r}")
    return float(s)


def _int_or_none(s: Optional[str]) -> Optional[int]:
    if s is None or s == "":
        return None
    return int(s)


# ---------------------------------------------------------------------------
# Skyline result record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkylineRunManifest:
    schema_version: str
    run_id: str
    query_set_id: str
    query_set_revision: str
    relocalizer_id: str
    relocalizer_version: str
    relocalizer_config: dict
    reference_db: dict
    compass_prior: dict
    environment: dict
    run_timestamp: str
    query_count: int
    processed_count: int
    completed: bool
    # additive (spec 006): historical/generic source block; see contracts/result-record.md.
    # Defaulted so existing (spec 004/005) constructors that pass only reference_db keep working.
    reference_source: dict = field(default_factory=dict)

    @property
    def is_target_hardware(self) -> bool:
        return bool(self.environment.get("is_target_hardware", False))


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    timestamp_s: float
    outcome: str
    est_east_m: Optional[float]
    est_north_m: Optional[float]
    est_heading_deg: Optional[float]
    heading_source: Optional[str]
    compass_prior_deg: Optional[float]
    confidence: Optional[float]
    best_score: Optional[float]
    second_best_score: Optional[float]
    score_margin: Optional[float]
    n_viable_candidates: Optional[int]
    alignment_residual: Optional[float]
    matched_east_m: Optional[float]
    matched_north_m: Optional[float]
    process_time_ns: int
    candidates: tuple = ()  # tuple of (east, north, score) if the record exposed a shortlist


@dataclass
class SkylineResultRecord:
    manifest: SkylineRunManifest
    results: list  # list[QueryResult], in file order


def load_skyline_record(root: Path | str) -> SkylineResultRecord:
    """Load and validate a skyline result-record directory."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    queries_path = root / "queries.csv"
    if not manifest_path.exists():
        raise ContractViolationError(f"Missing manifest.json in {root}")
    if not queries_path.exists():
        raise ContractViolationError(f"Missing queries.csv in {root}")

    with manifest_path.open("r", encoding="utf-8") as f:
        m = json.load(f)

    schema_version = m.get("schema_version", "")
    if _major_version(schema_version) != SUPPORTED_SCHEMA_MAJOR:
        raise SchemaVersionError(
            f"Unsupported skyline-record schema major version: {schema_version!r} "
            f"(this reader supports major version {SUPPORTED_SCHEMA_MAJOR})"
        )

    # Source block (spec 006, contracts/result-record.md): a record carries exactly one of
    # `reference_db` (DEM, spec 004/005) or `reference_source` (historical/generic). A present-but-
    # empty `reference_db: {}` counts as the (empty) DEM block, so existing records are unaffected;
    # the additive rule only rejects a record that declares BOTH non-empty, or declares NEITHER key.
    reference_db = m.get("reference_db", {})
    reference_source = m.get("reference_source", {})
    if "reference_db" not in m and "reference_source" not in m:
        raise ContractViolationError(
            "a skyline record must carry a reference_db or reference_source block "
            "(contracts/result-record.md)"
        )
    if reference_db and reference_source:
        raise ContractViolationError(
            "a skyline record must carry exactly one non-empty source block, not both "
            "reference_db and reference_source (contracts/result-record.md)"
        )

    manifest = SkylineRunManifest(
        schema_version=schema_version,
        run_id=m["run_id"],
        query_set_id=m["query_set_id"],
        query_set_revision=m["query_set_revision"],
        relocalizer_id=m.get("relocalizer_id", ""),
        relocalizer_version=m.get("relocalizer_version", ""),
        relocalizer_config=m.get("relocalizer_config", {}),
        reference_db=reference_db,
        reference_source=reference_source,
        compass_prior=m.get("compass_prior", {}),
        environment=m.get("environment", {}),
        run_timestamp=m.get("run_timestamp", ""),
        query_count=int(m["query_count"]),
        processed_count=int(m["processed_count"]),
        completed=bool(m["completed"]),
    )

    results: list = []
    seen_ids: set = set()
    with queries_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for row in reader:
            outcome = row["outcome"]
            if outcome not in VALID_OUTCOMES:
                raise ContractViolationError(
                    f"Invalid outcome {outcome!r}; expected one of {sorted(VALID_OUTCOMES)} "
                    f"(NOT_ATTEMPTED is trigger-time and invalid in a standalone D2 record)"
                )

            qid = row["query_id"]
            if qid in seen_ids:
                raise ContractViolationError(f"Duplicate query_id {qid!r}")
            seen_ids.add(qid)

            est_east = _float_or_none(row.get("est_east_m"))
            est_north = _float_or_none(row.get("est_north_m"))
            est_heading = _float_or_none(row.get("est_heading_deg"))
            heading_source = row.get("heading_source") or None
            confidence = _float_or_none(row.get("confidence"))

            # Per-outcome required/forbidden position fields.
            has_position = est_east is not None and est_north is not None
            if outcome in _OUTCOME_POSITION_REQUIRED and not has_position:
                raise ContractViolationError(
                    f"outcome={outcome} requires est_east_m/est_north_m (query_id {qid!r})"
                )
            if outcome in _OUTCOME_POSITION_FORBIDDEN and has_position:
                raise ContractViolationError(
                    f"outcome={outcome} must not carry a consumable position (query_id {qid!r})"
                )
            if outcome == "SUCCESS":
                if _float_or_none(row.get("matched_east_m")) is None or _float_or_none(row.get("matched_north_m")) is None:
                    raise ContractViolationError(
                        f"SUCCESS requires matched_east_m/matched_north_m (query_id {qid!r})"
                    )
                if confidence is None:
                    raise ContractViolationError(f"SUCCESS requires a confidence (query_id {qid!r})")
            if outcome in ("REJECTED", "AMBIGUOUS") and confidence is None:
                raise ContractViolationError(f"outcome={outcome} requires a confidence (query_id {qid!r})")

            # heading present => heading_source present and valid; heading in range.
            if est_heading is not None:
                if not (0.0 <= est_heading < 360.0):
                    raise ContractViolationError(
                        f"est_heading_deg must be in [0, 360), got {est_heading} (query_id {qid!r})"
                    )
                if heading_source is None:
                    raise ContractViolationError(
                        f"est_heading_deg present requires heading_source (query_id {qid!r})"
                    )
            if heading_source is not None and heading_source not in VALID_HEADING_SOURCES:
                raise ContractViolationError(
                    f"Invalid heading_source {heading_source!r}; expected one of {sorted(VALID_HEADING_SOURCES)}"
                )

            prior = _float_or_none(row.get("compass_prior_deg"))
            if prior is not None and not (0.0 <= prior < 360.0):
                raise ContractViolationError(
                    f"compass_prior_deg must be in [0, 360), got {prior} (query_id {qid!r})"
                )

            pt_raw = row.get("process_time_ns", "")
            if pt_raw == "" or pt_raw is None:
                raise ContractViolationError(f"process_time_ns must not be empty (query_id {qid!r})")
            process_time_ns = int(pt_raw)
            if process_time_ns < 0:
                raise ContractViolationError(f"process_time_ns must be >= 0 (query_id {qid!r})")

            # Optional ranked shortlist columns cand{k}_east_m/north_m/score.
            candidates = []
            k = 0
            while f"cand{k}_east_m" in headers:
                ce = _float_or_none(row.get(f"cand{k}_east_m"))
                cn = _float_or_none(row.get(f"cand{k}_north_m"))
                cs = _float_or_none(row.get(f"cand{k}_score"))
                if ce is not None and cn is not None:
                    candidates.append((ce, cn, cs))
                k += 1

            results.append(QueryResult(
                query_id=qid,
                timestamp_s=float(row["timestamp_s"]),
                outcome=outcome,
                est_east_m=est_east,
                est_north_m=est_north,
                est_heading_deg=est_heading,
                heading_source=heading_source,
                compass_prior_deg=prior,
                confidence=confidence,
                best_score=_float_or_none(row.get("best_score")),
                second_best_score=_float_or_none(row.get("second_best_score")),
                score_margin=_float_or_none(row.get("score_margin")),
                n_viable_candidates=_int_or_none(row.get("n_viable_candidates")),
                alignment_residual=_float_or_none(row.get("alignment_residual")),
                matched_east_m=_float_or_none(row.get("matched_east_m")),
                matched_north_m=_float_or_none(row.get("matched_north_m")),
                process_time_ns=process_time_ns,
                candidates=tuple(candidates),
            ))

    if not results:
        raise ContractViolationError("Skyline record has zero query results")
    if len(results) != manifest.processed_count:
        raise ContractViolationError(
            f"queries.csv has {len(results)} rows but manifest.processed_count is "
            f"{manifest.processed_count} -- the record is truncated or the manifest is stale"
        )

    return SkylineResultRecord(manifest=manifest, results=results)


# ---------------------------------------------------------------------------
# Skyline query set (base dataset + additive sidecar)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryGroundTruth:
    query_index: int
    east_m: Optional[float]
    north_m: Optional[float]
    heading_deg: Optional[float]
    roll_deg: Optional[float]
    pitch_deg: Optional[float]
    compass_prior_deg: Optional[float]
    in_coverage: bool
    conditions: dict = field(default_factory=dict)


@dataclass
class SkylineQuerySet:
    dataset: Dataset                 # reused base (evidence tier/caveat, quality classes, conventions)
    camera: dict                     # skyline.camera block
    compass_prior_spec: dict         # skyline.compass_prior block
    queries: dict                    # query_index -> QueryGroundTruth

    @property
    def query_set_id(self) -> str:
        return self.dataset.dataset_id

    @property
    def query_set_revision(self) -> str:
        return self.dataset.dataset_revision


def load_skyline_query_set(root: Path | str) -> SkylineQuerySet:
    """Load a skyline query set: the reused base dataset plus the additive sidecar."""
    root = Path(root)
    dataset = load_dataset(root)  # reuse: conventions, tiers/caveat, quality, geographic GT

    with (root / "dataset.json").open("r", encoding="utf-8") as f:
        descriptor = json.load(f)
    skyline_block = descriptor.get("skyline", {})
    camera = skyline_block.get("camera", {})
    compass_prior_spec = skyline_block.get("compass_prior", {})

    # Geographic + heading GT come from the reused base dataset. For a per-query set there
    # is one GT sample per query; associate by exact timestamp (index-aligned in practice).
    ts_to_gt: dict = {}
    for i in range(dataset.gt_timestamps.size):
        ts_to_gt[round(float(dataset.gt_timestamps[i]), 6)] = i

    def gt_for_query(query_index: int) -> tuple:
        ts = round(float(dataset.frame_timestamps[query_index]), 6)
        j = ts_to_gt.get(ts)
        if j is None:
            return (None, None, None)
        import math
        east = None if math.isnan(dataset.gt_east[j]) else float(dataset.gt_east[j])
        north = None if math.isnan(dataset.gt_north[j]) else float(dataset.gt_north[j])
        head = None if math.isnan(dataset.gt_heading[j]) else float(dataset.gt_heading[j])
        return (east, north, head)

    queries: dict = {}
    sidecar_path = root / "skyline_queries.csv"
    if not sidecar_path.exists():
        raise ContractViolationError(f"Missing skyline_queries.csv in {root}")
    with sidecar_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        base_cols = {"query_index", "roll_deg", "pitch_deg", "compass_prior_deg", "in_coverage"}
        for row in reader:
            qi = int(row["query_index"])
            east, north, head = gt_for_query(qi)
            conditions = {k: v for k, v in row.items() if k not in base_cols and v not in ("", None)}
            queries[qi] = QueryGroundTruth(
                query_index=qi,
                east_m=east,
                north_m=north,
                heading_deg=head,
                roll_deg=_float_or_none(row.get("roll_deg")),
                pitch_deg=_float_or_none(row.get("pitch_deg")),
                compass_prior_deg=_float_or_none(row.get("compass_prior_deg")),
                in_coverage=_parse_bool(row["in_coverage"]),
                conditions=conditions,
            )

    if not queries:
        raise ContractViolationError("Skyline query set has zero queries")
    return SkylineQuerySet(dataset=dataset, camera=camera, compass_prior_spec=compass_prior_spec, queries=queries)


def _parse_bool(s: str) -> bool:
    if s == "true":
        return True
    if s == "false":
        return False
    raise ContractViolationError(f"Invalid boolean field value: {s!r} (expected 'true' or 'false')")


def verify_record_matches_query_set(record: SkylineResultRecord, query_set: SkylineQuerySet) -> None:
    """Guard against evaluating a record against the wrong ground truth."""
    if record.manifest.query_set_id != query_set.query_set_id:
        raise DatasetMismatchError(
            f"Record references query_set_id={record.manifest.query_set_id!r} "
            f"but was evaluated against {query_set.query_set_id!r}"
        )
    if record.manifest.query_set_revision != query_set.query_set_revision:
        raise DatasetMismatchError(
            f"Record references query_set_revision={record.manifest.query_set_revision!r} "
            f"but the query set is at revision {query_set.query_set_revision!r}"
        )
