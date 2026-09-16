"""The matcher's view of a query set -- deliberately narrower than the evaluator's.

Spec FR-018 and the anti-leakage rule in ``plan.md``. Data model: ``data-model.md`` §4.

The spec-004 query set is **partitioned by consumer**. The evaluator gets everything; the matcher gets
only what a relocalizer could legitimately have at query time:

============================  =========================================================
the matcher may read          the matcher must never read
============================  =========================================================
``frames.csv``                ``groundtruth.csv`` (east/north/heading)
``skyline_queries.csv``:      ``skyline_queries.csv``: **``in_coverage``**
 ``query_index``,
 ``traversal_id``,
 ``compass_prior_deg``,
 ``roll_deg`` / ``pitch_deg``
``dataset.json`` descriptors
============================  =========================================================

This module is the enforcement, not a convention: it never opens ``groundtruth.csv`` and never parses
``in_coverage``, so a ``MatcherQuerySet`` has no field that could leak one. ``naveval``'s own
``load_skyline_query_set`` -- which *does* carry ground truth -- is used only by the evaluator, on the
other side of the file boundary.

Why it matters here specifically: 32 of the 61 spec-007 queries are out-of-database, and *whether the
matcher refuses them* is the acceptance result being measured. A matcher that could see
``in_coverage`` would be marking its own exam.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Field names that must never appear on a matcher-visible object. Asserted by the leakage test.
FORBIDDEN_FIELDS = ("in_coverage", "east_m", "north_m", "heading_deg", "gt_east_m", "gt_north_m")


class QuerySetError(Exception):
    """A query set is missing, malformed, or not the one the run expected."""


@dataclass(frozen=True)
class MatcherQuery:
    """One query, as the matcher may see it. Carries no ground truth by construction."""

    query_index: int
    query_id: str
    timestamp_s: float
    observation_id: str
    session_id: str
    compass_prior_deg: Optional[float]
    roll_deg: Optional[float]
    pitch_deg: Optional[float]
    #: The dataset's own declaration of how this query's skyline was obtained ("manual",
    #: "sim_exact", ...). Extraction provenance, not ground truth -- a relocalizer legitimately
    #: knows whether it was handed a human-verified curve or an automatically extracted one.
    oracle_provenance: Optional[str]


@dataclass(frozen=True)
class MatcherQuerySet:
    dataset_id: str
    dataset_revision: str
    evidence_tier: str
    evidence_caveat: Optional[str]
    image_width_px: int
    image_height_px: int
    local_frame_origin: Optional[dict]
    queries: list

    def session_map(self) -> dict:
        return {q.observation_id: q.session_id for q in self.queries}


def _opt_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def load_matcher_query_set(root: Path | str) -> MatcherQuerySet:
    """Read only the matcher-visible parts of a spec-004 skyline query set."""
    root = Path(root)
    descriptor_path = root / "dataset.json"
    frames_path = root / "frames.csv"
    sidecar_path = root / "skyline_queries.csv"
    for path in (descriptor_path, frames_path, sidecar_path):
        if not path.exists():
            raise QuerySetError(f"query set is missing {path.name} at {path}")

    with descriptor_path.open("r", encoding="utf-8") as f:
        descriptor = json.load(f)
    metadata = descriptor.get("metadata", {})
    width = metadata.get("image_width")
    height = metadata.get("image_height")
    if not width or not height:
        raise QuerySetError(
            f"{descriptor_path} declares no metadata.image_width/image_height; the matcher needs both "
            f"to validate that a curve covers every column"
        )

    frames: dict = {}
    with frames_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["frame_index"])
            frames[idx] = (float(row["timestamp_s"]), Path(row["image_path"]).stem)

    queries = []
    with sidecar_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qi = int(row["query_index"])
            if qi not in frames:
                raise QuerySetError(f"query_index {qi} has no matching row in frames.csv")
            timestamp_s, observation_id = frames[qi]
            session_id = row.get("traversal_id") or ""
            if not session_id:
                raise QuerySetError(
                    f"query_index {qi} has no traversal_id; the matcher cannot locate its oracle "
                    f"curve without knowing which session it came from (research R3)"
                )
            queries.append(MatcherQuery(
                query_index=qi,
                query_id=str(qi),          # the evaluator joins on the stringified query_index
                timestamp_s=timestamp_s,
                observation_id=observation_id,
                session_id=session_id,
                compass_prior_deg=_opt_float(row.get("compass_prior_deg")),
                roll_deg=_opt_float(row.get("roll_deg")),
                pitch_deg=_opt_float(row.get("pitch_deg")),
                oracle_provenance=(row.get("oracle_provenance") or None),
            ))
    if not queries:
        raise QuerySetError(f"{sidecar_path} has no query rows")
    queries.sort(key=lambda q: q.query_index)

    return MatcherQuerySet(
        dataset_id=descriptor["dataset_id"],
        dataset_revision=descriptor.get("dataset_revision", ""),
        evidence_tier=descriptor.get("evidence_tier", ""),
        evidence_caveat=descriptor.get("evidence_caveat"),
        image_width_px=int(width),
        image_height_px=int(height),
        local_frame_origin=descriptor.get("local_frame_origin"),
        queries=queries,
    )
