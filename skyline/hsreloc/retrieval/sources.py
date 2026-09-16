"""Curve sources: where a ``SkylineCurve`` comes from.

Contract: ``contracts/skyline-curve-source.md``. Research: ``research.md`` **R3** (the derived join),
**R11** (the synthetic construction).

Two producers ship in this feature. A third -- an automatic skyline extractor -- is a later spec's
job; it implements the same protocol, returns ``provenance="automatic:<method>"``, and requires no
change to any matcher module. Nothing here reads an image, and nothing here exposes ground truth.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from hsreloc.retrieval.skyline_curve import CurveError, SkylineCurve, make_curve

ORACLE_SUBDIR = "skylines_oracle"


class OracleCurveSource:
    """Human-verified oracle curves from a spec-006 observation store.

    Layout (frozen by ``contracts/observation-record.md``)::

        <root>/<session_id>/skylines_oracle/<observation_id>.csv     columns: col,row

    Research **R3**: nothing extra needs configuring, because the query set already determines which
    curve belongs to which query -- ``frames.csv``'s image stem *is* the ``observation_id`` and
    ``skyline_queries.csv``'s ``traversal_id`` *is* the ``session_id``. Only the store root is a
    configured value; the rest is derived. That is why this feature needs no change to a frozen
    contract.
    """

    kind = "observation_store"

    def __init__(self, root: Path | str, sessions: dict, image_height_px: int,
                 image_width_px: int, expect_provenance: str = "oracle:manual") -> None:
        """``sessions`` maps ``observation_id -> session_id`` (from the query set or reference set)."""
        self.root = Path(root)
        self.sessions = dict(sessions)
        self.image_height_px = int(image_height_px)
        self.image_width_px = int(image_width_px)
        self.expect_provenance = expect_provenance

    def curve_path(self, observation_id: str) -> Path:
        try:
            session_id = self.sessions[observation_id]
        except KeyError:
            raise CurveError(
                f"{observation_id}: no session known for this observation -- it is not in the "
                f"query/reference set this source was built from"
            ) from None
        return self.root / session_id / ORACLE_SUBDIR / f"{observation_id}.csv"

    def get(self, observation_id: str) -> SkylineCurve:
        path = self.curve_path(observation_id)
        if not path.exists():
            raise CurveError(f"{observation_id}: no oracle curve at {path}")
        rows: dict = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows[int(row["col"])] = float(row["row"])
        expected = list(range(self.image_width_px))
        if sorted(rows) != expected:
            raise CurveError(
                f"{observation_id}: stored curve at {path} does not cover columns "
                f"0..{self.image_width_px - 1} contiguously (has {len(rows)} columns)"
            )
        values = np.array([rows[c] for c in expected], dtype=np.float64)
        return make_curve(observation_id, values, self.image_width_px, self.image_height_px,
                          self.expect_provenance, str(path))

    def describe(self) -> dict:
        return {"kind": self.kind, "root": str(self.root), "n_known": len(self.sessions),
                "provenance": self.expect_provenance,
                "image_width_px": self.image_width_px, "image_height_px": self.image_height_px}


class SyntheticCurveSource:
    """In-memory curves with **constructed** answers -- Stage 1 only, tier T1.

    Holds curves it was handed. It exists so the matcher chain can be exercised with no filesystem
    and no dataset at all, which is what makes the unit tests independent of every real artifact.
    Provenance is ``oracle:sim_exact``: the curve is exactly known because it was constructed, which
    is a different claim from "a human drew it" and is labelled differently on purpose.
    """

    kind = "synthetic"

    def __init__(self, curves: dict, label: str = "synthetic") -> None:
        self._curves = dict(curves)
        self.label = label

    def get(self, observation_id: str) -> SkylineCurve:
        try:
            return self._curves[observation_id]
        except KeyError:
            raise CurveError(f"{observation_id}: not present in synthetic source {self.label!r}") from None

    def describe(self) -> dict:
        return {"kind": self.kind, "label": self.label, "n_curves": len(self._curves),
                "provenance": "oracle:sim_exact"}

    def ids(self) -> list:
        return sorted(self._curves)
