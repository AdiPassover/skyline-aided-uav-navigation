"""Load a spec-006 reference set: coordinates, curves, and the frame they live in.

Contract: ``006/contracts/reference-set.md`` (1.2.0). Research: ``research.md`` **R4**.

Two curve sources are accepted, and which one was used is recorded in the run manifest:

``curves.npz``
    The contract-canonical container. Preferred when present.

the observation store
    Fallback, keyed by ``reference_id`` -- which *is* the ``observation_id``. This matters more than
    it looks: ``curves.npz`` is gitignored, while the oracle curves it bundles are tracked, so the
    fallback is what lets the whole feature run in a clean lane checkout with no external data.

When **both** are available they are cross-checked and a disagreement is a hard error. Two sources of
truth that silently diverge would be a worse problem than either source being absent.

This module reads ``references.csv`` for coordinates and ``manifest.json`` for the reference-source
block and the set's ``local_frame_origin``. It never reads ground truth -- a reference's own position
is not ground truth about a query, it is the data being searched.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from hsreloc.retrieval.fix import GeodeticOrigin
from hsreloc.retrieval.profile import ProfileConfig, normalize
from hsreloc.retrieval.skyline_curve import SkylineCurve, make_curve

CURVES_NPZ = "curves.npz"
FROM_NPZ = "curves.npz"
FROM_STORE = "observation_store"
FROM_BOTH = "curves.npz+observation_store(cross-checked)"


class ReferenceSetError(Exception):
    """A reference set is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Reference:
    reference_id: str
    east_m: float
    north_m: float
    up_m: Optional[float]
    traversal_id: str
    extraction_mode: str
    curve: SkylineCurve
    profile: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ReferenceSet:
    ref_set_id: str
    references: list
    manifest: dict
    origin: Optional[GeodeticOrigin]
    curve_source_used: str

    @property
    def reference_source(self) -> dict:
        return dict(self.manifest.get("reference_source", {}))

    def with_profiles(self, config: ProfileConfig) -> "ReferenceSet":
        refs = [replace(r, profile=normalize(r.curve, config)) for r in self.references]
        return replace(self, references=refs)


def _read_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.exists():
        raise ReferenceSetError(f"reference set has no manifest.json at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_rows(root: Path) -> list:
    path = root / "references.csv"
    if not path.exists():
        raise ReferenceSetError(f"reference set has no references.csv at {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ReferenceSetError(f"{path} has no reference rows")
    return rows


def load_reference_set(root: Path | str, image_height_px: int,
                       observation_store_root: Optional[Path | str] = None,
                       profile_config: Optional[ProfileConfig] = None) -> ReferenceSet:
    """Load a reference set, its curves, and its declared frame."""
    root = Path(root)
    manifest = _read_manifest(root)
    rows = _read_rows(root)

    origin = None
    if manifest.get("local_frame_origin"):
        origin = GeodeticOrigin.from_dict(manifest["local_frame_origin"])

    npz_path = root / CURVES_NPZ
    npz = np.load(npz_path) if npz_path.exists() else None
    store_root = Path(observation_store_root) if observation_store_root else None
    if npz is None and store_root is None:
        raise ReferenceSetError(
            f"no reference curves available: {npz_path} is absent and no observation store root was "
            f"configured. curves.npz is gitignored, so a lane checkout must configure the store."
        )

    references = []
    for row in rows:
        rid = row["reference_id"]
        width = None
        from_npz = None
        if npz is not None:
            if rid not in npz.files:
                raise ReferenceSetError(f"{rid}: listed in references.csv but absent from {npz_path}")
            from_npz = np.asarray(npz[rid], dtype=np.float64)
            width = int(from_npz.size)

        from_store = None
        if store_root is not None:
            session = row.get("traversal_id") or ""
            store_path = store_root / session / "skylines_oracle" / f"{rid}.csv"
            if store_path.exists():
                cols = {}
                with store_path.open("r", encoding="utf-8", newline="") as f:
                    for r in csv.DictReader(f):
                        cols[int(r["col"])] = float(r["row"])
                if sorted(cols) != list(range(len(cols))):
                    raise ReferenceSetError(
                        f"{rid}: stored curve at {store_path} does not cover its columns contiguously")
                from_store = np.array([cols[c] for c in range(len(cols))], dtype=np.float64)
                width = int(from_store.size) if width is None else width
            elif from_npz is None:
                raise ReferenceSetError(
                    f"{rid}: no curve in {npz_path} and none at {store_path}")

        if from_npz is not None and from_store is not None:
            if from_npz.size != from_store.size or not np.allclose(from_npz, from_store, atol=1e-9):
                raise ReferenceSetError(
                    f"{rid}: curves.npz and the observation store disagree -- refusing to guess which "
                    f"is authoritative (npz {from_npz.size} samples, store {from_store.size})"
                )

        values = from_npz if from_npz is not None else from_store
        curve = make_curve(rid, values, width, image_height_px,
                           row.get("extraction_mode", "oracle:manual"),
                           str(npz_path if from_npz is not None else store_root))
        references.append(Reference(
            reference_id=rid,
            east_m=float(row["east_m"]),
            north_m=float(row["north_m"]),
            up_m=float(row["up_m"]) if row.get("up_m") else None,
            traversal_id=row.get("traversal_id", ""),
            extraction_mode=row.get("extraction_mode", ""),
            curve=curve,
        ))

    if npz is not None and store_root is not None:
        used = FROM_BOTH
    elif npz is not None:
        used = FROM_NPZ
    else:
        used = FROM_STORE

    ref_set = ReferenceSet(
        ref_set_id=manifest.get("reference_source", {}).get("ref_set_id", root.name),
        references=references,
        manifest=manifest,
        origin=origin,
        curve_source_used=used,
    )
    if profile_config is not None:
        ref_set = ref_set.with_profiles(profile_config)
    return ref_set
