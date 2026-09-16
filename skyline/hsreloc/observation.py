"""Canonical skyline observation record I/O (spec 006 contract; produced by spec 007).

One *session* (a traversal/season) is a directory ``observations/<session_id>/`` holding a
``session.json`` (session-level metadata) and ``observations.csv`` (one row per frame), plus an
``images/`` subdir and, once annotated, ``skylines_oracle/``. This is the source-agnostic superset the
006 ``observation-record`` contract defines; the 007 builders down-project it into the 004 query set +
006 reference set later. No matcher/ranking logic lives here -- it is pure data I/O + validation.

Fields we populate are the required core (id/session/frame/timestamp/image/dims/position/yaw/gt_source)
plus whatever the source provides; anything not a named core field is preserved verbatim in ``extra`` so
condition tags, ``sky_fraction``, and ``sim_*`` fields round-trip without a schema change (the contract's
minor-version rule).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# The named core columns, in file order. Everything else round-trips through ``extra``.
_CORE_COLUMNS = [
    "observation_id", "session_id", "frame_index", "timestamp_s",
    "image_path", "image_width_px", "image_height_px",
    "pos_east_m", "pos_north_m", "up_m", "lat", "lon", "alt_m",
    "yaw_deg", "roll_deg", "pitch_deg", "fov_deg",
    "gt_source", "gt_pos_sigma_m", "gt_yaw_sigma_deg",
    "oracle_skyline_path", "oracle_provenance", "auto_skyline_path",
]
_INT_COLUMNS = {"frame_index", "image_width_px", "image_height_px"}
_FLOAT_COLUMNS = {
    "timestamp_s", "pos_east_m", "pos_north_m", "up_m", "lat", "lon", "alt_m",
    "yaw_deg", "roll_deg", "pitch_deg", "fov_deg", "gt_pos_sigma_m", "gt_yaw_sigma_deg",
}


class ObservationError(Exception):
    """A session/observation record violates the 006 observation contract."""


def _num(value, is_int):
    if value is None or value == "":
        return None
    return int(float(value)) if is_int else float(value)


@dataclass
class Observation:
    observation_id: str
    session_id: str
    frame_index: int
    timestamp_s: float
    image_path: str
    image_width_px: int
    image_height_px: int
    gt_source: str
    pos_east_m: Optional[float] = None
    pos_north_m: Optional[float] = None
    up_m: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_m: Optional[float] = None
    yaw_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    fov_deg: Optional[float] = None
    gt_pos_sigma_m: Optional[float] = None
    gt_yaw_sigma_deg: Optional[float] = None
    oracle_skyline_path: Optional[str] = None
    oracle_provenance: Optional[str] = None
    auto_skyline_path: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Required-core check (006 observation-record 'minimum viable export')."""
        for name in ("observation_id", "session_id", "image_path", "gt_source"):
            if not getattr(self, name):
                raise ObservationError(f"observation missing required {name!r}")
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ObservationError(f"{self.observation_id}: image dimensions must be positive")
        has_enu = self.pos_east_m is not None and self.pos_north_m is not None
        has_geo = self.lat is not None and self.lon is not None
        if not (has_enu or has_geo):
            raise ObservationError(
                f"{self.observation_id}: needs a horizontal position (pos_east/north or lat/lon)"
            )

    def as_row(self) -> dict:
        row = {c: getattr(self, c) for c in _CORE_COLUMNS}
        for k, v in self.extra.items():
            if k not in row:
                row[k] = v
        return {k: ("" if v is None else v) for k, v in row.items()}


def _row_to_observation(row: dict) -> Observation:
    core = {}
    for c in _CORE_COLUMNS:
        raw = row.get(c, "")
        if c in _INT_COLUMNS:
            core[c] = _num(raw, True)
        elif c in _FLOAT_COLUMNS:
            core[c] = _num(raw, False)
        else:
            core[c] = (raw if raw != "" else None)
    extra = {k: v for k, v in row.items() if k not in _CORE_COLUMNS and v not in ("", None)}
    # required-core ints/strings must be present
    for req in ("observation_id", "session_id", "image_path", "gt_source"):
        if not core.get(req):
            raise ObservationError(f"observation row missing {req!r}")
    obs = Observation(
        observation_id=core["observation_id"], session_id=core["session_id"],
        frame_index=int(core["frame_index"]), timestamp_s=float(core["timestamp_s"]),
        image_path=core["image_path"], image_width_px=int(core["image_width_px"]),
        image_height_px=int(core["image_height_px"]), gt_source=core["gt_source"],
        pos_east_m=core["pos_east_m"], pos_north_m=core["pos_north_m"], up_m=core["up_m"],
        lat=core["lat"], lon=core["lon"], alt_m=core["alt_m"], yaw_deg=core["yaw_deg"],
        roll_deg=core["roll_deg"], pitch_deg=core["pitch_deg"], fov_deg=core["fov_deg"],
        gt_pos_sigma_m=core["gt_pos_sigma_m"], gt_yaw_sigma_deg=core["gt_yaw_sigma_deg"],
        oracle_skyline_path=core["oracle_skyline_path"], oracle_provenance=core["oracle_provenance"],
        auto_skyline_path=core["auto_skyline_path"], extra=extra,
    )
    obs.validate()
    return obs


# --- session-level metadata ---

_REQUIRED_SESSION_FIELDS = ("session_id", "platform", "frame_convention", "heading_convention",
                            "evidence_tier")


def write_session(session_dir: Path | str, session_meta: dict, observations: list[Observation]) -> Path:
    """Write ``session.json`` + ``observations.csv`` for one session; returns the dir."""
    session_dir = Path(session_dir)
    (session_dir / "images").mkdir(parents=True, exist_ok=True)
    for f in _REQUIRED_SESSION_FIELDS:
        if f not in session_meta:
            raise ObservationError(f"session.json missing required {f!r}")
    if session_meta.get("frame_convention") != "ENU":
        raise ObservationError("session frame_convention must be 'ENU' (DEC-004)")
    if not observations:
        raise ObservationError("a session must contain at least one observation")

    for obs in observations:
        obs.validate()
    # union of all columns (core + any extras across rows), core first in fixed order
    extra_cols = []
    for obs in observations:
        for k in obs.extra:
            if k not in _CORE_COLUMNS and k not in extra_cols:
                extra_cols.append(k)
    columns = _CORE_COLUMNS + sorted(extra_cols)

    with (session_dir / "session.json").open("w", encoding="utf-8") as f:
        json.dump(session_meta, f, indent=2, sort_keys=True)
    with (session_dir / "observations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for obs in observations:
            writer.writerow(obs.as_row())
    return session_dir


def read_session(session_dir: Path | str) -> tuple[dict, list[Observation]]:
    """Read ``session.json`` + ``observations.csv``; validate; return (meta, observations)."""
    session_dir = Path(session_dir)
    meta_path = session_dir / "session.json"
    obs_path = session_dir / "observations.csv"
    if not meta_path.exists() or not obs_path.exists():
        raise ObservationError(f"{session_dir} is not an observation session (missing session.json/observations.csv)")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    for fld in _REQUIRED_SESSION_FIELDS:
        if fld not in meta:
            raise ObservationError(f"session.json missing required {fld!r}")
    if meta.get("frame_convention") != "ENU":
        raise ObservationError("session frame_convention must be 'ENU' (DEC-004)")
    observations: list[Observation] = []
    with obs_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            observations.append(_row_to_observation(row))
    if not observations:
        raise ObservationError(f"{session_dir}/observations.csv has no rows")
    return meta, observations
