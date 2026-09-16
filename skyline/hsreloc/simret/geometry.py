"""Pose-only viewpoint geometry for the staged simulator skyline-retrieval experiment (PROT-SKY-001 §3).

Consumes the **frozen spec-006 observation record** (``hsreloc.observation.Observation`` — ENU
position, compass yaw, optional pitch/roll) and nothing else. It does not know what a simulator
export looks like: the adapter that turns one into an observation session is the only
format-dependent piece and is deliberately not written until data exists.

What it computes, per query against its nearest reference (all from poses):

* the horizontal **translation** and its decomposition into **along-track** (along the reference's
  heading) and **lateral** (to the reference's right) components — DEC-004 compass convention, a
  unit heading vector at heading ψ is ``(sin ψ, cos ψ)`` in (East, North);
* the signed, wrapped **yaw / pitch / roll differences**;
* the **stage** the pair belongs to (0 same pose, 1 translation only, 2 translation + yaw, 3 attitude)
  and the declared **translation / yaw bin** — the axes the unchanged evaluator will slice by
  through ``generic_axis_slices`` once these fields ride ``skyline_queries.csv`` as ``condition_*``;
* **pose-defined correctness**: the nearest grid reference within τ_pos = spacing / 2 (unique by the
  grid's construction), exactly as ``DEC-SKY-006`` R2 fixed it for ECL;
* the **first-order far-field prediction** (LIT-SKY-005): what uniform azimuth shift and horizontal
  scale a given translation would produce for skyline structure at a given range — the quantity a
  simulator's per-column skyline range lets us check against the observed curve disagreement.

Imports: standard library, numpy, ``hsreloc.observation`` and the grid rule already gated by
``DEC-SKY-006`` (``hsreloc.placeret.split.greedy_grid``). No extractor, matcher, or image library —
asserted by ``tests/test_simret_geometry.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from hsreloc.observation import Observation
from hsreloc.placeret.split import greedy_grid

GEOMETRY_VERSION = "1.0.0"

#: Declared bin edges (PROT-SKY-001 §3). Labels are ``"<lo>-<hi>"``; a value outside the last edge
#: has no bin. The translation bins bracket the brief's nominal 0 / ~5 / ~10 / ~25 / ~50 / ~100 m.
DEFAULT_TRANSLATION_BINS_M = [0.0, 1.0, 7.5, 17.5, 37.5, 75.0, 150.0]
#: |Δyaw| bins bracketing 0 / ±1 / ±2 / ±5 / ±10 degrees.
DEFAULT_YAW_BINS_DEG = [0.0, 0.5, 1.5, 3.5, 7.5, 15.0]
#: |Δpitch| / |Δroll| bins bracketing 0 / ±2 / ±5 / ±10 degrees.
DEFAULT_ATTITUDE_BINS_DEG = [0.0, 0.5, 3.5, 7.5, 15.0]

STAGES = ("stage0", "stage1", "stage2", "stage3")


class GeometryError(Exception):
    """A pose is missing a field the stage design needs, or a task set cannot be built as declared."""


# --------------------------------------------------------------------------------------------------
# angles and frames
# --------------------------------------------------------------------------------------------------

def wrap_deg(angle_deg: float) -> float:
    """Wrap to [-180, 180)."""
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def yaw_diff_deg(query_yaw_deg: float, reference_yaw_deg: float) -> float:
    """Signed shortest yaw difference query − reference, compass degrees, wrapped."""
    return wrap_deg(float(query_yaw_deg) - float(reference_yaw_deg))


def heading_vectors(yaw_deg: float) -> tuple:
    """(forward, right) unit vectors in (East, North) for a compass heading (DEC-004)."""
    psi = math.radians(float(yaw_deg))
    forward = np.array([math.sin(psi), math.cos(psi)])
    right = np.array([math.cos(psi), -math.sin(psi)])
    return forward, right


# --------------------------------------------------------------------------------------------------
# viewpoint offset between two poses
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ViewpointOffset:
    translation_m: float           # horizontal Euclidean distance
    along_m: float                 # query displacement along the reference heading (+ = ahead)
    lateral_m: float               # query displacement to the reference's right (+ = right)
    up_m: Optional[float]          # query − reference altitude, when both carry one
    yaw_diff_deg: float            # query − reference, wrapped
    pitch_diff_deg: Optional[float]
    roll_diff_deg: Optional[float]

    @property
    def abs_yaw_diff_deg(self) -> float:
        return abs(self.yaw_diff_deg)

    def as_dict(self) -> dict:
        return asdict(self)


def _require_pose(obs: Observation) -> None:
    if obs.pos_east_m is None or obs.pos_north_m is None:
        raise GeometryError(f"{obs.observation_id}: no ENU position — run the adapter's ENU conversion first")
    if obs.yaw_deg is None:
        raise GeometryError(f"{obs.observation_id}: no yaw — the stage design needs a heading for every "
                            f"observation (PROT-SKY-001 §1, REQUIRED for retrieval)")


def viewpoint_offset(query: Observation, reference: Observation) -> ViewpointOffset:
    """The query's pose relative to the reference's, in the reference camera's horizontal frame."""
    _require_pose(query)
    _require_pose(reference)
    d = np.array([query.pos_east_m - reference.pos_east_m, query.pos_north_m - reference.pos_north_m])
    forward, right = heading_vectors(reference.yaw_deg)
    up = None
    if query.up_m is not None and reference.up_m is not None:
        up = float(query.up_m - reference.up_m)
    pitch = None
    if query.pitch_deg is not None and reference.pitch_deg is not None:
        pitch = wrap_deg(query.pitch_deg - reference.pitch_deg)
    roll = None
    if query.roll_deg is not None and reference.roll_deg is not None:
        roll = wrap_deg(query.roll_deg - reference.roll_deg)
    return ViewpointOffset(
        translation_m=float(np.linalg.norm(d)),
        along_m=float(d @ forward),
        lateral_m=float(d @ right),
        up_m=up,
        yaw_diff_deg=yaw_diff_deg(query.yaw_deg, reference.yaw_deg),
        pitch_diff_deg=pitch,
        roll_diff_deg=roll,
    )


# --------------------------------------------------------------------------------------------------
# bins and stages
# --------------------------------------------------------------------------------------------------

def assign_bin(value: float, edges: list) -> Optional[str]:
    """Half-open bins ``[edges[i], edges[i+1])`` labelled ``"<lo>-<hi>"``; ``None`` outside."""
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise GeometryError(f"bin edges must be strictly increasing with at least two values, got {edges}")
    v = float(value)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= v < hi:
            return f"{lo:g}-{hi:g}"
    return None


@dataclass(frozen=True)
class StageRule:
    """Declared tolerances that decide which stage a query–reference pair belongs to.

    ``stage0``: same pose (translation ≤ ``same_pose_m``, |Δyaw| ≤ ``yaw_eps_deg``, attitude within
    ``attitude_eps_deg``) — appearance/condition change only. ``stage1``: translation with the yaw
    and attitude held. ``stage2``: yaw varies, attitude held. ``stage3``: pitch or roll vary.
    ``require_attitude`` refuses a pose without pitch/roll rather than guessing a stage.
    """

    same_pose_m: float = 0.5
    yaw_eps_deg: float = 0.5
    attitude_eps_deg: float = 0.5
    require_attitude: bool = True

    def stage_for(self, offset: ViewpointOffset) -> str:
        if offset.pitch_diff_deg is None or offset.roll_diff_deg is None:
            if self.require_attitude:
                raise GeometryError("pitch/roll missing on a pose — the stage design holds attitude fixed "
                                    "until Stage 3 and cannot tell a stage-1 pair from a stage-3 pair "
                                    "without them (set require_attitude=False to label 'attitude-unknown')")
            return "attitude-unknown"
        attitude_held = (abs(offset.pitch_diff_deg) <= self.attitude_eps_deg
                         and abs(offset.roll_diff_deg) <= self.attitude_eps_deg)
        if not attitude_held:
            return "stage3"
        if offset.abs_yaw_diff_deg > self.yaw_eps_deg:
            return "stage2"
        # A vertical-only sweep is a translation, not a same-pose pair: the 2026-09-02 batch
        # contains two UP runs whose horizontal displacement is ~0, and labelling them stage0
        # would poison every same-pose slice (research R5 of the viewpoint feature).
        moved_vertically = offset.up_m is not None and abs(offset.up_m) > self.same_pose_m
        if offset.translation_m > self.same_pose_m or moved_vertically:
            return "stage1"
        return "stage0"

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------------------
# far-field prediction (LIT-SKY-005, first order)
# --------------------------------------------------------------------------------------------------

def far_field_prediction(offset: ViewpointOffset, range_m: float) -> dict:
    """Predicted uniform profile change for skyline structure at horizontal range ``range_m`` straight
    ahead of the reference camera: lateral motion → azimuth shift ≈ −lateral / range; along-track
    motion → horizontal scale ≈ 1 + along / range (objects grow as the camera approaches). First
    order in ``t / range``; the point of computing it is to compare it with what the matcher sees."""
    if range_m <= 0:
        raise GeometryError(f"range must be positive, got {range_m}")
    return {
        "range_m": float(range_m),
        "shift_deg": math.degrees(-offset.lateral_m / range_m),
        "scale": 1.0 + offset.along_m / range_m,
        "t_over_range": offset.translation_m / range_m,
    }


# --------------------------------------------------------------------------------------------------
# reference grids, tasks, correctness
# --------------------------------------------------------------------------------------------------

QUERY_COLUMNS = [
    "spacing_m", "observation_id", "session_id", "frame_index", "east_m", "north_m", "up_m",
    "yaw_deg", "pitch_deg", "roll_deg", "nearest_reference_id", "in_coverage",
    "condition_translation_m", "condition_along_m", "condition_lateral_m", "condition_up_diff_m",
    "condition_yaw_diff_deg", "condition_abs_yaw_diff_deg", "condition_pitch_diff_deg",
    "condition_roll_diff_deg", "condition_translation_bin", "condition_yaw_bin",
    "condition_pitch_bin", "condition_roll_bin", "condition_stage", "condition_n_references",
]


def _tau(spacing: float) -> dict:
    return {"tau_pos_m": spacing / 2.0, "tau_near_m": float(spacing)}


def _fmt(v) -> str:
    return "" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))


def build_tasks(sessions: dict, config: dict) -> dict:
    """Pose-only reference grids and query rows.

    ``sessions`` maps ``session_id -> [Observation, ...]``. ``config`` declares the references in
    exactly one of two ways:

    * ``reference_session`` — the session whose observations, in ``frame_index`` order, are greedily
      subsampled into the grid (keep a frame when ≥ spacing from every kept frame); or
    * ``reference_observation_ids`` — an explicit list of observation ids (any sessions), used as
      the grid directly with no subsampling. This is the manually-collected-anchor case: the six
      declared references of the 2026-09-02 pilot span six single-observation sessions, which the
      single-session grid rule cannot express. The min-pairwise-distance ≥ spacing assertion is
      kept, so τ_pos = spacing/2 still guarantees a unique positive class.

    Also:

    * ``query_sessions`` — sessions whose every observation becomes a query (a session contributing
      a reference may be listed: its non-grid frames are then queries);
    * ``spacings_m`` — one grid, one τ pair and one task table per spacing;
    * optional ``translation_bins_m`` / ``yaw_bins_deg`` / ``attitude_bins_deg`` / ``stage_rule``.

    Correct = nearest grid reference; ``in_coverage`` (the spec-004 flag) iff within τ_pos. Every
    pairwise grid distance ≥ spacing is asserted, so the positive class is unique.
    """
    ref_session = config.get("reference_session")
    ref_ids_cfg = config.get("reference_observation_ids")
    if (ref_session is None) == (ref_ids_cfg is None):
        raise GeometryError("declare exactly one of reference_session or reference_observation_ids")
    query_sessions = list(config.get("query_sessions", []))
    missing = [s for s in query_sessions if s not in sessions]
    if missing:
        raise GeometryError(f"query sessions not found: {missing}")
    if not query_sessions:
        raise GeometryError("no query sessions declared")
    spacings = [float(s) for s in config["spacings_m"]]
    if not spacings or any(s <= 0 for s in spacings):
        raise GeometryError(f"spacings_m must be positive, got {spacings}")
    t_bins = list(config.get("translation_bins_m", DEFAULT_TRANSLATION_BINS_M))
    y_bins = list(config.get("yaw_bins_deg", DEFAULT_YAW_BINS_DEG))
    a_bins = list(config.get("attitude_bins_deg", DEFAULT_ATTITUDE_BINS_DEG))
    rule = StageRule(**config.get("stage_rule", {}))

    if ref_session is not None:
        if ref_session not in sessions:
            raise GeometryError(f"reference session {ref_session!r} not among {sorted(sessions)}")
        pool = sorted(sessions[ref_session], key=lambda o: o.frame_index)
    else:
        by_id = {o.observation_id: o for s in sessions.values() for o in s}
        absent = [i for i in ref_ids_cfg if i not in by_id]
        if absent:
            raise GeometryError(f"reference observation(s) not found in any session: {absent}")
        if len(set(ref_ids_cfg)) != len(ref_ids_cfg):
            raise GeometryError("reference_observation_ids contains duplicates")
        pool = [by_id[i] for i in ref_ids_cfg]
    for o in pool:
        _require_pose(o)
    xy = np.array([[o.pos_east_m, o.pos_north_m] for o in pool], dtype=np.float64)

    grids: dict = {}
    queries: list = []
    for spacing in spacings:
        kept = list(range(len(pool))) if ref_ids_cfg is not None else greedy_grid(xy, spacing)
        refs = [pool[i] for i in kept]
        P = xy[kept]
        if len(refs) > 1:
            d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            if float(d.min()) < spacing - 1e-9:
                raise GeometryError(f"spacing {spacing}: two references {d.min():.3f} m apart — the "
                                    f"positive class would not be unique")
        ref_ids = {o.observation_id for o in refs}
        entries = [{"reference_id": o.observation_id, "session_id": o.session_id,
                    "frame_index": o.frame_index, "east_m": o.pos_east_m, "north_m": o.pos_north_m,
                    "up_m": o.up_m, "yaw_deg": o.yaw_deg, "pitch_deg": o.pitch_deg, "roll_deg": o.roll_deg}
                   for o in refs]
        realised = [float(np.linalg.norm(P[i + 1] - P[i])) for i in range(len(refs) - 1)]
        key = f"{spacing:g}"
        grids[key] = {"spacing_m": spacing, **_tau(spacing), "n_references": len(refs),
                      "references": entries,
                      "realised_spacing_m": {"median": float(np.median(realised)) if realised else None,
                                             "min": min(realised) if realised else None,
                                             "max": max(realised) if realised else None},
                      "digest": hashlib.sha256("\n".join(e["reference_id"] for e in entries).encode()).hexdigest()}
        tau = _tau(spacing)
        for qs in query_sessions:
            for q in sorted(sessions[qs], key=lambda o: o.frame_index):
                if q.observation_id in ref_ids:
                    continue                      # a grid frame is never its own query
                _require_pose(q)
                dist = np.linalg.norm(P - np.array([q.pos_east_m, q.pos_north_m]), axis=1)
                j = int(np.argmin(dist))
                off = viewpoint_offset(q, refs[j])
                stage = rule.stage_for(off)
                queries.append({
                    "spacing_m": spacing, "observation_id": q.observation_id, "session_id": q.session_id,
                    "frame_index": q.frame_index, "east_m": q.pos_east_m, "north_m": q.pos_north_m,
                    "up_m": q.up_m, "yaw_deg": q.yaw_deg, "pitch_deg": q.pitch_deg, "roll_deg": q.roll_deg,
                    "nearest_reference_id": refs[j].observation_id,
                    "in_coverage": bool(off.translation_m <= tau["tau_pos_m"]),
                    "condition_translation_m": off.translation_m,
                    "condition_along_m": off.along_m,
                    "condition_lateral_m": off.lateral_m,
                    "condition_up_diff_m": off.up_m,
                    "condition_yaw_diff_deg": off.yaw_diff_deg,
                    "condition_abs_yaw_diff_deg": off.abs_yaw_diff_deg,
                    "condition_pitch_diff_deg": off.pitch_diff_deg,
                    "condition_roll_diff_deg": off.roll_diff_deg,
                    "condition_translation_bin": assign_bin(off.translation_m, t_bins),
                    "condition_yaw_bin": assign_bin(off.abs_yaw_diff_deg, y_bins),
                    "condition_pitch_bin": (None if off.pitch_diff_deg is None
                                            else assign_bin(abs(off.pitch_diff_deg), a_bins)),
                    "condition_roll_bin": (None if off.roll_diff_deg is None
                                           else assign_bin(abs(off.roll_diff_deg), a_bins)),
                    "condition_stage": stage,
                    "condition_n_references": len(refs),
                })

    manifest = {
        "geometry_version": GEOMETRY_VERSION,
        "reference_session": ref_session,
        "reference_observation_ids": list(ref_ids_cfg) if ref_ids_cfg is not None else None,
        "query_sessions": query_sessions,
        "spacings_m": spacings,
        "tolerances": {f"{s:g}": _tau(s) for s in spacings},
        "translation_bins_m": t_bins, "yaw_bins_deg": y_bins, "attitude_bins_deg": a_bins,
        "stage_rule": rule.as_dict(),
        "rule": ("greedy along-path grid over the reference session in frame order (ENU straight-line "
                 "distance, keep when >= spacing from every kept frame); tau_pos = spacing/2, tau_near = "
                 "spacing; correct reference = nearest grid reference; in_coverage iff translation <= tau_pos; "
                 "along/lateral in the reference heading frame (DEC-004 compass); stage by StageRule"),
        "grid_counts": {k: g["n_references"] for k, g in grids.items()},
        "n_query_rows": len(queries),
        "stage_counts": _counts(queries, "condition_stage"),
        "reads": "poses only — the observation record's position and yaw/pitch/roll; no image, curve or matcher output",
    }
    return {"grids": grids, "queries": queries, "manifest": manifest}


def _counts(rows: list, field: str) -> dict:
    out: dict = {}
    for r in rows:
        k = f"{r['spacing_m']:g}|{r[field]}"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def write_tasks(tasks: dict, out_dir: Path) -> dict:
    """``grids.json`` + ``queries.csv`` + ``manifest.json`` (with a content digest over the first two)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grids.json").write_text(json.dumps(tasks["grids"], indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    with (out_dir / "queries.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUERY_COLUMNS)
        w.writeheader()
        for q in tasks["queries"]:
            w.writerow({k: _fmt(q.get(k)) for k in QUERY_COLUMNS})
    h = hashlib.sha256()
    for name in ("grids.json", "queries.csv"):
        h.update((out_dir / name).read_bytes())
    manifest = dict(tasks["manifest"], content_digest=h.hexdigest())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return manifest


_FLOAT_COLUMNS = ("spacing_m", "east_m", "north_m", "up_m", "yaw_deg", "pitch_deg", "roll_deg",
                  "condition_translation_m", "condition_along_m", "condition_lateral_m", "condition_up_diff_m",
                  "condition_yaw_diff_deg", "condition_abs_yaw_diff_deg", "condition_pitch_diff_deg",
                  "condition_roll_diff_deg")


def load_tasks(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("geometry_version") != GEOMETRY_VERSION:
        raise GeometryError(f"task set version {manifest.get('geometry_version')!r} != {GEOMETRY_VERSION}")
    grids = json.loads((out_dir / "grids.json").read_text(encoding="utf-8"))
    queries = []
    with (out_dir / "queries.csv").open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for k in _FLOAT_COLUMNS:
                r[k] = None if r[k] == "" else float(r[k])
            r["frame_index"] = int(r["frame_index"])
            r["condition_n_references"] = int(r["condition_n_references"])
            r["in_coverage"] = r["in_coverage"] == "True"
            for k in ("condition_translation_bin", "condition_yaw_bin", "condition_pitch_bin", "condition_roll_bin"):
                r[k] = None if r[k] == "" else r[k]
            queries.append(r)
    return {"grids": grids, "queries": queries, "manifest": manifest}
