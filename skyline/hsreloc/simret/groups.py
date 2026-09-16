"""Experimental group membership for manually collected simulator runs (Part D).

The dataset this has to serve is not a flight. It is a human choosing camera positions: a handful of
**anchors**, some deliberate perturbations around each, and the same poses re-rendered under other
conditions. Nothing in the raw export knows which frames belong together that way, and nothing should
try to guess it.

The division of labour, which is the whole point of this module:

* **the human declares identity** — which anchor a frame belongs to, whether it is a reference or a
  query, and any grouping the geometry cannot see. Two authoring forms, both plain CSV a person can
  write and review in a spreadsheet;
* **the code derives geometry** — displacement from the anchor, its lateral and longitudinal
  components relative to North, altitude difference, attitude difference, the displacement bin, and
  the same-pose group. These are computed from poses through :mod:`hsreloc.simret.geometry`, never
  authored and never inferred from a filename.

Group membership is **never** taken from filename order or frame index. ``sky_000123.png`` sitting
between two frames of an anchor says nothing about it; only a declaration or a pose does.

Authoring form 1 — ``anchors.csv`` (usually enough)::

    anchor_id,east_m,north_m,up_m,heading_deg,radius_m,notes
    ridge_a,0,0,60,0,3,view down the valley
    ridge_a_e25,25,0,60,0,3,25 m East of ridge_a

Every observation is assigned to the nearest anchor within that anchor's ``radius_m``; an observation
matching none is assigned ``""`` and **counted in the manifest**, never quietly dropped.

Authoring form 2 — ``assignments.csv`` (explicit, wins over form 1)::

    observation_id,anchor_id,role,condition_group,notes
    Run_A__000017,ridge_a,reference,,hand-picked

Both are optional; with neither, every observation is emitted with derived geometry only and an empty
anchor, which is still a usable index for a Stage-0/E0 plumbing run.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from hsreloc.observation import Observation, read_session
from hsreloc.simret.geometry import (DEFAULT_TRANSLATION_BINS_M, GeometryError, assign_bin,
                                     heading_vectors, wrap_deg)

GROUPS_VERSION = "1.0.0"

ROLES = ("reference", "query", "both", "excluded")
DEFAULT_ROLE = "query"
DEFAULT_ANCHOR_RADIUS_M = 5.0
#: Two frames are the "same pose" when their positions agree to this and their attitudes to
#: ``POSE_GROUP_ANGLE_DECIMALS`` — the E2 same-pose cross-condition grouping.
POSE_GROUP_DECIMALS = 2
POSE_GROUP_ANGLE_DECIMALS = 2

ANCHOR_COLUMNS = ("anchor_id", "east_m", "north_m", "up_m", "heading_deg", "radius_m", "notes")
ASSIGNMENT_COLUMNS = ("observation_id", "anchor_id", "role", "condition_group", "notes")

INDEX_COLUMNS = [
    "observation_id", "session_id", "frame_index", "anchor_id", "role", "condition_group",
    "east_m", "north_m", "up_m", "yaw_deg", "pitch_deg", "roll_deg",
    "condition_level", "condition_time_of_day", "condition_hour", "condition_clouds",
    "condition_anchor_translation_m", "condition_anchor_along_m", "condition_anchor_lateral_m",
    "condition_anchor_up_m", "condition_anchor_yaw_diff_deg", "condition_anchor_pitch_diff_deg",
    "condition_anchor_roll_diff_deg", "condition_translation_bin", "condition_pose_group",
    "condition_gt_status", "notes",
]


class GroupError(Exception):
    """A sidecar cannot be read, or contradicts the sessions it describes."""


@dataclass(frozen=True)
class Anchor:
    """One declared physical location. ``heading_deg`` is the frame the along/lateral split uses."""

    anchor_id: str
    east_m: float
    north_m: float
    up_m: Optional[float] = None
    heading_deg: float = 0.0
    radius_m: float = DEFAULT_ANCHOR_RADIUS_M
    notes: str = ""

    def as_dict(self) -> dict:
        return {"anchor_id": self.anchor_id, "east_m": self.east_m, "north_m": self.north_m,
                "up_m": self.up_m, "heading_deg": self.heading_deg, "radius_m": self.radius_m,
                "notes": self.notes}


def _opt_float(value) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def read_anchors(path: Path | str) -> list:
    """Read ``anchors.csv``. Refuses duplicates and non-numeric coordinates by name."""
    path = Path(path)
    anchors, seen = [], set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("anchor_id", "east_m", "north_m") if c not in (reader.fieldnames or [])]
        if missing:
            raise GroupError(f"{path}: anchors.csv is missing required column(s) {missing}; "
                             f"expected a subset of {list(ANCHOR_COLUMNS)}")
        for i, row in enumerate(reader, start=2):
            aid = (row.get("anchor_id") or "").strip()
            if not aid:
                raise GroupError(f"{path}:{i}: blank anchor_id")
            if aid in seen:
                raise GroupError(f"{path}:{i}: duplicate anchor_id {aid!r} — an anchor is a physical "
                                 f"place and must be declared once")
            seen.add(aid)
            try:
                anchors.append(Anchor(
                    anchor_id=aid, east_m=float(row["east_m"]), north_m=float(row["north_m"]),
                    up_m=_opt_float(row.get("up_m")),
                    heading_deg=float(row.get("heading_deg") or 0.0),
                    radius_m=float(row.get("radius_m") or DEFAULT_ANCHOR_RADIUS_M),
                    notes=(row.get("notes") or "").strip()))
            except ValueError as exc:
                raise GroupError(f"{path}:{i} (anchor {aid!r}): {exc}") from exc
    if not anchors:
        raise GroupError(f"{path}: no anchor rows")
    return anchors


def read_assignments(path: Path | str) -> dict:
    """Read ``assignments.csv`` into ``observation_id -> {anchor_id, role, condition_group, notes}``."""
    path = Path(path)
    out: dict = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "observation_id" not in (reader.fieldnames or []):
            raise GroupError(f"{path}: assignments.csv needs an observation_id column; "
                             f"expected a subset of {list(ASSIGNMENT_COLUMNS)}")
        for i, row in enumerate(reader, start=2):
            oid = (row.get("observation_id") or "").strip()
            if not oid:
                raise GroupError(f"{path}:{i}: blank observation_id")
            if oid in out:
                raise GroupError(f"{path}:{i}: observation {oid!r} assigned twice")
            role = (row.get("role") or DEFAULT_ROLE).strip() or DEFAULT_ROLE
            if role not in ROLES:
                raise GroupError(f"{path}:{i}: role {role!r} is not one of {ROLES}")
            out[oid] = {"anchor_id": (row.get("anchor_id") or "").strip(), "role": role,
                        "condition_group": (row.get("condition_group") or "").strip(),
                        "notes": (row.get("notes") or "").strip()}
    return out


def assign_anchor(observation: Observation, anchors: list) -> tuple:
    """Nearest declared anchor within its own radius, or ``(None, None)``."""
    if observation.pos_east_m is None or observation.pos_north_m is None:
        raise GeometryError(f"{observation.observation_id}: no ENU position")
    best, best_d = None, None
    for a in anchors:
        d = float(np.hypot(observation.pos_east_m - a.east_m, observation.pos_north_m - a.north_m))
        if d <= a.radius_m and (best_d is None or d < best_d):
            best, best_d = a, d
    return best, best_d


def anchor_offset(observation: Observation, anchor: Anchor) -> dict:
    """Displacement from an anchor, split along/lateral in the anchor's heading frame (DEC-004).

    With the anchor heading at its default 0° (North) — which is what a world-locked-North camera
    means — ``along`` is the North/South component and ``lateral`` the East/West one, exactly the
    E3 split.
    """
    d = np.array([observation.pos_east_m - anchor.east_m, observation.pos_north_m - anchor.north_m])
    forward, right = heading_vectors(anchor.heading_deg)
    up = None
    if observation.up_m is not None and anchor.up_m is not None:
        up = float(observation.up_m - anchor.up_m)
    return {"translation_m": float(np.linalg.norm(d)), "along_m": float(d @ forward),
            "lateral_m": float(d @ right), "up_m": up,
            "yaw_diff_deg": (None if observation.yaw_deg is None
                             else wrap_deg(observation.yaw_deg - anchor.heading_deg))}


def pose_group_key(observation: Observation) -> str:
    """A stable key over the pose alone — frames sharing it are the *same camera pose*.

    This is what makes the E2 same-pose cross-condition comparison objective: two frames are grouped
    because their poses agree to the declared precision, not because their filenames or their order
    suggest they should be.
    """
    parts = []
    for v in (observation.pos_east_m, observation.pos_north_m, observation.up_m):
        parts.append("na" if v is None else f"{round(float(v), POSE_GROUP_DECIMALS):.{POSE_GROUP_DECIMALS}f}")
    for v in (observation.yaw_deg, observation.pitch_deg, observation.roll_deg):
        parts.append("na" if v is None else f"{round(float(v), POSE_GROUP_ANGLE_DECIMALS):.{POSE_GROUP_ANGLE_DECIMALS}f}")
    return "pose_" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def build_index(store_root: Path | str, session_ids: Optional[list] = None,
                anchors_csv: Optional[Path] = None, assignments_csv: Optional[Path] = None,
                translation_bins_m: Optional[list] = None) -> dict:
    """Join ingested sessions with the sidecar and derive every geometric grouping field."""
    root = Path(store_root)
    if session_ids is None:
        session_ids = sorted(p.name for p in root.iterdir() if (p / "observations.csv").exists())
    if not session_ids:
        raise GroupError(f"no ingested sessions under {root}")
    anchors = read_anchors(anchors_csv) if anchors_csv else []
    assignments = read_assignments(assignments_csv) if assignments_csv else {}
    by_id = {a.anchor_id: a for a in anchors}
    unknown = sorted({v["anchor_id"] for v in assignments.values()
                      if v["anchor_id"] and v["anchor_id"] not in by_id})
    if unknown and anchors:
        raise GroupError(f"assignments.csv names anchor(s) {unknown} that anchors.csv does not declare")
    bins = list(translation_bins_m or DEFAULT_TRANSLATION_BINS_M)

    rows, seen_ids = [], set()
    for sid in session_ids:
        meta, observations = read_session(root / sid)
        cond = meta.get("condition", {}) or {}
        level = (meta.get("scene", {}) or {}).get("level_id")
        for obs in observations:
            if obs.observation_id in seen_ids:
                raise GroupError(f"observation_id {obs.observation_id!r} occurs in more than one session")
            seen_ids.add(obs.observation_id)
            declared = assignments.get(obs.observation_id, {})
            anchor, _ = (by_id.get(declared["anchor_id"]), None) if declared.get("anchor_id") \
                else assign_anchor(obs, anchors)
            off = anchor_offset(obs, anchor) if anchor else {}
            pitch_diff = roll_diff = None
            if anchor is not None:
                # An anchor declares position and heading only; a nominal level camera is the
                # reference attitude, so a non-zero pitch/roll IS the perturbation being measured.
                pitch_diff = None if obs.pitch_deg is None else wrap_deg(obs.pitch_deg)
                roll_diff = None if obs.roll_deg is None else wrap_deg(obs.roll_deg)
            rows.append({
                "observation_id": obs.observation_id, "session_id": sid, "frame_index": obs.frame_index,
                "anchor_id": anchor.anchor_id if anchor else "",
                "role": declared.get("role", DEFAULT_ROLE),
                "condition_group": declared.get("condition_group", ""),
                "east_m": obs.pos_east_m, "north_m": obs.pos_north_m, "up_m": obs.up_m,
                "yaw_deg": obs.yaw_deg, "pitch_deg": obs.pitch_deg, "roll_deg": obs.roll_deg,
                "condition_level": obs.extra.get("condition_level", level),
                "condition_time_of_day": obs.extra.get("condition_time_of_day", cond.get("time_of_day")),
                "condition_hour": obs.extra.get("condition_hour", cond.get("hour")),
                "condition_clouds": obs.extra.get("condition_clouds", cond.get("clouds")),
                "condition_anchor_translation_m": off.get("translation_m"),
                "condition_anchor_along_m": off.get("along_m"),
                "condition_anchor_lateral_m": off.get("lateral_m"),
                "condition_anchor_up_m": off.get("up_m"),
                "condition_anchor_yaw_diff_deg": off.get("yaw_diff_deg"),
                "condition_anchor_pitch_diff_deg": pitch_diff,
                "condition_anchor_roll_diff_deg": roll_diff,
                "condition_translation_bin": (assign_bin(off["translation_m"], bins)
                                              if off.get("translation_m") is not None else None),
                "condition_pose_group": pose_group_key(obs),
                "condition_gt_status": obs.extra.get("sim_gt_status", ""),
                "notes": declared.get("notes", ""),
            })

    unassigned = [r["observation_id"] for r in rows if not r["anchor_id"]]
    pose_groups: dict = {}
    for r in rows:
        pose_groups.setdefault(r["condition_pose_group"], []).append(r["observation_id"])
    manifest = {
        "groups_version": GROUPS_VERSION,
        "store_root": str(root), "sessions": list(session_ids),
        "n_observations": len(rows),
        "anchors": [a.as_dict() for a in anchors],
        "n_anchors": len(anchors),
        "sidecar": {"anchors_csv": str(anchors_csv) if anchors_csv else None,
                    "assignments_csv": str(assignments_csv) if assignments_csv else None},
        "assignment_rule": ("declared assignments win; otherwise the nearest declared anchor within that "
                            "anchor's own radius_m. Frame index and filename NEVER assign a group."),
        "translation_bins_m": bins,
        "n_unassigned": len(unassigned),
        "unassigned_examples": unassigned[:10],
        "role_counts": _counts(rows, "role"),
        "anchor_counts": _counts(rows, "anchor_id"),
        "condition_counts": _counts(rows, "condition_time_of_day"),
        "cloud_counts": _counts(rows, "condition_clouds"),
        "level_counts": _counts(rows, "condition_level"),
        "n_pose_groups": len(pose_groups),
        "multi_frame_pose_groups": {k: v for k, v in sorted(pose_groups.items()) if len(v) > 1},
        "derived_fields": [c for c in INDEX_COLUMNS if c.startswith("condition_")],
    }
    return {"rows": rows, "manifest": manifest}


def _counts(rows: list, key: str) -> dict:
    out: dict = {}
    for r in rows:
        k = "" if r.get(key) is None else str(r[key])
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def write_index(index: dict, out_dir: Path) -> dict:
    """``experiment_index.csv`` + ``groups_manifest.json`` with a content digest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "experiment_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        for r in index["rows"]:
            w.writerow({k: _fmt(r.get(k)) for k in INDEX_COLUMNS})
    digest = hashlib.sha256((out_dir / "experiment_index.csv").read_bytes()).hexdigest()
    manifest = dict(index["manifest"], content_digest=digest)
    (out_dir / "groups_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8")
    return manifest


def load_index(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "groups_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("groups_version") != GROUPS_VERSION:
        raise GroupError(f"index version {manifest.get('groups_version')!r} != {GROUPS_VERSION}")
    rows = []
    float_cols = {c for c in INDEX_COLUMNS if c.endswith(("_m", "_deg")) or c in ("condition_hour",)}
    with (out_dir / "experiment_index.csv").open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for c in float_cols:
                r[c] = None if r.get(c, "") == "" else float(r[c])
            r["frame_index"] = int(r["frame_index"])
            rows.append(r)
    return {"rows": rows, "manifest": manifest}


def select(index: dict, **criteria) -> list:
    """Rows matching every ``field=value`` (or ``field=[values]``) criterion. The grouping query."""
    out = []
    for r in index["rows"]:
        keep = True
        for key, want in criteria.items():
            got = r.get(key)
            if isinstance(want, (list, tuple, set)):
                keep = keep and got in want
            else:
                keep = keep and got == want
            if not keep:
                break
        if keep:
            out.append(r)
    return out
