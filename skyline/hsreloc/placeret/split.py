"""Pose-only reference grids, tasks and attainability (research R1/R9; contracts/ecl-task-set.md).

Reads the tracked ECL index's ``groups.csv`` (poses, sequence, frame, eligibility, condition
availability, DEV membership) with the standard library and numpy **only**. It never imports an
extractor, the silver conversion, the matcher, or an image library, and never opens a file outside
the index directory — asserted by ``tests/test_placeret_split.py``. That is what makes the design
independent of the thing the experiment measures (spec FR-001; DEC-SKY-006 R6).

Rules (DEC-SKY-006 R1, R2):

* reference grid = greedy along-path subsampling over the **eligible Originals** of the reference
  traversal: walking the sequence in frame order, keep a frame when it is at least ``spacing`` from
  every frame already kept (straight-line distance in the scene's synthetic ENU frame);
* τ_pos = spacing / 2, τ_near = spacing; the grid's pairwise distance ≥ spacing guarantees at most
  one reference within τ_pos of any query, which is asserted;
* families: ``control`` (Summer/Winter/Evening of the grid frames), ``cross-orig`` / ``cross-gen``
  (Original / generated conditions of the scene's *other* sequences), ``same-orig`` / ``same-gen``
  (the reference traversal's non-grid eligible frames); DEV population: ``identity`` (the grid
  frames' own Originals — the known-answer plumbing case) and ``control`` over a grid drawn on the
  DEV eligible Originals of each scene.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from hsreloc.placeret.frame import SYNTHETIC_ORIGIN, UNITS, fit_scene_frame, write_frames

SPLIT_VERSION = "1.0.0"
CONDITIONS = ("original", "summer", "winter", "evening")
GENERATED = ("summer", "winter", "evening")
FAMILIES_ALL = ("control", "cross-orig", "cross-gen", "same-orig", "same-gen")
FAMILIES_DEV = ("identity", "control")
TIER_BY_FAMILY = {"control": "T2", "cross-orig": "T3", "cross-gen": "T2",
                  "same-orig": "T3", "same-gen": "T2", "identity": "T3"}
TASK_BY_FAMILY = {"control": "control", "cross-orig": "cross", "cross-gen": "cross",
                  "same-orig": "same", "same-gen": "same", "identity": "identity"}
CONDITIONS_BY_FAMILY = {"control": GENERATED, "cross-orig": ("original",), "cross-gen": GENERATED,
                        "same-orig": ("original",), "same-gen": GENERATED, "identity": ("original",)}
QUERY_COLUMNS = ["family", "spacing_m", "observation_id", "group_id", "scene", "sequence", "condition",
                 "task", "tier", "east_m", "north_m", "up_m", "nearest_reference_id", "nearest_ref_m",
                 "rotation_deg", "attainable", "dev_subset", "n_scene_references", "image_relpath"]


class SplitError(Exception):
    """The split cannot be built as the contract requires."""


def observation_id(group_id: str, condition: str) -> str:
    return f"{group_id}__{condition}"


def parse_observation_id(obs_id: str) -> tuple:
    if "__" not in obs_id:
        raise SplitError(f"observation id {obs_id!r} carries no condition suffix")
    gid, cond = obs_id.rsplit("__", 1)
    if cond not in CONDITIONS:
        raise SplitError(f"observation id {obs_id!r}: unknown condition {cond!r}")
    return gid, cond


def rel_angle_deg(qa, qb) -> float:
    """Geodesic angle between two orientations (w, x, y, z) — convention-invariant."""
    qa = np.asarray(qa, dtype=np.float64); qb = np.asarray(qb, dtype=np.float64)
    qa = qa / np.linalg.norm(qa); qb = qb / np.linalg.norm(qb)
    return float(np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.dot(qa, qb)))))))


def greedy_grid(points: np.ndarray, spacing: float) -> list:
    """Indices kept by the greedy along-path rule over planar (N, 2) points in sequence order."""
    if points.shape[0] == 0:
        return []
    kept = [0]
    for i in range(1, points.shape[0]):
        if np.min(np.linalg.norm(points[kept] - points[i], axis=1)) >= spacing:
            kept.append(i)
    return kept


def read_groups(index_dir: Path) -> tuple:
    """(manifest, rows sorted by (scene, seq, frame number)) from the index's groups.csv."""
    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    with (index_dir / "groups.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_frame_no"] = int(r["frame"].replace("frame", ""))
        r["_eligible"] = r["eligible"] == "True"
        r["_dev"] = r["dev_subset"] == "True"
        r["_pos"] = np.array([float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])])
        r["_quat"] = np.array([float(r["q1"]), float(r["q2"]), float(r["q3"]), float(r["q4"])])
        r["_available"] = tuple(c for c in CONDITIONS
                                if c == "original" or r.get(f"{c}_relpath"))
    rows.sort(key=lambda r: (r["scene"], r["seq"], r["_frame_no"]))
    return manifest, rows


def _tau(spacing: float) -> dict:
    return {"tau_pos_m": spacing / 2.0, "tau_near_m": float(spacing)}


def build_split(index_dir: Path, config: dict, out_dir: Path) -> dict:
    """Write frames.json, grids.json, queries.csv, manifest.json; return the manifest."""
    index_dir, out_dir = Path(index_dir), Path(out_dir)
    population = config.get("population", "all")
    if population not in ("all", "dev"):
        raise SplitError(f"population must be 'all' or 'dev', got {population!r}")
    spacings = [float(s) for s in config["spacings_m"]]
    if any(s >= 100.0 for s in spacings):
        raise SplitError("a 100 m grid is degenerate on this geometry (DEC-SKY-006 R2) — refused")
    roles = config["roles"]
    excluded = set(roles.get("excluded_scenes", []))
    ref_trav = roles["reference_traversals"]            # scene -> seq
    query_trav = roles.get("query_traversals", {})      # scene -> [seq, ...]

    manifest_idx, rows = read_groups(index_dir)
    if config.get("expected_index_revision") and manifest_idx.get("dataset_revision") != config["expected_index_revision"]:
        raise SplitError(f"index revision {manifest_idx.get('dataset_revision')!r} != expected "
                         f"{config['expected_index_revision']!r}")

    # --- scene frames from ALL test poses of the scene (pose only) ---
    by_scene: dict = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)
    offsets = dict(config.get("scene_offsets_m", {}))       # explicit overrides / test scenes
    frames = {s: fit_scene_frame(s, np.array([r["_pos"] for r in rs]),
                                 offset_m=offsets.get(s)) for s, rs in by_scene.items()}
    for r in rows:
        r["_enu"] = frames[r["scene"]].to_enu(r["_pos"])

    grids: dict = {}
    query_rows: list = []
    for spacing in spacings:
        tau = _tau(spacing)
        grid_entries: list = []
        scene_ref_count: dict = {}
        ref_rows_by_scene: dict = {}
        for scene, rs in sorted(by_scene.items()):
            if scene in excluded:
                continue
            if population == "dev":
                pool = [r for r in rs if r["_dev"] and r["_eligible"]]
            else:
                seq = ref_trav.get(scene)
                if seq is None:
                    continue
                pool = [r for r in rs if r["seq"] == seq and r["_eligible"]]
            if not pool:
                continue
            kept = greedy_grid(np.array([r["_enu"][:2] for r in pool]), spacing)
            refs = [pool[i] for i in kept]
            ref_rows_by_scene[scene] = refs
            scene_ref_count[scene] = len(refs)
            for r in refs:
                grid_entries.append({"reference_id": observation_id(r["group_id"], "original"),
                                     "group_id": r["group_id"], "scene": scene, "sequence": r["seq"],
                                     "east_m": float(r["_enu"][0]), "north_m": float(r["_enu"][1]),
                                     "up_m": float(r["_enu"][2]), "image_relpath": r["original_relpath"]})
        if not grid_entries:
            raise SplitError(f"spacing {spacing}: empty grid")
        # uniqueness of the positive class: every pair of references >= spacing apart
        P = np.array([[g["east_m"], g["north_m"]] for g in grid_entries])
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        if float(d.min()) < spacing - 1e-9:
            raise SplitError(f"spacing {spacing}: two references {d.min():.2f} m apart (< spacing) — "
                             f"the positive class would not be unique")
        realised = []
        for scene, refs in ref_rows_by_scene.items():
            for a, b in zip(refs[:-1], refs[1:]):
                realised.append(float(np.linalg.norm(a["_enu"][:2] - b["_enu"][:2])))
        digest = hashlib.sha256("\n".join(g["reference_id"] for g in grid_entries).encode()).hexdigest()
        grids[str(spacing)] = {"spacing_m": spacing, **tau, "references": grid_entries,
                               "n_references": len(grid_entries),
                               "per_scene": scene_ref_count, "digest": digest,
                               "realised_spacing_m": {"median": float(np.median(realised)) if realised else None,
                                                      "min": float(min(realised)) if realised else None,
                                                      "max": float(max(realised)) if realised else None}}
        ref_ids = {g["group_id"] for g in grid_entries}
        ref_by_scene_arrays = {s: (np.array([[g["east_m"], g["north_m"]] for g in grid_entries if g["scene"] == s]),
                                   [g for g in grid_entries if g["scene"] == s]) for s in scene_ref_count}
        all_ref_xy = P
        group_row = {r["group_id"]: r for r in rows}

        def nearest(r):
            xy = r["_enu"][:2]
            dist = np.linalg.norm(all_ref_xy - xy, axis=1)
            j = int(np.argmin(dist))
            g = grid_entries[j]
            rot = rel_angle_deg(r["_quat"], group_row[g["group_id"]]["_quat"])
            return g["reference_id"], float(dist[j]), rot

        families = FAMILIES_DEV if population == "dev" else FAMILIES_ALL
        for family in families:
            conds = CONDITIONS_BY_FAMILY[family]
            task = TASK_BY_FAMILY[family]
            for scene, rs in sorted(by_scene.items()):
                if scene in excluded or scene not in scene_ref_count:
                    continue
                if family in ("control", "identity"):
                    cand = [r for r in rs if r["group_id"] in ref_ids]
                elif family.startswith("cross"):
                    cand = [r for r in rs if r["seq"] in query_trav.get(scene, []) and r["_eligible"]]
                else:  # same-traversal: the reference sequence's other eligible frames
                    cand = [r for r in rs if r["seq"] == ref_trav.get(scene) and r["_eligible"]
                            and r["group_id"] not in ref_ids]
                for r in cand:
                    nid, ndist, rot = nearest(r)
                    for cond in conds:
                        if cond not in r["_available"]:
                            continue
                        query_rows.append({
                            "family": family, "spacing_m": spacing,
                            "observation_id": observation_id(r["group_id"], cond),
                            "group_id": r["group_id"], "scene": scene, "sequence": r["seq"],
                            "condition": cond, "task": task, "tier": TIER_BY_FAMILY[family],
                            "east_m": float(r["_enu"][0]), "north_m": float(r["_enu"][1]),
                            "up_m": float(r["_enu"][2]),
                            "nearest_reference_id": nid, "nearest_ref_m": round(ndist, 4),
                            "rotation_deg": round(rot, 3),
                            "attainable": bool(ndist <= tau["tau_pos_m"]),
                            "dev_subset": bool(r["_dev"]),
                            "n_scene_references": scene_ref_count[scene],
                            "image_relpath": r["original_relpath"] if cond == "original" else r[f"{cond}_relpath"],
                        })

    # --- write ---
    out_dir.mkdir(parents=True, exist_ok=True)
    write_frames(frames, out_dir / "frames.json")
    (out_dir / "grids.json").write_text(json.dumps(grids, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "queries.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUERY_COLUMNS)
        w.writeheader()
        for q in query_rows:
            w.writerow(q)
    h = hashlib.sha256()
    for name in ("frames.json", "grids.json", "queries.csv"):
        h.update((out_dir / name).read_bytes())
    counts = defaultdict(int)
    for q in query_rows:
        counts[f"{q['family']}|{q['spacing_m']:g}"] += 1
    manifest = {
        "split_version": SPLIT_VERSION,
        "population": population,
        "index": {"dir": str(index_dir.as_posix()), "dataset_revision": manifest_idx.get("dataset_revision"),
                  "content_digest": manifest_idx.get("content_digest")},
        "spacings_m": spacings,
        "tolerances": {str(s): _tau(s) for s in spacings},
        "roles": roles,
        "geodetic_origin": SYNTHETIC_ORIGIN,
        "units": UNITS,
        "rule": ("greedy along-path grid on eligible Originals of the reference traversal (planar "
                 "synthetic ENU, straight-line distance, keep when >= spacing from every kept frame); "
                 "tau_pos = spacing/2, tau_near = spacing; correct reference = nearest grid reference; "
                 "attainable iff nearest_ref_m <= tau_pos"),
        "families": list(FAMILIES_DEV if population == "dev" else FAMILIES_ALL),
        "grid_counts": {s: g["per_scene"] for s, g in grids.items()},
        "query_counts": dict(sorted(counts.items())),
        "n_query_rows": len(query_rows),
        "content_digest": h.hexdigest(),
        "reads": "poses only — groups.csv and manifest.json of the index; no image, curve, or matcher output",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_split(split_dir: Path) -> dict:
    """{manifest, grids, queries (list of dicts with typed fields), frames}"""
    from hsreloc.placeret.frame import read_frames
    split_dir = Path(split_dir)
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    grids = json.loads((split_dir / "grids.json").read_text(encoding="utf-8"))
    queries = []
    with (split_dir / "queries.csv").open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["spacing_m"] = float(r["spacing_m"])
            for k in ("east_m", "north_m", "up_m", "nearest_ref_m", "rotation_deg"):
                r[k] = float(r[k])
            r["attainable"] = r["attainable"] == "True"
            r["dev_subset"] = r["dev_subset"] == "True"
            r["n_scene_references"] = int(r["n_scene_references"])
            queries.append(r)
    return {"manifest": manifest, "grids": grids, "queries": queries,
            "frames": read_frames(split_dir / "frames.json")}
