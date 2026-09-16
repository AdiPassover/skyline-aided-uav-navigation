"""Synthetic fixtures for the place-retrieval bench tests (EXP-SKY-006).

A mini ECL-style index (poses, sequences, eligibility, DEV flags), a mini observation store with
``skylines_auto`` curves per condition, tiny images, and synthetic SegFormer-style label maps.
Everything is constructed; nothing here is evidence about anything real.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

W, H = 64, 36                                   # tiny images: fast tests, same conventions
SCENES = ("SceneA", "SceneB")


def _quat_yaw(deg: float) -> np.ndarray:
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


def make_index(root: Path, n_per_seq: int = 12, step_m: float = 2.5) -> Path:
    """Two scenes; SceneA has seq1 (reference) + seq2 (parallel walk 1.5 m aside); SceneB has
    seq1 only. Poses are planar in a tilted plane so the PCA frame does real work. Frames 0 and
    n-1 of SceneA/seq2 are ineligible; one SceneB frame lacks Evening. DEV = every 4th frame."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(0.4), -np.sin(0.4)], [0.0, np.sin(0.4), np.cos(0.4)]])
    rows = []
    for scene, seqs, base in (("SceneA", ("seq1", "seq2"), (0.0, 0.0)), ("SceneB", ("seq1",), (500.0, 300.0))):
        for si, seq in enumerate(seqs):
            for i in range(n_per_seq):
                local = np.array([base[0] + i * step_m, base[1] + 1.5 * si, 0.0])
                pos = tilt @ local
                q = _quat_yaw(5.0 * i + 90.0 * si)
                gid = f"{scene}__{seq}__frame{i + 1:05d}"
                eligible = not (scene == "SceneA" and seq == "seq2" and i in (0, n_per_seq - 1))
                has_evening = not (scene == "SceneB" and i == 3)
                rows.append({
                    "group_id": gid, "scene": scene, "seq": seq, "frame": f"frame{i + 1:05d}",
                    "original_relpath": f"raw/{scene}/{seq}/frame{i + 1:05d}.png",
                    "summer_relpath": f"raw/{scene}/{seq}/frame{i + 1:05d}_summer.png",
                    "winter_relpath": f"raw/{scene}/{seq}/frame{i + 1:05d}_winter.png",
                    "evening_relpath": (f"raw/{scene}/{seq}/frame{i + 1:05d}_evening.png" if has_evening else ""),
                    "n_conditions": 4 if has_evening else 3, "width_px": W, "height_px": H,
                    "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
                    "q1": q[0], "q2": q[1], "q3": q[2], "q4": q[3],
                    "split": "test", "eligible": str(eligible),
                    "eligibility_reason": "ok" if eligible else "low_sky_fraction",
                    "thin_mask": "False", "dev_subset": str(i % 4 == 0),
                })
    with (root / "groups.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (root / "manifest.json").write_text(json.dumps({"dataset_revision": "synth-index-v1",
                                                    "content_digest": "deadbeef" * 8}, indent=2),
                                        encoding="utf-8")
    # the consistency index also carries masks; the bench reads them only for diagnostics
    packed = np.packbits(np.ones((len(rows), W), dtype=bool), axis=1)
    np.savez(root / "column_masks.npz", width=W, group_ids=np.array([r["group_id"] for r in rows]),
             packed=packed)
    return root


def place_curve(gid: str, condition: str = "original", offset_px: float = 0.0,
                switch: bool = False) -> np.ndarray:
    """A deterministic per-group skyline: a low-frequency shape keyed by the group, +offset, and
    optionally a half-width switch to a different structure."""
    import hashlib
    seed = int.from_bytes(hashlib.sha256(gid.encode()).digest()[:4], "little")   # distinct per group
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, W)
    rows = H * 0.4 + H * 0.15 * np.sin(2 * np.pi * (rng.uniform(1, 3) * x + rng.uniform())) \
        + H * 0.05 * np.sin(2 * np.pi * (rng.uniform(5, 9) * x))
    rows = rows + offset_px
    if switch:
        rows[: W // 2] = H * 0.8
    return np.clip(rows, 0, H - 1)


def make_store(root: Path, index_root: Path, offsets: dict | None = None, switches: set | None = None) -> Path:
    """observations_ecl-style store: sessions ecl_<condition> with images/ and skylines_auto/."""
    root = Path(root)
    offsets = offsets or {}
    switches = switches or set()
    with (Path(index_root) / "groups.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for cond in ("original", "summer", "winter", "evening"):
        sess = root / f"ecl_{cond}"
        (sess / "images").mkdir(parents=True, exist_ok=True)
        (sess / "skylines_auto").mkdir(parents=True, exist_ok=True)
        obs_rows = []
        for r in rows:
            if cond != "original" and not r[f"{cond}_relpath"]:
                continue
            gid = r["group_id"]
            curve = place_curve(gid, cond, offsets.get((gid, cond), 0.0), (gid, cond) in switches)
            with (sess / "skylines_auto" / f"{gid}.csv").open("w", encoding="utf-8", newline="") as f:
                f.write("col,row\n")
                for c in range(W):
                    f.write(f"{c},{float(curve[c])}\n")
            img = np.full((H, W, 3), 200, np.uint8)
            for c in range(W):
                img[int(curve[c]):, c] = 60
            try:
                import cv2
                cv2.imwrite(str(sess / "images" / f"{gid}.png"), img)
            except ImportError:  # pragma: no cover
                pass
            obs_rows.append({"observation_id": gid, "session_id": f"ecl_{cond}", "status": "ok"})
        (sess / "session.json").write_text(json.dumps({"session_id": f"ecl_{cond}", "local_frame_origin": None,
                                                       "frame_convention": "cambridge_sfm_local"}),
                                           encoding="utf-8")
        with (sess / "extraction_outcomes.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["observation_id", "session_id", "status", "reason", "confidence"])
            w.writeheader()
            for o in obs_rows:
                w.writerow({**o, "reason": "", "confidence": 0.9})
    return root


def make_label_maps(root: Path, index_root: Path, statuses: dict | None = None) -> Path:
    """SegFormer-style label maps (uint8, class 2 = sky) under <root>/<condition>/s512/<gid>.npz,
    following the group's curve: sky above the curve, building below; ``statuses`` may force
    'insufficient_sky' (sky only in a narrow band) or 'no_top_connected_sky' (an enclosed patch)."""
    root = Path(root)
    statuses = statuses or {}
    with (Path(index_root) / "groups.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for cond in ("original", "summer", "winter", "evening"):
        d = root / cond / "s512"
        d.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if cond != "original" and not r[f"{cond}_relpath"]:
                continue
            gid = r["group_id"]
            lab = np.full((H, W), 1, np.uint8)          # building
            st = statuses.get((gid, cond), "ok")
            curve = place_curve(gid, cond)
            if st == "ok":
                for c in range(W):
                    lab[: int(curve[c]), c] = 2
                lab[:, W - 8:] = 1                       # a tower reaching the top: invalid columns
            elif st == "insufficient_sky":
                lab[:5, : W // 4] = 2
            elif st == "no_top_connected_sky":
                lab[10:15, 10:20] = 2
            np.savez_compressed(d / f"{gid}.npz", label=lab)
    (root / "inference_manifest.json").write_text(json.dumps({
        "model": {"repo_id": "nvidia/segformer-b0-finetuned-ade-512-512",
                  "revision": "489d5cd81a0b59fab9b7ea758d3548ebe99677da", "sky_class_index": 2},
        "preprocessing": {"scales_short_side": [512]}, "output_digest": "0" * 64,
        "population": "all"}, indent=2), encoding="utf-8")
    return root


def split_config(population: str = "all") -> dict:
    return {"population": population, "spacings_m": [10.0, 25.0],
            "scene_offsets_m": {"SceneA": [0.0, 0.0], "SceneB": [10_000.0, 0.0]},
            "roles": {"reference_traversals": {"SceneA": "seq1", "SceneB": "seq1"},
                      "query_traversals": {"SceneA": ["seq2"]},
                      "excluded_scenes": []}}
