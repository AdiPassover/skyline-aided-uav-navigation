"""Synthetic sim_* fixture generator (spec 007, T027/US3). Proves the ingest -> QC -> split -> build
path is dataset-agnostic: a simulator source needs **no CVAT step at all** -- its oracle skyline is
known exactly (`sim_skyline_exact`) and is written directly to `skylines_oracle/`, the same location
`store.py` places a CVAT-derived curve. No real sim/UAV data; NOT the Java/UE5 simulator itself, just a
minimal analytically-defined stand-in for "a future sim adapter's output."
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

W, H = 64, 48
N_FRAMES = 12
ORIGIN_LAT, ORIGIN_LON = 50.0, 8.0


def exact_curve(frame_index: int) -> np.ndarray:
    """The known-exact skyline row per column -- deterministic, no annotation involved."""
    x = np.linspace(0, 2 * np.pi, W)
    return (H * 0.5 + 8 * np.sin(x + frame_index * 0.3)).astype(np.float64)


def _render(curve: np.ndarray) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    rows = np.arange(H)[:, None]
    sky = rows < curve[None, :]
    img[..., 0] = np.where(sky, 230, 40)   # B
    img[..., 1] = np.where(sky, 220, 90)   # G
    img[..., 2] = np.where(sky, 200, 55)   # R (sky: high B-R gap; ground: low)
    return img


def generate(root: Path | str, session_id: str, n_frames: int = N_FRAMES) -> tuple[Path, list[dict]]:
    """Write ``<root>/<session_id>/images/*.png`` + ``skylines_oracle/*.csv`` directly (no CVAT).
    Returns (session_dir, records) where each record has frame_index/lat/lon/image_path."""
    session_dir = Path(root) / session_id
    (session_dir / "images").mkdir(parents=True, exist_ok=True)
    (session_dir / "skylines_oracle").mkdir(parents=True, exist_ok=True)
    records = []
    for fi in range(n_frames):
        curve = exact_curve(fi)
        img = _render(curve)
        name = f"frame_{fi:06d}.png"
        cv2.imwrite(str(session_dir / "images" / name), img)
        obs_id = f"{session_id}_{fi:06d}"
        with (session_dir / "skylines_oracle" / f"{obs_id}.csv").open(
                "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["col", "row"])
            for c, row in enumerate(curve):
                w.writerow([c, round(float(row), 3)])
        records.append({
            "frame_index": fi, "image_path": name,
            "lat": round(ORIGIN_LAT + 0.0004 * fi, 6), "lon": round(ORIGIN_LON + 0.0006 * fi, 6),
        })
    return session_dir, records
