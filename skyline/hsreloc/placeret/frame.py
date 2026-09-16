"""Per-scene synthetic ENU frames (research R1; DEC-SKY-006 R3).

Cambridge Landmarks poses live in a scene-local SfM frame whose axes are arbitrary — in two of the
four scenes the SfM ``z`` spans 34–73 m across a planar walk, so it is not "up". Each scene's frame
is therefore the PCA plane of **all** its test-frame positions: east = first principal axis, north =
second, up = the normal. Scenes are placed at declared 10 km offsets so that cross-scene candidates
are distant negatives by construction, and the geodetic origin every spec-006 artifact must declare
is a **synthetic** one — no absolute geographic quantity is ever meaningful here, and every manifest
says so (``is_synthetic``).

Pose only: this module imports numpy and the standard library and reads nothing but position
arrays handed to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FRAME_VERSION = "1.0.0"
SYNTHETIC_ORIGIN = {"lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0, "is_synthetic": True}
SCENE_OFFSETS_M = {
    "KingsCollege": (0.0, 0.0),
    "OldHospital": (10_000.0, 0.0),
    "ShopFacade": (0.0, 10_000.0),
    "StMarysChurch": (10_000.0, 10_000.0),
}
UNITS = "SfM m (Cambridge Landmarks SfM frame; approximately metric, scale unverified)"


class FrameError(Exception):
    """A scene frame cannot be fitted or read as the contract requires."""


@dataclass(frozen=True)
class SceneFrame:
    scene: str
    centroid: np.ndarray            # (3,) SfM
    basis: np.ndarray               # (3, 3) rows: east, north, up — orthonormal
    offset_m: tuple                 # declared synthetic offset (east, north)
    off_plane_rms_m: float
    n_points: int
    is_synthetic: bool = True
    version: str = FRAME_VERSION

    def to_enu(self, pos_xyz) -> np.ndarray:
        """SfM (N, 3) or (3,) -> synthetic ENU (N, 3) or (3,): east, north, up."""
        p = np.atleast_2d(np.asarray(pos_xyz, dtype=np.float64)) - self.centroid
        enu = p @ self.basis.T
        enu[:, 0] += self.offset_m[0]
        enu[:, 1] += self.offset_m[1]
        return enu[0] if np.ndim(pos_xyz) == 1 else enu

    def as_dict(self) -> dict:
        return {"scene": self.scene, "centroid_sfm": self.centroid.tolist(),
                "basis_rows_east_north_up": self.basis.tolist(),
                "offset_m": list(self.offset_m), "off_plane_rms_m": self.off_plane_rms_m,
                "n_points": self.n_points, "is_synthetic": self.is_synthetic, "version": self.version}

    @classmethod
    def from_dict(cls, d: dict) -> "SceneFrame":
        return cls(scene=d["scene"], centroid=np.asarray(d["centroid_sfm"], dtype=np.float64),
                   basis=np.asarray(d["basis_rows_east_north_up"], dtype=np.float64),
                   offset_m=tuple(d["offset_m"]), off_plane_rms_m=float(d["off_plane_rms_m"]),
                   n_points=int(d["n_points"]), is_synthetic=bool(d.get("is_synthetic", True)),
                   version=d.get("version", FRAME_VERSION))


def fit_scene_frame(scene: str, positions, offset_m=None) -> SceneFrame:
    """PCA plane of the scene's positions; the sign of each axis is fixed deterministically
    (largest-magnitude component positive) so a rebuild reproduces the frame byte-for-byte."""
    P = np.asarray(positions, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        raise FrameError(f"{scene}: need at least three 3-D positions, got shape {P.shape}")
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    basis = vt.copy()
    for i in range(3):
        j = int(np.argmax(np.abs(basis[i])))
        if basis[i, j] < 0:
            basis[i] = -basis[i]
    if np.linalg.det(basis) < 0:            # keep a right-handed frame: flip "up"
        basis[2] = -basis[2]
    off = (P - c) @ basis[2]
    if offset_m is None:
        if scene not in SCENE_OFFSETS_M:
            raise FrameError(f"{scene}: no declared offset; add it to SCENE_OFFSETS_M explicitly")
        offset_m = SCENE_OFFSETS_M[scene]
    return SceneFrame(scene=scene, centroid=c, basis=basis, offset_m=tuple(float(v) for v in offset_m),
                      off_plane_rms_m=float(np.sqrt(np.mean(off ** 2))), n_points=int(P.shape[0]))


def write_frames(frames: dict, path: Path) -> None:
    payload = {"version": FRAME_VERSION, "units": UNITS, "geodetic_origin": SYNTHETIC_ORIGIN,
               "scene_offsets_m": {k: list(v) for k, v in SCENE_OFFSETS_M.items()},
               "scenes": {s: f.as_dict() for s, f in sorted(frames.items())}}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_frames(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("version") != FRAME_VERSION:
        raise FrameError(f"frames file version {d.get('version')!r} != {FRAME_VERSION}")
    if not d.get("geodetic_origin", {}).get("is_synthetic"):
        raise FrameError("frames file does not declare a synthetic origin — refusing to read it")
    return {s: SceneFrame.from_dict(v) for s, v in d["scenes"].items()}
