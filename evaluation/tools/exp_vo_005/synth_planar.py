"""EXP-VO-005: render exact synthetic nadir sequences of a textured plane with known intrinsics,
known XY / altitude / heading / roll trajectories, and write them as `contracts/dataset.md`
datasets (`source_type: synthetic`, `evidence_tier: T1`) plus the matching VoRunnerApp configs.

Geometry (LIT-VO-003 §1-2). Ground plane z = 0 in ENU. Camera centre C = (E, N, h). Camera axes in
world coordinates for heading psi (compass, clockwise from north), zero attitude:
    x_cam = right  = ( cos psi, -sin psi, 0)
    y_cam = image-down = -forward = (-sin psi, -cos psi, 0)
    z_cam = boresight = (0, 0, -1)
Roll phi rotates the axes about the forward vector, pitch theta about the right vector. With
K = [[f,0,cx],[0,f,cy],[0,0,1]] the ground->image homography is H = K R_cw [e1 e2 -C], and the
frame is rendered by warping the texture through H composed with the texture->ground map.
Everything is exact; no lens distortion, no relief, no noise, PNG output.

Usage::

    python evaluation/tools/exp_vo_005/synth_planar.py --out datasets --configs evaluation/eval_configs/exp-vo-005
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

SEED = 20260825
F_PX = 735.0
W, H = 1224, 1024
CX, CY = W / 2.0, H / 2.0
FPS = 10.0
TEXEL_M = 0.10
TEX_SIZE = 8192
SPEED_MS = 3.0
H0 = 80.0

SEQUENCES = {
    "const":     dict(frames=1800, alt=("const", 80.0), xy="straight", yaw="const", roll=None),
    "climb":     dict(frames=600,  alt=("linear", 80.0, 120.0), xy="straight", yaw="const", roll=None),
    "descent":   dict(frames=600,  alt=("linear", 80.0, 50.0), xy="straight", yaw="const", roll=None),
    "climbdesc": dict(frames=800,  alt=("updown", 80.0, 120.0), xy="straight", yaw="const", roll=None),
    "altonly":   dict(frames=400,  alt=("updown", 80.0, 110.0), xy="none", yaw="const", roll=None),
    "yawonly":   dict(frames=400,  alt=("const", 80.0), xy="none", yaw=("linear", 0.0, 360.0), roll=None),
    "combo":     dict(frames=800,  alt=("updown", 80.0, 100.0), xy="along_heading", yaw=("linear", 0.0, 180.0), roll=None),
    "tilt":      dict(frames=600,  alt=("const", 80.0), xy="straight", yaw="const", roll=("sin", 10.0, 0.1)),
}


# ----------------------------------------------------------------------------- texture
def make_texture(seed: int = SEED, size: int = TEX_SIZE) -> np.ndarray:
    """Band-limited multi-octave noise plus sparse high-contrast blobs, uint8."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size), np.float32)
    for octave, amp in ((16, 1.0), (64, 0.7), (256, 0.5), (1024, 0.3)):
        small = rng.normal(0, 1, (octave, octave)).astype(np.float32)
        acc += amp * cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    img = (40 + 150 * acc).astype(np.uint8)
    n_blobs = (size // 64) ** 2 // 2
    xs = rng.integers(0, size, n_blobs); ys = rng.integers(0, size, n_blobs)
    for x, y in zip(xs, ys):
        r = int(rng.integers(3, 12))
        cv2.circle(img, (int(x), int(y)), r, int(rng.integers(0, 255)), -1)
    return img


# ----------------------------------------------------------------------------- trajectories
def profile(kind, n):
    t = np.arange(n) / (n - 1)
    if kind[0] == "const":
        return np.full(n, kind[1])
    if kind[0] == "linear":
        return kind[1] + (kind[2] - kind[1]) * t
    if kind[0] == "updown":
        return kind[1] + (kind[2] - kind[1]) * np.sin(math.pi * t)
    raise ValueError(kind)


def trajectory(spec: dict) -> dict:
    n = spec["frames"]
    ts = np.arange(n) / FPS
    alt = profile(spec["alt"], n)
    yaw = np.zeros(n) if spec["yaw"] == "const" else profile(spec["yaw"], n)
    roll = np.zeros(n)
    if spec["roll"]:
        _, amp, hz = spec["roll"]
        roll = amp * np.sin(2 * math.pi * hz * ts)
    east = np.zeros(n); north = np.zeros(n)
    if spec["xy"] == "straight":
        north = SPEED_MS * ts - SPEED_MS * ts[-1] / 2.0      # centred on the texture
    elif spec["xy"] == "along_heading":
        d = SPEED_MS / FPS
        for k in range(1, n):
            psi = math.radians(yaw[k - 1])
            east[k] = east[k - 1] + d * math.sin(psi)
            north[k] = north[k - 1] + d * math.cos(psi)
    return {"t": ts, "east": east, "north": north, "up": alt, "heading": yaw, "roll": roll}


# ----------------------------------------------------------------------------- rendering
def camera_axes(heading_deg: float, roll_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    psi = math.radians(heading_deg)
    fwd = np.array([math.sin(psi), math.cos(psi), 0.0])
    right = np.array([math.cos(psi), -math.sin(psi), 0.0])
    down = np.array([0.0, 0.0, -1.0])

    def rot(axis, ang):
        axis = axis / np.linalg.norm(axis); a = math.radians(ang)
        Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + math.sin(a) * Kx + (1 - math.cos(a)) * Kx @ Kx

    R = rot(fwd, roll_deg) @ rot(right, pitch_deg)
    x_cam = R @ right; y_cam = R @ (-fwd); z_cam = R @ down
    return np.stack([x_cam, y_cam, z_cam])      # rows: camera axes in world = R_cw


def ground_to_image(east, north, up, heading, roll) -> np.ndarray:
    K = np.array([[F_PX, 0, CX], [0, F_PX, CY], [0, 0, 1.0]])
    Rcw = camera_axes(heading, roll)
    C = np.array([east, north, up])
    P = np.column_stack([np.array([1, 0, 0.0]), np.array([0, 1, 0.0]), -C])
    return K @ Rcw @ P


def texture_to_ground(size: int, texel: float) -> np.ndarray:
    # texel (tx, ty) -> ground (E, N): E = (tx - size/2) * texel ; N = -(ty - size/2) * texel
    return np.array([[texel, 0, -size / 2 * texel], [0, -texel, size / 2 * texel], [0, 0, 1.0]])


def render(texture: np.ndarray, Hg2i: np.ndarray) -> np.ndarray:
    Ht2i = Hg2i @ texture_to_ground(texture.shape[0], TEXEL_M)
    return cv2.warpPerspective(texture, Ht2i, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def footprint_extent_m(up: float) -> float:
    return W * up / F_PX


# ----------------------------------------------------------------------------- dataset writing
def write_dataset(out_root: Path, seq_id: str, spec: dict, texture: np.ndarray) -> Path:
    tr = trajectory(spec)
    d = out_root / f"synth-scale-{seq_id}"
    (d / "images").mkdir(parents=True, exist_ok=True)
    n = len(tr["t"])
    with (d / "frames.csv").open("w", newline="") as f:
        f.write("frame_index,timestamp_s,image_path\n")
        for k in range(n):
            Hg2i = ground_to_image(tr["east"][k], tr["north"][k], tr["up"][k], tr["heading"][k], tr["roll"][k])
            img = render(texture, Hg2i)
            assert img.min() >= 0 and (img > 0).mean() > 0.999, f"frame {k} leaves the texture"
            name = f"images/frame_{k:05d}.png"
            cv2.imwrite(str(d / name), img)
            f.write(f"{k},{tr['t'][k]:.6f},{name}\n")
    with (d / "groundtruth.csv").open("w", newline="") as f:
        f.write("timestamp_s,east_m,north_m,up_m,heading_deg,fix_quality,valid\n")
        for k in range(n):
            f.write(f"{tr['t'][k]:.6f},{tr['east'][k]:.6f},{tr['north'][k]:.6f},{tr['up'][k]:.6f},"
                    f"{tr['heading'][k] % 360.0:.6f},synthetic_exact,true\n")
    with (d / "attitude.csv").open("w", newline="") as f:
        f.write("timestamp_s,roll_deg,pitch_deg,tilt_deg,yaw_compass_deg\n")
        for k in range(n):
            f.write(f"{tr['t'][k]:.6f},{tr['roll'][k]:.6f},0.0,{abs(tr['roll'][k]):.6f},{tr['heading'][k] % 360.0:.6f}\n")
    meta = {
        "schema_version": "1.0.0", "dataset_id": f"synth-scale-{seq_id}", "dataset_revision": "v1",
        "source_type": "synthetic", "evidence_tier": "T1",
        "evidence_caveat": "EXP-VO-005 rendered planar sequence: one procedural texture warped through the exact "
                           "plane-induced homography of a pinhole camera with known intrinsics. The imagery satisfies the "
                           "estimator's own planar model exactly (no distortion, no relief, no noise, PNG). Establishes "
                           "estimator behaviour when its assumptions hold; says nothing about real imagery.",
        "frame_clock": "synthetic", "gt_clock": "synthetic", "clock_offset_s": 0.0,
        "clock_offset_source": "same clock by construction", "clock_drift_s_per_s": 0.0,
        "frame_convention": "ENU", "heading_convention": "compass_cw_from_north", "local_frame_origin": None,
        "position_quality": {"class": "simulator_exact", "nominal_accuracy": 0.0, "accuracy_source": "exact rendered pose", "notes": ""},
        "heading_quality": {"class": "simulator_exact", "nominal_accuracy": 0.0, "accuracy_source": "exact rendered pose", "notes": ""},
        "height_quality": {"class": "simulator_exact", "nominal_accuracy": 0.0, "accuracy_source": "exact rendered pose", "notes": "altitude is scripted and exact"},
        "metadata": {
            "flight_id": f"synth-scale-{seq_id}", "trajectory_type": "custom", "frame_rate_hz": FPS,
            "image_width": W, "image_height": H, "environment": "synthetic textured plane",
            "nominal_altitude_m": H0, "nominal_speed_ms": SPEED_MS if spec["xy"] != "none" else 0.0,
            "capture_date": "2026-08-25",
            "notes": f"EXP-VO-005 sequence '{seq_id}': {json.dumps({k: str(v) for k, v in spec.items()})}",
            "camera_intrinsics": {"fx": F_PX, "fy": F_PX, "cx": CX, "cy": CY, "k1": 0.0, "k2": 0.0, "k3": 0.0,
                                  "p1": 0.0, "p2": 0.0, "width": W, "height": H, "source": "synthetic, exact"},
            "gsd0_m_per_px": H0 / F_PX, "texel_m": TEXEL_M, "seed": SEED,
        },
    }
    (d / "dataset.json").write_text(json.dumps(meta, indent=2))
    return d


def write_configs(cfg_dir: Path, out_root: Path, dataset_ids: list[str]) -> None:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for sid in dataset_ids:
        for model in ("homography", "affine"):
            variants = [("", 10)] + ([("-border60", 60)] if sid == "climb" else [])
            for suffix, border in variants:
                cfg = {
                    "dataset_dir": str(out_root / f"synth-scale-{sid}").replace("\\", "/"),
                    "output_dir": "runs", "run_id": f"synth-scale-{sid}-{model}{suffix}-v1",
                    "estimator_id": "stitching-vo", "estimator_version": "exp-vo-005",
                    "is_target_hardware": False, "downsampleFactor": 1, "motion_model": model,
                    "minDistanceFromBorder": border,
                    "navigation_source": "logical_frame", "logical_transform_sidecar": True,
                }
                (cfg_dir / f"run-synth-{sid}-{model}{suffix}.json").write_text(json.dumps(cfg, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out_root = Path(a.out)
    tex = make_texture()
    ids = a.only or list(SEQUENCES)
    for sid in ids:
        d = write_dataset(out_root, sid, SEQUENCES[sid], tex)
        print("wrote", d, SEQUENCES[sid]["frames"], "frames", flush=True)
    write_configs(Path(a.configs), out_root, ids)


if __name__ == "__main__":
    main()
