"""`EXP-VO-012`: render nadir sequences over a textured **height field**, with the takeoff datum and
the true AGL recorded as separate ground truth.

`EXP-VO-005`'s renderer warps one texture through the exact plane-induced homography of a *flat*
ground plane. That is enough for altitude experiments and not enough for this one, because the whole
question here is what happens when the **terrain** moves under a camera whose barometer only knows
its height above the **takeoff datum** (`LIT-VO-006` §3). So this renderer casts a ray per pixel and
intersects it with a terrain height field `e(x, y)`.

**It reduces exactly to `EXP-VO-005`'s renderer when `e ≡ 0`** — asserted by test (gate G1), which is
what lets the flat regimes here be read alongside the frozen `EXP-VO-005` results. It also imports
`EXP-VO-005`'s `make_texture` rather than re-deriving one, so the imagery family is identical.

**The three heights, kept apart** — this is the point of the module and the reason for the extra
ground-truth file:

    up_m          altitude above the TAKEOFF DATUM. What a barometer reports (LIT-VO-006 §2).
                  Written to groundtruth.csv, same column and meaning as the real datasets.
    terrain_m     terrain elevation above that same datum, under the camera. Unobservable to
                  every sensor this system has.
    agl_m         camera-to-imaged-surface height = up_m - terrain_m. What the geometry needs
                  (LIT-VO-003 §2) and what no channel on the author's platform supplies.

`groundtruth.csv` keeps exactly the contract columns; `terrain.csv` carries the other two, the same
way `attitude.csv` already carries attitude. Nothing about the dataset contract changes.

Geometry, frames and conventions are `EXP-VO-005`'s unchanged: ground datum `z = 0` in ENU, camera
centre `C = (E, N, h)`, `x_cam = right`, `y_cam = image-down`, `z_cam = boresight`, compass heading
clockwise from north, `K = [[f,0,cx],[0,f,cy],[0,0,1]]`.

Usage::

    python evaluation/tools/exp_vo_012/synth_terrain.py \
        --out datasets --configs evaluation/eval_configs/exp-vo-012
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
EVALUATION_DIR = HERE.parents[1]
REPO = EVALUATION_DIR.parent


def _load_exp_vo_005():
    """Import `EXP-VO-005`'s renderer by path — same texture, same seed, same constants."""
    path = EVALUATION_DIR / "tools" / "exp_vo_005" / "synth_planar.py"
    spec = importlib.util.spec_from_file_location("exp_vo_005_synth_planar", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = _load_exp_vo_005()

SEED = sp.SEED
F_PX = sp.F_PX
W, H = sp.W, sp.H
CX, CY = sp.CX, sp.CY
FPS = sp.FPS
TEXEL_M = sp.TEXEL_M
TEX_SIZE = sp.TEX_SIZE
SPEED_MS = sp.SPEED_MS
H0 = sp.H0

#: Ray/height-field intersection: iterate to a **stated tolerance**, not to a fixed count.
#:
#: The fixed count this started with (6) converges on flat ground and on the 14° ramp but **not** on
#: the 21° ridge, where it left a 6.9 mm residual and tripped `EXP-VO-012`'s pre-registered 1 mm gate
#: (G2) — which is the gate working. Convergence is geometric with factor `|∇e · d_xy / d_z|`, so the
#: iterations a terrain needs depend on its slope, and pinning a count pins the wrong thing.
#:
#: `RAY_MIN_ITERS` is a floor rather than a target, and it exists for one specific reason: the flat
#: and ramp datasets were rendered with exactly six unconditional iterations, and their run records
#: are committed. Keeping the floor at six means those regimes are reproduced **bit-for-bit** by this
#: version, while the ridge simply iterates further. (Flat terrain is exact after one iteration, so
#: only the ramp actually depends on the floor.)
RAY_TOLERANCE_M = 1e-3
RAY_MIN_ITERS = 6
RAY_MAX_ITERS = 40


# ------------------------------------------------------------------------------- terrain models

def terrain_flat(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """`e ≡ 0`. The regime in which `h₀ + h_baro == h_AGL` exactly (`LIT-VO-006` eq. 4)."""
    return np.zeros_like(north)


def terrain_ramp(east: np.ndarray, north: np.ndarray, *, rise_m: float, start_n: float,
                 end_n: float) -> np.ndarray:
    """A linear ramp along the flight direction, flat before and after.

    The clean failure case: `h_AGL` falls monotonically while the barometer reports nothing at all,
    so the two quantities are maximally separated and the resulting error must accumulate.
    """
    t = np.clip((north - start_n) / (end_n - start_n), 0.0, 1.0)
    return rise_m * t


def terrain_ridge(east: np.ndarray, north: np.ndarray, *, rise_m: float, centre_n: float,
                  width_n: float) -> np.ndarray:
    """A smooth ridge that comes back down — raised cosine, C¹ at both feet.

    Terrain that *returns* to its starting elevation separates a datum error from an accumulating
    estimator error without any fitting: a datum error must return to zero with the terrain, and an
    accumulating error must not.
    """
    u = np.clip((north - centre_n) / (0.5 * width_n), -1.0, 1.0)
    return rise_m * 0.5 * (1.0 + np.cos(math.pi * u))


TERRAINS = {
    "flat": (terrain_flat, {}),
    "ramp30": (terrain_ramp, dict(rise_m=30.0, start_n=-60.0, end_n=60.0)),
    "ridge25": (terrain_ridge, dict(rise_m=25.0, centre_n=0.0, width_n=200.0)),
}


def terrain_fn(name: str):
    fn, kw = TERRAINS[name]
    return lambda e, n: fn(e, n, **kw)


# ------------------------------------------------------------------------------- sequences
#
# Fixed in the EXP-VO-012 pre-registration. `alt` is altitude above the TAKEOFF DATUM -- i.e. exactly
# what a barometer reports -- NOT height above the ground.

SEQUENCES = {
    # R1 -- sanity / regression. Nothing should happen.
    "bvo-flat-const": dict(frames=600, terrain="flat", alt=("const", 80.0),
                           xy="straight", yaw="const"),
    # R2 -- THE PRIMARY TEST. Flat ground, +50 % altitude.
    "bvo-flat-climb": dict(frames=600, terrain="flat", alt=("linear", 80.0, 120.0),
                           xy="straight", yaw="const"),
    # R2b -- a harder altitude profile, and a turn, so H4's straight-leg cancellation is testable.
    "bvo-flat-profile": dict(frames=900, terrain="flat",
                             alt=("piecewise", [(0.0, 80.0), (0.34, 125.0), (0.67, 65.0),
                                                (1.0, 95.0)]),
                             xy="along_heading", yaw=("step", 450, 0.0, 90.0)),
    # R7a -- THE CRITICAL FAILURE CASE. Barometer says nothing; the ground rises 30 m.
    "bvo-ramp-constalt": dict(frames=600, terrain="ramp30", alt=("const", 80.0),
                              xy="straight", yaw="const"),
    # R7b -- terrain that returns to its own datum.
    "bvo-ridge-constalt": dict(frames=900, terrain="ridge25", alt=("const", 80.0),
                               xy="straight", yaw="const"),
    # R8 -- stress: vehicle vertical motion AND terrain elevation change together.
    "bvo-ramp-climb": dict(frames=900, terrain="ramp30", alt=("linear", 80.0, 110.0),
                           xy="straight", yaw="const"),
}

#: Capture arms. Homography is the reference configuration (`VO_CLOSURE.md` §3.1). Affine appears on
#: two regimes as a CONTROL on whether a conclusion is model-specific -- not as a model comparison,
#: which is closed (`VO_CLOSURE.md` §6.1).
MODELS = {sid: (("homography",) if sid not in ("bvo-flat-climb", "bvo-ramp-constalt")
                else ("homography", "affine"))
          for sid in SEQUENCES}


# ------------------------------------------------------------------------------- trajectories

def profile(kind, n: int) -> np.ndarray:
    t = np.arange(n) / (n - 1)
    if kind[0] == "const":
        return np.full(n, kind[1])
    if kind[0] == "linear":
        return kind[1] + (kind[2] - kind[1]) * t
    if kind[0] == "updown":
        return kind[1] + (kind[2] - kind[1]) * np.sin(math.pi * t)
    if kind[0] == "piecewise":
        pts = kind[1]
        return np.interp(t, [p[0] for p in pts], [p[1] for p in pts])
    raise ValueError(kind)


def heading_profile(spec, n: int) -> np.ndarray:
    if spec == "const":
        return np.zeros(n)
    if spec[0] == "linear":
        return profile(spec, n)
    if spec[0] == "step":
        # A turn executed over 100 frames centred on `at`, smooth (raised cosine) so the airframe is
        # never asked for an impulsive yaw the tracker would see as a discontinuity.
        _, at, a0, a1 = spec
        k = np.arange(n)
        u = np.clip((k - (at - 50)) / 100.0, 0.0, 1.0)
        return a0 + (a1 - a0) * 0.5 * (1.0 - np.cos(math.pi * u))
    raise ValueError(spec)


def trajectory(spec: dict) -> dict:
    n = spec["frames"]
    ts = np.arange(n) / FPS
    alt = profile(spec["alt"], n)
    yaw = heading_profile(spec["yaw"], n)
    east = np.zeros(n)
    north = np.zeros(n)
    if spec["xy"] == "straight":
        north = SPEED_MS * ts - SPEED_MS * ts[-1] / 2.0          # centred on the texture
    elif spec["xy"] == "along_heading":
        d = SPEED_MS / FPS
        north[0] = -SPEED_MS * ts[-1] / 2.0
        for k in range(1, n):
            psi = math.radians(yaw[k - 1])
            east[k] = east[k - 1] + d * math.sin(psi)
            north[k] = north[k - 1] + d * math.cos(psi)
    elif spec["xy"] == "none":
        pass
    else:
        raise ValueError(spec["xy"])
    return {"t": ts, "east": east, "north": north, "up": alt, "heading": yaw,
            "roll": np.zeros(n)}


# ------------------------------------------------------------------------------- rendering

def ray_directions(heading_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """World-frame unit ray direction for every pixel, shape (H, W, 3).

    `K⁻¹ p` gives the direction in camera coordinates; `R_cw` has the camera axes as ROWS (it maps
    world -> camera), so `R_cwᵀ` maps camera -> world.
    """
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    d_cam = np.stack([(u - CX) / F_PX, (v - CY) / F_PX, np.ones_like(u)], axis=-1)
    r_cw = sp.camera_axes(heading_deg, roll_deg)
    return d_cam @ r_cw                    # (H,W,3) @ (3,3) == d_cam · R_cw == R_cwᵀ d_cam per-ray


def intersect_terrain(camera: np.ndarray, dirs: np.ndarray, elev,
                      tolerance_m: float = RAY_TOLERANCE_M, min_iters: int = RAY_MIN_ITERS,
                      max_iters: int = RAY_MAX_ITERS):
    """Intersect each ray with `z = e(x, y)` by fixed-point iteration, to a stated tolerance.

    Start at the `z = 0` plane and re-solve the ray parameter against the terrain height at the
    current guess. Convergence is geometric with factor `|∇e · d_xy / d_z|`, so it holds for any
    terrain whose slope is gentler than the ray's own inclination — true for a near-nadir camera over
    every terrain used here — but the *rate* falls as the slope rises, which is why the loop is
    tolerance-driven rather than count-driven (see the constants above).

    Returns the ground point and the **worst residual over all rays**, which the caller asserts
    against `EXP-VO-012`'s pre-registered 1 mm gate. Convergence is therefore checked per frame, not
    assumed once.
    """
    dx, dy, dz = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    t = (0.0 - camera[2]) / dz                                   # z = 0 plane

    def refine(t):
        return (elev(camera[0] + t * dx, camera[1] + t * dy) - camera[2]) / dz

    # The first `min_iters` refinements are unconditional, which is exactly what the fixed-count
    # version did -- so a terrain that converged under it lands on the same `t`, bit for bit.
    for _ in range(min_iters):
        t = refine(t)
    for _ in range(max_iters - min_iters):
        e = elev(camera[0] + t * dx, camera[1] + t * dy)
        if float(np.abs((camera[2] + t * dz) - e).max()) < tolerance_m:
            break
        t = (e - camera[2]) / dz

    x = camera[0] + t * dx
    y = camera[1] + t * dy
    residual = float(np.abs((camera[2] + t * dz) - elev(x, y)).max())
    return x, y, residual


def render_terrain(texture: np.ndarray, camera: np.ndarray, heading_deg: float, elev,
                   roll_deg: float = 0.0):
    """One frame, by inverse mapping through the terrain. Returns (image, max residual in m)."""
    dirs = ray_directions(heading_deg, roll_deg)
    gx, gy, residual = intersect_terrain(camera, dirs, elev)
    # ground (E, N) -> texel, the inverse of EXP-VO-005's `texture_to_ground`
    map_x = (gx / TEXEL_M + TEX_SIZE / 2.0).astype(np.float32)
    map_y = (TEX_SIZE / 2.0 - gy / TEXEL_M).astype(np.float32)
    img = cv2.remap(texture, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, residual


# ------------------------------------------------------------------------------- dataset writing

def write_dataset(out_root: Path, seq_id: str, spec: dict, texture: np.ndarray,
                  render: bool = True) -> Path:
    tr = trajectory(spec)
    elev = terrain_fn(spec["terrain"])
    d = out_root / seq_id
    (d / "images").mkdir(parents=True, exist_ok=True)
    n = len(tr["t"])

    terrain_under = np.array([float(elev(np.array([tr["east"][k]]), np.array([tr["north"][k]]))[0])
                              for k in range(n)])
    agl = tr["up"] - terrain_under
    assert agl.min() > 5.0, f"{seq_id}: camera would be within 5 m of the ground"

    worst = 0.0
    with (d / "frames.csv").open("w", newline="") as f:
        f.write("frame_index,timestamp_s,image_path\n")
        for k in range(n):
            name = f"images/frame_{k:05d}.png"
            if render:
                cam = np.array([tr["east"][k], tr["north"][k], tr["up"][k]])
                img, res = render_terrain(texture, cam, tr["heading"][k], elev, tr["roll"][k])
                worst = max(worst, res)
                # The frame must lie entirely on the texture, or the estimator is tracking a border.
                assert (img > 0).mean() > 0.999, f"{seq_id} frame {k} leaves the texture"
                cv2.imwrite(str(d / name), img)
            f.write(f"{k},{tr['t'][k]:.6f},{name}\n")
    assert worst < 1e-3, f"{seq_id}: ray/terrain intersection residual {worst:.3g} m exceeds 1 mm"

    # groundtruth.csv keeps EXACTLY the contract columns. `up_m` is altitude above the TAKEOFF
    # DATUM, the same meaning it has in the real datasets -- i.e. what a barometer reports.
    with (d / "groundtruth.csv").open("w", newline="") as f:
        f.write("timestamp_s,east_m,north_m,up_m,heading_deg,fix_quality,valid\n")
        for k in range(n):
            f.write(f"{tr['t'][k]:.6f},{tr['east'][k]:.6f},{tr['north'][k]:.6f},"
                    f"{tr['up'][k]:.6f},{tr['heading'][k] % 360.0:.6f},synthetic_exact,true\n")

    # The two quantities the barometer cannot see, in their own file (same convention as
    # attitude.csv). agl_m is the ORACLE height; terrain_m is what breaks the flat-terrain
    # assumption.
    with (d / "terrain.csv").open("w", newline="") as f:
        f.write("timestamp_s,terrain_m,agl_m,up_m,baro_relative_m\n")
        for k in range(n):
            f.write(f"{tr['t'][k]:.6f},{terrain_under[k]:.6f},{agl[k]:.6f},{tr['up'][k]:.6f},"
                    f"{tr['up'][k] - tr['up'][0]:.6f}\n")

    with (d / "attitude.csv").open("w", newline="") as f:
        f.write("timestamp_s,roll_deg,pitch_deg,tilt_deg,yaw_compass_deg\n")
        for k in range(n):
            f.write(f"{tr['t'][k]:.6f},{tr['roll'][k]:.6f},0.0,{abs(tr['roll'][k]):.6f},"
                    f"{tr['heading'][k] % 360.0:.6f}\n")

    meta = {
        "schema_version": "1.0.0", "dataset_id": seq_id, "dataset_revision": "v1",
        "source_type": "synthetic", "evidence_tier": "T1",
        "evidence_caveat":
            "EXP-VO-012 rendered terrain sequence: one procedural texture (EXP-VO-005's, same seed) "
            "sampled by per-pixel ray intersection with a known height field, through a pinhole "
            "camera with exact intrinsics. No distortion, no noise, no JPEG. On the flat regimes the "
            "imagery satisfies the estimator's planar model exactly (same caveat as EXP-VO-005); on "
            "the terrain regimes it deliberately does not. Establishes estimator and readout "
            "behaviour under known geometry; says nothing about real imagery or any real barometer.",
        "frame_clock": "synthetic", "gt_clock": "synthetic", "clock_offset_s": 0.0,
        "clock_offset_source": "same clock by construction", "clock_drift_s_per_s": 0.0,
        "frame_convention": "ENU", "heading_convention": "compass_cw_from_north",
        "local_frame_origin": None,
        "position_quality": {"class": "simulator_exact", "nominal_accuracy": 0.0,
                             "accuracy_source": "exact rendered pose", "notes": ""},
        "heading_quality": {"class": "simulator_exact", "nominal_accuracy": 0.0,
                            "accuracy_source": "exact rendered pose", "notes": ""},
        "height_quality": {"class": "simulator_exact", "nominal_accuracy": 0.0,
                           "accuracy_source": "exact rendered pose",
                           "notes": "up_m is altitude above the TAKEOFF DATUM -- what a barometer "
                                    "reports. Height above the imaged surface is agl_m in "
                                    "terrain.csv and is a DIFFERENT quantity wherever terrain_m "
                                    "is nonzero. See LIT-VO-006 section 3."},
        "metadata": {
            "flight_id": seq_id, "trajectory_type": "custom", "frame_rate_hz": FPS,
            "image_width": W, "image_height": H,
            "environment": f"synthetic textured terrain ({spec['terrain']})",
            "nominal_altitude_m": float(tr["up"][0]),
            "nominal_speed_ms": SPEED_MS if spec["xy"] != "none" else 0.0,
            "capture_date": "2026-08-27",
            "notes": f"EXP-VO-012 regime '{seq_id}': "
                     f"{json.dumps({k: str(v) for k, v in spec.items()})}",
            "camera_intrinsics": {"fx": F_PX, "fy": F_PX, "cx": CX, "cy": CY,
                                  "k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0,
                                  "width": W, "height": H, "source": "synthetic, exact"},
            "gsd0_m_per_px": float(agl[0]) / F_PX,
            "h0_agl_m": float(agl[0]),
            "terrain_model": spec["terrain"],
            "terrain_range_m": [float(terrain_under.min()), float(terrain_under.max())],
            "agl_range_m": [float(agl.min()), float(agl.max())],
            "texel_m": TEXEL_M, "seed": SEED,
        },
    }
    (d / "dataset.json").write_text(json.dumps(meta, indent=2))
    return d


def write_configs(cfg_dir: Path, out_root: Path, ids: list[str]) -> None:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        for model in MODELS[sid]:
            cfg = {
                "dataset_dir": str(out_root / sid).replace("\\", "/"),
                "output_dir": "runs", "run_id": f"{sid}-{model}-rigid-v1",
                "estimator_id": "stitching-vo", "estimator_version": "exp-vo-012",
                "is_target_hardware": False, "downsampleFactor": 1, "motion_model": model,
                "navigation_source": "rigid_motion", "logical_transform_sidecar": True,
            }
            (cfg_dir / f"run-{sid}-{model}.json").write_text(json.dumps(cfg, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-render", action="store_true",
                    help="write descriptors and configs only (for inspecting the trajectories)")
    a = ap.parse_args()
    out_root = Path(a.out)
    ids = a.only or list(SEQUENCES)
    tex = sp.make_texture()
    for sid in ids:
        d = write_dataset(out_root, sid, SEQUENCES[sid], tex, render=not a.no_render)
        print(f"wrote {d}  {SEQUENCES[sid]['frames']} frames  terrain={SEQUENCES[sid]['terrain']}",
              flush=True)
    write_configs(Path(a.configs), out_root, ids)
    print(f"wrote configs to {a.configs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
