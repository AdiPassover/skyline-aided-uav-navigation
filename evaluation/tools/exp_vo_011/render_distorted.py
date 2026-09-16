"""EXP-VO-011 Phase 3b: the rendered lens-distortion arm.

The correspondence-level harness (`ScaleBiasMonteCarlo`) answers what distortion does to a *fit*.
This answers whether that survives the real pipeline -- KLT tracking, respawn, keyframing, canvas
re-origin and composition -- because a tracker could in principle interact with a warped image in a
way a point-correspondence model does not capture.

Three arms over the SAME trajectory and the SAME texture, so the only difference is the warp:

- `none`       -- ideal pinhole frames. Must reproduce `EXP-VO-005` R2's flat scale.
- `distorted`  -- frames carrying a Brown-Conrady radial distortion, fed to the estimator raw, which
                  is how the pipeline treats the real imagery (it applies no distortion model
                  anywhere, `COMP-001` section 2).
- `rectified`  -- those distorted frames, un-warped with the correct coefficients before the
                  estimator sees them. This is the H2 arm.

**Two rendering decisions worth stating, because they change what the arms mean.** `none` and
`distorted` are both sampled in a SINGLE pass from the 8192-texel texture, with the distortion
folded into the inverse map -- so `distorted` has no black rim and no double-resampling loss, and
the two arms differ by the geometry alone. `rectified`, by contrast, is produced the way a real
pipeline produces it: the distorted frame, resampled a second time through the rectification map.
It therefore carries the resampling loss and the black rim a real rectification has, which is the
honest thing for it to carry -- rectifying the ideal render instead would just be the control arm
with extra steps.

One heading only. The travel-direction dependence is the correspondence-level harness's job, where
it can be swept exactly; here the question is magnitude through the real tracker.

Usage::

    python evaluation/tools/exp_vo_011/render_distorted.py --out datasets \
        --configs evaluation/eval_configs/exp-vo-011
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "evaluation" / "tools" / "exp_vo_005"))

import synth_planar as sp  # noqa: E402

# The realistic middle of the sweep: about 19 px of corner barrel at this geometry, ~2.4 % of the
# corner radius -- inside the 1-5 % a wide machine-vision lens shows at 79.5 degrees HFOV.
K1 = -0.02
FRAMES = 600
SEQ = dict(frames=FRAMES, alt=("const", 80.0), xy="straight", yaw="const", roll=None)
ARMS = ("none", "distorted", "rectified")


def corner_distortion_px(k1: float) -> float:
    r = float(math.hypot(sp.CX, sp.CY) / sp.F_PX)
    return abs(k1 * r * r) * r * sp.F_PX


def _grid() -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(np.arange(sp.W, dtype=np.float64), np.arange(sp.H, dtype=np.float64))


def _undistorted_grid(k1: float) -> tuple[np.ndarray, np.ndarray]:
    """For every DISTORTED output pixel, the ideal pixel it must show. Fixed-point inversion."""
    u, v = _grid()
    xd, yd = (u - sp.CX) / sp.F_PX, (v - sp.CY) / sp.F_PX
    x, y = xd.copy(), yd.copy()
    for _ in range(20):
        f = 1.0 + k1 * (x * x + y * y)
        x, y = xd / f, yd / f
    return sp.CX + sp.F_PX * x, sp.CY + sp.F_PX * y


def _rectify_maps(k1: float) -> tuple[np.ndarray, np.ndarray]:
    """For every RECTIFIED output pixel, the distorted pixel to sample: the forward warp."""
    u, v = _grid()
    x, y = (u - sp.CX) / sp.F_PX, (v - sp.CY) / sp.F_PX
    f = 1.0 + k1 * (x * x + y * y)
    return ((sp.CX + sp.F_PX * x * f).astype(np.float32),
            (sp.CY + sp.F_PX * y * f).astype(np.float32))


def _render(texture: np.ndarray, Hg2i: np.ndarray,
            ideal_u: np.ndarray | None, ideal_v: np.ndarray | None) -> np.ndarray:
    """One sampling pass from the texture. `ideal_*` is None for the pinhole arm."""
    Ht2i = Hg2i @ sp.texture_to_ground(texture.shape[0], sp.TEXEL_M)
    if ideal_u is None:
        return cv2.warpPerspective(texture, Ht2i, (sp.W, sp.H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    Hi2t = np.linalg.inv(Ht2i)
    w = Hi2t[2, 0] * ideal_u + Hi2t[2, 1] * ideal_v + Hi2t[2, 2]
    tx = (Hi2t[0, 0] * ideal_u + Hi2t[0, 1] * ideal_v + Hi2t[0, 2]) / w
    ty = (Hi2t[1, 0] * ideal_u + Hi2t[1, 1] * ideal_v + Hi2t[1, 2]) / w
    return cv2.remap(texture, tx.astype(np.float32), ty.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def write_arm(out_root: Path, arm: str, texture: np.ndarray) -> Path:
    tr = sp.trajectory(SEQ)
    seq_id = f"synth-dist-{arm}"
    d = out_root / seq_id
    (d / "images").mkdir(parents=True, exist_ok=True)

    ideal_u, ideal_v = (None, None) if arm == "none" else _undistorted_grid(K1)
    rect = _rectify_maps(K1) if arm == "rectified" else None

    rows, gt_rows, att_rows = [], [], []
    covered = []
    for i in range(SEQ["frames"]):
        Hg2i = sp.ground_to_image(tr["east"][i], tr["north"][i], tr["up"][i],
                                  tr["heading"][i], tr["roll"][i])
        img = _render(texture, Hg2i, ideal_u, ideal_v)
        if rect is not None:
            img = cv2.remap(img, rect[0], rect[1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        covered.append(float((img > 0).mean()))
        name = f"frame_{i:05d}.png"
        cv2.imwrite(str(d / "images" / name), img)
        t = tr["t"][i]
        rows.append(f"{i},{t:.6f},images/{name}")
        gt_rows.append(f"{t:.6f},{tr['east'][i]:.6f},{tr['north'][i]:.6f},{tr['up'][i]:.6f},"
                       f"{tr['heading'][i] % 360.0:.6f},synthetic_exact,true")
        att_rows.append(f"{t:.6f},{tr['roll'][i]:.6f},0.0,{abs(tr['roll'][i]):.6f},"
                        f"{tr['heading'][i] % 360.0:.6f}")

    (d / "frames.csv").write_text("frame_index,timestamp_s,image_path\n" + "\n".join(rows) + "\n")
    (d / "groundtruth.csv").write_text(
        "timestamp_s,east_m,north_m,up_m,heading_deg,fix_quality,valid\n" + "\n".join(gt_rows) + "\n")
    (d / "attitude.csv").write_text(
        "timestamp_s,roll_deg,pitch_deg,tilt_deg,yaw_compass_deg\n" + "\n".join(att_rows) + "\n")

    caveat = {
        "none": "Ideal pinhole projection of an exact plane -- the control arm, the same geometry "
                "EXP-VO-005 used, single sampling pass from the texture.",
        "distorted": f"Brown-Conrady radial distortion k1={K1} ({corner_distortion_px(K1):.1f} px "
                     f"of corner barrel) folded into the inverse map, so it is a single sampling "
                     f"pass with no black rim, and NOT corrected before estimation -- which is how "
                     f"the pipeline treats real imagery.",
        "rectified": f"The k1={K1} distorted frames, correctly un-warped before estimation. Two "
                     f"sampling passes and a black rim, as a real rectification pipeline has.",
    }[arm]
    quality = {"class": "simulator_exact", "nominal_accuracy": 0.0,
               "accuracy_source": "exact rendered pose", "notes": ""}

    (d / "dataset.json").write_text(json.dumps({
        "schema_version": "1.0.0", "dataset_id": seq_id, "dataset_revision": "v1",
        "source_type": "synthetic", "evidence_tier": "T1",
        "evidence_caveat": ("EXP-VO-011 Phase 3b lens-distortion arm. One procedural texture warped "
                            "through the exact plane-induced homography, so the underlying geometry "
                            "satisfies the estimator's own planar model exactly -- the same caveat "
                            "as EXP-VO-005. " + caveat),
        "frame_clock": "synthetic", "gt_clock": "synthetic", "clock_offset_s": 0.0,
        "clock_offset_source": "same clock by construction", "clock_drift_s_per_s": 0.0,
        "frame_convention": "ENU", "heading_convention": "compass_cw_from_north",
        "local_frame_origin": None,
        "position_quality": quality, "heading_quality": quality, "height_quality": quality,
        "metadata": {
            "flight_id": seq_id, "trajectory_type": "straight", "frame_rate_hz": sp.FPS,
            "image_width": sp.W, "image_height": sp.H, "nominal_altitude_m": sp.H0,
            "nominal_speed_ms": sp.SPEED_MS, "capture_date": "2026-08-27",
            "environment": "procedural texture on an exact plane",
            "notes": f"EXP-VO-011 Phase 3b arm '{arm}', k1={K1 if arm != 'none' else 0.0}",
            "distortion_k1": K1 if arm != "none" else 0.0,
            "distortion_corrected": arm == "rectified",
            "corner_distortion_px": corner_distortion_px(K1) if arm != "none" else 0.0,
            "min_frame_coverage": min(covered), "mean_frame_coverage": sum(covered) / len(covered),
            "camera_intrinsics": {"fx": sp.F_PX, "fy": sp.F_PX, "cx": sp.CX, "cy": sp.CY,
                                  "k1": K1 if arm != "none" else 0.0, "k2": 0.0, "k3": 0.0,
                                  "p1": 0.0, "p2": 0.0, "width": sp.W, "height": sp.H,
                                  "source": "exact, by construction"},
            "gsd0_m_per_px": sp.H0 / sp.F_PX, "texel_m": sp.TEXEL_M, "seed": sp.SEED,
            "provenance": "evaluation/tools/exp_vo_011/render_distorted.py",
        },
    }, indent=2))
    print(f"{seq_id}: {SEQ['frames']} frames, coverage min {min(covered):.4f}")
    return d


def write_configs(cfg_dir: Path) -> None:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for model in ("affine", "homography"):
            (cfg_dir / f"run-synth-dist-{arm}-{model}.json").write_text(json.dumps({
                "dataset_dir": f"datasets/synth-dist-{arm}",
                "output_dir": "runs",
                "run_id": f"synth-dist-{arm}-{model}-v1",
                "estimator_id": "stitching-vo",
                "estimator_version": "exp-vo-011",
                "is_target_hardware": False,
                "downsampleFactor": 1,
                "motion_model": model,
                "navigation_source": "rigid_motion",
                "logical_transform_sidecar": True,
            }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", required=True)
    a = ap.parse_args()
    corner = corner_distortion_px(K1)
    print(f"k1 = {K1}  ->  {corner:.1f} px of corner barrel "
          f"({100 * corner / math.hypot(sp.CX, sp.CY):.2f} % of the corner radius)")
    texture = sp.make_texture()
    for arm in ARMS:
        write_arm(Path(a.out), arm, texture)
    write_configs(Path(a.configs))


if __name__ == "__main__":
    main()
