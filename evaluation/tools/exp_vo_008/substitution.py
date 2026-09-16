"""`EXP-VO-008` Phases 5 and 6d: known-answer substitution diagnostics.

Each variant replaces exactly one ingredient of the rigid readout with a ground-truth or
LiDAR-derived quantity, leaving everything else identical and recomposing from the same recorded
transforms. That isolates how much of the position error each ingredient is responsible for.

**These are diagnostics and nothing else.** Every variant that consumes GT or LiDAR height is
unusable at runtime by construction, and none is a candidate implementation. Their only job is to
put an upper bound on what fixing that one ingredient could buy, so the record can say which
ingredient the error actually lives in rather than guessing.

Variants:

| variant | what is substituted | answers |
|---|---|---|
| `baseline` | nothing (= `RIGID_MOTION`) | reference |
| `edge_rotation` | polar rotation -> legacy top-edge angle | how much is the rotation estimator worth |
| `gt_heading` | accumulated heading -> GT heading | **upper bound on fixing heading** |
| `visual_scale_applied` | XY divided by accumulated visual scale | `DEC-VO-004` alternative B, as-is |
| `lidar_scale_applied` | XY divided by LiDAR `h_k/h_0` | alternative B with a *correct* scale |
| `gt_heading_and_lidar_scale` | both | what is left when heading and height are both right |
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parents[2]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import recompose as rc                                                   # noqa: E402
from height_profile_shim import load_profile                             # noqa: E402
from naveval.dataset import load_dataset                                 # noqa: E402
from naveval.runrecord import load_run_record                            # noqa: E402
from rotation_diagnostic import gt_heading_increments                    # noqa: E402
from schedule_sweep import score                                         # noqa: E402


def integrate_with(rec: rc.Recording, width: int, height: int,
                   rotation: str = "polar",
                   heading_increments: np.ndarray | None = None,
                   scale_series: np.ndarray | None = None) -> dict:
    """Per-frame rigid integration with optional substituted heading and/or scale divisor.

    `heading_increments` replaces the estimated per-frame rotation; `scale_series[k]` divides frame
    `k`'s displacement, i.e. re-expresses it in first-frame pixel units.
    """
    n = len(rec)
    cx, cy = width / 2.0, height / 2.0
    Ginv = np.linalg.inv(rec.G)
    x = np.zeros(n)
    y = np.zeros(n)
    yaw = np.zeros(n)
    Tx = Ty = theta = 0.0
    for k in range(1, n):
        S = rec.G[k - 1] @ Ginv[k]
        qx, qy = rc._apply(S, cx, cy)
        qx -= cx
        qy -= cy
        if heading_increments is not None:
            dth = float(heading_increments[k])
        elif rotation == "polar":
            dth = rc.polar_rotation(rc._jacobian(S, cx, cy))
        else:
            dth = rc.edge_angle(S, width, height)
        s = float(scale_series[k]) if scale_series is not None else 1.0
        c, si = math.cos(theta), math.sin(theta)
        Tx += (c * qx - si * qy) / s
        Ty += (si * qx + c * qy) / s
        theta += dth
        x[k], y[k], yaw[k] = Tx, Ty, theta
    return {"x": x, "y": -y, "yaw_deg": np.degrees(yaw) % 360.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--height-profile", default="")
    ap.add_argument("--scene", default="")
    ap.add_argument("--width", type=int, default=1224)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = load_dataset(REPO / a.dataset)
    run = REPO / a.run
    rec = rc.load_recording(run)
    rr = load_run_record(run)
    t = rr.timestamps_s
    n = len(rec)

    dgt = gt_heading_increments(ds, t)
    # BUGFIX 2026-08-27 (EXP-VO-010): this probe used only the first 199 frames. On
    # `hkairport01-b` the aircraft turns by -1.00 deg over that prefix, so the correlation there is
    # pure noise (measured -0.043 to +0.044, flipping sign between otherwise-identical arms) while
    # over the whole run it is +0.86. A coin-flip sign silently negates the GT heading and makes the
    # `gt_heading` variant report ~6.6 % normalised ATE against a ~1.0 % baseline. Determined over
    # the WHOLE run now, and refused rather than guessed when the correlation is not decisive.
    probe = np.array([rc.polar_rotation(rc._jacobian(
        rec.G[k - 1] @ np.linalg.inv(rec.G[k]), a.width / 2, a.height / 2))
        for k in range(1, len(rec))])
    corr = float(np.corrcoef(probe, dgt[1:])[0, 1])
    if abs(corr) < 0.2:
        raise SystemExit(
            f"refusing to guess the rotation sign convention: correlation with GT heading is "
            f"{corr:+.4f} over {len(rec) - 1} frames, which is not decisive. A sequence with too "
            f"little net turn cannot fix the convention, and guessing it wrong inverts every "
            f"heading-substituted variant.")
    sign = 1.0 if corr > 0 else -1.0
    gt_inc = sign * dgt                       # in the estimator's rotation convention

    # accumulated visual scale, from the run's own diagnostics
    import csv
    with (run / "logical_transform.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    visual = np.array([float(r["rigid_accum_scale"]) for r in rows])

    variants = {
        "baseline": dict(),
        "edge_rotation": dict(rotation="edge"),
        "gt_heading": dict(heading_increments=gt_inc),
        "visual_scale_applied": dict(scale_series=visual),
        "gt_heading_visual_scale": dict(heading_increments=gt_inc, scale_series=visual),
    }

    if a.height_profile:
        prof = load_profile(a.height_profile, a.scene or None)
        hp = np.interp(t, np.asarray(prof["t"], dtype=float),
                       np.asarray(prof["height_m"], dtype=float))
        lidar_scale = hp[0] / hp        # pixels shrink as height grows: units relative to frame 0
        variants["lidar_scale_applied"] = dict(scale_series=lidar_scale)
        variants["gt_heading_lidar_scale"] = dict(heading_increments=gt_inc,
                                                  scale_series=lidar_scale)
        agree = {
            "visual_accum_end": float(visual[-1]),
            "lidar_scale_end": float(lidar_scale[-1]),
            "lidar_scale_min": float(lidar_scale.min()),
            "lidar_scale_max": float(lidar_scale.max()),
            "visual_over_lidar_end": float(visual[-1] / lidar_scale[-1]),
            "corr_visual_vs_lidar": float(np.corrcoef(np.log(visual), np.log(lidar_scale))[0, 1]),
        }
    else:
        agree = None

    out = {"label": a.label, "n_frames": n, "gt_sign_convention": sign,
           "scale_agreement": agree, "variants": {}}
    for name, kw in variants.items():
        v = integrate_with(rec, a.width, a.height, **kw)
        s = score(ds, t, v["x"], v["y"], v["yaw_deg"])
        out["variants"][name] = {k: val for k, val in s.items() if not k.startswith("_")}
        print(f"  {name:28s} normATE {100 * s['ate_rmse_normalised']:7.3f} %   "
              f"rpe50 {s['rpe_50m']:6.2f}   endpoint {s['endpoint_error_m']:7.1f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    if agree:
        print("\nscale agreement:", json.dumps(agree, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
