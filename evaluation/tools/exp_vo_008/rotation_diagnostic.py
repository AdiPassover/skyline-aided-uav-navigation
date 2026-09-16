"""`EXP-VO-008` Phase 6: which per-frame rotation estimator is right, measured against GT heading.

The schedule sweep localised the `AMtown01` affine reversal to the **rotation estimator**, not to
reduction granularity: at identical granularity, swapping the polar rotation for the legacy top-edge
angle moves normalised ATE from 2.429 % to 0.787 %, and reproduces the live legacy figure to
0.002 pp at the legacy granularity. This script asks the direct question the sweep cannot: *which
one tracks the aircraft's actual heading?*

Both estimators are legitimate and they agree exactly on a similarity. For an affine
`A = R(theta) P` with `P` symmetric positive definite,

    polar rotation  atan2(a21 - a12, a11 + a22) = theta                        (exact)
    top-edge angle  atan2(a21, a11)             = theta + atan2(q, p)

where `q` is `P`'s off-diagonal — the shear term. So the two differ by the *stretch orientation*, and
which is closer to the true camera rotation depends on whether the estimator's non-rigid residual is
a real stretch of the scene (polar right) or an artefact that also rotates the fitted frame (edge
possibly closer). Nothing in `LIT-VO-003` or `DEC-VO-004` settles that empirically; this does.

Ground-truth heading is used **to measure**, never to choose: no parameter is fitted here, and both
estimators are pre-existing, documented formulations (`COMP-001` section 3.6 for the edge angle,
`DEC-VO-004` for the polar).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parents[2]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import recompose as rc                                                   # noqa: E402
from naveval.dataset import load_dataset                                 # noqa: E402
from naveval.runrecord import load_run_record                            # noqa: E402


def per_frame_rotations(rec: rc.Recording, width: int, height: int) -> dict:
    n = len(rec)
    cx, cy = width / 2.0, height / 2.0
    Ginv = np.linalg.inv(rec.G)
    pol = np.zeros(n)
    edg = np.zeros(n)
    shear = np.zeros(n)      # P's off-diagonal / its mean diagonal, the quantity that separates them
    for k in range(1, n):
        S = rec.G[k - 1] @ Ginv[k]
        J = rc._jacobian(S, cx, cy)
        pol[k] = rc.polar_rotation(J)
        edg[k] = rc.edge_angle(S, width, height)
        # symmetric (stretch) factor P = R^T J
        c, s = np.cos(pol[k]), np.sin(pol[k])
        R = np.array([[c, -s], [s, c]])
        P = R.T @ J
        shear[k] = 0.5 * (P[0, 1] + P[1, 0]) / (0.5 * (P[0, 0] + P[1, 1]))
    return {"polar": pol, "edge": edg, "shear": shear}


def gt_heading_increments(ds, timestamps: np.ndarray) -> np.ndarray:
    """GT heading change per frame, unwrapped, in radians."""
    h = np.unwrap(np.radians(np.asarray(ds.gt_heading)))
    hi = np.interp(timestamps, np.asarray(ds.gt_timestamps), h)
    d = np.zeros_like(hi)
    d[1:] = np.diff(hi)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, help="a *-rigid-v1 run dir with the sidecar")
    ap.add_argument("--label", required=True)
    ap.add_argument("--width", type=int, default=1224)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = load_dataset(REPO / a.dataset)
    run = REPO / a.run
    rec = rc.load_recording(run)
    rr = load_run_record(run)
    r = per_frame_rotations(rec, a.width, a.height)
    dgt = gt_heading_increments(ds, rr.timestamps_s)

    # The estimator turns in image coordinates (y down, CCW positive); GT heading is compass
    # (CW from north). The sign relation is fixed by the conventions, not fitted: determine it once
    # from the sign of the correlation and report it, so a convention slip cannot hide here.
    sign = 1.0 if np.corrcoef(r["polar"][1:], dgt[1:])[0, 1] > 0 else -1.0

    out: dict = {"label": a.label, "n_frames": len(rec), "gt_sign_convention": sign,
                 "total_gt_turn_deg": float(np.degrees(dgt.sum()))}
    for name in ("polar", "edge"):
        inc = sign * r[name]
        err = inc[1:] - dgt[1:]
        out[name] = {
            "mean_increment_error_deg": float(np.degrees(err.mean())),
            "sd_increment_error_deg": float(np.degrees(err.std(ddof=1))),
            "t_of_mean": float(err.mean() / (err.std(ddof=1) / np.sqrt(err.size))),
            "accumulated_error_deg": float(np.degrees(err.sum())),
            "total_turn_deg": float(np.degrees(inc.sum())),
            "corr_with_gt_increment": float(np.corrcoef(inc[1:], dgt[1:])[0, 1]),
            "rms_increment_error_deg": float(np.degrees(np.sqrt((err ** 2).mean()))),
        }
    d = sign * (r["edge"] - r["polar"])
    out["edge_minus_polar"] = {
        "mean_deg": float(np.degrees(d[1:].mean())),
        "sd_deg": float(np.degrees(d[1:].std(ddof=1))),
        "accumulated_deg": float(np.degrees(d[1:].sum())),
    }
    out["shear_q_over_p"] = {
        "mean": float(r["shear"][1:].mean()), "sd": float(r["shear"][1:].std(ddof=1)),
        "t_of_mean": float(r["shear"][1:].mean()
                           / (r["shear"][1:].std(ddof=1) / np.sqrt(len(rec) - 1))),
    }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    np.savez(str(Path(a.out).with_suffix(".npz")),
             polar=r["polar"], edge=r["edge"], shear=r["shear"], dgt=dgt,
             t=rr.timestamps_s, sign=sign)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
