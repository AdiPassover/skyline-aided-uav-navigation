"""EXP-CONF-001 P3: ground-truth-derived per-frame targets T-A / T-B / T-C (+ the T-E mask).

Implements exactly the frozen definitions of the EXP-CONF-001 pre-registration (front half, P1,
amended 2026-08-30):

- **T-A** ``|sign * inc_rotation_deg - RTK heading increment|`` (deg). The sidecar's
  ``inc_rotation_deg`` is the frozen readout's per-frame rotation; the RTK increment is the
  frame-timestamp-interpolated unwrapped heading difference (the ``exp_vo_008`` tooling). The
  sign relation between the image-plane rotation and the compass heading is fixed by the
  conventions, not fitted: it is determined once per sequence from the correlation sign and
  DECLARED in the output (same discipline as ``rotation_diagnostic.py``). Constant heading
  offsets cancel in the increment, so the sequence-specific yaw-offset caveat does not apply.
- **T-B** angle (deg) between the estimated per-frame translation increment, rotated by the
  per-sequence global Sim(2) alignment rotation (``naveval.alignment.fit_sim2``, whole-sequence,
  declared), and the RTK position increment. Scale-free. Undefined when either increment is
  near zero (``EPS_INCREMENT_M`` on the GT side and on the scale-converted est side).
- **T-C** forward local drift: ``|| s R (p_est[k+W] - p_est[k]) - (p_gt[k+W] - p_gt[k]) ||``
  normalised by the GT path length over the window, ``W = 50`` frames (frozen).
- **T-E** operational mask: ``restart`` rows and the K = 10 frames preceding one (frozen K).

Gradability (the frozen "healthy reference population" rule): a frame is gradable iff
``success``, ``event not in {init, restart}``, and its RTK increment is valid -- both endpoint
timestamps bracketed by ``valid`` ground-truth samples. T-B additionally requires non-degenerate
increments; T-C requires the full window ahead with every frame timestamp validly bracketed.

Output: ``<out>.csv`` per frame + ``<out>.json`` with every declared parameter.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parents[2]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))

from naveval.alignment import fit_sim2, rotation_matrix          # noqa: E402
from naveval.dataset import load_dataset                         # noqa: E402
from naveval.runrecord import load_run_record                    # noqa: E402

T_C_WINDOW_FRAMES = 50     # frozen (EXP-CONF-001 T-C)
T_E_K = 10                 # frozen at P1 (EXP-CONF-001 T-E)
EPS_INCREMENT_M = 0.05     # T-B degeneracy floor, metres (declared; cruise increments are ~0.3-0.8 m)


def bracket_valid(ds, t: np.ndarray) -> np.ndarray:
    """True where t is inside the GT range and both bracketing GT samples are valid."""
    gt_t = ds.gt_timestamps
    valid = ds.gt_valid
    idx = np.searchsorted(gt_t, t, side="right")
    ok = (idx > 0) & (idx < len(gt_t))
    lo = np.clip(idx - 1, 0, len(gt_t) - 1)
    hi = np.clip(idx, 0, len(gt_t) - 1)
    return ok & valid[lo] & valid[hi]


def read_sidecar(run_dir: Path) -> dict:
    with (run_dir / "logical_transform.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "inc_rotation_deg": np.array([float(r["inc_rotation_deg"]) for r in rows]),
        "inc_proper": np.array([r["inc_proper"] == "1" for r in rows]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, help="a *-conf-v1 run dir with the sidecar")
    ap.add_argument("--out", required=True, help="output stem (writes .csv and .json)")
    a = ap.parse_args()

    ds = load_dataset(REPO / a.dataset)
    rr = load_run_record(REPO / a.run)
    sc = read_sidecar(REPO / a.run)
    n = len(rr.frame_indices)
    t = rr.timestamps_s

    events = np.array([fe.event for fe in rr.frame_estimates])
    success = rr.success

    # --- ground truth interpolated at frame timestamps -------------------------------------
    h_unwrapped = np.unwrap(np.radians(ds.gt_heading))
    gt_h = np.interp(t, ds.gt_timestamps, h_unwrapped)
    gt_e = np.interp(t, ds.gt_timestamps, ds.gt_east)
    gt_n = np.interp(t, ds.gt_timestamps, ds.gt_north)
    frame_gt_ok = bracket_valid(ds, t)

    # per-frame increments (frame k = motion k-1 -> k; frame 0 has none)
    dgt_h = np.zeros(n)
    dgt_h[1:] = np.diff(gt_h)                     # radians
    dgt_en = np.zeros((n, 2))
    dgt_en[1:, 0] = np.diff(gt_e)
    dgt_en[1:, 1] = np.diff(gt_n)
    inc_gt_ok = np.zeros(n, dtype=bool)
    inc_gt_ok[1:] = frame_gt_ok[1:] & frame_gt_ok[:-1]

    # --- gradability (frozen healthy-population rule) ---------------------------------------
    gradable_a = success & ~np.isin(events, ("init", "restart")) & inc_gt_ok

    # --- T-A ---------------------------------------------------------------------------------
    inc_rot = np.radians(sc["inc_rotation_deg"])
    corr = float(np.corrcoef(inc_rot[gradable_a], dgt_h[gradable_a])[0, 1])
    sign = 1.0 if corr > 0 else -1.0
    t_a = np.degrees(np.abs(sign * inc_rot - dgt_h))

    # --- global Sim(2) alignment (whole sequence, FR-023; declared) --------------------------
    fit_mask = frame_gt_ok
    est_pts = np.stack([rr.est_x, rr.est_y], axis=1)
    gt_pts = np.stack([gt_e, gt_n], axis=1)
    align = fit_sim2(est_pts[fit_mask], gt_pts[fit_mask])
    R = rotation_matrix(align.rotation_deg)
    s = align.scale

    # --- T-B ---------------------------------------------------------------------------------
    d_est = np.zeros((n, 2))
    d_est[1:] = np.diff(est_pts, axis=0)
    d_est_rot = d_est @ R.T
    gt_mag = np.linalg.norm(dgt_en, axis=1)
    est_mag_m = s * np.linalg.norm(d_est, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cross = d_est_rot[:, 0] * dgt_en[:, 1] - d_est_rot[:, 1] * dgt_en[:, 0]
        dot = (d_est_rot * dgt_en).sum(axis=1)
        t_b = np.degrees(np.abs(np.arctan2(cross, dot)))
    gradable_b = gradable_a & (gt_mag >= EPS_INCREMENT_M) & (est_mag_m >= EPS_INCREMENT_M)

    # --- T-C ---------------------------------------------------------------------------------
    W = T_C_WINDOW_FRAMES
    t_c = np.full(n, np.nan)
    gradable_c = np.zeros(n, dtype=bool)
    est_aligned = s * (est_pts @ R.T)
    step_len = np.zeros(n)
    step_len[1:] = np.linalg.norm(np.diff(gt_pts, axis=0), axis=1)
    cum_len = np.cumsum(step_len)
    all_ok_cum = np.cumsum(frame_gt_ok.astype(int))
    for k in range(n - W):
        # every frame timestamp in [k, k+W] validly bracketed
        if all_ok_cum[k + W] - (all_ok_cum[k - 1] if k > 0 else 0) != W + 1:
            continue
        if not gradable_a[k]:
            continue
        L = cum_len[k + W] - cum_len[k]
        if L <= 0:
            continue
        err = np.linalg.norm((est_aligned[k + W] - est_aligned[k]) - (gt_pts[k + W] - gt_pts[k]))
        t_c[k] = err / L
        gradable_c[k] = True

    # --- T-E mask ----------------------------------------------------------------------------
    restart = events == "restart"
    t_e = restart.copy()
    for i in np.flatnonzero(restart):
        t_e[max(0, i - T_E_K):i] = True

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame_index", "gradable_a", "t_a_deg", "gradable_b", "t_b_deg",
                    "gradable_c", "t_c_frac", "t_e", "event", "success"])
        for i in range(n):
            w.writerow([int(rr.frame_indices[i]),
                        int(gradable_a[i]), f"{t_a[i]:.9g}" if gradable_a[i] else "",
                        int(gradable_b[i]), f"{t_b[i]:.9g}" if gradable_b[i] else "",
                        int(gradable_c[i]), f"{t_c[i]:.9g}" if gradable_c[i] else "",
                        int(t_e[i]), events[i], str(bool(success[i])).lower()])

    decl = {
        "dataset": a.dataset, "run": a.run, "n_frames": n,
        "t_a_sign_convention": sign, "t_a_sign_correlation": corr,
        "alignment": {"rotation_deg": align.rotation_deg, "scale": align.scale,
                      "translation": list(align.translation), "n_points": align.n_points,
                      "reflection_rejected": align.reflection_rejected,
                      "near_straight": align.near_straight},
        "t_c_window_frames": W, "t_e_k": T_E_K, "eps_increment_m": EPS_INCREMENT_M,
        "counts": {"frames": n,
                   "gradable_a": int(gradable_a.sum()),
                   "gradable_b": int(gradable_b.sum()),
                   "gradable_c": int(gradable_c.sum()),
                   "t_e_frames": int(t_e.sum()),
                   "restarts": int(restart.sum()),
                   "gt_invalid_frames": int((~frame_gt_ok).sum())},
    }
    out.with_suffix(".json").write_text(json.dumps(decl, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(decl["counts"], sign=sign, align_rot_deg=align.rotation_deg,
                          align_scale=align.scale), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
