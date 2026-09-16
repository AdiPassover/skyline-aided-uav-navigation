"""EXP-VO-004: the estimator's internal scale state, read directly from the per-frame logical
transform (`logical_transform.csv`, DEC-VO-003 + VoRunnerConfig.logical_transform_sidecar), compared
with the physically predicted scale from ground-truth altitude/attitude, and re-composed under
pre-declared scale-normalisation schedules (Phase 5: what the legacy reset was doing).

Every observable is defined in `LIT-VO-003`; every hypothesis and schedule in the EXP-VO-004
pre-registration, fixed before this ran. `naveval` is imported read-only; the VO is never
re-run here.

Conventions. Sidecar rows carry G_k : L -> C_k in homography form (row-major g00..g22), with
column-vector matrices: x_C ~ G x_L. M_k = G_k^-1 : C_k -> L. Image centre c = (W/2, H/2) in the
downsampled working frame. Footprint magnification at the principal point
m_k = sqrt(|det J_M(c)|) (logical px per current px); lambda = log m. Under nadir-planar
assumptions m_k = h_k / h_0 (LIT-VO-003 eq. 3).

Usage::

    python evaluation/tools/exp_vo_004/internal_scale.py --out evaluations/exp-vo-004
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parents[2]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))

from naveval.alignment import fit_sim2                       # noqa: E402
from naveval.dataset import load_dataset                     # noqa: E402
from naveval.metrics import absolute_trajectory_error, path_length, yaw_error_metric  # noqa: E402
from naveval.runrecord import load_run_record                # noqa: E402
from naveval.sync import synchronize                         # noqa: E402

# ------------------------------------------------------------------ pre-registered constants
RUNS = {
    ("a", "homography"): "runs/hkairport01-a-homography-logical-v1",
    ("a", "affine"): "runs/hkairport01-a-affine-logical-v1",
    ("b", "homography"): "runs/hkairport01-b-homography-logical-v1",
    ("b", "affine"): "runs/hkairport01-b-affine-logical-v1",
}
BASELINES = {
    ("a", "homography"): "runs/hkairport01-a-run-v1",
    ("a", "affine"): "runs/hkairport01-a-affine-v1",
    ("b", "homography"): "runs/hkairport01-b-run-v1",
    ("b", "affine"): "runs/hkairport01-b-affine-v1",
}
DATASETS = {"a": "datasets/hkairport01-a", "b": "datasets/hkairport01-b"}
WORKING_W, WORKING_H = 1224, 1024
W_EDGE, H_EDGE = WORKING_W, WORKING_H       # frame span used by the se2_edge / legacy emulation variants          # EXP-002 working resolution (2x box downsample)
PHYSICAL_BAND = (0.90, 1.10)
KEYFRAME_PROXY_RISE = 0.20                 # track_count rises >= 20 % over previous frame
TURN_RATE_DPS = 5.0
LAG_RANGE = (1, 500)
SCHEDULE_PERIODS = (1, 2, 5, 10, 25, 50, 100, 200, 400, 800)
VARIANTS = ("scale_only", "se2", "se2_edge", "legacy")
# se2_edge: SE(2) reduction with rotation from the top edge, position from the centre.
# legacy:   the same, but position from the CENTROID of the four corners (COMP-001 §3.6 verbatim).


# ------------------------------------------------------------------ IO
def read_csv(p: Path) -> list[dict]:
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def load_sidecar(run_dir: Path) -> tuple[np.ndarray, list[str]]:
    rows = read_csv(run_dir / "logical_transform.csv")
    G = np.array([[float(r[f"g{i}{j}"]) for i in range(3) for j in range(3)] for r in rows]).reshape(-1, 3, 3)
    events = [r["event"] for r in rows]
    return G, events


# ------------------------------------------------------------------ observables
def jacobian_at(M: np.ndarray, x: float, y: float) -> np.ndarray:
    """2x2 Jacobian of the projective map M (3x3, column vectors) at pixel (x, y)."""
    p = np.array([x, y, 1.0])
    q = M @ p
    w = q[2]
    u, v = q[0] / w, q[1] / w
    J = np.array([
        [M[0, 0] - u * M[2, 0], M[0, 1] - u * M[2, 1]],
        [M[1, 0] - v * M[2, 0], M[1, 1] - v * M[2, 1]],
    ]) / w
    return J


def apply(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    q = M @ np.array([x, y, 1.0])
    return q[0] / q[2], q[1] / q[2]


def polar_rotation_deg(J: np.ndarray) -> float:
    """Rotation angle of the polar factor R of J = R P (nearest rotation in Frobenius norm)."""
    U, _, Vt = np.linalg.svd(J)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def observables(G: np.ndarray, W: int, H: int) -> dict:
    """Per-frame scale observables from G_k : L -> C_k."""
    M = np.linalg.inv(G)
    cx, cy = W / 2.0, H / 2.0
    J = jacobian_at(M, cx, cy)
    det = np.linalg.det(J)
    sv = np.linalg.svd(J, compute_uv=False)
    corners = [apply(M, 0, 0), apply(M, W, 0), apply(M, W, H), apply(M, 0, H)]
    area = 0.5 * abs(sum(corners[i][0] * corners[(i + 1) % 4][1] - corners[(i + 1) % 4][0] * corners[i][1]
                         for i in range(4)))
    qx, qy = apply(M, cx, cy)
    ax, ay = corners[0]
    bx, by = corners[1]
    edge_yaw = (math.degrees(math.atan2(by - ay, bx - ax)) + 360.0) % 360.0
    Gn = G / G[2, 2]
    return {
        "sigma1": float(sv[0]), "sigma2": float(sv[1]),
        "m_centre": math.sqrt(abs(det)),
        "det_sign": 1.0 if det > 0 else -1.0,
        "m_area": math.sqrt(area / (W * H)),
        "anisotropy": float(sv[0] / sv[1]) if sv[1] > 0 else float("inf"),
        "perspective": max(abs(Gn[2, 0]) * W, abs(Gn[2, 1]) * H),
        "logical_x": qx - cx, "logical_y": -(qy - cy),
        "yaw_edge_deg": edge_yaw,
        "yaw_polar_deg": (polar_rotation_deg(J) + 360.0) % 360.0,
    }


def increment_observables(G_prev: np.ndarray, G_cur: np.ndarray, W: int, H: int) -> dict:
    """Per-frame relative transform D = G_cur G_prev^-1 (C_{k-1} -> C_k): its centre scale, and its
    stretch ALONG and ACROSS its own translation direction (the image-flow direction). A
    systematic along/across asymmetry is the signature of a tracking bias tied to the flow."""
    D = G_cur @ np.linalg.inv(G_prev)
    cx, cy = W / 2.0, H / 2.0
    J = jacobian_at(D, cx, cy)
    tx, ty = apply(D, cx, cy)
    t = np.array([tx - cx, ty - cy])
    nrm = np.linalg.norm(t)
    if nrm < 1e-9:
        return {"inc_scale": math.sqrt(abs(np.linalg.det(J))), "inc_along": float("nan"),
                "inc_across": float("nan"), "inc_flow_px": 0.0, "inc_rot_deg": polar_rotation_deg(J)}
    u = t / nrm
    v = np.array([-u[1], u[0]])
    return {"inc_scale": math.sqrt(abs(np.linalg.det(J))), "inc_along": float(np.linalg.norm(J @ u)),
            "inc_across": float(np.linalg.norm(J @ v)), "inc_flow_px": float(nrm),
            "inc_rot_deg": polar_rotation_deg(J)}


# ------------------------------------------------------------------ ground truth
def physical_scale(dataset, frame_ts: np.ndarray, attitude: list[dict]) -> dict:
    """m_phys,k = (h_k/h_0) * (cos th_0 / cos th_k)^1.5 from GT altitude and tilt, interpolated to
    frame timestamps. Also heading rate for the turn mask."""
    gt_t = dataset.gt_timestamps[dataset.gt_valid]
    gt_up = dataset.gt_up[dataset.gt_valid]
    h = np.interp(frame_ts, gt_t, gt_up)
    if attitude:
        at = np.array([float(r["timestamp_s"]) for r in attitude])
        tilt = np.array([float(r["tilt_deg"]) for r in attitude])
        th = np.interp(frame_ts, at, tilt)
    else:
        th = np.zeros_like(frame_ts)
    ct = np.cos(np.radians(th))
    m_phys = (h / h[0]) * (ct[0] / ct) ** 1.5
    heading = np.interp(frame_ts, gt_t, np.unwrap(np.radians(dataset.gt_heading[dataset.gt_valid])))
    rate = np.gradient(np.degrees(heading), frame_ts)
    return {"h": h, "tilt": th, "m_phys": m_phys, "heading_rate_dps": rate}


# ------------------------------------------------------------------ statistics
def lag_variance_exponent(lam: np.ndarray, lo: int, hi: int) -> dict:
    """Log-log slope of the MEAN-SQUARE lag increment E[(lam_{k+tau} - lam_k)^2] against tau:
    ~1 for a random walk (diffusion), ~2 for a deterministic drift. (The plain variance of the
    increment would be ~0 for a pure drift, which is why the mean square is used.)"""
    lags = np.unique(np.geomspace(lo, min(hi, len(lam) // 4), 25).astype(int))
    v = np.array([np.mean((lam[l:] - lam[:-l]) ** 2) for l in lags])
    ok = v > 0
    slope = float(np.polyfit(np.log(lags[ok]), np.log(v[ok]), 1)[0])
    return {"lags": lags.tolist(), "variance": v.tolist(), "log_log_slope": slope}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def r2_linear(y: np.ndarray, X: np.ndarray) -> float:
    A = np.column_stack([X, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum(resid ** 2) / ss) if ss > 0 else float("nan")


# ------------------------------------------------------------------ phase 5: re-composition
class DegenerateTransform(Exception):
    """Composed transform singular at the image centre; normalisation undefined."""


LAST_DEGENERATE_COUNT = 0

def normalise(M: np.ndarray, cx: float, cy: float, variant: str) -> np.ndarray:
    """Return M' : C -> L with unit scale (and, for 'se2', pure rigid form) at the image centre,
    mapping the centre to the same logical point as M (no position jump)."""
    J = jacobian_at(M, cx, cy)
    qx, qy = apply(M, cx, cy)
    if not np.all(np.isfinite(J)) or abs(np.linalg.det(J)) < 1e-18:
        # The composed projective map has degenerated at the image centre (homography only, in
        # practice): no scale can be read there, so the normalisation is undefined. Counted by the
        # caller and reported; the running transform is left as it is.
        raise DegenerateTransform()
    if variant == "scale_only":
        m = math.sqrt(abs(np.linalg.det(J)))
        Z = np.array([[1 / m, 0, cx - cx / m], [0, 1 / m, cy - cy / m], [0, 0, 1]])   # scale about c
        return M @ Z
    if variant in ("se2", "se2_edge", "legacy"):
        if variant == "se2":
            th = math.radians(polar_rotation_deg(J))
        else:
            ax, ay = apply(M, 0, 0); bx, by = apply(M, W_EDGE, 0)
            th = math.atan2(by - ay, bx - ax)
        if variant == "legacy":
            cs = [apply(M, 0, 0), apply(M, W_EDGE, 0), apply(M, W_EDGE, H_EDGE), apply(M, 0, H_EDGE)]
            qx = sum(c[0] for c in cs) / 4.0; qy = sum(c[1] for c in cs) / 4.0
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        Mn = np.eye(3)
        Mn[:2, :2] = R
        Mn[:2, 2] = np.array([qx, qy]) - R @ np.array([cx, cy])
        return Mn
    raise ValueError(variant)


def recompose(G: np.ndarray, events: list[str], W: int, H: int, schedule, variant: str,
              drop_restart_increments: bool = False) -> np.ndarray:
    """Re-integrate the per-frame relative transforms D_k = G_k G_{k-1}^-1 (C_{k-1} -> C_k), applying
    `normalise` to the running M whenever `schedule(k, event)` is true. Returns (n, 4): logical
    x, y (pose convention), edge yaw, centre magnification.

    `drop_restart_increments`: treat D_k = I on restart frames. BoofCV's ImageMotionPointTrackerKey
    updates worldToCurr BEFORE StitchingFromMotion2D.checkLargeMotion can reject the frame, so the
    estimator's accumulated transform on a restart frame already contains the rejected estimate;
    the legacy readout never published it, the DEC-VO-003 fold did. This flag reproduces the
    legacy treatment (discard) for a like-for-like comparison."""
    global LAST_DEGENERATE_COUNT
    cx, cy = W / 2.0, H / 2.0
    n = G.shape[0]
    M = np.linalg.inv(G[0])
    out = np.zeros((n, 4))
    degenerate = 0
    for k in range(n):
        if k > 0 and not (drop_restart_increments and events[k] == "restart"):
            D = G[k] @ np.linalg.inv(G[k - 1])
            M = M @ np.linalg.inv(D)              # M_k = M_{k-1} ∘ D_k^-1
            M = M / M[2, 2]
        if schedule(k, events[k]):
            try:
                M = normalise(M, cx, cy, variant)
            except DegenerateTransform:
                degenerate += 1
        if variant == "legacy":
            cs = [apply(M, 0, 0), apply(M, W, 0), apply(M, W, H), apply(M, 0, H)]
            qx = sum(c[0] for c in cs) / 4.0; qy = sum(c[1] for c in cs) / 4.0
        else:
            qx, qy = apply(M, cx, cy)
        J = jacobian_at(M, cx, cy)
        ax, ay = apply(M, 0, 0); bx, by = apply(M, W, 0)
        out[k] = [qx - cx, -(qy - cy), (math.degrees(math.atan2(by - ay, bx - ax)) + 360) % 360,
                  math.sqrt(abs(np.linalg.det(J)))]
    LAST_DEGENERATE_COUNT = degenerate
    return out


def evaluate_trajectory(est_xy: np.ndarray, est_yaw: np.ndarray, run_record, dataset) -> dict:
    gt_heading = dataset.gt_heading if dataset.has_heading else None
    sync = synchronize(run_record.timestamps_s, dataset.gt_timestamps, dataset.gt_east, dataset.gt_north,
                       gt_heading, dataset.gt_valid, clock_offset_s=dataset.clock_offset_s,
                       clock_drift_s_per_s=dataset.clock_drift_s_per_s)
    idx = sync.frame_indices
    est = est_xy[idx]
    gt = np.stack([sync.east, sync.north], axis=1)
    al = fit_sim2(est, gt)
    aligned = al.apply(est)
    ate = absolute_trajectory_error(aligned, gt)
    out = {"ate_rmse_m": ate["rmse"], "ate_rmse_normalised": ate["rmse"] / path_length(gt),
           "alignment_scale": al.scale, "alignment_rotation_deg": al.rotation_deg, "n_points": int(idx.size)}
    if gt_heading is not None and sync.has_heading:
        y = yaw_error_metric(al.transform_yaw(est_yaw[idx]), sync.heading)
        out["yaw_rmse_deg"] = y["rmse"]
    return out


# ------------------------------------------------------------------ driver
def analyse_run(key, out: Path, replicate_check: bool = True) -> dict:
    exe, model = key
    run_dir = REPO / RUNS[key]
    base_dir = REPO / BASELINES[key]
    ds_dir = REPO / DATASETS[exe]
    G_raw, events = load_sidecar(run_dir)
    # Effective logical transform with restart-frame increments dropped (see `recompose`): BoofCV
    # updates worldToCurr before checkLargeMotion rejects a frame, so the raw sidecar folds a
    # rejected estimate at every restart. Observables are read from the corrected composition; the
    # raw end value is kept in the summary for the record.
    G = np.empty_like(G_raw); G[0] = G_raw[0]
    for k in range(1, G_raw.shape[0]):
        D = np.eye(3) if events[k] == "restart" else G_raw[k] @ np.linalg.inv(G_raw[k - 1])
        G[k] = D @ G[k - 1]; G[k] /= G[k][2, 2]
    frames = read_csv(run_dir / "frames.csv")
    base = read_csv(base_dir / "frames.csv")
    rr = load_run_record(run_dir)
    ds = load_dataset(ds_dir)
    att_path = ds_dir / "attitude.csv"
    attitude = read_csv(att_path) if att_path.exists() else []
    n = G.shape[0]
    W, H = WORKING_W, WORKING_H

    # R0: estimator untouched
    hk = [f"h{i}{j}" for i in range(3) for j in range(3)] + ["event", "reference_id", "track_count", "inlier_count", "success"]
    mismatches = sum(1 for x, y in zip(frames, base) if any(x[c] != y[c] for c in hk))

    obs = [observables(G[k], W, H) for k in range(n)]
    m = np.array([o["m_centre"] for o in obs]); lam = np.log(m)
    m_area = np.array([o["m_area"] for o in obs])
    aniso = np.array([o["anisotropy"] for o in obs])
    persp = np.array([o["perspective"] for o in obs])
    det_sign = np.array([o["det_sign"] for o in obs])
    ts = np.array([float(r["timestamp_s"]) for r in frames])
    tracks = np.array([int(r["track_count"]) for r in frames])
    ev = np.array(events)
    phys = physical_scale(ds, ts, attitude)
    dlam = np.diff(lam, prepend=lam[0])
    inc = [increment_observables(G[k - 1], G[k], W, H) if k > 0 else
           {"inc_scale": 1.0, "inc_along": float("nan"), "inc_across": float("nan"), "inc_flow_px": 0.0, "inc_rot_deg": 0.0}
           for k in range(n)]
    inc_along = np.array([i["inc_along"] for i in inc]); inc_across = np.array([i["inc_across"] for i in inc])
    inc_scale = np.array([i["inc_scale"] for i in inc]); inc_flow = np.array([i["inc_flow_px"] for i in inc])

    # sidecar pose consistency with frames.csv (LOGICAL_FRAME published est_x/est_y)
    est_x = np.array([float(r["est_x"]) for r in frames]); est_y = np.array([float(r["est_y"]) for r in frames])
    lx = np.array([o["logical_x"] for o in obs]); ly = np.array([o["logical_y"] for o in obs])
    raw_obs = [observables(G_raw[k], W, H) for k in range(n)]
    pose_consistency_px = float(np.max(np.hypot(est_x - np.array([o["logical_x"] for o in raw_obs]),
                                                est_y - np.array([o["logical_y"] for o in raw_obs]))))

    # masks
    keyframe_proxy = np.zeros(n, bool)
    keyframe_proxy[1:] = tracks[1:] >= (1 + KEYFRAME_PROXY_RISE) * np.maximum(tracks[:-1], 1)
    reorigin = ev == "recenter"; restart = ev == "restart"
    turn = np.abs(phys["heading_rate_dps"]) > TURN_RATE_DPS
    terc = np.percentile(tracks, [33.3, 66.7])
    low = tracks <= terc[0]; high = tracks >= terc[1]
    normal = ~(reorigin | restart | keyframe_proxy)
    normal[0] = False

    # epochs (between reference changes)
    ref = np.array([int(r["reference_id"]) for r in frames])
    epochs = []
    for rid in np.unique(ref):
        idx = np.where(ref == rid)[0]
        epochs.append({"reference_id": int(rid), "start": int(idx[0]), "end": int(idx[-1]),
                       "n_frames": int(idx.size), "dlam": float(lam[idx[-1]] - lam[idx[0]]),
                       "mean_tracks": float(tracks[idx].mean())})
    ep_len = np.array([e["n_frames"] for e in epochs]); ep_d = np.array([e["dlam"] for e in epochs])

    # H1
    outside = np.where((m < PHYSICAL_BAND[0]) | (m > PHYSICAL_BAND[1]))[0]
    first_exit = int(outside[0]) if outside.size else None
    inside_after_exit = bool(((m[first_exit:] >= PHYSICAL_BAND[0]) & (m[first_exit:] <= PHYSICAL_BAND[1])).any()) if first_exit is not None else None
    X = np.column_stack([phys["h"], phys["tilt"]])
    r2 = r2_linear(lam, X)
    # H2
    d = dlam[1:]
    t_stat = float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size)))
    lagv = lag_variance_exponent(lam, *LAG_RANGE)
    ac1 = float(np.corrcoef(d[:-1], d[1:])[0, 1])
    # H3
    rho_len = spearman(ep_len, ep_d) if len(epochs) > 3 else float("nan")
    med = lambda mask: float(np.median(np.abs(dlam[mask]))) if mask.any() else float("nan")  # noqa: E731
    # H4/H5
    summary = {
        "run": RUNS[key], "model": model, "execution": exe, "n_frames": n,
        "R0_estimator_mismatches_vs_baseline": mismatches,
        "R0_sidecar_pose_vs_frames_max_px": pose_consistency_px,
        "raw_sidecar_including_restart_folds": {
            "m_end": float(observables(G_raw[-1], W, H)["m_centre"]),
            "n_restarts": int(sum(1 for e in events if e == "restart"))},
        "H1": {"m_end": float(m[-1]), "lambda_end": float(lam[-1]), "m_min": float(m.min()), "m_max": float(m.max()),
               "m_phys_min": float(phys["m_phys"].min()), "m_phys_max": float(phys["m_phys"].max()),
               "first_exit_frame": first_exit, "returns_inside_after_exit": inside_after_exit,
               "r2_lambda_vs_altitude_tilt": r2,
               "corr_lambda_altitude": float(np.corrcoef(lam, phys["h"])[0, 1]),
               "corr_lambda_tilt": float(np.corrcoef(lam, phys["tilt"])[0, 1]),
               "det_negative_frames": int((det_sign < 0).sum())},
        "H2": {"mean_dlam_per_frame": float(d.mean()), "sd_dlam_per_frame": float(d.std(ddof=1)),
               "t_stat": t_stat, "lag1_autocorr": ac1, "lag_variance_log_log_slope": lagv["log_log_slope"],
               "predicted_mean_dlam_from_dec_vo_003": math.log(1 / 0.170) / 6158,
               "lambda_end_over_frames": float(lam[-1] / n)},
        "H3": {"n_epochs": len(epochs), "spearman_epoch_length_vs_dlam": rho_len,
               "max_abs_dlam_at_reorigin": float(np.max(np.abs(dlam[reorigin]))) if reorigin.any() else 0.0,
               "max_abs_dlam_at_restart": float(np.max(np.abs(dlam[restart]))) if restart.any() else 0.0,
               "median_abs_dlam_keyframe_proxy": med(keyframe_proxy), "median_abs_dlam_normal": med(normal),
               "n_keyframe_proxy": int(keyframe_proxy.sum()), "n_reorigin": int(reorigin.sum()), "n_restart": int(restart.sum()),
               "mean_dlam_keyframe_proxy": float(dlam[keyframe_proxy].mean()) if keyframe_proxy.any() else float("nan"),
               "mean_dlam_normal": float(dlam[normal].mean())},
        "H4": {"median_abs_dlam_low_support": med(low & normal), "median_abs_dlam_high_support": med(high & normal),
               "mean_dlam_low_support": float(dlam[low & normal].mean()), "mean_dlam_high_support": float(dlam[high & normal].mean()),
               "median_abs_dlam_turn": med(turn & normal), "median_abs_dlam_straight": med(~turn & normal),
               "mean_dlam_turn": float(dlam[turn & normal].mean()) if (turn & normal).any() else float("nan"),
               "mean_dlam_straight": float(dlam[~turn & normal].mean()),
               "turn_frames": int(turn.sum()),
               "anisotropy_p50": float(np.median(aniso)), "anisotropy_p95": float(np.percentile(aniso, 95)),
               "anisotropy_frac_gt_1p10": float((aniso > 1.10).mean())},
        "increments": {
            "mean_log_inc_scale": float(np.nanmean(np.log(inc_scale[1:]))),
            "mean_log_along": float(np.nanmean(np.log(inc_along[1:]))),
            "mean_log_across": float(np.nanmean(np.log(inc_across[1:]))),
            "t_log_along": float(np.nanmean(np.log(inc_along[1:])) / (np.nanstd(np.log(inc_along[1:]), ddof=1) / math.sqrt(np.isfinite(inc_along[1:]).sum()))),
            "t_log_across": float(np.nanmean(np.log(inc_across[1:])) / (np.nanstd(np.log(inc_across[1:]), ddof=1) / math.sqrt(np.isfinite(inc_across[1:]).sum()))),
            "median_flow_px": float(np.median(inc_flow[1:])),
            "sd_log_inc_scale": float(np.nanstd(np.log(inc_scale[1:]), ddof=1)),
        },
        "H5": {"perspective_max": float(persp.max()), "perspective_p95": float(np.percentile(persp, 95)),
               "m_area_over_m_centre_p95": float(np.percentile(m_area / m, 95)),
               "m_area_over_m_centre_p05": float(np.percentile(m_area / m, 5))},
        "epochs": epochs,
    }

    # per-frame table
    out.mkdir(parents=True, exist_ok=True)
    with (out / "internal_scale.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "timestamp_s", "event", "reference_id", "track_count", "keyframe_proxy",
                    "m_centre", "lambda", "dlambda", "m_area", "anisotropy", "perspective", "det_sign",
                    "m_phys", "gt_up_m", "gt_tilt_deg", "gt_heading_rate_dps", "logical_x", "logical_y",
                    "yaw_edge_deg", "yaw_polar_deg", "sigma1", "sigma2",
                    "inc_scale", "inc_along", "inc_across", "inc_flow_px", "inc_rot_deg"])
        for k in range(n):
            o = obs[k]
            w.writerow([k, ts[k], ev[k], ref[k], tracks[k], int(keyframe_proxy[k]), o["m_centre"], lam[k], dlam[k],
                        o["m_area"], o["anisotropy"], o["perspective"], o["det_sign"], phys["m_phys"][k],
                        phys["h"][k], phys["tilt"][k], phys["heading_rate_dps"][k], o["logical_x"], o["logical_y"],
                        o["yaw_edge_deg"], o["yaw_polar_deg"], o["sigma1"], o["sigma2"],
                        inc[k]["inc_scale"], inc[k]["inc_along"], inc[k]["inc_across"], inc[k]["inc_flow_px"], inc[k]["inc_rot_deg"]])

    # ---------------------------------------------------------- Phase 5 schedules
    sched = {}
    legacy_x = np.array([float(r["est_x"]) for r in base]); legacy_y = np.array([float(r["est_y"]) for r in base])
    legacy_yaw = np.array([float(r["est_yaw_deg"]) for r in base])
    sched["legacy_committed"] = evaluate_trajectory(np.stack([legacy_x, legacy_y], 1), legacy_yaw, rr, ds)
    rlx = np.array([o["logical_x"] for o in raw_obs]); rly = np.array([o["logical_y"] for o in raw_obs])
    sched["logical_none_as_published"] = evaluate_trajectory(np.stack([rlx, rly], 1), np.array([o["yaw_edge_deg"] for o in raw_obs]), rr, ds)
    sched["logical_none_polar_yaw"] = evaluate_trajectory(np.stack([lx, ly], 1), np.array([o["yaw_polar_deg"] for o in obs]), rr, ds)
    nd = recompose(G_raw, events, W, H, lambda k, e: False, "se2", drop_restart_increments=True)
    r = evaluate_trajectory(nd[:, :2], nd[:, 2], rr, ds)
    r["restart_increment_px"] = [float(np.hypot(*(np.array(apply(G_raw[k] @ np.linalg.inv(G_raw[k - 1]), W / 2, H / 2)) - np.array([W / 2, H / 2]))))
                                 for k in range(1, n) if events[k] == "restart"]
    r["restart_centroid_jump_px"] = [float(np.hypot(*(np.mean([apply(np.linalg.inv(G_raw[k] @ np.linalg.inv(G_raw[k - 1])), x, y) for x, y in ((0, 0), (W, 0), (W, H), (0, H))], axis=0) - np.array([W / 2, H / 2]))))
                                     for k in range(1, n) if events[k] == "restart"]
    sched["logical_none_restart_dropped"] = r
    for variant in VARIANTS:
        # recorded re-origin/restart schedule (H6a) -- restart increments dropped, as legacy did
        rec = recompose(G, events, W, H, lambda k, e: e in ("recenter", "restart"), variant, drop_restart_increments=True)
        r = evaluate_trajectory(rec[:, :2], rec[:, 2], rr, ds)
        r["degenerate_normalisations"] = LAST_DEGENERATE_COUNT
        r["rms_px_vs_legacy"] = float(np.sqrt(np.mean((rec[:, 0] - legacy_x) ** 2 + (rec[:, 1] - legacy_y) ** 2)))
        r["max_px_vs_legacy"] = float(np.max(np.hypot(rec[:, 0] - legacy_x, rec[:, 1] - legacy_y)))
        sched[f"{variant}@recorded_reorigins"] = r
        for N in SCHEDULE_PERIODS:
            rec = recompose(G, events, W, H, lambda k, e, N=N: (k % N == 0 and k > 0), variant, drop_restart_increments=True)
            sched[f"{variant}@{N}"] = evaluate_trajectory(rec[:, :2], rec[:, 2], rr, ds)
            sched[f"{variant}@{N}"]["degenerate_normalisations"] = LAST_DEGENERATE_COUNT
    summary["phase5"] = sched
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="e.g. a-homography b-affine")
    a = ap.parse_args()
    out = Path(a.out)
    keys = list(RUNS)
    if a.only:
        keys = [tuple(s.split("-")) for s in a.only]
    allsum = {}
    for key in keys:
        name = f"{key[0]}-{key[1]}"
        print("analysing", name, flush=True)
        allsum[name] = analyse_run(key, out / name)
    (out / "summary_all.json").write_text(json.dumps(allsum, indent=2))
    for name, s in allsum.items():
        print(name, "R0 mismatches", s["R0_estimator_mismatches_vs_baseline"], "m_end", round(s["H1"]["m_end"], 4),
              "mean dlam", f"{s['H2']['mean_dlam_per_frame']:.2e}", "t", round(s["H2"]["t_stat"], 1),
              "lag slope", round(s["H2"]["lag_variance_log_log_slope"], 2))


if __name__ == "__main__":
    main()
