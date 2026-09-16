"""EXP-CONF-002 Phase 2: audit of the RTK-derived per-frame rotation target (T-A/T-D).

Implements exactly the frozen analyses of the EXP-CONF-002 pre-registration (Phase 2):

1. native RTK sampling-interval distribution;
2. native heading resolution: nonzero step histogram, smallest step, plateau fraction and
   run lengths;
3. interpolated per-camera-frame increments (exactly ``labels.py``'s ``np.interp`` on the
   unwrapped heading): distribution, lag-1 autocorrelation, transition-bracketing structure;
4. clustering of T-D positives at native heading *transitions*: max-CDF-gap statistic vs the
   all-gradable distribution, permutation test (n=10000, seed 20260830);
5. accumulated-motion comparison: VO rotation vs native heading difference over one native RTK
   interval (no interpolation) and over time windows {0.2, 0.4, 0.5, 1.0} s;
6. attitude-stream cross-check (~100 Hz ``yaw_compass_deg``): resolution, rate, noise, and a
   three-way per-camera-interval comparison attitude vs interpolated RTK vs VO.

Existing data only. T-A/T-D are not modified. Output: ``evaluations/exp-conf-002/rtk_audit/``.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = TOOL_DIR.parents[1]
REPO = EVALUATION_DIR.parent

SEED = 20260830
N_PERM = 10_000
WINDOWS_S = (0.2, 0.4, 0.5, 1.0)

SEQUENCES = {
    # seq -> (dataset dir, labels dir, run dir)
    "hkairport01-a": ("datasets/hkairport01-a", "evaluations/exp-conf-001-dev/labels",
                      "runs/hkairport01-a-homography-rigid-conf-v1"),
    "hkairport01-b": ("datasets/hkairport01-b", "evaluations/exp-conf-001-dev/labels",
                      "runs/hkairport01-b-homography-rigid-conf-v1"),
    "amtown01-c": ("datasets/amtown01-c", "evaluations/exp-conf-001-dev/labels",
                   "runs/amtown01-c-homography-rigid-conf-v1"),
    "amtown01-d": ("datasets/amtown01-d", "evaluations/exp-conf-001-dev/labels",
                   "runs/amtown01-d-homography-rigid-conf-v1"),
    "amtown02": ("datasets/amtown02", "evaluations/exp-conf-001-heldout/labels",
                 "runs/amtown02-homography-rigid-conf-v1"),
    "hkisland01": ("datasets/hkisland01", "evaluations/exp-conf-001-heldout/labels",
                   "runs/hkisland01-homography-rigid-conf-v1"),
}

# Frozen T-D thresholds (dev-pool absolute; EXP-CONF-001 fitting_target_decision.json).
DECISION = REPO / "evaluations/exp-conf-001-dev/fitting_target_decision.json"


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k in rows[0]:
        col = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) if v != "" else np.nan for v in col])
        except ValueError:
            out[k] = np.array(col)
    return out


def qtiles(x: np.ndarray, qs=(5, 25, 50, 75, 90, 95, 99)) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    d = {f"p{q}": float(np.percentile(x, q)) for q in qs}
    d.update(n=int(len(x)), mean=float(np.mean(x)), sd=float(np.std(x)),
             min=float(np.min(x)), max=float(np.max(x)))
    return d


def run_lengths(mask: np.ndarray) -> list[int]:
    out, run = [], 0
    for v in mask:
        if v:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def max_cdf_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample max |CDF_a - CDF_b| (KS statistic, computed manually; numpy only)."""
    grid = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def perm_test(pos: np.ndarray, allv: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Permutation p-value for the max-CDF-gap of `pos` (subset) vs `allv` (population)."""
    obs = max_cdf_gap(pos, allv)
    n = len(pos)
    count = 0
    for _ in range(N_PERM):
        sub = rng.choice(allv, size=n, replace=False)
        if max_cdf_gap(sub, allv) >= obs:
            count += 1
    return obs, (count + 1) / (N_PERM + 1)


def circ_interp_deg(t: np.ndarray, tp: np.ndarray, deg: np.ndarray) -> np.ndarray:
    return np.interp(t, tp, np.unwrap(np.radians(deg)))


def audit_sequence(seq: str, paths: tuple[str, str, str], thresholds: dict,
                   rng: np.random.Generator) -> dict:
    ds_dir, labels_dir, run_dir = (REPO / p for p in paths)
    gt = read_csv(ds_dir / "groundtruth.csv")
    frames = read_csv(ds_dir / "frames.csv")
    labels = read_csv(REPO / paths[1] / f"{seq}.csv")
    labels_meta = json.load((REPO / paths[1] / f"{seq}.json").open(encoding="utf-8"))
    sidecar = read_csv(run_dir / "logical_transform.csv")

    t_gt = gt["timestamp_s"]
    heading = gt["heading_deg"]
    t_cam = frames["timestamp_s"]
    out: dict = {"n_native": len(t_gt), "n_frames": len(t_cam)}

    # -- 1. native sampling ------------------------------------------------------------------
    dt = np.diff(t_gt)
    out["native_dt_s"] = qtiles(dt)

    # -- 2. native heading resolution --------------------------------------------------------
    h_unwrapped_deg = np.degrees(np.unwrap(np.radians(heading)))
    dh = np.diff(h_unwrapped_deg)
    nonzero = np.abs(dh[np.abs(dh) > 1e-12])
    plateaus = np.abs(dh) <= 1e-12
    out["native_heading"] = {
        "increment_abs": qtiles(np.abs(dh)),
        "nonzero_step_abs": qtiles(nonzero),
        "smallest_nonzero_step_deg": float(np.min(nonzero)) if len(nonzero) else None,
        "plateau_fraction": float(np.mean(plateaus)),
        "plateau_run_lengths": qtiles(np.array(run_lengths(plateaus), dtype=float))
        if plateaus.any() else {"n": 0},
        # value granularity: how many distinct values mod 0.01 / 0.1 / 1.0 deg
        "frac_multiple_of_0p01": float(np.mean(np.abs(np.round(heading * 100) - heading * 100) < 1e-6)),
        "frac_multiple_of_0p1": float(np.mean(np.abs(np.round(heading * 10) - heading * 10) < 1e-6)),
        "frac_multiple_of_1": float(np.mean(np.abs(np.round(heading) - heading) < 1e-6)),
    }

    # -- 3. interpolated per-camera-frame increments (labels.py construction) ----------------
    h_interp = circ_interp_deg(t_cam, t_gt, heading)          # radians, unwrapped
    d_interp = np.degrees(np.diff(h_interp))                  # deg per camera frame
    gradable = labels["gradable_a"] == 1.0
    grad_inc = gradable[1:] if len(gradable) == len(t_cam) else gradable[1:len(t_cam)]
    x = d_interp[grad_inc[:len(d_interp)]] if len(d_interp) else d_interp
    ac1 = float(np.corrcoef(d_interp[:-1], d_interp[1:])[0, 1]) if len(d_interp) > 2 else None
    # does the camera interval bracket a native *value transition*?
    trans_idx = np.where(np.abs(dh) > 1e-12)[0] + 1           # native sample where value changed
    t_trans = t_gt[trans_idx] if len(trans_idx) else np.array([])
    brackets = np.zeros(len(d_interp), dtype=bool)
    if len(t_trans):
        lo = np.searchsorted(t_trans, t_cam[:-1], side="right")
        hi = np.searchsorted(t_trans, t_cam[1:], side="right")
        brackets = hi > lo
    out["interpolated_increment_deg"] = {
        "all": qtiles(d_interp), "gradable": qtiles(x), "lag1_autocorr": ac1,
        "frac_intervals_bracketing_transition": float(np.mean(brackets)) if len(brackets) else None,
        "abs_given_bracketing": qtiles(np.abs(d_interp[brackets])) if brackets.any() else {"n": 0},
        "abs_given_not_bracketing": qtiles(np.abs(d_interp[~brackets])) if (~brackets).any() else {"n": 0},
    }

    # -- 4. T-D positive clustering at native transitions ------------------------------------
    t_a = labels["t_a_deg"]
    out["td_transition_clustering"] = {}
    if len(t_trans):
        # distance of each camera timestamp to the nearest native transition
        def dist_to_trans(ts):
            i = np.searchsorted(t_trans, ts)
            left = np.where(i > 0, np.abs(ts - t_trans[np.clip(i - 1, 0, len(t_trans) - 1)]), np.inf)
            right = np.where(i < len(t_trans), np.abs(t_trans[np.clip(i, 0, len(t_trans) - 1)] - ts), np.inf)
            return np.minimum(left, right)

        d_all = dist_to_trans(t_cam[gradable])
        for name, thr in thresholds.items():
            pos_mask = gradable & (t_a > thr)
            n_pos = int(np.sum(pos_mask))
            if n_pos < 3:
                out["td_transition_clustering"][name] = {"n_pos": n_pos, "note": "too few positives"}
                continue
            d_pos = dist_to_trans(t_cam[pos_mask])
            gap, p = perm_test(d_pos, d_all, rng)
            out["td_transition_clustering"][name] = {
                "n_pos": n_pos, "max_cdf_gap": gap, "perm_p": p,
                "median_dist_pos_s": float(np.median(d_pos)),
                "median_dist_all_s": float(np.median(d_all)),
            }

    # -- 5. accumulated-motion comparison ----------------------------------------------------
    sign = float(labels_meta["t_a_sign_convention"])
    inc_rot = sign * sidecar["inc_rotation_deg"]
    # cumulative VO rotation on the camera-time grid; init/restart rows break the chain --
    # exclude windows containing a non-gradable frame instead of zeroing.
    cum_vo = np.concatenate([[0.0], np.cumsum(inc_rot[1:])])
    grad_all = gradable.copy()
    cum_bad = np.concatenate([[0], np.cumsum((~grad_all[1:]).astype(int))])

    acc: dict = {}
    # (a) one native RTK interval, native endpoints (no interpolation on the GT side)
    inside = (t_gt >= t_cam[0]) & (t_gt <= t_cam[-1])
    tg = t_gt[inside]
    hg = h_unwrapped_deg[inside]
    if len(tg) > 2:
        vo_at_native = np.interp(tg, t_cam, cum_vo)
        bad_at_native = np.interp(tg, t_cam, cum_bad)
        err = np.abs(np.diff(vo_at_native) - np.diff(hg))
        clean = np.abs(np.diff(bad_at_native)) < 1e-9      # no non-gradable frame inside
        acc["native_interval"] = {
            "err_deg_all": qtiles(err), "err_deg_clean": qtiles(err[clean]),
            "n_clean": int(np.sum(clean)),
            "typical_abs_rotation_deg": qtiles(np.abs(np.diff(hg))),
        }
    # (b) fixed time windows, camera-grid endpoints (GT side interpolated as labels.py does)
    h_cam_deg = np.degrees(h_interp)
    for w in WINDOWS_S:
        j = np.searchsorted(t_cam, t_cam + w, side="left")
        k = np.arange(len(t_cam))
        okw = j < len(t_cam)
        kk, jj = k[okw], j[okw]
        err = np.abs((cum_vo[jj] - cum_vo[kk]) - (h_cam_deg[jj] - h_cam_deg[kk]))
        clean = (cum_bad[jj] - cum_bad[kk]) == 0
        acc[f"window_{w}s"] = {
            "err_deg_all": qtiles(err), "err_deg_clean": qtiles(err[clean]),
            "n_clean": int(np.sum(clean)),
            "median_frames_per_window": float(np.median(jj - kk)),
        }
    out["accumulated"] = acc

    # -- 6. attitude-stream cross-check ------------------------------------------------------
    att_path = ds_dir / "attitude.csv"
    if att_path.exists():
        att = read_csv(att_path)
        t_att, yaw_att = att["timestamp_s"], att["yaw_compass_deg"]
        datt = np.diff(np.degrees(np.unwrap(np.radians(yaw_att))))
        nz = np.abs(datt[np.abs(datt) > 1e-12])
        att_at_cam = np.degrees(circ_interp_deg(t_cam, t_att, yaw_att))
        d_att_cam = np.diff(att_at_cam)
        m = grad_all[1:]
        vo_inc = inc_rot[1:]
        rtk_inc = d_interp
        att_inc = d_att_cam
        def rel(a, b):
            mm = m & np.isfinite(a) & np.isfinite(b)
            if np.sum(mm) < 10:
                return None
            return {"corr": float(np.corrcoef(a[mm], b[mm])[0, 1]),
                    "sd_diff_deg": float(np.std(a[mm] - b[mm])),
                    "median_abs_diff_deg": float(np.median(np.abs(a[mm] - b[mm]))), "n": int(np.sum(mm))}
        out["attitude"] = {
            "rate_hz": float(1.0 / np.median(np.diff(t_att))),
            "yaw_step_abs_nonzero": qtiles(nz),
            "smallest_nonzero_step_deg": float(np.min(nz)) if len(nz) else None,
            "plateau_fraction": float(np.mean(np.abs(datt) <= 1e-12)),
            "per_camera_interval": {
                "vo_vs_rtk": rel(vo_inc, rtk_inc),
                "vo_vs_attitude": rel(vo_inc, att_inc),
                "attitude_vs_rtk": rel(att_inc, rtk_inc),
            },
        }
    else:
        out["attitude"] = None

    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    decision = json.load(DECISION.open(encoding="utf-8"))
    thresholds = decision["t_a"]["thresholds"]

    out_dir = REPO / "evaluations/exp-conf-002/rtk_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"seed": SEED, "n_perm": N_PERM, "windows_s": list(WINDOWS_S),
              "thresholds_deg": thresholds, "sequences": {}}
    for seq, paths in SEQUENCES.items():
        print(f"[audit] {seq} ...", flush=True)
        report["sequences"][seq] = audit_sequence(seq, paths, thresholds, rng)

    with (out_dir / "audit.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"[audit] written {out_dir / 'audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
