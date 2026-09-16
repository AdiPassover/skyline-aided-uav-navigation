"""EXP-VO-005 analysis: compare the estimator's composed footprint magnification m_k (from the
`logical_transform.csv` sidecar, EXP-VO-004 observables) with the exactly known height ratio
h_k/h_0 on the rendered sequences; test the pure-yaw / pure-altitude false-XY conditions; and
quantify the XY error of the composed vs constant-scale (SE(2)-every-frame) readouts under a real
height change. Reuses `exp_vo_004.internal_scale` for every observable so the two records measure
the same thing.

Usage::

    python evaluation/tools/exp_vo_005/analyse_exp_vo_005.py --out evaluations/exp-vo-005
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp_vo_004"))
sys.path.insert(0, str(HERE.parents[1]))
import internal_scale as isc  # noqa: E402
import synth_planar as sp     # noqa: E402

REPO = HERE.parents[2]
SEQ = list(sp.SEQUENCES)
MODELS = ("homography", "affine")
GSD0 = sp.H0 / sp.F_PX


def load_run(run_id: str):
    run = REPO / "runs" / run_id
    if not (run / "logical_transform.csv").exists():
        return None
    G_raw, events = isc.load_sidecar(run)
    frames = isc.read_csv(run / "frames.csv")
    return G_raw, events, frames


def gt(seq_id: str):
    rows = isc.read_csv(REPO / "datasets" / f"synth-scale-{seq_id}" / "groundtruth.csv")
    return {k: np.array([float(r[k]) for r in rows]) for k in ("timestamp_s", "east_m", "north_m", "up_m", "heading_deg")}


def analyse(seq_id: str, model: str, suffix: str = "") -> dict | None:
    r = load_run(f"synth-scale-{seq_id}-{model}{suffix}-v1")
    if r is None:
        return None
    G_raw, events, frames = r
    W, H = sp.W, sp.H
    n = G_raw.shape[0]
    # restart-dropped composition, as in EXP-VO-004
    G = np.empty_like(G_raw); G[0] = G_raw[0]
    for k in range(1, n):
        D = np.eye(3) if events[k] == "restart" else G_raw[k] @ np.linalg.inv(G_raw[k - 1])
        G[k] = D @ G[k - 1]; G[k] /= G[k][2, 2]
    obs = [isc.observables(G[k], W, H) for k in range(n)]
    m = np.array([o["m_centre"] for o in obs]); lam = np.log(m)
    g = gt(seq_id)
    ratio_true = g["up_m"] / g["up_m"][0]
    rel_err = m / ratio_true - 1.0
    dlam = np.diff(lam)
    aniso = np.array([o["anisotropy"] for o in obs]); persp = np.array([o["perspective"] for o in obs])
    lx = np.array([o["logical_x"] for o in obs]); ly = np.array([o["logical_y"] for o in obs])
    centre_disp_px = np.hypot(lx, ly)
    # true XY displacement in FIRST-FRAME pixels (LIT-VO-003 eq. 3): metres / GSD0
    true_e = g["east_m"] - g["east_m"][0]; true_n = g["north_m"] - g["north_m"][0]
    true_px = np.hypot(true_e, true_n) / GSD0
    path_m = float(np.sum(np.hypot(np.diff(g["east_m"]), np.diff(g["north_m"]))))
    # composed readout, in metres via the KNOWN GSD0 (no fit): compare shapes via endpoint & per-frame error.
    # Logical frame L is the first camera frame: x right = east when heading 0, y forward = north.
    comp_e = lx * GSD0; comp_n = ly * GSD0
    err_comp = np.hypot(comp_e - true_e, comp_n - true_n)
    # constant-scale readout: SE(2) every frame (the legacy prior at its most frequent)
    rec = isc.recompose(G_raw, events, W, H, lambda k, e: k > 0, "se2", drop_restart_increments=True)
    cs_e = rec[:, 0] * GSD0; cs_n = rec[:, 1] * GSD0
    err_cs = np.hypot(cs_e - true_e, cs_n - true_n)
    tracks = np.array([int(f["track_count"]) for f in frames])
    out = {
        "sequence": seq_id, "model": model, "suffix": suffix, "n_frames": n,
        "n_restarts": int(sum(1 for e in events if e == "restart")),
        "n_reorigins": int(sum(1 for e in events if e == "recenter")),
        "success_rate": float(np.mean([f["success"] == "true" for f in frames])),
        "mean_tracks": float(tracks.mean()),
        "H1_scale": {"median_rel_err": float(np.median(rel_err)), "p95_abs_rel_err": float(np.percentile(np.abs(rel_err), 95)),
                     "end_rel_err": float(rel_err[-1]), "m_end": float(m[-1]), "true_ratio_end": float(ratio_true[-1]),
                     "max_true_ratio": float(ratio_true.max()), "min_true_ratio": float(ratio_true.min())},
        "H2_drift": {"mean_dlam": float(dlam.mean()), "sd_dlam": float(dlam.std(ddof=1)),
                     "t": float(dlam.mean() / (dlam.std(ddof=1) / math.sqrt(dlam.size))),
                     "m_min": float(m.min()), "m_max": float(m.max()),
                     "lag_slope": isc.lag_variance_exponent(lam, 1, 500)["log_log_slope"] if n > 200 else None},
        "H3H4_centre": {"max_centre_disp_frac_width": float(centre_disp_px.max() / W),
                        "end_centre_disp_frac_width": float(centre_disp_px[-1] / W),
                        "true_disp_end_px": float(true_px[-1])},
        "H5_xy": {"path_m": path_m,
                  "composed_endpoint_err_m": float(err_comp[-1]), "composed_rms_err_m": float(np.sqrt(np.mean(err_comp ** 2))),
                  "constscale_endpoint_err_m": float(err_cs[-1]), "constscale_rms_err_m": float(np.sqrt(np.mean(err_cs ** 2))),
                  "predicted_constscale_frac": float(np.mean(ratio_true) - 1.0)},
        "H7_tilt": {"anisotropy_p50": float(np.median(aniso)), "anisotropy_max": float(aniso.max()),
                    "perspective_max": float(persp.max())},
        "yaw": {"yaw_edge_end": float(obs[-1]["yaw_edge_deg"]), "yaw_polar_end": float(obs[-1]["yaw_polar_deg"]),
                "true_heading_end": float(g["heading_deg"][-1])},
        "series": {"m": m.tolist(), "true_ratio": ratio_true.tolist(), "centre_disp_px": centre_disp_px.tolist(),
                   "anisotropy": aniso.tolist(), "err_comp_m": err_comp.tolist(), "err_cs_m": err_cs.tolist()},
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for sid in SEQ:
        for model in MODELS:
            for suffix in ("", "-border60"):
                r = analyse(sid, model, suffix)
                if r is not None:
                    results[f"{sid}-{model}{suffix}"] = r
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "series"} for k, v in results.items()}
    (out / "summary.json").write_text(json.dumps(slim, indent=2))
    with (out / "series.json").open("w") as f:
        json.dump({k: v["series"] for k, v in results.items()}, f)
    # H6: canvas schedule invariance on climb
    h6 = {}
    for model in MODELS:
        a_ = results.get(f"climb-{model}"); b_ = results.get(f"climb-{model}-border60")
        if a_ and b_:
            ma = np.array(a_["series"]["m"]); mb = np.array(b_["series"]["m"])
            h6[model] = {"rms_rel_diff_m": float(np.sqrt(np.mean((ma / mb - 1) ** 2))),
                         "endpoint_diff_frac_path": float(abs(a_["H5_xy"]["composed_endpoint_err_m"] - b_["H5_xy"]["composed_endpoint_err_m"]) / a_["H5_xy"]["path_m"]),
                         "reorigins": (a_["n_reorigins"], b_["n_reorigins"])}
    (out / "h6_schedule.json").write_text(json.dumps(h6, indent=2))
    for k, v in slim.items():
        print(f"{k:28s} ok={v['success_rate']:.3f} restarts={v['n_restarts']} m_end={v['H1_scale']['m_end']:.4f} true={v['H1_scale']['true_ratio_end']:.4f} "
              f"med_err={v['H1_scale']['median_rel_err']:+.4f} mean_dlam={v['H2_drift']['mean_dlam']:+.2e} t={v['H2_drift']['t']:+.1f} "
              f"centre={v['H3H4_centre']['max_centre_disp_frac_width']:.4f} comp_end={v['H5_xy']['composed_endpoint_err_m']:.2f}m cs_end={v['H5_xy']['constscale_endpoint_err_m']:.2f}m aniso={v['H7_tilt']['anisotropy_p50']:.3f}")
    print("H6", json.dumps(h6))


if __name__ == "__main__":
    main()
