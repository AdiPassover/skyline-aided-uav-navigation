"""`EXP-VO-012`: score every regime, arm and sweep point, from the committed run records.

**No estimator is invoked and no imagery is read.** The eight capture runs are read once; every
sweep point in the pre-registration is then a different *height array* applied to the same
increments. That is what makes a sweep of this size affordable, and it is also the structural
guarantee that no sweep can perturb the estimator.

Output: `evaluations/exp-vo-012/results.json`, plus `series/` for the per-frame diagnostics the
figures need.

Three arms throughout (`DEC-VO-007`, `metric_readout`): **fixed** `h₀` for every frame, **baro**
`h₀ + h_baro(t)` under the flat-terrain assumption, **oracle** the renderer's true `h_AGL`. The
oracle is diagnostic only — no sensor on the author's platform supplies it at survey height.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVALUATION_DIR = HERE.parents[1]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


st = _load("exp_vo_012_synth_terrain", "synth_terrain.py")
baro = _load("exp_vo_012_baro", "baro.py")
mr = _load("exp_vo_012_metric_readout", "metric_readout.py")
mm = _load("exp_vo_012_metric_metrics", "metric_metrics.py")

#: Pre-registered in EXP-VO-012 (Metrics, B3). LIT-VO-003 section 10.6 is why it is not 1.
PATH_STEP = 5
RPE_LENGTHS = (10, 25, 50, 100)

REGIMES = {
    "bvo-flat-const": "R1", "bvo-flat-climb": "R2", "bvo-flat-profile": "R2b",
    "bvo-ramp-constalt": "R7a", "bvo-ridge-constalt": "R7b", "bvo-ramp-climb": "R8",
}


# --------------------------------------------------------------------------- loading

def load_case(seq_id: str, model: str) -> dict | None:
    ds = REPO / "datasets" / seq_id
    run = REPO / "runs" / f"{seq_id}-{model}-rigid-v1"
    if not (run / "logical_transform.csv").exists() or not (ds / "terrain.csv").exists():
        return None
    with (ds / "groundtruth.csv").open(newline="") as f:
        gt = list(csv.DictReader(f))
    with (ds / "terrain.csv").open(newline="") as f:
        terr = list(csv.DictReader(f))
    meta = json.loads((ds / "dataset.json").read_text())
    inc = mr.load_increments(run, st.W, st.H)

    col = lambda rows, k: np.array([float(r[k]) for r in rows])          # noqa: E731
    # The estimator may drop frames; index ground truth by the run's own frame_index so the two
    # are aligned by construction rather than by assuming a 1:1 schedule.
    idx = inc.frame_index
    return {
        "seq_id": seq_id, "model": model, "regime": REGIMES[seq_id],
        "east": col(gt, "east_m")[idx], "north": col(gt, "north_m")[idx],
        "up": col(gt, "up_m")[idx], "heading": col(gt, "heading_deg")[idx],
        "t": col(gt, "timestamp_s")[idx],
        "terrain": col(terr, "terrain_m")[idx], "agl": col(terr, "agl_m")[idx],
        "h0": float(meta["metadata"]["h0_agl_m"]),
        "f_working": mr.f_working(meta["metadata"]["camera_intrinsics"]["fx"], 1),
        "inc": inc, "events": inc.events,
        "n_restart": sum(1 for e in inc.events if e == "restart"),
        "n_frames": len(inc),
    }


def gt_xy(case: dict) -> np.ndarray:
    return np.stack([case["east"], case["north"]], axis=1)


# --------------------------------------------------------------------------- arms

def arm_heights(case: dict, spec: baro.BaroSpec | None = None, h0_rel_error: float = 0.0) -> dict:
    """The three arms' height series, plus the channel's validity flags."""
    h0_used = case["h0"] * (1.0 + h0_rel_error)
    sampled = (baro.ideal(case["t"], case["up"]) if spec is None
               else baro.sample(case["t"], case["up"], spec))
    return {
        "fixed": (np.full(case["n_frames"], h0_used), None, sampled),
        "baro": (mr.h_agl_from_baro(h0_used, sampled.h_baro), sampled.valid, sampled),
        # The oracle is not perturbed by h0 error: it IS the truth, by definition. Perturbing it
        # would make the h0 sweep measure two things at once.
        "oracle": (case["agl"], None, sampled),
    }


def score_arms(case: dict, spec: baro.BaroSpec | None = None, h0_rel_error: float = 0.0,
               arms=("fixed", "baro", "oracle")) -> dict:
    heights = arm_heights(case, spec, h0_rel_error)
    g = gt_xy(case)
    out = {}
    for arm in arms:
        h, valid, sampled = heights[arm]
        track = mr.integrate_metric(case["inc"], h, case["f_working"], arm=arm, valid=valid)
        s = mm.score(track.xy(), g, float(case["heading"][0]),
                     path_step=PATH_STEP, rpe_lengths=RPE_LENGTHS)
        s["mean_gsd_m_per_px"] = float(track.gsd_m_per_px.mean())
        s["mean_height_used_m"] = float(track.h_used_m.mean())
        s["mean_height_error_m"] = float(np.mean(track.h_used_m - case["agl"]))
        s["n_degraded_frames"] = int((~track.valid).sum())
        out[arm] = s
    return out


# --------------------------------------------------------------------------- phases

def phase_regimes(cases: dict) -> dict:
    """R1, R2, R2b, R7a, R7b, R8 — the ideal channel, all three arms."""
    out = {}
    for key, case in cases.items():
        res = score_arms(case)
        out[f"{case['seq_id']}/{case['model']}"] = {
            "regime": case["regime"], "n_frames": case["n_frames"],
            "n_restarts": case["n_restart"],
            "terrain_range_m": [float(case["terrain"].min()), float(case["terrain"].max())],
            "agl_range_m": [float(case["agl"].min()), float(case["agl"].max())],
            "baro_range_m": [float((case["up"] - case["up"][0]).min()),
                             float((case["up"] - case["up"][0]).max())],
            "h0_m": case["h0"], "arms": res,
        }
    return out


def phase_noise(cases: dict, sigmas, seeds: int = 20) -> dict:
    """R3. Zero-mean height noise, `seeds` draws per point (H4 is about a distribution)."""
    out = {}
    for key, case in cases.items():
        rows = []
        for sigma in sigmas:
            errs, ratios = [], []
            for seed in range(seeds if sigma > 0 else 1):
                s = baro.BaroSpec(noise_m=sigma, seed=seed)
                r = score_arms(case, s, arms=("baro",))["baro"]
                errs.append(r["ref_init"]["endpoint_error_pct_of_path"])
                ratios.append(r["path_length_ratio"])
            rows.append({
                "sigma_m": sigma, "n_seeds": len(errs),
                "endpoint_pct_mean": float(np.mean(errs)),
                "endpoint_pct_sd": float(np.std(errs)),
                "endpoint_pct_p95": float(np.percentile(errs, 95)),
                "path_ratio_mean": float(np.mean(ratios)),
                # H4's closed form: (sigma/h)/sqrt(n), as a PERCENTAGE of path.
                "h4_prediction_pct": float(100.0 * (sigma / float(np.mean(case["agl"])))
                                           / np.sqrt(case["n_frames"])),
            })
        out[key] = rows
    return out


def phase_drift(cases: dict, drifts) -> dict:
    """R4. Slow signed drift — the mechanism that does NOT cancel in a relative channel."""
    out = {}
    for key, case in cases.items():
        rows = []
        for d in drifts:
            r = score_arms(case, baro.BaroSpec(drift_m=d), arms=("baro",))["baro"]
            rows.append({
                "drift_m": d,
                "endpoint_pct": r["ref_init"]["endpoint_error_pct_of_path"],
                "path_length_ratio": r["path_length_ratio"],
                "mean_height_error_m": r["mean_height_error_m"],
                # H5's closed form: a linear drift to D has mean D/2, so scale error is D/(2h).
                "h5_prediction_ratio": float(1.0 + d / (2.0 * float(np.mean(case["agl"])))),
            })
        out[key] = rows
    return out


def phase_rate(cases: dict, rates, latencies) -> dict:
    """R5. Rate and latency, causal ZOH against offline linear interpolation."""
    out = {}
    for key, case in cases.items():
        rows = []
        for rate in rates:
            for policy in ("zoh", "linear"):
                r = score_arms(case, baro.BaroSpec(rate_hz=rate, policy=policy),
                               arms=("baro",))["baro"]
                rows.append({"rate_hz": rate, "latency_s": 0.0, "policy": policy,
                             "causal": policy == "zoh",
                             "endpoint_pct": r["ref_init"]["endpoint_error_pct_of_path"],
                             "mean_height_error_m": r["mean_height_error_m"],
                             "path_length_ratio": r["path_length_ratio"]})
        for lat in latencies:
            r = score_arms(case, baro.BaroSpec(rate_hz=10.0, latency_s=lat),
                           arms=("baro",))["baro"]
            rows.append({"rate_hz": 10.0, "latency_s": lat, "policy": "zoh", "causal": True,
                         "endpoint_pct": r["ref_init"]["endpoint_error_pct_of_path"],
                         "mean_height_error_m": r["mean_height_error_m"],
                         "path_length_ratio": r["path_length_ratio"]})
        out[key] = rows
    return out


def phase_dropout(cases: dict, durations, starts=(0.25, 0.50, 0.75), taus=(1.0, 2.0, 5.0, 10.0)):
    """R6. A dropout is a hold; its cost is bounded by the vertical motion inside it (H7)."""
    out = {}
    for key, case in cases.items():
        span = float(case["t"][-1] - case["t"][0])
        rows = []
        for dur in durations:
            for frac in starts:
                start = max(0.5, frac * span - dur / 2.0)
                if start + dur >= span:
                    continue
                for tau in taus:
                    s = baro.BaroSpec(rate_hz=10.0, dropouts=((start, dur),), tau_stale_s=tau)
                    r = score_arms(case, s, arms=("baro",))["baro"]
                    rows.append({"duration_s": dur, "start_frac": frac, "tau_stale_s": tau,
                                 "endpoint_pct": r["ref_init"]["endpoint_error_pct_of_path"],
                                 "n_degraded_frames": r["n_degraded_frames"],
                                 "path_length_ratio": r["path_length_ratio"]})
        out[key] = rows
    return out


def phase_h0(cases: dict, etas) -> dict:
    """Phase 6. `h₀` error is exactly multiplicative (H8) — the closed form is the baseline."""
    out = {}
    for key, case in cases.items():
        rows = []
        for eta in etas:
            r = score_arms(case, None, h0_rel_error=eta, arms=("baro",))["baro"]
            # H8: eta * h0 / h_AGL(t), path-averaged. Constant altitude over flat ground -> eta.
            pred = float(np.mean(eta * case["h0"] / case["agl"]))
            rows.append({"eta": eta, "eta_pct": 100.0 * eta,
                         "endpoint_pct": r["ref_init"]["endpoint_error_pct_of_path"],
                         "path_length_ratio": r["path_length_ratio"],
                         # Carried so Phase 5's methodological point is measurable rather than
                         # argued: a CONSTANT scale error is exactly what a Sim(2) alignment is free
                         # to absorb, so this column should not move with eta at constant altitude.
                         "sim2_norm_ate_pct": 100.0 * r["sim2"]["ate_rmse_normalised"],
                         "sim2_fitted_scale": r["sim2"]["alignment_scale"],
                         "h8_prediction_ratio": 1.0 + pred})
        out[key] = rows
    return out


def phase_diagnostics(cases: dict, out_dir: Path) -> dict:
    """Phase 7. What the three channels think, per frame. **Logged, never used.**"""
    summary = {}
    series_dir = out_dir / "series"
    series_dir.mkdir(parents=True, exist_ok=True)
    for key, case in cases.items():
        inc = case["inc"]
        visual_accum = np.exp(np.cumsum(inc.inc_log_scale))
        # The estimator's scale observable is sqrt|det J(D^-1)|, so an accumulated value ABOVE 1
        # means it believes the camera has receded. The physical height ratio it should equal, if
        # LIT-VO-003 section 2 held, is h_k/h_0.
        agl_ratio = case["agl"] / case["agl"][0]
        baro_ratio = (case["h0"] + (case["up"] - case["up"][0])) / case["h0"]
        name = key.replace("/", "_")
        with (series_dir / f"{name}.csv").open("w", newline="") as f:
            f.write("frame_index,t_s,terrain_m,agl_m,baro_relative_m,visual_accum_scale,"
                    "agl_ratio,baro_implied_ratio,inc_log_scale,event\n")
            for k in range(case["n_frames"]):
                f.write(f"{inc.frame_index[k]},{case['t'][k]:.4f},{case['terrain'][k]:.5f},"
                        f"{case['agl'][k]:.5f},{case['up'][k] - case['up'][0]:.5f},"
                        f"{visual_accum[k]:.8f},{agl_ratio[k]:.8f},{baro_ratio[k]:.8f},"
                        f"{inc.inc_log_scale[k]:.10f},{case['events'][k]}\n")

        finite = np.isfinite(visual_accum) & (agl_ratio > 0)
        # H11: on flat terrain the visual scale should track h/h0; over the ramp LIT-VO-003 section
        # 4b predicts it reads -1/2 of the true relative height change.
        d_log_h = np.log(agl_ratio[finite])
        d_log_v = np.log(visual_accum[finite])
        slope = (float(np.polyfit(d_log_h, d_log_v, 1)[0])
                 if np.ptp(d_log_h) > 1e-6 else float("nan"))
        summary[key] = {
            "visual_accum_end": float(visual_accum[-1]),
            "agl_ratio_end": float(agl_ratio[-1]),
            "baro_implied_ratio_end": float(baro_ratio[-1]),
            "visual_vs_agl_end_rel_error": float(visual_accum[-1] / agl_ratio[-1] - 1.0),
            "slope_dlogvisual_dlogagl": slope,
            "series_csv": f"series/{name}.csv",
        }
    return summary


# --------------------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="evaluations/exp-vo-012")
    a = ap.parse_args()
    out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = {}
    for seq_id in st.SEQUENCES:
        for model in st.MODELS[seq_id]:
            c = load_case(seq_id, model)
            if c is None:
                print(f"  MISSING runs/{seq_id}-{model}-rigid-v1")
                continue
            cases[f"{seq_id}/{model}"] = c
    if not cases:
        raise SystemExit("no capture runs found; run VoRunnerApp first")
    print(f"loaded {len(cases)} arms: {', '.join(sorted(cases))}")

    # The sensor sweeps run on the flat regimes, where the flat-terrain assumption holds and the
    # sensor's own contribution is therefore isolated. Running them over terrain would confound the
    # sweep with the datum error (R7/R8's subject) and measure neither cleanly.
    flat = {k: v for k, v in cases.items() if v["regime"].startswith("R2")}

    results = {
        "record": "EXP-VO-012", "path_step_frames": PATH_STEP, "rpe_lengths_m": list(RPE_LENGTHS),
        "arms": {"fixed": "h0 for every frame (what the system does today)",
                 "baro": "h0 + h_baro(t), flat-terrain assumption -- THE DESIGN UNDER TEST",
                 "oracle": "true h_AGL from the renderer -- DIAGNOSTIC ONLY"},
        "regimes": phase_regimes(cases),
        "r3_noise": phase_noise(flat, (0.0, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00)),
        "r4_drift": phase_drift(flat, (-8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0,
                                       0.25, 0.5, 1.0, 2.0, 4.0, 8.0)),
        "r5_rate": phase_rate(flat, (50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5),
                              (0.0, 0.1, 0.25, 0.5, 1.0)),
        "r6_dropout": phase_dropout(flat, (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)),
        "phase6_h0": phase_h0(cases, (-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10)),
        "phase7_diagnostics": phase_diagnostics(cases, out),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
