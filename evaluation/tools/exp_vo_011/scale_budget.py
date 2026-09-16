"""EXP-VO-011 Phase 2: the quantity to explain.

Reads the committed rigid-readout run records and produces the scale-bias budget: the per-frame
systematic log-scale bias on every real (window, model) arm, its robust spread, its significance
under an AR(1)-corrected effective sample size, the accumulated figure, the physically plausible
envelope from LiDAR height above the imaged surface, and the ratio between them.

Nothing is re-run and no estimator is invoked. Output: `evaluations/exp-vo-011/scale_budget.json`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import scale_series as ss


def physical_envelope(arm: dict, cov: dict) -> dict:
    """What the flight could physically have done to the scale, from LiDAR and RTK.

    Two envelopes, deliberately kept apart (`LIT-VO-004` section 4):

    - `lidar` -- height above the **imaged surface**, which is what actually sets the scale.
    - `takeoff_datum` -- `groundtruth.csv`'s `up_m`, which is what the telemetry reports and what
      the dataset paper means by "constant 80 m". Quoting this one as the scale envelope is the
      error `LIT-VO-004` names; it is reported here only to show the size of that error.

    Tilt contributes a bounded, reversible factor `cos^{3/2} theta` (`LIT-VO-003` section 4); its
    extreme is reported, not integrated, because it cannot accumulate.
    """
    out = {}
    for name, h in (("lidar", cov["lidar_height_m"]), ("takeoff_datum", cov["up_m"])):
        lam = np.log(h / h[0])                      # log(h_k / h_0), the physical accumulated scale
        d = np.diff(lam)
        out[name] = {
            "h_first_m": float(h[0]), "h_last_m": float(h[-1]),
            "h_min_m": float(h.min()), "h_max_m": float(h.max()),
            "accum_log_scale": float(lam[-1]),
            "accum_scale": float(math.exp(lam[-1])),
            "span_scale_min": float(math.exp(lam.min())),
            "span_scale_max": float(math.exp(lam.max())),
            "per_frame_mean_abs": float(np.mean(np.abs(d))) if len(d) else float("nan"),
            "per_frame_p95_abs": float(np.percentile(np.abs(d), 95)) if len(d) else float("nan"),
            "per_frame_mean": float(np.mean(d)) if len(d) else float("nan"),
        }
    tilt = cov["tilt_deg"]
    out["tilt"] = {
        "median_deg": float(np.median(tilt)), "max_deg": float(tilt.max()),
        "scale_factor_at_max": float(math.cos(math.radians(float(tilt.max()))) ** 1.5),
        "anisotropy_ceiling_at_max": float(1.0 / math.cos(math.radians(float(tilt.max())))),
    }
    return out


def budget_for(window: str, model: str) -> dict:
    arm = ss.load_arm(window, model)
    cov = ss.covariates_for(arm)
    inc = arm["inc_log_scale"]

    stats = ss.describe(inc)
    phys = physical_envelope(arm, cov)

    observed_accum = float(np.sum(inc))
    lidar_accum = phys["lidar"]["accum_log_scale"]

    # How far outside physical plausibility the accumulated figure is. Reported on the log scale
    # first (the honest comparison) and only then exponentiated, because exponentiation of a small
    # per-frame bias is what makes these numbers look enormous.
    ratio_log = (observed_accum / lidar_accum) if abs(lidar_accum) > 1e-12 else float("inf")

    # The per-frame bias against the largest per-frame rate the physical height series ever shows.
    phys_rate = phys["lidar"]["per_frame_p95_abs"]
    bias_over_phys_rate = abs(stats["mean"]) / phys_rate if phys_rate > 0 else float("inf")

    # A physical reading of the bias: at this height, what climb rate would produce it?
    h0 = phys["lidar"]["h_first_m"]
    dt = float(np.median(np.diff(arm["timestamp_s"])))
    phantom_climb_ms = stats["mean"] * h0 / dt

    return {
        "window": window, "model": model,
        "rows_total": arm["n_total_rows"], "rows_scored": stats["n"],
        "rows_dropped": arm["n_dropped"], "restarts": arm["n_restart"],
        "recenters": arm["n_recenter"],
        "frame_dt_s": dt,
        "inc_log_scale": stats,
        "accum_log_scale_observed": observed_accum,
        "accum_scale_observed": float(math.exp(observed_accum)),
        "accum_log_scale_from_record": float(arm["rigid_accum_log_scale"][-1]),
        "physical": phys,
        "observed_over_physical_log": ratio_log,
        "observed_over_physical_scale": (float(math.exp(observed_accum - lidar_accum))),
        "bias_over_physical_p95_rate": bias_over_phys_rate,
        "phantom_climb_ms": phantom_climb_ms,
        "anisotropy_median": float(np.median(arm["inc_anisotropy"])),
        "perspective_median": float(np.median(arm["inc_perspective"])),
        "flow_px_median": float(np.median(arm["inc_flow_px"])),
        "lidar_source": cov["lidar_source"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arms = [budget_for(w, m) for w, m in ss.REAL_ARMS]
    controls = []
    for w, m in ss.CONTROL_ARMS:
        try:
            controls.append(budget_for(w, m))
        except FileNotFoundError:
            pass

    report = {"record": "EXP-VO-011", "phase": 2, "arms": arms, "controls": controls}
    (out / "scale_budget.json").write_text(json.dumps(report, indent=1))

    hdr = (f"{'arm':<28} {'n':>6} {'mean/frame':>12} {'MADs':>10} {'t_eff':>8} "
           f"{'accum':>9} {'phys accum':>11} {'ratio':>9} {'climb m/s':>10}")
    print(hdr)
    print("-" * len(hdr))
    for a in arms + controls:
        s = a["inc_log_scale"]
        print(f"{a['window'] + '-' + a['model']:<28} {s['n']:>6} {s['mean']:>12.3e} "
              f"{s['mad_sigma']:>10.3e} {s['t_eff']:>8.2f} "
              f"{a['accum_scale_observed']:>9.3f} "
              f"{a['physical']['lidar']['accum_scale']:>11.4f} "
              f"{a['observed_over_physical_scale']:>9.3f} {a['phantom_climb_ms']:>10.3f}")
    print(f"\nwrote {out / 'scale_budget.json'}")


if __name__ == "__main__":
    main()
