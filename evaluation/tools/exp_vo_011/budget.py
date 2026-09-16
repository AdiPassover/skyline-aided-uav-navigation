"""EXP-VO-011 Phase 8: the scale-bias budget — how much of the *accumulated* real drift each
mechanism can account for, once its own parity and its own flight-long integral are taken into
account.

**Why a per-frame magnitude is not enough, and this module exists.** Phases 3-5 measure what each
mechanism does to one frame pair. Three of them reach the observed per-frame magnitude at some
parameter value. But the quantity that matters is the *accumulated* drift, and a mechanism that is
**odd** in some physical rate contributes

    accumulated  =  (bias per unit rate)  x  (sum of that rate over the flight)

so its total is set by the flight-long **integral** of its driving quantity, not by its per-frame
size. For two of the three that integral telescopes:

- **a coherent ground slope** is odd in `(grad z . t)`, whose sum along a path is the net terrain
  height change, `-1/2 log(h_end / h_0)` in log-scale terms (`LIT-VO-003` section 4b);
- **a changing boresight tilt** is odd in `dTilt` (measured, `parity.csv`), whose sum along a flight
  is the net attitude change -- and the airframe ends roughly as it started.

The third, **lens distortion**, does *not* telescope: its driving quantity is
`(feature centroid . travel)`, and the centroid's offset from the distortion centre is fixed in the
CAMERA frame, so with a concentrated travel direction it accumulates with distance flown. Its size is
bounded instead by how asymmetric the correspondence set can plausibly be, and the effect is linear
in that offset, so the deterministic one-quadrant lattice measurement scales down to the realistic
case exactly.

Every number here is read from committed artifacts. Output: `scale_bias_budget.json`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import scale_series as ss

REPO = Path(__file__).resolve().parents[3]

# Geometry of the deterministic lattice arms, needed to scale the distortion result to a realistic
# feature-centroid offset. The one-quadrant lattice covers [0, 0.4 W] x [0, 0.4 H], so its centroid
# sits at (0.2 W, 0.2 H) and its offset from the image centre is |(-0.3 W, -0.3 H)|.
W, H, FOCAL = 1224.0, 1024.0, 735.53
QUADRANT_OFFSET_PX = math.hypot(0.3 * W, 0.3 * H)
# The principal point at downsampleFactor 2 (MARS-LVIG calibration, halved), against the image
# centre where the pipeline evaluates the Jacobian. A correspondence set that is symmetric about the
# image centre is offset from the DISTORTION centre by exactly this much.
PRINCIPAL_POINT_OFFSET_PX = math.hypot(1224 / 2 - 1172.3577 / 2, 1024 / 2 - 1046.3674 / 2)


def rendered_distortion(model: str) -> dict | None:
    """Phase 3b: what distortion does through the REAL pipeline, not through a point model.

    The correspondence-level lattice arms bound the effect by geometry, but they have to assume a
    feature distribution. The rendered arms do not: the detector picks its own features from real
    rendered texture, and the tracker, the respawn logic, the keyframing and the composition are all
    the shipped ones. Where the two disagree, this is the better estimate of what a real lens would
    do -- with the caveat that it is one heading, and the effect is odd in travel direction, so a
    flight that varies its heading would accumulate less than this.
    """
    base, dist, rect = (REPO / "runs" / f"synth-dist-{a}-{model}-v1" / "logical_transform.csv"
                        for a in ("none", "distorted", "rectified"))
    if not (base.exists() and dist.exists() and rect.exists()):
        return None

    def mean_of(path: Path) -> float:
        vals = []
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["event"] in ss.DROPPED_EVENTS or not r["inc_log_scale"]:
                    continue
                vals.append(float(r["inc_log_scale"]))
        return float(np.mean(vals))

    b, d, r = mean_of(base), mean_of(dist), mean_of(rect)
    return {"none": b, "distorted": d, "rectified": r,
            "effect": d - b, "residual_after_rectification": r - b,
            "removed_fraction": 1.0 - abs(r - b) / abs(d - b) if d != b else float("nan")}


def load_parity(path: Path) -> dict:
    out = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["condition"], r["model"])] = {k: float(r[k]) for k in
                                                 ("bias_log_scale", "se_log_scale", "sd_log_scale")}
    return out


def odd_even(parity: dict, mech: str, model: str) -> dict:
    f = parity[(mech + "+", model)]["bias_log_scale"]
    b = parity[(mech + "-", model)]["bias_log_scale"]
    odd, even = (f - b) / 2, (f + b) / 2
    return {"forward": f, "reversed": b, "odd": odd, "even": even,
            "verdict": "ODD" if abs(odd) > 2 * abs(even)
            else "EVEN" if abs(even) > 2 * abs(odd) else "mixed/noise"}


def flight_integrals(window: str) -> dict:
    """The flight-long sums each odd mechanism's contribution is proportional to."""
    arm = ss.load_arm(window, "affine")
    cov = ss.covariates_for(arm)
    tilt = cov["tilt_deg"]
    d_tilt = np.diff(tilt)
    h = cov["lidar_height_m"]
    return {
        "n_frames": int(len(arm["inc_log_scale"])),
        "observed_accum_log_scale": float(np.sum(arm["inc_log_scale"])),
        # Slope: LIT-VO-003 (4b.2) integrates to -1/2 log(h_end / h_0).
        "lidar_log_height_change": float(math.log(h[-1] / h[0])),
        "slope_predicted_accum": float(-0.5 * math.log(h[-1] / h[0])),
        # Changing tilt: odd in dTilt, so its total is proportional to the NET attitude change.
        "sum_d_tilt_deg": float(np.sum(d_tilt)),
        "mean_abs_d_tilt_deg": float(np.mean(np.abs(d_tilt))),
        "mean_d_tilt_deg": float(np.mean(d_tilt)),
        "path_m": float(cov["distance_m"][-1]),
    }


def budget(parity: dict, mechanisms: list[dict], window: str, model: str) -> dict:
    """The explained fraction of one arm's accumulated drift, mechanism by mechanism."""
    arm = ss.load_arm(window, model)
    observed = float(np.sum(arm["inc_log_scale"]))
    integ = flight_integrals(window)
    n = len(arm["inc_log_scale"])
    up = model.upper()

    def one(cond, **kw):
        rows = [r for r in mechanisms if r["condition"] == cond and r["model"] == up]
        for k, v in kw.items():
            rows = [r for r in rows if abs(r[k] - v) < 1e-9]
        if len(rows) != 1:
            raise KeyError(f"{cond} {up} {kw}: {len(rows)} rows")
        return rows[0]

    out = {"window": window, "model": model, "observed_accum_log_scale": observed,
           "n_frames": n, "integrals": integ, "mechanisms": {}}

    # --- 1. Coherent ground slope: odd, telescopes to the net terrain height change -------------
    out["mechanisms"]["coherent_slope"] = {
        "parity": odd_even(parity, "slope", up),
        "per_frame_at_22m": [r for r in mechanisms if r["condition"] == "relief"
                             and r["model"] == up and r["relief"] == "SLOPE"
                             and abs(r["relief_spread_m"] - 22.0) < 1e-9
                             and abs(r["travel_angle_deg"]) < 1e-9][0]["bias_log_scale"],
        "accumulated_prediction": integ["slope_predicted_accum"],
        "explained_fraction": integ["slope_predicted_accum"] / observed,
        "why_bounded": "odd in (grad z . t); the sum along a path is the net terrain height change",
    }

    # --- 2. Changing boresight tilt: odd in dTilt, telescopes to the net attitude change --------
    dt = odd_even(parity, "dtilt", up)
    per_deg = dt["odd"] / 0.1                      # the parity arm ran at 0.1 deg/frame
    out["mechanisms"]["changing_tilt"] = {
        "parity": dt,
        "per_frame_at_0p1_deg": dt["forward"],
        "bias_per_deg_of_tilt_rate": per_deg,
        "accumulated_prediction": per_deg * integ["sum_d_tilt_deg"],
        "explained_fraction": per_deg * integ["sum_d_tilt_deg"] / observed,
        "why_bounded": "odd in dTilt; the sum along a flight is the net attitude change, and the "
                       "airframe ends roughly as it started",
    }

    # --- 3. Lens distortion: odd, and it does NOT telescope -------------------------------------
    q = one("dist-lattice-quadrant", k1=-0.02, travel_angle_deg=0.0)["bias_log_scale"]
    sym = one("dist-lattice", k1=-0.02, travel_angle_deg=0.0)["bias_log_scale"]
    lattice_extrapolated = q * PRINCIPAL_POINT_OFFSET_PX / QUADRANT_OFFSET_PX
    ren = rendered_distortion(model)
    # Prefer the rendered end-to-end measurement where it exists: it makes no assumption about the
    # feature distribution, which is the whole difficulty with this mechanism.
    realistic = ren["effect"] if ren else lattice_extrapolated
    out["mechanisms"]["lens_distortion"] = {
        "parity": odd_even(parity, "dist", up),
        "per_frame_one_quadrant_support": q,
        "per_frame_symmetric_support": sym,
        "quadrant_centroid_offset_px": QUADRANT_OFFSET_PX,
        "realistic_centroid_offset_px": PRINCIPAL_POINT_OFFSET_PX,
        "per_frame_lattice_extrapolated": lattice_extrapolated,
        "rendered": ren,
        "source_of_point_estimate": "rendered end-to-end" if ren else "lattice extrapolation",
        "per_frame_at_realistic_offset": realistic,
        "accumulated_prediction": realistic * n,
        "explained_fraction": realistic * n / observed,
        "why_not_bounded": "odd in (feature centroid . travel); the centroid offset is fixed in the "
                           "CAMERA frame, so with a concentrated travel direction it accumulates "
                           "with distance flown rather than telescoping",
        "centroid_offset_needed_for_the_whole_drift_px":
            QUADRANT_OFFSET_PX * (observed / n) / q if q else float("nan"),
    }

    # --- 4. Homogeneous relief and feature noise: no resolvable bias -----------------------------
    # The bound on these two is set by Monte Carlo error, so it is taken from the high-precision
    # parity arms (8,000 trials each) rather than from the grid (250), where available.
    rf = [r for r in mechanisms if r["condition"] == "relief" and r["model"] == up
          and r["relief"] == "RANDOM_FIELD" and abs(r["relief_spread_m"] - 22.0) < 1e-9
          and abs(r["travel_angle_deg"]) < 1e-9][0]
    ei = one("eiv", sigma_px=0.5)
    if ("reliefhi+", up) in parity:
        rf = {"bias_log_scale": parity[("reliefhi+", up)]["bias_log_scale"],
              "se_log_scale": parity[("reliefhi+", up)]["se_log_scale"],
              "sd_log_scale": parity[("reliefhi+", up)]["sd_log_scale"]}
    if ("noisehi+", up) in parity:
        ei = {"bias_log_scale": parity[("noisehi+", up)]["bias_log_scale"],
              "se_log_scale": parity[("noisehi+", up)]["se_log_scale"],
              "sd_log_scale": parity[("noisehi+", up)]["sd_log_scale"]}
    for name, row, note in (
            ("homogeneous_relief", rf,
             "produces variance, not bias: the per-frame SD is the largest of any arm"),
            ("feature_noise", ei,
             "sub-pixel localisation noise, at the level the sub-pixel RANSAC residuals imply")):
        # These two are NOT resolvable above their own Monte Carlo error, and multiplying an
        # unresolvable per-frame mean by 6,000 frames would turn sampling noise into a 10-20 %
        # "explanation". The point estimate is therefore ZERO and what is reported instead is the
        # one-sided bound `3 SE x n` -- the most these arms could contribute and still be consistent
        # with the measurement.
        res = abs(row["bias_log_scale"]) > 3 * row["se_log_scale"]
        out["mechanisms"][name] = {
            "parity": odd_even(parity,
                               ("reliefhi" if ("reliefhi+", up) in parity else "relief")
                               if name == "homogeneous_relief"
                               else ("noisehi" if ("noisehi+", up) in parity else "noise"), up),
            "per_frame": row["bias_log_scale"], "se": row["se_log_scale"],
            "per_frame_sd": row["sd_log_scale"],
            "resolvable": bool(res),
            "accumulated_prediction": row["bias_log_scale"] * n if res else 0.0,
            "explained_fraction": (row["bias_log_scale"] * n / observed) if res else 0.0,
            "upper_bound_fraction": abs(3 * row["se_log_scale"] * n / observed),
            "note": note,
        }

    explained = sum(m["explained_fraction"] for m in out["mechanisms"].values())
    explained_pos = sum(max(0.0, m["explained_fraction"]) for m in out["mechanisms"].values())
    bound = sum(m.get("upper_bound_fraction", abs(m["explained_fraction"]))
                for m in out["mechanisms"].values())
    out["total_explained_fraction"] = explained
    out["total_explained_fraction_ignoring_sign"] = explained_pos
    out["total_upper_bound_fraction"] = bound
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)

    parity = load_parity(out / "parity.csv")
    mechanisms = []
    with (out / "mechanisms.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            row = {}
            for k, v in r.items():
                row[k] = v if k in ("condition", "model", "relief", "spread") else (
                    v.strip().lower() == "true" if k == "correct_distortion" else float(v))
            mechanisms.append(row)

    arms = [budget(parity, mechanisms, w, m) for w, m in ss.REAL_ARMS]
    (out / "scale_bias_budget.json").write_text(json.dumps({"record": "EXP-VO-011", "phase": 8,
                                                            "arms": arms}, indent=1))

    print("=== parity of every candidate mechanism (odd survives time reversal, even does not) ===")
    print(f"{'mechanism':<18} {'model':<12} {'forward':>12} {'reversed':>12} {'odd':>12} "
          f"{'even':>12} {'verdict':>12}")
    for mech in ("slope", "dist", "dtilt", "noise", "relief"):
        for model in ("HOMOGRAPHY", "AFFINE", "SIMILARITY"):
            d = odd_even(parity, mech, model)
            print(f"{mech:<18} {model:<12} {d['forward']:>12.3e} {d['reversed']:>12.3e} "
                  f"{d['odd']:>12.3e} {d['even']:>12.3e} {d['verdict']:>12}")

    print("\n=== the scale-bias budget: accumulated drift each mechanism can account for ===")
    for a_ in arms:
        print(f"\n{a_['window']} {a_['model']}: observed accumulated log-scale "
              f"{a_['observed_accum_log_scale']:+.4f} over {a_['n_frames']} frames")
        for name, m in a_["mechanisms"].items():
            bound = ("" if "upper_bound_fraction" not in m
                     else f"  (<= {100 * m['upper_bound_fraction']:.1f} % bound, not resolvable)")
            print(f"   {name:<20} predicted {m['accumulated_prediction']:>+9.4f}  "
                  f"= {100 * m['explained_fraction']:>7.1f} %   "
                  f"[{m['parity']['verdict']}]{bound}")
        print(f"   {'TOTAL':<20} {'':>19}  = {100 * a_['total_explained_fraction']:>7.1f} %  "
              f"({100 * a_['total_explained_fraction_ignoring_sign']:.1f} % ignoring sign, "
              f"<= {100 * a_['total_upper_bound_fraction']:.1f} % as an upper bound)")
    print(f"\nwrote {out / 'scale_bias_budget.json'}")


if __name__ == "__main__":
    main()
