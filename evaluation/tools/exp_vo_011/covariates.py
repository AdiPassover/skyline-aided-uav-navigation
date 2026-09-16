"""EXP-VO-011 Phase 6: does the real per-frame log-scale increment covary with what each candidate
mechanism predicts it should?

Three things, in decreasing order of how much they decide:

1. **H5c, the direction-independence test.** Every mechanism this record was commissioned to test --
   lens distortion, terrain relief, and the principal-point/Jacobian-evaluation offset -- is, to
   leading order, ODD in image position: its apparent scale goes as `(centroid . travel)` or
   `(travel . grad z)`. Both vanish on a symmetric feature distribution and both FLIP SIGN WITH
   HEADING. So if the measured bias is the same sign and roughly the same size in every heading
   quadrant, none of them can be its dominant cause -- independently of what any synthetic sweep at
   one heading shows. Criterion M8: same sign in all four quadrants and max/min |bias| < 3.

2. **H5a / H5b, covariation with the physical quantities.** Partial correlation of the increment
   against LiDAR in-footprint depth spread, LiDAR height above the imaged surface, and airframe
   tilt, **controlling for elapsed time and cumulative travelled distance** -- because accumulated
   anything correlates with distance, and the increments must be shown to move with the covariate
   and not merely with the flight going on. Criterion M5.

3. **H6b, the errors-in-variables prediction.** Attenuation of the fitted linear block predicts a
   bias `~ sigma^2 / V` with `V` the spatial variance of the correspondence set: direction-
   independent, of the observed (positive) sign, and larger when the inlier set is spatially narrow.
   The spatial spread is not in the committed run records, so the proxy available here is the inlier
   count; the direct measurement needs the probe of Phase 6b.

Significance uses an AR(1) effective sample size throughout (the increments have lag-1
autocorrelation of order 0.1-0.4), and a normal approximation for the p-value, which at these
effective sample sizes -- hundreds to thousands -- is indistinguishable from the t distribution.

Output: `evaluations/exp-vo-011/covariates.json`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import scale_series as ss

CONTROLS = ("elapsed_s", "distance_m")
COVARIATES = ("lidar_spread_m", "lidar_spread_frac", "lidar_height_m", "tilt_deg",
              "roll_deg", "pitch_deg", "abs_d_tilt_deg", "d_pitch_deg",
              "flow_px", "anisotropy", "perspective", "inlier_count", "abs_rotation_deg")


def _norm_sf(z: float) -> float:
    """Two-sided normal tail. `math.erfc` only -- no scipy in this repository by design (R4)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def _residualise(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Residual of y after OLS on [1 | controls]."""
    X = np.column_stack([np.ones(len(y)), controls])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_correlation(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> dict:
    if float(np.std(x)) == 0.0:
        return {"r": float("nan"), "p": float("nan"), "n_eff": 0.0}
    ry, rx = _residualise(y, controls), _residualise(x, controls)
    sy, sx = float(np.std(ry)), float(np.std(rx))
    if sy == 0 or sx == 0:
        return {"r": float("nan"), "p": float("nan"), "n_eff": 0.0}
    r = float(np.mean((ry - ry.mean()) * (rx - rx.mean())) / (sy * sx))
    # Effective sample size from the residual autocorrelations of both series (Bartlett/Quenouille).
    ry_rho, rx_rho = ss.lag1(ry), ss.lag1(rx)
    n = len(y)
    f = (1.0 - ry_rho * rx_rho) / (1.0 + ry_rho * rx_rho) if abs(ry_rho * rx_rho) < 1 else 1.0
    n_eff = max(4.0, n * f)
    z = r * math.sqrt(max(n_eff - 3.0, 1.0)) / math.sqrt(max(1.0 - r * r, 1e-12))
    return {"r": r, "p": _norm_sf(z), "n_eff": float(n_eff), "z": z,
            "rho_y": ry_rho, "rho_x": rx_rho}


def pearson(y: np.ndarray, x: np.ndarray) -> float:
    """Guarded: a structurally constant covariate (the affine and similarity models' perspective
    magnitude is identically 0) has no correlation, and must report that rather than a warning."""
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y, x)[0, 1])


def spearman(y: np.ndarray, x: np.ndarray) -> float:
    if float(np.std(y)) == 0.0 or float(np.std(x)) == 0.0:
        return float("nan")
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        return r
    ry, rx = rank(y), rank(x)
    return float(np.corrcoef(ry, rx)[0, 1])


def heading_test(y: np.ndarray, heading_deg: np.ndarray,
                 controls: np.ndarray | None = None) -> dict:
    """H5c / M8. Bias per heading quadrant, plus the first-harmonic decomposition.

    The harmonic fit `y ~ c0 + c1 cos(psi) + c2 sin(psi)` separates the part of the bias that is
    heading-INVARIANT (`c0`) from the part that rotates with the aircraft (`sqrt(c1^2 + c2^2)`).
    An odd-in-position mechanism contributes only to the second; anything direction-independent
    contributes only to the first.
    """
    psi = np.radians(heading_deg)
    cols = [np.ones(len(y)), np.cos(psi), np.sin(psi)]
    names = ["const", "cos", "sin"]
    if controls is not None:
        # Heading and time are confounded on a legged flight -- the aircraft flies one leg then
        # another -- so the harmonic is fitted alongside the same two controls the partial
        # correlations use, and the constant is read off that fit.
        for j in range(controls.shape[1]):
            c = controls[:, j]
            sd = float(np.std(c))
            cols.append((c - c.mean()) / sd if sd > 0 else c - c.mean())
            names.append(f"control{j}")
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c0, c1, c2 = float(beta[0]), float(beta[1]), float(beta[2])
    amp = math.hypot(c1, c2)

    # The decisive accumulation split: how much of the run's total drift comes from the
    # heading-INVARIANT constant, and how much from the part that rotates with the aircraft and
    # therefore ought to cancel over varied headings.
    harmonic_series = c1 * np.cos(psi) + c2 * np.sin(psi)
    total = float(np.sum(y))
    accum_const = c0 * len(y)
    accum_harm = float(np.sum(harmonic_series))

    quads = {}
    signs, mags = [], []
    for q in range(4):
        lo, hi = 90.0 * q, 90.0 * (q + 1)
        m = (heading_deg >= lo) & (heading_deg < hi)
        if m.sum() < 30:
            quads[f"{int(lo)}-{int(hi)}"] = {"n": int(m.sum()), "mean": None}
            continue
        st = ss.describe(y[m])
        quads[f"{int(lo)}-{int(hi)}"] = {"n": st["n"], "mean": st["mean"], "t_eff": st["t_eff"],
                                         "mad_sigma": st["mad_sigma"]}
        signs.append(math.copysign(1.0, st["mean"]))
        mags.append(abs(st["mean"]))

    populated = len(mags)
    same_sign = populated > 0 and len(set(signs)) == 1
    ratio = (max(mags) / min(mags)) if populated > 1 and min(mags) > 0 else float("inf")
    return {
        "quadrants": quads,
        "populated_quadrants": populated,
        "same_sign_all": bool(same_sign),
        "max_over_min_abs": ratio,
        "m8_heading_invariant": bool(same_sign and ratio < 3.0 and populated >= 3),
        "harmonic": {"constant": c0, "cos": c1, "sin": c2, "amplitude": amp,
                     "amplitude_over_constant": (amp / abs(c0)) if c0 != 0 else float("inf")},
        "accumulation_split": {
            "total_log_scale": total,
            "from_constant": accum_const,
            "from_harmonic": accum_harm,
            "residual": total - accum_const - accum_harm,
            "constant_fraction": accum_const / total if total != 0 else float("nan"),
            "harmonic_fraction": accum_harm / total if total != 0 else float("nan"),
        },
    }


def windowed(y: np.ndarray, cov: dict, size: int = 100) -> dict:
    """Fixed-window means, so the LiDAR covariate (sampled at ~10 s) is compared like with like."""
    n = (len(y) // size) * size
    if n < 2 * size:
        return {}
    ybar = y[:n].reshape(-1, size).mean(axis=1)
    out = {"window_frames": size, "n_windows": int(len(ybar)),
           "mean_by_window": [float(v) for v in ybar]}
    for k in COVARIATES + CONTROLS:
        xb = cov[k][:n].reshape(-1, size).mean(axis=1)
        out[k] = {"pearson": pearson(ybar, xb),
                  "spearman": spearman(ybar, xb),
                  "partial": partial_correlation(
                      ybar, xb, np.column_stack([cov[c][:n].reshape(-1, size).mean(axis=1)
                                                 for c in CONTROLS]))}
    return out


def analyse(window: str, model: str) -> dict:
    arm = ss.load_arm(window, model)
    cov = ss.covariates_for(arm)
    y = arm["inc_log_scale"]
    controls = np.column_stack([cov[c] for c in CONTROLS])

    per_frame = {}
    for k in COVARIATES:
        per_frame[k] = {
            "pearson": pearson(y, cov[k]),
            "spearman": spearman(y, cov[k]),
            "partial": partial_correlation(y, cov[k], controls),
        }
    # The controls themselves, so "everything accumulates with distance" is visible rather than
    # assumed: a raw correlation with distance that survives is a finding, one that does not is a
    # warning about every other raw correlation in the table.
    for k in CONTROLS:
        per_frame[k] = {"pearson": pearson(y, cov[k]),
                        "spearman": spearman(y, cov[k]), "partial": None}

    return {
        "window": window, "model": model, "n": int(len(y)),
        "bias_mean": float(np.mean(y)),
        "heading": heading_test(y, cov["heading_deg"], controls),
        "per_frame": per_frame,
        "windowed_100": windowed(y, cov, 100),
        "covariate_ranges": {k: {"min": float(cov[k].min()), "median": float(np.median(cov[k])),
                                 "max": float(cov[k].max())} for k in COVARIATES},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arms = [analyse(w, m) for w, m in ss.REAL_ARMS + ss.CONTROL_ARMS]
    (out / "covariates.json").write_text(json.dumps({"record": "EXP-VO-011", "phase": 6,
                                                     "arms": arms}, indent=1))

    print(f"{'arm':<28} {'bias':>10} {'c0':>10} {'harm amp':>10} {'amp/c0':>8} "
          f"{'same':>5} {'max/min':>8} {'M8':>6} {'accum':>8} {'const%':>8} {'harm%':>8}")
    print("-" * 118)
    for a in arms:
        h = a["heading"]
        sp = h["accumulation_split"]
        print(f"{a['window'] + '-' + a['model']:<28} {a['bias_mean']:>10.3e} "
              f"{h['harmonic']['constant']:>10.3e} {h['harmonic']['amplitude']:>10.3e} "
              f"{h['harmonic']['amplitude_over_constant']:>8.2f} "
              f"{str(h['same_sign_all']):>5} "
              f"{h['max_over_min_abs']:>8.2f} {str(h['m8_heading_invariant']):>6} "
              f"{sp['total_log_scale']:>8.3f} {100 * sp['constant_fraction']:>7.1f}% "
              f"{100 * sp['harmonic_fraction']:>7.1f}%")

    print(f"\n{'arm':<28} " + " ".join(f"{c[:13]:>14}" for c in COVARIATES))
    print("-" * 120)
    for a in arms:
        cells = []
        for c in COVARIATES:
            p = a["per_frame"][c]["partial"]
            cells.append(f"{p['r']:>+8.3f}{'*' if p['p'] < 0.01 else ' '}     ")
        print(f"{a['window'] + '-' + a['model']:<28} " + " ".join(cells))
    print("\n(partial r, controlling for elapsed time and travelled distance; * = p < 0.01)")
    print(f"wrote {out / 'covariates.json'}")


if __name__ == "__main__":
    main()
