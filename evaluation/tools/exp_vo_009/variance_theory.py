"""`LIT-VO-005` section 4: verify the closed-form rotation-variance ratio between an affine fit
read through its polar rotation and a direct Sim(2) fit, on the *same* correspondences.

The identity under test, derived in `LIT-VO-005` eq. (8):

    Var(theta_affine) / Var(theta_similarity)  =  tr(M) tr(M^-1) / 4  =  (2 + k + 1/k) / 4  >= 1

where `M` is the sample's centred second-moment matrix and `k` its condition number, with equality
iff the sample is isotropic.

This is a check of algebra, not an experiment: no VO, no imagery, no flight data, no estimator. It
is run as `EXP-VO-009` procedure gate 1 and its numbers are quoted in `LIT-VO-005` section 4.

Two modes:

* ``--mode per-sample`` (the honest test) fixes the points and varies only the noise, which is the
  regime the identity is derived in. This is what must agree.
* ``--mode marginal`` re-draws the points every trial. The identity then does **not** describe the
  ratio of marginal variances, because `E[(2+k+1/k)/4]` is dominated by rare near-collinear samples
  while the marginal variance is not. Reported deliberately, so that the divergence is on the record
  rather than discovered later as a contradiction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def polar_rotation(J: np.ndarray) -> float:
    """Rotation of the polar factor of a 2x2 matrix — the same closed form as
    ``RigidMotionDecomposition.set`` (``atan2(a21 - a12, a11 + a22)``)."""
    return float(np.arctan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1]))


def fit_affine_linear_part(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Least-squares affine, returning only the 2x2 linear block (BoofCV ``GenerateAffine2D``)."""
    A = np.c_[p, np.ones(len(p))]
    c1, *_ = np.linalg.lstsq(A, q[:, 0], rcond=None)
    c2, *_ = np.linalg.lstsq(A, q[:, 1], rcond=None)
    return np.array([[c1[0], c1[1]], [c2[0], c2[1]]])


def fit_similarity(p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """Umeyama's closed form specialised to 2-D (`LIT-VO-005` eq. 4). Returns ``(theta, scale)``."""
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    a = float((pc * qc).sum())
    b = float((pc[:, 0] * qc[:, 1] - pc[:, 1] * qc[:, 0]).sum())
    denom = float((pc ** 2).sum())
    return float(np.arctan2(b, a)), float(np.hypot(a, b) / denom)


def kappa(p: np.ndarray) -> float:
    pc = p - p.mean(axis=0)
    return float(np.linalg.cond(pc.T @ pc))


def _draw(rng, n, spread):
    return rng.normal(0.0, 1.0, (n, 2)) * np.asarray(spread, dtype=float)


def run(mode: str, ns, spreads, sigma: float, s_true: float, theta_true: float,
        trials: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    R = s_true * np.array([[np.cos(theta_true), -np.sin(theta_true)],
                           [np.sin(theta_true), np.cos(theta_true)]])
    rows = []
    for n in ns:
        for spread in spreads:
            ea = np.empty(trials)
            es = np.empty(trials)
            ks = np.empty(trials)
            p = _draw(rng, n, spread)
            base = p @ R.T
            for i in range(trials):
                if mode == "marginal":
                    p = _draw(rng, n, spread)
                    base = p @ R.T
                ks[i] = kappa(p)
                q = base + rng.normal(0.0, sigma, (n, 2))
                ea[i] = polar_rotation(fit_affine_linear_part(p, q)) - theta_true
                es[i] = fit_similarity(p, q)[0] - theta_true
            k_used = float(ks[0]) if mode == "per-sample" else float(np.median(ks))
            rows.append({
                "mode": mode, "n": int(n), "spread": list(map(float, spread)),
                "sigma": sigma, "trials": int(trials),
                "kappa": k_used,
                "sd_affine_rad": float(ea.std(ddof=1)),
                "sd_similarity_rad": float(es.std(ddof=1)),
                "variance_ratio_observed": float(ea.var(ddof=1) / es.var(ddof=1)),
                "variance_ratio_predicted": float((2 + k_used + 1 / k_used) / 4),
                "mean_predicted_over_samples": float(np.mean((2 + ks + 1 / ks) / 4)),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("per-sample", "marginal"), default="per-sample")
    ap.add_argument("--trials", type=int, default=40000)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ns = (3, 5, 30) if a.mode == "per-sample" else (3, 8, 50, 2000)
    spreads = [(1.0, 1.0)] if a.mode == "per-sample" else [(1.0, 1.0), (3.0, 1.0)]
    rows = run(a.mode, ns, spreads, a.sigma, 1.0, 0.05, a.trials, a.seed)

    hdr = f"{'n':>6} {'kappa':>10} {'sd_aff':>10} {'sd_sim':>10} {'ratio_obs':>10} {'ratio_pred':>11}"
    print(f"mode={a.mode}  sigma={a.sigma}  trials={a.trials}")
    print(hdr)
    worst = 0.0
    for r in rows:
        print(f"{r['n']:>6} {r['kappa']:>10.4f} {r['sd_affine_rad']:>10.5f} "
              f"{r['sd_similarity_rad']:>10.5f} {r['variance_ratio_observed']:>10.4f} "
              f"{r['variance_ratio_predicted']:>11.4f}")
        if a.mode == "per-sample":
            rel = abs(r["variance_ratio_observed"] - r["variance_ratio_predicted"]) \
                / r["variance_ratio_predicted"]
            worst = max(worst, rel)
    if a.mode == "per-sample":
        print(f"\nworst relative disagreement with LIT-VO-005 eq. (8): {worst * 100:.3f} %  "
              f"(gate: <= 1 %)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=2))
    return 0 if (a.mode != "per-sample" or worst <= 0.01) else 1


if __name__ == "__main__":
    raise SystemExit(main())
