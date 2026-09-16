"""EXP-VO-011: test the mechanism the relief arm revealed, directly on the committed real data.

**The mechanism.** For a camera over a ground plane tilted by `s = tan(alpha)` relative to nadir,
pure horizontal translation `t` induces (LIT-VO-003 eq. 1, with `R = I`, `n` the tilted normal, and
`K = diag(f, f, 1)`) an image transform whose linear block is

    diag(1 - a s, 1),      a = t / (d sqrt(1 + s^2))

in the frame whose x axis is the slope direction -- the stretch is along ONE axis only, the one the
camera is climbing or descending along. Two consequences follow, and both are testable without
ground truth:

1. **The scale observable reads HALF the true relative-height rate, with the opposite sign.**
   `sqrt(|det|) = sqrt(1 - a s) ~ 1 - a s / 2`, so `log sqrt(|det|) = -a s / 2` while the true
   `d log h_local = -a s`. Half, because only one of the two singular values moved and `sqrt(det)`
   is their geometric mean. `ScaleBiasMonteCarlo`'s SLOPE arm measures exactly this ratio.
2. **The per-frame log-scale is half the per-frame log-anisotropy, and the stretch axis is aligned
   with the direction of travel.** That is a signature the shipped sidecar already records:
   `inc_anisotropy` and `inc_stretch_axis_deg` are written for every frame of every rigid run, and
   the image-frame travel direction is recoverable from the pose series (`flow_direction`).

This module tests (2). It is the difference between "a mechanism that could produce the observed
magnitude" and "the mechanism that is producing it".

**Sign convention.** `RigidMotionDecomposition.stretchAxisRad` is the major axis of the stretch,
modulo pi. If the transform is a compression along travel, the MAJOR axis is perpendicular to
travel; if an extension along travel, the major axis is parallel. `signed_half_log_anisotropy`
therefore carries `-cos(2 * delta)` with `delta` the axis-minus-travel angle, so that a compression
along travel (delta = 90 deg) contributes positively -- the sign the observed bias has.
"""
from __future__ import annotations

import json
import math

import numpy as np

import flow_direction as fd
import scale_series as ss


def signature(arm: dict) -> dict:
    y = arm["inc_log_scale"]
    aniso = arm["inc_anisotropy"]
    axis = np.radians(arm["inc_stretch_axis_deg"])
    fl = fd.image_flow(arm)
    travel = np.radians(fl["angle_deg"])
    flow = fl["flow_px"]

    # Axis is defined modulo pi, so only 2*delta is meaningful.
    delta = axis - travel
    c2 = np.cos(2 * delta)

    log_aniso = np.log(np.maximum(aniso, 1.0))
    predicted = -0.5 * log_aniso * c2

    ok = np.isfinite(predicted) & np.isfinite(y)
    denom = float(np.std(y[ok])) * float(np.std(predicted[ok]))
    corr = float(np.mean((y[ok] - y[ok].mean()) * (predicted[ok] - predicted[ok].mean())) / denom) \
        if denom > 0 else float("nan")

    # Where does the stretch axis sit relative to travel? Under the slope mechanism it should
    # concentrate at 90 degrees (compression along travel) or 0 (extension along travel), not be
    # uniform -- and its concentration is measured on the doubled angle, since the axis is mod pi.
    dbl = fd.circular_stats(np.degrees(2 * delta) % 360.0, weights=flow)

    return {
        "n": int(ok.sum()),
        "mean_observed": float(np.mean(y)),
        "mean_predicted_from_anisotropy": float(np.mean(predicted[ok])),
        "ratio_observed_over_predicted": (float(np.mean(y)) / float(np.mean(predicted[ok]))
                                          if float(np.mean(predicted[ok])) else float("nan")),
        "correlation": corr,
        "median_anisotropy": float(np.median(aniso)),
        "mean_half_log_anisotropy": float(np.mean(0.5 * log_aniso)),
        "axis_vs_travel_doubled": dbl,
        "fraction_axis_within_30deg_of_perpendicular": float(np.mean(np.abs(c2 + 1.0) < 0.5)),
        "fraction_axis_within_30deg_of_parallel": float(np.mean(np.abs(c2 - 1.0) < 0.5)),
    }


def main() -> None:
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = {}
    print(f"{'arm':<28} {'observed':>11} {'from aniso':>12} {'ratio':>8} {'corr':>7} "
          f"{'med aniso':>10} {'axis R':>8} {'perp frac':>10}")
    print("-" * 100)
    for w, m in ss.REAL_ARMS + ss.CONTROL_ARMS:
        arm = ss.load_arm(w, m)
        s = signature(arm)
        rows[f"{w}-{m}"] = s
        print(f"{w + '-' + m:<28} {s['mean_observed']:>11.3e} "
              f"{s['mean_predicted_from_anisotropy']:>12.3e} "
              f"{s['ratio_observed_over_predicted']:>8.3f} {s['correlation']:>7.3f} "
              f"{s['median_anisotropy']:>10.6f} "
              f"{s['axis_vs_travel_doubled']['resultant_R']:>8.3f} "
              f"{s['fraction_axis_within_30deg_of_perpendicular']:>10.3f}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "anisotropy_axis.json").write_text(json.dumps(
        {"record": "EXP-VO-011", "arms": rows}, indent=1))
    print(f"\nwrote {out / 'anisotropy_axis.json'}")


if __name__ == "__main__":
    main()
