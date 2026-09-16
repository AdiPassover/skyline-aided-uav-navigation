"""Airframe-attitude association analysis for `EXP-003`.

Answers one pre-registered question on data that already exists: **is airframe attitude
associated with the stitching VO's per-segment scale instability and horizontal error?**
`EXP-003` fixes the hypotheses, the seven tests, the aggregation rules and the decision
thresholds; this module implements them and nothing else.

**Association only.** The data are observational -- a recorded third-party flight in which
attitude was never manipulated, and in which roll co-varies with heading, turn rate and scene
content by flight physics. Nothing here supports a causal claim, and the naming throughout says
"association" rather than "effect" for that reason (`EXP-003` threats 1-2).

**Autocorrelation is the main statistical hazard and is handled explicitly.** At 10 Hz over a
3 m/s flight, adjacent frames are near-duplicates; treating 6,158 frames as 6,158 independent
samples would inflate significance by one to two orders of magnitude and manufacture a false
positive. The defences, all pre-registered rather than chosen after seeing results:

  * the **segment** is the primary analysis unit (44 in EXP-002 B, not 6,158 frames);
  * significance comes from a **permutation** null, never an analytic formula assuming
    independence;
  * confidence intervals come from a **moving-block bootstrap** whose block length is set from
    the measured decorrelation time;
  * an **effective sample size** accompanies every correlation, and a test with `n_eff < 10` is
    reported as underpowered rather than as a null;
  * a **circular-shift null** destroys the true temporal correspondence while preserving both
    series' autocorrelation exactly -- a correlation that survives it is an autocorrelation
    artifact, not a relationship. This is the strongest guard in the design.

No scipy (`naveval` is kept at numpy +
matplotlib), which is not a constraint worth regretting here: a permutation null is more
defensible than an analytic one for this data anyway.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_SEED = 20260814
DEFAULT_N_RESAMPLES = 10_000

# `EXP-003` "Analysis configuration". Fixed before the data existed; changing one of these
# after seeing a result would invalidate the pre-registration, so they live here as named
# constants rather than as call-site defaults scattered through the analysis.
RATE_SMOOTH_WINDOW_S = 0.25   # tilt-rate smoothing before differentiation
RATE_AGGREGATE_Q = 90         # p90, not max -- a rate hypothesis is about excursions, but a
                              # single EKF noise spike must not drive the result
EVENT_WINDOW_S = 2.0          # +/- window around restart/recenter events for test T6
MIN_EFFECTIVE_N = 10          # below this a test is "underpowered", not "null"


# ------------------------------------------------------------------------------------------
# Attitude series
# ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AttitudeSeries:
    """Native-rate attitude, as parallel arrays. Times are on the dataset's GPS-UTC clock."""
    timestamp_s: np.ndarray
    roll_deg: np.ndarray
    pitch_deg: np.ndarray
    tilt_deg: np.ndarray
    yaw_compass_deg: np.ndarray

    def __len__(self) -> int:
        return int(self.timestamp_s.shape[0])

    @property
    def measured_rate_hz(self) -> float:
        if len(self) < 2:
            return 0.0
        span = float(self.timestamp_s[-1] - self.timestamp_s[0])
        return (len(self) - 1) / span if span > 0 else 0.0

    @property
    def max_gap_s(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(np.max(np.diff(self.timestamp_s)))


def load_attitude_csv(path: Path | str) -> AttitudeSeries:
    """Read an `attitude.csv` sidecar (`ingest_mars_lvig.write_attitude_sidecar`)."""
    path = Path(path)
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((
                float(row["timestamp_s"]), float(row["roll_deg"]), float(row["pitch_deg"]),
                float(row["tilt_deg"]), float(row["yaw_compass_deg"]),
            ))
    if not rows:
        raise ValueError(f"No attitude samples in {path}")
    rows.sort(key=lambda r: r[0])
    a = np.asarray(rows, dtype=np.float64)
    return AttitudeSeries(
        timestamp_s=a[:, 0], roll_deg=a[:, 1], pitch_deg=a[:, 2],
        tilt_deg=a[:, 3], yaw_compass_deg=a[:, 4],
    )


def tilt_rate_dps(series: AttitudeSeries, smooth_window_s: float = RATE_SMOOTH_WINDOW_S) -> np.ndarray:
    """|d(tilt)/dt| in deg/s, computed on the NATIVE series.

    Differentiation happens before any resampling, deliberately: differentiating an already
    interpolated 10 Hz series would alias away most of the rate signal that `EXP-003`'s H3 is
    about. A short boxcar smooth precedes the difference because differentiating a 100 Hz EKF
    output amplifies its high-frequency noise (`EXP-003` threat 8).

    Returns an array the same length as the input, sampled at the input's own timestamps.
    """
    n = len(series)
    if n < 2:
        return np.zeros(n)
    dt_median = float(np.median(np.diff(series.timestamp_s)))
    width = max(1, int(round(smooth_window_s / dt_median))) if dt_median > 0 else 1
    if width > 1:
        kernel = np.ones(width) / width
        # Edge-replicate before convolving. `np.convolve(..., mode="same")` zero-pads, which
        # drags the first and last width/2 samples toward zero and then shows up as a large
        # spurious gradient at both ends -- on a constant-tilt input it manufactured a 16 deg/s
        # "rate" out of nothing (caught by test_constant_tilt_has_zero_rate). Segment-boundary
        # rate values are exactly what EXP-003's H3 aggregates, so this would have biased the
        # result rather than merely looking untidy.
        pad = width // 2
        padded = np.pad(series.tilt_deg, pad, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="same")[pad:pad + n]
    else:
        smoothed = series.tilt_deg
    return np.abs(np.gradient(smoothed, series.timestamp_s))


def interpolate_to(
    source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    """Linear interpolation onto `target_times`, NaN outside the source span.

    Not `np.interp` alone: `np.interp` clamps to the endpoint values outside the range, which
    would silently fabricate attitude for frames the attitude series does not cover. Out-of-span
    targets become NaN so they are visibly missing and get excluded downstream (the same
    "countable, not silently dropped" discipline `contracts/dataset.md` applies to ground truth).
    """
    out = np.interp(target_times, source_times, values)
    outside = (target_times < source_times[0]) | (target_times > source_times[-1])
    out = np.asarray(out, dtype=np.float64)
    out[outside] = np.nan
    return out


# ------------------------------------------------------------------------------------------
# Rank statistics -- no scipy
# ------------------------------------------------------------------------------------------

def rankdata(a: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, matching `scipy.stats.rankdata`'s default method."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, a.shape[0] + 1, dtype=np.float64)
    # Average the ranks within each group of equal values.
    sorted_a = a[order]
    i = 0
    while i < sorted_a.shape[0]:
        j = i
        while j + 1 < sorted_a.shape[0] and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    return ranks


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation. Returns NaN when either input has zero variance."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt(float(xc @ xc) * float(yc @ yc))
    if denom == 0.0:
        return float("nan")
    return float((xc @ yc) / denom)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho = Pearson on ranks. Chosen over Pearson because `EXP-003` expects
    monotone-but-nonlinear relationships and because the attitude distribution is heavy-tailed
    (`LIT-006`: roll mean 2.15 deg, sd 4.87, max 9.06)."""
    return pearson_r(rankdata(x), rankdata(y))


def lag1_autocorr(a: np.ndarray) -> float:
    """Lag-1 autocorrelation, in series order. NaN-safe only for finite input."""
    a = np.asarray(a, dtype=np.float64)
    if a.shape[0] < 3:
        return 0.0
    return pearson_r(a[:-1], a[1:])


def effective_sample_size(x: np.ndarray, y: np.ndarray) -> float:
    """`n_eff = n (1 - r1x r1y) / (1 + r1x r1y)`, the standard correction for correlating two
    autocorrelated series (Bartlett). Reported alongside every rho so that a correlation over
    highly-dependent samples cannot be read as if it came from independent ones.

    Clamped to [1, n]: a strongly *negatively* autocorrelated pair can drive the formula above n,
    which is not a meaningful claim to make here.
    """
    n = float(np.asarray(x).shape[0])
    r = lag1_autocorr(x) * lag1_autocorr(y)
    if not math.isfinite(r):
        return n
    denom = 1.0 + r
    if denom <= 0:
        return 1.0
    return float(min(max(n * (1.0 - r) / denom, 1.0), n))


def permutation_pvalue(
    x: np.ndarray, y: np.ndarray, n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> float:
    """Two-sided permutation p-value for Spearman rho, permuting `y`'s labels.

    Uses the (r + 1) / (n + 1) estimator rather than r / n, which cannot return an impossible
    p = 0 from a finite number of resamples.
    """
    rho_obs = spearman_rho(x, y)
    if not math.isfinite(rho_obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64)
    count = 0
    for _ in range(n_resamples):
        rho_perm = spearman_rho(x, rng.permutation(y))
        if math.isfinite(rho_perm) and abs(rho_perm) >= abs(rho_obs):
            count += 1
    return (count + 1) / (n_resamples + 1)


def moving_block_bootstrap_ci(
    x: np.ndarray, y: np.ndarray, block_length: int,
    n_resamples: int = DEFAULT_N_RESAMPLES, seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for Spearman rho by moving-block bootstrap over the paired series.

    Blocks of *consecutive* pairs are resampled with replacement, so within-block dependence
    survives into every resample -- an i.i.d. pair bootstrap would destroy exactly the structure
    that makes the naive CI too narrow.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    block_length = max(1, min(int(block_length), n))
    n_blocks = int(math.ceil(n / block_length))
    max_start = n - block_length
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
        rho = spearman_rho(x[idx], y[idx])
        if math.isfinite(rho):
            rhos.append(rho)
    if not rhos:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(rhos, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def decorrelation_block_length(x: np.ndarray, y: np.ndarray) -> int:
    """Block length for the bootstrap: `ceil(tau)` from the larger of the two lag-1
    decorrelation times, `tau = -1 / ln|r1|`, floored at 1 and capped at n // 4 so a resample
    still contains at least four blocks."""
    n = int(np.asarray(x).shape[0])
    taus = []
    for a in (x, y):
        r = abs(lag1_autocorr(a))
        if 0.0 < r < 1.0:
            taus.append(-1.0 / math.log(r))
    tau = max(taus) if taus else 1.0
    return int(min(max(1, math.ceil(tau)), max(1, n // 4)))


def circular_shift_null(
    x: np.ndarray, y: np.ndarray, n_shifts: int = 200,
) -> dict:
    """Null distribution of rho under circular shifts of `y`.

    A circular shift destroys the true temporal correspondence between the two series while
    preserving each one's autocorrelation *exactly*. If the observed rho is unremarkable against
    this null, it is an artifact of both series being smooth in time, not a relationship --
    which is the specific false positive this analysis is most at risk of.

    Returns the observed rho, the null's max |rho|, and the fraction of shifts at least as
    extreme as the observation (a p-value-like quantity, reported as such and not as a p-value).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    rho_obs = spearman_rho(x, y)
    # Skip shift 0 (the observation itself) and, at both ends, shifts so small they barely move.
    shifts = np.unique(np.linspace(1, n - 1, num=min(n_shifts, max(1, n - 1)), dtype=int))
    null = []
    for s in shifts:
        rho = spearman_rho(x, np.roll(y, int(s)))
        if math.isfinite(rho):
            null.append(rho)
    if not null:
        return {"rho_observed": rho_obs, "null_max_abs_rho": float("nan"),
                "fraction_null_as_extreme": float("nan"), "n_shifts": 0}
    null_arr = np.asarray(null)
    frac = float(np.mean(np.abs(null_arr) >= abs(rho_obs))) if math.isfinite(rho_obs) else float("nan")
    return {
        "rho_observed": rho_obs,
        "null_max_abs_rho": float(np.max(np.abs(null_arr))),
        "fraction_null_as_extreme": frac,
        "n_shifts": int(null_arr.shape[0]),
    }


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, order preserved.

    The family is `EXP-003`'s seven pre-registered tests, fixed before execution precisely so it
    cannot be trimmed to whichever subset looks best afterwards. NaN p-values (a degenerate test)
    stay NaN and do not consume a step.
    """
    idx = [i for i, p in enumerate(pvalues) if p is not None and math.isfinite(p)]
    m = len(idx)
    out: list[float] = [float("nan")] * len(pvalues)
    if m == 0:
        return out
    ordered = sorted(idx, key=lambda i: pvalues[i])
    running = 0.0
    for k, i in enumerate(ordered):
        adj = (m - k) * pvalues[i]
        running = max(running, adj)          # enforce monotonicity
        out[i] = float(min(1.0, running))
    return out


# ------------------------------------------------------------------------------------------
# The pre-registered test harness
# ------------------------------------------------------------------------------------------

@dataclass
class CorrelationResult:
    """One of `EXP-003`'s seven tests, with everything needed to judge it in one row."""
    test_id: str
    description: str
    hypothesis: str
    n: int
    n_effective: float
    rho: float
    p_permutation: float
    p_holm: Optional[float]
    ci_low: float
    ci_high: float
    block_length: int
    null_shift_max_abs_rho: float
    null_shift_fraction_as_extreme: float
    underpowered: bool

    @property
    def survives_circular_shift(self) -> bool:
        """True when the observation is not reproducible by autocorrelation alone."""
        f = self.null_shift_fraction_as_extreme
        return math.isfinite(f) and f < 0.05


def run_correlation_test(
    test_id: str, description: str, hypothesis: str,
    x: np.ndarray, y: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES, seed: int = DEFAULT_SEED,
) -> CorrelationResult:
    """Compute one pre-registered test in full: rho, permutation p, block-bootstrap CI,
    effective n, and the circular-shift null. Holm adjustment is applied later, across the
    whole family, by `apply_holm`."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = int(x.shape[0])
    if n < 3:
        return CorrelationResult(
            test_id=test_id, description=description, hypothesis=hypothesis, n=n,
            n_effective=float(n), rho=float("nan"), p_permutation=float("nan"), p_holm=None,
            ci_low=float("nan"), ci_high=float("nan"), block_length=0,
            null_shift_max_abs_rho=float("nan"), null_shift_fraction_as_extreme=float("nan"),
            underpowered=True,
        )
    block = decorrelation_block_length(x, y)
    n_eff = effective_sample_size(x, y)
    ci_low, ci_high = moving_block_bootstrap_ci(x, y, block, n_resamples, seed)
    shift = circular_shift_null(x, y)
    return CorrelationResult(
        test_id=test_id, description=description, hypothesis=hypothesis, n=n,
        n_effective=n_eff, rho=spearman_rho(x, y),
        p_permutation=permutation_pvalue(x, y, n_resamples, seed), p_holm=None,
        ci_low=ci_low, ci_high=ci_high, block_length=block,
        null_shift_max_abs_rho=shift["null_max_abs_rho"],
        null_shift_fraction_as_extreme=shift["fraction_null_as_extreme"],
        underpowered=n_eff < MIN_EFFECTIVE_N,
    )


def apply_holm(results: list[CorrelationResult]) -> list[CorrelationResult]:
    """Fill `p_holm` across the family, in place, and return the same list."""
    adjusted = holm_adjust([r.p_permutation for r in results])
    for r, p in zip(results, adjusted):
        r.p_holm = None if math.isnan(p) else p
    return results


def write_correlations_csv(results: Sequence[CorrelationResult], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(results[0]).keys()) + ["survives_circular_shift"] if results else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = asdict(r)
            row["survives_circular_shift"] = r.survives_circular_shift
            w.writerow(row)
    return path


# ------------------------------------------------------------------------------------------
# D1 -- the falsification test (H4)
# ------------------------------------------------------------------------------------------

def relative_spread(a: np.ndarray) -> float:
    """sd / |mean|, the scale-free spread `EXP-002` reports for `scale_ratio_to_global`
    (17.5 % in Execution A, 27.7 % in B). NaN when the mean is zero."""
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.shape[0] < 2:
        return float("nan")
    m = float(np.mean(a))
    if m == 0.0:
        return float("nan")
    return float(np.std(a, ddof=1) / abs(m))


def tercile_spread_analysis(tilt: np.ndarray, scale_ratio: np.ndarray) -> dict:
    """`EXP-003` D1: does the scale instability survive controlling for tilt?

    Splits segments into tilt terciles and reports the spread of `scale_ratio_to_global` within
    each. The decision statistic is `spread_ratio` = spread(lowest-tilt tercile) / spread(all).
    Per `EXP-003` H4, **spread_ratio >= 0.8 means attitude is ruled out as the principal
    mechanism** behind the scale instability: the instability is essentially undiminished among
    the segments where the aircraft was flattest.

    Deliberately NOT a significance test. A tercile of 44 segments holds ~15, which is
    underpowered by construction; the effect size is what carries the argument, so the effect
    size is what is reported.
    """
    tilt = np.asarray(tilt, dtype=np.float64)
    scale_ratio = np.asarray(scale_ratio, dtype=np.float64)
    finite = np.isfinite(tilt) & np.isfinite(scale_ratio)
    tilt, scale_ratio = tilt[finite], scale_ratio[finite]
    n = int(tilt.shape[0])
    if n < 6:
        return {"n": n, "error": "fewer than 6 segments; terciles are not meaningful"}

    q1, q2 = np.percentile(tilt, [100 / 3, 200 / 3])
    groups = {
        "low": scale_ratio[tilt <= q1],
        "mid": scale_ratio[(tilt > q1) & (tilt <= q2)],
        "high": scale_ratio[tilt > q2],
    }
    overall_sd = relative_spread(scale_ratio)
    overall_iqr = float(np.subtract(*np.percentile(scale_ratio, [75, 25])))
    low_sd = relative_spread(groups["low"])
    low_iqr = (
        float(np.subtract(*np.percentile(groups["low"], [75, 25])))
        if groups["low"].shape[0] >= 2 else float("nan")
    )
    return {
        "n": n,
        "tilt_tercile_bounds_deg": [float(q1), float(q2)],
        "all_segments": {
            "relative_spread": overall_sd, "iqr": overall_iqr,
            "min": float(np.min(scale_ratio)), "max": float(np.max(scale_ratio)),
        },
        "terciles": {
            name: {
                "n": int(g.shape[0]),
                "relative_spread": relative_spread(g),
                "iqr": (float(np.subtract(*np.percentile(g, [75, 25])))
                        if g.shape[0] >= 2 else float("nan")),
                "min": float(np.min(g)) if g.size else float("nan"),
                "max": float(np.max(g)) if g.size else float("nan"),
            }
            for name, g in groups.items()
        },
        "spread_ratio_sd": (low_sd / overall_sd) if overall_sd else float("nan"),
        "spread_ratio_iqr": (low_iqr / overall_iqr) if overall_iqr else float("nan"),
        "h4_threshold": 0.8,
        "interpretation_rule": (
            "spread_ratio >= 0.8 supports EXP-003 H4: the scale instability persists among the "
            "flattest segments, so attitude is ruled OUT as the principal mechanism. "
            "spread_ratio < 0.5 implicates attitude. Between the two is inconclusive."
        ),
    }


# ------------------------------------------------------------------------------------------
# Ingest gates (EXP-003 Procedure step 4)
# ------------------------------------------------------------------------------------------

def attitude_distribution(series: AttitudeSeries) -> dict:
    """The full-window roll/pitch/tilt distribution that `LIT-006` §463 flagged as missing --
    it only ever had a four-short-window sample. Reported regardless of every hypothesis."""
    def stats(a: np.ndarray) -> dict:
        a = np.asarray(a, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {}
        return {
            "mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
            "min": float(np.min(a)), "max": float(np.max(a)),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
        }
    return {
        "n_samples": len(series),
        "measured_rate_hz": series.measured_rate_hz,
        "max_gap_s": series.max_gap_s,
        "roll_deg": stats(series.roll_deg),
        "pitch_deg": stats(series.pitch_deg),
        "tilt_deg": stats(series.tilt_deg),
        "abs_roll_deg": stats(np.abs(series.roll_deg)),
        "abs_pitch_deg": stats(np.abs(series.pitch_deg)),
    }


def check_ingest_gates(
    series: AttitudeSeries,
    window: tuple[float, float],
    gt_heading_residual_deg: Optional[float] = None,
) -> dict:
    """`EXP-003` gates G1-G3. **If any gate fails the analysis does not proceed** and the failure
    is itself the reported result -- a wrong decode must not become a finding.

    G2's `gt_heading_residual_deg` is supplied by the caller (mean shortest-angle difference
    between this series' `yaw_compass_deg` and the RTK-derived heading over straight-flight
    frames). It is the check that the quaternion was decoded with the right convention: this
    dataset has already produced one ~90-degree convention trap in `rtk_yaw`, so the decode is
    verified against something independent rather than trusted.
    """
    t0, t1 = window
    covers = bool(len(series) and series.timestamp_s[0] <= t0 + 1.0
                  and series.timestamp_s[-1] >= t1 - 1.0)
    g1 = covers and series.max_gap_s <= 1.0 and abs(series.measured_rate_hz - 100.0) <= 10.0
    g2 = (gt_heading_residual_deg is None
          or abs(gt_heading_residual_deg) < 5.0)
    roll_sd = float(np.std(series.roll_deg, ddof=1)) if len(series) > 1 else 0.0
    pitch_sd = float(np.std(series.pitch_deg, ddof=1)) if len(series) > 1 else 0.0
    g3 = (float(np.max(np.abs(series.tilt_deg))) < 30.0
          and 0.5 < roll_sd < 20.0 and 0.5 < pitch_sd < 20.0)
    return {
        "G1_coverage": {
            "pass": bool(g1), "covers_window": covers,
            "measured_rate_hz": series.measured_rate_hz, "max_gap_s": series.max_gap_s,
            "rule": "spans the window, no gap > 1 s, rate within 10% of 100 Hz",
        },
        "G2_yaw_self_consistency": {
            "pass": bool(g2), "residual_deg": gt_heading_residual_deg,
            "rule": "|mean residual vs RTK-derived heading| < 5 deg (Stage 2 measured -1.95 +/- 1.62)",
        },
        "G3_plausibility": {
            "pass": bool(g3), "roll_sd_deg": roll_sd, "pitch_sd_deg": pitch_sd,
            "max_tilt_deg": float(np.max(np.abs(series.tilt_deg))) if len(series) else float("nan"),
            "rule": "max|tilt| < 30 deg and roll/pitch sd in (0.5, 20) deg; LIT-006 saw sd ~4.9/~4.1",
        },
        "all_pass": bool(g1 and g2 and g3),
    }


def write_json(obj: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path
