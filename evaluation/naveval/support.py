"""Support-level classification: (metric, ground-truth class) -> support level (FR-061).

The classification is deliberately a property of the pair alone, evaluated from the
*declared* ground-truth quality class -- never relaxed to make a given source look
adequate (FR-062). `unknown` fails closed: every metric keyed by an `unknown` class is
`unsupported`, never a default of "probably fine".

Position-scoped metrics are classified from `position_quality.quality_class`;
heading-scoped metrics from `heading_quality.quality_class` independently (FR-068) --
position and heading typically come from different sensors with unrelated error
characteristics, so one must never contaminate the other's classification.

Source: spec.md "Ground-truth quality determines which claims are possible" support
matrix, itself citing KITTI's documented precedent ([[LIT-004-geiger-2012-kitti]]) for
why short-baseline relative accuracy is unsupported at metre-level ground truth.
"""

from __future__ import annotations

from typing import Optional

SUPPORTED = "supported"
WEAKLY_SUPPORTED = "weakly_supported"
UNSUPPORTED = "unsupported"

# spec.md support matrix, columns "Onboard GNSS + FC telemetry" -> consumer_gnss,
# "RTK GNSS" -> rtk_gnss, "Java/UE5 simulator (exact)" -> simulator_exact. `unknown` is
# handled separately (always unsupported) rather than listed per metric below.
_POSITION_METRIC_MATRIX: dict[str, dict[str, str]] = {
    "ate_rmse": {"consumer_gnss": WEAKLY_SUPPORTED, "rtk_gnss": SUPPORTED, "simulator_exact": SUPPORTED},
    "ate_rmse_normalised": {"consumer_gnss": WEAKLY_SUPPORTED, "rtk_gnss": SUPPORTED, "simulator_exact": SUPPORTED},
    "drift_per_distance": {"consumer_gnss": SUPPORTED, "rtk_gnss": SUPPORTED, "simulator_exact": SUPPORTED},
    "rpe_short": {"consumer_gnss": UNSUPPORTED, "rtk_gnss": SUPPORTED, "simulator_exact": SUPPORTED},
    "rpe_long": {"consumer_gnss": WEAKLY_SUPPORTED, "rtk_gnss": SUPPORTED, "simulator_exact": SUPPORTED},
    "endpoint_error": {"consumer_gnss": WEAKLY_SUPPORTED, "rtk_gnss": SUPPORTED, "simulator_exact": SUPPORTED},
    "endpoint_error_closed_loop": {
        "consumer_gnss": SUPPORTED,
        "rtk_gnss": SUPPORTED,
        "simulator_exact": SUPPORTED,
    },
    "scale_drift_diagnostic": {
        "consumer_gnss": WEAKLY_SUPPORTED,
        "rtk_gnss": SUPPORTED,
        "simulator_exact": SUPPORTED,
    },
}

# "Weakly supported unless dual-antenna" (spec.md matrix): this schema has no
# dual-antenna flag yet, so rtk_gnss heading defaults to the conservative case rather
# than assuming the better one (FR-062: never relax a classification to look adequate).
_HEADING_METRIC_MATRIX: dict[str, dict[str, str]] = {
    "yaw_error": {"fc_heading": WEAKLY_SUPPORTED, "rtk_gnss": WEAKLY_SUPPORTED, "simulator_exact": SUPPORTED},
    "yaw_drift": {"fc_heading": WEAKLY_SUPPORTED, "rtk_gnss": WEAKLY_SUPPORTED, "simulator_exact": SUPPORTED},
    # Skyline relocalization evaluation (spec 004). Heading error is a per-query circular
    # error against the query set's heading ground truth; it is only as strong as that
    # ground truth's declared quality class, exactly like yaw_error above.
    "skyline_heading_error": {
        "fc_heading": WEAKLY_SUPPORTED, "rtk_gnss": WEAKLY_SUPPORTED, "simulator_exact": SUPPORTED
    },
}

# Skyline relocalization evaluation (spec 004): per-query absolute geographic position
# error. Same position-quality dependence as absolute trajectory error -- a metre-level
# GNSS query set only weakly supports a metric-accuracy claim; RTK or exact simulator
# ground truth supports it. Added additively so classify_position serves both features.
_POSITION_METRIC_MATRIX["skyline_position_error"] = {
    "consumer_gnss": WEAKLY_SUPPORTED, "rtk_gnss": SUPPORTED, "simulator_exact": SUPPORTED
}

# KITTI lengthened its evaluated sub-sequences from 5-400m to 100-800m specifically
# because short-baseline GNSS/INS ground-truth error biased results (LIT-004). This
# threshold is a documented placeholder in that same spirit, not a measurement --
# spec.md's Deferred Decisions table schedules the real value from the first EXP record.
RPE_SHORT_LONG_THRESHOLD_M = 100.0


def rpe_metric_name_for_length(length_m: float) -> str:
    """Which support-classification bucket a configured RPE sub-trajectory length falls
    into. A separate, tiny function rather than inlined logic, so the threshold has
    exactly one place to change when the first EXP record supplies a measured value."""
    return "rpe_long" if length_m >= RPE_SHORT_LONG_THRESHOLD_M else "rpe_short"


def classify_position(metric: str, quality_class: Optional[str]) -> str:
    """Support level for a position-scoped metric, given the dataset's declared
    `position_quality.quality_class` (or None if the dataset carries no position
    quality declaration at all, which fails closed identically to `unknown`)."""
    if quality_class is None or quality_class == "unknown":
        return UNSUPPORTED
    table = _POSITION_METRIC_MATRIX.get(metric)
    if table is None:
        raise KeyError(f"No support classification defined for position metric {metric!r}")
    return table.get(quality_class, UNSUPPORTED)


def classify_heading(metric: str, quality_class: Optional[str]) -> str:
    """Support level for a heading-scoped metric, given the dataset's declared
    `heading_quality.quality_class` -- independent of position quality (FR-068)."""
    if quality_class is None or quality_class == "unknown":
        return UNSUPPORTED
    table = _HEADING_METRIC_MATRIX.get(metric)
    if table is None:
        raise KeyError(f"No support classification defined for heading metric {metric!r}")
    return table.get(quality_class, UNSUPPORTED)
