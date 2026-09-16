"""Candidate profile matchers — an **additive** family beside the frozen chain, never in place of it.

``hsreloc/retrieval/`` is unchanged and remains the baseline; ``C0`` here calls its scorer directly.
The ladder, in the order ``LIT-SKY-005`` ranked it and the order ``PROT-SKY-001`` §5 requires runs to
follow:

===== ================================ ===================================== ====================
key   variant                          added freedom                          status
===== ================================ ===================================== ====================
C0    ``c0_frozen_ncc``                none                                   the frozen baseline
C1    ``c1_bounded_lag_ncc``           horizontal shift, bounded              implemented, untuned
C2    ``c2_angular_units``             x-axis in azimuth (a representation)   implemented, untuned
C3    ``c3_shift_scale_ncc``           shift x horizontal scale, bounded      implemented, untuned
C4    ``c4_banded_dtw_ncc``            banded monotone local warp             synthetic-test only
===== ================================ ===================================== ====================

"Untuned" is a load-bearing word: every bound is a declared configuration value with a stated
provenance (heading uncertainty, camera FOV, the far-field model), and **none of them may be chosen
from retrieval correctness**. The first simulator experiment runs C0 on all three curve sources before
any of C1–C4 touches real data (``PROT-SKY-001`` §5; the brief's Part H).

C2 is a *representation*, not a scorer: it changes what the x-axis of a profile means, and any of
C0/C1/C3/C4 then scores in it. That is why it is listed in the ladder but built through
``hsreloc.matchers.representation`` rather than ``build_matcher``.
"""

from hsreloc.matchers.base import (DEFAULT_MIN_OVERLAP_FRAC, MATCHERS_VERSION, BaseMatcher,
                                   MatcherError, MatchResult, ProfileMatcher)
from hsreloc.matchers.dtw import C4, ConstrainedDtwMatcher, align_by_path, banded_dtw
from hsreloc.matchers.representation import (C2, AngularProfileConfig, CameraModel,
                                             RepresentationError, angular_profile, azimuth_grid,
                                             curvature_profile, degrees_per_sample,
                                             derivative_profile, local_extrema, multiscale_profiles,
                                             roughness, smoothed_profile)
from hsreloc.matchers.variants import (C0, C1, C3, BoundedLagNccMatcher, FrozenNccMatcher,
                                       ShiftScaleNccMatcher, lag_samples_for_degrees)

#: The pre-registered experimental order. A run may not skip a rung without saying why.
LADDER = (C0, C1, C2, C3, C4)

_BUILDERS = {
    C0: lambda spec: FrozenNccMatcher(),
    C1: lambda spec: BoundedLagNccMatcher(
        max_lag_samples=spec.get("max_lag_samples", 0),
        min_overlap_frac=spec.get("min_overlap_frac", DEFAULT_MIN_OVERLAP_FRAC),
        subsample_step=spec.get("subsample_step", 1.0),
        max_lag_deg=spec.get("max_lag_deg"), fov_deg=spec.get("fov_deg"),
        n_samples=spec.get("n_samples")),
    C3: lambda spec: ShiftScaleNccMatcher(
        max_lag_samples=spec.get("max_lag_samples", 0),
        scales=spec.get("scales") or ShiftScaleNccMatcher.scale_grid(
            spec.get("max_scale_deviation", 0.0), spec.get("n_scale_steps", 1)),
        min_overlap_frac=spec.get("min_overlap_frac", DEFAULT_MIN_OVERLAP_FRAC),
        subsample_step=spec.get("subsample_step", 1.0)),
    C4: lambda spec: ConstrainedDtwMatcher(
        band=spec.get("band", 8), step_penalty=spec.get("step_penalty", 0.05),
        max_mean_warp=spec.get("max_mean_warp", 4.0)),
}


_KNOWN_KEYS = {
    C0: set(),
    C1: {"max_lag_samples", "min_overlap_frac", "subsample_step", "max_lag_deg", "fov_deg",
         "n_samples"},
    C3: {"max_lag_samples", "scales", "max_scale_deviation", "n_scale_steps", "min_overlap_frac",
         "subsample_step"},
    C4: {"band", "step_penalty", "max_mean_warp"},
}


def build_matcher(variant: str, spec: dict | None = None):
    """Construct one variant from its config block.

    Unknown keys are **refused**, not ignored: the 2026-09-02 pilot passed ``max_lag: 64`` where
    the key is ``max_lag_samples``, and the silent default of 0 turned its C1 diagnostic into a
    lag-0 NCC (recorded as an addendum to ``EXP-SKY-007``). A declared bound that never reaches
    the matcher is worse than an error, so now it is one.
    """
    try:
        builder = _BUILDERS[variant]
    except KeyError:
        raise MatcherError(
            f"unknown matcher variant {variant!r}; expected one of {sorted(_BUILDERS)} "
            f"({C2} is a representation, applied through hsreloc.matchers.representation)") from None
    spec = dict(spec or {})
    unknown = sorted(set(spec) - _KNOWN_KEYS[variant])
    if unknown:
        raise MatcherError(f"{variant}: unknown spec key(s) {unknown}; declared keys are "
                           f"{sorted(_KNOWN_KEYS[variant])} — a bound that never reaches the "
                           f"matcher is a silent lag-0/identity run (EXP-SKY-007 addendum)")
    return builder(spec)


__all__ = [
    "MATCHERS_VERSION", "LADDER", "MatchResult", "MatcherError", "ProfileMatcher", "BaseMatcher",
    "DEFAULT_MIN_OVERLAP_FRAC", "build_matcher",
    "C0", "C1", "C2", "C3", "C4",
    "FrozenNccMatcher", "BoundedLagNccMatcher", "ShiftScaleNccMatcher", "ConstrainedDtwMatcher",
    "lag_samples_for_degrees", "banded_dtw", "align_by_path",
    "CameraModel", "AngularProfileConfig", "angular_profile", "azimuth_grid", "degrees_per_sample",
    "RepresentationError", "derivative_profile", "smoothed_profile", "multiscale_profiles",
    "curvature_profile", "local_extrema", "roughness",
]
