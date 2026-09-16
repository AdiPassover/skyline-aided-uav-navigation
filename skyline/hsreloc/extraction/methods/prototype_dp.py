"""Adapter over the existing prototype DP extractor — the measured baseline-to-beat.

Wraps ``skyline.extraction.extract_skyline`` (Lie et al. 2005 lineage, maturity **prototype**,
``skyline/skyline/extraction.py``) exactly as shipped. The prototype file is
**not modified**: its ``EXP-006`` provenance (21/42 alpine failures from snow/haze cue inversion)
must stay attached to exactly the code that produced it (task T018).

The one thing the adapter adds is the shared padding guard, applied *outside* the prototype —
EXP-006 established that letterboxed input invalidates the prototype's global region cue, so
feeding it bars silently would measure the defect class, not the method. The guard is part of the
adapter's config and digest; the prototype's own parameters are its shipped defaults, recorded
verbatim below.
"""

from __future__ import annotations

import numpy as np

from hsreloc.extraction.methods.base import (
    STATUS_FAILURE,
    STATUS_OK,
    ExtractionOutput,
    PaddingRejected,
    apply_padding_policy,
    curve_from_cropped,
    register,
    timed_extract,
)

from skyline.extraction import extract_skyline

#: The prototype's shipped defaults (``extract_skyline`` signature), recorded verbatim — the
#: frozen "as-is" configuration of the falsification runs.
SHIPPED_DEFAULTS = {"work_width": 480, "max_jump": 10, "smooth": 0.12,
                    "w_edge": 0.45, "w_region": 0.55}


@register
class PrototypeDp:
    method_id = "prototype_dp"

    def __init__(self) -> None:
        self.params = dict(SHIPPED_DEFAULTS)
        self.padding_policy = "crop"

    def configure(self, config: dict) -> None:
        unknown = set(config) - set(SHIPPED_DEFAULTS) - {"padding_policy"}
        if unknown:
            raise ValueError(f"{self.method_id}: unknown parameters {sorted(unknown)}")
        self.params = {**SHIPPED_DEFAULTS, **{k: config[k] for k in config
                                              if k in SHIPPED_DEFAULTS}}
        self.padding_policy = config.get("padding_policy", "crop")

    @timed_extract
    def extract(self, image_bgr: np.ndarray) -> ExtractionOutput:
        h0, w0 = image_bgr.shape[:2]
        try:
            content, crop_meta = apply_padding_policy(image_bgr, self.padding_policy)
        except PaddingRejected as e:
            return ExtractionOutput(status=STATUS_FAILURE, reason=f"padding_rejected: {e}",
                                    curve=None, runtime_s=0.0)
        # Measured on SkyFinder EVAL (2026-08-24): near-dark night frames crop to a band whose DP
        # working height is at or below max_jump, and the prototype's shortest-path then indexes
        # past its own grid. The prototype is not edited (EXP-006 provenance); the adapter converts
        # the condition — and any other prototype exception — into an explicit per-image
        # extraction failure (Principle X), which the harness counts as catastrophic.
        ch, cw = content.shape[:2]
        work_height = max(2, int(round(ch * self.params["work_width"] / float(cw))))
        if work_height <= self.params["max_jump"]:
            return ExtractionOutput(
                status=STATUS_FAILURE,
                reason=f"content_too_small: working height {work_height} <= max_jump "
                       f"{self.params['max_jump']} (content {cw}x{ch} after padding policy)",
                curve=None, runtime_s=0.0)
        try:
            result = extract_skyline(content, **self.params)
        except Exception as e:  # the prototype offers no failure channel of its own
            return ExtractionOutput(status=STATUS_FAILURE,
                                    reason=f"prototype_error: {type(e).__name__}: {e}",
                                    curve=None, runtime_s=0.0)
        rows = np.asarray(result.y, dtype=np.float64)
        curve = curve_from_cropped(rows, crop_meta, w0, h0,
                                   producer=f"method:{self.method_id}",
                                   meta={"roll_deg": float(result.roll_deg)})
        return ExtractionOutput(status=STATUS_OK, reason=None, curve=curve, runtime_s=0.0)
