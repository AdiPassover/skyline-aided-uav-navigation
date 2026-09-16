"""Trivial baseline: global Otsu intensity split → per-column topmost bright→dark transition.

The second "no method" bound (task T020): assumes sky is the globally brighter class. Per column,
the boundary is the first pixel (top→down) falling in the darker class; a column whose top pixel is
already dark has no decidable sky and is an invalid column — stated, not guessed (spec FR-013).
Night scenes, bright terrain and dark skies defeat it; again, that is the point of a bound.
"""

from __future__ import annotations

import cv2
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


@register
class ThresholdBaseline:
    method_id = "baseline_threshold"

    def __init__(self) -> None:
        self.padding_policy = "crop"

    def configure(self, config: dict) -> None:
        self.padding_policy = config.get("padding_policy", "crop")

    @timed_extract
    def extract(self, image_bgr: np.ndarray) -> ExtractionOutput:
        h0, w0 = image_bgr.shape[:2]
        try:
            content, crop_meta = apply_padding_policy(image_bgr, self.padding_policy)
        except PaddingRejected as e:
            return ExtractionOutput(status=STATUS_FAILURE, reason=f"padding_rejected: {e}",
                                    curve=None, runtime_s=0.0)
        gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sky = binary > 0                       # the brighter Otsu class
        h, w = sky.shape
        nonsky = ~sky
        has_nonsky = nonsky.any(axis=0)
        first_nonsky = nonsky.argmax(axis=0)
        rows = np.full(w, np.nan, dtype=np.float64)
        for c in range(w):
            if has_nonsky[c] and first_nonsky[c] > 0:
                rows[c] = float(first_nonsky[c])
        # NaN columns (no sky at top / boundary below frame) become method_no_output in the
        # prediction frame: the method has no decidable output there.
        curve = curve_from_cropped(rows, crop_meta, w0, h0,
                                   producer=f"method:{self.method_id}")
        return ExtractionOutput(status=STATUS_OK, reason=None, curve=curve, runtime_s=0.0)
