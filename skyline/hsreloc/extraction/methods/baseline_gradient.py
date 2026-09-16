"""Trivial baseline: per-column strongest downward intensity drop. No smoothing, no DP.

The "no method" bound (task T019): what a single hand-picked cue achieves with no spatial
coherence at all. At a sky/terrain boundary intensity typically falls moving down the image, so
each column independently picks the row below its largest negative vertical difference. Clouds,
glare and texture defeat it trivially — that is the point of a bound.
"""

from __future__ import annotations

import cv2
import numpy as np

from hsreloc.extraction.methods.base import (
    STATUS_OK,
    ExtractionOutput,
    PaddingRejected,
    STATUS_FAILURE,
    apply_padding_policy,
    curve_from_cropped,
    register,
    timed_extract,
)


@register
class GradientBaseline:
    method_id = "baseline_gradient"

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
        gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dy = np.diff(gray, axis=0)                      # dy[r] = I[r+1] - I[r]
        rows = (np.argmin(dy, axis=0) + 1).astype(np.float64)
        curve = curve_from_cropped(rows, crop_meta, w0, h0,
                                   producer=f"method:{self.method_id}")
        return ExtractionOutput(status=STATUS_OK, reason=None, curve=curve, runtime_s=0.0)
