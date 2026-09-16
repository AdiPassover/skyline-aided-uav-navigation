"""The harness-facing method interface, the registry, and the shared padding guard.

Contract: ``contracts/extraction-method.md``. Every extractor — prototype adapter, trivial
baselines, the POC — implements ``ExtractionMethod`` and is scored by the identical harness, so a
method comparison compares methods, never harness variants.

The padding guard exists because of a measured defect, not a hypothetical: ``EXP-006`` found 24/42
extractions pinned to letterbox bars that nothing in the pipeline had detected. Preprocessing is
part of the method (and of its config digest); an image the guard rejects is an explicit
``extraction_failure``, and a crop is recorded in the output metadata with the curve mapped back to
**original** image coordinates — predictions and GT always live in the same frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from hsreloc.extraction.curves import (
    CONVERSION_KIND_PRED,
    CanonicalCurve,
    REASON_METHOD_NO_OUTPUT,
    REASON_NONE,
    make_curve,
)

STATUS_OK = "ok"
STATUS_FAILURE = "extraction_failure"

REASON_PADDING_REJECTED = "padding_rejected"


class MethodError(Exception):
    """An unknown method id or a misconfigured method."""


@dataclass(frozen=True)
class ExtractionOutput:
    """What one extractor produced for one image — a curve or an explicit failure, never both."""

    status: str
    reason: str | None
    curve: CanonicalCurve | None
    runtime_s: float
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.status == STATUS_OK:
            if self.curve is None:
                raise MethodError("status 'ok' with no curve")
            self.curve.validate()
        elif self.status == STATUS_FAILURE:
            if not self.reason:
                raise MethodError("extraction_failure with no reason")
            if self.curve is not None:
                raise MethodError("extraction_failure must not carry a curve")
        else:
            raise MethodError(f"unknown status {self.status!r}")


@runtime_checkable
class ExtractionMethod(Protocol):
    """One image in, one curve or one explicit failure out. Pixels only — no GT, no retrieval."""

    method_id: str

    def configure(self, config: dict) -> None: ...

    def extract(self, image_bgr: np.ndarray) -> ExtractionOutput: ...


_REGISTRY: dict = {}


def register(cls):
    """Class decorator: register an ``ExtractionMethod`` implementation under its ``method_id``."""
    mid = getattr(cls, "method_id", None)
    if not mid:
        raise MethodError(f"{cls.__name__} has no method_id")
    if mid in _REGISTRY:
        raise MethodError(f"duplicate method_id {mid!r}")
    _REGISTRY[mid] = cls
    return cls


def get_method(method_id: str) -> "ExtractionMethod":
    try:
        cls = _REGISTRY[method_id]
    except KeyError:
        raise MethodError(
            f"unknown method {method_id!r}; registered: {registered_methods()}") from None
    return cls()


def registered_methods() -> tuple:
    return tuple(sorted(_REGISTRY))


# ---------------------------------------------------------------------------
# The shared padding guard (the EXP-006 defect class, promoted into the interface)
# ---------------------------------------------------------------------------

PADDING_INTENSITY_THRESHOLD = 8      # a pixel darker than this in every channel counts as "bar"
PADDING_MIN_FRACTION = 0.02          # bars thinner than 2 % of the dimension are ignored


def content_bbox(image_bgr: np.ndarray, threshold: int = PADDING_INTENSITY_THRESHOLD):
    """(top, bottom, left, right) of the non-bar content, half-open rows/cols.

    A row/column is a bar row/column when *every* pixel in it is darker than ``threshold`` in all
    channels — the same black-border detector EXP-006 introduced (its one parameter is a border
    detector, not a performance knob).
    """
    dark = (image_bgr < threshold).all(axis=2) if image_bgr.ndim == 3 else (image_bgr < threshold)
    row_is_bar = dark.all(axis=1)
    col_is_bar = dark.all(axis=0)
    h, w = dark.shape
    top = 0
    while top < h and row_is_bar[top]:
        top += 1
    bottom = h
    while bottom > top and row_is_bar[bottom - 1]:
        bottom -= 1
    left = 0
    while left < w and col_is_bar[left]:
        left += 1
    right = w
    while right > left and col_is_bar[right - 1]:
        right -= 1
    return top, bottom, left, right


def apply_padding_policy(image_bgr: np.ndarray, policy: str = "crop"):
    """Detect letterbox/pillarbox bars and apply the configured policy.

    Returns ``(content_image, crop_meta)`` where ``crop_meta`` records the bbox (empty dict when
    nothing was cropped). ``policy="reject"`` raises ``PaddingRejected`` instead of cropping.
    Bars thinner than ``PADDING_MIN_FRACTION`` of the dimension are treated as absent.
    """
    h, w = image_bgr.shape[:2]
    top, bottom, left, right = content_bbox(image_bgr)
    if bottom <= top or right <= left:
        raise PaddingRejected("the whole image is below the bar-intensity threshold")
    min_rows = max(1, int(round(h * PADDING_MIN_FRACTION)))
    min_cols = max(1, int(round(w * PADDING_MIN_FRACTION)))
    if top < min_rows and (h - bottom) < min_rows and left < min_cols and (w - right) < min_cols:
        return image_bgr, {}
    if policy == "reject":
        raise PaddingRejected(
            f"letterbox/pillarbox bars detected (content bbox rows {top}:{bottom}, "
            f"cols {left}:{right} of {h}x{w})")
    if policy != "crop":
        raise MethodError(f"unknown padding policy {policy!r} (expected 'crop' or 'reject')")
    meta = {"padding_crop": {"top": int(top), "bottom": int(bottom),
                             "left": int(left), "right": int(right),
                             "orig_height": int(h), "orig_width": int(w)}}
    return image_bgr[top:bottom, left:right], meta


class PaddingRejected(Exception):
    """The padding guard refused this image under the 'reject' policy (or found no content)."""


def curve_from_cropped(rows_cropped: np.ndarray, crop_meta: dict, orig_width: int,
                       orig_height: int, producer: str, meta: dict | None = None) -> CanonicalCurve:
    """Map a curve computed on the cropped content back into original image coordinates.

    Columns inside the bars carry no content and become invalid ``method_no_output`` — the method
    genuinely has no output there, and saying so beats extrapolating into a black bar.
    """
    if not crop_meta:
        reasons = tuple(REASON_NONE if np.isfinite(v) else REASON_METHOD_NO_OUTPUT
                        for v in rows_cropped)
        return make_curve(orig_width, orig_height, rows_cropped, reasons,
                          CONVERSION_KIND_PRED, producer, meta=meta)
    box = crop_meta["padding_crop"]
    rows = np.full(orig_width, np.nan, dtype=np.float64)
    rows[box["left"]:box["right"]] = np.asarray(rows_cropped, dtype=np.float64) + box["top"]
    reasons = tuple(REASON_NONE if np.isfinite(v) else REASON_METHOD_NO_OUTPUT for v in rows)
    merged = dict(meta or {})
    merged.update(crop_meta)
    return make_curve(orig_width, orig_height, rows, reasons, CONVERSION_KIND_PRED, producer,
                      meta=merged)


def timed_extract(fn):
    """Decorator for ``extract``: stamps wall-clock runtime (descriptive only, Principle IX)."""

    def wrapper(self, image_bgr: np.ndarray) -> ExtractionOutput:
        t0 = time.perf_counter()
        out = fn(self, image_bgr)
        elapsed = time.perf_counter() - t0
        out = ExtractionOutput(status=out.status, reason=out.reason, curve=out.curve,
                               runtime_s=elapsed, meta=out.meta)
        out.validate()
        return out

    return wrapper
