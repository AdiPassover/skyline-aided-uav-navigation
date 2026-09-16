"""The POC extractor: interpretable improved-cost DP with indeterminate-frame refusal.

Decision: ``DEC-SKY-003`` (Family 5 of ``LIT-SKY-002``). No training, no new dependency; every cue
below states what it measures and which *measured* failure it targets (research log 2026-08-24;
``EXP-006``). The DP recovery is the literature-persistent back-end (Lie et al. 2005 — the same
formulation the prototype uses); reimplemented here rather than imported so the collapsed-content
condition is guarded and the working resolution is configurable, and so the prototype file keeps
its EXP-006 provenance untouched.

Cues (all cheap, all inspectable):

1. **Oriented boundary edge** — the signed vertical derivative of the sky-likelihood map, gated by
   the *verticality* of the local gray-level gradient (|gy| / (|gx|+|gy|)). A sky/terrain boundary
   is a near-horizontal structure (vertical gradient); cloud texture and glare produce edges at all
   orientations. Targets: cloud-edge captures, glare (prototype failure class).
2. **Region split** — mean sky-likelihood above minus below the candidate row (the prototype's
   global term, kept: it is what bridges haze).
3. **Structure-above penalty** — the mean gray-gradient energy above the candidate row. The
   EXP-006 failure is one-sided (boundary placed *too low*, under sunlit snow/bright terrain);
   a path below the true ridge must cross terrain structure, which this term prices. Targets the
   too-low class directly.
4. **Refusal, before and after** (Principle X; the frozen definition's own reject rule):
   a near-black frame (night) or a near-zero-dynamic-range frame (fog/washout) has no visible
   boundary — the POC refuses *before* extracting; after extraction, a path whose across-boundary
   luminance contrast is negligible is refused rather than reported. Explicit failure beats a
   fabricated curve (spec FR-013), even though the frozen catastrophe rule counts both the same.

Maturity: **prototype** (component doc: COMP-SKY-002). Tuned on SkyFinder DEV cameras + synthetic
fixtures only; the frozen final-split config lives in ``configs/`` with its digest.
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

DEFAULTS = {
    "work_width": 640,
    "max_jump": 16,
    "smooth": 0.10,
    "w_edge": 0.40,
    "w_region": 0.35,
    "w_above": 0.25,
    "dark_p95_floor": 30.0,        # gray levels; p95 below this = night, refuse
    "flat_range_floor": 25.0,      # p95 - p5 below this = washout/fog, refuse
    "polarity_refusal_floor": None,  # enables when a number: refuse if mean(top 20 % rows) -
                                     # mean(bottom 20 % rows) falls below it. A negative polarity
                                     # is a night-lit / inverted scene (dark sky over lit
                                     # structures) that the bright-sky cue family cannot
                                     # represent; DEV measurement 2026-08-24: 70 % of
                                     # wrong-output frames sit below -10 vs 5 % of good frames.
                                     # Refusing fabricates nothing (spec FR-013) at a small
                                     # over-refusal cost, accepted deliberately.
    "min_boundary_contrast": 10.0, # mean |above-below| luminance across the path, gray levels
    "contrast_band_px": 5,
}


def _norm01(a, lo_pct=1.0, hi_pct=99.0):
    lo = np.percentile(a, lo_pct)
    hi = np.percentile(a, hi_pct)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def _sky_likelihood(image_bgr):
    """Per-pixel sky evidence: blue-red opponency, brightness, smoothness (per-image normalized)."""
    img = image_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(img)
    colour = _norm01(b - r)
    bright = _norm01((b + g + r) / 3.0)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mu = cv2.boxFilter(gray, -1, (9, 9))
    mu2 = cv2.boxFilter(gray * gray, -1, (9, 9))
    texture = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    smooth_cue = 1.0 - _norm01(texture)
    sky = 0.45 * colour + 0.25 * bright + 0.30 * smooth_cue
    return cv2.GaussianBlur(sky, (5, 5), 0), gray


def _node_cost(image_bgr, w_edge, w_region, w_above):
    sky, gray = _sky_likelihood(image_bgr)
    h, w = sky.shape

    # Cue 1: oriented boundary edge.
    gy_s = cv2.Sobel(sky, cv2.CV_32F, 0, 1, ksize=3)
    gx_g = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_g = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    verticality = np.abs(gy_g) / (np.abs(gx_g) + np.abs(gy_g) + 1e-6)
    edge = _norm01(np.maximum(-gy_s, 0.0) * verticality, hi_pct=99.5)

    # Cue 2: region split (identical formulation to the prototype's global term).
    cum = np.cumsum(sky, axis=0)
    total = cum[-1:, :]
    n_above = np.arange(1, h + 1, dtype=np.float32)[:, None]
    n_below = np.maximum(h - n_above, 1.0)
    region = _norm01(cum / n_above - (total - cum) / n_below)

    # Cue 3: structure-above penalty — gradient energy accumulated above the candidate row.
    gmag = np.abs(gx_g) + np.abs(gy_g)
    above = _norm01(np.cumsum(gmag, axis=0) / n_above)

    return 1.0 - (w_edge * edge + w_region * region) + w_above * above, gray


def _dp_shortest_path(cost, max_jump, smooth):
    """Minimum-cost left-to-right path (Lie et al. 2005 formulation), guarded for short grids."""
    h, w = cost.shape
    if h <= max_jump + 1:
        return None  # collapsed content; caller refuses explicitly
    acc = np.empty((h, w), dtype=np.float32)
    back = np.zeros((h, w), dtype=np.int32)
    acc[:, 0] = cost[:, 0]
    shifts = np.arange(-max_jump, max_jump + 1)
    penalties = smooth * np.abs(shifts).astype(np.float32)
    idx = np.arange(h)
    for x in range(1, w):
        prev = acc[:, x - 1]
        candidates = np.full((len(shifts), h), np.inf, dtype=np.float32)
        for k, (s, pen) in enumerate(zip(shifts, penalties)):
            if s == 0:
                candidates[k] = prev + pen
            elif s > 0:
                candidates[k, s:] = prev[:h - s] + pen
            else:
                candidates[k, :s] = prev[-s:] + pen
        best_k = np.argmin(candidates, axis=0)
        acc[:, x] = cost[:, x] + candidates[best_k, idx]
        back[:, x] = idx - shifts[best_k]
    path = np.empty(w, dtype=np.int32)
    path[-1] = int(np.argmin(acc[:, -1]))
    for x in range(w - 1, 0, -1):
        path[x - 1] = back[path[x], x]
    return path


@register
class PocRobustDp:
    method_id = "poc_robust_dp"

    def __init__(self) -> None:
        self.params = dict(DEFAULTS)
        self.padding_policy = "crop"

    def configure(self, config: dict) -> None:
        unknown = set(config) - set(DEFAULTS) - {"padding_policy"}
        if unknown:
            raise ValueError(f"{self.method_id}: unknown parameters {sorted(unknown)}")
        self.params = {**DEFAULTS, **{k: config[k] for k in config if k in DEFAULTS}}
        self.padding_policy = config.get("padding_policy", "crop")

    @timed_extract
    def extract(self, image_bgr: np.ndarray) -> ExtractionOutput:
        p = self.params
        h0, w0 = image_bgr.shape[:2]
        try:
            content, crop_meta = apply_padding_policy(image_bgr, self.padding_policy)
        except PaddingRejected as e:
            return ExtractionOutput(status=STATUS_FAILURE, reason=f"padding_rejected: {e}",
                                    curve=None, runtime_s=0.0)

        gray_full = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
        p95 = float(np.percentile(gray_full, 95))
        p5 = float(np.percentile(gray_full, 5))
        if p95 < p["dark_p95_floor"]:
            return ExtractionOutput(status=STATUS_FAILURE,
                                    reason=f"indeterminate_scene:dark (p95 luminance {p95:.1f})",
                                    curve=None, runtime_s=0.0)
        if (p95 - p5) < p["flat_range_floor"]:
            return ExtractionOutput(status=STATUS_FAILURE,
                                    reason=f"indeterminate_scene:flat (range {p95 - p5:.1f})",
                                    curve=None, runtime_s=0.0)
        if p["polarity_refusal_floor"] is not None:
            band_h = max(1, gray_full.shape[0] // 5)
            polarity = float(gray_full[:band_h].mean()) - float(gray_full[-band_h:].mean())
            if polarity < p["polarity_refusal_floor"]:
                return ExtractionOutput(
                    status=STATUS_FAILURE,
                    reason=f"indeterminate_scene:inverted_polarity (top-bottom luminance "
                           f"{polarity:.1f})",
                    curve=None, runtime_s=0.0)

        ch, cw = content.shape[:2]
        scale = p["work_width"] / float(cw)
        work_h = max(2, int(round(ch * scale)))
        small = cv2.resize(content, (p["work_width"], work_h), interpolation=cv2.INTER_AREA)
        cost, gray_small = _node_cost(small, p["w_edge"], p["w_region"], p["w_above"])
        path_small = _dp_shortest_path(cost, int(p["max_jump"]), float(p["smooth"]))
        if path_small is None:
            return ExtractionOutput(
                status=STATUS_FAILURE,
                reason=f"content_too_small: working height {work_h} <= max_jump+1",
                curve=None, runtime_s=0.0)

        # Post-refusal: measure the across-boundary luminance contrast on the working image.
        band = int(p["contrast_band_px"])
        contrasts = []
        for c in range(p["work_width"]):
            r = int(path_small[c])
            above = gray_small[max(0, r - band):r, c]
            below = gray_small[r:min(work_h, r + band), c]
            if above.size and below.size:
                contrasts.append(abs(float(above.mean()) - float(below.mean())) * 255.0)
        mean_contrast = float(np.mean(contrasts)) if contrasts else 0.0
        if mean_contrast < p["min_boundary_contrast"]:
            return ExtractionOutput(
                status=STATUS_FAILURE,
                reason=f"low_confidence_boundary (mean across-boundary contrast "
                       f"{mean_contrast:.1f} gray levels)",
                curve=None, runtime_s=0.0)

        xs_small = np.arange(p["work_width"])
        xs_full = np.linspace(0, p["work_width"] - 1, cw)
        rows = np.interp(xs_full, xs_small, path_small.astype(np.float64)) / scale
        rows = np.clip(rows, 0, ch - 1)
        curve = curve_from_cropped(rows, crop_meta, w0, h0,
                                   producer=f"method:{self.method_id}",
                                   meta={"mean_boundary_contrast": mean_contrast})
        return ExtractionOutput(status=STATUS_OK, reason=None, curve=curve, runtime_s=0.0)
