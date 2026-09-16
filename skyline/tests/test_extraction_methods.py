"""Method-contract tests (task T021): conformance, determinism, failure paths, padding rule."""

from __future__ import annotations

import numpy as np
import pytest

from hsreloc.extraction.methods import get_method, registered_methods
from hsreloc.extraction.methods.base import (
    STATUS_FAILURE,
    STATUS_OK,
    ExtractionMethod,
    MethodError,
    apply_padding_policy,
    content_bbox,
    curve_from_cropped,
)

EXPECTED = {"baseline_gradient", "baseline_threshold", "prototype_dp", "poc_robust_dp"}


def _scene(h=60, w=80, boundary=30, sky=200, ground=60, seed=0):
    """Sky above ``boundary`` (bright, bluish), textured ground below."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:boundary] = (sky + 30, sky, sky - 40)          # bluish bright sky (BGR)
    ground_tex = rng.integers(ground - 30, ground + 30, size=(h - boundary, w, 3))
    img[boundary:] = ground_tex.astype(np.uint8)
    return img


class TestRegistry:
    def test_expected_methods_registered(self):
        assert EXPECTED.issubset(set(registered_methods()))

    def test_unknown_method_is_error(self):
        with pytest.raises(MethodError, match="unknown method"):
            get_method("nope")


@pytest.mark.parametrize("method_id", sorted(EXPECTED))
class TestConformance:
    def test_protocol_and_determinism(self, method_id):
        m = get_method(method_id)
        assert isinstance(m, ExtractionMethod)
        m.configure({})
        img = _scene()
        out1 = m.extract(img)
        out2 = m.extract(img)
        out1.validate(), out2.validate()
        assert out1.status == STATUS_OK
        assert out1.curve.rows.tolist() == out2.curve.rows.tolist()

    def test_finds_the_easy_boundary(self, method_id):
        m = get_method(method_id)
        m.configure({})
        out = m.extract(_scene(boundary=30))
        assert out.status == STATUS_OK
        valid = out.curve.valid_mask
        assert valid.mean() > 0.9
        err = np.abs(out.curve.rows[valid] - 30.0)
        assert np.median(err) <= 2.0

    def test_padding_reject_policy_is_explicit_failure(self, method_id):
        m = get_method(method_id)
        m.configure({"padding_policy": "reject"})
        img = np.zeros((80, 60, 3), dtype=np.uint8)
        img[20:60] = _scene(h=40, w=60, boundary=20)     # letterboxed content
        out = m.extract(img)
        assert out.status == STATUS_FAILURE
        assert "padding_rejected" in out.reason

    def test_padding_crop_maps_back_to_original_frame(self, method_id):
        m = get_method(method_id)
        m.configure({"padding_policy": "crop"})
        img = np.zeros((80, 60, 3), dtype=np.uint8)
        img[20:60] = _scene(h=40, w=60, boundary=20)     # true boundary at original row 40
        out = m.extract(img)
        assert out.status == STATUS_OK
        assert out.curve.width_px == 60 and out.curve.height_px == 80
        valid = out.curve.valid_mask
        assert valid.any()
        err = np.abs(out.curve.rows[valid] - 40.0)
        assert np.median(err) <= 3.0
        assert "padding_crop" in out.curve.meta

    def test_runtime_is_stamped(self, method_id):
        m = get_method(method_id)
        m.configure({})
        out = m.extract(_scene())
        assert out.runtime_s >= 0.0


class TestPaddingGuard:
    def test_content_bbox_on_clean_image(self):
        img = _scene()
        assert content_bbox(img) == (0, 60, 0, 80)

    def test_all_dark_image_raises(self):
        from hsreloc.extraction.methods.base import PaddingRejected
        with pytest.raises(PaddingRejected):
            apply_padding_policy(np.zeros((20, 20, 3), dtype=np.uint8), "crop")

    def test_thin_bars_ignored(self):
        img = _scene(h=100, w=100)
        img[0] = 0                     # a single dark row — below the 2 % floor
        out, meta = apply_padding_policy(img, "crop")
        assert meta == {} and out.shape == img.shape

    def test_curve_from_cropped_offsets(self):
        rows = np.array([5.0, 6.0])
        meta = {"padding_crop": {"top": 10, "bottom": 50, "left": 1, "right": 3,
                                 "orig_height": 60, "orig_width": 5}}
        curve = curve_from_cropped(rows, meta, 5, 60, "t")
        assert np.isnan(curve.rows[0]) and np.isnan(curve.rows[3]) and np.isnan(curve.rows[4])
        assert curve.rows[1] == 15.0 and curve.rows[2] == 16.0


class TestPrototypeConfig:
    def test_unknown_parameter_rejected(self):
        m = get_method("prototype_dp")
        with pytest.raises(ValueError, match="unknown parameters"):
            m.configure({"learning_rate": 0.1})

    def test_shipped_defaults_are_the_default(self):
        from hsreloc.extraction.methods.prototype_dp import SHIPPED_DEFAULTS
        m = get_method("prototype_dp")
        m.configure({})
        assert m.params == SHIPPED_DEFAULTS
