"""naveval.confidence: calibration loading, refusals, and scoring (schema 2.0.0).

Mirrors the Java loader's refusal set (CalibrationConfigTest) so a configuration both languages
must read identically is validated identically. Evidence tier: T1 (analytical) — validates the
instrument, never the VO.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from naveval.confidence import (
    CalibrationRefusalError,
    digest_of,
    load_calibration,
    parse_calibration,
)

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "evaluation" / "confidence_configs"

MINIMAL = {
    "schema_version": "2.0.0",
    "calibration_id": "test-unvalidated-v1",
    "validated": False,
    "validated_by": None,
    "probability_semantics": False,
    "required_signals": ["track_count", "inlier_ratio"],
    "configuration_binding": {"motion_model": "homography", "downsampleFactor": 2},
    "temporal": {"window_w": 10, "warmup_m": 5},
    "rejection_rules": [
        {"signal": "track_count", "op": "lt", "threshold": 30.0, "reason": "insufficient_features"}
    ],
    "score_model": {
        "type": "weighted_sum",
        "terms": [
            {"signal": "inlier_ratio", "normalise": "identity", "reference": None, "weight": 1.0}
        ],
    },
    "usable_score_bound": 0.6,
}


def _text(overrides=None, drop=None) -> str:
    d = json.loads(json.dumps(MINIMAL))
    for key in (drop or []):
        d.pop(key, None)
    for key, value in (overrides or {}).items():
        d[key] = value
    return json.dumps(d, indent=2)


class TestLoader:
    def test_parses_and_exposes_every_field(self):
        c = parse_calibration(_text())
        assert c.schema_version == "2.0.0"
        assert c.calibration_id == "test-unvalidated-v1"
        assert not c.validated and c.validated_by is None
        assert not c.probability_semantics
        assert c.window_w == 10 and c.warmup_m == 5
        assert c.score_model.type == "weighted_sum"
        assert c.usable_score_bound == 0.6
        assert len(c.configuration_binding) == 2

    def test_schema_1_is_refused(self):
        with pytest.raises(CalibrationRefusalError, match="2.0.0"):
            parse_calibration(_text({"schema_version": "1.0.0"}))

    def test_unknown_top_level_field_is_refused(self):
        with pytest.raises(CalibrationRefusalError, match="Unknown calibration fields"):
            parse_calibration(_text({"usable_score_bnd": 0.6}, drop=["usable_score_bound"]))

    def test_unknown_signal_name_is_refused(self):
        with pytest.raises(CalibrationRefusalError, match="Unknown signal name"):
            parse_calibration(_text({"required_signals": ["track_cout", "inlier_ratio"]}))

    def test_probability_semantics_requires_validation(self):
        with pytest.raises(CalibrationRefusalError, match="A1"):
            parse_calibration(_text({"probability_semantics": True}))

    def test_binding_must_be_non_empty(self):
        with pytest.raises(CalibrationRefusalError, match="A3"):
            parse_calibration(_text({"configuration_binding": {}}))
        with pytest.raises(CalibrationRefusalError, match="configuration_binding"):
            parse_calibration(_text(drop=["configuration_binding"]))

    def test_temporal_bounds_are_enforced(self):
        with pytest.raises(CalibrationRefusalError, match="temporal"):
            parse_calibration(_text(drop=["temporal"]))
        with pytest.raises(CalibrationRefusalError, match="warmup_m"):
            parse_calibration(_text({"temporal": {"window_w": 10, "warmup_m": 11}}))

    def test_weights_must_sum_to_one(self):
        bad = json.loads(json.dumps(MINIMAL))
        bad["score_model"]["terms"][0]["weight"] = 0.8
        with pytest.raises(CalibrationRefusalError, match="sum to 1.0"):
            parse_calibration(json.dumps(bad))

    def test_undeclared_signal_in_rule_or_model_is_refused(self):
        with pytest.raises(CalibrationRefusalError, match="required_signals"):
            parse_calibration(_text({"required_signals": ["inlier_ratio"]}))

    def test_rejection_rule_must_name_a_condition(self):
        bad = json.loads(json.dumps(MINIMAL))
        bad["rejection_rules"][0]["reason"] = "low_score"
        with pytest.raises(CalibrationRefusalError, match="reason"):
            parse_calibration(json.dumps(bad))

    def test_isotonic_shape_errors_are_refused(self):
        iso = json.loads(json.dumps(MINIMAL))
        iso["required_signals"] = ["relative_support"]
        iso["rejection_rules"] = []
        iso["score_model"] = {
            "type": "isotonic",
            "signal": "relative_support",
            "thresholds": [0.25, 0.5],
            "values": [0.1, 0.5],   # wrong length
        }
        with pytest.raises(CalibrationRefusalError, match="thresholds\\+1"):
            parse_calibration(json.dumps(iso))
        iso["score_model"]["values"] = [0.1, 0.5, 0.9]
        parse_calibration(json.dumps(iso))  # correct shape parses
        iso["score_model"]["thresholds"] = [0.5, 0.25]
        with pytest.raises(CalibrationRefusalError, match="ascending"):
            parse_calibration(json.dumps(iso))

    def test_logistic_requires_intercept_and_coefficients(self):
        lo = json.loads(json.dumps(MINIMAL))
        lo["required_signals"] = ["relative_support"]
        lo["rejection_rules"] = []
        lo["score_model"] = {"type": "logistic", "coefficients": [
            {"signal": "relative_support", "coefficient": 2.0}]}
        with pytest.raises(CalibrationRefusalError, match="intercept"):
            parse_calibration(json.dumps(lo))
        lo["score_model"]["intercept"] = -1.0
        parse_calibration(json.dumps(lo))

    def test_unknown_model_type_is_refused(self):
        bad = json.loads(json.dumps(MINIMAL))
        bad["score_model"] = {"type": "max"}
        with pytest.raises(CalibrationRefusalError, match="score_model.type"):
            parse_calibration(json.dumps(bad))


class TestDigest:
    def test_digest_is_byte_based_and_line_ending_insensitive(self):
        text = _text()
        assert digest_of(text) == digest_of(text.replace("\n", "\r\n"))
        assert digest_of(text).startswith("sha256:")
        assert digest_of(text) != digest_of(text.replace("30.0", "31.0"))

    def test_shipped_calibrations_load_and_declare_themselves_unvalidated(self):
        for name in ("bootstrap-unvalidated-v2.json", "contrast-unvalidated-v2.json"):
            c = load_calibration(CONFIGS / name)
            assert not c.validated
            assert not c.probability_semantics
            assert "unvalidated" in c.calibration_id
            assert c.window_w == 10 and c.warmup_m == 5

    def test_digest_matches_the_java_written_manifest_value(self):
        """Cross-language digest agreement, against the Java-produced fixture manifest."""
        fixture = REPO / "evaluation" / "tests" / "fixtures" / "confidence_agreement"
        manifest = json.loads((fixture / "run_bootstrap" / "manifest.json").read_text("utf-8"))
        java_digest = manifest["confidence"]["calibration_digest"]
        python_digest = load_calibration(CONFIGS / "bootstrap-unvalidated-v2.json").digest
        assert python_digest == java_digest, (
            "the two loaders hashed different bytes for the same file — the digest is the identity "
            "that proves both scored under identical configuration"
        )
