"""The online/offline agreement invariant (contracts/agreement.md).

Runs against the **Java-produced** fixture (`ConfidenceAgreementFixtureTest` generated it through
the real SignalExtractor → ConfidenceScorer → RunRecordWriter pipeline; a Python-authored fixture
would test Python against itself and pass while the invariant was broken). Exact means exact:
string equality on outcome/reason, **bitwise** equality on scores — a tolerance would silently
absorb precisely the divergence this test exists to catch.

Also verifies the amendment-A4 temporal replay: the persisted windowed signals are re-derivable
bit-for-bit from the raw per-frame columns.

Evidence tier: T1 (analytical).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from naveval.confidence import (
    CalibrationRefusalError,
    load_calibration,
    replay_windowed_signals,
    rescore_run,
    score_frame,
    verify_agreement,
)
from naveval.runrecord import load_run_record

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "evaluation" / "tests" / "fixtures" / "confidence_agreement"
CONFIGS = REPO / "evaluation" / "confidence_configs"


@pytest.fixture(scope="module")
def bootstrap():
    return load_calibration(CONFIGS / "bootstrap-unvalidated-v2.json")


@pytest.fixture(scope="module")
def contrast():
    return load_calibration(CONFIGS / "contrast-unvalidated-v2.json")


@pytest.fixture(scope="module")
def run_bootstrap():
    return load_run_record(FIXTURE / "run_bootstrap")


@pytest.fixture(scope="module")
def run_contrast():
    return load_run_record(FIXTURE / "run_contrast")


class TestAgreement:
    def test_every_frame_agrees_bitwise_under_bootstrap(self, run_bootstrap, bootstrap):
        mismatches = verify_agreement(run_bootstrap, bootstrap)
        assert mismatches == [], "\n".join(mismatches)

    def test_every_frame_agrees_bitwise_under_contrast(self, run_contrast, contrast):
        mismatches = verify_agreement(run_contrast, contrast)
        assert mismatches == [], "\n".join(mismatches)

    def test_first_frame_is_not_established_with_null_score(self, run_bootstrap, bootstrap):
        s = score_frame(bootstrap, run_bootstrap.frame_estimates[0], first_frame=True)
        assert (s.outcome, s.reason, s.score) == ("rejected", "not_established", None)
        c = run_bootstrap.frame_estimates[0].confidence
        assert (c.outcome, c.reason, c.score) == ("rejected", "not_established", None)

    def test_absent_residuals_are_unavailable_not_zero(self, run_bootstrap, bootstrap):
        # f7 carries no residual summary; the bootstrap calibration requires residual_mean_sq_px.
        fe = run_bootstrap.frame_estimates[7]
        assert fe.confidence.residual_mean_sq_px is None
        s = score_frame(bootstrap, fe, first_frame=False)
        assert (s.outcome, s.reason, s.score) == ("rejected", "signals_unavailable", None)

    def test_empty_match_set_is_a_real_zero_beside_absent_statistics(self, run_bootstrap, bootstrap):
        fe = run_bootstrap.frame_estimates[8]
        assert fe.confidence.residual_inlier_count == 0
        assert fe.confidence.residual_mean_sq_px is None
        s = score_frame(bootstrap, fe, first_frame=False)
        # The count-of-zero is not read as a statistic: the required mean is absent.
        assert (s.outcome, s.reason) == ("rejected", "signals_unavailable")

    def test_estimator_failed_frame_is_not_produced(self, run_bootstrap, bootstrap):
        fe = run_bootstrap.frame_estimates[9]
        assert not fe.success
        s = score_frame(bootstrap, fe, first_frame=False)
        assert (s.outcome, s.reason, s.score) == ("not_produced", "estimator_failed", None)

    def test_exactly_on_threshold_values_stay_unrejected_in_both_languages(
            self, run_bootstrap, bootstrap):
        # f15: track_count == 30 (lt strict), inlier_ratio == 0.3 exactly, residual mean == 3.0
        # (gt strict). No rule fires; the low score lands the frame in DEGRADED.
        fe = run_bootstrap.frame_estimates[15]
        assert fe.track_count == 30 and fe.inlier_count == 9
        s = score_frame(bootstrap, fe, first_frame=False)
        assert (s.outcome, s.reason) == ("degraded", "low_score")
        assert s.score == fe.confidence.score  # bitwise

    def test_the_two_calibrations_visibly_disagree(self, run_bootstrap, run_contrast,
                                                   bootstrap, contrast):
        # FR-019: without this case, two implementations that both ignored the configuration file
        # would pass every other case. Frame 2 is scored usable by bootstrap but is still in the
        # contrast calibration's warm-up (relative_support absent) — different outcome families.
        b = run_bootstrap.frame_estimates[2].confidence
        c = run_contrast.frame_estimates[2].confidence
        assert b.outcome == "usable" and c.outcome == "rejected"
        assert c.reason == "signals_unavailable"
        # And the re-scorers agree with each side under its own calibration.
        assert score_frame(bootstrap, run_bootstrap.frame_estimates[2], False).outcome == "usable"
        assert score_frame(contrast, run_contrast.frame_estimates[2], False).outcome == "rejected"

    def test_rescore_run_reproduces_the_whole_stream(self, run_bootstrap, bootstrap):
        rescored = rescore_run(run_bootstrap, bootstrap)
        for fe, s in zip(run_bootstrap.frame_estimates, rescored):
            c = fe.confidence
            assert (s.outcome, s.reason) == (c.outcome, c.reason)
            assert s.score == c.score  # None == None, or bitwise-equal floats

    def test_scoring_under_the_wrong_calibration_digest_is_flagged(self, run_bootstrap, contrast):
        # run_bootstrap names the bootstrap digest; verify_agreement under contrast must flag the
        # identity mismatch rather than comparing apples to oranges.
        mismatches = verify_agreement(run_bootstrap, contrast)
        assert mismatches and "digest mismatch" in mismatches[0]


class TestTemporalReplay:
    def test_windowed_signals_replay_bitwise_from_raw_columns(self, run_bootstrap, bootstrap):
        replayed = replay_windowed_signals(run_bootstrap, bootstrap.window_w, bootstrap.warmup_m)
        assert len(replayed) == len(run_bootstrap.frame_estimates)
        for fe, (rs, disp) in zip(run_bootstrap.frame_estimates, replayed):
            c = fe.confidence
            assert rs == c.relative_support, f"relative_support at frame {fe.frame_index}"
            assert disp == c.inc_log_scale_dispersion, f"dispersion at frame {fe.frame_index}"

    def test_the_restart_resets_the_windows(self, run_bootstrap):
        # f9 is the restart; the five frames after it are the rebuilt warm-up, so the windowed
        # signals must be absent on f10..f14 and present again on f15.
        for i in range(10, 15):
            assert run_bootstrap.frame_estimates[i].confidence.relative_support is None
        assert run_bootstrap.frame_estimates[15].confidence.relative_support is not None


class TestRefusals:
    def test_a_v1_0_record_is_refused_for_confidence_signals(self, bootstrap):
        # The golden fixture is a v1.0.0 record: it does not carry residual_mean_sq_px, so a
        # calibration requiring it is refused with the signal and run named — never defaulted.
        run = load_run_record(REPO / "evaluation" / "tests" / "fixtures" / "golden_run")
        with pytest.raises(CalibrationRefusalError) as exc:
            rescore_run(run, bootstrap)
        assert "residual_mean_sq_px" in str(exc.value)

    def test_a_binding_mismatch_is_refused(self, run_bootstrap, tmp_path, bootstrap):
        # Rewrite the fixture manifest with a different downsample factor: A3 must refuse.
        import shutil
        run_dir = tmp_path / "run"
        shutil.copytree(FIXTURE / "run_bootstrap", run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
        manifest["estimator_config"]["downsampleFactor"] = 1
        (run_dir / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        run = load_run_record(run_dir)
        with pytest.raises(CalibrationRefusalError, match="A3"):
            rescore_run(run, bootstrap)

    def test_a_window_identity_mismatch_is_refused(self, tmp_path, run_contrast):
        # The contrast calibration reads windowed signals; a run captured under W=20 must refuse
        # under a W=10 calibration (amendment A4: a different W is a different calibration).
        import shutil
        run_dir = tmp_path / "run"
        shutil.copytree(FIXTURE / "run_contrast", run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
        manifest["confidence"]["signal_window_w"] = 20
        (run_dir / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        run = load_run_record(run_dir)
        contrast = load_calibration(CONFIGS / "contrast-unvalidated-v2.json")
        with pytest.raises(CalibrationRefusalError, match="A4"):
            rescore_run(run, contrast)
