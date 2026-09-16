package org.boofcv.confidence;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The scorer is a pure, deterministic function that never invents a value (feature spec FR-006,
 * FR-016, and the invariants in {@code contracts/confidence-result.md}).
 *
 * <p>Constitution Principle XIII names "confidence calculations" as a required test target.
 *
 * <p>Evidence tier: <b>T1 (analytical)</b>. These validate the instrument, never the VO.
 */
public class ConfidenceScorerTest {

    /** Rejection rules in the default order: track count, then inlier ratio, then residual. */
    static final String RULES_DEFAULT_ORDER =
            "{\"signal\": \"track_count\", \"op\": \"lt\", \"threshold\": 30.0, \"reason\": \"insufficient_features\"},"
            + "{\"signal\": \"inlier_ratio\", \"op\": \"lt\", \"threshold\": 0.3, \"reason\": \"poor_inlier_support\"},"
            + "{\"signal\": \"residual_mean_sq_px\", \"op\": \"gt\", \"threshold\": 3.0, \"reason\": \"high_reprojection_error\"}";

    /** The same three rules with the first two swapped. */
    static final String RULES_SWAPPED_ORDER =
            "{\"signal\": \"inlier_ratio\", \"op\": \"lt\", \"threshold\": 0.3, \"reason\": \"poor_inlier_support\"},"
            + "{\"signal\": \"track_count\", \"op\": \"lt\", \"threshold\": 30.0, \"reason\": \"insufficient_features\"},"
            + "{\"signal\": \"residual_mean_sq_px\", \"op\": \"gt\", \"threshold\": 3.0, \"reason\": \"high_reprojection_error\"}";

    /**
     * Builds a calibration with the given rejection rules.
     *
     * <p>Assembled from a parameter rather than by substring-editing a text block, so that a test
     * about rule ORDER cannot silently pass because its string surgery failed to match.
     */
    static String configWithRules(String rulesJson) {
        return """
                {
                  "schema_version": "2.0.0",
                  "calibration_id": "test-unvalidated-v1",
                  "validated": false,
                  "validated_by": null,
                  "probability_semantics": false,
                  "required_signals": ["track_count", "inlier_ratio", "residual_mean_sq_px"],
                  "configuration_binding": {"motion_model": "homography"},
                  "temporal": {"window_w": 10, "warmup_m": 5},
                  "rejection_rules": [%s],
                  "score_model": {
                    "type": "weighted_sum",
                    "terms": [
                      {"signal": "inlier_ratio", "normalise": "identity", "reference": null, "weight": 0.5},
                      {"signal": "track_count", "normalise": "saturating", "reference": 200.0, "weight": 0.25},
                      {"signal": "residual_mean_sq_px", "normalise": "inverse_saturating", "reference": 3.0, "weight": 0.25}
                    ]
                  },
                  "usable_score_bound": 0.6
                }
                """.formatted(rulesJson);
    }

    static final String CONFIG = configWithRules(RULES_DEFAULT_ORDER);

    static ConfidenceScorer scorer() {
        return new ConfidenceScorer(CalibrationConfig.parse(CONFIG));
    }

    /** A healthy frame: 200 tracks, 90% inliers, sub-threshold residual. */
    static SignalBlock healthy() {
        return SignalBlock.builder()
                .trackCount(200)
                .inlierCount(180)
                .residualInlierCount(180)
                .residualMeanSqPx(1.0)
                .residualRmsPx(1.0)
                .inlierThresholdSqPx(3.0)
                .build();
    }

    @Test
    public void aHealthyFrameIsUsableWithReasonOk() {
        ConfidenceResult r = scorer().score(healthy(), true, false);

        assertEquals(ConfidenceOutcome.USABLE, r.outcome());
        assertEquals(ConfidenceReason.OK, r.reason());
        assertNotNull(r.score());
        assertTrue(r.poseConsumable());
        assertEquals("test-unvalidated-v1", r.calibrationId());
        assertTrue(r.calibrationDigest().startsWith("sha256:"));
    }

    @Test
    public void theScoreMatchesTheDocumentedWeightedSumExactly() {
        // 0.5*0.9 + 0.25*min(200/200,1) + 0.25*(1/(1+1/3))
        //   = 0.45 + 0.25 + 0.25*0.75 = 0.8875
        // Computed here in the same left-to-right order the scorer uses, because
        // contracts/agreement.md makes the order part of the result.
        double expected = 0.0;
        expected += 0.5 * (180.0 / 200.0);
        expected += 0.25 * Math.min(200.0 / 200.0, 1.0);
        expected += 0.25 * (1.0 / (1.0 + 1.0 / 3.0));

        ConfidenceResult r = scorer().score(healthy(), true, false);
        assertEquals(expected, r.score(), 0.0, "score must match the documented arithmetic bitwise");
    }

    @Test
    public void identicalInputsAlwaysProduceIdenticalResults() {
        ConfidenceScorer s = scorer();
        ConfidenceResult a = s.score(healthy(), true, false);
        ConfidenceResult b = s.score(healthy(), true, false);

        assertEquals(a, b, "the scorer must be a pure function of its inputs");

        // And no hidden state accumulates across calls: interleaving other frames must not change
        // the verdict for an identical input. This is what lets the offline re-scorer reproduce a
        // verdict from a single persisted row.
        s.score(SignalBlock.UNAVAILABLE, true, false);
        s.score(healthy(), false, false);
        assertEquals(a, s.score(healthy(), true, false));
    }

    @Test
    public void anEstimatorFailureIsNotProducedAndNotAJudgement() {
        ConfidenceResult r = scorer().score(healthy(), false, false);

        assertEquals(ConfidenceOutcome.NOT_PRODUCED, r.outcome());
        assertEquals(ConfidenceReason.ESTIMATOR_FAILED, r.reason());
        assertNull(r.score(), "there is no estimate to score");
        assertTrue(!r.poseConsumable());
    }

    @Test
    public void theFirstFrameIsWithheldRatherThanScoredOrCalledNotProduced() {
        // A pose exists on frame 0 (the origin) but no motion was estimated against anything.
        // NOT_PRODUCED is reserved for success == false -- the run-record contract pins that as a
        // biconditional -- so the first frame must be REJECTED/NOT_ESTABLISHED instead.
        ConfidenceResult r = scorer().score(healthy(), true, true);

        assertEquals(ConfidenceOutcome.REJECTED, r.outcome());
        assertEquals(ConfidenceReason.NOT_ESTABLISHED, r.reason());
        assertNull(r.score(), "nothing can be judged on the first frame");
    }

    @Test
    public void anAbsentRequiredSignalYieldsSignalsUnavailableAndNeverADefault() {
        // The single most important behaviour in the class. A "reasonable default" here is exactly
        // how an unmeasured frame becomes a confident one.
        SignalBlock noResidual = SignalBlock.builder()
                .trackCount(200)
                .inlierCount(180)
                .build();

        ConfidenceResult r = scorer().score(noResidual, true, false);

        assertEquals(ConfidenceReason.SIGNALS_UNAVAILABLE, r.reason());
        assertEquals(ConfidenceOutcome.REJECTED, r.outcome());
        assertNull(r.score(), "an unjudgeable frame must not receive a score at all");
    }

    @Test
    public void absenceShortCircuitsBeforeAnyRejectionRuleIsEvaluated() {
        // A frame with an absent residual but a track count below the minimum must report the
        // absence, not the track-count rejection: we cannot claim to have applied a criterion we
        // could not fully evaluate.
        SignalBlock partial = SignalBlock.builder()
                .trackCount(10)
                .inlierCount(9)
                .build();

        assertEquals(ConfidenceReason.SIGNALS_UNAVAILABLE, scorer().score(partial, true, false).reason());
    }

    @Test
    public void completelyAbsentSignalsAreUnavailableRatherThanMaximallyBad() {
        ConfidenceResult r = scorer().score(SignalBlock.UNAVAILABLE, true, false);

        assertEquals(ConfidenceReason.SIGNALS_UNAVAILABLE, r.reason());
        assertNull(r.score());
    }

    @Test
    public void aLowScoreDegradesButDoesNotBorrowARejectionReason() {
        // Below the usable bound, but no named condition fired: 0.5*0.35 + 0.25*(50/200)
        //   + 0.25*(1/(1+2.9/3)) = 0.175 + 0.0625 + ~0.1272 = ~0.3647
        SignalBlock weak = SignalBlock.builder()
                .trackCount(50)
                .inlierCount(18)          // ratio 0.36, above the 0.30 rejection threshold
                .residualInlierCount(18)
                .residualMeanSqPx(2.9)    // below the 3.0 rejection threshold
                .build();

        ConfidenceResult r = scorer().score(weak, true, false);

        assertEquals(ConfidenceOutcome.DEGRADED, r.outcome());
        assertEquals(ConfidenceReason.LOW_SCORE, r.reason(),
                "a degradation must not carry a rejection condition's name");
        assertNotNull(r.score());
        assertTrue(r.score() < 0.6, "expected a score below the usable bound, got " + r.score());
        assertTrue(r.poseConsumable(), "a degraded estimate is still a real estimate");
    }

    @Test
    public void rejectionRulesAreEvaluatedBeforeTheScore() {
        // This frame would also score low, but the verdict must name the condition, not the score.
        SignalBlock fewTracks = SignalBlock.builder()
                .trackCount(10)
                .inlierCount(9)
                .residualInlierCount(9)
                .residualMeanSqPx(0.5)
                .build();

        ConfidenceResult r = scorer().score(fewTracks, true, false);

        assertEquals(ConfidenceOutcome.REJECTED, r.outcome());
        assertEquals(ConfidenceReason.INSUFFICIENT_FEATURES, r.reason());
        assertNull(r.score(), "a rejected frame carries no score");
    }

    @Test
    public void theScoreIsAlwaysInRangeOrAbsent() {
        ConfidenceScorer s = scorer();
        SignalBlock[] cases = {
                healthy(),
                SignalBlock.builder().trackCount(100000).inlierCount(100000)
                        .residualInlierCount(100000).residualMeanSqPx(0.0).build(),
                SignalBlock.builder().trackCount(31).inlierCount(31)
                        .residualInlierCount(31).residualMeanSqPx(2.999).build(),
        };
        for (SignalBlock b : cases) {
            Double score = s.score(b, true, false).score();
            if (score != null) {
                assertTrue(score >= 0.0 && score <= 1.0, "score out of range: " + score);
            }
        }
    }

    @Test
    public void aPerfectFrameScoresOneAndIsUsable() {
        SignalBlock perfect = SignalBlock.builder()
                .trackCount(200)
                .inlierCount(200)
                .residualInlierCount(200)
                .residualMeanSqPx(0.0)   // a genuine perfect fit, distinct from absence
                .build();

        ConfidenceResult r = scorer().score(perfect, true, false);
        assertEquals(1.0, r.score(), 0.0);
        assertEquals(ConfidenceOutcome.USABLE, r.outcome());
    }

    @Test
    public void resultConstructionRefusesContradictoryVerdicts() {
        SignalBlock s = healthy();

        // A score attached to a frame nobody could judge.
        assertThrows(IllegalArgumentException.class, () -> new ConfidenceResult(
                ConfidenceOutcome.REJECTED, ConfidenceReason.SIGNALS_UNAVAILABLE, 0.5, s, "id", "d"));
        // A verdict formed with no value behind it.
        assertThrows(IllegalArgumentException.class, () -> new ConfidenceResult(
                ConfidenceOutcome.USABLE, ConfidenceReason.OK, null, s, "id", "d"));
        // A rejection that names no condition.
        assertThrows(IllegalArgumentException.class, () -> new ConfidenceResult(
                ConfidenceOutcome.REJECTED, ConfidenceReason.LOW_SCORE, null, s, "id", "d"));
        // A degradation borrowing a condition's name.
        assertThrows(IllegalArgumentException.class, () -> new ConfidenceResult(
                ConfidenceOutcome.DEGRADED, ConfidenceReason.INSUFFICIENT_FEATURES, 0.2, s, "id", "d"));
        // NOT_PRODUCED for anything other than an estimator failure.
        assertThrows(IllegalArgumentException.class, () -> new ConfidenceResult(
                ConfidenceOutcome.NOT_PRODUCED, ConfidenceReason.NOT_ESTABLISHED, null, s, "id", "d"));
        // A score outside [0,1].
        assertThrows(IllegalArgumentException.class, () -> new ConfidenceResult(
                ConfidenceOutcome.USABLE, ConfidenceReason.OK, 1.5, s, "id", "d"));
    }

    @Test
    public void everyResultCarriesItsCalibrationIdentity() {
        // FR-005/FR-018: a number must always be traceable to the configuration that produced it,
        // and the bootstrap identity carries the word 'unvalidated' with it.
        ConfidenceScorer s = scorer();
        for (ConfidenceResult r : new ConfidenceResult[]{
                s.score(healthy(), true, false),
                s.score(healthy(), false, false),
                s.score(SignalBlock.UNAVAILABLE, true, false),
                s.score(healthy(), true, true)}) {
            assertEquals("test-unvalidated-v1", r.calibrationId());
            assertTrue(r.calibrationDigest().startsWith("sha256:"));
        }
    }

    // ------------------------------------------------------------------
    // Schema-2.0.0 score models (DEC-CONF-002 as amended)
    // ------------------------------------------------------------------

    static final String LOGISTIC_CONFIG = """
            {
              "schema_version": "2.0.0",
              "calibration_id": "test-logistic-unvalidated-v1",
              "validated": false,
              "validated_by": null,
              "probability_semantics": false,
              "required_signals": ["relative_support", "inc_flow_px"],
              "configuration_binding": {"motion_model": "homography"},
              "temporal": {"window_w": 10, "warmup_m": 5},
              "rejection_rules": [],
              "score_model": {
                "type": "logistic",
                "intercept": -1.0,
                "coefficients": [
                  {"signal": "relative_support", "coefficient": 2.0},
                  {"signal": "inc_flow_px", "coefficient": -0.05}
                ]
              },
              "usable_score_bound": 0.5
            }
            """;

    static final String ISOTONIC_CONFIG = """
            {
              "schema_version": "2.0.0",
              "calibration_id": "test-isotonic-unvalidated-v1",
              "validated": false,
              "validated_by": null,
              "probability_semantics": false,
              "required_signals": ["relative_support"],
              "configuration_binding": {"motion_model": "homography"},
              "temporal": {"window_w": 10, "warmup_m": 5},
              "rejection_rules": [],
              "score_model": {
                "type": "isotonic",
                "signal": "relative_support",
                "thresholds": [0.25, 0.5, 0.9],
                "values": [0.05, 0.2, 0.6, 0.9]
              },
              "usable_score_bound": 0.5
            }
            """;

    private static SignalBlock withWindowedSignals(double relativeSupport, double incFlowPx) {
        return SignalBlock.builder()
                .trackCount(200)
                .inlierCount(180)
                .relativeSupport(relativeSupport)
                .incFlowPx(incFlowPx)
                .build();
    }

    @Test
    public void logisticScoreMatchesTheAlgebraicSquashExactly() {
        // z = -1.0 + 2.0*0.95 + (-0.05)*4.0 = 0.7; s = 0.5 + 0.5*(z/(1+|z|)).
        // Computed here in the same left-to-right order the scorer uses; the squash replaces the
        // sigmoid because exp() is not identically rounded across languages
        // (contracts/calibration-config.md).
        double z = -1.0;
        z += 2.0 * 0.95;
        z += -0.05 * 4.0;
        double expected = 0.5 + 0.5 * (z / (1.0 + Math.abs(z)));

        ConfidenceScorer s = new ConfidenceScorer(CalibrationConfig.parse(LOGISTIC_CONFIG));
        ConfidenceResult r = s.score(withWindowedSignals(0.95, 4.0), true, false);

        assertEquals(expected, r.score(), 0.0, "logistic score must match the squash bitwise");
        assertEquals(ConfidenceOutcome.USABLE, r.outcome());
        assertTrue(r.score() > 0.0 && r.score() < 1.0, "the squash lies strictly inside (0,1)");
    }

    @Test
    public void logisticScoreIsMonotoneInItsPredictor() {
        ConfidenceScorer s = new ConfidenceScorer(CalibrationConfig.parse(LOGISTIC_CONFIG));
        double low = s.score(withWindowedSignals(0.2, 4.0), true, false).score();
        double mid = s.score(withWindowedSignals(0.9, 4.0), true, false).score();
        double high = s.score(withWindowedSignals(1.4, 4.0), true, false).score();
        assertTrue(low < mid && mid < high, "score must be strictly monotone in relative support");

        double fast = s.score(withWindowedSignals(0.9, 60.0), true, false).score();
        assertTrue(fast < mid, "more flow must lower the score under a negative coefficient");
    }

    @Test
    public void logisticWithAnAbsentSignalShortCircuitsToUnavailable() {
        ConfidenceScorer s = new ConfidenceScorer(CalibrationConfig.parse(LOGISTIC_CONFIG));
        // relative_support absent (warm-up), inc_flow_px present: no partial arithmetic.
        SignalBlock warmup = SignalBlock.builder()
                .trackCount(200).inlierCount(180).incFlowPx(4.0).build();
        ConfidenceResult r = s.score(warmup, true, false);
        assertEquals(ConfidenceOutcome.REJECTED, r.outcome());
        assertEquals(ConfidenceReason.SIGNALS_UNAVAILABLE, r.reason());
        assertNull(r.score());
    }

    @Test
    public void isotonicScoreIsThePiecewiseValueWithStrictThresholds() {
        ConfidenceScorer s = new ConfidenceScorer(CalibrationConfig.parse(ISOTONIC_CONFIG));

        assertEquals(0.05, s.score(withWindowedSignals(0.1, 0.0), true, false).score(), 0.0);
        // Exactly on a threshold: strict '>' means the value stays in the LOWER bin — the same
        // strictness discipline Op.GT uses, so both languages agree on the boundary.
        assertEquals(0.05, s.score(withWindowedSignals(0.25, 0.0), true, false).score(), 0.0);
        assertEquals(0.2, s.score(withWindowedSignals(0.3, 0.0), true, false).score(), 0.0);
        assertEquals(0.6, s.score(withWindowedSignals(0.7, 0.0), true, false).score(), 0.0);
        assertEquals(0.9, s.score(withWindowedSignals(2.0, 0.0), true, false).score(), 0.0);
    }
}
