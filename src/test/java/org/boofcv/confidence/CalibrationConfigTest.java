package org.boofcv.confidence;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The calibration is read, not compiled in, and it refuses what it cannot honour
 * (feature spec FR-015, FR-017, FR-018, FR-019; schema 2.0.0 per {@code DEC-CONF-002} as amended).
 *
 * <p>Constitution Principle XIII names "serialization of experiment configuration" as a required
 * test target; this is that.
 *
 * <p>Evidence tier: <b>T1 (analytical)</b>.
 */
public class CalibrationConfigTest {

    private static final String MINIMAL = """
            {
              "schema_version": "2.0.0",
              "calibration_id": "test-unvalidated-v1",
              "validated": false,
              "validated_by": null,
              "probability_semantics": false,
              "required_signals": ["track_count", "inlier_ratio"],
              "configuration_binding": {"motion_model": "homography", "downsampleFactor": 2},
              "temporal": {"window_w": 10, "warmup_m": 5},
              "rejection_rules": [
                {"signal": "track_count", "op": "lt", "threshold": 30.0, "reason": "insufficient_features"}
              ],
              "score_model": {
                "type": "weighted_sum",
                "terms": [
                  {"signal": "inlier_ratio", "normalise": "identity", "reference": null, "weight": 1.0}
                ]
              },
              "usable_score_bound": 0.6
            }
            """;

    private static final String LOGISTIC = """
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

    private static final String ISOTONIC = """
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

    @Test
    public void parsesAndExposesEveryField() {
        CalibrationConfig c = CalibrationConfig.parse(MINIMAL);

        assertEquals("2.0.0", c.schemaVersion());
        assertEquals("test-unvalidated-v1", c.calibrationId());
        assertFalse(c.validated());
        assertNull(c.validatedBy());
        assertFalse(c.probabilitySemantics());
        assertEquals(2, c.requiredSignals().size());
        assertEquals(1, c.rejectionRules().size());
        assertEquals(ConfidenceReason.INSUFFICIENT_FEATURES, c.rejectionRules().get(0).reason());
        assertEquals(CalibrationConfig.Op.LT, c.rejectionRules().get(0).op());
        assertEquals(0.6, c.usableScoreBound(), 0.0);
        assertEquals(10, c.windowW());
        assertEquals(5, c.warmupM());
        assertEquals("weighted_sum", c.scoreModel().typeName());
        assertEquals(2, c.configurationBinding().size());
    }

    @Test
    public void schemaOneIsRefusedWithAMessageNamingTheReason() {
        String v1 = MINIMAL.replace("\"schema_version\": \"2.0.0\"", "\"schema_version\": \"1.0.0\"");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> CalibrationConfig.parse(v1));
        assertTrue(e.getMessage().contains("2.0.0"), e.getMessage());
    }

    @Test
    public void logisticAndIsotonicModelsParse() {
        CalibrationConfig lo = CalibrationConfig.parse(LOGISTIC);
        assertEquals("logistic", lo.scoreModel().typeName());
        assertEquals(2, lo.scoreModel().signals().size());

        CalibrationConfig iso = CalibrationConfig.parse(ISOTONIC);
        assertEquals("isotonic", iso.scoreModel().typeName());
        assertEquals("relative_support", iso.scoreModel().signals().get(0));
    }

    @Test
    public void probabilitySemanticsRequiresAValidatedCalibration() {
        // Amendment A1's structural half: no probability claim without a recorded validation.
        String bad = MINIMAL.replace("\"probability_semantics\": false",
                "\"probability_semantics\": true");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> CalibrationConfig.parse(bad));
        assertTrue(e.getMessage().contains("A1"), e.getMessage());
    }

    @Test
    public void bindingAndTemporalBlocksAreRequired() {
        // Amendment A3: no stated validity domain, no calibration.
        String noBinding = MINIMAL.replace(
                "\"configuration_binding\": {\"motion_model\": \"homography\", \"downsampleFactor\": 2},", "");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(noBinding));

        String emptyBinding = MINIMAL.replace(
                "\"configuration_binding\": {\"motion_model\": \"homography\", \"downsampleFactor\": 2}",
                "\"configuration_binding\": {}");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(emptyBinding));

        // Amendment A4: the temporal identity is part of the calibration.
        String noTemporal = MINIMAL.replace("\"temporal\": {\"window_w\": 10, \"warmup_m\": 5},", "");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(noTemporal));

        String badTemporal = MINIMAL.replace("\"warmup_m\": 5", "\"warmup_m\": 11");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(badTemporal));
    }

    @Test
    public void bindingCheckRefusesMismatchAndAbsence() {
        CalibrationConfig c = CalibrationConfig.parse(MINIMAL);

        // Matching context passes.
        c.checkBinding(Map.of("motion_model", "homography", "downsampleFactor", 2, "extra", 1), "run-x");

        // Integer-vs-double representations of the same number match (Jackson may give either).
        c.checkBinding(Map.of("motion_model", "homography", "downsampleFactor", 2.0), "run-x");

        // A mismatched value refuses.
        IllegalArgumentException mismatch = assertThrows(IllegalArgumentException.class,
                () -> c.checkBinding(Map.of("motion_model", "homography", "downsampleFactor", 1), "run-x"));
        assertTrue(mismatch.getMessage().contains("downsampleFactor"), mismatch.getMessage());

        // An absent key refuses — an unknown context cannot be shown to match (A3).
        IllegalArgumentException absent = assertThrows(IllegalArgumentException.class,
                () -> c.checkBinding(Map.of("motion_model", "homography"), "run-x"));
        assertTrue(absent.getMessage().contains("does not carry"), absent.getMessage());
    }

    @Test
    public void digestIsStableAcrossReadsAndInsensitiveToLineEndings() {
        // The digest hashes bytes, not a re-serialisation, because Python's json.dumps and Java's
        // Double.toString disagree on the textual form of some doubles (1e-07 vs 1.0E-7) -- a
        // canonical-re-serialisation digest would be a latent cross-language mismatch, exactly the
        // class of divergence the agreement invariant exists to prevent. Line endings are
        // normalised because this repository checks out CRLF on Windows.
        String d1 = CalibrationConfig.parse(MINIMAL).digest();
        String d2 = CalibrationConfig.parse(MINIMAL).digest();
        assertEquals(d1, d2, "same bytes must give the same digest");

        String crlf = MINIMAL.replace("\n", "\r\n");
        assertEquals(d1, CalibrationConfig.parse(crlf).digest(),
                "line-ending normalisation must not change the digest");

        assertTrue(d1.startsWith("sha256:"));
    }

    @Test
    public void digestChangesWhenAnyValueChanges() {
        String other = MINIMAL.replace("\"threshold\": 30.0", "\"threshold\": 31.0");
        assertNotEquals(CalibrationConfig.parse(MINIMAL).digest(),
                CalibrationConfig.parse(other).digest(),
                "a changed threshold must produce a different digest");

        String otherW = MINIMAL.replace("\"window_w\": 10", "\"window_w\": 20");
        assertNotEquals(CalibrationConfig.parse(MINIMAL).digest(),
                CalibrationConfig.parse(otherW).digest(),
                "a different W is a different calibration (amendment A4)");
    }

    @Test
    public void unknownFieldsAreRefusedRatherThanIgnored() {
        // A typo must be an error, not a silently ignored term that changes the score.
        String typo = MINIMAL.replace("\"usable_score_bound\"", "\"usable_score_bnd\"");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(typo));
    }

    @Test
    public void unknownSignalNamesAreRefusedAtLoadTime() {
        // Failing when the configuration is read, not silently at frame 4000.
        String bad = MINIMAL.replace("\"track_count\", \"inlier_ratio\"", "\"track_cout\", \"inlier_ratio\"");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(bad));
    }

    @Test
    public void unknownOpNormalisationReasonAndModelTypeAreRefused() {
        assertThrows(IllegalArgumentException.class,
                () -> CalibrationConfig.parse(MINIMAL.replace("\"op\": \"lt\"", "\"op\": \"leq\"")));
        assertThrows(IllegalArgumentException.class,
                () -> CalibrationConfig.parse(MINIMAL.replace("\"normalise\": \"identity\"", "\"normalise\": \"linear\"")));
        assertThrows(IllegalArgumentException.class,
                () -> CalibrationConfig.parse(MINIMAL.replace("\"reason\": \"insufficient_features\"", "\"reason\": \"vibes\"")));
        assertThrows(IllegalArgumentException.class,
                () -> CalibrationConfig.parse(MINIMAL.replace("\"type\": \"weighted_sum\"", "\"type\": \"max\"")));
    }

    @Test
    public void mixedModelFieldsAreRefused() {
        // A weighted_sum carrying an intercept means someone edited the type and forgot the body.
        String bad = LOGISTIC.replace("\"type\": \"logistic\"", "\"type\": \"weighted_sum\"");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(bad));
    }

    @Test
    public void isotonicShapeErrorsAreRefused() {
        String badLen = ISOTONIC.replace("\"values\": [0.05, 0.2, 0.6, 0.9]",
                "\"values\": [0.05, 0.2, 0.6]");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(badLen));

        String notAscending = ISOTONIC.replace("\"thresholds\": [0.25, 0.5, 0.9]",
                "\"thresholds\": [0.25, 0.25, 0.9]");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(notAscending));

        String outOfRange = ISOTONIC.replace("\"values\": [0.05, 0.2, 0.6, 0.9]",
                "\"values\": [0.05, 0.2, 0.6, 1.9]");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(outOfRange));
    }

    @Test
    public void weightsNotSummingToOneAreRefused() {
        // Not a mathematical necessity for a weighted sum -- but a set that does not sum to 1
        // almost always means a term was edited and another forgotten, and the score would
        // silently leave [0,1].
        String bad = MINIMAL.replace("\"weight\": 1.0", "\"weight\": 0.8");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(bad));
    }

    @Test
    public void aRuleOrModelReadingAnUndeclaredSignalIsRefused() {
        // required_signals is what drives FR-017's refusal to score a run lacking a signal. A rule
        // reading something outside it would make that refusal silently incomplete.
        String bad = MINIMAL.replace("\"required_signals\": [\"track_count\", \"inlier_ratio\"]",
                "\"required_signals\": [\"inlier_ratio\"]");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> CalibrationConfig.parse(bad));
        assertTrue(e.getMessage().contains("required_signals"), e.getMessage());

        String badModel = LOGISTIC.replace(
                "\"required_signals\": [\"relative_support\", \"inc_flow_px\"]",
                "\"required_signals\": [\"relative_support\"]");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(badModel));
    }

    @Test
    public void aRejectionRuleMustNameAConditionNotALowScore() {
        String bad = MINIMAL.replace("\"reason\": \"insufficient_features\"", "\"reason\": \"low_score\"");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(bad));
    }

    @Test
    public void aNormalisationRequiringAReferenceIsRefusedWithoutOne() {
        String bad = MINIMAL.replace(
                "{\"signal\": \"inlier_ratio\", \"normalise\": \"identity\", \"reference\": null, \"weight\": 1.0}",
                "{\"signal\": \"inlier_ratio\", \"normalise\": \"saturating\", \"reference\": null, \"weight\": 1.0}");
        assertThrows(IllegalArgumentException.class, () -> CalibrationConfig.parse(bad));
    }

    @Test
    public void rejectionRuleOrderIsPreserved() {
        // Order determines which reason code a frame receives when several conditions hold, so it
        // is part of the configuration and part of the digest.
        String twoRules = MINIMAL.replace(
                "{\"signal\": \"track_count\", \"op\": \"lt\", \"threshold\": 30.0, \"reason\": \"insufficient_features\"}",
                "{\"signal\": \"inlier_ratio\", \"op\": \"lt\", \"threshold\": 0.3, \"reason\": \"poor_inlier_support\"},"
                        + "{\"signal\": \"track_count\", \"op\": \"lt\", \"threshold\": 30.0, \"reason\": \"insufficient_features\"}");
        CalibrationConfig c = CalibrationConfig.parse(twoRules);

        assertEquals(ConfidenceReason.POOR_INLIER_SUPPORT, c.rejectionRules().get(0).reason());
        assertEquals(ConfidenceReason.INSUFFICIENT_FEATURES, c.rejectionRules().get(1).reason());
    }

    @Test
    public void comparisonsAreStrictSoBothLanguagesAgreeOnAThresholdValue() {
        // '<' vs '<=' is the classic way two implementations of "the same rule" quietly disagree,
        // and it only shows up on a value sitting exactly on the threshold.
        assertFalse(CalibrationConfig.Op.LT.test(30.0, 30.0), "lt must be strict");
        assertTrue(CalibrationConfig.Op.LT.test(29.999, 30.0));
        assertFalse(CalibrationConfig.Op.GT.test(3.0, 3.0), "gt must be strict");
        assertTrue(CalibrationConfig.Op.GT.test(3.001, 3.0));
    }

    @Test
    public void theShippedCalibrationsLoadAndDeclareThemselvesUnvalidated() throws IOException {
        for (String name : new String[]{"bootstrap-unvalidated-v2.json", "contrast-unvalidated-v2.json"}) {
            Path p = Paths.get("evaluation/confidence_configs/" + name);
            assertTrue(Files.exists(p), "calibration missing at " + p.toAbsolutePath());

            CalibrationConfig c = CalibrationConfig.load(p);

            // SC-009: no shipped artifact may present an unvalidated threshold as validated. The
            // word lives in the identity so it travels into frames.csv and any exported table.
            assertFalse(c.validated(), name + " must declare itself unvalidated");
            assertNull(c.validatedBy());
            assertFalse(c.probabilitySemantics(),
                    name + " must not claim probability semantics (amendment A1)");
            assertTrue(c.calibrationId().contains("unvalidated"),
                    "the calibration id must carry the word 'unvalidated' so it survives copy-paste "
                            + "into a results table; got '" + c.calibrationId() + "'");
            assertEquals(10, c.windowW(), name + " must carry the frozen W (EXP-CONF-001 P1)");
            assertEquals(5, c.warmupM(), name + " must carry the frozen m");
        }
    }

    @Test
    public void normalisationsMatchTheirDocumentedDefinitions() {
        assertEquals(0.5, CalibrationConfig.Normalisation.IDENTITY.apply(0.5, null), 0.0);
        assertEquals(1.0, CalibrationConfig.Normalisation.IDENTITY.apply(1.7, null), 0.0, "clamped");
        assertEquals(0.0, CalibrationConfig.Normalisation.IDENTITY.apply(-0.2, null), 0.0, "clamped");

        assertEquals(0.5, CalibrationConfig.Normalisation.SATURATING.apply(100.0, 200.0), 0.0);
        assertEquals(1.0, CalibrationConfig.Normalisation.SATURATING.apply(400.0, 200.0), 0.0, "saturates");

        // 1 / (1 + 3/3) = 0.5
        assertEquals(0.5, CalibrationConfig.Normalisation.INVERSE_SATURATING.apply(3.0, 3.0), 0.0);
        assertEquals(1.0, CalibrationConfig.Normalisation.INVERSE_SATURATING.apply(0.0, 3.0), 0.0,
                "a perfect fit normalises to 1");
    }
}
