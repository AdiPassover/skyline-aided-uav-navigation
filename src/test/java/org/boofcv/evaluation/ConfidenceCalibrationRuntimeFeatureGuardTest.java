package org.boofcv.evaluation;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A latent integration hazard between the confidence layer and the metric/heading readout: the
 * {@code confidence_calibration} instrumented-estimator path ({@code DEC-CONF-002}) is built on a
 * separate construction line from {@link VoRunnerApp#buildEstimator}, and never calls
 * {@code enableMetricReadout}/{@code enableHeadingReadout}. Left unguarded, a config combining
 * {@code confidence_calibration} with {@code metric_readout} or {@code heading_readout} would not
 * fail -- it would silently produce a pixel-valued pose while the manifest still advertised the
 * requested metric/heading block (and, with {@code publish_as_navigation_source}, a navigation-unit
 * mismatch).
 *
 * <p>This fixes nothing about the underlying incompatibility: it only makes
 * {@link VoRunnerApp#checkConfidenceCalibrationVsRuntimeFeatures} refuse the combination
 * explicitly, at config-validation time, before any estimator is built. The combined path itself
 * remains INT's responsibility (bit-equivalence, metric units, authoritative heading, manifest
 * truthfulness, all interacting with the instrumented estimator).
 */
class ConfidenceCalibrationRuntimeFeatureGuardTest {

    private static VoRunnerConfig confConfig() {
        VoRunnerConfig c = new VoRunnerConfig();
        c.datasetDir = "d";
        c.confidenceCalibration = "some-calibration.json";
        return c;
    }

    private static MetricReadoutSettings enabledMetric() {
        MetricReadoutSettings m = new MetricReadoutSettings();
        m.enabled = true;
        m.fxNativePx = 512.0;
        m.h0AglM = 40.0;
        return m;
    }

    private static HeadingReadoutSettings enabledHeading() {
        HeadingReadoutSettings h = new HeadingReadoutSettings();
        h.enabled = true;
        h.headingSemantics = "sim_nadir_camera_heading";
        return h;
    }

    // ================================================================ rejected combinations

    @Test
    @DisplayName("confidence_calibration + metric_readout.enabled is refused, not silently inert")
    void confidenceCalibrationWithMetricReadoutIsRejected() {
        VoRunnerConfig c = confConfig();
        c.metricReadout = enabledMetric();

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c));
        assertTrue(e.getMessage().contains("confidence_calibration")
                        && e.getMessage().contains("metric_readout")
                        && e.getMessage().contains("not yet jointly supported"),
                "message must name both features and say they are not yet jointly supported: "
                        + e.getMessage());
    }

    @Test
    @DisplayName("confidence_calibration + heading_readout.enabled is refused, not silently inert")
    void confidenceCalibrationWithHeadingReadoutIsRejected() {
        VoRunnerConfig c = confConfig();
        c.headingReadout = enabledHeading();

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c));
        assertTrue(e.getMessage().contains("confidence_calibration")
                        && e.getMessage().contains("heading_readout")
                        && e.getMessage().contains("not yet jointly supported"),
                "message must name both features and say they are not yet jointly supported: "
                        + e.getMessage());
    }

    @Test
    @DisplayName("confidence_calibration + metric_readout + heading_readout is refused on the "
            + "metric_readout check first")
    void confidenceCalibrationWithBothIsRejected() {
        VoRunnerConfig c = confConfig();
        c.metricReadout = enabledMetric();
        c.headingReadout = enabledHeading();

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c));
        assertTrue(e.getMessage().contains("metric_readout"));
    }

    @Test
    @DisplayName("publish_as_navigation_source needs no separate check: it is covered by the "
            + "metric_readout guard, since it only takes effect when metric_readout.enabled is true")
    void publishAsNavigationSourceIsCoveredByTheMetricReadoutGuard() {
        VoRunnerConfig c = confConfig();
        c.metricReadout = enabledMetric();
        c.metricReadout.publishAsNavigationSource = true;

        assertThrows(IllegalArgumentException.class,
                () -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c));
    }

    // ================================================================ genuinely supported combinations

    @Test
    @DisplayName("confidence_calibration alone, with no metric/heading readout, is unaffected")
    void confidenceCalibrationAloneStillWorks() {
        VoRunnerConfig c = confConfig();
        assertDoesNotThrow(() -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c));
    }

    @Test
    @DisplayName("a metric_readout block present but not enabled does not trip the guard")
    void aPresentButDisabledMetricBlockDoesNotTripTheGuard() {
        VoRunnerConfig c = confConfig();
        MetricReadoutSettings m = new MetricReadoutSettings();
        m.enabled = false; // block present (e.g. templated config), but the master switch is off
        c.metricReadout = m;
        assertDoesNotThrow(() -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c));
    }

    @Test
    @DisplayName("metric_readout alone, with no confidence_calibration, still works "
            + "(the ordinary VO metric-readout path is untouched by this guard)")
    void metricReadoutAloneStillWorks() {
        VoRunnerConfig c = new VoRunnerConfig();
        c.datasetDir = "d";
        c.metricReadout = enabledMetric();

        assertDoesNotThrow(() -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c));
    }

    @Test
    @DisplayName("heading_readout alone, with no confidence_calibration, still works "
            + "(the ordinary VO heading-readout path is untouched by this guard)")
    void headingReadoutAloneStillWorks() {
        VoRunnerConfig c = new VoRunnerConfig();
        c.datasetDir = "d";
        c.metricReadout = enabledMetric(); // heading_readout requires metric_readout (DEC-VO-010)
        c.headingReadout = enabledHeading();

        assertDoesNotThrow(() -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c));
    }

    @Test
    @DisplayName("a plain config with neither feature set is unaffected")
    void aPlainConfigIsUnaffected() {
        VoRunnerConfig c = new VoRunnerConfig();
        c.datasetDir = "d";
        assertDoesNotThrow(() -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c));
    }

    // ================================================================ real committed configs
    //
    // The synthetic-object tests above cover the guard's logic; these tie it to the actual
    // committed configs so a change here cannot silently start rejecting a config someone already
    // relies on. Loading only -- these configs point at datasets that are not part of the
    // repository and are never run here.

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static VoRunnerConfig loadConfig(String repoRelativePath) throws IOException {
        return MAPPER.readValue(new File(repoRelativePath), VoRunnerConfig.class);
    }

    @Test
    @DisplayName("the committed EXP-CONF-001 confidence-calibration configs still pass the guard")
    void committedConfConfigsStillWork() throws IOException {
        for (String path : new String[]{
                "evaluation/eval_configs/exp-conf-001/run-amtown01-c-conf.json",
                "evaluation/eval_configs/exp-conf-001/run-amtown01-d-conf.json",
                "evaluation/eval_configs/exp-conf-001/run-amtown02-conf.json",
                "evaluation/eval_configs/exp-conf-001/run-hkairport01-a-conf.json",
                "evaluation/eval_configs/exp-conf-001/run-hkairport01-b-conf.json",
                "evaluation/eval_configs/exp-conf-001/run-hkisland01-conf.json"}) {
            VoRunnerConfig c = loadConfig(path);
            assertTrue(c.confidenceCalibration != null, path + ": expected confidence_calibration set");
            assertDoesNotThrow(() -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c),
                    path + " must still pass now that CONF/VO combinations are guarded");
        }
    }

    @Test
    @DisplayName("the committed DEC-VO-009 metric-readout-only configs still pass the guard")
    void committedMetricOnlyConfigsStillWork() throws IOException {
        for (String path : new String[]{
                "evaluation/eval_configs/vo-metric-runtime/run-uevo-fig8-vary-metric.json",
                "evaluation/eval_configs/vo-metric-runtime/run-uevo-fig8-const-metric-published.json",
                "evaluation/eval_configs/vo-metric-runtime/run-timing-bvo-metric-on.json"}) {
            VoRunnerConfig c = loadConfig(path);
            assertTrue(c.metricReadout != null && c.metricReadout.enabled,
                    path + ": expected metric_readout.enabled");
            assertTrue(c.confidenceCalibration == null,
                    path + ": expected no confidence_calibration -- this is the metric-only case");
            assertDoesNotThrow(() -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c), path);
        }
    }

    @Test
    @DisplayName("the committed DEC-VO-010 heading-readout configs still pass the guard")
    void committedHeadingConfigsStillWork() throws IOException {
        for (String path : new String[]{
                "evaluation/eval_configs/vo-heading-runtime/run-uevo-fig8-vary-heading.json",
                "evaluation/eval_configs/vo-heading-runtime/run-amtown01-d-heading.json",
                "evaluation/eval_configs/vo-heading-runtime/run-hkairport01-a-heading.json"}) {
            VoRunnerConfig c = loadConfig(path);
            assertTrue(c.headingReadout != null && c.headingReadout.enabled,
                    path + ": expected heading_readout.enabled");
            assertTrue(c.confidenceCalibration == null,
                    path + ": expected no confidence_calibration -- this is the heading case");
            assertDoesNotThrow(() -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c), path);
        }
    }
}
