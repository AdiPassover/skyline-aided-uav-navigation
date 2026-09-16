package org.boofcv.evaluation;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The relocalization layer's pose-contract guard (test A). It began the post-merge compatibility
 * pass as a temporary fail-fast refusing INT with {@code METRIC_LOCAL}; once the layer consumed
 * the metric, externally-heading-referenced state and the invariant tests proved it, the guard was
 * inverted: INT now <b>requires</b> the metric readout and the heading channel, and is refused
 * without them. {@code metric_readout} as a diagnostic without INT is untouched.
 */
class RelocalizationNavigationContractGuardTest {

    private static MetricReadoutSettings enabledMetric(boolean publish) {
        MetricReadoutSettings m = new MetricReadoutSettings();
        m.enabled = true;
        m.fxNativePx = 512.0;
        m.h0AglM = 40.0;
        m.publishAsNavigationSource = publish;
        return m;
    }

    private static HeadingReadoutSettings enabledHeading() {
        HeadingReadoutSettings h = new HeadingReadoutSettings();
        h.enabled = true;
        h.headingSemantics = "sim_nadir_camera_heading";
        return h;
    }

    @Test
    @DisplayName("supported: INT + metric_readout + heading_readout, published as METRIC_LOCAL (route A)")
    void intWithMetricAndHeadingPublishedIsSupported() {
        VoRunnerConfig c = new VoRunnerConfig();
        c.datasetDir = "d";
        c.alignmentSidecar = true;
        c.metricReadout = enabledMetric(true);
        c.headingReadout = enabledHeading();
        assertDoesNotThrow(() -> VoRunnerApp.checkRelocalizationVsNavigationContract(c));
    }

    @Test
    @DisplayName("supported: INT + metric_readout + heading_readout with navigation_source metric_local (route B)")
    void intWithMetricAndHeadingNamedIsSupported() {
        VoRunnerConfig c = new VoRunnerConfig();
        c.datasetDir = "d";
        c.relocalizationConfig = "reloc.json";
        c.navigationSource = "metric_local";
        c.metricReadout = enabledMetric(false);
        c.headingReadout = enabledHeading();
        assertDoesNotThrow(() -> VoRunnerApp.checkRelocalizationVsNavigationContract(c));
    }

    @Test
    @DisplayName("supported: INT + metric_readout + heading_readout while the published source stays RIGID_MOTION")
    void intWithMetricDiagnosticOnlyIsSupported() {
        VoRunnerConfig c = new VoRunnerConfig();
        c.datasetDir = "d";
        c.relocalizationConfig = "reloc.json";
        c.metricReadout = enabledMetric(false);
        c.headingReadout = enabledHeading();
        assertDoesNotThrow(() -> VoRunnerApp.checkRelocalizationVsNavigationContract(c),
                "the layer reads the metric state directly; which source publishes frames.csv is irrelevant");
    }

    @Test
    @DisplayName("refused: INT on the pixel readout (no metric_readout) — there is no pixel contract")
    void intWithoutMetricReadoutIsRefused() {
        VoRunnerConfig c = new VoRunnerConfig();
        c.datasetDir = "d";
        c.alignmentSidecar = true;
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> VoRunnerApp.checkRelocalizationVsNavigationContract(c));
        assertTrue(e.getMessage().contains("metric_readout.enabled"), e.getMessage());
        assertTrue(e.getMessage().contains("pixel"), e.getMessage());
    }

    @Test
    @DisplayName("refused: INT on the metric state rotated by the VISUAL yaw (no heading_readout)")
    void intWithoutHeadingReadoutIsRefused() {
        VoRunnerConfig c = new VoRunnerConfig();
        c.datasetDir = "d";
        c.relocalizationConfig = "reloc.json";
        c.metricReadout = enabledMetric(true);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> VoRunnerApp.checkRelocalizationVsNavigationContract(c));
        assertTrue(e.getMessage().contains("heading_readout.enabled"), e.getMessage());
        assertTrue(e.getMessage().contains("visual yaw"), e.getMessage());
    }

    @Test
    @DisplayName("untouched: metric_readout as a diagnostic, or METRIC_LOCAL, without any INT layer")
    void metricWithoutIntIsUntouched() {
        VoRunnerConfig a = new VoRunnerConfig();
        a.datasetDir = "d";
        a.metricReadout = enabledMetric(false);
        assertDoesNotThrow(() -> VoRunnerApp.checkRelocalizationVsNavigationContract(a));
        VoRunnerConfig b = new VoRunnerConfig();
        b.datasetDir = "d";
        b.metricReadout = enabledMetric(true);
        b.headingReadout = enabledHeading();
        assertDoesNotThrow(() -> VoRunnerApp.checkRelocalizationVsNavigationContract(b));
        VoRunnerConfig plain = new VoRunnerConfig();
        plain.datasetDir = "d";
        assertDoesNotThrow(() -> VoRunnerApp.checkRelocalizationVsNavigationContract(plain));
    }

    @Test
    @DisplayName("the CONF x VO guard is neither bypassed nor weakened by the INT guard")
    void confVoGuardStillStands() {
        VoRunnerConfig c = new VoRunnerConfig();
        c.datasetDir = "d";
        c.confidenceCalibration = "cal.json";
        c.metricReadout = enabledMetric(true);
        c.headingReadout = enabledHeading();
        c.relocalizationConfig = "reloc.json";
        // INT's own guard is satisfied (metric + heading present) ...
        assertDoesNotThrow(() -> VoRunnerApp.checkRelocalizationVsNavigationContract(c));
        // ... and the CONF x VO guard still refuses the confidence_calibration combination.
        assertThrows(IllegalArgumentException.class,
                () -> VoRunnerApp.checkConfidenceCalibrationVsRuntimeFeatures(c));
    }
}
