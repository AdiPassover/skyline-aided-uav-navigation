package org.boofcv.evaluation;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.boofcv.stitching.NavigationSource;
import org.boofcv.stitching.NavigationUnits;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The evaluation-side plumbing of the runtime metric readout: config parsing, the declared
 * height semantics, and the causal replay of a recorded height stream.
 *
 * <p>The theme is that every way of getting a metric run wrong that this code can detect, it
 * detects — an undeclared {@code h0}, an oracle column masquerading as a barometer, a
 * published-metres flag on a readout that is not the one it scales, or a config that has silently
 * lost its metric block.
 */
class MetricReadoutPlumbingTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    // ================================================================ backward compatibility

    @Test
    @DisplayName("a config with no metric_readout block is unchanged in every respect")
    void absentBlockChangesNothing() throws IOException {
        VoRunnerConfig c = MAPPER.readValue(
                "{\"dataset_dir\":\"d\",\"navigation_source\":\"rigid_motion\"}", VoRunnerConfig.class);
        assertNull(c.metricReadout, "every pre-2026-09-06 config must parse to no metric readout");
        assertEquals(NavigationSource.RIGID_MOTION, VoRunnerApp.resolveNavigationSource(c));
        assertEquals(NavigationUnits.IMAGE_PIXELS, VoRunnerApp.resolveNavigationSource(c).units());
    }

    @Test
    @DisplayName("an enabled block still leaves the published source alone unless it says otherwise")
    void enablingTheReadoutDoesNotPublishMetresByItself() throws IOException {
        VoRunnerConfig c = MAPPER.readValue("{\"dataset_dir\":\"d\",\"metric_readout\":"
                + "{\"enabled\":true,\"fx_native_px\":512.0,\"h0_agl_m\":40.0}}", VoRunnerConfig.class);
        assertEquals(NavigationSource.RIGID_MOTION, VoRunnerApp.resolveNavigationSource(c),
                "computing a metric pose and publishing it are two separate, deliberate acts");
        assertFalse(c.metricReadout.publishAsNavigationSource);
    }

    @Test
    @DisplayName("publish_as_navigation_source selects METRIC_LOCAL and declares metres")
    void publishingSelectsMetricLocal() throws IOException {
        VoRunnerConfig c = MAPPER.readValue("{\"dataset_dir\":\"d\",\"metric_readout\":"
                + "{\"enabled\":true,\"fx_native_px\":512.0,\"h0_agl_m\":40.0,"
                + "\"publish_as_navigation_source\":true}}", VoRunnerConfig.class);
        assertEquals(NavigationSource.METRIC_LOCAL, VoRunnerApp.resolveNavigationSource(c));
        assertEquals(NavigationUnits.METRES, VoRunnerApp.resolveNavigationSource(c).units());
    }

    @Test
    @DisplayName("publishing metres over a legacy or logical readout is refused, not reconciled")
    void publishingOverANonRigidSourceIsRefused() throws IOException {
        for (String src : new String[]{"mosaic_legacy", "logical_frame"}) {
            VoRunnerConfig c = MAPPER.readValue("{\"dataset_dir\":\"d\",\"navigation_source\":\""
                    + src + "\",\"metric_readout\":{\"enabled\":true,\"fx_native_px\":512.0,"
                    + "\"h0_agl_m\":40.0,\"publish_as_navigation_source\":true}}", VoRunnerConfig.class);
            assertThrows(IllegalArgumentException.class, () -> VoRunnerApp.resolveNavigationSource(c),
                    src + ": the metric readout scales RIGID_MOTION's increments and nothing else");
        }
    }

    @Test
    @DisplayName("metric_local is a parseable navigation_source name")
    void metricLocalParses() {
        assertEquals(NavigationSource.METRIC_LOCAL, VoRunnerApp.parseNavigationSource("metric_local"));
        assertEquals(NavigationSource.DEFAULT, VoRunnerApp.parseNavigationSource(null));
        assertThrows(IllegalArgumentException.class, () -> VoRunnerApp.parseNavigationSource("metres"));
    }

    // ================================================================ declared parameters

    @Test
    @DisplayName("h0 and fx_native are required when enabled -- neither is defaulted or inferred")
    void theDeclaredParametersAreRequired() throws IOException {
        VoRunnerConfig noFx = MAPPER.readValue("{\"dataset_dir\":\"d\",\"metric_readout\":"
                + "{\"enabled\":true,\"h0_agl_m\":40.0}}", VoRunnerConfig.class);
        IllegalArgumentException a = assertThrows(IllegalArgumentException.class,
                () -> VoRunnerApp.metricReadoutConfig(noFx));
        assertTrue(a.getMessage().contains("fx_native_px"));

        VoRunnerConfig noH0 = MAPPER.readValue("{\"dataset_dir\":\"d\",\"metric_readout\":"
                + "{\"enabled\":true,\"fx_native_px\":512.0}}", VoRunnerConfig.class);
        IllegalArgumentException b = assertThrows(IllegalArgumentException.class,
                () -> VoRunnerApp.metricReadoutConfig(noH0));
        assertTrue(b.getMessage().contains("h0_agl_m") && b.getMessage().contains("outside"),
                "the message must say h0 comes from outside the system: " + b.getMessage());
    }

    @Test
    @DisplayName("f_working follows the run's own downsampleFactor, so the two cannot drift apart")
    void fWorkingFollowsTheDownsampleFactor() throws IOException {
        VoRunnerConfig c = MAPPER.readValue("{\"dataset_dir\":\"d\",\"downsampleFactor\":2,"
                + "\"metric_readout\":{\"enabled\":true,\"fx_native_px\":1471.0653076171875,"
                + "\"h0_agl_m\":80.0}}", VoRunnerConfig.class);
        assertEquals(735.5326538, VoRunnerApp.metricReadoutConfig(c).fWorkingPx(), 1e-6);
    }

    // ================================================================ the oracle guard

    @Test
    @DisplayName("the height column's MEANING is declared, and an unknown declaration is refused")
    void heightSemanticsMustBeDeclaredFromTheKnownSet() {
        MetricReadoutSettings m = new MetricReadoutSettings();
        assertEquals("takeoff_relative", m.heightSemantics, "production-like is the default");
        assertFalse(m.isOracleArm());

        m.heightSemantics = "oracle_agl_diagnostic";
        assertTrue(m.isOracleArm());

        m.heightSemantics = "true_agl";
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, m::isOracleArm);
        assertTrue(e.getMessage().contains("DIAGNOSTIC ONLY"),
                "the refusal must say why the oracle arm is special: " + e.getMessage());
    }

    @Test
    @DisplayName("the oracle arm is re-zeroed to its own datum; the production-like arm never is")
    void onlyTheOracleArmIsRezeroed(@TempDir Path dir) throws IOException {
        Path csv = dir.resolve("h.csv");
        Files.write(csv, List.of("timestamp_s,baro_relative_m,agl_m",
                                 "0.0,0.0,80.0", "1.0,5.0,85.0", "2.0,9.0,89.0"),
                StandardCharsets.UTF_8);

        // Production-like: the takeoff datum is already zero, so re-zeroing would be a second,
        // silent subtraction.
        HeightSampleStream baro = HeightSampleStream.fromCsv(csv, "timestamp_s", "baro_relative_m",
                0.0, 1, false);
        assertEquals(0.0, baro.drainUpTo(0.0).get(0).relativeHeightM(), 0.0);
        assertEquals(5.0, baro.drainUpTo(1.0).get(0).relativeHeightM(), 0.0);

        // Oracle: an ABSOLUTE AGL column, re-zeroed so the same arithmetic applies. Feeding it
        // through unchanged would give h_AGL = h0 + 80 m on the first frame.
        HeightSampleStream oracle = HeightSampleStream.fromCsv(csv, "timestamp_s", "agl_m",
                0.0, 1, true);
        assertEquals(0.0, oracle.drainUpTo(0.0).get(0).relativeHeightM(), 0.0);
        assertEquals(5.0, oracle.drainUpTo(1.0).get(0).relativeHeightM(), 0.0);
    }

    // ================================================================ causal replay

    @Test
    @DisplayName("a replayed stream never hands out a sample from the frame's future")
    void theReplayIsCausal(@TempDir Path dir) throws IOException {
        Path csv = dir.resolve("h.csv");
        Files.write(csv, List.of("timestamp_s,baro_relative_m",
                                 "0.0,0.0", "1.0,1.0", "2.0,2.0", "3.0,3.0"), StandardCharsets.UTF_8);
        HeightSampleStream s = HeightSampleStream.fromCsv(csv, "timestamp_s", "baro_relative_m", 0.0, 1);
        assertEquals(4, s.size());

        assertEquals(1, s.drainUpTo(0.5).size(), "only the 0.0 s sample is available at t=0.5");
        assertEquals(1, s.drainUpTo(1.0).size(), "the 1.0 s sample becomes available exactly at 1.0");
        assertEquals(0, s.drainUpTo(1.9).size(), "no new sample yet, and none is invented");
        assertEquals(2, s.drainUpTo(9.0).size(), "the remainder, each delivered exactly once");
        assertEquals(4, s.delivered());
        assertEquals(0, s.drainUpTo(9.0).size(), "a delivered sample is never handed out twice");
    }

    @Test
    @DisplayName("latency delays availability; decimation thins the stream")
    void latencyAndDecimation(@TempDir Path dir) throws IOException {
        Path csv = dir.resolve("h.csv");
        Files.write(csv, List.of("timestamp_s,baro_relative_m",
                                 "0.0,0.0", "1.0,1.0", "2.0,2.0", "3.0,3.0"), StandardCharsets.UTF_8);
        HeightSampleStream late = HeightSampleStream.fromCsv(csv, "timestamp_s", "baro_relative_m",
                0.5, 1);
        assertEquals(0, late.drainUpTo(0.4).size(), "a sample taken at 0.0 arrives at 0.5");
        assertEquals(1, late.drainUpTo(0.5).size());

        HeightSampleStream thin = HeightSampleStream.fromCsv(csv, "timestamp_s", "baro_relative_m",
                0.0, 2);
        assertEquals(2, thin.size());
    }

    @Test
    @DisplayName("a missing column names itself rather than failing obscurely")
    void aMissingColumnIsNamed(@TempDir Path dir) throws IOException {
        Path csv = dir.resolve("h.csv");
        Files.write(csv, List.of("timestamp_s,other", "0.0,1.0"), StandardCharsets.UTF_8);
        IOException e = assertThrows(IOException.class,
                () -> HeightSampleStream.fromCsv(csv, "timestamp_s", "baro_relative_m", 0.0, 1));
        assertTrue(e.getMessage().contains("baro_relative_m"));
    }
}
