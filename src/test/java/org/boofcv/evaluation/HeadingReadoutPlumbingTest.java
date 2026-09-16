package org.boofcv.evaluation;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.boofcv.stitching.metric.HeadingReadoutConfig;
import org.boofcv.stitching.metric.HeadingSemantics;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The evaluation-side plumbing of the authoritative heading channel: config parsing, the declared
 * semantics, the refusal of a missing mounting calibration, and the causal replay of a recorded
 * heading stream.
 *
 * <p>The theme is the same as the height side's: every way of getting a heading run wrong that this
 * code can detect, it detects — an undeclared mounting calibration, an airframe column labelled as a
 * camera one, a heading block with no metric block under it, or a stream replayed acausally.
 */
class HeadingReadoutPlumbingTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    // ================================================================ backward compatibility

    @Test
    @DisplayName("a config with no heading_readout block is unchanged in every respect")
    void absentBlockChangesNothing() throws IOException {
        VoRunnerConfig c = MAPPER.readValue(
                "{\"dataset_dir\":\"d\",\"navigation_source\":\"rigid_motion\"}", VoRunnerConfig.class);
        assertNull(c.headingReadout,
                "every pre-DEC-VO-010 config must parse to no heading readout, so the metric pose "
                + "keeps being rotated by the visual yaw exactly as DEC-VO-009 shipped it");
    }

    // ================================================================ declared semantics

    @Test
    @DisplayName("the production arm parses, and its mounting calibration travels with it")
    void productionArmParses() throws IOException {
        VoRunnerConfig c = MAPPER.readValue("""
                {"dataset_dir":"d","heading_readout":{
                   "enabled": true,
                   "heading_semantics": "fc_ahrs_body_compass",
                   "delta_mount_deg": 90.6,
                   "delta_mount_provenance": "one-time platform calibration",
                   "heading_csv": "datasets/x/attitude.csv",
                   "heading_column": "yaw_compass_deg"}}""", VoRunnerConfig.class);
        assertTrue(c.headingReadout.enabled);
        assertEquals(HeadingSemantics.FC_AHRS_BODY_COMPASS, c.headingReadout.semantics());
        assertEquals(90.6, c.headingReadout.toConfig().mountOffsetDeg(), 0.0);
        assertEquals("timestamp_s", c.headingReadout.headingTimeColumn, "sensible default");
        assertEquals(0.01, c.headingReadout.nominalSampleIntervalS, 0.0,
                "defaults to the 100 Hz rate the real channel actually runs at");
        assertEquals(HeadingReadoutConfig.DEFAULT_MAX_RATE_DEG_PER_S,
                c.headingReadout.maxRateDegPerS, 0.0);
    }

    @Test
    @DisplayName("the simulator arm parses and needs no calibration")
    void simulatorArmParses() throws IOException {
        VoRunnerConfig c = MAPPER.readValue("""
                {"dataset_dir":"d","heading_readout":{
                   "enabled": true,
                   "heading_semantics": "sim_nadir_camera_heading",
                   "heading_csv": "datasets/x/groundtruth.csv",
                   "heading_column": "heading_deg"}}""", VoRunnerConfig.class);
        assertEquals(HeadingSemantics.SIM_NADIR_CAMERA_HEADING, c.headingReadout.semantics());
        assertEquals(0.0, c.headingReadout.toConfig().mountOffsetDeg(), 0.0,
                "exactly zero, because that column is already the navigation camera's");
    }

    @Test
    @DisplayName("an airframe column with NO declared mounting calibration is refused, not defaulted")
    void productionArmRefusesAMissingCalibration() throws IOException {
        VoRunnerConfig c = MAPPER.readValue("""
                {"dataset_dir":"d","heading_readout":{
                   "enabled": true,
                   "heading_semantics": "fc_ahrs_body_compass",
                   "heading_csv": "a.csv", "heading_column": "yaw_compass_deg"}}""",
                VoRunnerConfig.class);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> c.headingReadout.toConfig());
        assertTrue(e.getMessage().contains("delta_mount_deg is REQUIRED"));
        // Defaulting it to zero would look entirely ordinary and would be EXP-VO-013 R6's ~66 deg
        // error, silently, on every frame.
        assertTrue(e.getMessage().contains("EXP-VO-013 R6"));
    }

    @Test
    @DisplayName("a mounting calibration on the simulator arm is refused rather than ignored")
    void simulatorArmRefusesACalibration() throws IOException {
        VoRunnerConfig c = MAPPER.readValue("""
                {"dataset_dir":"d","heading_readout":{
                   "enabled": true,
                   "heading_semantics": "sim_nadir_camera_heading",
                   "delta_mount_deg": 90.0,
                   "heading_csv": "a.csv", "heading_column": "heading_deg"}}""",
                VoRunnerConfig.class);
        assertThrows(IllegalArgumentException.class, () -> c.headingReadout.toConfig());
    }

    @Test
    @DisplayName("an unrecognised semantics is refused with both valid spellings named")
    void unknownSemanticsIsRefused() throws IOException {
        VoRunnerConfig c = MAPPER.readValue("""
                {"dataset_dir":"d","heading_readout":{
                   "enabled": true, "heading_semantics": "body_yaw_deg",
                   "heading_csv": "a.csv", "heading_column": "x"}}""", VoRunnerConfig.class);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> c.headingReadout.semantics());
        assertTrue(e.getMessage().contains("fc_ahrs_body_compass"));
        assertTrue(e.getMessage().contains("sim_nadir_camera_heading"));
    }

    @Test
    @DisplayName("an unknown key in the block is refused -- a typo must not silently disable a channel")
    void unknownKeysAreRefused() {
        assertThrows(IOException.class, () -> MAPPER.readValue("""
                {"dataset_dir":"d","heading_readout":{
                   "enabled": true, "heading_semantics": "sim_nadir_camera_heading",
                   "delta_mount_degrees": 90.0}}""", VoRunnerConfig.class));
    }

    // ================================================================ causal replay

    private static Path headingCsv(Path dir, String name, String body) throws IOException {
        Path p = dir.resolve(name);
        Files.writeString(p, body, StandardCharsets.UTF_8);
        return p;
    }

    @Test
    @DisplayName("the recording is replayed causally -- a frame can never reach a later sample")
    void replayIsCausal(@TempDir Path dir) throws IOException {
        Path csv = headingCsv(dir, "attitude.csv",
                "timestamp_s,yaw_compass_deg\n0.0,10.0\n0.1,11.0\n0.2,12.0\n0.3,13.0\n");
        HeadingSampleStream s = HeadingSampleStream.fromCsv(csv, "timestamp_s", "yaw_compass_deg",
                0.0, 1);
        assertEquals(4, s.size());
        List<HeadingSampleStream.Sample> due = s.drainUpTo(0.15);
        assertEquals(2, due.size(), "only the samples available by 0.15");
        assertEquals(10.0, due.get(0).headingDeg(), 0.0);
        assertEquals(11.0, due.get(1).headingDeg(), 0.0);
        assertEquals(0, s.drainUpTo(0.15).size(), "each sample is delivered exactly once");
        assertEquals(2, s.drainUpTo(10.0).size());
        assertEquals(4, s.delivered());
    }

    @Test
    @DisplayName("latency shifts availability, decimation thins the stream, and both are declared")
    void latencyAndDecimation(@TempDir Path dir) throws IOException {
        Path csv = headingCsv(dir, "a.csv",
                "timestamp_s,h\n0.0,1.0\n0.1,2.0\n0.2,3.0\n0.3,4.0\n");
        HeadingSampleStream lagged = HeadingSampleStream.fromCsv(csv, "timestamp_s", "h", 0.25, 1);
        assertEquals(0, lagged.drainUpTo(0.2).size(), "nothing has arrived yet at 0.2 s");
        assertEquals(1, lagged.drainUpTo(0.25).size());

        HeadingSampleStream thinned = HeadingSampleStream.fromCsv(csv, "timestamp_s", "h", 0.0, 2);
        assertEquals(2, thinned.size());
        assertEquals(1.0, thinned.drainUpTo(10.0).get(0).headingDeg(), 0.0);
    }

    @Test
    @DisplayName("an empty heading cell is an absent measurement, not a zero")
    void emptyCellsAreDroppedNotReadAsNorth(@TempDir Path dir) throws IOException {
        // groundtruth.csv's contract says heading_deg is "Empty when no heading source." Parsing
        // that as 0.0 would silently point the whole trajectory North.
        Path csv = headingCsv(dir, "gt.csv",
                "timestamp_s,heading_deg\n0.0,10.0\n0.1,\n0.2,12.0\n");
        HeadingSampleStream s = HeadingSampleStream.fromCsv(csv, "timestamp_s", "heading_deg",
                0.0, 1);
        assertEquals(2, s.size(), "the blank row is dropped so the channel's own staleness "
                + "policy describes the outage");
        assertEquals(12.0, s.drainUpTo(10.0).get(1).headingDeg(), 0.0);
    }

    @Test
    @DisplayName("a missing column names itself and the header, rather than failing on an index")
    void missingColumnIsNamed(@TempDir Path dir) throws IOException {
        Path csv = headingCsv(dir, "a.csv", "timestamp_s,yaw\n0.0,1.0\n");
        IOException e = assertThrows(IOException.class,
                () -> HeadingSampleStream.fromCsv(csv, "timestamp_s", "yaw_compass_deg", 0.0, 1));
        assertTrue(e.getMessage().contains("yaw_compass_deg"));
        assertTrue(e.getMessage().contains("timestamp_s,yaw"));
    }

    @Test
    @DisplayName("an out-of-order recording is refused rather than sorted behind the caller's back")
    void outOfOrderRecordingIsRefused(@TempDir Path dir) throws IOException {
        Path csv = headingCsv(dir, "a.csv", "timestamp_s,h\n0.0,1.0\n0.3,2.0\n0.1,3.0\n");
        assertThrows(IllegalArgumentException.class,
                () -> HeadingSampleStream.fromCsv(csv, "timestamp_s", "h", 0.0, 1));
    }

    // ================================================================ the real committed feeds

    @Test
    @DisplayName("the committed MARS-LVIG attitude feed loads and is ~100 Hz, in order, wrap-clean")
    void marsLvigAttitudeFeedIsUsable() throws IOException {
        Path csv = Path.of("datasets/hkairport01-a/attitude.csv");
        org.junit.jupiter.api.Assumptions.assumeTrue(Files.exists(csv),
                "attitude.csv is an ingest product; skip where the checkout has no imagery side");
        HeadingSampleStream s = HeadingSampleStream.fromCsv(csv, "timestamp_s", "yaw_compass_deg",
                0.0, 1);
        assertTrue(s.size() > 17000, "the 180 s window holds ~18 000 samples at ~100 Hz");
        List<HeadingSampleStream.Sample> all = s.drainUpTo(Double.MAX_VALUE);
        for (HeadingSampleStream.Sample sample : all) {
            assertTrue(sample.headingDeg() >= 0.0 && sample.headingDeg() < 360.0,
                    "the ingest publishes compass degrees in [0,360)");
        }
        double span = all.get(all.size() - 1).availabilityTimeS() - all.get(0).availabilityTimeS();
        assertEquals(100.0, (all.size() - 1) / span, 5.0, "roughly 100 Hz");
    }
}
