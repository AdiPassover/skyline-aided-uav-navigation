package org.boofcv.evaluation;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class RunRecordWriterTest {

    private static RunManifest baseManifest() {
        RunManifest m = new RunManifest();
        m.runId = "test-run-001";
        m.datasetId = "test-dataset";
        m.datasetRevision = "rev1";
        m.estimatorId = "stitching-vo";
        m.estimatorVersion = "abc123";
        m.estimatorConfig = Map.of("shrinkScale", 0.5, "maxFeatures", 300);
        m.evaluatorCaptureVersion = "abc123";
        m.environment = new Environment("host", "os", "cpu", "19.0.2", 4096, false);
        m.runTimestamp = "2026-08-12T00:00:00Z";
        return m;
    }

    @Test
    public void testRoundTripKnownValues(@TempDir Path tempDir) throws IOException {
        Path runDir = tempDir.resolve("run1");
        try (RunRecordWriter writer = new RunRecordWriter(runDir, baseManifest(), 2)) {
            FrameEstimateRow row0 = new FrameEstimateRow(
                    0, 0.0, -112.1217, -32.0736, null, 200.0, true, "init", 0,
                    0.5, 0.0, 160.0, 0.0, 0.5, 128.0, 0.0, 0.0, 1.0,
                    285, 268, 12000000L);
            FrameEstimateRow row1 = new FrameEstimateRow(
                    1, 0.1, -77.7891, -38.1274, null, 190.0, true, "none", 0,
                    0.5, 0.0, 159.1, 0.0, 0.5, 127.9, 0.0, 0.0, 1.0,
                    281, 262, 12100000L);
            writer.writeFrame(row0);
            writer.writeFrame(row1);
        }

        List<String> lines = Files.readAllLines(runDir.resolve("frames.csv"));
        assertEquals(3, lines.size()); // header + 2 rows
        assertEquals(
                "frame_index,timestamp_s,est_x,est_y,est_z,est_yaw_deg,success,event,reference_id,"
                        + "h00,h01,h02,h10,h11,h12,h20,h21,h22,track_count,inlier_count,process_time_ns",
                lines.get(0));

        String[] cols = lines.get(1).split(",", -1);
        assertEquals("0", cols[0]);
        assertEquals("0.0", cols[1]);
        assertEquals("-112.1217", cols[2]);
        assertEquals("-32.0736", cols[3]);
        assertEquals("", cols[4]); // est_z always empty
        assertEquals("200.0", cols[5]);
        assertEquals("true", cols[6]);
        assertEquals("init", cols[7]);
        assertEquals("0", cols[8]);
        assertEquals("0.5", cols[9]);
        assertEquals("285", cols[18]);
        assertEquals("268", cols[19]);
        assertEquals("12000000", cols[20]);

        Map<String, Object> manifest = new ObjectMapper().readValue(runDir.resolve("manifest.json").toFile(), Map.class);
        assertEquals("1.0.0", manifest.get("schema_version"));
        assertEquals("test-run-001", manifest.get("run_id"));
        assertEquals(2, manifest.get("frame_count"));
        assertEquals(2, manifest.get("processed_count"));
        assertEquals(true, manifest.get("completed"));
        assertEquals(Boolean.FALSE, ((Map<?, ?>) manifest.get("environment")).get("is_target_hardware"));
    }

    @Test
    public void testEmptyVersusZeroPreserved(@TempDir Path tempDir) throws IOException {
        Path runDir = tempDir.resolve("run2");
        try (RunRecordWriter writer = new RunRecordWriter(runDir, baseManifest(), 2)) {
            // Frame with zero measured tracks (a real measurement of 0).
            writer.writeFrame(new FrameEstimateRow(
                    0, 0.0, 0.0, 0.0, null, 0.0, false, "init", 0,
                    0.5, 0.0, 100.0, 0.0, 0.5, 100.0, 0.0, 0.0, 1.0,
                    0, 0, 1000L));
            // Frame with diagnostics unavailable (AccessPointTracks cast failed).
            writer.writeFrame(new FrameEstimateRow(
                    1, 0.1, 1.0, 1.0, null, 0.0, true, "none", 0,
                    null, null, null, null, null, null, null, null, null,
                    null, null, 1000L));
        }

        List<String> lines = Files.readAllLines(runDir.resolve("frames.csv"));
        String[] measured = lines.get(1).split(",", -1);
        String[] unavailable = lines.get(2).split(",", -1);

        assertEquals("0", measured[18]); // track_count = 0, a real measurement
        assertEquals("0", measured[19]);
        assertEquals("", unavailable[9]); // h00 empty, not 0.0
        assertEquals("", unavailable[18]); // track_count empty, not 0
        assertEquals("", unavailable[19]);
        assertNotEquals(measured[18], unavailable[18]);
    }

    @Test
    public void testNaNRejected(@TempDir Path tempDir) throws IOException {
        Path runDir = tempDir.resolve("run3");
        try (RunRecordWriter writer = new RunRecordWriter(runDir, baseManifest(), 1)) {
            FrameEstimateRow badRow = new FrameEstimateRow(
                    0, 0.0, Double.NaN, 0.0, null, 0.0, true, "init", 0,
                    0.5, 0.0, 100.0, 0.0, 0.5, 100.0, 0.0, 0.0, 1.0,
                    1, 1, 1000L);
            assertThrows(IllegalArgumentException.class, () -> writer.writeFrame(badRow));
        }
    }

    @Test
    public void testInfinityRejectedInOptionalField(@TempDir Path tempDir) throws IOException {
        Path runDir = tempDir.resolve("run4");
        try (RunRecordWriter writer = new RunRecordWriter(runDir, baseManifest(), 1)) {
            FrameEstimateRow badRow = new FrameEstimateRow(
                    0, 0.0, 0.0, 0.0, null, 0.0, true, "init", 0,
                    Double.POSITIVE_INFINITY, 0.0, 100.0, 0.0, 0.5, 100.0, 0.0, 0.0, 1.0,
                    1, 1, 1000L);
            assertThrows(IllegalArgumentException.class, () -> writer.writeFrame(badRow));
        }
    }

    @Test
    public void testManifestFieldsPopulated(@TempDir Path tempDir) throws IOException {
        Path runDir = tempDir.resolve("run5");
        RunRecordWriter writer = new RunRecordWriter(runDir, baseManifest(), 3);
        writer.writeFrame(new FrameEstimateRow(
                0, 0.0, 0.0, 0.0, null, 0.0, true, "init", 0,
                0.5, 0.0, 100.0, 0.0, 0.5, 100.0, 0.0, 0.0, 1.0,
                1, 1, 1000L));
        writer.markIncomplete();
        writer.close();

        Map<String, Object> manifest = new ObjectMapper().readValue(runDir.resolve("manifest.json").toFile(), Map.class);
        assertEquals(3, manifest.get("frame_count"));
        assertEquals(1, manifest.get("processed_count")); // only 1 frame actually written
        assertEquals(false, manifest.get("completed"));
        assertEquals("stitching-vo", manifest.get("estimator_id"));
        assertEquals(0.5, ((Map<?, ?>) manifest.get("estimator_config")).get("shrinkScale"));
    }

    @Test
    public void testStrictlyIncreasingFrameIndexEnforced(@TempDir Path tempDir) throws IOException {
        Path runDir = tempDir.resolve("run6");
        try (RunRecordWriter writer = new RunRecordWriter(runDir, baseManifest(), 2)) {
            writer.writeFrame(new FrameEstimateRow(
                    1, 0.0, 0.0, 0.0, null, 0.0, true, "init", 0,
                    0.5, 0.0, 100.0, 0.0, 0.5, 100.0, 0.0, 0.0, 1.0,
                    1, 1, 1000L));
            FrameEstimateRow outOfOrder = new FrameEstimateRow(
                    1, 0.1, 0.0, 0.0, null, 0.0, true, "none", 0,
                    0.5, 0.0, 100.0, 0.0, 0.5, 100.0, 0.0, 0.0, 1.0,
                    1, 1, 1000L);
            assertThrows(IllegalArgumentException.class, () -> writer.writeFrame(outOfOrder));
        }
    }
}
