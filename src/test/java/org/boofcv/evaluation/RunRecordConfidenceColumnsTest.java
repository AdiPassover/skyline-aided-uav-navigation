package org.boofcv.evaluation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The writer-side v1.1.0 obligations (contracts/run-record-v1.1.md): the sixteen appended columns,
 * absence encoded as emptiness and never zero, and the row invariants enforced at the producer so
 * a malformed record can never be written.
 *
 * <p>Evidence tier: <b>T1 (analytical)</b>.
 */
public class RunRecordConfidenceColumnsTest {

    private static RunManifest manifest() {
        RunManifest m = new RunManifest();
        m.runId = "t";
        m.datasetId = "d";
        m.datasetRevision = "v1";
        m.estimatorId = "e";
        m.estimatorVersion = "v";
        m.evaluatorCaptureVersion = "v";
        m.runTimestamp = "1970-01-01T00:00:00Z";
        return m;
    }

    private static FrameEstimateRow row(int index, boolean success, String event,
                                        FrameEstimateRow.Confidence c) {
        return new FrameEstimateRow(index, index * 0.1, 1.0, 2.0, null, 3.0, success, event, 0,
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 100, 90, 1000L, c);
    }

    private static FrameEstimateRow.Confidence usable() {
        return new FrameEstimateRow.Confidence(90, 0.8, 0.894, 0.7, 2.9, 3.0, 0.5, null,
                1.02, 4.5, 0.001, 0.0004, "usable", "ok", 0.9, 500L);
    }

    @Test
    public void confidenceColumnsRoundTripWithAbsenceAsEmptiness(@TempDir Path tmp) throws IOException {
        Path dir = tmp.resolve("run");
        try (RunRecordWriter w = new RunRecordWriter(dir, manifest(), 2, true)) {
            // Frame 0: the not-established frame — every windowed/residual signal absent, no score.
            w.writeFrame(row(0, true, "init", new FrameEstimateRow.Confidence(
                    null, null, null, null, null, null, null, null, null, null, null, null,
                    "rejected", "not_established", null, 400L)));
            w.writeFrame(row(1, true, "none", usable()));
        }

        List<String> lines = Files.readAllLines(dir.resolve("frames.csv"));
        String[] header = lines.get(0).split(",", -1);
        assertEquals(37, header.length);
        assertEquals("residual_inlier_count", header[21]);
        assertEquals("relative_support", header[29]);
        assertEquals("inc_flow_px", header[30]);
        assertEquals("inc_log_scale", header[31]);
        assertEquals("inc_log_scale_dispersion", header[32]);
        assertEquals("confidence_outcome", header[33]);
        assertEquals("confidence_time_ns", header[36]);

        String[] init = lines.get(1).split(",", -1);
        assertEquals("", init[21], "absent count must be empty, never zero");
        assertEquals("", init[35], "withheld score must be empty, never zero");
        assertEquals("rejected", init[33]);

        String[] ok = lines.get(2).split(",", -1);
        assertEquals("90", ok[21]);
        assertEquals("0.9", ok[35]);
        assertEquals("usable", ok[33]);
    }

    @Test
    public void aConfidenceRunRefusesARowWithoutAVerdict(@TempDir Path tmp) throws IOException {
        try (RunRecordWriter w = new RunRecordWriter(tmp.resolve("r"), manifest(), 1, true)) {
            assertThrows(IllegalArgumentException.class,
                    () -> w.writeFrame(row(0, true, "init", null)),
                    "a missing verdict must be indistinguishable from a lost record, so it is refused");
            w.markIncomplete();
        }
    }

    @Test
    public void aPlainRunRefusesAConfidenceRow(@TempDir Path tmp) throws IOException {
        try (RunRecordWriter w = new RunRecordWriter(tmp.resolve("r"), manifest(), 1)) {
            assertThrows(IllegalArgumentException.class,
                    () -> w.writeFrame(row(0, true, "init", usable())));
            w.markIncomplete();
        }
    }

    @Test
    public void theScoreNullityInvariantIsEnforcedAtTheProducer(@TempDir Path tmp) throws IOException {
        try (RunRecordWriter w = new RunRecordWriter(tmp.resolve("r"), manifest(), 4, true)) {
            // A rejected frame with a score: refused — a number for a refused estimate.
            assertThrows(IllegalArgumentException.class, () -> w.writeFrame(row(0, true, "none",
                    new FrameEstimateRow.Confidence(null, null, null, null, null, null, null, null,
                            null, null, null, null, "rejected", "signals_unavailable", 0.4, 1L))));
            // A usable frame without a score: refused — a verdict with no basis on record.
            assertThrows(IllegalArgumentException.class, () -> w.writeFrame(row(0, true, "none",
                    new FrameEstimateRow.Confidence(null, null, null, null, null, null, null, null,
                            null, null, null, null, "usable", "ok", null, 1L))));
            // not_produced on a successful frame: refused — the contract pins it to success=false.
            assertThrows(IllegalArgumentException.class, () -> w.writeFrame(row(0, true, "none",
                    new FrameEstimateRow.Confidence(null, null, null, null, null, null, null, null,
                            null, null, null, null, "not_produced", "estimator_failed", null, 1L))));
            // Residual statistics without a positive match-set count: refused.
            assertThrows(IllegalArgumentException.class, () -> w.writeFrame(row(0, true, "none",
                    new FrameEstimateRow.Confidence(0, 0.5, null, null, null, null, null, null,
                            null, null, null, null, "usable", "ok", 0.9, 1L))));
            w.markIncomplete();
        }
    }

    @Test
    public void theManifestCarriesTheConfidenceSchemaVersion(@TempDir Path tmp) throws IOException {
        Path dir = tmp.resolve("run");
        RunManifest m = manifest();
        m.schemaVersion = RunManifest.SCHEMA_VERSION_CONFIDENCE;
        m.confidence = new java.util.LinkedHashMap<>(java.util.Map.of(
                "calibration_id", "t-unvalidated-v1", "calibration_validated", false));
        try (RunRecordWriter w = new RunRecordWriter(dir, m, 1, true)) {
            w.writeFrame(row(0, true, "init", new FrameEstimateRow.Confidence(
                    null, null, null, null, null, null, null, null, null, null, null, null,
                    "rejected", "not_established", null, 1L)));
        }
        String json = Files.readString(dir.resolve("manifest.json"));
        assertTrue(json.contains("\"schema_version\" : \"1.1.0\""), json);
        assertTrue(json.contains("t-unvalidated-v1"), json);
    }
}
