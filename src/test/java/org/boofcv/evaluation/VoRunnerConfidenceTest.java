package org.boofcv.evaluation;

import boofcv.struct.image.GrayF32;
import org.boofcv.confidence.CalibrationConfig;
import org.boofcv.confidence.ConfidenceScorer;
import org.boofcv.confidence.SignalExtractor;
import org.boofcv.stitching.InstrumentedStitching;
import org.boofcv.stitching.StitchingEstimator;
import org.boofcv.stitching.StitchingFactory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * <b>T027 — the FR-013 bit-identity test.</b> A run with confidence capture enabled and one with
 * it disabled must produce identical poses, success flags, events, homographies and counts — exact
 * string equality on every persisted field except {@code process_time_ns}, which is wall-clock
 * timing and the one column that legitimately differs between two runs of the same computation.
 *
 * <p>Real KLT + RANSAC over synthetic textured imagery on a drifting crop path (the
 * {@code StitchingFactoryInstrumentationEquivalenceTest} sequence, which is asserted there to
 * exercise keyframe respawn and recentering rather than only steady state). The capture-off arm is
 * built by {@code buildGray()}; the capture-on arm by {@code buildGrayInstrumented()} plus the full
 * confidence pipeline — extractor, scorer, v1.1 writer — so the comparison covers everything the
 * capture path adds.
 *
 * <p>Evidence tier: <b>T1 (synthetic)</b>. This validates that observation does not perturb the
 * estimator; it says nothing about what the signals mean on real imagery.
 */
public class VoRunnerConfidenceTest {

    private static final int WORLD_SIZE = 512;
    private static final int FRAME_SIZE = 320;
    private static final int FRAME_COUNT = 24;

    private static GrayF32 syntheticTexturedWorld() {
        GrayF32 world = new GrayF32(WORLD_SIZE, WORLD_SIZE);
        Random random = new Random(4242);
        for (int y = 0; y < WORLD_SIZE; y++) {
            for (int x = 0; x < WORLD_SIZE; x++) {
                world.set(x, y, random.nextInt(256));
            }
        }
        return world;
    }

    private static GrayF32 frameAt(GrayF32 world, int index) {
        int offsetX = 4 + index * 8;
        int offsetY = 4 + index * 5;
        GrayF32 frame = new GrayF32(FRAME_SIZE, FRAME_SIZE);
        for (int y = 0; y < FRAME_SIZE; y++) {
            for (int x = 0; x < FRAME_SIZE; x++) {
                frame.set(x, y, world.get(x + offsetX, y + offsetY));
            }
        }
        return frame;
    }

    /** An in-memory frame source over the synthetic sequence. */
    private static FrameSource frames(GrayF32 world) {
        return new FrameSource() {
            @Override
            public int frameCount() {
                return FRAME_COUNT;
            }

            @Override
            public TimestampedFrame frame(int index) {
                return new TimestampedFrame(index, index * 0.1, frameAt(world, index));
            }
        };
    }

    /** A calibration whose binding matches this synthetic capture, W/m at the frozen values. */
    private static final String TEST_CALIBRATION = """
            {
              "schema_version": "2.0.0",
              "calibration_id": "t027-unvalidated-v1",
              "validated": false,
              "validated_by": null,
              "probability_semantics": false,
              "required_signals": ["track_count", "inlier_count", "inlier_ratio"],
              "configuration_binding": {"motion_model": "homography"},
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

    private static RunManifest manifest(String runId) {
        RunManifest m = new RunManifest();
        m.runId = runId;
        m.datasetId = "t027-synthetic";
        m.datasetRevision = "v1";
        m.estimatorId = "stitching-vo";
        m.estimatorVersion = "test";
        m.evaluatorCaptureVersion = "test";
        m.runTimestamp = "1970-01-01T00:00:00Z";
        return m;
    }

    private static StitchingEstimator<GrayF32> configure(StitchingEstimator<GrayF32> e) {
        e.setShrinkScale(0.5);
        e.setMinDistanceFromBorder(10);
        return e;
    }

    @Test
    public void captureOnAndCaptureOffProduceBitIdenticalEstimation(@TempDir Path tmp) throws IOException {
        GrayF32 world = syntheticTexturedWorld();

        // Arm 1: capture off, plain factory build, v1.0.0 record.
        Path offDir = tmp.resolve("off");
        StitchingEstimator<GrayF32> plain = configure(
                new StitchingEstimator<>(StitchingFactory.builder().buildGray()));
        try (RunRecordWriter writer = new RunRecordWriter(offDir, manifest("t027-off"), FRAME_COUNT)) {
            new VoRunner(plain, frames(world)).run(writer);
        }

        // Arm 2: capture on, instrumented build, full confidence pipeline, v1.1.0 record.
        Path onDir = tmp.resolve("on");
        CalibrationConfig calibration = CalibrationConfig.parse(TEST_CALIBRATION);
        InstrumentedStitching<GrayF32> instrumented = StitchingFactory.builder().buildGrayInstrumented();
        StitchingEstimator<GrayF32> observed = configure(new StitchingEstimator<>(instrumented.stitch()));
        VoRunner runner = new VoRunner(observed, frames(world));
        runner.setConfidence(instrumented.diagnostics(),
                new SignalExtractor(calibration.windowW(), calibration.warmupM()),
                new ConfidenceScorer(calibration));
        try (RunRecordWriter writer = new RunRecordWriter(onDir, manifest("t027-on"), FRAME_COUNT, true)) {
            runner.run(writer);
        }

        List<String> off = Files.readAllLines(offDir.resolve("frames.csv"));
        List<String> on = Files.readAllLines(onDir.resolve("frames.csv"));
        assertEquals(off.size(), on.size(), "row count");

        // Header: the capture-on record extends the v1.0.0 header, never alters it.
        String[] offHeader = off.get(0).split(",", -1);
        String[] onHeader = on.get(0).split(",", -1);
        assertEquals(21, offHeader.length);
        assertEquals(37, onHeader.length, "v1.1.0 carries sixteen appended columns");
        for (int c = 0; c < 21; c++) {
            assertEquals(offHeader[c], onHeader[c], "header column " + c);
        }

        // Every data field except process_time_ns (column 20): EXACT string equality, no tolerance.
        List<String> diffs = new ArrayList<>();
        for (int r = 1; r < off.size(); r++) {
            String[] a = off.get(r).split(",", -1);
            String[] b = on.get(r).split(",", -1);
            for (int c = 0; c < 21; c++) {
                if (c == 20) {
                    continue;   // wall-clock timing; both must still be present and non-negative
                }
                if (!a[c].equals(b[c])) {
                    diffs.add("row " + r + " col " + offHeader[c] + ": '" + a[c] + "' vs '" + b[c] + "'");
                }
            }
        }
        assertTrue(diffs.isEmpty(), "FR-013 bit-identity violated:\n" + String.join("\n", diffs));

        // The capture-on record's own obligations, spot-checked here (the full contract check is
        // RunRecordContractTest's): every row carries a verdict; frame 0 is the rejected
        // not-established frame; a restart row is not_produced.
        for (int r = 1; r < on.size(); r++) {
            String[] b = on.get(r).split(",", -1);
            assertFalse(b[33].isEmpty(), "confidence_outcome missing at row " + r);
            assertFalse(b[34].isEmpty(), "confidence_reason missing at row " + r);
            boolean success = Boolean.parseBoolean(b[6]);
            if (!success) {
                assertEquals("not_produced", b[33], "failed frame must be not_produced at row " + r);
            }
        }
        String[] first = on.get(1).split(",", -1);
        assertEquals("rejected", first[33], "frame 0 outcome");
        assertEquals("not_established", first[34], "frame 0 reason");
        assertTrue(first[35].isEmpty(), "frame 0 score must be empty, not zero");
    }
}
