package org.boofcv.evaluation;

import boofcv.struct.image.GrayF32;
import georegression.struct.homography.Homography2D_F64;
import org.boofcv.relocalization.AlignmentSidecarWriter;
import org.boofcv.stitching.MotionModelStitchingEstimator;
import org.boofcv.stitching.ScriptedStitchingHarness;
import org.boofcv.stitching.ScriptedStitchingHarness.ScriptedStitching;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * <b>Deterministic hard-loss injection on the capture path.</b>
 *
 * <p>Natural hard loss is rare and cannot be scheduled by hand in the simulator, yet the whole
 * downstream recovery contract depends on it. A {@code synthetic_hard_loss} block declares the
 * frames; the estimator's own hard-loss branch is taken on each, so the semantics are not a
 * re-implementation of the real thing but literally it. What must never happen is the two becoming
 * indistinguishable afterwards, so both the run manifest and the sidecar label every loss.
 *
 * <p>Evidence tier: T1 (synthetic). The estimator-level equivalence of a forced and an observed
 * loss is proved by {@code RecenterVersusRestartTest}; this class is about the runner, the config
 * and the record.
 */
public class SyntheticHardLossTest {

    private static final int FRAME = ScriptedStitchingHarness.FRAME;
    private static final int FRAMES = 8;

    /** Eight frames of steady 10 px motion, with no natural failure anywhere. */
    private static FrameSource frames(ScriptedStitching stitch) {
        return new FrameSource() {
            @Override public int frameCount() { return FRAMES; }
            @Override public TimestampedFrame frame(int i) {
                if (i > 0) {
                    stitch.motion.advance(ScriptedStitchingHarness.translation(10.0));
                }
                return new TimestampedFrame(i, i * 0.1, new GrayF32(FRAME, FRAME));
            }
        };
    }

    /** Runs the capture path with the alignment sidecar on, injecting losses on {@code injectAt}. */
    private static VoRunner run(Path runDir, int... injectAt) throws IOException {
        Files.createDirectories(runDir);
        StringBuilder height = new StringBuilder("timestamp_s,baro_relative_m\n");
        StringBuilder heading = new StringBuilder("timestamp_s,heading_deg\n");
        for (int i = 0; i < FRAMES; i++) {
            height.append(String.format(java.util.Locale.ROOT, "%.3f,0.0%n", i * 0.1));
            heading.append(String.format(java.util.Locale.ROOT, "%.3f,90.0%n", i * 0.1));
        }
        Path heightCsv = runDir.resolve("height.csv");
        Path headingCsv = runDir.resolve("heading.csv");
        Files.writeString(heightCsv, height.toString(), StandardCharsets.UTF_8);
        Files.writeString(headingCsv, heading.toString(), StandardCharsets.UTF_8);

        ScriptedStitching stitch = new ScriptedStitching(0);
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> estimator =
                ScriptedStitchingHarness.headingEstimator(stitch,
                        org.boofcv.stitching.metric.HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null);
        VoRunner runner = new VoRunner(estimator, frames(stitch));
        runner.setMetricTrackCsv(runDir.resolve("metric_track.csv"),
                HeightSampleStream.fromCsv(heightCsv, "timestamp_s", "baro_relative_m", 0.0, 1, false));
        runner.setHeadingStream(HeadingSampleStream.fromCsv(headingCsv, "timestamp_s", "heading_deg", 0.0, 1));
        runner.setAlignmentSidecar(runDir);
        TreeSet<Integer> injected = new TreeSet<>();
        for (int f : injectAt) {
            injected.add(f);
        }
        runner.setSyntheticHardLossFrames(injected);

        RunManifest manifest = new RunManifest();
        manifest.runId = "synthetic-loss-test";
        manifest.datasetId = "d";
        manifest.datasetRevision = "r";
        manifest.estimatorId = "stitching-vo";
        manifest.estimatorVersion = "test";
        manifest.evaluatorCaptureVersion = "test";
        manifest.estimatorConfig = Map.of();
        manifest.environment = new Environment("h", "os", "cpu", "19", 1024, false);
        manifest.runTimestamp = "2026-09-08T00:00:00Z";
        try (RunRecordWriter writer = new RunRecordWriter(runDir, manifest, FRAMES)) {
            runner.run(writer);
        }
        return runner;
    }

    private static int col(List<String> lines, String name) {
        String[] header = lines.get(0).split(",", -1);
        for (int i = 0; i < header.length; i++) {
            if (header[i].equals(name)) return i;
        }
        throw new AssertionError("no column " + name);
    }

    // ================================================================ the record

    @Test
    @DisplayName("an injected loss is a real hard loss in the record, and is labelled synthetic")
    void anInjectedLossIsARealLossAndIsLabelled(@TempDir Path tmp) throws IOException {
        VoRunner runner = run(tmp.resolve("run"), 3);
        Path runDir = tmp.resolve("run");

        assertEquals(List.of(3), runner.syntheticHardLossesFired());
        assertEquals(List.of(), runner.naturalHardLosses(),
                "the stitcher never failed on its own — nothing here is an observed failure");

        // frames.csv (the shared run-record contract) reports it as a restart, exactly as a real one.
        List<String> frames = Files.readAllLines(runDir.resolve("frames.csv"));
        String[] row3 = frames.get(4).split(",", -1);
        assertEquals("restart", row3[col(frames, "event")]);
        assertEquals("false", row3[col(frames, "success")]);

        // The sidecar carries the label, and it is the ONLY place the two are distinguishable.
        List<String> events = Files.readAllLines(runDir.resolve(AlignmentSidecarWriter.EVENTS_FILE));
        int kind = col(events, "kind");
        int lossSource = col(events, "loss_source");
        int frameIdx = col(events, "frame_index");
        int found = 0;
        for (String line : events.subList(1, events.size())) {
            String[] r = line.split(",", -1);
            if ("hard_loss".equals(r[kind])) {
                found++;
                assertEquals("3", r[frameIdx]);
                assertEquals(AlignmentSidecarWriter.LOSS_SYNTHETIC, r[lossSource]);
            }
        }
        assertEquals(1, found, "exactly one hard loss");

        // ... and the downstream contract really did fire: segment 1, position UNKNOWN, heading kept.
        List<String> af = Files.readAllLines(runDir.resolve(AlignmentSidecarWriter.FRAMES_FILE));
        String[] at3 = af.get(4).split(",", -1);
        assertEquals("1", at3[col(af, "segment_id")], "the segment terminated");
        assertEquals("false", at3[col(af, "global_position_valid")], "position UNKNOWN");
        assertEquals("true", at3[col(af, "heading_valid")], "heading survives the translation gap");
        assertEquals("false", at3[col(af, "global_pose_valid")]);
        assertEquals("hard_loss", at3[col(af, "event")]);
        // The new segment starts at its own origin, and nothing bridged the gap.
        assertEquals(0.0, Double.parseDouble(at3[col(af, "local_east_m")]), 1e-12);
        String[] at4 = af.get(5).split(",", -1);
        assertEquals("false", at4[col(af, "global_position_valid")],
                "and healthy VO alone does not restore it");
    }

    @Test
    @DisplayName("a natural loss in the same run is labelled natural, and the two are counted apart")
    void naturalAndSyntheticLossesAreCountedApart(@TempDir Path tmp) throws IOException {
        Path runDir = tmp.resolve("run");
        Files.createDirectories(runDir);
        StringBuilder height = new StringBuilder("timestamp_s,baro_relative_m\n");
        StringBuilder heading = new StringBuilder("timestamp_s,heading_deg\n");
        for (int i = 0; i < FRAMES; i++) {
            height.append(String.format(java.util.Locale.ROOT, "%.3f,0.0%n", i * 0.1));
            heading.append(String.format(java.util.Locale.ROOT, "%.3f,90.0%n", i * 0.1));
        }
        Path heightCsv = runDir.resolve("height.csv");
        Path headingCsv = runDir.resolve("heading.csv");
        Files.writeString(heightCsv, height.toString(), StandardCharsets.UTF_8);
        Files.writeString(headingCsv, heading.toString(), StandardCharsets.UTF_8);

        ScriptedStitching stitch = new ScriptedStitching(0);
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> estimator =
                ScriptedStitchingHarness.headingEstimator(stitch,
                        org.boofcv.stitching.metric.HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null);
        // Frame 5 fails on its own; frame 3 is forced.
        FrameSource source = new FrameSource() {
            @Override public int frameCount() { return FRAMES; }
            @Override public TimestampedFrame frame(int i) {
                stitch.forceFailure = i == 5;
                if (i > 0) {
                    stitch.motion.advance(ScriptedStitchingHarness.translation(10.0));
                }
                return new TimestampedFrame(i, i * 0.1, new GrayF32(FRAME, FRAME));
            }
        };
        VoRunner runner = new VoRunner(estimator, source);
        runner.setMetricTrackCsv(runDir.resolve("metric_track.csv"),
                HeightSampleStream.fromCsv(heightCsv, "timestamp_s", "baro_relative_m", 0.0, 1, false));
        runner.setHeadingStream(HeadingSampleStream.fromCsv(headingCsv, "timestamp_s", "heading_deg", 0.0, 1));
        runner.setAlignmentSidecar(runDir);
        runner.setSyntheticHardLossFrames(new TreeSet<>(List.of(3)));

        RunManifest manifest = new RunManifest();
        manifest.runId = "mixed";
        manifest.datasetId = "d";
        manifest.datasetRevision = "r";
        manifest.estimatorId = "stitching-vo";
        manifest.estimatorVersion = "test";
        manifest.evaluatorCaptureVersion = "test";
        manifest.estimatorConfig = Map.of();
        manifest.environment = new Environment("h", "os", "cpu", "19", 1024, false);
        manifest.runTimestamp = "2026-09-08T00:00:00Z";
        try (RunRecordWriter writer = new RunRecordWriter(runDir, manifest, FRAMES)) {
            runner.run(writer);
        }

        assertEquals(List.of(3), runner.syntheticHardLossesFired());
        assertEquals(List.of(5), runner.naturalHardLosses(), "the observed failure is reported as observed");

        List<String> events = Files.readAllLines(runDir.resolve(AlignmentSidecarWriter.EVENTS_FILE));
        int kind = col(events, "kind");
        int lossSource = col(events, "loss_source");
        int frameIdx = col(events, "frame_index");
        for (String line : events.subList(1, events.size())) {
            String[] r = line.split(",", -1);
            if (!"hard_loss".equals(r[kind])) {
                continue;
            }
            String expected = "3".equals(r[frameIdx])
                    ? AlignmentSidecarWriter.LOSS_SYNTHETIC : AlignmentSidecarWriter.LOSS_NATURAL;
            assertEquals(expected, r[lossSource], "frame " + r[frameIdx]);
        }
    }

    @Test
    @DisplayName("multiple injections, and a run with none, both behave")
    void multipleInjectionsAndTheDefaultOfNone(@TempDir Path tmp) throws IOException {
        VoRunner two = run(tmp.resolve("two"), 2, 5);
        assertEquals(List.of(2, 5), two.syntheticHardLossesFired());
        assertEquals(List.of(), two.naturalHardLosses());
        List<String> af = Files.readAllLines(tmp.resolve("two").resolve(AlignmentSidecarWriter.FRAMES_FILE));
        assertEquals("2", af.get(FRAMES).split(",", -1)[col(af, "segment_id")],
                "two losses, two segment increments");

        VoRunner none = run(tmp.resolve("none"));
        assertEquals(List.of(), none.syntheticHardLossesFired(), "off unless declared");
        assertEquals(List.of(), none.naturalHardLosses());
        List<String> clean = Files.readAllLines(tmp.resolve("none").resolve(AlignmentSidecarWriter.FRAMES_FILE));
        assertEquals("0", clean.get(FRAMES).split(",", -1)[col(clean, "segment_id")],
                "an ordinary run is one uninterrupted segment");
        assertEquals(List.of(), none.recenters(), "and this schedule re-origins never");
    }

    // ================================================================ the config guard

    @Test
    @DisplayName("the config refuses anything it would have to guess, and demands an explicit acknowledgement")
    void theConfigBlockIsExplicitOrRefused() {
        VoRunnerConfig.SyntheticHardLossSettings s = new VoRunnerConfig.SyntheticHardLossSettings();
        s.frames = new int[]{3};
        assertTrue(assertThrows(IllegalArgumentException.class, () -> s.validated(10))
                        .getMessage().contains("acknowledge_synthetic"),
                "a frame list alone arms nothing: the departure from what was observed is spelled out");

        s.acknowledgeSynthetic = true;
        assertEquals(new TreeSet<>(List.of(3)), s.validated(10));

        s.frames = new int[]{0};
        assertTrue(assertThrows(IllegalArgumentException.class, () -> s.validated(10))
                        .getMessage().contains("frame 0"),
                "frame 0 integrates nothing, so it has no segment to terminate");

        s.frames = new int[]{99};
        assertThrows(IllegalArgumentException.class, () -> s.validated(10), "beyond the dataset");

        s.frames = new int[]{3, 3};
        assertTrue(assertThrows(IllegalArgumentException.class, () -> s.validated(10))
                .getMessage().contains("repeats frame"));

        s.frames = new int[]{};
        assertThrows(IllegalArgumentException.class, () -> s.validated(10));

        // A config that says nothing arms nothing — the default for every existing run config.
        assertFalse(new VoRunnerConfig().syntheticHardLoss != null);
    }
}
