package org.boofcv.evaluation;

import boofcv.abst.sfm.d2.ImageMotion2D;
import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.struct.image.GrayF32;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.point.Point2D_F64;
import georegression.struct.shapes.Quadrilateral_F64;
import org.boofcv.relocalization.AlignmentSidecarWriter;
import org.boofcv.stitching.MotionModelStitchingEstimator;
import org.boofcv.stitching.MotionModelSupport;
import org.boofcv.stitching.metric.HeadingReadoutConfig;
import org.boofcv.stitching.metric.HeadingSemantics;
import org.boofcv.stitching.metric.MetricReadoutConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The opt-in alignment sidecar ({@code DEC-INT-001}, amended) on the capture path, driven by the
 * real estimator with the metric readout and an authoritative heading over a scripted
 * init / none / restart / none / none sequence: the restart must open VO segment 1 with the
 * persistent position absent and the heading still valid, the local positions must be metres
 * (0.1 m/px at h0 = 80 m, f = 800 px), {@code frames.csv} must be identical with the flag off or
 * on, and the layer must refuse an estimator without the metric+heading contract (tests A, B, E).
 */
public class VoRunnerAlignmentSidecarTest {

    private static final int FRAME = 320;

    private static final class ScriptedMotion implements ImageMotion2D<GrayF32, Homography2D_F64> {
        final Homography2D_F64 firstToCurrent = new Homography2D_F64();

        @Override public boolean process(GrayF32 input) { return true; }
        @Override public void reset() { firstToCurrent.reset(); }
        @Override public void setToFirst() { firstToCurrent.reset(); }
        @Override public long getFrameID() { return 0; }
        @Override public Homography2D_F64 getFirstToCurrent() { return firstToCurrent; }
        @Override public Class<Homography2D_F64> getTransformType() { return Homography2D_F64.class; }

        void advance(Homography2D_F64 increment) {
            Homography2D_F64 next = new Homography2D_F64();
            firstToCurrent.concat(increment, next);
            firstToCurrent.setTo(next);
        }
    }

    private static final class ScriptedStitching extends StitchingFromMotion2D<GrayF32, Homography2D_F64> {
        final ScriptedMotion motion = new ScriptedMotion();
        final GrayF32 stitched = new GrayF32(200, 200);
        boolean forceFailure = false;

        @SuppressWarnings("unchecked")
        ScriptedStitching() {
            super(mock(ImageMotion2D.class), mock(boofcv.alg.distort.ImageDistort.class),
                    mock(boofcv.alg.sfm.d2.StitchingTransform.class), 0.5);
        }

        @SuppressWarnings("unchecked")
        private static <I> I mock(Class<I> iface) {
            return (I) java.lang.reflect.Proxy.newProxyInstance(
                    iface.getClassLoader(), new Class<?>[]{iface},
                    (proxy, method, args) -> {
                        switch (method.getName()) {
                            case "equals": return proxy == args[0];
                            case "hashCode": return System.identityHashCode(proxy);
                            case "toString": return "Mock";
                            default: break;
                        }
                        Class<?> rt = method.getReturnType();
                        if (rt == boolean.class) return false;
                        if (rt == int.class) return 0;
                        if (rt == double.class) return 0.0;
                        if (rt == float.class) return 0.0f;
                        if (rt == long.class) return 0L;
                        if (rt.isAssignableFrom(Homography2D_F64.class)) return new Homography2D_F64();
                        return null;
                    });
        }

        @Override public ImageMotion2D<GrayF32, Homography2D_F64> getMotion() { return motion; }
        @Override public boolean process(GrayF32 image) { return !forceFailure; }
        @Override public void reset() { motion.reset(); }
        @Override public void configure(int w, int h, Homography2D_F64 worldToInit) { }
        @Override public Homography2D_F64 getWorldToCurr() { return new Homography2D_F64(); }
        @Override public Homography2D_F64 getWorldToCurr(Homography2D_F64 s) {
            if (s == null) s = new Homography2D_F64();
            s.reset();
            return s;
        }
        @Override public GrayF32 getStitchedImage() { return stitched; }
        @Override public void setOriginToCurrent() { motion.setToFirst(); }

        @Override
        public Quadrilateral_F64 getImageCorners(int w, int h, Quadrilateral_F64 storage) {
            return new Quadrilateral_F64(new Point2D_F64(100, 100), new Point2D_F64(140, 100),
                    new Point2D_F64(140, 140), new Point2D_F64(100, 140));
        }
    }

    /** Pure image-plane translation of {@code px} pixels to image-right. */
    private static Homography2D_F64 translation(double px) {
        return new Homography2D_F64(1, 0, -px, 0, 1, 0, 0, 0, 1);
    }

    /** The scripted frames: 10 px of image-right motion per frame, a stitching failure on frame 2. */
    private static FrameSource frames(ScriptedStitching stitch) {
        return new FrameSource() {
            @Override public int frameCount() { return 5; }
            @Override public TimestampedFrame frame(int i) {
                stitch.forceFailure = i == 2;
                if (i > 0) {
                    stitch.motion.advance(translation(10.0));
                }
                return new TimestampedFrame(i, i * 0.1, new GrayF32(FRAME, FRAME));
            }
        };
    }

    private static MotionModelStitchingEstimator<GrayF32, Homography2D_F64> estimator(
            ScriptedStitching stitch, boolean metric, boolean heading) {
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est =
                new MotionModelStitchingEstimator<>(stitch, MotionModelSupport.HOMOGRAPHY);
        est.setShrinkScale(0.5);
        est.setMinDistanceFromBorder(10);
        if (metric) {
            est.enableMetricReadout(new MetricReadoutConfig(800.0, 1, 80.0, 2.0, 0.1));   // 0.1 m/px
        }
        if (heading) {
            est.enableHeadingReadout(new HeadingReadoutConfig(HeadingSemantics.SIM_NADIR_CAMERA_HEADING,
                    null, 2.0, 0.1, HeadingReadoutConfig.DEFAULT_MAX_RATE_DEG_PER_S));
        }
        return est;
    }

    private static Path run(Path runDir, boolean alignmentSidecar, double firstHeadingSampleTimeS)
            throws IOException {
        Files.createDirectories(runDir);
        Path heightCsv = runDir.resolve("height.csv");
        Files.writeString(heightCsv, "timestamp_s,baro_relative_m\n0.0,0.0\n0.1,0.0\n0.2,0.0\n0.3,0.0\n0.4,0.0\n",
                StandardCharsets.UTF_8);
        Path headingCsv = runDir.resolve("heading.csv");
        StringBuilder h = new StringBuilder("timestamp_s,heading_deg\n");
        for (double t = firstHeadingSampleTimeS; t <= 0.4001; t += 0.1) {
            h.append(String.format(java.util.Locale.ROOT, "%.3f,%.1f%n", t, 90.0));   // camera image-up = East
        }
        Files.writeString(headingCsv, h.toString(), StandardCharsets.UTF_8);

        ScriptedStitching stitch = new ScriptedStitching();
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> estimator = estimator(stitch, true, true);
        VoRunner runner = new VoRunner(estimator, frames(stitch));
        runner.setMetricTrackCsv(runDir.resolve("metric_track.csv"),
                HeightSampleStream.fromCsv(heightCsv, "timestamp_s", "baro_relative_m", 0.0, 1, false));
        runner.setHeadingStream(HeadingSampleStream.fromCsv(headingCsv, "timestamp_s", "heading_deg", 0.0, 1));
        if (alignmentSidecar) {
            runner.setAlignmentSidecar(runDir);
        }
        RunManifest manifest = new RunManifest();
        manifest.runId = "test";
        manifest.datasetId = "d";
        manifest.datasetRevision = "r";
        manifest.estimatorId = "stitching-vo";
        manifest.estimatorVersion = "test";
        manifest.evaluatorCaptureVersion = "test";
        manifest.estimatorConfig = Map.of();
        manifest.environment = new Environment("h", "os", "cpu", "19", 1024, false);
        manifest.runTimestamp = "2026-09-07T00:00:00Z";
        try (RunRecordWriter writer = new RunRecordWriter(runDir, manifest, 5)) {
            runner.run(writer);
        }
        return runDir;
    }

    private static int col(List<String> lines, String name) {
        String[] header = lines.get(0).split(",", -1);
        for (int i = 0; i < header.length; i++) {
            if (header[i].equals(name)) return i;
        }
        throw new AssertionError("no column " + name);
    }

    private static String[][] rows(List<String> lines) {
        return lines.subList(1, lines.size()).stream().map(l -> l.split(",", -1)).toArray(String[][]::new);
    }

    @Test
    public void restartOpensTheVoSegmentWithPositionAbsentHeadingValidAndMetresThroughout(@TempDir Path tmp)
            throws IOException {
        Path runDir = run(tmp.resolve("run"), true, 0.0);
        List<String> frames = Files.readAllLines(runDir.resolve(AlignmentSidecarWriter.FRAMES_FILE));
        assertEquals(6, frames.size(), "header + 5 frames");
        String[][] r = rows(frames);
        int event = col(frames, "event"), seg = col(frames, "segment_id");
        int posValid = col(frames, "global_position_valid"), hdgValid = col(frames, "heading_valid");
        int poseValid = col(frames, "global_pose_valid"), gx = col(frames, "global_east_m");
        int lx = col(frames, "local_east_m"), ly = col(frames, "local_north_m"), hdg = col(frames, "heading_deg");
        int epoch = col(frames, "alignment_epoch_id"), eSince = col(frames, "e_since"), usable = col(frames, "translation_usable");

        assertEquals(List.of("init", "none", "hard_loss", "none", "none"),
                List.of(r[0][event], r[1][event], r[2][event], r[3][event], r[4][event]));
        assertEquals(List.of("0", "0", "1", "1", "1"),
                List.of(r[0][seg], r[1][seg], r[2][seg], r[3][seg], r[4][seg]), "the VO's segment index");
        assertEquals(List.of("true", "true", "false", "false", "false"),
                List.of(r[0][posValid], r[1][posValid], r[2][posValid], r[3][posValid], r[4][posValid]));
        assertEquals(List.of("true", "true", "true", "true", "true"),
                List.of(r[0][hdgValid], r[1][hdgValid], r[2][hdgValid], r[3][hdgValid], r[4][hdgValid]),
                "the heading channel never saw the visual failure");
        assertEquals(List.of("true", "true", "false", "false", "false"),
                List.of(r[0][poseValid], r[1][poseValid], r[2][poseValid], r[3][poseValid], r[4][poseValid]));
        assertEquals("", r[2][gx], "absence is an empty field, never zero");
        assertEquals("", r[4][gx]);
        // 10 px image-right per frame at 0.1 m/px, camera heading 90 (image-up = East) →
        // image-right = South: 1 m south per frame, in METRES.
        assertEquals(0.0, Double.parseDouble(r[1][lx]), 1e-9);
        assertEquals(-1.0, Double.parseDouble(r[1][ly]), 1e-9, "frame 1: 1 m south, metres not pixels");
        assertEquals(-1.0, Double.parseDouble(r[1][col(frames, "global_north_m")]), 1e-9);
        assertEquals(0.0, Double.parseDouble(r[2][ly]), 0.0, "restart: the segment origin");
        assertEquals(-1.0, Double.parseDouble(r[3][ly]), 1e-9, "segment 1: measured from the restart");
        assertEquals(-2.0, Double.parseDouble(r[4][ly]), 1e-9);
        for (String[] row : r) {
            assertEquals(90.0, Double.parseDouble(row[hdg]), 0.0, "the authoritative heading, every frame");
            assertEquals("true", row[usable]);
        }
        assertEquals(List.of("0", "0", "1", "1", "1"),
                List.of(r[0][epoch], r[1][epoch], r[2][epoch], r[3][epoch], r[4][epoch]));
        assertEquals(List.of("0", "1", "1", "2", "3"),
                List.of(r[0][eSince], r[1][eSince], r[2][eSince], r[3][eSince], r[4][eSince]),
                "exposure counts usable increments only, and a loss does not reset it");

        List<String> events = Files.readAllLines(runDir.resolve(AlignmentSidecarWriter.EVENTS_FILE));
        assertEquals(2, events.size(), "header + one hard loss");
        String[] loss = events.get(1).split(",", -1);
        assertEquals("2", loss[col(events, "frame_index")]);
        assertEquals("hard_loss", loss[col(events, "kind")]);
        assertEquals("0", loss[col(events, "segment_id")]);
        assertTrue(loss[col(events, "detail")].contains("new_segment_id=1"));
        assertTrue(loss[col(events, "detail")].contains("heading_known_across_gap=true"));

        // The VO's own metric sidecar agrees on the segment index — the invariant in the logs.
        List<String> metric = Files.readAllLines(runDir.resolve("metric_track.csv"));
        int mseg = col(metric, "segment_index"), gap = col(metric, "unknown_translation_gap");
        String[][] m = rows(metric);
        assertEquals(List.of("0", "0", "1", "1", "1"), List.of(m[0][mseg], m[1][mseg], m[2][mseg], m[3][mseg], m[4][mseg]));
        assertEquals("1", m[2][gap]);
    }

    @Test
    public void theSidecarDoesNotChangeTheRunRecord(@TempDir Path tmp) throws IOException {
        List<String> off = Files.readAllLines(run(tmp.resolve("off"), false, 0.0).resolve("frames.csv"));
        List<String> on = Files.readAllLines(run(tmp.resolve("on"), true, 0.0).resolve("frames.csv"));
        assertEquals(off.size(), on.size());
        for (int i = 0; i < off.size(); i++) {
            String[] a = off.get(i).split(",", -1), b = on.get(i).split(",", -1);
            assertEquals(a.length, b.length);
            for (int c = 0; c < a.length - 1; c++) {      // last column is wall-clock time
                assertEquals(a[c], b[c], "row " + i + " column " + c);
            }
        }
        assertFalse(Files.exists(tmp.resolve("off").resolve(AlignmentSidecarWriter.FRAMES_FILE)),
                "off means no file, not an empty one");
    }

    /** Test F on the capture path: no heading at the start → no increment → position UNKNOWN, no yaw invented. */
    @Test
    public void headingUnavailableAtTheStartIsADropoutNotAFabricatedPose(@TempDir Path tmp) throws IOException {
        Path runDir = run(tmp.resolve("late"), true, 0.15);   // first heading sample at 0.15 s
        List<String> frames = Files.readAllLines(runDir.resolve(AlignmentSidecarWriter.FRAMES_FILE));
        String[][] r = rows(frames);
        int event = col(frames, "event"), posValid = col(frames, "global_position_valid");
        int hdgValid = col(frames, "heading_valid"), hdg = col(frames, "heading_deg"), status = col(frames, "heading_status");
        assertEquals(List.of("init", "translation_dropout", "hard_loss", "none", "none"),
                List.of(r[0][event], r[1][event], r[2][event], r[3][event], r[4][event]),
                "frame 1's increment had no k-1 heading (the first sample arrived at 0.15 s); the "
                        + "restart frame at 0.2 s re-latched a FRESH heading, so frame 3's increment "
                        + "is usable — but the position stays UNKNOWN: segment 1 was never aligned");
        assertEquals(List.of("true", "false", "false", "false", "false"),
                List.of(r[0][posValid], r[1][posValid], r[2][posValid], r[3][posValid], r[4][posValid]));
        assertEquals(List.of("false", "false", "true", "true", "true"),
                List.of(r[0][hdgValid], r[1][hdgValid], r[2][hdgValid], r[3][hdgValid], r[4][hdgValid]));
        assertEquals("", r[0][hdg], "no heading exists: an empty cell, not 0 and not the visual yaw");
        assertEquals("UNAVAILABLE", r[0][status]);
        assertEquals("90.0", r[2][hdg]);

        List<String> events = Files.readAllLines(runDir.resolve(AlignmentSidecarWriter.EVENTS_FILE));
        assertTrue(events.stream().anyMatch(l -> l.contains(",translation_dropout,")));
        assertTrue(events.stream().filter(l -> l.contains(",translation_dropout,")).count() == 1,
                "recorded once per known → unknown transition (segment 1 was never known)");
    }

    @Test
    public void theLayerRefusesAnEstimatorWithoutTheMetricHeadingContract(@TempDir Path tmp) {
        ScriptedStitching s1 = new ScriptedStitching();
        VoRunner pixels = new VoRunner(estimator(s1, false, false), frames(s1));
        IllegalStateException e1 = assertThrows(IllegalStateException.class, () -> pixels.setAlignmentSidecar(tmp));
        assertTrue(e1.getMessage().contains("metric readout"));

        ScriptedStitching s2 = new ScriptedStitching();
        VoRunner visualYaw = new VoRunner(estimator(s2, true, false), frames(s2));
        IllegalStateException e2 = assertThrows(IllegalStateException.class, () -> visualYaw.setAlignmentSidecar(tmp));
        assertTrue(e2.getMessage().contains("heading"));
    }
}
