package org.boofcv.evaluation;

import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.struct.image.GrayF32;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.shapes.Quadrilateral_F64;
import org.boofcv.stitching.StitchingEstimator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class VoRunnerTest {

    /**
     * Minimal scripted stub, adapted from StitchingEstimatorTest's MockStitching: static corners
     * (not derived from the homography) so a test can directly control when nearBorder() fires,
     * and a per-call scripted process() result so success/failure sequences are deterministic.
     */
    private static class MockStitching extends StitchingFromMotion2D<GrayF32, Homography2D_F64> {
        boolean[] processResults;
        int processCallCount = 0;
        Quadrilateral_F64[] cornersPerCall; // corners returned on the Nth *post-configure* getImageCorners call
        final Homography2D_F64 worldToCurr = new Homography2D_F64();
        final GrayF32 stitchedImage = new GrayF32(200, 200);
        private boolean justConfigured = false;
        int cornerCallIndex = -1;

        @SuppressWarnings("unchecked")
        MockStitching() {
            super(
                    mock(boofcv.abst.sfm.d2.ImageMotion2D.class),
                    mock(boofcv.alg.distort.ImageDistort.class),
                    mock(boofcv.alg.sfm.d2.StitchingTransform.class),
                    0.5
            );
            worldToCurr.setTo(1, 0, 0, 0, 1, 0, 0, 0, 1);
        }

        @SuppressWarnings("unchecked")
        private static <I> I mock(Class<I> iface) {
            return (I) java.lang.reflect.Proxy.newProxyInstance(
                    iface.getClassLoader(),
                    new Class<?>[]{iface},
                    (proxy, method, args) -> {
                        if (method.getName().equals("equals")) return proxy == args[0];
                        if (method.getName().equals("hashCode")) return System.identityHashCode(proxy);
                        if (method.getName().equals("toString")) return "Mock " + iface.getSimpleName();
                        Class<?> rt = method.getReturnType();
                        if (rt == boolean.class) return false;
                        if (rt == int.class) return 0;
                        if (rt == double.class) return 0.0;
                        if (rt == float.class) return 0.0f;
                        if (rt == long.class) return 0L;
                        if (rt.isAssignableFrom(Homography2D_F64.class)) return new Homography2D_F64();
                        return null;
                    }
            );
        }

        @Override
        public boolean process(GrayF32 image) {
            if (justConfigured) {
                justConfigured = false;
                processCallCount++;
                return true;
            }
            boolean result = processResults[processCallCount];
            processCallCount++;
            return result;
        }

        @Override
        public void reset() {
        }

        @Override
        public void configure(int widthStitch, int heightStitch, Homography2D_F64 worldToInitArg) {
            this.worldToCurr.setTo(worldToInitArg);
            this.justConfigured = true;
        }

        @Override
        public Homography2D_F64 getWorldToCurr() {
            return worldToCurr;
        }

        @Override
        public Quadrilateral_F64 getImageCorners(int width, int height, Quadrilateral_F64 storage) {
            cornerCallIndex++;
            Quadrilateral_F64 result = cornersPerCall[cornerCallIndex];
            if (storage == null) storage = new Quadrilateral_F64();
            storage.a.setTo(result.a);
            storage.b.setTo(result.b);
            storage.c.setTo(result.c);
            storage.d.setTo(result.d);
            return storage;
        }

        @Override
        public GrayF32 getStitchedImage() {
            return stitchedImage;
        }

        @Override
        public void setOriginToCurrent() {
        }
    }

    private static class FixedFrameSource implements FrameSource {
        private final int count;

        FixedFrameSource(int count) {
            this.count = count;
        }

        @Override
        public int frameCount() {
            return count;
        }

        @Override
        public TimestampedFrame frame(int index) {
            return new TimestampedFrame(index, index * 0.1, new GrayF32(100, 100));
        }
    }

    private static final Quadrilateral_F64 CENTERED =
            new Quadrilateral_F64(20, 20, 180, 20, 180, 180, 20, 180);
    private static final Quadrilateral_F64 NEAR_BORDER =
            new Quadrilateral_F64(1, 1, 199, 1, 199, 199, 1, 199);

    /**
     * Scripts 5 frames: init, none, restart (process fails), recenter (corners near border),
     * none.
     *
     * <p>{@code process()} call indices 0 and 3 are consumed internally by
     * {@code StitchingEstimator} re-initializing after {@code configure()} (frame 0's init, and
     * frame 2's post-restart re-init) — those calls short-circuit to {@code true} without
     * reading {@code processResults}, so those slots are unused placeholders.
     */
    private static MockStitching scriptedMock() {
        MockStitching mock = new MockStitching();
        mock.processResults = new boolean[]{
                true,  // 0: unused (frame 0 init re-init call)
                true,  // 1: frame 1
                false, // 2: frame 2 (triggers restart)
                true,  // 3: unused (frame 2's post-restart re-init call)
                true,  // 4: frame 3
                true,  // 5: frame 4
        };
        // Corner-read schedule. Invariant to the navigation source, and this is worth stating
        // because it is not obvious: every getImageCorners call in MotionModelStitchingEstimator is
        // UNCONDITIONAL (the per-frame canvas-border check, and the reference-centre capture at
        // init / restart / re-origin), so flipping the default to RIGID_MOTION in DEC-VO-008 left
        // the schedule below untouched. Only legacyUpdateOffsetFromCentre / legacyUpdatePose are
        // gated on the source, and neither reads corners. This test asserts run-record EVENT
        // classification, which is likewise source-independent -- no pose value is asserted.
        mock.cornersPerCall = new Quadrilateral_F64[]{
                CENTERED, // frame 0 init
                CENTERED, // frame 1 (none)
                CENTERED, // frame 2 restart re-init corners (post-reset)
                NEAR_BORDER, // frame 3: triggers the canvas re-origin
                CENTERED,   // frame 3: corners re-read after the re-origin
                CENTERED,   // frame 4 (none)
        };
        return mock;
    }

    private static List<String[]> writeAndReadRows(Path runDir) throws IOException {
        MockStitching mock = scriptedMock();
        StitchingEstimator<GrayF32> estimator = new StitchingEstimator<>(mock);
        estimator.setShrinkScale(0.5);
        estimator.setMinDistanceFromBorder(10);
        VoRunner runner = new VoRunner(estimator, new FixedFrameSource(5));

        RunManifest manifest = new RunManifest();
        manifest.runId = "test";
        manifest.datasetId = "d";
        manifest.datasetRevision = "r";
        manifest.estimatorId = "stitching-vo";
        manifest.estimatorVersion = "test";
        manifest.evaluatorCaptureVersion = "test";
        manifest.estimatorConfig = Map.of();
        manifest.environment = new Environment("h", "os", "cpu", "19", 1024, false);
        manifest.runTimestamp = "2026-08-12T00:00:00Z";

        try (RunRecordWriter writer = new RunRecordWriter(runDir, manifest, 5)) {
            runner.run(writer);
        }

        List<String[]> rows = new ArrayList<>();
        List<String> lines = Files.readAllLines(runDir.resolve("frames.csv"));
        for (int i = 1; i < lines.size(); i++) {
            rows.add(lines.get(i).split(",", -1));
        }
        return rows;
    }

    @Test
    public void testEventClassification(@TempDir Path tempDir) throws IOException {
        List<String[]> rows = writeAndReadRows(tempDir.resolve("run"));

        assertEquals(5, rows.size());
        assertEquals("init", rows.get(0)[7]);
        assertEquals("none", rows.get(1)[7]);
        assertEquals("restart", rows.get(2)[7]);
        assertEquals("recenter", rows.get(3)[7]);
        assertEquals("none", rows.get(4)[7]);

        assertEquals("false", rows.get(2)[6]); // success column for the restart row
        assertEquals("true", rows.get(3)[6]);

        // reference_id increments on restart and recenter (data-model.md), starts at 0
        assertEquals("0", rows.get(0)[8]);
        assertEquals("0", rows.get(1)[8]);
        assertEquals("1", rows.get(2)[8]); // restart
        assertEquals("2", rows.get(3)[8]); // recenter
        assertEquals("2", rows.get(4)[8]);
    }

    @Test
    public void testIdenticalInputYieldsIdenticalTrajectory(@TempDir Path tempDir) throws IOException {
        List<String[]> rowsA = writeAndReadRows(tempDir.resolve("runA"));
        List<String[]> rowsB = writeAndReadRows(tempDir.resolve("runB"));

        assertEquals(rowsA.size(), rowsB.size());
        for (int i = 0; i < rowsA.size(); i++) {
            // Every column except process_time_ns (last, real wall-clock time) must match exactly.
            String[] a = rowsA.get(i);
            String[] b = rowsB.get(i);
            for (int c = 0; c < a.length - 1; c++) {
                assertEquals(a[c], b[c], "column " + c + " differs at row " + i);
            }
        }
    }
}
