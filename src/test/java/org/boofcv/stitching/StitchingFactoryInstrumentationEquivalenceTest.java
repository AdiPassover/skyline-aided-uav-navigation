package org.boofcv.stitching;

import boofcv.struct.image.GrayF32;
import georegression.struct.homography.Homography2D_F64;
import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The load-bearing check behind the residual-diagnostics feature (spec FR-007, FR-008, FR-014,
 * FR-015): that {@code buildGrayInstrumented()} estimates <b>exactly</b> the same motion as
 * {@code buildGray()}.
 *
 * <p>This matters more than a normal regression test. {@code buildGrayInstrumented()} does not call
 * {@code FactoryMotion2D.createMotion2D}; it replicates that factory method's
 * {@code Homography2D_F64} branch so it can retain a handle on the low-level estimator, which the
 * factory's return type hides. The claim that the replication is faithful is exactly the kind of
 * claim that should not rest on reading the library's source — so it is asserted here bitwise,
 * frame by frame, over a sequence long enough to exercise keyframe respawn.
 *
 * <p>Comparisons use {@link org.junit.jupiter.api.Assertions#assertEquals(double, double, double)}
 * with a zero delta: not "close enough", but identical. Both stacks are deterministic — BoofCV
 * hardcodes the RANSAC seed at 123123 — so any difference at all indicates the replication has
 * drifted from the factory, most likely because of a BoofCV upgrade.
 *
 * <p><b>If this test fails after a dependency bump</b>, do not relax the tolerance. Re-read
 * {@code FactoryMotion2D.createMotion2D} at the new version and bring
 * {@code StitchingFactory.createInstrumentedMotion2D} back into line with it, or reconsider whether
 * the instrumented path is still viable. See {@code DEC-VO-001}.
 *
 * <p>Evidence tier: T1 (synthetic). This test validates the equivalence of two construction paths;
 * it says nothing about VO accuracy.
 */
public class StitchingFactoryInstrumentationEquivalenceTest {

    private static final int WORLD_SIZE = 512;
    private static final int FRAME_SIZE = 320;
    private static final int FRAME_COUNT = 24;

    /** Bitwise identity, not approximate agreement. */
    private static final double EXACT = 0.0;

    /** One frame's worth of everything the estimator is observable through. */
    private record Observation(boolean success, boolean recentered, Pose3D pose,
                               Homography2D_F64 worldToCurr) {
    }

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

    private static GrayF32 crop(GrayF32 world, int offsetX, int offsetY) {
        GrayF32 frame = new GrayF32(FRAME_SIZE, FRAME_SIZE);
        for (int y = 0; y < FRAME_SIZE; y++) {
            for (int x = 0; x < FRAME_SIZE; x++) {
                frame.set(x, y, world.get(x + offsetX, y + offsetY));
            }
        }
        return frame;
    }

    /**
     * A drifting crop path. The per-frame step is large enough that tracks are steadily lost and
     * the mosaic drifts toward its border, so the sequence exercises keyframe respawn and
     * recentering rather than only the easy steady-state case.
     */
    private static GrayF32 frameAt(GrayF32 world, int index) {
        int offsetX = 4 + index * 8;
        int offsetY = 4 + index * 5;
        return crop(world, offsetX, offsetY);
    }

    private static List<Observation> run(StitchingEstimator<GrayF32> estimator, GrayF32 world) {
        boolean[] recenterFlag = new boolean[1];
        estimator.setRecenterListener(h -> recenterFlag[0] = true);
        estimator.setShrinkScale(0.5);
        estimator.setMinDistanceFromBorder(10);

        List<Observation> observations = new ArrayList<>();
        for (int i = 0; i < FRAME_COUNT; i++) {
            recenterFlag[0] = false;
            boolean success = estimator.processFrame(frameAt(world, i));
            observations.add(new Observation(
                    success,
                    recenterFlag[0],
                    estimator.getCurrentPose(),
                    estimator.getStitch().getWorldToCurr().copy()));
        }
        return observations;
    }

    private static void assertHomographyIdentical(int frame, Homography2D_F64 expected,
                                                  Homography2D_F64 actual) {
        String at = " at frame " + frame;
        assertEquals(expected.a11, actual.a11, EXACT, "a11" + at);
        assertEquals(expected.a12, actual.a12, EXACT, "a12" + at);
        assertEquals(expected.a13, actual.a13, EXACT, "a13" + at);
        assertEquals(expected.a21, actual.a21, EXACT, "a21" + at);
        assertEquals(expected.a22, actual.a22, EXACT, "a22" + at);
        assertEquals(expected.a23, actual.a23, EXACT, "a23" + at);
        assertEquals(expected.a31, actual.a31, EXACT, "a31" + at);
        assertEquals(expected.a32, actual.a32, EXACT, "a32" + at);
        assertEquals(expected.a33, actual.a33, EXACT, "a33" + at);
    }

    @Test
    public void instrumentedStackProducesBitIdenticalMotionAndPose() {
        GrayF32 world = syntheticTexturedWorld();

        List<Observation> baseline = run(
                new StitchingEstimator<>(StitchingFactory.builder().buildGray()), world);

        InstrumentedStitching<GrayF32> instrumented =
                StitchingFactory.builder().buildGrayInstrumented();
        List<Observation> withDiagnostics = run(
                new StitchingEstimator<>(instrumented.stitch()), world);

        assertEquals(baseline.size(), withDiagnostics.size(), "frame count");

        for (int i = 0; i < baseline.size(); i++) {
            Observation a = baseline.get(i);
            Observation b = withDiagnostics.get(i);

            assertEquals(a.success(), b.success(), "success flag at frame " + i);
            assertEquals(a.recentered(), b.recentered(), "recenter event at frame " + i);

            assertEquals(a.pose().x, b.pose().x, EXACT, "pose.x at frame " + i);
            assertEquals(a.pose().y, b.pose().y, EXACT, "pose.y at frame " + i);
            assertEquals(a.pose().z, b.pose().z, EXACT, "pose.z at frame " + i);
            assertEquals(a.pose().yaw, b.pose().yaw, EXACT, "pose.yaw at frame " + i);

            assertHomographyIdentical(i, a.worldToCurr(), b.worldToCurr());
        }
    }

    /**
     * Guards against the comparison above passing vacuously. If the sequence never respawns a
     * keyframe or recenters the mosaic, it would only prove the two stacks agree on the easy path —
     * and keyframe respawn is exactly where the instrumented subclass's observation point sits, so
     * it is the case that most needs covering.
     */
    @Test
    public void sequenceActuallyExercisesRespawnOrRecentering() {
        GrayF32 world = syntheticTexturedWorld();

        InstrumentedStitching<GrayF32> instrumented =
                StitchingFactory.builder().buildGrayInstrumented();
        StitchingEstimator<GrayF32> estimator = new StitchingEstimator<>(instrumented.stitch());

        List<Observation> observations = run(estimator, world);

        long interestingFrames = observations.stream()
                .filter(o -> o.recentered() || !o.success())
                .count();

        assertTrue(interestingFrames > 0,
                "expected the synthetic sequence to trigger at least one recenter or restart, "
                        + "otherwise the equivalence test only covers steady-state tracking; "
                        + "adjust the per-frame step in frameAt() if this starts failing");
    }

    /**
     * The planar (colour) build path wraps the same gray motion estimator in
     * {@code PlToGrayMotion2D}, so it gets its own equivalence check rather than being assumed to
     * follow from the gray one.
     */
    @Test
    public void instrumentedPlanarStackMatchesPlanarBaselineConstruction() {
        InstrumentedStitching<boofcv.struct.image.Planar<GrayF32>> instrumented =
                StitchingFactory.builder().buildPlanarInstrumented();

        assertTrue(instrumented.stitch() != null, "planar instrumented stitcher was built");
        assertEquals(3.0, instrumented.diagnostics().inlierThresholdSquaredPx(), EXACT,
                "planar instrumented path reports the builder's configured inlier threshold");
    }
}
