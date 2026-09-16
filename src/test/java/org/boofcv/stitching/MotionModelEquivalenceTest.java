package org.boofcv.stitching;

import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.struct.image.GrayF32;
import georegression.struct.affine.Affine2D_F64;
import georegression.struct.homography.Homography2D_F64;
import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Guards {@code DEC-VO-002}'s generalisation of the pose estimator over the motion-model type, and
 * establishes that the affine arm of {@code EXP-VO-002} is actually constructible and actually
 * different.
 *
 * <p>Three things are asserted, in increasing order of what they buy:
 *
 * <ol>
 *   <li><b>The generalisation did not change the shipped path.</b> {@link StitchingEstimator} (now a
 *       homography-fixed subclass) and an explicitly-constructed
 *       {@link MotionModelStitchingEstimator} with {@link MotionModelSupport#HOMOGRAPHY} produce
 *       bitwise-identical poses over a sequence that exercises respawn/recentering. This is the
 *       cheap, local half of the reproduction guarantee; the expensive half is
 *       {@code EXP-VO-002}'s reproduction of {@code EXP-002}'s committed run record.</li>
 *   <li><b>The affine path runs end to end</b> — builds, configures, processes, recentres, and
 *       yields finite poses. Without this the experiment's decision boundary ("can a fair
 *       comparison even be constructed?") would be untested.</li>
 *   <li><b>The two models actually diverge</b>, so a null result in {@code EXP-VO-002} could never
 *       be an artifact of accidentally running the same estimator twice.</li>
 * </ol>
 *
 * <p>Evidence tier: T1 (synthetic). Says nothing about which model navigates better — that is
 * {@code EXP-VO-002}'s question, on real data.
 */
public class MotionModelEquivalenceTest {

    private static final int WORLD_SIZE = 512;
    private static final int FRAME_SIZE = 320;
    private static final int FRAME_COUNT = 24;
    private static final double EXACT = 0.0;

    private record Observation(boolean success, boolean recentered, double x, double y, double yaw) {
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

    private static GrayF32 frameAt(GrayF32 world, int index) {
        GrayF32 frame = new GrayF32(FRAME_SIZE, FRAME_SIZE);
        int offsetX = 4 + index * 8;
        int offsetY = 4 + index * 5;
        for (int y = 0; y < FRAME_SIZE; y++) {
            for (int x = 0; x < FRAME_SIZE; x++) {
                frame.set(x, y, world.get(x + offsetX, y + offsetY));
            }
        }
        return frame;
    }

    private static List<Observation> run(MotionModelStitchingEstimator<GrayF32, ?> estimator,
                                          GrayF32 world) {
        boolean[] recenterFlag = new boolean[1];
        estimator.setRecenterListener(h -> recenterFlag[0] = true);
        estimator.setShrinkScale(0.5);
        estimator.setMinDistanceFromBorder(10);

        List<Observation> observations = new ArrayList<>();
        for (int i = 0; i < FRAME_COUNT; i++) {
            recenterFlag[0] = false;
            boolean success = estimator.processFrame(frameAt(world, i));
            Pose3D p = estimator.getCurrentPose();
            observations.add(new Observation(success, recenterFlag[0], p.x, p.y, p.yaw));
        }
        return observations;
    }

    private static MotionModelStitchingEstimator<GrayF32, Homography2D_F64> genericHomography() {
        StitchingFromMotion2D<GrayF32, Homography2D_F64> stitch =
                StitchingFactory.builder().buildGray();
        return new MotionModelStitchingEstimator<>(stitch, MotionModelSupport.HOMOGRAPHY);
    }

    private static MotionModelStitchingEstimator<GrayF32, Affine2D_F64> genericAffine() {
        StitchingFromMotion2D<GrayF32, Affine2D_F64> stitch =
                StitchingFactory.builder().buildGrayAffine();
        return new MotionModelStitchingEstimator<>(stitch, MotionModelSupport.AFFINE);
    }

    @Test
    public void generalisedEstimatorMatchesShippedHomographyPathBitwise() {
        GrayF32 world = syntheticTexturedWorld();

        List<Observation> shipped =
                run(new StitchingEstimator<>(StitchingFactory.builder().buildGray()), world);
        List<Observation> generalised = run(genericHomography(), world);

        assertEquals(shipped.size(), generalised.size(), "frame count");
        for (int i = 0; i < shipped.size(); i++) {
            Observation a = shipped.get(i);
            Observation b = generalised.get(i);
            assertEquals(a.success(), b.success(), "success flag at frame " + i);
            assertEquals(a.recentered(), b.recentered(), "recenter event at frame " + i);
            assertEquals(a.x(), b.x(), EXACT, "pose.x at frame " + i);
            assertEquals(a.y(), b.y(), EXACT, "pose.y at frame " + i);
            assertEquals(a.yaw(), b.yaw(), EXACT, "pose.yaw at frame " + i);
        }
    }

    /** Guards the comparison above against passing on a trivial steady-state sequence. */
    @Test
    public void sequenceExercisesRespawnOrRecentering() {
        List<Observation> obs = run(genericHomography(), syntheticTexturedWorld());
        assertTrue(obs.stream().anyMatch(o -> o.recentered() || !o.success()),
                "expected at least one recenter or restart; otherwise the equivalence assertions "
                        + "only cover steady-state tracking");
    }

    @Test
    public void affinePathRunsEndToEndAndReportsItsModel() {
        MotionModelStitchingEstimator<GrayF32, Affine2D_F64> affine = genericAffine();
        assertEquals("affine", affine.motionModelId());
        assertEquals("homography", genericHomography().motionModelId());

        List<Observation> obs = run(affine, syntheticTexturedWorld());
        assertEquals(FRAME_COUNT, obs.size(), "affine arm processed every frame");
        for (int i = 0; i < obs.size(); i++) {
            assertTrue(Double.isFinite(obs.get(i).x()), "finite pose.x at frame " + i);
            assertTrue(Double.isFinite(obs.get(i).y()), "finite pose.y at frame " + i);
            assertTrue(Double.isFinite(obs.get(i).yaw()), "finite pose.yaw at frame " + i);
        }
    }

    /**
     * The two models must produce genuinely different trajectories on the same input. If they did
     * not, every comparative result in {@code EXP-VO-002} would be vacuous — so this asserts the
     * ablation has an effect at all, without asserting anything about which model is better.
     */
    @Test
    public void affineAndHomographyDivergeOnTheSameInput() {
        GrayF32 world = syntheticTexturedWorld();
        List<Observation> h = run(genericHomography(), world);
        List<Observation> a = run(genericAffine(), world);

        assertEquals(h.size(), a.size(), "frame count");
        boolean anyDifference = false;
        for (int i = 0; i < h.size() && !anyDifference; i++) {
            anyDifference = h.get(i).x() != a.get(i).x()
                    || h.get(i).y() != a.get(i).y()
                    || h.get(i).yaw() != a.get(i).yaw();
        }
        assertTrue(anyDifference,
                "affine and homography produced identical trajectories, which would mean the "
                        + "motion model is not actually being switched");

        // And the divergence must not be a degenerate collapse to nothing.
        assertNotEquals(0.0, a.get(a.size() - 1).x() * a.get(a.size() - 1).x()
                + a.get(a.size() - 1).y() * a.get(a.size() - 1).y(),
                "affine arm produced a zero trajectory");
    }
}
