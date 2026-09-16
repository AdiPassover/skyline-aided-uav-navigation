package org.boofcv.stitching;

import boofcv.abst.feature.detect.interest.ConfigPointDetector;
import boofcv.abst.feature.detect.interest.PointDetectorTypes;
import boofcv.abst.sfm.d2.ImageMotion2D;
import boofcv.abst.tracker.PointTracker;
import boofcv.factory.sfm.FactoryMotion2D;
import boofcv.factory.tracker.FactoryPointTracker;
import boofcv.struct.image.GrayF32;
import georegression.struct.affine.Affine2D_F64;
import georegression.struct.homography.Homography2D_F64;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A synthetic sequence long enough to exercise keyframe respawn and mosaic recentering, plus the
 * plumbing to drive two differently-constructed motion estimators over it and compare them bitwise.
 *
 * <p>Extracted so {@code RefinementSemanticsTest} can make the same equivalence claim
 * {@code StitchingFactoryInstrumentationEquivalenceTest} makes for the residual probe, without
 * copying its harness — the two probes are separate features but the standard of evidence for
 * "observing changes nothing" is the same, and a copied harness is one refactor away from silently
 * diverging.
 *
 * <p>Bitwise means bitwise: comparisons use a zero delta. Both stacks are deterministic (BoofCV
 * hardcodes the RANSAC seed at 123123), so any difference indicates a real behavioural change.
 */
final class EquivalenceHarness {

    private static final int WORLD_SIZE = 512;
    private static final int FRAME_SIZE = 320;
    private static final int FRAME_COUNT = 24;
    private static final double EXACT = 0.0;

    private EquivalenceHarness() {
    }

    /** One frame's worth of everything the estimator is observable through. */
    record Observation(boolean success, boolean recentered, double x, double y, double yaw,
                       Homography2D_F64 worldToCurr) {
    }

    static GrayF32 syntheticTexturedWorld() {
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

    private static PointTracker<GrayF32> tracker() {
        ConfigPointDetector detector = new ConfigPointDetector();
        detector.type = PointDetectorTypes.SHI_TOMASI;
        detector.general.maxFeatures = 300;
        detector.general.radius = 3;
        detector.general.threshold = 1;
        return FactoryPointTracker.klt(4, detector, 3, GrayF32.class, GrayF32.class);
    }

    /** The stack BoofCV's own factory builds, for the given model and refinement setting. */
    static ImageMotion2D<GrayF32, ?> plain(StitchingFactory.ProbeModel model, boolean refine) {
        return model == StitchingFactory.ProbeModel.AFFINE
                ? FactoryMotion2D.createMotion2D(220, 3.0, 2, 30, 0.6, 0.5, refine, tracker(),
                                                 new Affine2D_F64())
                : FactoryMotion2D.createMotion2D(220, 3.0, 2, 30, 0.6, 0.5, refine, tracker(),
                                                 new Homography2D_F64());
    }

    /** The same stack with {@code EXP-VO-010}'s observing tracker key attached. */
    static StitchingFactory.ProbedMotion<?> probed(StitchingFactory.ProbeModel model,
                                                   boolean refine) {
        return StitchingFactory.createProbedMotion2D(
                model, 220, 3.0, 2, 30, 0.6, 0.5, refine, tracker());
    }

    /** The probed stack additionally carrying {@code EXP-CONF-004}'s observer-only refiner. */
    static StitchingFactory.ProbedMotion<?> probedWithDiagnosticRefit(
            StitchingFactory.ProbeModel model, boolean refine) {
        return StitchingFactory.createProbedMotion2D(
                model, 220, 3.0, 2, 30, 0.6, 0.5, refine, true, tracker());
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static List<Observation> run(ImageMotion2D<GrayF32, ?> motion,
                                 StitchingFactory.ProbeModel model, GrayF32 world) {
        MotionModelSupport support = model == StitchingFactory.ProbeModel.AFFINE
                ? MotionModelSupport.AFFINE : MotionModelSupport.HOMOGRAPHY;
        boolean[] recenterFlag = new boolean[1];

        boofcv.alg.sfm.d2.StitchingFromMotion2D stitch = FactoryMotion2D.createVideoStitch(
                0.55, (ImageMotion2D) motion, boofcv.struct.image.ImageType.single(GrayF32.class));
        MotionModelStitchingEstimator<GrayF32, ?> estimator =
                new MotionModelStitchingEstimator<>(stitch, support);
        estimator.setRecenterListener(h -> recenterFlag[0] = true);
        estimator.setShrinkScale(0.5);
        estimator.setMinDistanceFromBorder(10);
        estimator.setNavigationSource(NavigationSource.RIGID_MOTION);

        List<Observation> observations = new ArrayList<>();
        for (int i = 0; i < FRAME_COUNT; i++) {
            recenterFlag[0] = false;
            boolean success = estimator.processFrame(frameAt(world, i));
            var pose = estimator.getCurrentPose();
            observations.add(new Observation(success, recenterFlag[0], pose.x, pose.y, pose.yaw,
                    estimator.getStitch().getWorldToCurr(new Homography2D_F64())));
        }
        return observations;
    }

    static void assertIdentical(String at, Observation expected, Observation actual) {
        assertEquals(expected.success(), actual.success(), "success " + at);
        assertEquals(expected.recentered(), actual.recentered(), "recentered " + at);
        assertEquals(expected.x(), actual.x(), EXACT, "x " + at);
        assertEquals(expected.y(), actual.y(), EXACT, "y " + at);
        assertEquals(expected.yaw(), actual.yaw(), EXACT, "yaw " + at);
        Homography2D_F64 e = expected.worldToCurr();
        Homography2D_F64 a = actual.worldToCurr();
        assertEquals(e.a11, a.a11, EXACT, "a11 " + at);
        assertEquals(e.a12, a.a12, EXACT, "a12 " + at);
        assertEquals(e.a13, a.a13, EXACT, "a13 " + at);
        assertEquals(e.a21, a.a21, EXACT, "a21 " + at);
        assertEquals(e.a22, a.a22, EXACT, "a22 " + at);
        assertEquals(e.a23, a.a23, EXACT, "a23 " + at);
        assertEquals(e.a31, a.a31, EXACT, "a31 " + at);
        assertEquals(e.a32, a.a32, EXACT, "a32 " + at);
        assertEquals(e.a33, a.a33, EXACT, "a33 " + at);
    }
}
