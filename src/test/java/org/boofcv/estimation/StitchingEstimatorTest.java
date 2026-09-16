package org.boofcv.estimation;

import boofcv.abst.sfm.d2.ImageMotion2D;
import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.struct.image.GrayF32;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.shapes.Quadrilateral_F64;
import org.boofcv.stitching.StitchingEstimator;
import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Bookkeeping-layer tests for {@link StitchingEstimator}.
 *
 * <p><b>Migrated by {@code DEC-VO-003}.</b> These tests previously drove the mock's <i>canvas</i>
 * transform ({@code worldToCurr}) and asserted that the pose followed — i.e. they tested the
 * mosaic-derived pose path that decision removed. They now drive the <b>motion model's</b>
 * accumulated transform, which is the estimator's actual navigation input, and assert against the
 * logical-frame contract. The expected numeric values are unchanged where the quantity is unchanged,
 * which is itself evidence that the refactor preserved the intended semantics rather than redefining
 * them.
 *
 * <p>The canvas transform is still driven, because it still decides when the finite raster needs
 * re-originning — but it no longer contributes to the pose.
 *
 * <p>As before ({@code COMP-001} §11) these are T1 bookkeeping tests: no KLT, RANSAC or homography
 * estimation executes here. Real-stack behaviour is covered by
 * {@code LogicalNavigationStateTest} and {@code NavigationFrameCouplingDiagnosticTest}.
 */
public class StitchingEstimatorTest {

    private static final double EPS = 1e-6;

    /** Controllable stand-in for BoofCV's motion estimator: the navigation input under test. */
    private static class MockMotion implements ImageMotion2D<GrayF32, Homography2D_F64> {
        final Homography2D_F64 firstToCurrent = new Homography2D_F64();
        int setToFirstCount = 0;

        @Override public boolean process(GrayF32 input) { return true; }
        @Override public void reset() { firstToCurrent.reset(); }
        @Override public void setToFirst() { firstToCurrent.reset(); setToFirstCount++; }
        @Override public long getFrameID() { return 0; }
        @Override public Homography2D_F64 getFirstToCurrent() { return firstToCurrent; }
        @Override public Class<Homography2D_F64> getTransformType() { return Homography2D_F64.class; }
    }

    private static class MockStitching extends StitchingFromMotion2D<GrayF32, Homography2D_F64> {
        public final MockMotion mockMotion = new MockMotion();
        public Homography2D_F64 worldToCurr = new Homography2D_F64();
        public Homography2D_F64 worldToInit = new Homography2D_F64();
        public boolean processResult = true;
        public int processCallCount = 0;
        public int resetCallCount = 0;
        public int configureCallCount = 0;
        public int recenterCallCount = 0;
        public GrayF32 stitchedImage = new GrayF32(200, 200);
        private boolean justConfigured = false;

        @SuppressWarnings("unchecked")
        public MockStitching() {
            super(
                mock(boofcv.abst.sfm.d2.ImageMotion2D.class),
                mock(boofcv.alg.distort.ImageDistort.class),
                mock(boofcv.alg.sfm.d2.StitchingTransform.class),
                0.5
            );
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

        /** The navigation input. Overridden so tests can script the accumulated motion. */
        @Override
        public ImageMotion2D<GrayF32, Homography2D_F64> getMotion() { return mockMotion; }

        @Override
        public boolean process(GrayF32 image) {
            processCallCount++;
            if (justConfigured) {
                justConfigured = false;
                return true;
            }
            return processResult;
        }

        /** Faithful to BoofCV: {@code StitchingFromMotion2D.reset()} calls {@code motion.reset()}. */
        @Override
        public void reset() {
            resetCallCount++;
            mockMotion.reset();
        }

        @Override
        public void configure(int widthStitch, int heightStitch, Homography2D_F64 worldToInit) {
            configureCallCount++;
            this.worldToInit.setTo(worldToInit);
            this.worldToCurr.setTo(worldToInit);
            this.justConfigured = true;
        }

        @Override
        public Homography2D_F64 getWorldToCurr() { return worldToCurr; }

        @Override
        public Homography2D_F64 getWorldToCurr(Homography2D_F64 storage) {
            if (storage == null) storage = new Homography2D_F64();
            storage.setTo(worldToCurr);
            return storage;
        }

        @Override
        public Quadrilateral_F64 getImageCorners(int width, int height, Quadrilateral_F64 storage) {
            if (storage == null) storage = new Quadrilateral_F64();
            Homography2D_F64 currToWorld = worldToCurr.invert(null);
            georegression.transform.homography.HomographyPointOps_F64.transform(currToWorld, 0, 0, storage.a);
            georegression.transform.homography.HomographyPointOps_F64.transform(currToWorld, width, 0, storage.b);
            georegression.transform.homography.HomographyPointOps_F64.transform(currToWorld, width, height, storage.c);
            georegression.transform.homography.HomographyPointOps_F64.transform(currToWorld, 0, height, storage.d);
            return storage;
        }

        @Override
        public GrayF32 getStitchedImage() { return stitchedImage; }

        /** Faithful to BoofCV: {@code setOriginToCurrent()} calls {@code motion.setToFirst()}. */
        @Override
        public void setOriginToCurrent() {
            recenterCallCount++;
            worldToCurr.setTo(worldToInit);
            mockMotion.setToFirst();
        }

        /** Keeps the canvas transform consistent with a scripted image motion, as BoofCV does. */
        void applyMotion(Homography2D_F64 firstToCurrent) {
            mockMotion.firstToCurrent.setTo(firstToCurrent);
            worldToInit.concat(firstToCurrent, worldToCurr);
        }
    }

    private static Homography2D_F64 translation(double dx, double dy) {
        return new Homography2D_F64(1, 0, dx, 0, 1, dy, 0, 0, 1);
    }

    /** Rotation by {@code deg} about the image centre of a {@code size}×{@code size} frame. */
    private static Homography2D_F64 rotationAboutCentre(double deg, double size) {
        double r = Math.toRadians(deg), c = Math.cos(r), s = Math.sin(r), m = size / 2.0;
        return new Homography2D_F64(c, -s, m - (c * m - s * m), s, c, m - (s * m + c * m), 0, 0, 1);
    }

    private static StitchingEstimator<GrayF32> newEstimator(MockStitching mock) {
        StitchingEstimator<GrayF32> est = new StitchingEstimator<>(mock);
        est.setShrinkScale(0.5);
        est.setMinDistanceFromBorder(10);
        // These tests assert the DEC-VO-003 logical-frame contract; the legacy source is the
        // shipped default and is covered by the real-data reproduction check instead.
        est.setNavigationSource(org.boofcv.stitching.NavigationSource.LOGICAL_FRAME);
        return est;
    }

    @Test
    public void testInitialPose() {
        MockStitching mock = new MockStitching();
        StitchingEstimator<GrayF32> estimator = newEstimator(mock);

        assertTrue(estimator.processFrame(new GrayF32(100, 100)));
        assertEquals(1, mock.processCallCount);
        assertEquals(1, mock.configureCallCount);

        Pose3D pose = estimator.getCurrentPose();
        assertEquals(0.0, pose.x, EPS);
        assertEquals(0.0, pose.y, EPS);
        assertEquals(0.0, pose.z, EPS);
        assertEquals(0.0, pose.yaw, EPS);
    }

    /**
     * A camera translation appears in the motion model as the <i>opposite</i> image-content shift.
     * With {@code firstToCurrent} translating content by (−10, +20) px, the image centre lands at
     * (+10, −20) in the logical frame, which is a pose of (+10, +20) after the documented
     * image-down → pose-forward y negation. Same expected values as before the refactor.
     */
    @Test
    public void testTranslationPose() {
        MockStitching mock = new MockStitching();
        StitchingEstimator<GrayF32> estimator = newEstimator(mock);
        GrayF32 frame = new GrayF32(100, 100);

        estimator.processFrame(frame);

        mock.applyMotion(translation(-10, 20));
        estimator.processFrame(frame);

        Pose3D pose = estimator.getCurrentPose();
        assertEquals(10.0, pose.x, EPS);
        assertEquals(20.0, pose.y, EPS);
        assertEquals(0.0, pose.yaw, EPS);
    }

    /**
     * A pure rotation about the image centre must produce yaw and <b>no translation whatsoever</b> —
     * the property an arbitrary transformed image point would violate.
     */
    @Test
    public void testRotationPose() {
        MockStitching mock = new MockStitching();
        StitchingEstimator<GrayF32> estimator = newEstimator(mock);
        GrayF32 frame = new GrayF32(100, 100);

        estimator.processFrame(frame);

        mock.applyMotion(rotationAboutCentre(-30, 100));
        estimator.processFrame(frame);

        Pose3D pose = estimator.getCurrentPose();
        assertEquals(0.0, pose.x, 1e-4);
        assertEquals(0.0, pose.y, 1e-4);
        assertEquals(30.0, pose.yaw, 1e-4);
    }

    /**
     * Canvas re-origin must not move the logical pose. This is no longer vacuous: real accumulated
     * motion exists before the event, the mock faithfully zeroes {@code firstToCurrent} the way
     * BoofCV's {@code setToFirst()} does, and the pose is required to survive it exactly.
     */
    @Test
    public void testRecenterPoseContinuity() {
        MockStitching mock = new MockStitching();
        StitchingEstimator<GrayF32> estimator = newEstimator(mock);
        GrayF32 frame = new GrayF32(100, 100);

        estimator.processFrame(frame);

        // Enough image translation to push a canvas corner inside the 10 px margin.
        mock.applyMotion(translation(40, 12));
        estimator.processFrame(frame);

        assertEquals(1, mock.recenterCallCount, "expected exactly one canvas re-origin");
        assertEquals(1, mock.mockMotion.setToFirstCount, "canvas re-origin must reset the estimator");

        Pose3D beforeRecenter = estimator.getCurrentPose();
        assertEquals(-40.0, beforeRecenter.x, EPS, "pose must reflect the motion, not the canvas");
        assertEquals(12.0, beforeRecenter.y, EPS);

        // Next frame with no further motion: accumulation is identity, anchor carries everything.
        estimator.processFrame(frame);

        Pose3D afterRecenter = estimator.getCurrentPose();
        assertEquals(beforeRecenter.x, afterRecenter.x, EPS);
        assertEquals(beforeRecenter.y, afterRecenter.y, EPS);
        assertEquals(beforeRecenter.yaw, afterRecenter.yaw, EPS);
    }

    /** Motion accumulated after a canvas re-origin must compose onto the anchor, not restart from zero. */
    @Test
    public void testMotionComposesAcrossRecenter() {
        MockStitching mock = new MockStitching();
        StitchingEstimator<GrayF32> estimator = newEstimator(mock);
        GrayF32 frame = new GrayF32(100, 100);

        estimator.processFrame(frame);
        mock.applyMotion(translation(40, 12));   // triggers the canvas re-origin
        estimator.processFrame(frame);
        assertEquals(1, mock.recenterCallCount);

        // A further 10 px of content shift in the new epoch.
        mock.applyMotion(translation(10, 0));
        estimator.processFrame(frame);

        Pose3D pose = estimator.getCurrentPose();
        assertEquals(-50.0, pose.x, EPS, "the two epochs must sum: 40 + 10 px of content shift");
        assertEquals(12.0, pose.y, EPS);
    }

    @Test
    public void testResetPoseContinuity() {
        MockStitching mock = new MockStitching();
        StitchingEstimator<GrayF32> estimator = newEstimator(mock);
        GrayF32 frame = new GrayF32(100, 100);

        estimator.processFrame(frame);
        mock.applyMotion(translation(15, -25));
        estimator.processFrame(frame);

        Pose3D beforeReset = estimator.getCurrentPose();
        assertEquals(-15.0, beforeReset.x, EPS);
        assertEquals(-25.0, beforeReset.y, EPS);

        mock.processResult = false;   // simulate a stitching failure
        estimator.processFrame(frame);
        assertEquals(1, mock.resetCallCount);

        Pose3D afterReset = estimator.getCurrentPose();
        assertEquals(beforeReset.x, afterReset.x, EPS);
        assertEquals(beforeReset.y, afterReset.y, EPS);
        assertEquals(beforeReset.yaw, afterReset.yaw, EPS);
    }

    /**
     * EXP-VO-004 R0. BoofCV's {@code ImageMotionPointTrackerKey.process} writes the accumulated
     * transform BEFORE {@code StitchingFromMotion2D.checkLargeMotion} can reject the frame, so on a
     * restart the live accumulation already contains the rejected estimate. The fold must use the
     * last GOOD transform; the rejected motion must never reach the pose. Measured on real data:
     * folding the live value moved the logical position 288 px in one frame.
     */
    @Test
    public void testRestartFoldDiscardsTheRejectedFrameMotion() {
        MockStitching mock = new MockStitching();
        StitchingEstimator<GrayF32> estimator = newEstimator(mock);
        GrayF32 frame = new GrayF32(100, 100);

        estimator.processFrame(frame);
        mock.applyMotion(translation(15, -25));
        estimator.processFrame(frame);
        Pose3D beforeFailure = estimator.getCurrentPose();

        // The estimator updates its accumulation with a wild estimate, then the frame is rejected.
        mock.applyMotion(translation(300, 400));
        mock.processResult = false;
        estimator.processFrame(frame);
        assertEquals(1, mock.resetCallCount);

        Pose3D afterRestart = estimator.getCurrentPose();
        assertEquals(beforeFailure.x, afterRestart.x, EPS, "rejected motion must not be folded");
        assertEquals(beforeFailure.y, afterRestart.y, EPS, "rejected motion must not be folded");

        // And subsequent good motion continues from the last good pose, not from the rejected one.
        mock.processResult = true;
        mock.applyMotion(translation(5, 0));
        estimator.processFrame(frame);
        assertEquals(beforeFailure.x - 5.0, estimator.getCurrentPose().x, EPS);
        assertEquals(beforeFailure.y, estimator.getCurrentPose().y, EPS);
    }
}
