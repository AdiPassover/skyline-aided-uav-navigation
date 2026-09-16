package org.boofcv.stitching;

import boofcv.abst.sfm.d2.ImageMotion2D;
import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.struct.image.GrayF32;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.point.Point2D_F64;
import georegression.struct.shapes.Quadrilateral_F64;
import org.boofcv.stitching.metric.HeadingReadoutConfig;
import org.boofcv.stitching.metric.HeadingSemantics;
import org.boofcv.stitching.metric.MetricReadoutConfig;

/**
 * The scripted stand-in for BoofCV's stitching stack, shared by every test that needs to hold the
 * accepted image motion fixed while the <em>canvas schedule</em> (re-origins) or the <em>failure
 * schedule</em> (hard losses) varies.
 *
 * <p>Only a mock can do that: on the real stack a forced failure also changes tracking, so a
 * comparison would measure the tracker rather than the thing under test. Extracted from
 * {@link MetricRuntimeIntegrationTest}, which was its first user, so
 * {@link RecenterVersusRestartTest} and the synthetic hard-loss tests measure the same harness
 * rather than a second copy of it.
 *
 * <p>Evidence tier: T1 (synthetic). No KLT, RANSAC or homography estimation runs here.
 */
public final class ScriptedStitchingHarness {

    public static final int FRAME = 320;
    public static final double FX = 800.0;
    /** gsd at h0 = 80/800 = 0.1 m/px. */
    public static final double H0 = 80.0;
    public static final double DT = 0.1;

    private ScriptedStitchingHarness() {
    }

    /** The accumulated image motion, advanced by the test rather than estimated. */
    public static final class ScriptedMotion implements ImageMotion2D<GrayF32, Homography2D_F64> {
        public final Homography2D_F64 firstToCurrent = new Homography2D_F64();

        @Override public boolean process(GrayF32 input) { return true; }
        @Override public void reset() { firstToCurrent.reset(); }
        @Override public void setToFirst() { firstToCurrent.reset(); }
        @Override public long getFrameID() { return 0; }
        @Override public Homography2D_F64 getFirstToCurrent() { return firstToCurrent; }
        @Override public Class<Homography2D_F64> getTransformType() { return Homography2D_F64.class; }

        public void advance(Homography2D_F64 increment) {
            Homography2D_F64 next = new Homography2D_F64();
            firstToCurrent.concat(increment, next);
            firstToCurrent.setTo(next);
        }
    }

    /**
     * A stitching stack whose re-origin schedule and failure schedule are declared by the test.
     *
     * <p>{@code recenterPeriod > 0} puts the frame corners at the canvas border on every
     * {@code recenterPeriod}-th frame, which is what makes the estimator re-origin; {@code
     * forceFailure} makes {@code process} return false, which is a natural hard loss.
     */
    public static final class ScriptedStitching extends StitchingFromMotion2D<GrayF32, Homography2D_F64> {
        public final ScriptedMotion motion = new ScriptedMotion();
        final GrayF32 stitched = new GrayF32(200, 200);
        public int recenterPeriod;
        public int frame = -1;
        public boolean forceFailure = false;

        @SuppressWarnings("unchecked")
        public ScriptedStitching(int recenterPeriod) {
            super(mock(ImageMotion2D.class), mock(boofcv.alg.distort.ImageDistort.class),
                    mock(boofcv.alg.sfm.d2.StitchingTransform.class), 0.5);
            this.recenterPeriod = recenterPeriod;
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
        @Override public boolean process(GrayF32 image) { frame++; return !forceFailure; }
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
            boolean atBorder = recenterPeriod > 0 && frame > 0 && frame % recenterPeriod == 0;
            double v = atBorder ? 1 : 100;
            return new Quadrilateral_F64(new Point2D_F64(v, v), new Point2D_F64(v + 40, v),
                    new Point2D_F64(v + 40, v + 40), new Point2D_F64(v, v + 40));
        }
    }

    /** Pure image-plane translation of {@code px} pixels east, no rotation, no scale. */
    public static Homography2D_F64 translation(double px) {
        return new Homography2D_F64(1, 0, -px, 0, 1, 0, 0, 0, 1);
    }

    /** Pure image-plane rotation of {@code deg} about the image centre. */
    public static Homography2D_F64 rotation(double deg) {
        double c = Math.cos(Math.toRadians(deg)), sn = Math.sin(Math.toRadians(deg));
        double cx = FRAME / 2.0, cy = FRAME / 2.0;
        return new Homography2D_F64(c, -sn, cx - c * cx + sn * cy,
                                    sn, c, cy - sn * cx - c * cy, 0, 0, 1);
    }

    /** An estimator over {@code stitch}, with the metric readout on or off. */
    public static MotionModelStitchingEstimator<GrayF32, Homography2D_F64> estimator(
            ScriptedStitching stitch, boolean metric) {
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est =
                new MotionModelStitchingEstimator<>(stitch, MotionModelSupport.HOMOGRAPHY);
        est.setShrinkScale(0.5);
        est.setMinDistanceFromBorder(10);
        if (metric) {
            est.enableMetricReadout(new MetricReadoutConfig(FX, 1, H0, 2.0, DT));
        }
        return est;
    }

    /** An estimator with the metric readout on and an external heading owning the direction. */
    public static MotionModelStitchingEstimator<GrayF32, Homography2D_F64> headingEstimator(
            ScriptedStitching stitch, HeadingSemantics semantics, Double deltaMountDeg) {
        var est = estimator(stitch, true);
        est.enableHeadingReadout(new HeadingReadoutConfig(semantics, deltaMountDeg, 2.0, DT,
                HeadingReadoutConfig.DEFAULT_MAX_RATE_DEG_PER_S));
        return est;
    }
}
