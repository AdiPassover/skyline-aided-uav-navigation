package org.boofcv.stitching;

import boofcv.abst.sfm.d2.ImageMotion2D;
import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.struct.image.GrayF32;
import georegression.struct.affine.Affine2D_F64;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.point.Point2D_F64;
import georegression.struct.shapes.Quadrilateral_F64;
import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Estimator-level invariance tests for {@link NavigationSource#RIGID_MOTION} — the properties
 * {@code DEC-VO-004} Alternative E claims of the <i>whole pipeline</i> rather than of the state
 * object: independence from the mosaic canvas schedule, continuity across estimator keyframe
 * changes, immunity to rejected estimates, and indifference to whether anything renders.
 *
 * <p>Two harnesses, deliberately:
 *
 * <ul>
 *   <li><b>A scripted mock</b> for the schedule-invariance and rejected-estimate cases. Only a mock
 *       can hold the <i>accepted motion</i> fixed while the canvas schedule varies — on the real
 *       stack BoofCV welds a track re-anchoring to every canvas re-origin ({@code DEC-VO-003}), so a
 *       different schedule is a different estimator and comparing poses would measure that instead.</li>
 *   <li><b>The real BoofCV stack on synthetic imagery</b> for keyframe continuity and the
 *       renderer-attached comparison, where the point is precisely that the real tracker is
 *       running.</li>
 * </ul>
 *
 * <p>Evidence tier: T1 (synthetic). Says nothing about accuracy on real imagery.
 */
public class RigidMotionEstimatorInvarianceTest {

    private static final int FRAME = 320;
    private static final int WORLD = 1400;

    // ================================================================ scripted mock harness

    /** Motion estimator whose accumulation is advanced by scripted <i>relative</i> increments. */
    private static final class ScriptedMotion implements ImageMotion2D<GrayF32, Homography2D_F64> {
        final Homography2D_F64 firstToCurrent = new Homography2D_F64();
        int setToFirstCount = 0;

        @Override public boolean process(GrayF32 input) { return true; }
        @Override public void reset() { firstToCurrent.reset(); }
        @Override public void setToFirst() { firstToCurrent.reset(); setToFirstCount++; }
        @Override public long getFrameID() { return 0; }
        @Override public Homography2D_F64 getFirstToCurrent() { return firstToCurrent; }
        @Override public Class<Homography2D_F64> getTransformType() { return Homography2D_F64.class; }

        /** Applies one frame of real image motion, whatever the accumulation currently is. */
        void advance(Homography2D_F64 increment) {
            Homography2D_F64 next = new Homography2D_F64();
            firstToCurrent.concat(increment, next);
            firstToCurrent.setTo(next);
        }
    }

    /** Stitcher stub: reports corners that trigger a canvas re-origin on a chosen schedule. */
    private static final class ScriptedStitching extends StitchingFromMotion2D<GrayF32, Homography2D_F64> {
        final ScriptedMotion motion = new ScriptedMotion();
        final GrayF32 stitched = new GrayF32(200, 200);
        int recenterPeriod;                 // 0 = never
        int frame = -1;
        boolean forceFailure = false;

        @SuppressWarnings("unchecked")
        ScriptedStitching(int recenterPeriod) {
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

        @Override
        public boolean process(GrayF32 image) {
            frame++;
            return !forceFailure;
        }

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

        /** Corner positions drive the re-origin: put them at the border on the chosen schedule. */
        @Override
        public Quadrilateral_F64 getImageCorners(int w, int h, Quadrilateral_F64 storage) {
            boolean atBorder = recenterPeriod > 0 && frame > 0 && frame % recenterPeriod == 0;
            double v = atBorder ? 1 : 100;
            return new Quadrilateral_F64(new Point2D_F64(v, v), new Point2D_F64(v + 40, v),
                    new Point2D_F64(v + 40, v + 40), new Point2D_F64(v, v + 40));
        }
    }

    private static Homography2D_F64 increment(int i) {
        double deg = 0.4 * Math.sin(i * 0.11);
        double s = 1.0 + 0.0015 * Math.cos(i * 0.07);
        double c = Math.cos(Math.toRadians(deg)) * s, sn = Math.sin(Math.toRadians(deg)) * s;
        double cx = FRAME / 2.0, cy = FRAME / 2.0;
        // (rotate+scale about the centre) then translate
        double tx = 2.5 + 0.01 * i, ty = -1.5;
        return new Homography2D_F64(
                c, -sn, cx - (c * cx - sn * cy) + tx,
                sn, c, cy - (sn * cx + c * cy) + ty,
                0, 0, 1);
    }

    private static List<Pose3D> runScripted(int frames, int recenterPeriod, NavigationSource source) {
        ScriptedStitching stitch = new ScriptedStitching(recenterPeriod);
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est =
                new MotionModelStitchingEstimator<>(stitch, MotionModelSupport.HOMOGRAPHY);
        est.setShrinkScale(0.5);
        est.setMinDistanceFromBorder(10);
        est.setNavigationSource(source);

        GrayF32 img = new GrayF32(FRAME, FRAME);
        List<Pose3D> poses = new ArrayList<>();
        est.processFrame(img);                       // frame 0: the logical origin
        poses.add(copy(est.getCurrentPose()));
        for (int i = 1; i < frames; i++) {
            stitch.motion.advance(increment(i));     // identical accepted motion in every run
            est.processFrame(img);
            poses.add(copy(est.getCurrentPose()));
        }
        return poses;
    }

    private static Pose3D copy(Pose3D p) { return new Pose3D(p.x, p.y, p.z, p.yaw); }

    // ================================================================ 6. canvas-schedule invariance

    /**
     * <b>The architectural claim.</b> With the accepted motion held identical, the canvas re-origin
     * schedule — a pure rendering parameter — must not move the rigid trajectory. Agreement is
     * asserted to 1e-9 px rather than bitwise: after a re-origin the same increment is obtained as
     * {@code F_k} directly instead of {@code F_k ∘ F_{k-1}⁻¹}, so the two arithmetic paths round
     * differently in the last bits. The measured maximum below is the size of that rounding, not of
     * any schedule dependence.
     */
    @Test
    public void rigidPosesAreInvariantToTheCanvasReoriginSchedule() {
        List<Pose3D> never = runScripted(200, 0, NavigationSource.RIGID_MOTION);
        double worst = 0;
        for (int period : new int[]{3, 11, 37}) {
            List<Pose3D> often = runScripted(200, period, NavigationSource.RIGID_MOTION);
            assertEquals(never.size(), often.size());
            for (int i = 0; i < never.size(); i++) {
                worst = Math.max(worst, Math.abs(never.get(i).x - often.get(i).x));
                worst = Math.max(worst, Math.abs(never.get(i).y - often.get(i).y));
                assertEquals(never.get(i).x, often.get(i).x, 1e-9, "period " + period + " frame " + i);
                assertEquals(never.get(i).y, often.get(i).y, 1e-9, "period " + period + " frame " + i);
                assertEquals(never.get(i).yaw, often.get(i).yaw, 1e-9, "period " + period + " frame " + i);
            }
        }
        System.out.println("[RIGID_MOTION] worst schedule-induced pose difference: " + worst + " px");
        assertTrue(worst < 1e-9, "worst deviation " + worst);
    }

    /** The contrast: the same intervention on the legacy readout does move the trajectory. */
    @Test
    public void theLegacyReadoutIsNotInvariantToTheSameIntervention() {
        List<Pose3D> never = runScripted(200, 0, NavigationSource.MOSAIC_LEGACY);
        List<Pose3D> often = runScripted(200, 11, NavigationSource.MOSAIC_LEGACY);
        double worst = 0;
        for (int i = 0; i < never.size(); i++) {
            worst = Math.max(worst, Math.hypot(never.get(i).x - often.get(i).x,
                    never.get(i).y - often.get(i).y));
        }
        System.out.println("[MOSAIC_LEGACY] worst schedule-induced pose difference: " + worst + " px");
        assertTrue(worst > 1e-6, "expected the legacy readout to depend on the schedule: " + worst);
    }

    // ================================================================ 8. rejected estimates

    /**
     * {@code EXP-VO-004} R0's failure, at the pipeline level: BoofCV writes {@code worldToCurr}
     * before {@code checkLargeMotion} can reject the frame, so the accumulation already holds the
     * rejected estimate when {@code process} returns false. The rigid readout must not integrate it.
     */
    @Test
    public void aRejectedEstimateNeverEntersTheRigidTrajectory() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est =
                new MotionModelStitchingEstimator<>(stitch, MotionModelSupport.HOMOGRAPHY);
        est.setNavigationSource(NavigationSource.RIGID_MOTION);
        GrayF32 img = new GrayF32(FRAME, FRAME);

        est.processFrame(img);
        for (int i = 1; i <= 10; i++) {
            stitch.motion.advance(increment(i));
            est.processFrame(img);
        }
        Pose3D good = copy(est.getCurrentPose());

        // A wild estimate lands in the accumulation, then the stitcher rejects the frame.
        stitch.motion.advance(new Homography2D_F64(1, 0, 900, 0, 1, -1300, 0, 0, 1));
        stitch.forceFailure = true;
        boolean ok = est.processFrame(img);
        stitch.forceFailure = false;

        assertTrue(!ok, "the frame must be reported as failed");
        assertEquals(good.x, est.getCurrentPose().x, 1e-12, "rejected motion must not be integrated");
        assertEquals(good.y, est.getCurrentPose().y, 1e-12);
        assertEquals(good.yaw, est.getCurrentPose().yaw, 1e-12);

        // And the restart leaves the trajectory continuing from the last good pose.
        stitch.motion.advance(increment(12));
        est.processFrame(img);
        Pose3D after = est.getCurrentPose();
        assertTrue(Math.hypot(after.x - good.x, after.y - good.y) < 20,
                "continues from the last good pose, not from the rejected one: " + after.x + "," + after.y);
    }

    // ================================================================ real stack: 7, 9, 10

    private static GrayF32 texturedWorld() {
        GrayF32 w = new GrayF32(WORLD, WORLD);
        Random r = new Random(7);
        for (int y = 0; y < WORLD; y++)
            for (int x = 0; x < WORLD; x++)
                w.set(x, y, r.nextInt(256));
        return w;
    }

    private static GrayF32 frameAt(GrayF32 world, int i) {
        GrayF32 f = new GrayF32(FRAME, FRAME);
        int ox = 20 + i * 6, oy = 20 + i * 4;
        for (int y = 0; y < FRAME; y++)
            for (int x = 0; x < FRAME; x++)
                f.set(x, y, world.get(x + ox, y + oy));
        return f;
    }

    private record RealRun(List<Pose3D> poses, List<Integer> trackCounts, int recenters) {}

    private static RealRun runReal(String model, int frames, boolean attachRenderer) {
        MotionModelStitchingEstimator<GrayF32, ?> est;
        StitchingFromMotion2D<GrayF32, ?> stitch;
        if ("affine".equals(model)) {
            StitchingFromMotion2D<GrayF32, Affine2D_F64> s = StitchingFactory.builder().buildGrayAffine();
            est = new MotionModelStitchingEstimator<>(s, MotionModelSupport.AFFINE);
            stitch = s;
        } else {
            StitchingFromMotion2D<GrayF32, Homography2D_F64> s = StitchingFactory.builder().buildGray();
            est = new MotionModelStitchingEstimator<>(s, MotionModelSupport.HOMOGRAPHY);
            stitch = s;
        }
        est.setShrinkScale(0.5);
        est.setMinDistanceFromBorder(10);
        est.setNavigationSource(NavigationSource.RIGID_MOTION);

        int[] recenters = new int[1];
        if (attachRenderer) {
            est.setRecenterListener(h -> recenters[0]++);
        }

        GrayF32 world = texturedWorld();
        List<Pose3D> poses = new ArrayList<>();
        List<Integer> tracks = new ArrayList<>();
        for (int i = 0; i < frames; i++) {
            GrayF32 f = frameAt(world, i);
            est.processFrame(f);
            if (attachRenderer) {                     // exactly what the GUI does each frame
                Quadrilateral_F64 c = stitch.getImageCorners(f.width, f.height, null);
                GrayF32 mosaic = stitch.getStitchedImage();
                assertTrue(mosaic.width > 0 && c != null);
            }
            poses.add(copy(est.getCurrentPose()));
            Object m = stitch.getMotion();
            tracks.add(m instanceof boofcv.abst.sfm.AccessPointTracks t ? t.getTotalTracks() : -1);
        }
        return new RealRun(poses, tracks, recenters[0]);
    }

    /**
     * Keyframe/respawn continuity on the real tracker. SmartRespawn changes the keyframe whenever
     * track support drops; BoofCV's {@code changeKeyFrame()} leaves the accumulation numerically
     * unchanged, so the rigid readout — which consumes only within-epoch increments — must show no
     * discontinuity there. Asserted as: no per-frame step at a respawn exceeds the largest ordinary
     * step by more than a factor of three.
     */
    @Test
    public void keyframeChangesLeaveTheRigidTrajectoryContinuous() {
        for (String model : new String[]{"homography", "affine"}) {
            RealRun run = runReal(model, 140, false);
            List<Integer> tracks = run.trackCounts();
            double worstRespawn = 0, worstOrdinary = 0;
            int respawns = 0;
            for (int i = 2; i < run.poses().size(); i++) {
                double step = Math.hypot(run.poses().get(i).x - run.poses().get(i - 1).x,
                        run.poses().get(i).y - run.poses().get(i - 1).y);
                boolean respawn = tracks.get(i) > 1.2 * Math.max(1, tracks.get(i - 1));
                if (respawn) {
                    respawns++;
                    worstRespawn = Math.max(worstRespawn, step);
                } else {
                    worstOrdinary = Math.max(worstOrdinary, step);
                }
            }
            assertTrue(respawns > 0, model + ": expected the sequence to trigger respawns");
            System.out.println("[" + model + "] respawns=" + respawns + " worst step at respawn="
                    + worstRespawn + " worst ordinary step=" + worstOrdinary);
            assertTrue(worstRespawn <= 3 * worstOrdinary,
                    model + ": respawn step " + worstRespawn + " vs ordinary " + worstOrdinary);
        }
    }

    /** Attaching the mosaic/GUI consumer must not change the rigid navigation at all. */
    @Test
    public void attachingTheRendererDoesNotChangeRigidNavigation() {
        for (String model : new String[]{"homography", "affine"}) {
            RealRun headless = runReal(model, 120, false);
            RealRun rendered = runReal(model, 120, true);
            assertTrue(rendered.recenters() > 0, model + ": expected canvas re-origins in this run");
            for (int i = 0; i < headless.poses().size(); i++) {
                assertEquals(headless.poses().get(i).x, rendered.poses().get(i).x, 0.0,
                        model + " frame " + i);
                assertEquals(headless.poses().get(i).y, rendered.poses().get(i).y, 0.0);
                assertEquals(headless.poses().get(i).yaw, rendered.poses().get(i).yaw, 0.0);
            }
        }
    }

    /** The diagnostics are populated on the real stack, for both models, and behave as derived. */
    @Test
    public void diagnosticsArePopulatedAndBehaveAsDerivedForBothModels() {
        for (String model : new String[]{"homography", "affine"}) {
            StitchingFromMotion2D<GrayF32, ?> stitch;
            MotionModelStitchingEstimator<GrayF32, ?> est;
            if ("affine".equals(model)) {
                StitchingFromMotion2D<GrayF32, Affine2D_F64> s = StitchingFactory.builder().buildGrayAffine();
                est = new MotionModelStitchingEstimator<>(s, MotionModelSupport.AFFINE);
                stitch = s;
            } else {
                StitchingFromMotion2D<GrayF32, Homography2D_F64> s = StitchingFactory.builder().buildGray();
                est = new MotionModelStitchingEstimator<>(s, MotionModelSupport.HOMOGRAPHY);
                stitch = s;
            }
            est.setNavigationSource(NavigationSource.RIGID_MOTION);
            GrayF32 world = texturedWorld();
            for (int i = 0; i < 60; i++) {
                est.processFrame(frameAt(world, i));
            }
            RigidNavigationState<?> r = est.getRigidNavigation();
            assertTrue(r.hasObservation());
            assertTrue(r.getIncrementScale() > 0.5 && r.getIncrementScale() < 2.0,
                    model + " per-frame scale sane: " + r.getIncrementScale());
            assertTrue(r.getIncrementAnisotropy() >= 1.0, model + " anisotropy >= 1");
            assertEquals(0, r.getImproperIncrementCount(), model + " no mirrored fits expected here");
            if ("affine".equals(model)) {
                assertEquals(0.0, r.getIncrementPerspective(), 0.0,
                        "the affine model has no perspective row, by construction");
            } else {
                assertTrue(r.getIncrementPerspective() >= 0.0);
            }
            // Pure translation of the synthetic world: the accumulated scale must stay near 1.
            assertEquals(1.0, r.accumulatedScale(), 0.05,
                    model + " accumulated scale on a pure-translation sequence: " + r.accumulatedScale());
            assertTrue(stitch.getStitchedImage().width > 0);
        }
    }
}
