package org.boofcv.stitching;

import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.struct.image.GrayF32;
import georegression.struct.affine.Affine2D_F64;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.point.Point2D_F64;
import georegression.struct.shapes.Quadrilateral_F64;
import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * <b>Phase-2 diagnostic for the navigation/mosaic reference-frame investigation.</b>
 *
 * <p>Establishes, on the real BoofCV stack rather than by reading source alone, which of the three
 * distinct events the codebase loosely calls "recentering" actually perturbs the estimator's
 * accumulated motion, and therefore whether logical navigation state is coupled to mosaic canvas
 * bookkeeping.
 *
 * <p>The three events must be kept apart (they are conflated in the existing code and docs):
 *
 * <ul>
 *   <li><b>(A) estimator keyframe change</b> — {@code ImageMotionPtkSmartRespawn} calls
 *       {@code ImageMotionPointTrackerKey.changeKeyFrame()} for tracking robustness. Source
 *       (BoofCV 0.44) shows this does {@code worldToKey.setTo(worldToCurr); keyToCurr.reset()},
 *       which leaves {@code worldToCurr} — i.e. {@code getFirstToCurrent()} — <i>numerically
 *       unchanged</i>.</li>
 *   <li><b>(B) mosaic canvas-origin change</b> — {@code StitchingFromMotion2D.setOriginToCurrent()}
 *       re-renders the canvas <i>and</i> calls {@code motion.setToFirst()}, which is
 *       {@code changeKeyFrame(); resetTransforms()}. {@code resetTransforms()} zeroes
 *       {@code worldToCurr}. This is a canvas operation that destroys estimator accumulation.</li>
 *   <li><b>(C) logical navigation-coordinate change</b> — in the current architecture this is
 *       <i>forced</i> by (B), and re-anchored by the wrapper's centroid/yaw bookkeeping.</li>
 * </ul>
 *
 * <p>These tests assert the (A)-vs-(B) distinction directly, which is the load-bearing fact for the
 * proposed separation: (A) needs no compensation at all, (B) needs exact composition.
 *
 * <p>Evidence tier: T1 (synthetic). Says nothing about VO accuracy.
 */
public class NavigationFrameCouplingDiagnosticTest {

    private static final int WORLD = 1400;
    private static final int FRAME = 320;
    private static final Path REPORT_DIR = Paths.get(System.getProperty(
            "expvo003.report.dir", System.getProperty("java.io.tmpdir")));

    // ---------------------------------------------------------------- synthetic imagery

    private static GrayF32 texturedWorld(long seed) {
        GrayF32 w = new GrayF32(WORLD, WORLD);
        Random r = new Random(seed);
        for (int y = 0; y < WORLD; y++)
            for (int x = 0; x < WORLD; x++)
                w.set(x, y, r.nextInt(256));
        return w;
    }

    /** Pure translation along a diagonal; guaranteed to walk the frame toward the canvas border. */
    private static GrayF32 translatingFrame(GrayF32 world, int i) {
        return crop(world, 20 + i * 6, 20 + i * 4);
    }

    private static GrayF32 crop(GrayF32 world, int ox, int oy) {
        GrayF32 f = new GrayF32(FRAME, FRAME);
        for (int y = 0; y < FRAME; y++)
            for (int x = 0; x < FRAME; x++)
                f.set(x, y, world.get(x + ox, y + oy));
        return f;
    }

    // ---------------------------------------------------------------- probe

    /** One frame's observable state, from public API only. */
    private record Sample(int frame, boolean success, boolean recentered,
                          double poseX, double poseY, double poseYaw,
                          double firstToCurrTx, double firstToCurrTy, boolean firstToCurrIsIdentity,
                          double cornerCentroidX, double cornerCentroidY, int trackCount) {}

    private static boolean isIdentity(Homography2D_F64 h) {
        return Math.abs(h.a11 - 1) < 1e-12 && Math.abs(h.a22 - 1) < 1e-12
                && Math.abs(h.a33 - 1) < 1e-12 && Math.abs(h.a12) < 1e-12
                && Math.abs(h.a13) < 1e-12 && Math.abs(h.a21) < 1e-12
                && Math.abs(h.a23) < 1e-12 && Math.abs(h.a31) < 1e-12 && Math.abs(h.a32) < 1e-12;
    }

    private static List<Sample> runHomography(int frameCount, int minDistanceFromBorder) {
        StitchingFromMotion2D<GrayF32, Homography2D_F64> stitch =
                StitchingFactory.builder().buildGray();
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est =
                new MotionModelStitchingEstimator<>(stitch, MotionModelSupport.HOMOGRAPHY);
        est.setShrinkScale(0.5);
        est.setMinDistanceFromBorder(minDistanceFromBorder);
        // This is a diagnostic OF the legacy mosaic-coupled readout -- the pose-continuity and
        // recenter-sensitivity assertions below are statements about MOSAIC_LEGACY specifically.
        // Pinned explicitly by DEC-VO-008: RIGID_MOTION is canvas-invariant by construction, so
        // under the new default these assertions would still pass while measuring nothing.
        est.setNavigationSource(NavigationSource.MOSAIC_LEGACY);

        boolean[] recentered = new boolean[1];
        est.setRecenterListener(h -> recentered[0] = true);

        GrayF32 world = texturedWorld(7);
        List<Sample> out = new ArrayList<>();
        for (int i = 0; i < frameCount; i++) {
            recentered[0] = false;
            boolean ok = est.processFrame(translatingFrame(world, i));
            Pose3D p = est.getCurrentPose();
            Homography2D_F64 f = stitch.getMotion().getFirstToCurrent();
            Quadrilateral_F64 c = stitch.getImageCorners(FRAME, FRAME, null);
            double cx = (c.a.x + c.b.x + c.c.x + c.d.x) / 4.0;
            double cy = (c.a.y + c.b.y + c.c.y + c.d.y) / 4.0;
            int tracks = trackCount(stitch);
            out.add(new Sample(i, ok, recentered[0], p.x, p.y, p.yaw,
                    f.a13, f.a23, isIdentity(f), cx, cy, tracks));
        }
        return out;
    }

    private static int trackCount(StitchingFromMotion2D<GrayF32, ?> stitch) {
        Object m = stitch.getMotion();
        if (m instanceof boofcv.abst.sfm.AccessPointTracks t) return t.getTotalTracks();
        return -1;
    }

    // ---------------------------------------------------------------- tests

    /**
     * <b>The central Phase-2 finding.</b> {@code getFirstToCurrent()} — the estimator's own
     * accumulated first-frame-to-current-frame transform — is reset to identity <i>exactly</i> at
     * mosaic-canvas re-origin events, and at no other time. In particular it survives every
     * SmartRespawn keyframe change, of which this sequence contains several (evidenced by the track
     * count jumping back toward the cap without any recenter being reported).
     */
    @Test
    public void accumulatedMotionIsDestroyedOnlyByCanvasReorigin() throws IOException {
        List<Sample> s = runHomography(140, 10);

        List<Integer> recenterFrames = new ArrayList<>();
        List<Integer> identityFrames = new ArrayList<>();
        for (Sample x : s) {
            if (x.recentered()) recenterFrames.add(x.frame());
            // frame 0 is the estimator's own initialisation, not a canvas re-origin
            if (x.firstToCurrIsIdentity() && x.frame() > 0) identityFrames.add(x.frame());
        }

        assertTrue(recenterFrames.size() >= 2,
                "diagnostic needs at least two canvas re-origin events; got " + recenterFrames);

        // Every frame where the accumulated transform reads identity must be a recenter frame.
        assertEquals(recenterFrames, identityFrames,
                "getFirstToCurrent() went to identity on frames that were NOT canvas re-origins, "
                        + "or survived a canvas re-origin -- either would contradict the source trace");

        writeReport("phase2_homography.csv", s);
    }

    /**
     * The wrapper's own pose is <i>not</i> reset at those frames — it stays continuous — which shows
     * the existing architecture is already compensating for (B) by hand. The question the refactor
     * answers is whether that hand-compensation is lossless, not whether it exists.
     */
    @Test
    public void wrapperPoseIsContinuousAcrossCanvasReorigin() {
        List<Sample> s = runHomography(140, 10);
        for (int i = 1; i < s.size(); i++) {
            Sample prev = s.get(i - 1), cur = s.get(i);
            if (!cur.recentered()) continue;
            double jump = Math.hypot(cur.poseX() - prev.poseX(), cur.poseY() - prev.poseY());
            // per-frame motion in this sequence is ~7 px of image translation; a reference reset
            // leaking into the pose would show up as a jump far larger than that.
            assertTrue(jump < 50.0,
                    "pose jumped " + jump + " px at the canvas re-origin on frame " + cur.frame()
                            + " -- the wrapper's re-anchoring failed outright");
        }
    }

    /**
     * The canvas-management knob {@code minDistanceFromBorder} has no physical meaning for where the
     * UAV is — it only decides when the finite raster needs re-originning. Changing it nevertheless
     * changes the reported navigation trajectory.
     *
     * <p><b>This is deliberately reported, not asserted as a clean rendering-only intervention.</b>
     * Changing the re-origin schedule also changes when {@code setToFirst()} fires, which changes
     * track re-anchoring and therefore the RANSAC inputs on subsequent frames. So the difference
     * measured here is an <i>upper bound</i> that mixes the coupling under investigation with a
     * genuine estimator-state difference. The clean separation is impossible in the current
     * architecture precisely because BoofCV welds the two into one call — which is the finding.
     */
    @Test
    public void canvasScheduleChangesTheReportedTrajectory() throws IOException {
        List<Sample> tight = runHomography(140, 10);
        List<Sample> loose = runHomography(140, 60);

        long tightRecenters = tight.stream().filter(Sample::recentered).count();
        long looseRecenters = loose.stream().filter(Sample::recentered).count();
        assertTrue(tightRecenters != looseRecenters,
                "the two canvas schedules produced the same number of re-origins ("
                        + tightRecenters + "); the diagnostic is vacuous");

        Sample tEnd = tight.get(tight.size() - 1), lEnd = loose.get(loose.size() - 1);
        double endDiff = Math.hypot(tEnd.poseX() - lEnd.poseX(), tEnd.poseY() - lEnd.poseY());

        StringBuilder sb = new StringBuilder();
        sb.append("minDistanceFromBorder,recenters,final_x,final_y,final_yaw\n");
        sb.append("10,").append(tightRecenters).append(',').append(tEnd.poseX()).append(',')
          .append(tEnd.poseY()).append(',').append(tEnd.poseYaw()).append('\n');
        sb.append("60,").append(looseRecenters).append(',').append(lEnd.poseX()).append(',')
          .append(lEnd.poseY()).append(',').append(lEnd.poseYaw()).append('\n');
        sb.append("endpoint_difference_px,").append(endDiff).append('\n');
        Files.createDirectories(REPORT_DIR);
        Files.writeString(REPORT_DIR.resolve("phase2_canvas_schedule.csv"), sb.toString());

        // Recorded, not asserted in a direction: the point is that a canvas-only knob moves it at all.
        System.out.println("[phase2] canvas schedule endpoint difference: " + endDiff + " px");
    }

    /** The same reference-reset structure must hold for the affine model, not just homography. */
    @Test
    public void affineShowsTheSameReferenceResetStructure() {
        StitchingFromMotion2D<GrayF32, Affine2D_F64> stitch =
                StitchingFactory.builder().buildGrayAffine();
        MotionModelStitchingEstimator<GrayF32, Affine2D_F64> est =
                new MotionModelStitchingEstimator<>(stitch, MotionModelSupport.AFFINE);
        est.setShrinkScale(0.5);
        est.setMinDistanceFromBorder(10);
        est.setNavigationSource(NavigationSource.MOSAIC_LEGACY);   // see runHomography, DEC-VO-008
        boolean[] recentered = new boolean[1];
        est.setRecenterListener(h -> recentered[0] = true);

        GrayF32 world = texturedWorld(7);
        int recenters = 0, identities = 0, matched = 0;
        for (int i = 0; i < 140; i++) {
            recentered[0] = false;
            est.processFrame(translatingFrame(world, i));
            Affine2D_F64 f = stitch.getMotion().getFirstToCurrent();
            boolean ident = Math.abs(f.a11 - 1) < 1e-12 && Math.abs(f.a22 - 1) < 1e-12
                    && Math.abs(f.a12) < 1e-12 && Math.abs(f.a21) < 1e-12
                    && Math.abs(f.tx) < 1e-12 && Math.abs(f.ty) < 1e-12;
            if (recentered[0]) recenters++;
            if (ident && i > 0) identities++;
            if (recentered[0] && ident && i > 0) matched++;
        }
        assertTrue(recenters >= 2, "affine diagnostic needs recenters; got " + recenters);
        assertEquals(recenters, identities, "affine: identity frames != recenter frames");
        assertEquals(recenters, matched, "affine: identity and recenter frames did not coincide");
    }

    /**
     * <b>The (A) half of the finding, demonstrated positively on the real stack.</b> With canvas
     * re-origin disabled (a negative border margin makes {@code nearBorder} unsatisfiable), the
     * sequence still drives SmartRespawn to change keyframes for tracking robustness — visible as
     * the track count recovering toward the cap. Across every one of those keyframe changes the
     * accumulated {@code getFirstToCurrent()} keeps growing and never returns to identity.
     *
     * <p>So an estimator keyframe change is <i>already</i> composed correctly by BoofCV and needs no
     * compensation from us; only the canvas re-origin destroys accumulation. That asymmetry is what
     * makes a clean separation possible.
     */
    @Test
    public void respawnKeyframeChangesPreserveAccumulatedMotion() throws IOException {
        // Negative margin => nearBorder() can never be true => setOriginToCurrent() is never called.
        List<Sample> s = runHomography(160, -10_000);

        long recenters = s.stream().filter(Sample::recentered).count();
        assertEquals(0, recenters, "canvas re-origin should be disabled in this run");

        // Track count recovering after a decline is the observable signature of a respawn.
        int respawnLike = 0;
        for (int i = 1; i < s.size(); i++)
            if (s.get(i).trackCount() > s.get(i - 1).trackCount() + 20) respawnLike++;
        assertTrue(respawnLike >= 1,
                "expected at least one SmartRespawn keyframe change; track count never recovered, "
                        + "so this run does not exercise case (A)");

        // The accumulated transform must never collapse to identity after frame 0.
        for (Sample x : s)
            assertTrue(x.frame() == 0 || !x.firstToCurrIsIdentity(),
                    "accumulated motion was reset on frame " + x.frame() + " with no canvas re-origin");

        // ...and must grow monotonically in magnitude for this monotone translation sequence.
        double last = 0;
        for (Sample x : s) {
            double mag = Math.hypot(x.firstToCurrTx(), x.firstToCurrTy());
            assertTrue(mag >= last - 1e-6,
                    "accumulated translation shrank at frame " + x.frame() + " (" + last + " -> " + mag + ")");
            last = mag;
        }
        System.out.println("[phase2] no-reorigin run: " + respawnLike
                + " respawn-like track recoveries, final |f2c| = " + last);
        writeReport("phase2_no_reorigin.csv", s);
    }

    /**
     * <b>The single-point failure mode, shown analytically.</b> Taking any fixed image point other
     * than the rotation centre as "the UAV position" manufactures translation out of pure rotation.
     *
     * <p>This is a statement about transform algebra, not about BoofCV, so it is tested directly on
     * the transform rather than through imagery: a rotation about the image centre leaves the centre
     * fixed and moves every other point by up to the frame half-diagonal.
     */
    @Test
    public void arbitraryImagePointFabricatesTranslationUnderPureRotation() {
        double cx = FRAME / 2.0, cy = FRAME / 2.0;
        double theta = Math.toRadians(30);
        double c = Math.cos(theta), sn = Math.sin(theta);
        // Rotation about the image centre, expressed as an affine map.
        Affine2D_F64 rotAboutCentre = new Affine2D_F64(
                c, -sn, sn, c, cx - (c * cx - sn * cy), cy - (sn * cx + c * cy));

        Point2D_F64 centre = apply(rotAboutCentre, cx, cy);
        Point2D_F64 origin = apply(rotAboutCentre, 0, 0);
        Point2D_F64 corner = apply(rotAboutCentre, FRAME, 0);

        double centreShift = Math.hypot(centre.x - cx, centre.y - cy);
        double originShift = Math.hypot(origin.x - 0, origin.y - 0);
        double cornerShift = Math.hypot(corner.x - FRAME, corner.y - 0);

        assertTrue(centreShift < 1e-9,
                "the image centre must be fixed by a rotation about itself; moved " + centreShift);
        assertTrue(originShift > 80, "expected a large spurious shift at the image origin, got " + originShift);
        assertTrue(cornerShift > 80, "expected a large spurious shift at a corner, got " + cornerShift);

        System.out.printf("[phase2] pure 30deg rotation about centre: centre moves %.6f px, "
                + "image origin moves %.2f px, corner moves %.2f px%n",
                centreShift, originShift, cornerShift);
    }

    /**
     * For an affine model the centroid of the transformed corners equals the transform of the
     * centre, so a corner-centroid readout and a centre readout agree exactly. For a projective
     * homography they do <i>not</i>, because the perspective divide is not affine-invariant. This
     * pins the difference so the chosen readout is a deliberate decision rather than an accident.
     */
    @Test
    public void cornerCentroidEqualsCentreForAffineButNotForHomography() {
        double w = FRAME, h = FRAME;
        Affine2D_F64 aff = new Affine2D_F64(1.3, 0.2, -0.1, 0.9, 15, -7);
        Point2D_F64 affCentroid = centroidOfCorners(p -> apply(aff, p.x, p.y), w, h);
        Point2D_F64 affCentre = apply(aff, w / 2, h / 2);
        assertEquals(affCentre.x, affCentroid.x, 1e-9, "affine: centroid != centre in x");
        assertEquals(affCentre.y, affCentroid.y, 1e-9, "affine: centroid != centre in y");

        Homography2D_F64 hom = new Homography2D_F64(1.3, 0.2, 15, -0.1, 0.9, -7, 0.0008, 0.0005, 1);
        Point2D_F64 homCentroid = centroidOfCorners(p -> applyH(hom, p.x, p.y), w, h);
        Point2D_F64 homCentre = applyH(hom, w / 2, h / 2);
        double gap = Math.hypot(homCentre.x - homCentroid.x, homCentre.y - homCentroid.y);
        assertTrue(gap > 1.0,
                "expected a measurable centroid-vs-centre gap under perspective; got " + gap + " px");
        System.out.printf("[phase2] homography centroid-vs-centre gap: %.3f px%n", gap);
    }

    private interface PtMap { Point2D_F64 map(Point2D_F64 p); }

    private static Point2D_F64 centroidOfCorners(PtMap f, double w, double h) {
        Point2D_F64[] c = {
                f.map(new Point2D_F64(0, 0)), f.map(new Point2D_F64(w, 0)),
                f.map(new Point2D_F64(w, h)), f.map(new Point2D_F64(0, h))};
        double x = 0, y = 0;
        for (Point2D_F64 p : c) { x += p.x; y += p.y; }
        return new Point2D_F64(x / 4, y / 4);
    }

    private static Point2D_F64 apply(Affine2D_F64 a, double x, double y) {
        return new Point2D_F64(a.a11 * x + a.a12 * y + a.tx, a.a21 * x + a.a22 * y + a.ty);
    }

    private static Point2D_F64 applyH(Homography2D_F64 hh, double x, double y) {
        double z = hh.a31 * x + hh.a32 * y + hh.a33;
        return new Point2D_F64((hh.a11 * x + hh.a12 * y + hh.a13) / z,
                               (hh.a21 * x + hh.a22 * y + hh.a23) / z);
    }

    private static void writeReport(String name, List<Sample> s) throws IOException {
        StringBuilder sb = new StringBuilder(
                "frame,success,recentered,pose_x,pose_y,pose_yaw,f2c_tx,f2c_ty,f2c_identity,"
                        + "corner_cx,corner_cy,track_count\n");
        for (Sample x : s) {
            sb.append(x.frame()).append(',').append(x.success()).append(',').append(x.recentered())
              .append(',').append(x.poseX()).append(',').append(x.poseY()).append(',').append(x.poseYaw())
              .append(',').append(x.firstToCurrTx()).append(',').append(x.firstToCurrTy())
              .append(',').append(x.firstToCurrIsIdentity()).append(',')
              .append(x.cornerCentroidX()).append(',').append(x.cornerCentroidY())
              .append(',').append(x.trackCount()).append('\n');
        }
        Files.createDirectories(REPORT_DIR);
        Files.writeString(REPORT_DIR.resolve(name), sb.toString());
    }
}
