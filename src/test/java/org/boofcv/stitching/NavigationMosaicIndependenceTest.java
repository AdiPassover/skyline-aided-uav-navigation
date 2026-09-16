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
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Real-stack tests for {@code DEC-VO-003}: navigation state must be independent of the mosaic, and
 * of whether anything is consuming the mosaic.
 *
 * <p>Unlike {@link LogicalNavigationStateTest}, these run genuine KLT tracking, RANSAC and model
 * estimation on synthetic texture, so they check the wiring rather than the algebra.
 *
 * <p>Evidence tier: T1 (synthetic).
 */
public class NavigationMosaicIndependenceTest {

    private static final int WORLD = 1400;
    private static final int FRAME = 320;

    private static GrayF32 texturedWorld() {
        GrayF32 w = new GrayF32(WORLD, WORLD);
        Random r = new Random(7);
        for (int y = 0; y < WORLD; y++)
            for (int x = 0; x < WORLD; x++)
                w.set(x, y, r.nextInt(256));
        return w;
    }

    private static GrayF32 frameAt(GrayF32 world, int i) {
        int ox = 20 + i * 6, oy = 20 + i * 4;
        GrayF32 f = new GrayF32(FRAME, FRAME);
        for (int y = 0; y < FRAME; y++)
            for (int x = 0; x < FRAME; x++)
                f.set(x, y, world.get(x + ox, y + oy));
        return f;
    }

    // ---------------------------------------------------------------- E / I. GUI consumer

    /**
     * <b>E + I.</b> Attaching a mosaic/GUI consumer must not perturb navigation. One run reads the
     * stitched image, the frame corners and the recenter notifications on every frame exactly as
     * {@code DatasetReplayApp} and {@code NavigationApp} do; the other reads nothing. The logical
     * poses must be <b>bitwise identical</b>.
     */
    @Test
    public void attachingAMosaicConsumerDoesNotChangeNavigation() {
        List<Pose3D> withoutGui = runHomography(120, false);
        List<Pose3D> withGui = runHomography(120, true);

        assertEquals(withoutGui.size(), withGui.size());
        for (int i = 0; i < withoutGui.size(); i++) {
            assertEquals(withoutGui.get(i).x, withGui.get(i).x, 0.0, "pose.x differs at frame " + i);
            assertEquals(withoutGui.get(i).y, withGui.get(i).y, 0.0, "pose.y differs at frame " + i);
            assertEquals(withoutGui.get(i).yaw, withGui.get(i).yaw, 0.0, "pose.yaw differs at frame " + i);
        }
    }

    private static List<Pose3D> runHomography(int frames, boolean simulateGuiConsumer) {
        StitchingFromMotion2D<GrayF32, Homography2D_F64> stitch =
                StitchingFactory.builder().buildGray();
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est =
                new MotionModelStitchingEstimator<>(stitch, MotionModelSupport.HOMOGRAPHY);
        est.setShrinkScale(0.5);
        est.setMinDistanceFromBorder(10);
        est.setNavigationSource(NavigationSource.LOGICAL_FRAME);   // the path under test

        int[] recenterCount = new int[1];
        if (simulateGuiConsumer) {
            est.setRecenterListener(h -> recenterCount[0]++);
        }

        GrayF32 world = texturedWorld();
        List<Pose3D> poses = new ArrayList<>();
        for (int i = 0; i < frames; i++) {
            GrayF32 f = frameAt(world, i);
            est.processFrame(f);
            if (simulateGuiConsumer) {
                // Exactly what the GUI does each frame.
                Quadrilateral_F64 c = stitch.getImageCorners(f.width, f.height, null);
                GrayF32 mosaic = stitch.getStitchedImage();
                assertTrue(mosaic.width > 0 && c != null);
            }
            Pose3D p = est.getCurrentPose();
            poses.add(new Pose3D(p.x, p.y, p.z, p.yaw));
        }
        if (simulateGuiConsumer) {
            assertTrue(recenterCount[0] > 0, "expected the GUI run to observe canvas re-origins");
        }
        return poses;
    }

    // ---------------------------------------------------------------- G. pre-re-origin equivalence

    /**
     * <b>G. Pre-first-re-origin equivalence with the previous implementation.</b>
     *
     * <p>Before the first canvas re-origin the old mosaic-derived readout and the new logical readout
     * are the <i>same quantity</i>, and this asserts it rather than assuming it. Algebraically, with
     * {@code S} the shrink transform (a uniform scale plus translation) and {@code F} the accumulated
     * motion, the legacy readout was
     * {@code (centroid(S(F⁻¹(corners))) − centroid(S(corners)))/shrinkScale}; because {@code S} is
     * affine it commutes with the centroid, and its scale cancels against the division, leaving
     * exactly the displacement of the <b>logical footprint centroid</b>.
     *
     * <p>Two separate facts are checked, because conflating them is what made the first version of
     * this test misleading:
     *
     * <ol>
     *   <li>legacy canvas readout ≡ logical footprint <b>centroid</b> displacement — for both models,
     *       to float32 precision. BoofCV computes canvas corners through a
     *       {@code PixelTransform<Point2D_F32>} ({@code StitchingFromMotion2D:82,299}), so the legacy
     *       path carries single-precision rounding while the logical path is double throughout; the
     *       tolerance below exists for that reason and for no other.</li>
     *   <li>logical footprint <b>centroid</b> vs the transformed <b>centre</b> — identical for affine,
     *       measurably different for a projective homography. This is the deliberate change of
     *       quantity, and it is measured and reported rather than bounded by an arbitrary constant.</li>
     * </ol>
     */
    @Test
    public void legacyCanvasReadoutEqualsLogicalFootprintCentroid_affine() {
        StitchingFromMotion2D<GrayF32, Affine2D_F64> stitch =
                StitchingFactory.builder().buildGrayAffine();
        MotionModelStitchingEstimator<GrayF32, Affine2D_F64> est =
                new MotionModelStitchingEstimator<>(stitch, MotionModelSupport.AFFINE);
        double shrink = 0.5;
        est.setShrinkScale(shrink);
        est.setMinDistanceFromBorder(10);
        est.setNavigationSource(NavigationSource.LOGICAL_FRAME);   // the path under test
        int[] recenters = new int[1];
        est.setRecenterListener(h -> recenters[0]++);

        GrayF32 world = texturedWorld();
        Point2D_F64 canvasRef = null;
        double worstCentroid = 0, worstCentreVsCentroid = 0;
        int compared = 0;

        for (int i = 0; i < 120 && recenters[0] == 0; i++) {
            GrayF32 f = frameAt(world, i);
            est.processFrame(f);
            if (recenters[0] > 0) break;

            Point2D_F64 canvasCentroid = centroid(stitch.getImageCorners(f.width, f.height, null));
            if (canvasRef == null) { canvasRef = canvasCentroid; continue; }

            Point2D_F64 logicalCentroid = centroid(est.getLogicalFootprint());
            double legacyX = (canvasCentroid.x - canvasRef.x) / shrink;
            double legacyY = (canvasCentroid.y - canvasRef.y) / shrink;
            worstCentroid = Math.max(worstCentroid, Math.hypot(
                    legacyX - (logicalCentroid.x - FRAME / 2.0),
                    legacyY - (logicalCentroid.y - FRAME / 2.0)));

            Pose3D p = est.getCurrentPose();
            worstCentreVsCentroid = Math.max(worstCentreVsCentroid, Math.hypot(
                    (logicalCentroid.x - FRAME / 2.0) - p.x,
                    (logicalCentroid.y - FRAME / 2.0) - (-p.y)));
            compared++;
        }

        assertTrue(compared >= 10, "needed at least 10 pre-re-origin frames; got " + compared);
        assertTrue(worstCentroid < 1e-3,
                "legacy canvas readout should equal the logical footprint centroid to float32 "
                        + "precision; worst gap " + worstCentroid + " px");
        assertEquals(0.0, worstCentreVsCentroid, 1e-9,
                "affine must preserve centroids, so centre and centroid readouts must coincide exactly");
        System.out.printf("[navsep] affine: legacy-vs-logical-centroid %.2e px, "
                + "centroid-vs-centre %.2e px over %d frames%n",
                worstCentroid, worstCentreVsCentroid, compared);
    }

    @Test
    public void legacyCanvasReadoutEqualsLogicalFootprintCentroid_homography() {
        StitchingFromMotion2D<GrayF32, Homography2D_F64> stitch =
                StitchingFactory.builder().buildGray();
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est =
                new MotionModelStitchingEstimator<>(stitch, MotionModelSupport.HOMOGRAPHY);
        double shrink = 0.5;
        est.setShrinkScale(shrink);
        est.setMinDistanceFromBorder(10);
        est.setNavigationSource(NavigationSource.LOGICAL_FRAME);   // the path under test
        int[] recenters = new int[1];
        est.setRecenterListener(h -> recenters[0]++);

        GrayF32 world = texturedWorld();
        Point2D_F64 canvasRef = null;
        double worstCentroid = 0, worstCentreVsCentroid = 0;
        int compared = 0;

        for (int i = 0; i < 120 && recenters[0] == 0; i++) {
            GrayF32 f = frameAt(world, i);
            est.processFrame(f);
            if (recenters[0] > 0) break;

            Point2D_F64 canvasCentroid = centroid(stitch.getImageCorners(f.width, f.height, null));
            if (canvasRef == null) { canvasRef = canvasCentroid; continue; }

            Point2D_F64 logicalCentroid = centroid(est.getLogicalFootprint());
            double legacyX = (canvasCentroid.x - canvasRef.x) / shrink;
            double legacyY = (canvasCentroid.y - canvasRef.y) / shrink;
            worstCentroid = Math.max(worstCentroid, Math.hypot(
                    legacyX - (logicalCentroid.x - FRAME / 2.0),
                    legacyY - (logicalCentroid.y - FRAME / 2.0)));

            Pose3D p = est.getCurrentPose();
            worstCentreVsCentroid = Math.max(worstCentreVsCentroid, Math.hypot(
                    (logicalCentroid.x - FRAME / 2.0) - p.x,
                    (logicalCentroid.y - FRAME / 2.0) - (-p.y)));
            compared++;
        }

        assertTrue(compared >= 10, "needed at least 10 pre-re-origin frames; got " + compared);
        assertTrue(worstCentroid < 1e-3,
                "legacy canvas readout should equal the logical footprint centroid to float32 "
                        + "precision even under homography; worst gap " + worstCentroid + " px");
        // Measured and reported, not bounded by an invented constant: this is the deliberate
        // change of quantity, and how large it is depends entirely on the perspective content.
        System.out.printf("[navsep] homography: legacy-vs-logical-centroid %.2e px, "
                + "centroid-vs-centre %.4f px over %d frames%n",
                worstCentroid, worstCentreVsCentroid, compared);
    }

    private static Point2D_F64 centroid(Quadrilateral_F64 q) {
        return new Point2D_F64((q.a.x + q.b.x + q.c.x + q.d.x) / 4.0,
                               (q.a.y + q.b.y + q.c.y + q.d.y) / 4.0);
    }

    private static double legacyYaw(Quadrilateral_F64 q) {
        double angle = Math.toDegrees(Math.atan2(q.b.y - q.a.y, q.b.x - q.a.x));
        return (angle + 360.0) % 360.0;
    }

    // ---------------------------------------------------------------- headless capability

    /**
     * <b>Question 7: can navigation run with no mosaic at all?</b> Yes — this run never constructs a
     * {@code StitchingFromMotion2D}, allocates no canvas and renders nothing, and still produces a
     * continuous logical pose from real tracking.
     *
     * <p>It deliberately does <b>not</b> assert equality with the mosaic-backed run: without canvas
     * re-origins there are no {@code setToFirst()} track re-anchorings, so the two are different
     * estimators and diverge over time ({@code MotionOnlyNavigationEstimator} documents this). The
     * assertion is that headless operation works and stays continuous, and the divergence is
     * reported rather than hidden.
     */
    @Test
    public void navigationRunsWithNoMosaicAtAll() {
        ImageMotion2D<GrayF32, Homography2D_F64> motion =
                StitchingFactory.builder().buildGrayMotionOnly();
        MotionOnlyNavigationEstimator<GrayF32, Homography2D_F64> headless =
                new MotionOnlyNavigationEstimator<>(motion, MotionModelSupport.HOMOGRAPHY);

        GrayF32 world = texturedWorld();
        List<Pose3D> poses = new ArrayList<>();
        for (int i = 0; i < 120; i++) {
            headless.processFrame(frameAt(world, i));
            Pose3D p = headless.getCurrentPose();
            poses.add(new Pose3D(p.x, p.y, p.z, p.yaw));
        }

        // Continuous and non-degenerate.
        for (int i = 1; i < poses.size(); i++) {
            double step = Math.hypot(poses.get(i).x - poses.get(i - 1).x,
                                     poses.get(i).y - poses.get(i - 1).y);
            assertTrue(step < 60.0, "implausible pose jump at frame " + i + ": " + step + " px");
        }
        Pose3D last = poses.get(poses.size() - 1);
        assertTrue(Math.hypot(last.x, last.y) > 100.0,
                "headless run produced no motion; final |pose| = " + Math.hypot(last.x, last.y));

        List<Pose3D> mosaicBacked = runHomography(120, false);
        double endpointGap = Math.hypot(last.x - mosaicBacked.get(119).x,
                                        last.y - mosaicBacked.get(119).y);
        System.out.printf("[navsep] headless vs mosaic-backed endpoint gap after 120 frames: %.3f px "
                + "(expected non-zero: different keyframe schedules)%n", endpointGap);
    }

    /** The headless path must work for the affine model too. */
    @Test
    public void headlessNavigationWorksForAffineAsWell() {
        ImageMotion2D<GrayF32, Affine2D_F64> motion =
                StitchingFactory.builder().buildGrayAffineMotionOnly();
        MotionOnlyNavigationEstimator<GrayF32, Affine2D_F64> headless =
                new MotionOnlyNavigationEstimator<>(motion, MotionModelSupport.AFFINE);
        assertEquals("affine", headless.motionModelId());

        GrayF32 world = texturedWorld();
        for (int i = 0; i < 120; i++) {
            headless.processFrame(frameAt(world, i));
        }
        Pose3D last = headless.getCurrentPose();
        assertTrue(Math.hypot(last.x, last.y) > 100.0,
                "affine headless run produced no motion; final |pose| = " + Math.hypot(last.x, last.y));
        assertNotEquals(0.0, headless.getLogicalFootprint().b.x - headless.getLogicalFootprint().a.x,
                "footprint should be non-degenerate");
    }
}
