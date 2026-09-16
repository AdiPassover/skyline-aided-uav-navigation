package org.boofcv.stitching;

import georegression.struct.InvertibleTransform;
import georegression.struct.affine.Affine2D_F64;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.point.Point2D_F64;
import georegression.struct.shapes.Quadrilateral_F64;
import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Known-answer and invariance tests for {@link LogicalNavigationState} ({@code DEC-VO-003}).
 *
 * <p>These are exact algebraic tests — no imagery, no tracking, no RANSAC — so every assertion here
 * is a statement about the reference-frame composition itself rather than about estimation quality.
 * Tolerances are at floating-point noise, deliberately: the composition is either right or it is not.
 *
 * <p>Every property is checked for <b>both</b> supported motion models, because a navigation
 * abstraction that happened to be correct only for one transform family would be a silent trap
 * ({@code DEC-VO-003} constraint 8).
 *
 * <p>Evidence tier: T1 (analytic).
 */
public class LogicalNavigationStateTest {

    private static final int W = 320, H = 240;
    private static final double EXACT = 1e-9;

    // ---------------------------------------------------------------- transform builders

    private static Homography2D_F64 hTranslate(double dx, double dy) {
        return new Homography2D_F64(1, 0, dx, 0, 1, dy, 0, 0, 1);
    }

    private static Homography2D_F64 hRotateAboutCentre(double deg) {
        double r = Math.toRadians(deg), c = Math.cos(r), s = Math.sin(r);
        double cx = W / 2.0, cy = H / 2.0;
        return new Homography2D_F64(c, -s, cx - (c * cx - s * cy), s, c, cy - (s * cx + c * cy), 0, 0, 1);
    }

    private static Homography2D_F64 hScaleAboutCentre(double k) {
        double cx = W / 2.0, cy = H / 2.0;
        return new Homography2D_F64(k, 0, cx * (1 - k), 0, k, cy * (1 - k), 0, 0, 1);
    }

    private static Homography2D_F64 hShear(double sh) {
        return new Homography2D_F64(1, sh, 0, 0, 1, 0, 0, 0, 1);
    }

    private static Homography2D_F64 hProjective(double p, double q) {
        return new Homography2D_F64(1, 0, 0, 0, 1, 0, p, q, 1);
    }

    private static Affine2D_F64 aOf(Homography2D_F64 h) {
        // Only valid for h with a zero last row; used to mirror the affine-representable cases.
        return new Affine2D_F64(h.a11, h.a12, h.a21, h.a22, h.a13, h.a23);
    }

    // ---------------------------------------------------------------- harness

    private static <IT extends InvertibleTransform<IT>> LogicalNavigationState<IT> fresh(
            MotionModelSupport<IT> support, IT prototype) {
        return new LogicalNavigationState<>(support, prototype);
    }

    /** Cumulative product of a step list, in application order, as the independent ground truth. */
    private static <IT extends InvertibleTransform<IT>> IT cumulative(List<IT> steps, IT identity) {
        IT acc = identity.createInstance();
        acc.reset();
        for (IT s : steps) {
            IT next = acc.concat(s, null);
            acc = next;
        }
        return acc;
    }

    // ---------------------------------------------------------------- A. known answers

    @Test
    public void pureTranslationGivesExactPositionAndZeroYaw_homography() {
        LogicalNavigationState<Homography2D_F64> nav =
                fresh(MotionModelSupport.HOMOGRAPHY, new Homography2D_F64());
        nav.observe(hTranslate(-10, 20), W, H);

        Pose3D p = nav.pose();
        assertEquals(10.0, p.x, EXACT);
        assertEquals(20.0, p.y, EXACT);
        assertEquals(0.0, p.yaw, EXACT);
        assertEquals(0.0, p.z, EXACT, "z is never estimated by this component");
    }

    @Test
    public void pureTranslationGivesExactPositionAndZeroYaw_affine() {
        LogicalNavigationState<Affine2D_F64> nav =
                fresh(MotionModelSupport.AFFINE, new Affine2D_F64());
        nav.observe(aOf(hTranslate(-10, 20)), W, H);

        Pose3D p = nav.pose();
        assertEquals(10.0, p.x, EXACT);
        assertEquals(20.0, p.y, EXACT);
        assertEquals(0.0, p.yaw, EXACT);
    }

    /** <b>B. Rotation without translation.</b> The defining property of a centre-based readout. */
    @Test
    public void pureRotationProducesYawWithZeroTranslation_bothModels() {
        LogicalNavigationState<Homography2D_F64> h =
                fresh(MotionModelSupport.HOMOGRAPHY, new Homography2D_F64());
        h.observe(hRotateAboutCentre(-30), W, H);
        assertEquals(0.0, h.pose().x, 1e-9, "homography: pure rotation must not fabricate x");
        assertEquals(0.0, h.pose().y, 1e-9, "homography: pure rotation must not fabricate y");
        assertEquals(30.0, h.pose().yaw, 1e-9);

        LogicalNavigationState<Affine2D_F64> a =
                fresh(MotionModelSupport.AFFINE, new Affine2D_F64());
        a.observe(aOf(hRotateAboutCentre(-30)), W, H);
        assertEquals(0.0, a.pose().x, 1e-9, "affine: pure rotation must not fabricate x");
        assertEquals(0.0, a.pose().y, 1e-9, "affine: pure rotation must not fabricate y");
        assertEquals(30.0, a.pose().yaw, 1e-9);
    }

    @Test
    public void translationAndRotationCompose_homography() {
        LogicalNavigationState<Homography2D_F64> nav =
                fresh(MotionModelSupport.HOMOGRAPHY, new Homography2D_F64());
        // apply rotation, then translation
        Homography2D_F64 f = hRotateAboutCentre(-30).concat(hTranslate(-10, 20), null);
        nav.observe(f, W, H);

        assertEquals(30.0, nav.pose().yaw, 1e-9);
        // The centre maps back through f^-1; verify against a direct computation.
        Point2D_F64 expected = applyH(f.invert(null), W / 2.0, H / 2.0);
        assertEquals(expected.x - W / 2.0, nav.pose().x, EXACT);
        assertEquals(-(expected.y - H / 2.0), nav.pose().y, EXACT);
    }

    /** Scaling changes footprint size but must leave a centred pose at the origin. */
    @Test
    public void scalingChangesFootprintButNotCentrePosition_bothModels() {
        LogicalNavigationState<Homography2D_F64> h =
                fresh(MotionModelSupport.HOMOGRAPHY, new Homography2D_F64());
        h.observe(hScaleAboutCentre(2.0), W, H);
        assertEquals(0.0, h.pose().x, 1e-9);
        assertEquals(0.0, h.pose().y, 1e-9);
        // content magnified 2x => the frame covers half the logical extent in each axis
        Quadrilateral_F64 q = h.logicalFootprint();
        assertEquals(W / 2.0, q.b.x - q.a.x, 1e-9, "footprint width should halve at 2x magnification");

        LogicalNavigationState<Affine2D_F64> a =
                fresh(MotionModelSupport.AFFINE, new Affine2D_F64());
        a.observe(aOf(hScaleAboutCentre(2.0)), W, H);
        assertEquals(0.0, a.pose().x, 1e-9);
        assertEquals(W / 2.0, a.logicalFootprint().b.x - a.logicalFootprint().a.x, 1e-9);
    }

    /** Affine shear must survive into the footprint rather than being silently dropped. */
    @Test
    public void affineShearAppearsInFootprint() {
        LogicalNavigationState<Affine2D_F64> nav =
                fresh(MotionModelSupport.AFFINE, new Affine2D_F64());
        nav.observe(aOf(hShear(0.25)), W, H);

        Quadrilateral_F64 q = nav.logicalFootprint();
        // A sheared mapping makes the footprint a parallelogram: its left edge is no longer vertical.
        double leftEdgeDx = q.d.x - q.a.x;
        assertTrue(Math.abs(leftEdgeDx) > 1.0,
                "expected shear to tilt the footprint's left edge; dx = " + leftEdgeDx);
        // ...but opposite edges stay parallel under affine.
        assertEquals(q.b.x - q.a.x, q.c.x - q.d.x, 1e-9, "affine must preserve parallelism");
        assertEquals(q.b.y - q.a.y, q.c.y - q.d.y, 1e-9, "affine must preserve parallelism");
    }

    /** Projective deformation must survive too — and must break parallelism, unlike affine. */
    @Test
    public void projectiveDeformationAppearsInFootprintAndBreaksParallelism() {
        LogicalNavigationState<Homography2D_F64> nav =
                fresh(MotionModelSupport.HOMOGRAPHY, new Homography2D_F64());
        nav.observe(hProjective(0.0008, 0.0005), W, H);

        Quadrilateral_F64 q = nav.logicalFootprint();
        double topWidth = q.b.x - q.a.x;
        double bottomWidth = q.c.x - q.d.x;
        assertTrue(Math.abs(topWidth - bottomWidth) > 1.0,
                "expected perspective to make opposite edges non-parallel; " + topWidth + " vs " + bottomWidth);
    }

    // ---------------------------------------------------------------- C/D. reference changes

    /**
     * <b>C. Reference-change continuity, and D. fold-schedule invariance — the architectural
     * invariant.</b>
     *
     * <p>The same physical motion is replayed under several different epoch-boundary schedules. A
     * canvas re-origin is exactly an epoch boundary, so this is the algebraic form of "changing the
     * mosaic canvas coordinate system must not change the logical navigation trajectory": whatever
     * the schedule, the logical transform must equal the cumulative product of the steps.
     */
    @Test
    public void logicalTrajectoryIsInvariantToFoldSchedule_homography() {
        List<Homography2D_F64> steps = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            steps.add(hTranslate(-3 - 0.1 * i, 2).concat(hRotateAboutCentre(-1.5), null));
        }

        int[][] schedules = {{}, {3}, {1, 2, 3}, {5, 9}, {1, 3, 5, 7, 9, 11}, {11}};
        List<Pose3D> reference = null;

        for (int[] folds : schedules) {
            LogicalNavigationState<Homography2D_F64> nav =
                    fresh(MotionModelSupport.HOMOGRAPHY, new Homography2D_F64());
            Homography2D_F64 epochAccum = new Homography2D_F64();   // F, reset at each fold
            List<Pose3D> poses = new ArrayList<>();

            for (int i = 0; i < steps.size(); i++) {
                Homography2D_F64 next = epochAccum.concat(steps.get(i), null);
                epochAccum = next;
                nav.observe(epochAccum, W, H);
                poses.add(nav.pose());

                if (contains(folds, i)) {          // simulate a canvas re-origin here
                    nav.foldEpoch(epochAccum);
                    epochAccum = new Homography2D_F64();   // BoofCV zeroes it
                    nav.observe(epochAccum, W, H);
                    // continuity: the pose must not move across the fold
                    assertEquals(poses.get(poses.size() - 1).x, nav.pose().x, EXACT,
                            "pose.x moved across a fold at step " + i);
                    assertEquals(poses.get(poses.size() - 1).y, nav.pose().y, EXACT,
                            "pose.y moved across a fold at step " + i);
                    assertEquals(poses.get(poses.size() - 1).yaw, nav.pose().yaw, EXACT,
                            "pose.yaw moved across a fold at step " + i);
                }
            }

            // independent ground truth: the cumulative product, with no epochs at all
            Homography2D_F64 truth = cumulative(steps, new Homography2D_F64());
            LogicalNavigationState<Homography2D_F64> direct =
                    fresh(MotionModelSupport.HOMOGRAPHY, new Homography2D_F64());
            direct.observe(truth, W, H);
            assertEquals(direct.pose().x, poses.get(poses.size() - 1).x, 1e-8);
            assertEquals(direct.pose().y, poses.get(poses.size() - 1).y, 1e-8);
            assertEquals(direct.pose().yaw, poses.get(poses.size() - 1).yaw, 1e-8);

            if (reference == null) {
                reference = poses;
            } else {
                for (int i = 0; i < poses.size(); i++) {
                    assertEquals(reference.get(i).x, poses.get(i).x, 1e-8, "step " + i + " x");
                    assertEquals(reference.get(i).y, poses.get(i).y, 1e-8, "step " + i + " y");
                    assertEquals(reference.get(i).yaw, poses.get(i).yaw, 1e-8, "step " + i + " yaw");
                }
            }
        }
    }

    /** Same invariant, affine model — the abstraction must not be homography-only. */
    @Test
    public void logicalTrajectoryIsInvariantToFoldSchedule_affine() {
        List<Affine2D_F64> steps = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            steps.add(aOf(hTranslate(-3 - 0.1 * i, 2).concat(hRotateAboutCentre(-1.5), null)));
        }

        int[][] schedules = {{}, {2, 6}, {0, 1, 2, 3, 4}, {11}};
        List<Pose3D> reference = null;

        for (int[] folds : schedules) {
            LogicalNavigationState<Affine2D_F64> nav =
                    fresh(MotionModelSupport.AFFINE, new Affine2D_F64());
            Affine2D_F64 epochAccum = new Affine2D_F64();
            List<Pose3D> poses = new ArrayList<>();

            for (int i = 0; i < steps.size(); i++) {
                Affine2D_F64 next = epochAccum.concat(steps.get(i), null);
                epochAccum = next;
                nav.observe(epochAccum, W, H);
                poses.add(nav.pose());
                if (contains(folds, i)) {
                    nav.foldEpoch(epochAccum);
                    epochAccum = new Affine2D_F64();
                    nav.observe(epochAccum, W, H);
                    assertEquals(poses.get(poses.size() - 1).x, nav.pose().x, EXACT);
                    assertEquals(poses.get(poses.size() - 1).y, nav.pose().y, EXACT);
                }
            }

            if (reference == null) {
                reference = poses;
            } else {
                for (int i = 0; i < poses.size(); i++) {
                    assertEquals(reference.get(i).x, poses.get(i).x, 1e-8, "step " + i + " x");
                    assertEquals(reference.get(i).y, poses.get(i).y, 1e-8, "step " + i + " y");
                    assertEquals(reference.get(i).yaw, poses.get(i).yaw, 1e-8, "step " + i + " yaw");
                }
            }
        }
    }

    /**
     * A fold must carry <b>scale</b> across the epoch boundary. This is the specific loss in the
     * pre-{@code DEC-VO-003} SE(2) re-anchoring, and it is the mechanism behind {@code COMP-001}
     * §4's "recentering may silently rescale the position units".
     */
    @Test
    public void foldCarriesScaleAcrossEpochBoundary() {
        LogicalNavigationState<Homography2D_F64> nav =
                fresh(MotionModelSupport.HOMOGRAPHY, new Homography2D_F64());

        // Epoch 1: magnify 2x (e.g. descending), then fold as a canvas re-origin would.
        Homography2D_F64 epoch1 = hScaleAboutCentre(2.0);
        nav.observe(epoch1, W, H);
        double widthAfterZoom = nav.logicalFootprint().b.x - nav.logicalFootprint().a.x;
        nav.foldEpoch(epoch1);
        nav.observe(new Homography2D_F64(), W, H);

        assertEquals(widthAfterZoom, nav.logicalFootprint().b.x - nav.logicalFootprint().a.x, EXACT,
                "the footprint scale must survive the fold; dropping it is the old defect");

        // Epoch 2: translate 10 px of content. Because the logical frame is still at 2x, that
        // corresponds to 5 logical px -- the unit consistency the old SE(2) re-anchoring lost.
        nav.observe(hTranslate(-10, 0), W, H);
        assertEquals(5.0, nav.pose().x, 1e-9,
                "post-fold translation must be expressed in logical units, not raw epoch pixels");
    }

    // ---------------------------------------------------------------- helpers

    private static boolean contains(int[] a, int v) {
        for (int x : a) if (x == v) return true;
        return false;
    }

    private static Point2D_F64 applyH(Homography2D_F64 h, double x, double y) {
        double z = h.a31 * x + h.a32 * y + h.a33;
        return new Point2D_F64((h.a11 * x + h.a12 * y + h.a13) / z,
                               (h.a21 * x + h.a22 * y + h.a23) / z);
    }
}
