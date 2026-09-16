package org.boofcv.stitching;

import boofcv.struct.geo.AssociatedPair;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.point.Point2D_F64;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Known-answer tests for the {@code DEC-VO-006} similarity model: the transform's group law, the
 * Umeyama generator, the distance function, and the exactness properties {@code EXP-VO-009}'s
 * fairness argument depends on.
 *
 * <p>These are <b>tier 1</b> and exact. Every case here has a closed-form right answer, so the
 * tolerances are machine-precision rather than empirical — a similarity fitted to correspondences
 * generated <em>by</em> a similarity must recover it to rounding, and any looser assertion would
 * hide exactly the class of defect that matters ({@code DEC-VO-006} <i>Trade-offs accepted</i>: this
 * is code the project maintains itself, so its correctness evidence has to be stronger than
 * agreement with a library).
 */
class Sim2ModelTest {

    /** Machine-precision tolerance for quantities that are exact up to rounding. */
    private static final double EXACT = 1e-10;

    private static List<AssociatedPair> apply(Sim2_F64 t, double[][] pts) {
        List<AssociatedPair> out = new ArrayList<>();
        for (double[] p : pts) {
            Point2D_F64 q = MotionModelSupport.SIMILARITY.apply(t, p[0], p[1], null);
            out.add(new AssociatedPair(p[0], p[1], q.x, q.y));
        }
        return out;
    }

    /** A spread that is not symmetric, so a degenerate fit cannot pass by accident. */
    private static final double[][] POINTS = {
            {12.0, 7.0}, {480.0, 33.0}, {91.0, 620.0}, {700.0, 512.0}, {305.0, 288.0},
    };

    private static void assertRecovers(String label, double scale, double thetaRad,
                                       double tx, double ty) {
        Sim2_F64 truth = Sim2_F64.of(scale, thetaRad, tx, ty);
        Sim2_F64 fitted = new Sim2_F64();
        assertTrue(new GenerateSimilarity2D().generate(apply(truth, POINTS), fitted), label);

        assertEquals(scale, fitted.scale(), EXACT * Math.max(1.0, scale), label + " scale");
        assertEquals(Math.sin(thetaRad), Math.sin(fitted.thetaRad()), EXACT, label + " sin theta");
        assertEquals(Math.cos(thetaRad), Math.cos(fitted.thetaRad()), EXACT, label + " cos theta");
        assertEquals(truth.tx, fitted.tx, EXACT * Math.max(1.0, Math.abs(tx)), label + " tx");
        assertEquals(truth.ty, fitted.ty, EXACT * Math.max(1.0, Math.abs(ty)), label + " ty");
    }

    // ---------------------------------------------------------------- the ten specified cases

    @Nested
    @DisplayName("Known-answer recovery of an exact Sim(2)")
    class KnownAnswer {

        @Test
        @DisplayName("1. pure translation")
        void pureTranslation() {
            assertRecovers("pure translation", 1.0, 0.0, -13.75, 41.5);
        }

        @Test
        @DisplayName("2. pure yaw")
        void pureYaw() {
            assertRecovers("pure yaw", 1.0, Math.toRadians(23.4), 0.0, 0.0);
        }

        @Test
        @DisplayName("3. pure uniform scale")
        void pureScale() {
            assertRecovers("pure scale", 1.137, 0.0, 0.0, 0.0);
        }

        @Test
        @DisplayName("4. translation + yaw")
        void translationYaw() {
            assertRecovers("translation+yaw", 1.0, Math.toRadians(-7.25), 22.0, -9.0);
        }

        @Test
        @DisplayName("5. translation + scale")
        void translationScale() {
            assertRecovers("translation+scale", 0.884, 0.0, -55.0, 17.5);
        }

        @Test
        @DisplayName("6. yaw + scale")
        void yawScale() {
            assertRecovers("yaw+scale", 1.061, Math.toRadians(112.0), 0.0, 0.0);
        }

        @Test
        @DisplayName("7. all four parameters at once, including a rotation past a quadrant")
        void combined() {
            assertRecovers("combined", 0.93, Math.toRadians(-164.0), 310.5, -212.25);
        }

        @Test
        @DisplayName("2 points already determine the model exactly (4 DoF, 4 equations)")
        void twoPointsSuffice() {
            Sim2_F64 truth = Sim2_F64.of(1.08, Math.toRadians(31.0), 5.0, -8.0);
            double[][] two = {{10.0, 20.0}, {400.0, 350.0}};
            Sim2_F64 fitted = new Sim2_F64();
            assertTrue(new GenerateSimilarity2D().generate(apply(truth, two), fitted));
            assertEquals(truth.a, fitted.a, EXACT);
            assertEquals(truth.b, fitted.b, EXACT);
            assertEquals(truth.tx, fitted.tx, 1e-9);
            assertEquals(truth.ty, fitted.ty, 1e-9);
        }
    }

    // ---------------------------------------------------------------- noise, outliers, violation

    @Test
    @DisplayName("8. noisy correspondences: the fit is consistent and unbiased at the truth")
    void noisyCorrespondences() {
        Random rand = new Random(4242);
        Sim2_F64 truth = Sim2_F64.of(1.02, Math.toRadians(3.0), 12.0, -4.0);
        double[][] grid = grid(20, 20, 1224, 1024);

        // With 400 correspondences over a full frame and sigma = 0.3 px, the closed-form standard
        // error of theta is sigma / (s * sqrt(tr M)) -- a few times 1e-5 rad. Assert well inside a
        // band that a systematically wrong estimator could not sit in.
        double sumTheta = 0.0, sumScale = 0.0;
        final int reps = 200;
        for (int r = 0; r < reps; r++) {
            List<AssociatedPair> pairs = apply(truth, grid);
            for (AssociatedPair p : pairs) {
                p.p2.x += rand.nextGaussian() * 0.3;
                p.p2.y += rand.nextGaussian() * 0.3;
            }
            Sim2_F64 fitted = new Sim2_F64();
            assertTrue(new GenerateSimilarity2D().generate(pairs, fitted));
            sumTheta += fitted.thetaRad();
            sumScale += fitted.scale();
        }
        assertEquals(truth.thetaRad(), sumTheta / reps, 1e-4, "mean fitted theta");
        assertEquals(truth.scale(), sumScale / reps, 1e-4, "mean fitted scale");
    }

    @Test
    @DisplayName("9. outlier-contaminated: the generator itself is NOT robust, which is RANSAC's job")
    void outlierContamination() {
        Sim2_F64 truth = Sim2_F64.of(1.0, 0.0, 0.0, 0.0);
        List<AssociatedPair> pairs = apply(truth, POINTS);
        pairs.get(0).p2.setTo(9000.0, -9000.0);   // one gross outlier

        Sim2_F64 fitted = new Sim2_F64();
        assertTrue(new GenerateSimilarity2D().generate(pairs, fitted));
        assertTrue(Math.abs(fitted.thetaRad()) > Math.toRadians(1.0),
                "a least-squares fit must be corrupted by a gross outlier -- if it were not, the "
                        + "generator would be doing robust estimation that RANSAC is not expecting");

        // ...and the distance function must expose it, which is what lets RANSAC reject it.
        DistanceSimilarity2DSq d = new DistanceSimilarity2DSq();
        d.setModel(truth);
        assertTrue(d.distance(pairs.get(0)) > 3.0, "outlier above the inlier threshold");
        for (int i = 1; i < pairs.size(); i++) {
            assertEquals(0.0, d.distance(pairs.get(i)), EXACT, "inlier " + i);
        }
    }

    @Test
    @DisplayName("10. projective violation: similarity is intentionally imperfect and says so")
    void projectiveViolation() {
        // A homography with a small perspective row -- the tilt case of LIT-VO-003 section 4.
        Homography2D_F64 h = new Homography2D_F64(1, 0, 0, 0, 1, 0, 2.0e-5, -1.0e-5, 1);
        double[][] grid = grid(12, 12, 1224, 1024);
        List<AssociatedPair> pairs = new ArrayList<>();
        for (double[] p : grid) {
            Point2D_F64 q = MotionModelSupport.HOMOGRAPHY.apply(h, p[0], p[1], null);
            pairs.add(new AssociatedPair(p[0], p[1], q.x, q.y));
        }

        Sim2_F64 fitted = new Sim2_F64();
        assertTrue(new GenerateSimilarity2D().generate(pairs, fitted));

        // The fit succeeds but leaves a real residual -- the model cannot represent perspective.
        // That residual is the predicted regression mechanism of EXP-VO-009 H6, so it is asserted
        // to be present rather than assumed away.
        DistanceSimilarity2DSq d = new DistanceSimilarity2DSq();
        d.setModel(fitted);
        double worst = 0.0;
        for (AssociatedPair p : pairs) worst = Math.max(worst, d.distance(p));
        assertTrue(worst > 3.0,
                "expected the similarity fit to exceed the inlier threshold somewhere under a "
                        + "perspective violation; worst squared residual was " + worst);
    }

    // ---------------------------------------------------------------- degeneracy

    @Test
    @DisplayName("Degenerate samples are refused, never returned as NaN")
    void degeneracyRefused() {
        GenerateSimilarity2D gen = new GenerateSimilarity2D();
        Sim2_F64 out = new Sim2_F64();

        List<AssociatedPair> coincident = new ArrayList<>();
        for (int i = 0; i < 3; i++) coincident.add(new AssociatedPair(5, 5, 9, 9));
        assertFalse(gen.generate(coincident, out), "coincident p1 has no spatial extent");

        List<AssociatedPair> collapsed = new ArrayList<>();
        collapsed.add(new AssociatedPair(0, 0, 4, 4));
        collapsed.add(new AssociatedPair(10, 0, 4, 4));
        collapsed.add(new AssociatedPair(0, 10, 4, 4));
        assertFalse(gen.generate(collapsed, out), "all p2 coincident collapses scale to zero");

        assertTrue(Double.isFinite(out.a) && Double.isFinite(out.b), "output left untouched/finite");
        assertThrows(IllegalStateException.class, () -> new Sim2_F64(0, 0, 1, 2).invert(null),
                "a zero-scale model must fail loudly rather than emit infinities");
    }

    // ---------------------------------------------------------------- group law and readout

    @Test
    @DisplayName("concat and invert are exact within the family, and match the homography form")
    void groupLaw() {
        Sim2_F64 t1 = Sim2_F64.of(1.07, Math.toRadians(19.0), 31.0, -12.0);
        Sim2_F64 t2 = Sim2_F64.of(0.93, Math.toRadians(-47.0), -8.0, 66.0);

        // this.concat(second, r) means r(p) = second(this(p)) -- georegression's convention.
        Sim2_F64 composed = t1.concat(t2, null);
        for (double[] p : POINTS) {
            Point2D_F64 stepwise = MotionModelSupport.SIMILARITY.apply(t1, p[0], p[1], null);
            stepwise = MotionModelSupport.SIMILARITY.apply(t2, stepwise.x, stepwise.y, null);
            Point2D_F64 direct = MotionModelSupport.SIMILARITY.apply(composed, p[0], p[1], null);
            assertEquals(stepwise.x, direct.x, EXACT);
            assertEquals(stepwise.y, direct.y, EXACT);
        }
        assertEquals(t1.scale() * t2.scale(), composed.scale(), EXACT, "scales multiply");

        Sim2_F64 roundTrip = t1.concat(t1.invert(null), null);
        assertEquals(1.0, roundTrip.scale(), EXACT);
        assertEquals(0.0, roundTrip.thetaRad(), EXACT);
        assertEquals(0.0, roundTrip.tx, 1e-9);
        assertEquals(0.0, roundTrip.ty, 1e-9);

        // The homography form must compose the same way, since that is what the run record stores.
        Homography2D_F64 h1 = MotionModelSupport.SIMILARITY.asHomography(t1, null);
        Homography2D_F64 h2 = MotionModelSupport.SIMILARITY.asHomography(t2, null);
        Homography2D_F64 hc = MotionModelSupport.SIMILARITY.asHomography(composed, null);
        Homography2D_F64 hh = h1.concat(h2, null);
        assertEquals(hh.a11, hc.a11, EXACT);
        assertEquals(hh.a12, hc.a12, EXACT);
        assertEquals(hh.a13, hc.a13, 1e-9);
        assertEquals(hh.a21, hc.a21, EXACT);
        assertEquals(hh.a22, hc.a22, EXACT);
        assertEquals(hh.a23, hc.a23, 1e-9);
    }

    @Test
    @DisplayName("RIGID_MOTION reads a similarity back as its own parameters, with no special case")
    void rigidReadoutIsTheIdentity() {
        // This is the property EXP-VO-009 procedure gate 3 requires, and the reason the task's
        // "anisotropy/shear/perspective structurally zero, not fabricated" requirement is met
        // without a branch anywhere in RigidMotionDecomposition.
        for (double thetaDeg : new double[]{-173.0, -40.0, -0.35, 0.0, 0.9, 88.0, 150.0}) {
            for (double scale : new double[]{0.71, 0.999, 1.0, 1.0004, 1.44}) {
                Sim2_F64 t = Sim2_F64.of(scale, Math.toRadians(thetaDeg), 3.0, -4.0);
                double[] j = MotionModelSupport.SIMILARITY.jacobian(t, 612.0, 512.0, null);
                RigidMotionDecomposition d = new RigidMotionDecomposition().set(j[0], j[1], j[2], j[3]);

                assertTrue(d.isProperRotation());
                assertEquals(Math.toRadians(thetaDeg), d.getRotationRad(), EXACT,
                        "rotation at theta=" + thetaDeg);
                assertEquals(scale, d.getUniformScale(), EXACT, "scale at s=" + scale);
                assertEquals(1.0, d.getAnisotropy(), EXACT, "anisotropy is exactly 1");
                assertEquals(0.0, d.getDeformationMagnitude(), EXACT, "deformation is exactly 0");
                assertEquals(0.0, MotionModelSupport.SIMILARITY.perspectiveMagnitude(t, 1224, 1024),
                        0.0, "perspective is exactly 0");
            }
        }
    }

    @Test
    @DisplayName("the legacy edge angle and the polar rotation coincide under a similarity")
    void edgeAngleEqualsPolarRotation() {
        // EXP-VO-008 showed these differ by atan2(q, p) for an affine, where q is the shear. A
        // similarity has q = 0 identically, so the discrepancy that produced the AMtown01 affine
        // reversal cannot arise under this model. Asserted rather than argued.
        Sim2_F64 t = Sim2_F64.of(1.03, Math.toRadians(11.5), 40.0, -20.0);
        double[] j = MotionModelSupport.SIMILARITY.jacobian(t, 0, 0, null);
        double polar = Math.atan2(j[2] - j[1], j[0] + j[3]);
        double edge = Math.atan2(j[2], j[0]);          // top-edge angle, COMP-001 section 3.6
        assertEquals(polar, edge, EXACT);
        assertEquals(t.thetaRad(), polar, EXACT);
    }

    @Test
    @DisplayName("shrinkTransform is exact: a scale-and-translate IS a similarity")
    void shrinkTransformExact() {
        Sim2_F64 t = MotionModelSupport.SIMILARITY.shrinkTransform(0.5, 100.0, 250.0);
        assertEquals(0.5, t.scale(), EXACT);
        assertEquals(0.0, t.thetaRad(), EXACT);
        Point2D_F64 q = MotionModelSupport.SIMILARITY.apply(t, 40.0, 60.0, null);
        assertEquals(120.0, q.x, EXACT);
        assertEquals(280.0, q.y, EXACT);
    }

    private static double[][] grid(int nx, int ny, int width, int height) {
        double[][] out = new double[nx * ny][2];
        int k = 0;
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                out[k][0] = (i + 0.5) * width / nx;
                out[k][1] = (j + 0.5) * height / ny;
                k++;
            }
        }
        return out;
    }
}
