package org.boofcv.stitching;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Known-answer tests for the polar/SVD decomposition {@code DEC-VO-004}'s rigid readout rests on.
 *
 * <p>Each property the implementation's closed forms claim is checked against an independently
 * constructed matrix — a rotation times a symmetric stretch built from chosen singular values and a
 * chosen axis — rather than against another implementation of the same formula.
 *
 * <p>Evidence tier: T1 (analytical).
 */
public class RigidMotionDecompositionTest {

    private static final double EPS = 1e-12;

    /** J = R(rotDeg) · V(axisDeg) diag(s1, s2) V(axisDeg)ᵀ — the polar form, built by hand. */
    private static double[] build(double rotDeg, double s1, double s2, double axisDeg) {
        double a = Math.toRadians(axisDeg), ca = Math.cos(a), sa = Math.sin(a);
        // P = V S V^T
        double p11 = s1 * ca * ca + s2 * sa * sa;
        double p12 = (s1 - s2) * ca * sa;
        double p22 = s1 * sa * sa + s2 * ca * ca;
        double r = Math.toRadians(rotDeg), cr = Math.cos(r), sr = Math.sin(r);
        return new double[]{
                cr * p11 - sr * p12, cr * p12 - sr * p22,
                sr * p11 + cr * p12, sr * p12 + cr * p22};
    }

    private static RigidMotionDecomposition decompose(double[] j) {
        return new RigidMotionDecomposition().set(j[0], j[1], j[2], j[3]);
    }

    @Test
    public void identityIsRigidWithUnitScale() {
        RigidMotionDecomposition d = decompose(new double[]{1, 0, 0, 1});
        assertEquals(0.0, d.getRotationRad(), EPS);
        assertEquals(1.0, d.getUniformScale(), EPS);
        assertEquals(1.0, d.getAnisotropy(), EPS);
        assertEquals(0.0, d.getDeformationMagnitude(), EPS);
        assertEquals(0.0, d.logUniformScale(), EPS);
        assertTrue(d.isProperRotation());
    }

    @Test
    public void pureRotationRecoversTheAngleAndLeavesScaleAtOne() {
        for (double deg : new double[]{-179, -90, -30, -0.5, 0.5, 17, 90, 179}) {
            RigidMotionDecomposition d = decompose(build(deg, 1, 1, 0));
            assertEquals(Math.toRadians(deg), d.getRotationRad(), 1e-12, "rotation " + deg);
            assertEquals(1.0, d.getUniformScale(), EPS);
            assertEquals(1.0, d.getAnisotropy(), 1e-12);
        }
    }

    @Test
    public void pureUniformScaleRecoversTheScaleAndZeroRotation() {
        for (double s : new double[]{0.25, 0.9, 1.0, 1.1, 4.0}) {
            RigidMotionDecomposition d = decompose(build(0, s, s, 0));
            assertEquals(0.0, d.getRotationRad(), EPS);
            assertEquals(s, d.getUniformScale(), 1e-12, "scale " + s);
            assertEquals(Math.log(s), d.logUniformScale(), 1e-12);
            assertEquals(1.0, d.getAnisotropy(), 1e-12);
        }
    }

    /** The geometric mean, not the arithmetic mean: this is the choice the class comment defends. */
    @Test
    public void uniformScaleIsTheGeometricMeanOfTheSingularValues() {
        RigidMotionDecomposition d = decompose(build(0, 4.0, 1.0, 0));
        assertEquals(Math.sqrt(4.0), d.getUniformScale(), 1e-12);   // NOT (4+1)/2 = 2.5
        assertEquals(4.0, d.getAnisotropy(), 1e-12);
    }

    @Test
    public void rotationScaleAndStretchAreSeparatedForArbitraryCombinations() {
        Random rng = new Random(20260826);
        for (int i = 0; i < 500; i++) {
            double rot = rng.nextDouble() * 340 - 170;
            double s1 = 0.2 + rng.nextDouble() * 4;
            double s2 = 0.2 + rng.nextDouble() * 4;
            if (s2 > s1) { double t = s1; s1 = s2; s2 = t; }
            double axis = rng.nextDouble() * 180;
            RigidMotionDecomposition d = decompose(build(rot, s1, s2, axis));
            assertEquals(Math.toRadians(rot), d.getRotationRad(), 1e-9, "rotation");
            assertEquals(Math.sqrt(s1 * s2), d.getUniformScale(), 1e-9, "scale");
            assertEquals(s1 / s2, d.getAnisotropy(), 1e-9, "anisotropy");
            assertEquals(Math.abs(Math.log(s1 / s2)) / Math.sqrt(2.0),
                    d.getDeformationMagnitude(), 1e-9, "deformation");
            if (s1 / s2 > 1.001) {
                double got = Math.toDegrees(d.getStretchAxisRad()) % 180.0;
                double want = ((axis % 180.0) + 180.0) % 180.0;
                double diff = Math.abs(((got - want + 270.0) % 180.0) - 90.0);
                assertEquals(0.0, diff, 1e-6, "stretch axis: got " + got + " want " + want);
            }
        }
    }

    /** The polar rotation is the closest proper rotation in the Frobenius norm — checked numerically. */
    @Test
    public void polarRotationIsTheNearestRotationInFrobeniusNorm() {
        double[] j = build(23.0, 2.5, 0.7, 40.0);
        RigidMotionDecomposition d = decompose(j);
        double best = Math.toDegrees(d.getRotationRad());
        double bestCost = frobeniusDistanceToRotation(j, best);
        for (double deg = best - 5; deg <= best + 5; deg += 0.05) {
            assertTrue(frobeniusDistanceToRotation(j, deg) >= bestCost - 1e-12,
                    "a nearer rotation exists at " + deg);
        }
    }

    private static double frobeniusDistanceToRotation(double[] j, double deg) {
        double r = Math.toRadians(deg), c = Math.cos(r), s = Math.sin(r);
        return sq(j[0] - c) + sq(j[1] + s) + sq(j[2] - s) + sq(j[3] - c);
    }

    private static double sq(double v) { return v * v; }

    @Test
    public void reflectionIsFlaggedRatherThanReportedAsScale() {
        RigidMotionDecomposition d = decompose(new double[]{1, 0, 0, -1});   // det = -1
        assertFalse(d.isProperRotation());
        assertEquals(1.0, d.getUniformScale(), EPS, "magnitude still reported");
        assertEquals(1.0, d.getAnisotropy(), 1e-12);
    }

    @Test
    public void singularJacobianIsFlagged() {
        RigidMotionDecomposition d = decompose(new double[]{1, 0, 0, 0});    // det = 0
        assertFalse(d.isProperRotation());
        assertEquals(Double.POSITIVE_INFINITY, d.getAnisotropy());
    }

    /** Under a camera tilt of θ the anisotropy is 1/cos θ and the centre scale cos^{3/2}θ (LIT-VO-003 §4). */
    @Test
    public void tiltSignatureMatchesTheDerivedGeometry() {
        double theta = Math.toRadians(15.5);              // the maximum measured on HKairport01
        double along = Math.cos(theta) * Math.cos(theta);  // foreshortening along the tilt axis
        double across = Math.cos(theta);
        RigidMotionDecomposition d = decompose(build(0, across, along, 0));
        assertEquals(1 / Math.cos(theta), d.getAnisotropy(), 1e-12);
        assertEquals(Math.pow(Math.cos(theta), 1.5), d.getUniformScale(), 1e-12);
        assertTrue(d.getAnisotropy() < 1.04, "physical ceiling quoted in LIT-VO-003 §4");
    }
}
