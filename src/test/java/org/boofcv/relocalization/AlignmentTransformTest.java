package org.boofcv.relocalization;

import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The translation-only alignment of {@code DEC-INT-001} (amendment 2026-09-07), pinned by known
 * answers computed by hand. There is no rotational degree of freedom and no scale: metres in,
 * metres out, heading untouched. T1.
 */
public class AlignmentTransformTest {

    private static final double EPS = 1e-12;

    private static PlanarPosition p(double e, double n) {
        return new PlanarPosition(e, n);
    }

    private static void assertPos(PlanarPosition expected, PlanarPosition actual) {
        assertEquals(expected.eastM(), actual.eastM(), EPS, "east");
        assertEquals(expected.northM(), actual.northM(), EPS, "north");
    }

    @Test
    public void identityLeavesAPositionAndAPoseUnchanged() {
        assertEquals(p(12.5, -3.25), AlignmentTransform.IDENTITY.apply(p(12.5, -3.25)));
        Pose3D q = new Pose3D(12.5, -3.25, 0.0, 47.0);
        Pose3D r = AlignmentTransform.IDENTITY.apply(q);
        assertEquals(12.5, r.x, 0.0);
        assertEquals(-3.25, r.y, 0.0);
        assertEquals(47.0, r.yaw, 0.0);
        assertTrue(AlignmentTransform.IDENTITY.isIdentity(0.0));
    }

    /** Hand computation: (1, 2) + (10, 20) = (11, 22). */
    @Test
    public void applyIsAPureTranslationInMetres() {
        AlignmentTransform a = AlignmentTransform.of(10, 20);
        assertEquals(p(11.0, 22.0), a.apply(p(1, 2)));
        assertEquals(p(10.0, 20.0), a.apply(PlanarPosition.ORIGIN));
        assertEquals(Math.hypot(10, 20), a.norm(), EPS);
    }

    /** The heading is the VO's authoritative heading and passes through untouched, as does z. */
    @Test
    public void applyOnAPoseNeverChangesYawOrZ() {
        AlignmentTransform a = AlignmentTransform.of(-100, 55);
        for (double yaw : new double[]{0.0, 89.999, 180.0, 271.25, 359.9}) {
            Pose3D out = a.apply(new Pose3D(1, 2, 7.0, yaw));
            // Pose3D's own constructor re-normalises every yaw ((yaw + 360) % 360), which can move
            // it by an ULP; the alignment adds nothing to it.
            assertEquals(yaw, out.yaw, 1e-9, "yaw untouched at " + yaw);
            assertEquals(7.0, out.z, 0.0);
            assertEquals(-99.0, out.x, EPS);
            assertEquals(57.0, out.y, EPS);
        }
        // NaN (no authoritative heading) passes through as NaN — never replaced by a number here.
        assertTrue(Double.isNaN(a.apply(new Pose3D(1, 2, 0, Double.NaN)).yaw));
    }

    @Test
    public void composeMatchesSequentialApplicationAndIsAssociative() {
        AlignmentTransform a = AlignmentTransform.of(-4.5, 8.0);
        AlignmentTransform b = AlignmentTransform.of(15.0, -2.0);
        AlignmentTransform c = AlignmentTransform.of(0.5, -0.25);
        PlanarPosition x = p(3.3, -7.7);
        assertPos(a.apply(b.apply(x)), a.compose(b).apply(x));
        AlignmentTransform left = a.compose(b).compose(c), right = a.compose(b.compose(c));
        assertEquals(left.tE(), right.tE(), EPS);
        assertEquals(left.tN(), right.tN(), EPS);
    }

    @Test
    public void inverseCancelsOnBothSides() {
        AlignmentTransform a = AlignmentTransform.of(-37.5, 122.25);
        assertTrue(a.compose(a.inverse()).isIdentity(EPS));
        assertTrue(a.inverse().compose(a).isIdentity(EPS));
    }

    /** {@code reanchor(p_ref, p_q).apply(p_q) == p_ref} — the design's re-anchor identity, exactly. */
    @Test
    public void reanchorMapsTheQueryExactlyOntoTheReference() {
        PlanarPosition ref = p(-37.5, 122.25);
        PlanarPosition local = p(41.2, -8.8);
        AlignmentTransform a = AlignmentTransform.reanchor(ref, local);
        assertPos(ref, a.apply(local));
        assertEquals(-78.7, a.tE(), EPS);
        assertEquals(131.05, a.tN(), EPS);
        // A genuine transform of the frame: a second local position displaced by a known metric
        // step maps to the reference displaced by the same step — the same East/North step, since
        // both frames share the North datum (DEC-VO-010 D2).
        PlanarPosition step = p(5.0, 10.0);
        assertPos(ref.plus(step), a.apply(local.plus(step)));
    }

    /** Known segment query + known reference → the expected East/North translation (test D). */
    @Test
    public void knownQueryAndReferenceGiveTheExpectedTranslation() {
        // The aircraft is 120 m east and 30 m south of its segment origin; the reference says that
        // place is at (500, −200) in the persistent frame. t_e = (380, −170).
        AlignmentTransform t = AlignmentTransform.reanchor(p(500, -200), p(120, -30));
        assertEquals(380.0, t.tE(), EPS);
        assertEquals(-170.0, t.tN(), EPS);
    }

    @Test
    public void reanchorOntoTheCurrentPositionIsTheIdentity() {
        PlanarPosition local = p(41.2, -8.8);
        assertTrue(AlignmentTransform.reanchor(local, local).isIdentity(EPS));
    }

    /** Neither rotation nor scale can be expressed: the class has two parameters and both are metres. */
    @Test
    public void thereIsNoRotationalOrScaleDegreeOfFreedom() {
        // Two references at the same position but with different headings yield the same alignment:
        // the heading cannot enter, because the type carries none.
        AlignmentTransform a = AlignmentTransform.reanchor(p(10, 10), p(1, 1));
        assertEquals(9.0, a.tE(), 0.0);
        assertEquals(9.0, a.tN(), 0.0);
        // Distances are preserved, always (no scale), and so are directions (no rotation).
        AlignmentTransform t = AlignmentTransform.of(100, -250);
        PlanarPosition x = p(1, 1), y = p(-30, 44);
        assertEquals(x.distanceTo(y), t.apply(x).distanceTo(t.apply(y)), EPS);
        assertPos(y.minus(x), t.apply(y).minus(t.apply(x)));
    }

    @Test
    public void deltaFromRecoversTheJumpBetweenTwoAlignments() {
        AlignmentTransform old = AlignmentTransform.of(3, 4);
        AlignmentTransform neu = AlignmentTransform.of(-8, 21);
        AlignmentTransform delta = neu.deltaFrom(old);
        assertEquals(-11.0, delta.tE(), EPS);
        assertEquals(17.0, delta.tN(), EPS);
        PlanarPosition local = p(7, -2);
        assertPos(neu.apply(local), delta.apply(old.apply(local)));
    }

    @Test
    public void wrapDegreesIsRetainedForReportingHeadingDifferences() {
        assertEquals(-10.0, AlignmentTransform.wrapDegrees(350), EPS);
        assertEquals(180.0, AlignmentTransform.wrapDegrees(-180), EPS);
        assertEquals(180.0, AlignmentTransform.wrapDegrees(180), EPS);
        assertEquals(5.0, AlignmentTransform.wrapDegrees(725), EPS);
    }

    @Test
    public void refusesNonFiniteComponents() {
        assertThrows(IllegalArgumentException.class, () -> AlignmentTransform.of(Double.NaN, 0));
        assertThrows(IllegalArgumentException.class, () -> AlignmentTransform.of(0, Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> new PlanarPosition(Double.NaN, 0));
    }
}
