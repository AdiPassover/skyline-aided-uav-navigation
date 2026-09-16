package org.boofcv.relocalization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The C0 primitive mirrored from {@code skyline.descriptors.profile_distance}: NCC on mean-removed
 * profiles, degenerate profiles refused. Known answers are hand-computed. T1.
 */
public class SkylineDescriptorTest {

    private static SkylineDescriptor of(double... p) {
        return SkylineDescriptor.of(p, p.length, 1e-9);
    }

    /**
     * p = [1,2,3,4], q = [2,2,3,5]: a = [−1.5,−.5,.5,1.5], b = [−1,−1,0,2], ⟨a,b⟩ = 5,
     * ‖a‖ = √5, ‖b‖ = √6 → NCC = 5/√30.
     */
    @Test
    public void matchesThePythonFormulaOnAHandExample() {
        SkylineDescriptor p = of(1, 2, 3, 4);
        SkylineDescriptor q = of(2, 2, 3, 5);
        assertEquals(5.0 / Math.sqrt(30.0), p.ncc(q), 1e-12);
        assertEquals(p.ncc(q), q.ncc(p), 0.0, "symmetric");
        assertEquals(1.0 - 5.0 / Math.sqrt(30.0), p.distance(q), 1e-12);
    }

    @Test
    public void selfSimilarityIsOne() {
        SkylineDescriptor d = TestDescriptors.sine(0.3);
        assertEquals(1.0, d.ncc(d), 1e-12);
        assertEquals(0.0, d.distance(d), 1e-12);
    }

    /** Mean removal: a constant vertical offset (pitch/altitude) does not change the score. */
    @Test
    public void invariantToConstantOffset() {
        double[] base = TestDescriptors.sineProfile(0.1);
        double[] shifted = base.clone();
        for (int i = 0; i < shifted.length; i++) {
            shifted[i] += 0.37;
        }
        assertEquals(1.0, of(base).ncc(of(shifted)), 1e-12);
    }

    /** Norm division: amplitude is discarded, as C0 does and L1/L2 would not. */
    @Test
    public void invariantToPositiveGain() {
        double[] base = TestDescriptors.sineProfile(0.1);
        double[] scaled = base.clone();
        for (int i = 0; i < scaled.length; i++) {
            scaled[i] *= 2.5;
        }
        assertEquals(1.0, of(base).ncc(of(scaled)), 1e-12);
    }

    @Test
    public void negatedShapeScoresMinusOne() {
        double[] base = TestDescriptors.sineProfile(0.1);
        double[] neg = base.clone();
        for (int i = 0; i < neg.length; i++) {
            neg[i] = -neg[i];
        }
        assertEquals(-1.0, of(base).ncc(of(neg)), 1e-12);
    }

    @Test
    public void similarityFallsOffWithPhaseDifference() {
        SkylineDescriptor a = TestDescriptors.sine(0.0);
        double previous = 1.0;
        for (double phase : new double[]{0.01, 0.03, 0.06, 0.10, 0.15}) {
            double s = a.ncc(TestDescriptors.sine(phase));
            assertTrue(s < previous, "phase " + phase + ": " + s + " !< " + previous);
            previous = s;
        }
    }

    @Test
    public void degenerateProfilesAreRefusedNotScored() {
        double[] flat = new double[16];
        java.util.Arrays.fill(flat, 0.42);
        assertNull(SkylineDescriptor.of(flat, 16, 1e-3), "a flat skyline has nothing to match on");

        double[] nearlyFlat = flat.clone();
        nearlyFlat[3] += 1e-6;
        assertNull(SkylineDescriptor.of(nearlyFlat, 16, 1e-3), "below the floor is degenerate");
        assertNotNull(SkylineDescriptor.of(nearlyFlat, 16, 0.0), "floor 0 admits it");
    }

    @Test
    public void lengthMismatchAndNonFiniteAreConstructionErrors() {
        assertThrows(IllegalArgumentException.class, () -> SkylineDescriptor.of(new double[10], 256, 0));
        assertThrows(IllegalArgumentException.class,
                () -> SkylineDescriptor.of(new double[]{1, Double.NaN, 3}, 3, 0));
        SkylineDescriptor a = of(1, 2, 3, 4);
        SkylineDescriptor b = of(1, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> a.ncc(b));
    }

    @Test
    public void centredProfileIsACopy() {
        SkylineDescriptor d = of(1, 2, 3, 4);
        double[] c = d.centredProfile();
        c[0] = 999;
        assertEquals(-1.5, d.centredProfile()[0], 0.0);
    }
}
