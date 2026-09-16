package org.boofcv.relocalization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * C1 bounded-lag NCC semantics, on constructed truths (the Python-parity check is
 * {@link C1ParityTest}). Sign convention, as in {@code hsreloc.matchers}: {@code q[i]} is compared
 * with {@code r[i + lag]}, so a reference that is the query shifted <em>left</em> by 5 samples
 * ({@code r(t) = q(t + 5)}) wins at lag {@code -5}, and one shifted right wins at {@code +5}. T1.
 */
public class SkylineMatcherTest {

    @Test
    public void identicalProfilesScoreOneAtLagZeroInBothVariants() {
        SkylineDescriptor q = TestProfiles.descriptor(0, 0);
        MatchResult c0 = SkylineMatcher.frozenNcc().match(q, q);
        MatchResult c1 = SkylineMatcher.boundedLagNcc(32, 0.6).match(q, q);
        assertEquals(1.0, c0.score(), 1e-12);
        assertEquals(1.0, c1.score(), 1e-12);
        assertEquals(0, c1.shiftSamples());
        assertEquals(TestProfiles.N, c1.overlap());
        assertEquals(1.0, c1.overlapFraction(), 0.0);
        assertTrue(c1.accepted());
        assertEquals(65, c1.alignmentsSearched(), "all 2·32+1 lags keep overlap ≥ 0.6·256");
    }

    @Test
    public void aShiftedCopyIsRecoveredExactlyWithTheDocumentedSign() {
        SkylineDescriptor q = TestProfiles.descriptor(1, 0);
        SkylineDescriptor left = TestProfiles.descriptor(1, +5);    // r(t) = q(t + 5)
        SkylineDescriptor right = TestProfiles.descriptor(1, -7);   // r(t) = q(t − 7)
        SkylineMatcher c1 = SkylineMatcher.boundedLagNcc(32, 0.6);

        MatchResult a = c1.match(q, left);
        assertEquals(-5, a.shiftSamples());
        assertEquals(1.0, a.score(), 1e-12, "exact: the samples coincide at that lag");
        assertEquals(TestProfiles.N - 5, a.overlap());

        MatchResult b = c1.match(q, right);
        assertEquals(7, b.shiftSamples());
        assertEquals(1.0, b.score(), 1e-12);

        // C0 has no lag freedom and scores the same pairs strictly lower.
        assertTrue(SkylineMatcher.frozenNcc().match(q, left).score() < 0.99);
        assertNotNull(a.scoreAtZeroLag());
        assertEquals(SkylineMatcher.frozenNcc().match(q, left).score(), a.scoreAtZeroLag(), 1e-12,
                "the lag-0 score is the C0 score");
    }

    @Test
    public void aShiftBeyondTheBoundIsNotRecoveredAndTheBoundIsRespected() {
        SkylineDescriptor q = TestProfiles.descriptor(2, 0);
        SkylineDescriptor far = TestProfiles.descriptor(2, +40);
        MatchResult m = SkylineMatcher.boundedLagNcc(8, 0.6).match(q, far);
        assertTrue(Math.abs(m.shiftSamples()) <= 8);
        assertTrue(m.score() < 0.999, "cannot reach the true alignment: " + m.score());
        assertEquals(17, m.alignmentsSearched());
    }

    @Test
    public void theOverlapFloorSkipsAlignmentsThatCompareTooFewSamples() {
        SkylineDescriptor q = TestProfiles.descriptor(3, 0);
        // floor = ceil(0.9 · 256) = 231 → only |lag| ≤ 25 keeps 256 − |lag| ≥ 231
        MatchResult m = SkylineMatcher.boundedLagNcc(32, 0.9).match(q, TestProfiles.descriptor(3, +3));
        assertEquals(51, m.alignmentsSearched());
        assertEquals(-3, m.shiftSamples());
    }

    @Test
    public void lagZeroIsAlwaysAdmissibleSoC1NeverRefusesAWholeProfile() {
        SkylineDescriptor q = TestProfiles.descriptor(0, 0);
        MatchResult m = SkylineMatcher.boundedLagNcc(8, 1.0).match(q, TestProfiles.descriptor(1, 0));
        assertTrue(m.accepted());
        assertEquals(1, m.alignmentsSearched(), "min_overlap 1.0 admits lag 0 only");
        assertEquals(0, m.shiftSamples());
    }

    @Test
    public void unrelatedPlacesScoreWellBelowTheAcceptanceThresholdEvenWithLagFreedom() {
        SkylineMatcher c1 = SkylineMatcher.boundedLagNcc(32, 0.6);
        for (int a = 0; a < 4; a++) {
            for (int b = 0; b < 4; b++) {
                if (a == b) continue;
                double s = c1.match(TestProfiles.descriptor(a, 0), TestProfiles.descriptor(b, 0)).score();
                assertTrue(s < 0.75, "families " + a + "/" + b + " scored " + s);
            }
        }
    }

    @Test
    public void configSelectsTheVariantAndMismatchedLengthsAreRefused() {
        RelocalizationConfig c = new RelocalizationConfig();
        assertEquals(RelocalizationConfig.MATCHER_C1, SkylineMatcher.fromConfig(c).variant());
        c.matcher = RelocalizationConfig.MATCHER_C0;
        assertEquals(RelocalizationConfig.MATCHER_C0, SkylineMatcher.fromConfig(c).variant());
        SkylineDescriptor a = TestProfiles.descriptor(0, 0);
        SkylineDescriptor b = SkylineDescriptor.of(new double[]{1, 2, 3, 5}, 4, 0.0);
        assertThrows(IllegalArgumentException.class, () -> SkylineMatcher.boundedLagNcc(2, 0.5).match(a, b));
        assertThrows(IllegalArgumentException.class, () -> SkylineMatcher.boundedLagNcc(-1, 0.5));
        assertThrows(IllegalArgumentException.class, () -> SkylineMatcher.boundedLagNcc(1, 0.0));
    }
}
