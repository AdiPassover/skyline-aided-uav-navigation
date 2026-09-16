package org.boofcv.confidence;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Absence is a value, and it is not zero (feature spec FR-004).
 *
 * <p>The failure this guards against is silent and produces plausible numbers: a missing residual
 * read as {@code 0.0} looks like a <i>perfect fit</i>, and a missing track count read as {@code 0}
 * looks like total tracking failure. Both would be confident statements about a frame nobody
 * measured.
 *
 * <p>Evidence tier: <b>T1 (analytical)</b>. These validate the instrument's bookkeeping, not the VO.
 */
public class SignalBlockAbsenceTest {

    @Test
    public void unavailableHasEveryFieldAbsentRatherThanZero() {
        SignalBlock b = SignalBlock.UNAVAILABLE;

        assertNull(b.trackCount(), "track count must be absent, not 0");
        assertNull(b.inlierCount(), "inlier count must be absent, not 0");
        assertNull(b.residualInlierCount(), "residual inlier count must be absent, not 0");
        assertNull(b.residualMeanSqPx(), "mean residual must be absent, not 0.0");
        assertNull(b.residualRmsPx());
        assertNull(b.residualMedianSqPx());
        assertNull(b.residualMaxSqPx());
        assertNull(b.inlierThresholdSqPx());
        assertNull(b.inlierCoverage());
        assertNull(b.keyframeAge());
        assertTrue(b.isEmpty());
    }

    @Test
    public void absentIsDistinguishableFromAMeasuredZero() {
        SignalBlock absent = SignalBlock.UNAVAILABLE;
        SignalBlock measuredZero = SignalBlock.builder()
                .trackCount(0)
                .inlierCount(0)
                .inlierCoverage(0.0)
                .build();

        assertNotEquals(absent, measuredZero,
                "a block of measured zeros must not equal a block of absences");
        assertNull(absent.trackCount());
        assertEquals(0, measuredZero.trackCount());
        assertFalse(measuredZero.isEmpty());
    }

    @Test
    public void emptyMatchSetIsARealZeroAlongsideGenuinelyAbsentStatistics() {
        // VO's documented case: estimation succeeded over an empty match set. The count is a real
        // measurement of zero; the statistics genuinely do not exist. Both facts, separately.
        SignalBlock b = SignalBlock.builder()
                .residualInlierCount(0)
                .build();

        assertEquals(0, b.residualInlierCount(), "count of zero is a measurement");
        assertNull(b.residualMeanSqPx(), "statistics over an empty set do not exist");
        assertNull(b.residualRmsPx());
        assertNull(b.residualMaxSqPx());
    }

    @Test
    public void aStatisticWithoutAPopulationIsRejected() {
        // A mean over an empty or unknown match set is not a measurement; it is a bug upstream,
        // and it must fail loudly at construction rather than reach a score.
        assertThrows(IllegalArgumentException.class, () -> SignalBlock.builder()
                .residualInlierCount(0)
                .residualMeanSqPx(1.5)
                .build());

        assertThrows(IllegalArgumentException.class, () -> SignalBlock.builder()
                .residualMeanSqPx(1.5)
                .build());
    }

    @Test
    public void inlierRatioIsDerivedAndAbsentWhenEitherOperandIsAbsent() {
        assertNull(SignalBlock.builder().trackCount(100).build().inlierRatio(),
                "ratio with no inlier count must be absent, not 0.0");
        assertNull(SignalBlock.builder().inlierCount(50).build().inlierRatio(),
                "ratio with no track count must be absent, not 0.0");
        assertNull(SignalBlock.UNAVAILABLE.inlierRatio());

        SignalBlock b = SignalBlock.builder().trackCount(200).inlierCount(150).build();
        assertEquals(0.75, b.inlierRatio(), 0.0);
    }

    @Test
    public void inlierRatioIsAbsentRatherThanInfiniteWhenTrackCountIsZero() {
        // 0 tracks with 0 inliers is a real measurement of both, but their ratio is undefined.
        // Undefined must be absent -- never NaN, which the run record forbids as a value.
        SignalBlock b = SignalBlock.builder().trackCount(0).inlierCount(0).build();
        assertNull(b.inlierRatio(), "an undefined ratio is absent, not NaN and not 0.0");
    }

    @Test
    public void inlierRatioMatchesPlainDoubleDivision() {
        // Pinned because contracts/agreement.md requires the Python side to reproduce this
        // bitwise: any pre-rounding here would be invisible locally and fatal cross-language.
        SignalBlock b = SignalBlock.builder().trackCount(3).inlierCount(1).build();
        assertEquals(1.0 / 3.0, b.inlierRatio(), 0.0, "ratio must be plain double division");
    }

    @Test
    public void nanAndInfinityAreRejectedRatherThanTreatedAsAbsent() {
        // The run record's encoding rule: NaN/Inf are invalid values, not a way to spell absence.
        // Absence has its own representation and it is null.
        assertThrows(IllegalArgumentException.class,
                () -> SignalBlock.builder().residualMeanSqPx(Double.NaN).residualInlierCount(5).build());
        assertThrows(IllegalArgumentException.class,
                () -> SignalBlock.builder().inlierCoverage(Double.POSITIVE_INFINITY).build());
    }

    @Test
    public void negativeCountsAndResidualsAreRejectedRatherThanClamped() {
        assertThrows(IllegalArgumentException.class,
                () -> SignalBlock.builder().trackCount(-1).build());
        assertThrows(IllegalArgumentException.class,
                () -> SignalBlock.builder().residualMaxSqPx(-0.5).residualInlierCount(2).build());
        assertThrows(IllegalArgumentException.class,
                () -> SignalBlock.builder().inlierCoverage(1.5).build());
    }

    @Test
    public void signalLookupReturnsAbsenceForAbsentAndThrowsForUnknownNames() {
        SignalBlock b = SignalBlock.builder().trackCount(120).build();

        assertEquals(120.0, b.signal(SignalBlock.TRACK_COUNT), 0.0);
        assertNull(b.signal(SignalBlock.RESIDUAL_MEAN_SQ_PX), "absent signal reads as absent");
        assertFalse(b.hasSignal(SignalBlock.RESIDUAL_MEAN_SQ_PX));
        assertTrue(b.hasSignal(SignalBlock.TRACK_COUNT));

        // A typo must not be indistinguishable from a genuinely absent measurement.
        assertThrows(IllegalArgumentException.class, () -> b.signal("track_cout"));
    }

    @Test
    public void keyframeAgeIsModelledAndCurrentlyAlwaysAbsent() {
        // Not observable: WrapImageMotionPtkSmartRespawn.alg is package-private and no method on
        // its public surface reports keyframe state. Modelled anyway so the confound stays visible
        // in every run rather than being an omission a reader would have to already know about.
        assertNull(SignalBlock.UNAVAILABLE.keyframeAge());
        assertNull(SignalBlock.builder().trackCount(100).build().keyframeAge());
    }
}
