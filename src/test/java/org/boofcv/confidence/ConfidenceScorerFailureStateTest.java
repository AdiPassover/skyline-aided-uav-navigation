package org.boofcv.confidence;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Constitution Principle X's VO-side failure states: each is either provoked here and asserted, or
 * documented as undetectable with its blocking dependency named (feature spec FR-020, SC-005).
 *
 * <p><b>Silent absence is not permitted.</b> A state with no test and no explanation would be
 * indistinguishable from one nobody thought about, and a green suite would then be evidence of
 * coverage that does not exist. {@link #everyPrincipleXVoStateIsAccountedFor()} enforces that
 * directly, so the accounting cannot rot as the enum changes.
 *
 * <p>Evidence tier: <b>T1 (analytical)</b>.
 */
public class ConfidenceScorerFailureStateTest {

    /** The five VO-side states named in Principle X. The other three are skyline-side. */
    private static final Set<String> PRINCIPLE_X_VO_STATES = new LinkedHashSet<>(List.of(
            "insufficient tracked features",
            "poor RANSAC support",
            "high reprojection error",
            "inconsistent motion",
            "invalid homography"));

    private static ConfidenceScorer scorer() {
        return ConfidenceScorerTest.scorer();
    }

    // ---------------------------------------------------------------- online-detectable states

    @Test
    public void insufficientTrackedFeaturesIsDetectedWithItsOwnReason() {
        SignalBlock b = SignalBlock.builder()
                .trackCount(12)           // below the configured minimum of 30
                .inlierCount(11)
                .residualInlierCount(11)
                .residualMeanSqPx(0.4)
                .build();

        ConfidenceResult r = scorer().score(b, true, false);

        assertEquals(ConfidenceOutcome.REJECTED, r.outcome());
        assertEquals(ConfidenceReason.INSUFFICIENT_FEATURES, r.reason());
        assertEquals("insufficient tracked features", r.reason().principleXState());
    }

    @Test
    public void poorRansacSupportIsDistinguishableFromInsufficientFeatures() {
        // Plenty of tracks, but few of them agree with the fitted model. Previously a raw count
        // with no verdict (COMP-001 s8); now a criterion, and one that must not be confused with
        // the "not enough features to try" case.
        SignalBlock b = SignalBlock.builder()
                .trackCount(400)
                .inlierCount(40)          // ratio 0.10, below the configured 0.30
                .residualInlierCount(40)
                .residualMeanSqPx(0.4)
                .build();

        ConfidenceResult r = scorer().score(b, true, false);

        assertEquals(ConfidenceReason.POOR_INLIER_SUPPORT, r.reason());
        assertEquals("poor RANSAC support", r.reason().principleXState());
    }

    @Test
    public void highReprojectionErrorIsDetected() {
        // Previously neither detected nor reported (COMP-001 s8); reportable since DEC-VO-001 and
        // detected here. This is the only online signal that measures geometric agreement rather
        // than support count -- EXP-002 showed the estimator degrading badly while its counts
        // stayed healthy, which is the reason it matters.
        SignalBlock b = SignalBlock.builder()
                .trackCount(400)
                .inlierCount(380)
                .residualInlierCount(380)
                .residualMeanSqPx(12.0)   // above the configured 3.0
                .residualMaxSqPx(40.0)
                .build();

        ConfidenceResult r = scorer().score(b, true, false);

        assertEquals(ConfidenceReason.HIGH_REPROJECTION_ERROR, r.reason());
        assertEquals("high reprojection error", r.reason().principleXState());
    }

    @Test
    public void healthyCountsWithABadResidualStillRejects() {
        // The EXP-002 shape, in miniature: counts look fine, geometry does not. A counts-only
        // indicator would call this frame usable.
        SignalBlock b = SignalBlock.builder()
                .trackCount(2500)
                .inlierCount(2450)        // ratio 0.98
                .residualInlierCount(2450)
                .residualMeanSqPx(9.0)
                .build();

        assertEquals(ConfidenceReason.HIGH_REPROJECTION_ERROR, scorer().score(b, true, false).reason());
    }

    @Test
    public void estimatorFailureAndUnjudgeabilityAreTheirOwnStates() {
        assertEquals(ConfidenceReason.ESTIMATOR_FAILED,
                scorer().score(ConfidenceScorerTest.healthy(), false, false).reason());
        assertEquals(ConfidenceReason.SIGNALS_UNAVAILABLE,
                scorer().score(SignalBlock.UNAVAILABLE, true, false).reason());
        assertEquals(ConfidenceReason.NOT_ESTABLISHED,
                scorer().score(ConfidenceScorerTest.healthy(), true, true).reason());
    }

    // ------------------------------------------------------------------- offline-only states

    @Test
    public void inconsistentMotionIsDocumentedAsOfflineOnlyRatherThanSilentlyAbsent() {
        // NOT a skipped test: an assertion that the code is declared offline-only, so a reader of
        // a green suite cannot mistake this for online coverage.
        //
        // Why it cannot be online: the incremental frame-to-keyframe model does not cross the VO
        // boundary -- only its residuals do (DEC-VO-001). The online path therefore has no
        // frame-to-frame transform to test. The offline re-scorer composes consecutive mosaic
        // homographies from the run record instead (research.md R2), which is valid across keyframe
        // changes and invalid across a reference_id change.
        //
        // Approximating it online with something weaker under the same name is what FR-012 forbids.
        assertEquals(ConfidenceReason.Detectability.OFFLINE_ONLY,
                ConfidenceReason.INCONSISTENT_MOTION.detectability());
        assertEquals("inconsistent motion", ConfidenceReason.INCONSISTENT_MOTION.principleXState());
    }

    @Test
    public void invalidHomographyIsDocumentedAsOfflineOnlyRatherThanSilentlyAbsent() {
        // Same mechanism and same reason as inconsistent motion above.
        assertEquals(ConfidenceReason.Detectability.OFFLINE_ONLY,
                ConfidenceReason.INVALID_HOMOGRAPHY.detectability());
        assertEquals("invalid homography", ConfidenceReason.INVALID_HOMOGRAPHY.principleXState());
    }

    // ------------------------------------------------------------------------- the accounting

    @Test
    public void everyPrincipleXVoStateIsAccountedFor() {
        // SC-005, enforced rather than asserted in prose: every VO-side state named in Principle X
        // must map to exactly one reason code, and that code must declare where it is detected.
        List<String> covered = new ArrayList<>();
        for (ConfidenceReason r : ConfidenceReason.values()) {
            String state = r.principleXState();
            if (state == null) {
                continue;
            }
            assertTrue(PRINCIPLE_X_VO_STATES.contains(state),
                    "reason " + r + " claims a Principle X state this test does not know about: " + state);
            assertTrue(r.detectability() != ConfidenceReason.Detectability.NOT_A_FAILURE_STATE,
                    "reason " + r + " names a failure state but declares itself not one");
            covered.add(state);
        }

        for (String state : PRINCIPLE_X_VO_STATES) {
            assertTrue(covered.contains(state),
                    "Principle X names '" + state + "' but no reason code reports it. Either add a "
                            + "code, or record it as undetectable with its blocking dependency -- "
                            + "silent absence is not permitted (FR-020).");
        }
        assertEquals(PRINCIPLE_X_VO_STATES.size(), covered.size(), "one code per state, no duplicates");
    }

    @Test
    public void whenTwoConditionsHoldTheFirstConfiguredRuleWins() {
        // Determinism of the reason code. Both the track-count and inlier-ratio rules fire here;
        // the configured order puts track_count first, so that is the reason -- not whichever the
        // iteration order happened to reach.
        SignalBlock b = SignalBlock.builder()
                .trackCount(10)           // < 30
                .inlierCount(1)           // ratio 0.10 < 0.30
                .residualInlierCount(1)
                .residualMeanSqPx(0.5)
                .build();

        ConfidenceResult r = scorer().score(b, true, false);
        assertEquals(ConfidenceReason.INSUFFICIENT_FEATURES, r.reason(),
                "first configured matching rule must win");
        assertNull(r.score());
    }

    @Test
    public void reorderingTheRulesChangesWhichReasonIsReported() {
        // Proves the previous test is testing configuration rather than a hardcoded precedence:
        // same frame, same signals, rules swapped, different reason reported.
        ConfidenceScorer swapped = new ConfidenceScorer(
                CalibrationConfig.parse(ConfidenceScorerTest.configWithRules(
                        ConfidenceScorerTest.RULES_SWAPPED_ORDER)));

        SignalBlock b = SignalBlock.builder()
                .trackCount(10)           // < 30
                .inlierCount(1)           // ratio 0.10 < 0.30
                .residualInlierCount(1)
                .residualMeanSqPx(0.5)
                .build();

        assertEquals(ConfidenceReason.INSUFFICIENT_FEATURES, scorer().score(b, true, false).reason(),
                "default order puts track_count first");
        assertEquals(ConfidenceReason.POOR_INLIER_SUPPORT, swapped.score(b, true, false).reason(),
                "swapped order puts inlier_ratio first");
    }
}
