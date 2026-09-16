package org.boofcv.relocalization;

import org.junit.jupiter.api.Test;

import javax.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The acceptance gate (design §8) on the dual-view evidence model, the provenance rule, and the
 * configurable temporal requirement (tests K, L, M). T1.
 */
public class AcceptanceGateTest {

    private static RelocalizationConfig config() {
        RelocalizationConfig c = new RelocalizationConfig();
        c.matchThreshold = 0.9;
        c.marginThreshold = 0.15;
        c.confirmQueries = 2;
        c.searchExposureBound = 1000;
        return c;
    }

    private static RelocalizationConfig config(String rule, String temporal) {
        RelocalizationConfig c = config();
        c.fusionRule = rule;
        c.temporalConfirmation = temporal;
        return c;
    }

    /** A memory holding references 0..n-1 so the provenance rule can be exercised. */
    private static ReferenceMemory memoryWith(int n, RelocalizationConfig c) {
        ReferenceMemory m = new ReferenceMemory(c);
        NavigationAligner a = new NavigationAligner();
        a.observe(TestSamples.ok(0, 0, 0, 0, 10.0));
        for (int i = 0; i < n; i++) {
            NavigationOutput out = a.observe(TestSamples.ok(i + 1, 0, i, 0, 10.0));
            InsertionDecision d = m.offer(TestProfiles.observation(i + 1, (i + 1) * 0.1, i % 4, 3.0 * i), out);
            assertTrue(d.inserted(), "" + d.outcome());
        }
        return m;
    }

    private static RetrievalResult.ViewRanking ranking(String view, int bestId, double top,
                                                       @Nullable Integer competingId, @Nullable Double competing) {
        List<RetrievalResult.Region> regions = new ArrayList<>();
        List<RetrievalResult.Candidate> cands = new ArrayList<>();
        regions.add(new RetrievalResult.Region(0, bestId, top, List.of(bestId), new PlanarPosition(bestId, 0)));
        cands.add(new RetrievalResult.Candidate(1, bestId, top, 0, 1.0, null, 1, 0.1, 0, 0));
        if (competingId != null) {
            regions.add(new RetrievalResult.Region(1, competingId, competing, List.of(competingId),
                    new PlanarPosition(competingId, 0)));
            cands.add(new RetrievalResult.Candidate(2, competingId, competing, 0, 1.0, null, 2, 0.2, 0, 1));
        }
        Double margin = competing == null ? null : top - competing;
        return new RetrievalResult.ViewRanking(view, cands, regions, top, competing, margin, List.of(), 10);
    }

    /** A North-only retrieval. */
    private static RetrievalResult retrieval(int bestId, double top, @Nullable Integer competingId,
                                             @Nullable Double competing) {
        RetrievalResult.ViewRanking north = ranking(SkylineView.NORTH, bestId, top, competingId, competing);
        RetrievalResult.DualRegionEvidence d = new RetrievalResult.DualRegionEvidence(bestId, 0, top, null,
                north.regionMargin(), null, null, null, bestId, null, null, null,
                RetrievalResult.DualStatus.NOT_USED);
        return new RetrievalResult(50, 5.0, RelocalizationConfig.MATCHER_C1, RelocalizationConfig.FUSION_NORTH_ONLY,
                10, 8, List.of(), north, null, null, false, false, 0, d, null);
    }

    /** A dual retrieval under {@code rule}, with an optional West ranking and fused ranking. */
    private static RetrievalResult dual(String rule, RetrievalResult.ViewRanking north,
                                        @Nullable RetrievalResult.ViewRanking west,
                                        @Nullable RetrievalResult.ViewRanking fused,
                                        @Nullable Boolean agreement, RetrievalResult.DualStatus status) {
        RetrievalResult.ViewRanking primary = fused != null ? fused : north;
        int cand = primary.top().referenceId();
        RetrievalResult.DualRegionEvidence d = new RetrievalResult.DualRegionEvidence(cand, 0,
                north.candidate(cand) == null ? null : north.candidate(cand).score(),
                west == null || west.candidate(cand) == null ? null : west.candidate(cand).score(),
                north.regionMargin(), west == null ? null : west.regionMargin(),
                fused == null ? null : fused.topRegionScore(), fused == null ? null : fused.regionMargin(),
                north.top().referenceId(), west == null ? null : west.top().referenceId(), agreement,
                agreement == null ? null : 0.0, status);
        return new RetrievalResult(50, 5.0, RelocalizationConfig.MATCHER_C1, rule, 10, 8, List.of(), north, west,
                fused, west != null || status == RetrievalResult.DualStatus.WEST_INVALID, west != null, 8, d, null);
    }

    private static RegionTracker.Track track(int bestId, int support) {
        return new RegionTracker.Track(bestId, List.of(bestId), new PlanarPosition(bestId, 0), support, 40, 50,
                0.95, RegionTracker.Traversal.STATIONARY);
    }

    // ---------------------------------------------------------------- the North-only gate

    @Test
    public void weakMatchIsRejected() {
        RelocalizationConfig c = config();
        AcceptanceGate g = new AcceptanceGate(c);
        AcceptanceGate.Decision d = g.evaluate(retrieval(2, 0.85, 5, 0.2), track(2, 5), memoryWith(6, c));
        assertEquals(AcceptanceGate.Verdict.REJECT, d.verdict());
        assertEquals(AcceptanceGate.Reason.WEAK_MATCH, d.reason());
        assertEquals(2, d.candidateReferenceId());
    }

    @Test
    public void ambiguousRegionIsRetainedNotAccepted() {
        RelocalizationConfig c = config();
        AcceptanceGate g = new AcceptanceGate(c);
        AcceptanceGate.Decision d = g.evaluate(retrieval(2, 0.95, 5, 0.9), track(2, 5), memoryWith(6, c));
        assertEquals(AcceptanceGate.Verdict.RETAIN, d.verdict());
        assertEquals(AcceptanceGate.Reason.AMBIGUOUS_REGION, d.reason());
        assertEquals(0.05, d.margin(), 1e-12);
        assertFalse(d.accepted());
    }

    @Test
    public void insufficientTemporalSupportIsRetained() {
        RelocalizationConfig c = config();
        AcceptanceGate g = new AcceptanceGate(c);
        ReferenceMemory m = memoryWith(6, c);
        AcceptanceGate.Decision first = g.evaluate(retrieval(2, 0.95, 5, 0.3), track(2, 1), m);
        assertEquals(AcceptanceGate.Verdict.RETAIN, first.verdict());
        assertEquals(AcceptanceGate.Reason.UNCONFIRMED, first.reason());
        assertTrue(first.temporalRequired());
        AcceptanceGate.Decision none = g.evaluate(retrieval(2, 0.95, 5, 0.3), null, m);
        assertEquals(AcceptanceGate.Reason.UNCONFIRMED, none.reason());
        AcceptanceGate.Decision otherTrack = g.evaluate(retrieval(2, 0.95, 5, 0.3), track(9, 4), m);
        assertEquals(AcceptanceGate.Reason.UNCONFIRMED, otherTrack.reason(), "support belongs to another region");
    }

    @Test
    public void allConditionsTogetherAccept() {
        RelocalizationConfig c = config();
        AcceptanceGate g = new AcceptanceGate(c);
        AcceptanceGate.Decision d = g.evaluate(retrieval(2, 0.95, 5, 0.3), track(2, 2), memoryWith(6, c));
        assertTrue(d.accepted());
        assertEquals(AcceptanceGate.Reason.ACCEPTED, d.reason());
        assertEquals(2, d.supportCount());
        assertEquals(AcceptanceGate.DualVerdict.NOT_USED, d.dualVerdict());
    }

    @Test
    public void noCompetingHypothesisIsRetainedHoweverStrongTheWinnerIs() {
        // Until 2026-09-08 a null margin satisfied condition (2), so this same call accepted.
        // "No competitor was available" is not "the winner beat the alternatives", and a discrete
        // pose correction may not be authorised by condition (1) alone.
        RelocalizationConfig c = config();
        AcceptanceGate g = new AcceptanceGate(c);
        ReferenceMemory m = memoryWith(6, c);
        for (double score : new double[]{0.95, 0.99, 0.999, 1.0}) {
            for (int support : new int[]{0, 2, 9}) {
                AcceptanceGate.Decision solo =
                        g.evaluate(retrieval(2, score, null, null), track(2, support), m);
                assertEquals(AcceptanceGate.Verdict.RETAIN, solo.verdict(),
                        "score " + score + ", support " + support);
                assertEquals(AcceptanceGate.Reason.NO_COMPETING_HYPOTHESIS, solo.reason());
                assertNull(solo.margin(), "there is no margin to report, and none is invented");
                assertFalse(solo.accepted());
            }
        }
        // And such a query cannot lend its region temporal support either.
        assertFalse(g.primaryPassesGate(retrieval(2, 1.0, null, null)),
                "an unconfirmable query must not build a confirmation chain");
        assertTrue(g.primaryPassesGate(retrieval(2, 0.95, 5, 0.3)),
                "while a query that really did beat a competitor still does");
    }

    @Test
    public void noCandidateAndUntrustedCandidateAreRejected() {
        RelocalizationConfig c = config();
        AcceptanceGate g = new AcceptanceGate(c);
        ReferenceMemory m = memoryWith(3, c);
        RetrievalResult empty = new RetrievalResult(1, 0.1, RelocalizationConfig.MATCHER_C1, "", 3, 0,
                List.of(), null, null, null, false, false, 0, null, "empty database");
        assertEquals(AcceptanceGate.Reason.NO_CANDIDATE, g.evaluate(empty, null, m).reason());
        AcceptanceGate.Decision untrusted = g.evaluate(retrieval(42, 0.99, null, null), track(42, 9), m);
        assertEquals(AcceptanceGate.Verdict.REJECT, untrusted.verdict());
        assertEquals(AcceptanceGate.Reason.NOT_TRUSTED, untrusted.reason());
    }

    // ---------------------------------------------------------------- the dual-view condition

    @Test
    public void strictAgreementAcceptsOnOneCaptureWhenBothViewsPassAndAgree() {
        RelocalizationConfig c = config(RelocalizationConfig.FUSION_STRICT, RelocalizationConfig.TEMPORAL_FALLBACK);
        AcceptanceGate g = new AcceptanceGate(c);
        RetrievalResult r = dual(c.fusionRule, ranking("north", 2, 0.95, 5, 0.3), ranking("west", 2, 0.93, 6, 0.4),
                null, true, RetrievalResult.DualStatus.COMPLETE);
        AcceptanceGate.Decision d = g.evaluate(r, null, memoryWith(6, c));
        assertTrue(d.accepted(), d.reason().toString());
        assertEquals(AcceptanceGate.DualVerdict.AGREED, d.dualVerdict());
        assertFalse(d.temporalRequired(), "agreement confirmed the region; temporal is the fallback only");
        assertEquals(0.93, d.westScore(), 0.0);
        assertEquals(Boolean.TRUE, d.agreement());
    }

    /** Test L: two confident views naming different regions are never accepted. */
    @Test
    public void disagreeingViewsAreRetainedNeverAccepted() {
        for (String temporal : new String[]{RelocalizationConfig.TEMPORAL_REQUIRED,
                RelocalizationConfig.TEMPORAL_FALLBACK, RelocalizationConfig.TEMPORAL_DISABLED}) {
            RelocalizationConfig c = config(RelocalizationConfig.FUSION_STRICT, temporal);
            AcceptanceGate g = new AcceptanceGate(c);
            RetrievalResult r = dual(c.fusionRule, ranking("north", 2, 0.99, 5, 0.3), ranking("west", 5, 0.98, 2, 0.4),
                    null, false, RetrievalResult.DualStatus.COMPLETE);
            AcceptanceGate.Decision d = g.evaluate(r, track(2, 9), memoryWith(6, c));
            assertEquals(AcceptanceGate.Verdict.RETAIN, d.verdict(), temporal);
            assertEquals(AcceptanceGate.Reason.VIEW_DISAGREEMENT, d.reason(), temporal);
            assertEquals(AcceptanceGate.DualVerdict.DISAGREE, d.dualVerdict());
            assertFalse(d.accepted());
        }
    }

    /**
     * EXP-SKY-012 R2: on the flat city the West's protection is refusal — on every North false accept
     * it failed its own gate. A strict rule that fell back to North persistence on refusal would have
     * accepted them (R5: North k = 3 still 30 / 87). So a refusing West retains, in every temporal mode,
     * whatever the support.
     */
    @Test
    public void aWestBelowItsOwnGateIsARefusalThatNoNorthPersistenceOverrides() {
        for (String temporal : new String[]{RelocalizationConfig.TEMPORAL_REQUIRED,
                RelocalizationConfig.TEMPORAL_FALLBACK, RelocalizationConfig.TEMPORAL_DISABLED}) {
            RelocalizationConfig c = config(RelocalizationConfig.FUSION_STRICT, temporal);
            AcceptanceGate g = new AcceptanceGate(c);
            ReferenceMemory m = memoryWith(6, c);
            RetrievalResult r = dual(c.fusionRule, ranking("north", 2, 0.95, 5, 0.3), ranking("west", 2, 0.80, 5, 0.5),
                    null, true, RetrievalResult.DualStatus.COMPLETE);
            for (int support : new int[]{0, 1, 2, 9}) {
                AcceptanceGate.Decision d = g.evaluate(r, support == 0 ? null : track(2, support), m);
                assertEquals(AcceptanceGate.Verdict.RETAIN, d.verdict(), temporal + " support " + support);
                assertEquals(AcceptanceGate.Reason.WEST_REFUSED, d.reason(), temporal + " support " + support);
                assertEquals(AcceptanceGate.DualVerdict.WEST_BELOW_GATE, d.dualVerdict());
                assertFalse(d.temporalRequired(), "temporal persistence was not consulted as a substitute");
                assertEquals(support, d.supportCount(), "the tracker's support is still reported, for the record");
            }
        }
    }

    /**
     * Test J/L: a missing or invalid West is incomplete evidence — never agreement, never a pose
     * action, and under a dual rule never accepted through North-only persistence in any temporal
     * mode (EXP-SKY-011 R6, EXP-SKY-012 R5). The single-view path exists only under north_only.
     */
    @Test
    public void aMissingOrInvalidWestIsNeverAgreementAndNeverAcceptedUnderADualRule() {
        ReferenceMemory m = memoryWith(6, config());
        RetrievalResult.ViewRanking north = ranking("north", 2, 0.95, 5, 0.3);   // strong, unambiguous North

        for (String rule : new String[]{RelocalizationConfig.FUSION_STRICT, RelocalizationConfig.FUSION_WEAKEST,
                RelocalizationConfig.FUSION_MEAN}) {
            for (String temporal : new String[]{RelocalizationConfig.TEMPORAL_REQUIRED,
                    RelocalizationConfig.TEMPORAL_FALLBACK, RelocalizationConfig.TEMPORAL_DISABLED}) {
                AcceptanceGate g = new AcceptanceGate(config(rule, temporal));
                for (RetrievalResult.DualStatus status : new RetrievalResult.DualStatus[]{
                        RetrievalResult.DualStatus.WEST_UNAVAILABLE, RetrievalResult.DualStatus.WEST_INVALID}) {
                    RetrievalResult r = dual(rule, north, null, null, null, status);
                    for (int support : new int[]{0, 2, 9}) {
                        AcceptanceGate.Decision d = g.evaluate(r, support == 0 ? null : track(2, support), m);
                        String where = rule + "/" + temporal + "/" + status + "/support " + support;
                        assertEquals(AcceptanceGate.Verdict.RETAIN, d.verdict(), where);
                        assertEquals(AcceptanceGate.Reason.WEST_INCOMPLETE, d.reason(), where);
                        assertEquals(AcceptanceGate.DualVerdict.INCOMPLETE, d.dualVerdict(), where);
                        assertNull(d.agreement(), where);
                        assertFalse(d.accepted(), where);
                    }
                }
            }
        }

        // The only rule under which North alone can accept is north_only — and there the temporal
        // requirement is what the mode says.
        RelocalizationConfig single = config(RelocalizationConfig.FUSION_NORTH_ONLY, RelocalizationConfig.TEMPORAL_FALLBACK);
        AcceptanceGate gs = new AcceptanceGate(single);
        assertEquals(AcceptanceGate.Reason.UNCONFIRMED, gs.evaluate(retrieval(2, 0.95, 5, 0.3), track(2, 1), m).reason());
        assertTrue(gs.evaluate(retrieval(2, 0.95, 5, 0.3), track(2, 2), m).accepted());
        assertTrue(single.singleViewAcceptance().startsWith("after 2"));
        RelocalizationConfig dualRule = config(RelocalizationConfig.FUSION_WEAKEST, RelocalizationConfig.TEMPORAL_FALLBACK);
        assertTrue(dualRule.singleViewAcceptance().startsWith("never"));
    }

    /** Under a dual rule the three temporal modes differ only in whether an AGREED capture also needs a chain. */
    @Test
    public void underADualRuleTemporalModesOnlyAddAChainToAnAgreedCapture() {
        ReferenceMemory m = memoryWith(6, config());
        RetrievalResult agreed = dual(RelocalizationConfig.FUSION_WEAKEST, ranking("north", 2, 0.99, 5, 0.3),
                ranking("west", 2, 0.92, 5, 0.50), ranking("fused", 2, 0.92, 5, 0.50), true,
                RetrievalResult.DualStatus.COMPLETE);
        AcceptanceGate required = new AcceptanceGate(config(RelocalizationConfig.FUSION_WEAKEST, RelocalizationConfig.TEMPORAL_REQUIRED));
        AcceptanceGate fallback = new AcceptanceGate(config(RelocalizationConfig.FUSION_WEAKEST, RelocalizationConfig.TEMPORAL_FALLBACK));
        AcceptanceGate disabled = new AcceptanceGate(config(RelocalizationConfig.FUSION_WEAKEST, RelocalizationConfig.TEMPORAL_DISABLED));

        AcceptanceGate.Decision r1 = required.evaluate(agreed, track(2, 1), m);
        assertEquals(AcceptanceGate.Reason.UNCONFIRMED, r1.reason(), "required: dual AND chain");
        assertTrue(r1.temporalRequired());
        assertTrue(required.evaluate(agreed, track(2, 2), m).accepted());

        AcceptanceGate.Decision f = fallback.evaluate(agreed, null, m);
        assertTrue(f.accepted(), "fallback: one agreeing dual capture accepts (EXP-SKY-012: false-free where North k=3 was not)");
        assertFalse(f.temporalRequired());
        assertTrue(disabled.evaluate(agreed, null, m).accepted());
    }

    /** Test M: the temporal requirement is a switch on the gate; the candidate state is untouched. */
    @Test
    public void theTemporalRequirementSwitchesWithoutChangingTheCandidateState() {
        ReferenceMemory m = memoryWith(6, config());
        RetrievalResult r = retrieval(2, 0.95, 5, 0.3);
        RegionTracker.Track t = track(2, 1);

        AcceptanceGate required = new AcceptanceGate(config(RelocalizationConfig.FUSION_NORTH_ONLY,
                RelocalizationConfig.TEMPORAL_REQUIRED));
        AcceptanceGate fallback = new AcceptanceGate(config(RelocalizationConfig.FUSION_NORTH_ONLY,
                RelocalizationConfig.TEMPORAL_FALLBACK));
        AcceptanceGate disabled = new AcceptanceGate(config(RelocalizationConfig.FUSION_NORTH_ONLY,
                RelocalizationConfig.TEMPORAL_DISABLED));

        assertEquals(AcceptanceGate.Reason.UNCONFIRMED, required.evaluate(r, t, m).reason());
        assertEquals(AcceptanceGate.Reason.UNCONFIRMED, fallback.evaluate(r, t, m).reason(),
                "North alone confirms nothing, so fallback == required there");
        AcceptanceGate.Decision d = disabled.evaluate(r, t, m);
        assertTrue(d.accepted());
        assertFalse(d.temporalRequired());
        assertEquals(1, d.supportCount(), "the tracker's count is still reported, just not consulted");
        assertEquals(1, t.supportCount(), "the track object is the same, untouched");
    }

    @Test
    public void weakestViewUsesTheFusedRankingAndItsOwnMargin() {
        RelocalizationConfig c = config(RelocalizationConfig.FUSION_WEAKEST, RelocalizationConfig.TEMPORAL_DISABLED);
        AcceptanceGate g = new AcceptanceGate(c);
        // North alone would be ambiguous (0.99 vs 0.98); the fused ranking separates the places.
        RetrievalResult r = dual(c.fusionRule, ranking("north", 2, 0.99, 5, 0.98), ranking("west", 2, 0.92, 5, 0.50),
                ranking("fused", 2, 0.92, 5, 0.50), true, RetrievalResult.DualStatus.COMPLETE);
        AcceptanceGate.Decision d = g.evaluate(r, null, memoryWith(6, c));
        assertTrue(d.accepted(), d.reason().toString());
        assertEquals(0.92, d.topScore(), 0.0, "the fused score is what was gated");
        assertEquals(0.42, d.margin(), 1e-12);
        assertEquals(AcceptanceGate.DualVerdict.AGREED, d.dualVerdict());

        // Without a West view the same rule falls back to North, incomplete, and is then ambiguous.
        RetrievalResult noWest = dual(c.fusionRule, ranking("north", 2, 0.99, 5, 0.98), null, null, null,
                RetrievalResult.DualStatus.WEST_UNAVAILABLE);
        AcceptanceGate.Decision e = g.evaluate(noWest, null, memoryWith(6, c));
        assertEquals(AcceptanceGate.Reason.AMBIGUOUS_REGION, e.reason());
    }
}
