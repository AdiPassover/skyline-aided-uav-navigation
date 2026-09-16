package org.boofcv.relocalization;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Region persistence across queries, in forward, reverse and stationary traversal (design §7). T1. */
public class RegionTrackerTest {

    private static RetrievalResult result(int frame, List<Integer> topMembers, int bestId, double score,
                                          PlanarPosition bestPosition) {
        RetrievalResult.Region top = new RetrievalResult.Region(0, bestId, score, topMembers, bestPosition);
        RetrievalResult.Candidate c = new RetrievalResult.Candidate(1, bestId, score, 0, 1.0, null,
                bestId * 10, bestId, 0, 0);
        RetrievalResult.ViewRanking north = new RetrievalResult.ViewRanking(SkylineView.NORTH, List.of(c),
                List.of(top), score, null, null, List.of(), 15);
        RetrievalResult.DualRegionEvidence d = new RetrievalResult.DualRegionEvidence(bestId, 0, score, null,
                null, null, null, null, bestId, null, null, null, RetrievalResult.DualStatus.NOT_USED);
        return new RetrievalResult(frame, frame * 0.1, RelocalizationConfig.MATCHER_C1,
                RelocalizationConfig.FUSION_NORTH_ONLY, 20, 15, List.of(), north, null, null, false, false, 0,
                d, null);
    }

    private static RetrievalResult result(int frame, List<Integer> topMembers, int bestId, double score) {
        return result(frame, topMembers, bestId, score, new PlanarPosition(bestId * 10.0, 0));
    }

    private static RetrievalResult empty(int frame) {
        return new RetrievalResult(frame, frame * 0.1, RelocalizationConfig.MATCHER_C1, "", 20, 0,
                List.of(), null, null, null, false, false, 0, null, "empty database");
    }

    @Test
    public void supportCountsConsecutiveConfirmationsOfTheSameRegion() {
        RegionTracker t = new RegionTracker(2);
        RegionTracker.Track a = t.observe(result(10, List.of(5, 6), 6, 0.95));
        assertEquals(1, a.supportCount());
        assertEquals(RegionTracker.Traversal.FIRST, a.lastTraversal());
        RegionTracker.Track b = t.observe(result(11, List.of(6, 7), 7, 0.96));
        assertEquals(2, b.supportCount());
        assertEquals(RegionTracker.Traversal.FORWARD, b.lastTraversal());
        assertEquals(10, b.firstQueryFrame());
        assertEquals(11, b.lastQueryFrame());
        assertTrue(b.covers(7));
        assertTrue(b.covers(6));
        assertFalse(b.covers(9));
    }

    @Test
    public void reverseAndStationaryTraversalCountToo() {
        RegionTracker t = new RegionTracker(2);
        t.observe(result(1, List.of(8, 9), 9, 0.9));
        RegionTracker.Track r = t.observe(result(2, List.of(7, 8), 7, 0.9));
        assertEquals(2, r.supportCount());
        assertEquals(RegionTracker.Traversal.REVERSE, r.lastTraversal());
        RegionTracker.Track s = t.observe(result(3, List.of(7), 7, 0.9));
        assertEquals(3, s.supportCount());
        assertEquals(RegionTracker.Traversal.STATIONARY, s.lastTraversal());
    }

    @Test
    public void aDifferentRegionRestartsSupportAndNoCandidatesDropsTheTrack() {
        RegionTracker t = new RegionTracker(1);
        t.observe(result(1, List.of(3, 4), 4, 0.9));
        RegionTracker.Track other = t.observe(result(2, List.of(30), 30, 0.9));
        assertEquals(1, other.supportCount());
        assertEquals(30, other.bestReferenceId());
        assertNull(t.observe(empty(3)));
        assertNull(t.current());
        assertEquals(1, t.observe(result(4, List.of(30), 30, 0.9)).supportCount(), "chain was broken");
    }

    /** {@code EXP-SKY-011}'s temporal definition: every query in the window must pass the gate. */
    @Test
    public void aQueryThatFailsTheGateBreaksTheChainInsteadOfLendingSupport() {
        RegionTracker t = new RegionTracker(2);
        t.observe(result(1, List.of(5), 5, 0.95), true);
        assertNull(t.observe(result(2, List.of(5), 5, 0.6), false), "a weak query is no confirmation");
        assertEquals(1, t.observe(result(3, List.of(5), 5, 0.97), true).supportCount(), "the chain restarted");
        assertEquals(2, t.observe(result(4, List.of(6), 6, 0.96), true).supportCount());
    }

    @Test
    public void identityIsSymmetricIdProximityWithinTheConfiguredGap() {
        RegionTracker t = new RegionTracker(2);
        assertTrue(t.sameRegion(List.of(10), List.of(12)));
        assertTrue(t.sameRegion(List.of(12), List.of(10)));
        assertFalse(t.sameRegion(List.of(10), List.of(13)));
        assertTrue(t.sameRegion(List.of(1, 2), List.of(4, 9)), "any pair within the gap suffices");
        t.reset();
        assertNull(t.current());
    }

    /** Under the positional rule, identity follows the stored metric positions, not the ids. */
    @Test
    public void positionalIdentityFollowsStoredPositionsNotIds() {
        RelocalizationConfig c = new RelocalizationConfig();
        c.regionRule = RelocalizationConfig.REGION_POSITION;
        c.regionRadiusM = 15.0;
        RegionTracker t = new RegionTracker(c);
        t.observe(result(1, List.of(3), 3, 0.9, new PlanarPosition(0, 0)));
        RegionTracker.Track revisit = t.observe(result(2, List.of(40), 40, 0.9, new PlanarPosition(12, 0)));
        assertEquals(2, revisit.supportCount(), "id 40 is 12 m from id 3: the same place, forward in id");
        assertEquals(RegionTracker.Traversal.FORWARD, revisit.lastTraversal());
        RegionTracker.Track far = t.observe(result(3, List.of(41), 41, 0.9, new PlanarPosition(100, 0)));
        assertEquals(1, far.supportCount(), "adjacent id, 88 m away: a different place");
    }
}
