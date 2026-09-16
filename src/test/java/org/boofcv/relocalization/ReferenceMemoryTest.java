package org.boofcv.relocalization;

import org.boofcv.stitching.metric.HeadingStatus;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Insertion policy, reference pose semantics, recent-reference exclusion, exact per-view top-K
 * retrieval with region-level margins, and the dual-view banks (design §6, §7;
 * {@code DEC-INT-003}; tests H, J, K). Every expected ranking is known from how the synthetic
 * profiles were built. T1.
 */
public class ReferenceMemoryTest {

    /** The P0 memory tests use the pointwise C0 matcher: they are about the rules, not the lag. */
    private static RelocalizationConfig config() {
        RelocalizationConfig c = new RelocalizationConfig();
        c.descriptorLength = TestDescriptors.N;
        c.degenerateStdFloor = TestDescriptors.FLOOR;
        c.matcher = RelocalizationConfig.MATCHER_C0;
        c.noveltyMinDistance = 0.05;
        c.maxSpacingFrames = 100;
        c.searchExposureBound = 1000;
        c.recentExclusionMode = RelocalizationConfig.MODE_COUNT;
        c.recentExclusionCount = 2;
        c.recentExclusionSeconds = null;
        c.regionGapReferences = 1;
        c.topK = 5;
        return c;
    }

    /** A running aligner on the metric contract, so the memory sees genuine {@link NavigationOutput}s. */
    private static final class World {
        final NavigationAligner aligner = new NavigationAligner();
        int frame = -1;
        int segment = 0;

        NavigationOutput step(boolean success) {
            frame++;
            if (!success) {
                segment++;
                return aligner.observe(TestSamples.loss(frame, segment, 0.1 * frame));
            }
            return aligner.observe(TestSamples.ok(frame, segment, 2.0 * frame, 0.5 * frame, 0.1 * frame));
        }

        NavigationOutput advance(int frames) {
            NavigationOutput o = null;
            for (int i = 0; i < frames; i++) {
                o = step(true);
            }
            return o;
        }

        SkylineObservation obs(double phase) {
            return SkylineObservation.valid(frame, frame * 0.1, TestDescriptors.sine(phase));
        }
    }

    // ---------------------------------------------------------------- insertion and pose semantics

    /** Test H: a reference stores metres ENU, the authoritative heading, and its schema. */
    @Test
    public void firstValidObservationBecomesReferenceZeroWithMetricPoseAndProvenance() {
        ReferenceMemory m = new ReferenceMemory(config());
        World w = new World();
        NavigationOutput nav = w.advance(6);

        InsertionDecision d = m.offer(w.obs(0.0), nav);

        assertTrue(d.inserted());
        TrustedReference r = d.reference();
        assertEquals(0, r.id());
        assertEquals(5, r.frameIndex());
        assertEquals(nav.globalPosition().orElseThrow(), r.positionGlobal(), "metres ENU, the persistent position");
        assertEquals(10.0, r.positionGlobal().eastM(), 0.0);
        assertEquals(nav.headingDeg(), r.headingDeg(), 0.0, "the VO's authoritative heading");
        assertEquals(HeadingStatus.FRESH, r.headingStatus());
        assertTrue(r.headingValid());
        assertEquals(TrustedReference.POSE_SCHEMA, r.poseSchema());
        assertEquals(0, r.segmentId());
        assertEquals(0, r.alignmentEpochId());
        assertNull(r.sourceAnchorId(), "root anchor");
        assertNull(r.sourceReanchorFrameIndex());
        assertFalse(r.descendsFromRelocalization());
        assertFalse(r.hasWest(), "a North-only observation stores no West");
        assertEquals(nav.lineage().effectiveExposure(), r.baselineExposure());
        assertFalse(r.baselineInstability());
        assertNull(d.noveltyDistance(), "nothing to compare with yet");
        assertEquals(1, m.size());
    }

    /** Test H: an old pixel / visual-yaw record cannot be constructed, let alone reinterpreted. */
    @Test
    public void aLegacyPoseRecordIsRefusedNotReinterpreted() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new TrustedReference(0, 1, 0.1, TestDescriptors.sine(0.1), null,
                        new PlanarPosition(100, 200), 45.0, HeadingStatus.FRESH, 0, 0, null, null, 0, false,
                        "int-reference-pose/1: first-frame pixels, start-relative visual yaw"));
        assertTrue(e.getMessage().contains("incompatible reference pose record"));
        // And a heading that claims validity without a number, or vice versa, is refused too.
        assertThrows(IllegalArgumentException.class,
                () -> new TrustedReference(0, 1, 0.1, TestDescriptors.sine(0.1), null,
                        new PlanarPosition(1, 1), Double.NaN, HeadingStatus.FRESH, 0, 0, null, null, 0, false,
                        TrustedReference.POSE_SCHEMA));
    }

    @Test
    public void invalidSkylineIsRefusedFirst() {
        ReferenceMemory m = new ReferenceMemory(config());
        World w = new World();
        NavigationOutput nav = w.advance(3);
        InsertionDecision d = m.offer(SkylineObservation.invalid(w.frame, w.frame * 0.1, "extraction refused"), nav);
        assertEquals(InsertionDecision.Outcome.REJECTED_SKYLINE_INVALID, d.outcome());
        assertEquals(0, m.size());
    }

    @Test
    public void unknownGlobalPositionRefusesInsertion() {
        ReferenceMemory m = new ReferenceMemory(config());
        World w = new World();
        w.advance(3);
        NavigationOutput lost = w.step(false);
        assertFalse(lost.globalPositionValid());
        assertTrue(lost.headingValid(), "the heading alone does not make a reference storable");
        InsertionDecision d = m.offer(w.obs(0.0), lost);
        assertEquals(InsertionDecision.Outcome.REJECTED_GLOBAL_POSITION_INVALID, d.outcome());
        NavigationOutput later = w.advance(5);
        assertEquals(InsertionDecision.Outcome.REJECTED_GLOBAL_POSITION_INVALID,
                m.offer(w.obs(0.3), later).outcome(), "still unknown until a re-anchor");
    }

    @Test
    public void unresolvedInstabilityRefusesInsertionUntilAnAcceptedAnchorAndDescendantsAreTraceable() {
        ReferenceMemory m = new ReferenceMemory(config());
        World w = new World();
        w.advance(3);
        w.aligner.noteInstability();
        NavigationOutput nav = w.advance(40);
        assertEquals(InsertionDecision.Outcome.REJECTED_UNRESOLVED_INSTABILITY,
                m.offer(w.obs(0.0), nav).outcome());

        // A trusted anchor clears I_since; insertion becomes possible again, and the new reference
        // records exactly which accepted event established its lineage.
        w.aligner.acceptReference(new TrustedReference(99, 1, 0.1, TestDescriptors.sine(0.9), null,
                new PlanarPosition(1, 1), 1.0, HeadingStatus.FRESH, 0, 0, null, null, 3, false,
                TrustedReference.POSE_SCHEMA));
        int reanchorFrame = w.frame;
        NavigationOutput after = w.advance(2);
        assertTrue(m.offer(w.obs(0.0), after).inserted());
        assertEquals(99, m.get(0).sourceAnchorId());
        assertEquals(reanchorFrame, m.get(0).sourceReanchorFrameIndex());
        assertTrue(m.get(0).descendsFromRelocalization());
    }

    @Test
    public void exposureBeyondTheSearchBoundRefusesTrustedInsertion() {
        RelocalizationConfig c = config();
        c.searchExposureBound = 10;
        ReferenceMemory m = new ReferenceMemory(c);
        World w = new World();
        NavigationOutput atBound = w.advance(11);          // E_eff = 10
        assertTrue(m.offer(w.obs(0.0), atBound).inserted(), "E_eff == bound is still allowed");
        NavigationOutput over = w.advance(1);              // E_eff = 11
        assertEquals(InsertionDecision.Outcome.REJECTED_EXPOSURE_BEYOND_SEARCH_BOUND,
                m.offer(w.obs(0.5), over).outcome());
    }

    @Test
    public void noveltyOrMaxSpacingGovernsInsertion() {
        ReferenceMemory m = new ReferenceMemory(config());
        World w = new World();
        NavigationOutput nav = w.advance(1);
        assertTrue(m.offer(w.obs(0.0), nav).inserted());

        nav = w.advance(10);
        InsertionDecision same = m.offer(w.obs(0.0), nav);
        assertEquals(InsertionDecision.Outcome.REJECTED_NOT_NOVEL, same.outcome());
        assertEquals(0.0, same.noveltyDistance(), 1e-12);
        assertEquals(10, same.framesSinceLast());

        nav = w.advance(1);
        InsertionDecision novel = m.offer(w.obs(0.25), nav);
        assertTrue(novel.inserted(), "a different view is novel");
        assertTrue(novel.noveltyDistance() >= 0.05);

        nav = w.advance(100);
        InsertionDecision spaced = m.offer(w.obs(0.25), nav);
        assertTrue(spaced.inserted(), "identical view, but max spacing reached");
        assertEquals(0.0, spaced.noveltyDistance(), 1e-12);
        assertEquals(100, spaced.framesSinceLast());
        assertEquals(3, m.size());
    }

    /** Test J: the West view is stored when valid, omitted when unavailable or invalid, never copied. */
    @Test
    public void theWestViewIsStoredWhenValidOmittedOtherwiseAndNeverCopiedFromNorth() {
        RelocalizationConfig c = config();
        c.descriptorLength = TestProfiles.N;
        c.degenerateStdFloor = 1e-9;
        ReferenceMemory m = new ReferenceMemory(c);
        World w = new World();
        NavigationOutput nav = w.advance(2);
        TrustedReference dual = m.offer(TestProfiles.dual(w.frame, w.frame * 0.1, 0, 0, 0), nav).reference();
        assertTrue(dual.hasWest());
        assertNotSame(dual.northDescriptor(), dual.westDescriptor());
        assertTrue(dual.westDescriptor().ncc(TestProfiles.descriptor(4, 0)) > 0.999, "the West profile, not North's");
        assertTrue(dual.northDescriptor().ncc(TestProfiles.descriptor(0, 0)) > 0.999);

        nav = w.advance(10);
        TrustedReference invalidWest = m.offer(TestProfiles.westInvalid(w.frame, w.frame * 0.1, 1, 0), nav).reference();
        assertFalse(invalidWest.hasWest(), "captured but invalid: nothing is stored for it");

        nav = w.advance(10);
        TrustedReference northOnly = m.offer(TestProfiles.observation(w.frame, w.frame * 0.1, 2, 0), nav).reference();
        assertFalse(northOnly.hasWest());
    }

    // ---------------------------------------------------------------- retrieval

    private static ReferenceMemory populated(RelocalizationConfig c, World w, int n) {
        ReferenceMemory m = new ReferenceMemory(c);
        for (int i = 0; i < n; i++) {
            NavigationOutput nav = w.advance(10);
            InsertionDecision d = m.offer(w.obs(0.1 * i), nav);
            assertTrue(d.inserted(), "reference " + i + ": " + d.outcome());
        }
        return m;
    }

    @Test
    public void exactTopKIsExhaustiveDescendingAndExcludesTheRecent() {
        World w = new World();
        ReferenceMemory m = populated(config(), w, 10);   // phases 0.0 .. 0.9
        w.advance(50);

        RetrievalResult r = m.retrieve(w.obs(0.3));        // identical to reference 3

        assertNull(r.refusedReason());
        assertEquals(10, r.databaseSize());
        assertEquals(List.of(8, 9), r.excludedRecentIds());
        assertEquals(8, r.consideredCount());
        assertEquals(5, r.candidates().size());
        assertEquals(3, r.top().referenceId());
        assertEquals(1.0, r.top().score(), 1e-12);
        for (int i = 1; i < r.candidates().size(); i++) {
            assertTrue(r.candidates().get(i - 1).score() >= r.candidates().get(i).score(), "descending");
            assertEquals(i + 1, r.candidates().get(i).rank());
        }
        assertTrue(r.candidates().stream().noneMatch(c -> c.referenceId() >= 8));
        assertTrue(r.candidates().stream().anyMatch(c -> c.referenceId() == 2));
        assertTrue(r.candidates().stream().anyMatch(c -> c.referenceId() == 4));
        assertEquals(RetrievalResult.DualStatus.NOT_USED, r.dual().status());
        assertNull(r.dual().agreement(), "North-only rule: no agreement is defined");
        assertNull(r.west());
    }

    @Test
    public void timeModeExcludesByAgeOnlyAndIgnoresTheCountParameter() {
        RelocalizationConfig c = config();
        c.recentExclusionMode = RelocalizationConfig.MODE_TIME;
        c.recentExclusionCount = 4;        // must be ignored in TIME mode
        c.recentExclusionSeconds = 4.05;   // references stored within the last 4.05 s are out
        World w = new World();
        ReferenceMemory m = populated(c, w, 6);   // stored at frames 9,19,...,59 → 0.9 s .. 5.9 s
        w.advance(20);                            // query at frame 79 → 7.9 s
        RetrievalResult r = m.retrieve(w.obs(0.5));
        assertEquals(List.of(3, 4, 5), r.excludedRecentIds(), "3.9, 4.9 and 5.9 s are newer than 7.9 − 4.05");
        assertEquals(3, r.consideredCount());
        assertEquals(2, r.top().referenceId(), "the nearest non-excluded phase");
    }

    @Test
    public void countModeExcludesByCountOnlyAndTheTwoMechanismsAreNeverCombined() {
        RelocalizationConfig c = config();          // COUNT, M = 2
        World w = new World();
        ReferenceMemory m = populated(c, w, 6);
        w.advance(1);                               // the latest references are seconds old
        RetrievalResult r = m.retrieve(w.obs(0.5));
        assertEquals(List.of(4, 5), r.excludedRecentIds(), "exactly the latest two, whatever their age");

        RelocalizationConfig both = config();
        both.recentExclusionSeconds = 2.0;          // set while the mode is COUNT
        assertThrows(IllegalArgumentException.class, both::validate,
                "a second mechanism configured alongside the active one is refused, not unioned");
        RelocalizationConfig timeWithout = config();
        timeWithout.recentExclusionMode = RelocalizationConfig.MODE_TIME;
        assertThrows(IllegalArgumentException.class, timeWithout::validate, "TIME needs its seconds");
        RelocalizationConfig unknownMode = config();
        unknownMode.recentExclusionMode = "both";
        assertThrows(IllegalArgumentException.class, unknownMode::validate);
    }

    @Test
    public void c1CandidatesCarryTheirLagAndOverlapAsEvidenceOnly() {
        RelocalizationConfig c = config();
        c.recentExclusionCount = 0;
        c.matcher = RelocalizationConfig.MATCHER_C1;
        c.maxLagSamples = 16;
        c.descriptorLength = TestProfiles.N;
        c.degenerateStdFloor = 1e-9;
        ReferenceMemory m = new ReferenceMemory(c);
        World w = new World();
        for (int fam = 0; fam < 4; fam++) {
            NavigationOutput nav = w.advance(10);
            assertTrue(m.offer(TestProfiles.observation(w.frame, w.frame * 0.1, fam, 0), nav).inserted());
        }
        w.advance(5);
        // q(t) = r(t − 4): q[i] equals r[i − 4], so the winning lag is −4 (Python convention).
        RetrievalResult r = m.retrieve(TestProfiles.observation(w.frame, w.frame * 0.1, 2, -4));
        assertEquals(RelocalizationConfig.MATCHER_C1, r.matcherVariant());
        assertEquals(2, r.top().referenceId());
        assertEquals(-4, r.top().lagSamples());
        assertTrue(r.top().overlapFraction() < 1.0);
        assertNotNull(r.top().scoreAtZeroLag());
        assertTrue(r.top().scoreAtZeroLag() < r.top().score());
        assertTrue(r.north().refusedAlignmentIds().isEmpty());
    }

    @Test
    public void refusalsAreExplicit() {
        World w = new World();
        ReferenceMemory empty = new ReferenceMemory(config());
        w.advance(2);
        assertEquals("empty database", empty.retrieve(w.obs(0.0)).refusedReason());
        assertFalse(empty.retrieve(w.obs(0.0)).hasCandidates());

        ReferenceMemory m = populated(config(), w, 2);   // both excluded by count 2
        RetrievalResult allRecent = m.retrieve(w.obs(0.0));
        assertEquals("every reference excluded as recent", allRecent.refusedReason());
        assertEquals(List.of(0, 1), allRecent.excludedRecentIds());
        assertNull(allRecent.dual());

        RetrievalResult invalid = m.retrieve(SkylineObservation.invalid(w.frame, 0.0, "degenerate profile"));
        assertTrue(invalid.refusedReason().startsWith("query invalid"));
        assertEquals(2, m.retrievals().size(), "every attempt on this memory is recorded");
    }

    @Test
    public void kLargerThanTheDatabaseReturnsEverythingConsidered() {
        RelocalizationConfig c = config();
        c.topK = 50;
        World w = new World();
        ReferenceMemory m = populated(c, w, 6);
        RetrievalResult r = m.retrieve(w.obs(0.2));
        assertEquals(4, r.candidates().size());
    }

    /**
     * Region grouping: references 3 and 4 (adjacent ids) are the two best and form one region;
     * reference 7, next best, is a separate region. Top-1 vs top-2 as <em>frames</em> would call
     * 3-vs-4 an ambiguity; as regions it is not, and the competing score is reference 7's.
     */
    @Test
    public void candidatesAreGroupedIntoChronologicalRegionsWithARegionLevelMargin() {
        RelocalizationConfig c = config();
        c.recentExclusionCount = 0;
        c.topK = 3;
        ReferenceMemory m = new ReferenceMemory(c);
        World w = new World();
        // For TestDescriptors.sine, NCC(Δ) = (0.04·cos 2πΔ + 0.0025·cos 6πΔ) / 0.0425. Query phase
        // 0.0: id 4 (0.00) = 1.000, id 3 (0.05) = 0.930, id 7 (0.07) = 0.866, id 0 (0.9 ≡ −0.1) =
        // 0.743, the rest lower — so top-3 is [4, 3, 7] and 3/4 are adjacent ids.
        double[] phases = {0.9, 0.8, 0.7, 0.05, 0.0, 0.6, 0.5, 0.07, 0.4, 0.3};
        for (double phase : phases) {
            NavigationOutput nav = w.advance(10);
            assertTrue(m.offer(w.obs(phase), nav).inserted());
        }
        w.advance(5);

        RetrievalResult r = m.retrieve(w.obs(0.0));

        assertEquals(List.of(4, 3, 7), r.candidates().stream().map(RetrievalResult.Candidate::referenceId).toList());
        assertEquals(2, r.regions().size());
        RetrievalResult.Region best = r.regions().get(0);
        assertEquals(0, best.regionId());
        assertEquals(List.of(3, 4), best.memberIds());
        assertEquals(4, best.bestReferenceId());
        assertEquals(1.0, best.bestScore(), 1e-12);
        assertEquals(m.get(4).positionGlobal(), best.bestPosition());
        RetrievalResult.Region competing = r.regions().get(1);
        assertEquals(List.of(7), competing.memberIds());
        assertEquals(0, r.candidates().get(0).regionId());
        assertEquals(0, r.candidates().get(1).regionId());
        assertEquals(1, r.candidates().get(2).regionId());
        assertNotNull(r.regionMargin());
        assertEquals(best.bestScore() - competing.bestScore(), r.regionMargin(), 1e-12);
        assertTrue(r.regionMargin() > 0.0);
    }

    /**
     * Test K, the dense-memory case: when the whole top-K is one region, the competing score is
     * still the best reference <em>outside</em> that region over everything scored — a dense
     * memory must not hide its real competitor ({@code EXP-SKY-010}/{@code -011}).
     */
    @Test
    public void competingScoreIsTakenOverAllScoredReferencesNotOnlyTheTopK() {
        RelocalizationConfig c = config();
        c.recentExclusionCount = 0;
        c.topK = 3;
        c.regionGapReferences = 1;
        c.noveltyMinDistance = 0.0;                          // dense insertions of a slowly changing skyline
        ReferenceMemory m = new ReferenceMemory(c);
        World w = new World();
        double[] phases = {0.00, 0.02, 0.04, 0.06, 0.08, 0.10};
        for (double phase : phases) {
            NavigationOutput nav = w.advance(10);
            assertTrue(m.offer(w.obs(phase), nav).inserted());
        }
        w.advance(5);
        // Query phase 0.04: id 2 = 1.000; ids 1 and 3 (Δ 0.02) = 0.988; ids 0 and 4 (Δ 0.04) =
        // 0.954; id 5 (Δ 0.06) = 0.900. The top-3 is {2, 1, 3} — entirely inside the ±1 ball.
        RetrievalResult r = m.retrieve(w.obs(0.04));
        assertEquals(List.of(2, 1, 3), r.candidates().stream().map(RetrievalResult.Candidate::referenceId).toList());
        assertEquals(1, r.regions().size(), "every candidate is in the top region");
        assertNotNull(r.competingRegionScore(), "... yet a competitor exists outside it");
        assertEquals(0.954, r.competingRegionScore(), 0.002);
        assertEquals(1.0 - r.competingRegionScore(), r.regionMargin(), 1e-12);
    }

    @Test
    public void aSingleRegionHasNoCompetingScoreAndNoMargin() {
        RelocalizationConfig c = config();
        c.recentExclusionCount = 0;
        c.topK = 2;
        c.regionGapReferences = 5;
        World w = new World();
        ReferenceMemory m = populated(c, w, 4);
        RetrievalResult r = m.retrieve(w.obs(0.1));
        assertEquals(1, r.regions().size());
        assertNotNull(r.topRegionScore());
        assertNull(r.competingRegionScore(), "every scored reference lies inside the top region");
        assertNull(r.regionMargin());
    }

    /** The positional region rule groups by stored metric position, so a revisit is not its own competitor. */
    @Test
    public void positionalRegionRuleGroupsByStoredPositionNotById() {
        RelocalizationConfig c = config();
        c.recentExclusionCount = 0;
        c.topK = 3;
        c.regionRule = RelocalizationConfig.REGION_POSITION;
        c.regionRadiusM = 5.0;
        c.regionGapReferences = 1;
        ReferenceMemory m = new ReferenceMemory(c);
        NavigationAligner a = new NavigationAligner();
        // A at (0,0) phase 0.0; B at (100,0) phase 0.5; C at (1,1) — a revisit of A, phase 0.0.
        a.observe(TestSamples.ok(0, 0, 0, 0, 10.0));
        assertTrue(m.offer(SkylineObservation.valid(0, 0.0, TestDescriptors.sine(0.0)), a.current()).inserted());
        a.observe(TestSamples.ok(1, 0, 100, 0, 10.0));
        assertTrue(m.offer(SkylineObservation.valid(1, 0.1, TestDescriptors.sine(0.5)), a.current()).inserted());
        a.observe(TestSamples.ok(2, 0, 1, 1, 10.0));
        assertTrue(m.offer(SkylineObservation.valid(2, 0.2, TestDescriptors.sine(0.0)), a.current()).inserted());
        a.observe(TestSamples.ok(3, 0, 50, 50, 10.0));

        RetrievalResult r = m.retrieve(SkylineObservation.valid(3, 0.3, TestDescriptors.sine(0.0)));
        assertEquals(0, r.top().referenceId(), "tie between A and C broken by id");
        assertEquals(List.of(0, 2), r.topRegion().memberIds(), "the revisit C joins A's region by position");
        assertEquals(-1.0, r.competingRegionScore(), 1e-9, "the competitor is B, 100 m away");
        assertEquals(2.0, r.regionMargin(), 1e-9);

        RelocalizationConfig bad = config();
        bad.regionRule = RelocalizationConfig.REGION_POSITION;
        assertThrows(IllegalArgumentException.class, bad::validate, "the radius is declared, never defaulted");
    }

    // ---------------------------------------------------------------- dual-view banks (tests J, K)

    private static RelocalizationConfig dualConfig(String rule) {
        RelocalizationConfig c = new RelocalizationConfig();
        c.descriptorLength = TestProfiles.N;
        c.degenerateStdFloor = 1e-9;
        c.matcher = RelocalizationConfig.MATCHER_C1;
        c.maxLagSamples = 32;
        c.searchExposureBound = 1000;
        c.recentExclusionMode = RelocalizationConfig.MODE_COUNT;
        c.recentExclusionCount = 0;
        c.regionGapReferences = 0;                       // every reference its own region
        c.topK = 5;
        c.fusionRule = rule;
        return c;
    }

    /** Four places (families 0–3), each stored with both views, 10 frames apart. */
    private static ReferenceMemory dualPopulated(RelocalizationConfig c, World w) {
        ReferenceMemory m = new ReferenceMemory(c);
        for (int fam = 0; fam < 4; fam++) {
            NavigationOutput nav = w.advance(10);
            assertTrue(m.offer(TestProfiles.dual(w.frame, w.frame * 0.1, fam, 0, 0), nav).inserted());
        }
        w.advance(5);
        return m;
    }

    @Test
    public void theWestBankIsScoredOnlyUnderAFusingRuleAndAMissingWestIsNeverAgreement() {
        World w = new World();
        ReferenceMemory northOnly = dualPopulated(dualConfig(RelocalizationConfig.FUSION_NORTH_ONLY), w);
        RetrievalResult r0 = northOnly.retrieve(TestProfiles.dual(w.frame, w.frame * 0.1, 1, 0, 0));
        assertNull(r0.west(), "north_only never scores the West bank, even when the query has one");
        assertEquals(RetrievalResult.DualStatus.NOT_USED, r0.dual().status());
        assertNull(r0.dual().agreement());
        assertTrue(r0.westQueryAvailable());

        World w2 = new World();
        ReferenceMemory strict = dualPopulated(dualConfig(RelocalizationConfig.FUSION_STRICT), w2);
        RetrievalResult complete = strict.retrieve(TestProfiles.dual(w2.frame, w2.frame * 0.1, 1, -3, 5));
        assertNotNull(complete.west());
        assertEquals(RetrievalResult.DualStatus.COMPLETE, complete.dual().status());
        assertEquals(1, complete.north().top().referenceId());
        assertEquals(1, complete.west().top().referenceId());
        assertEquals(-3, complete.north().top().lagSamples(), "North lag: evidence, never a position");
        assertEquals(5, complete.west().top().lagSamples(), "West lag: evidence, never a position");
        assertEquals(Boolean.TRUE, complete.dual().agreement());
        assertEquals(0.0, complete.dual().topSeparationM(), 0.0);
        assertEquals(4, complete.westReferencesScored());
        assertNull(complete.fused(), "strict agreement forms no fused ranking");
        assertEquals(1, complete.top().referenceId(), "the primary ranking is North's");

        RetrievalResult noWest = strict.retrieve(TestProfiles.observation(w2.frame + 1, 0.0, 1, 0));
        assertNull(noWest.west());
        assertEquals(RetrievalResult.DualStatus.WEST_UNAVAILABLE, noWest.dual().status());
        assertNull(noWest.dual().agreement(), "unavailable is not agreement");
        assertFalse(noWest.westQueryAvailable());

        RetrievalResult badWest = strict.retrieve(TestProfiles.westInvalid(w2.frame + 2, 0.0, 1, 0));
        assertEquals(RetrievalResult.DualStatus.WEST_INVALID, badWest.dual().status());
        assertNull(badWest.dual().agreement(), "invalid is not agreement");
        assertTrue(badWest.westQueryAvailable());
        assertFalse(badWest.westQueryValid());
    }

    @Test
    public void strictAgreementEvidenceDetectsADisagreeingWest() {
        World w = new World();
        ReferenceMemory strict = dualPopulated(dualConfig(RelocalizationConfig.FUSION_STRICT), w);
        // North sees place 0, West sees place 2: two confident answers naming different regions.
        RetrievalResult r = strict.retrieve(TestProfiles.crossed(w.frame, w.frame * 0.1, 0, 2));
        assertEquals(0, r.north().top().referenceId());
        assertEquals(2, r.west().top().referenceId());
        assertEquals(RetrievalResult.DualStatus.COMPLETE, r.dual().status());
        assertEquals(Boolean.FALSE, r.dual().agreement());
        assertEquals(strict.get(0).positionGlobal().distanceTo(strict.get(2).positionGlobal()),
                r.dual().topSeparationM(), 1e-9);
        assertEquals(2, r.dual().westTopReferenceId());
        assertTrue(r.north().top().score() > 0.99);
        assertTrue(r.west().top().score() > 0.99);
    }

    @Test
    public void weakestViewFusionRanksByTheMinimumAndUnpairedReferencesLose() {
        RelocalizationConfig c = dualConfig(RelocalizationConfig.FUSION_WEAKEST);
        ReferenceMemory m = new ReferenceMemory(c);
        World w = new World();
        NavigationOutput nav = w.advance(10);
        assertTrue(m.offer(TestProfiles.dual(w.frame, w.frame * 0.1, 0, 0, 0), nav).inserted());     // id 0: both
        nav = w.advance(10);
        assertTrue(m.offer(TestProfiles.observation(w.frame, w.frame * 0.1, 1, 0), nav).inserted());  // id 1: North only
        nav = w.advance(10);
        assertTrue(m.offer(TestProfiles.dual(w.frame, w.frame * 0.1, 2, 0, 0), nav).inserted());     // id 2: both
        w.advance(5);

        RetrievalResult r = m.retrieve(TestProfiles.dual(w.frame, w.frame * 0.1, 1, 0, 0));
        assertNotNull(r.fused(), "both views present: a fused ranking exists");
        assertTrue(r.fused().candidates().stream().noneMatch(cand -> cand.referenceId() == 1),
                "the North-only reference has no West score and loses under min()");
        assertEquals(RetrievalResult.DualStatus.COMPLETE, r.dual().status());
        assertEquals(1, r.north().top().referenceId(), "North alone would have picked place 1 ...");
        assertTrue(r.fused().top().referenceId() != 1, "... the fused ranking cannot");
        for (RetrievalResult.Candidate f : r.fused().candidates()) {
            double n = r.north().candidate(f.referenceId()).score();
            double wv = r.west().candidate(f.referenceId()).score();
            assertEquals(Math.min(n, wv), f.score(), 1e-12, "min(s_N, s_W) for " + f.referenceId());
        }
        assertEquals(r.fused().top(), r.top(), "the primary ranking is the fused one");

        // A query without West under the same rule falls back to North, marked incomplete.
        RetrievalResult noWest = m.retrieve(TestProfiles.observation(w.frame + 1, 0.0, 1, 0));
        assertNull(noWest.fused());
        assertEquals(RetrievalResult.DualStatus.WEST_UNAVAILABLE, noWest.dual().status());
        assertEquals(1, noWest.top().referenceId());
    }

    @Test
    public void configRefusesUnknownKeysAndBadValues(@org.junit.jupiter.api.io.TempDir java.nio.file.Path dir)
            throws java.io.IOException {
        java.nio.file.Path good = dir.resolve("good.json");
        java.nio.file.Files.writeString(good, "{\"top_k\": 7, \"recent_exclusion_count\": 3, "
                + "\"fusion_rule\": \"strict_agreement\", \"temporal_confirmation\": \"fallback\"}");
        RelocalizationConfig c = RelocalizationConfig.load(good);
        assertEquals(7, c.topK);
        assertEquals(3, c.recentExclusionCount);
        assertEquals(256, c.descriptorLength, "defaults survive");
        assertEquals(RelocalizationConfig.FUSION_STRICT, c.fusionRule);
        assertEquals(RelocalizationConfig.TEMPORAL_FALLBACK, c.temporalConfirmation);

        java.nio.file.Path typo = dir.resolve("typo.json");
        java.nio.file.Files.writeString(typo, "{\"topk\": 7}");
        assertThrows(java.io.IOException.class, () -> RelocalizationConfig.load(typo),
                "a misspelt key must not be silently ignored");

        java.nio.file.Path bad = dir.resolve("bad.json");
        java.nio.file.Files.writeString(bad, "{\"top_k\": 0}");
        assertThrows(IllegalArgumentException.class, () -> RelocalizationConfig.load(bad));

        java.nio.file.Path badRule = dir.resolve("rule.json");
        java.nio.file.Files.writeString(badRule, "{\"fusion_rule\": \"product\"}");
        assertThrows(IllegalArgumentException.class, () -> RelocalizationConfig.load(badRule),
                "no fusion equation outside the three evaluated ones");
    }
}
