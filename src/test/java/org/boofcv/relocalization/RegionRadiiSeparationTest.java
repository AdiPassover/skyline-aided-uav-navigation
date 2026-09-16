package org.boofcv.relocalization;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * <b>One radius answered three different questions; now three do (2026-09-08).</b>
 *
 * <p>{@code region_radius_m} was set to 125 m because that is the τ-ball SKY's recognition study
 * evaluated correctness against — an <em>evaluation tolerance</em>, silently promoted to a runtime
 * parameter and then used for all three of:
 *
 * <ol type="A">
 *   <li>ambiguity grouping — which references may not count as competing hypotheses;</li>
 *   <li>dual-view agreement — whether North and West identified the same place;</li>
 *   <li>temporal continuity — whether this query's candidate is the previous query's.</li>
 * </ol>
 *
 * <p>These tests hold two of the three fixed and move the third, and assert that only the matching
 * behaviour changes. <b>They deliberately fix no numeric value</b>: the radii used here are chosen
 * to sit either side of the synthetic reference spacing so the switch is unambiguous, and carry no
 * recommendation. Choosing operating values waits on the SKY matcher study and on INT trajectory
 * evidence.
 *
 * <p>Evidence tier: T1 (synthetic).
 */
public class RegionRadiiSeparationTest {

    /**
     * References are laid down every 20 frames on the {@code World} track, i.e. 41.23 m apart:
     * {@code |Δp| = 20 · sqrt(2.0² + 0.5²)}. Every radius below is chosen relative to that.
     */
    private static final double SPACING_M = 20.0 * Math.sqrt(2.0 * 2.0 + 0.5 * 0.5);

    private static RelocalizationConfig positional(double ambiguity, double dual, double temporal) {
        RelocalizationConfig c = base();
        c.regionRule = RelocalizationConfig.REGION_POSITION;
        c.ambiguityRegionRadiusM = ambiguity;
        c.dualAgreementRadiusM = dual;
        c.temporalRegionRadiusM = temporal;
        return c;
    }

    private static RelocalizationConfig base() {
        RelocalizationConfig c = new RelocalizationConfig();
        c.descriptorLength = TestProfiles.N;
        c.degenerateStdFloor = 1e-9;
        c.matcher = RelocalizationConfig.MATCHER_C0;
        c.noveltyMinDistance = 0.05;
        c.maxSpacingFrames = 1000;
        c.searchExposureBound = 100000;
        c.recentExclusionMode = RelocalizationConfig.MODE_COUNT;
        c.recentExclusionCount = 0;
        // strict_agreement so the PRIMARY ranking stays North: these tests are about the region
        // predicates, and a fused primary would make the ranking itself the variable.
        c.fusionRule = RelocalizationConfig.FUSION_STRICT;
        c.topK = 5;
        c.matchThreshold = 0.9;
        c.marginThreshold = 0.15;
        c.confirmQueries = 2;
        return c;
    }

    /** A running aligner, so the memory stores genuine metric positions (2 m E, 0.5 m N per frame). */
    private static final class World {
        final NavigationAligner aligner = new NavigationAligner();
        int frame = -1;

        NavigationOutput advance(int frames) {
            NavigationOutput o = null;
            for (int i = 0; i < frames; i++) {
                frame++;
                o = aligner.observe(TestSamples.ok(frame, 0, 2.0 * frame, 0.5 * frame, 30.0));
            }
            return o;
        }
    }

    /** Four places, 41.23 m apart, each stored with both views. */
    private static ReferenceMemory fourPlaces(RelocalizationConfig c) {
        ReferenceMemory m = new ReferenceMemory(c);
        World w = new World();
        for (int place = 0; place < 4; place++) {
            NavigationOutput nav = w.advance(20);
            InsertionDecision d = m.offer(TestProfiles.dual(w.frame, w.frame * 0.1, place, 0, 0), nav);
            assertTrue(d.inserted(), "place " + place + ": " + d.outcome());
        }
        assertEquals(4, m.size());
        assertEquals(SPACING_M, m.get(0).positionGlobal().distanceTo(m.get(1).positionGlobal()), 1e-9);
        return m;
    }

    // ================================================================ A: ambiguity grouping only

    @Test
    @DisplayName("A: the ambiguity radius decides whether a competitor exists — and nothing else")
    void theAmbiguityRadiusOnlyGovernsCompetingGrouping() {
        // Wide enough to swallow all four places (3 x 41.23 = 123.7 m apart end to end).
        RelocalizationConfig wide = positional(130.0, 60.0, 60.0);
        // Narrow enough that each place is its own ambiguity region.
        RelocalizationConfig narrow = positional(20.0, 60.0, 60.0);

        RetrievalResult rWide = fourPlaces(wide).retrieve(TestProfiles.dual(99, 9.9, 1, 0, 0));
        RetrievalResult rNarrow = fourPlaces(narrow).retrieve(TestProfiles.dual(99, 9.9, 1, 0, 0));

        // The RANKING is identical — the radius is not a scorer.
        assertEquals(rWide.top().referenceId(), rNarrow.top().referenceId());
        assertEquals(rWide.top().score(), rNarrow.top().score(), 0.0);
        assertEquals(1, rWide.top().referenceId(), "place 1 is the match either way");

        // Only the grouping, and therefore the margin, moved.
        assertEquals(1, rWide.regions().size(), "one region swallows the whole memory");
        assertNull(rWide.competingRegionScore(), "so nothing competes");
        assertNull(rWide.regionMargin());
        assertTrue(rNarrow.regions().size() > 1, "each place is its own hypothesis");
        assertNotNull(rNarrow.competingRegionScore());
        assertNotNull(rNarrow.regionMargin());

        // B is untouched: both views still name place 1, so they still agree.
        assertEquals(Boolean.TRUE, rWide.dual().agreement());
        assertEquals(Boolean.TRUE, rNarrow.dual().agreement());
        assertEquals(rWide.dual().topSeparationM(), rNarrow.dual().topSeparationM(), 0.0);

        // C is untouched: with the gate's verdict held fixed, the tracker sees the same region.
        assertEquals(2, supportOverTwoQueries(wide, 1, 1),
                "temporal continuity does not depend on the ambiguity radius");
        assertEquals(2, supportOverTwoQueries(narrow, 1, 1));
    }

    // ================================================================ B: dual agreement only

    @Test
    @DisplayName("B: the dual-agreement radius decides whether two views named one place — and nothing else")
    void theDualAgreementRadiusOnlyGovernsViewAgreement() {
        RelocalizationConfig generous = positional(20.0, SPACING_M + 1.0, 60.0);
        RelocalizationConfig strict = positional(20.0, SPACING_M - 1.0, 60.0);

        // North sees place 1, West sees place 2: neighbouring places, exactly SPACING_M apart.
        SkylineObservation crossed = TestProfiles.crossed(99, 9.9, 1, 2);
        RetrievalResult rGenerous = fourPlaces(generous).retrieve(crossed);
        RetrievalResult rStrict = fourPlaces(strict).retrieve(crossed);

        assertEquals(1, rGenerous.dual().northTopReferenceId());
        assertEquals(2, rGenerous.dual().westTopReferenceId());
        assertEquals(SPACING_M, rGenerous.dual().topSeparationM(), 1e-9);

        assertEquals(Boolean.TRUE, rGenerous.dual().agreement(), "within the agreement radius");
        assertEquals(Boolean.FALSE, rStrict.dual().agreement(), "outside it — a disagreement");

        // A is untouched: same ambiguity radius, so the same regions and the same margin.
        assertEquals(rGenerous.regions().size(), rStrict.regions().size());
        assertEquals(rGenerous.regionMargin(), rStrict.regionMargin());
        assertEquals(rGenerous.competingRegionScore(), rStrict.competingRegionScore());
        // C is untouched.
        assertEquals(supportOverTwoQueries(generous, 1, 1), supportOverTwoQueries(strict, 1, 1));
    }

    // ================================================================ C: temporal continuity only

    @Test
    @DisplayName("C: the temporal radius decides whether a confirmation chain continues — and nothing else")
    void theTemporalRadiusOnlyGovernsTrackerContinuity() {
        RelocalizationConfig generous = positional(20.0, 60.0, SPACING_M + 1.0);
        RelocalizationConfig strict = positional(20.0, 60.0, SPACING_M - 1.0);

        // Two consecutive queries whose winners are neighbouring places, SPACING_M apart.
        assertEquals(2, supportOverTwoQueries(generous, 1, 2), "one hypothesis, still moving");
        assertEquals(1, supportOverTwoQueries(strict, 1, 2), "a different hypothesis — chain broken");

        // ... and both agree that the SAME place twice is one chain, whatever the radius.
        assertEquals(2, supportOverTwoQueries(generous, 1, 1));
        assertEquals(2, supportOverTwoQueries(strict, 1, 1));

        // A and B are untouched by the temporal radius.
        RetrievalResult a = fourPlaces(generous).retrieve(TestProfiles.crossed(99, 9.9, 1, 2));
        RetrievalResult b = fourPlaces(strict).retrieve(TestProfiles.crossed(99, 9.9, 1, 2));
        assertEquals(a.regions().size(), b.regions().size());
        assertEquals(a.regionMargin(), b.regionMargin());
        assertEquals(a.dual().agreement(), b.dual().agreement());
    }

    /**
     * Support after observing place {@code first} then place {@code second}, with the gate's own
     * verdict forced true so that the only thing under test is the region rule's identity decision.
     */
    private static int supportOverTwoQueries(RelocalizationConfig c, int first, int second) {
        ReferenceMemory m = fourPlaces(c);
        RegionTracker t = new RegionTracker(c);
        t.observe(m.retrieve(TestProfiles.dual(98, 9.8, first, 0, 0)), true);
        RegionTracker.Track track = t.observe(m.retrieve(TestProfiles.dual(99, 9.9, second, 0, 0)), true);
        return track == null ? 0 : track.supportCount();
    }

    // ================================================================ backward compatibility

    @Test
    @DisplayName("a legacy region_radius_m still means exactly what it meant, and says so")
    void theLegacyRadiusIsMappedExplicitlyAndLabelled() {
        RelocalizationConfig legacy = base();
        legacy.regionRule = RelocalizationConfig.REGION_POSITION;
        legacy.regionRadiusM = 125.0;
        legacy.validate();

        assertTrue(legacy.regionRadiiAreLegacyMapped(),
                "a run made under one radius is never reported as though three had been chosen");
        assertEquals(125.0, legacy.effectiveAmbiguityRadiusM());
        assertEquals(125.0, legacy.effectiveDualAgreementRadiusM());
        assertEquals(125.0, legacy.effectiveTemporalRadiusM());

        RelocalizationConfig split = positional(125.0, 125.0, 125.0);
        split.validate();
        assertFalse(split.regionRadiiAreLegacyMapped(), "declared explicitly, not inherited");
        // Same numbers => same behaviour: the mapping is a relabelling, not a reinterpretation.
        RetrievalResult viaLegacy = fourPlaces(legacy).retrieve(TestProfiles.dual(99, 9.9, 1, 0, 0));
        RetrievalResult viaSplit = fourPlaces(split).retrieve(TestProfiles.dual(99, 9.9, 1, 0, 0));
        assertEquals(viaLegacy.regions().size(), viaSplit.regions().size());
        assertEquals(viaLegacy.regionMargin(), viaSplit.regionMargin());
        assertEquals(viaLegacy.dual().agreement(), viaSplit.dual().agreement());
    }

    @Test
    @DisplayName("the two forms are never mixed, and neither may be omitted")
    void radiiAreDeclaredOrRefusedNeverGuessed() {
        RelocalizationConfig mixed = base();
        mixed.regionRule = RelocalizationConfig.REGION_POSITION;
        mixed.regionRadiusM = 125.0;
        mixed.ambiguityRegionRadiusM = 30.0;
        assertTrue(assertThrows(IllegalArgumentException.class, mixed::validate)
                .getMessage().contains("never combined"));

        RelocalizationConfig none = base();
        none.regionRule = RelocalizationConfig.REGION_POSITION;
        assertTrue(assertThrows(IllegalArgumentException.class, none::validate)
                .getMessage().contains("never defaulted"));

        RelocalizationConfig partial = base();
        partial.regionRule = RelocalizationConfig.REGION_POSITION;
        partial.ambiguityRegionRadiusM = 30.0;
        partial.dualAgreementRadiusM = 30.0;                  // temporal missing
        assertTrue(assertThrows(IllegalArgumentException.class, partial::validate)
                .getMessage().contains("temporal_region_radius_m"));

        RelocalizationConfig onIdGap = base();                // id_gap takes no radius at all
        onIdGap.temporalRegionRadiusM = 30.0;
        assertThrows(IllegalArgumentException.class, onIdGap::validate);

        RelocalizationConfig idGap = base();
        idGap.validate();
        assertNull(idGap.effectiveAmbiguityRadiusM(), "no radius is in force under id_gap");
        assertNull(idGap.effectiveDualAgreementRadiusM());
        assertNull(idGap.effectiveTemporalRadiusM());
        assertFalse(idGap.regionRadiiAreLegacyMapped());
    }

    // ================================================================ the moving ball (audit)

    @Test
    @DisplayName("AUDIT: a temporal chain WALKS — consecutive steps are bounded, the chain is not")
    void theTemporalRegionIsAMovingBallAndTheChainHasNoTotalBound() {
        // Documented, not changed. RegionTracker.observe replaces the tracked representative with
        // each confirmation, so identity is "within R of the PREVIOUS query", never "within R of
        // where the chain started". Four places 41.23 m apart with a 60 m temporal radius are one
        // unbroken chain of support 4, although its ends are 123.7 m apart — three times the step
        // and twice the radius.
        RelocalizationConfig c = positional(20.0, 60.0, 60.0);
        ReferenceMemory m = fourPlaces(c);
        RegionTracker t = new RegionTracker(c);

        RegionTracker.Track track = null;
        for (int place = 0; place < 4; place++) {
            track = t.observe(m.retrieve(TestProfiles.dual(90 + place, 9.0 + place, place, 0, 0)), true);
            assertEquals(place, track.bestReferenceId());
            assertEquals(place + 1, track.supportCount(), "place " + place);
        }
        assertEquals(RegionTracker.Traversal.FORWARD, track.lastTraversal());
        double endToEnd = m.get(0).positionGlobal().distanceTo(m.get(3).positionGlobal());
        assertEquals(3 * SPACING_M, endToEnd, 1e-9);
        assertTrue(endToEnd > 2 * 60.0,
                "the chain's extent exceeds twice the radius that was supposed to bound it");

        // A step LARGER than the radius does break it, so the per-step bound is real.
        RegionTracker strict = new RegionTracker(positional(20.0, 60.0, SPACING_M - 1.0));
        strict.observe(m.retrieve(TestProfiles.dual(90, 9.0, 0, 0, 0)), true);
        RegionTracker.Track broken =
                strict.observe(m.retrieve(TestProfiles.dual(91, 9.1, 1, 0, 0)), true);
        assertEquals(1, broken.supportCount(), "a step beyond the radius restarts the chain");

        // And reverse / stationary traversal still count, as the design requires (§7).
        RegionTracker back = new RegionTracker(c);
        back.observe(m.retrieve(TestProfiles.dual(90, 9.0, 2, 0, 0)), true);
        assertEquals(RegionTracker.Traversal.REVERSE,
                back.observe(m.retrieve(TestProfiles.dual(91, 9.1, 1, 0, 0)), true).lastTraversal());
        assertEquals(RegionTracker.Traversal.STATIONARY,
                back.observe(m.retrieve(TestProfiles.dual(92, 9.2, 1, 0, 0)), true).lastTraversal());
        assertEquals(3, back.current().supportCount());
    }

    @Test
    @DisplayName("AUDIT: a query that fails the gate still cannot lend support to the chain it walked into")
    void aFailingQueryBreaksTheChainRatherThanExtendingIt() {
        RelocalizationConfig c = positional(20.0, 60.0, 60.0);
        ReferenceMemory m = fourPlaces(c);
        RegionTracker t = new RegionTracker(c);
        t.observe(m.retrieve(TestProfiles.dual(90, 9.0, 0, 0, 0)), true);
        assertEquals(2, t.observe(m.retrieve(TestProfiles.dual(91, 9.1, 1, 0, 0)), true).supportCount());
        assertNull(t.observe(m.retrieve(TestProfiles.dual(92, 9.2, 2, 0, 0)), false),
                "a query that did not pass the match/margin gate drops the track");
        assertEquals(1, t.observe(m.retrieve(TestProfiles.dual(93, 9.3, 2, 0, 0)), true).supportCount(),
                "and the chain restarts rather than resuming");
        assertEquals(List.of(2), t.current().memberIds());
    }
}
