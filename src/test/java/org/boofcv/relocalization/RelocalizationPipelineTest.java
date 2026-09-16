package org.boofcv.relocalization;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The whole loop on a synthetic metric flight: references inserted early, an exposure-scheduled
 * search, two-query confirmation, a translation-only re-anchor (with a C1 lag that is logged and
 * not applied, and a heading that is the VO's), a weak-match rejection with the retry gap, an
 * instability escalation with attribution, recovery after a hard loss, dual-view observations,
 * the pose no-op guarantees, and reference provenance. T1 — mechanism only.
 */
public class RelocalizationPipelineTest {

    private static final double EPS = 1e-9;

    private static RelocalizationConfig config() {
        RelocalizationConfig c = new RelocalizationConfig();
        c.descriptorLength = TestProfiles.N;
        c.degenerateStdFloor = 1e-9;
        c.noveltyMinDistance = 0.05;
        c.maxSpacingFrames = 100;
        c.searchExposureBound = 20;
        c.recentExclusionMode = RelocalizationConfig.MODE_COUNT;
        c.recentExclusionCount = 1;
        c.regionGapReferences = 1;
        c.topK = 5;
        c.matcher = RelocalizationConfig.MATCHER_C1;
        c.maxLagSamples = 32;
        c.minOverlapFrac = 0.6;
        c.minRetryGapFrames = 5;
        c.instabilityThresholdDeg = 1.0;
        c.matchThreshold = 0.9;
        c.marginThreshold = 0.15;
        c.confirmQueries = 2;
        c.regionTrackMaxGap = 2;
        return c;
    }

    /** A metric VO track on segment 0: 3 m steps along a slowly turning heading, metres ENU. */
    private static List<LocalPoseSample> voTrack(int frames) {
        List<LocalPoseSample> track = new ArrayList<>();
        double e = 0, n = 0;
        for (int i = 0; i < frames; i++) {
            double heading = 20.0 + 0.4 * i;
            if (i > 0) {
                e += 3.0 * Math.sin(Math.toRadians(heading));
                n += 3.0 * Math.cos(Math.toRadians(heading));
            }
            track.add(TestSamples.ok(i, 0, e, n, heading));
        }
        return track;
    }

    private static void assertPos(PlanarPosition expected, PlanarPosition actual, String what) {
        assertEquals(expected.eastM(), actual.eastM(), EPS, what + " east");
        assertEquals(expected.northM(), actual.northM(), EPS, what + " north");
    }

    /** Frames 0..7 with four distinct places stored on frames 1, 3, 5, 7 (North only). */
    private static RelocalizationPipeline seeded(RelocalizationConfig c, List<LocalPoseSample> vo) {
        RelocalizationPipeline p = new RelocalizationPipeline(c);
        for (int f = 0; f <= 7; f++) {
            SkylineObservation obs = (f % 2 == 1) ? TestProfiles.observation(f, f * 0.1, (f - 1) / 2, 0) : null;
            RelocalizationPipeline.FrameStep s = p.step(vo.get(f), obs, 0.2);
            if (obs != null) {
                assertTrue(s.insertion().inserted(), "frame " + f + ": " + s.insertion().outcome());
            }
            assertNull(s.retrieval(), "no request is pending this early");
        }
        assertEquals(4, p.memory().size());
        return p;
    }

    private static RelocalizationPipeline seeded(List<LocalPoseSample> vo) {
        return seeded(config(), vo);
    }

    @Test
    public void scheduledSearchConfirmsOverTwoQueriesAndSnapsThePositionExactlyOntoTheReferenceIgnoringTheLag() {
        List<LocalPoseSample> vo = voTrack(80);
        // One reference per region here, so that excluding the most recent one still leaves a
        // reference OUTSIDE the winner's region and a real competing hypothesis exists. With the
        // default gap of 1 the four seeded places collapse to {0,1,2} plus the excluded 3, leaving
        // no competitor at all — which since 2026-09-08 is retained, not accepted, and is covered
        // by recencyExclusionLeavingOneRegionCannotAccept().
        RelocalizationConfig c = config();
        c.regionGapReferences = 0;
        RelocalizationPipeline p = seeded(c, vo);

        RelocalizationPipeline.FrameStep armedAt = null;
        for (int f = 8; f <= 30; f++) {
            RelocalizationPipeline.FrameStep s = p.step(vo.get(f), null, 0.2);
            if (s.requestArmed() != null) {
                assertNull(armedAt, "armed once");
                armedAt = s;
            }
            assertNull(s.retrieval());
        }
        assertNotNull(armedAt);
        assertEquals(20, armedAt.output().frameIndex());
        assertEquals(SearchScheduler.Cause.SCHEDULED_EXPOSURE, armedAt.requestArmed().cause());
        assertTrue(p.scheduler().requestPending());

        // Query 1: place 1 seen 5 samples shifted. C1 aligns it; the gate retains it.
        RelocalizationPipeline.FrameStep q1 = p.step(vo.get(31), TestProfiles.observation(31, 3.1, 1, -5), 0.2);
        assertNotNull(q1.retrieval());
        assertEquals(1, q1.retrieval().top().referenceId());
        assertEquals(-5, q1.retrieval().top().lagSamples(), "q(t) = r(t − 5) → lag −5");
        assertEquals(1.0, q1.retrieval().top().score(), 1e-9);
        assertEquals(List.of(3), q1.retrieval().excludedRecentIds(), "COUNT mode, M = 1");
        assertEquals(AcceptanceGate.Verdict.RETAIN, q1.decision().verdict());
        assertEquals(AcceptanceGate.Reason.UNCONFIRMED, q1.decision().reason());
        assertEquals(SearchScheduler.AttemptOutcome.RETAINED_FOR_CONFIRMATION, q1.attempt().outcome());
        assertNull(q1.reanchor());
        assertTrue(p.scheduler().requestPending(), "retained → still pending");
        assertEquals(InsertionDecision.Outcome.REJECTED_EXPOSURE_BEYOND_SEARCH_BOUND, q1.insertion().outcome());
        assertTrue(p.aligner().alignment().orElseThrow().isIdentity(0.0), "a retained candidate moves nothing");
        assertPos(vo.get(31).segmentPosition(), q1.output().globalPosition().orElseThrow(), "still the VO position");
        // A pending search executes on the observation's frame, with THAT frame's pose (test: sync).
        assertEquals(31, q1.evidence().queryFrameIndex());
        assertPos(vo.get(31).segmentPosition(), q1.evidence().queryLocalPosition(), "query position");
        assertEquals(vo.get(31).headingDeg(), q1.evidence().queryHeadingDeg(), 0.0);
        assertEquals(20, q1.evidence().originFrame(), "... for a request armed 11 frames earlier");

        // Query 2: support reaches 2 → ACCEPT. Position = stored reference position EXACTLY; the
        // heading stays the VO's authoritative one; the lag is evidence and nowhere in the pose.
        double headingAtQuery = vo.get(32).headingDeg();
        RelocalizationPipeline.FrameStep q2 = p.step(vo.get(32), TestProfiles.observation(32, 3.2, 1, -6), 0.2);
        assertTrue(q2.decision().accepted(), q2.decision().reason().toString());
        assertNotNull(q2.reanchor());
        TrustedReference ref1 = p.memory().get(1);
        assertPos(ref1.positionGlobal(), q2.output().globalPosition().orElseThrow(), "snapped");
        assertPos(ref1.positionGlobal(), q2.reanchor().relocalizedPosition(), "p_reloc = p_ref");
        assertEquals(headingAtQuery, q2.output().headingDeg(), 0.0, "heading: the VO's, bit-exact ...");
        assertEquals(headingAtQuery, q2.output().globalPose().orElseThrow().yaw, 1e-9,
                "... and in the pose (Pose3D's constructor re-normalises by an ULP), not the reference's");
        assertEquals(vo.get(3).headingDeg(), ref1.headingDeg(), 0.0,
                "(which differs from what the reference, stored on frame 3, carries)");
        assertTrue(Math.abs(headingAtQuery - ref1.headingDeg()) > 1.0, "premise: the two headings differ");
        assertEquals("discrete_reference", q2.reanchor().poseRuleId());
        assertEquals(-6, q2.evidence().winningLagSamples(), "lag preserved as evidence");
        assertEquals(2, q2.evidence().temporalSupportCount());
        assertTrue(q2.evidence().accepted());
        assertEquals(SearchScheduler.Cause.SCHEDULED_EXPOSURE, q2.evidence().cause());
        assertEquals("not_used", q2.evidence().dualStatus());
        assertEquals(RegionTracker.Traversal.STATIONARY, q2.track().lastTraversal());
        assertNotNull(q2.reanchor().appliedDelta(), "the segment had a valid alignment before");
        assertEquals(ref1.baselineExposure(), p.aligner().lineage().exposureAnchor(), "inherited, not zero");
        assertEquals(0, p.aligner().lineage().exposureSince());
        assertFalse(p.scheduler().requestPending(), "accepted → request resolved");
        assertNull(p.tracker().current(), "confirmation starts afresh");

        AlignmentTransform tNew = q2.reanchor().alignmentAfter();
        for (int f = 33; f < 40; f++) {
            RelocalizationPipeline.FrameStep s = p.step(vo.get(f), null, 0.2);
            assertPos(tNew.apply(s.output().localPosition()), s.output().globalPosition().orElseThrow(), "frame " + f);
            assertEquals(vo.get(f).headingDeg(), s.output().headingDeg(), 0.0);
        }
    }

    @Test
    public void aWeakMatchIsRejectedWithoutMovingAnythingTheRetryGapThrottlesAndProvenanceStaysRoot() {
        List<LocalPoseSample> vo = voTrack(80);
        RelocalizationPipeline p = seeded(vo);
        for (int f = 8; f <= 20; f++) {
            p.step(vo.get(f), null, 0.2);
        }
        assertTrue(p.scheduler().requestPending());
        AlignmentTransform before = p.aligner().alignment().orElseThrow();
        AnchorLineage.Snapshot lineageBefore = p.aligner().lineage();

        // Family 3 is the latest reference and COUNT mode (M = 1) excludes it, so only the three
        // other places are scored; a 40-sample-shifted family-3 query matches none of them.
        RelocalizationPipeline.FrameStep r = p.step(vo.get(21), TestProfiles.observation(21, 2.1, 3, 40), 0.2);
        assertNotNull(r.retrieval());
        assertTrue(r.retrieval().topRegionScore() < 0.9, "premise: " + r.retrieval().topRegionScore());
        assertEquals(AcceptanceGate.Verdict.REJECT, r.decision().verdict());
        assertEquals(AcceptanceGate.Reason.WEAK_MATCH, r.decision().reason());
        assertNotNull(r.rejection());
        assertEquals(SearchScheduler.AttemptOutcome.REJECTED, r.attempt().outcome());
        assertSame(before, p.aligner().alignment().orElseThrow(), "alignment object untouched");
        assertEquals(lineageBefore.effectiveExposure() + 1, p.aligner().lineage().effectiveExposure());
        assertNull(p.aligner().lineage().anchorReferenceId(), "a rejected attempt creates no lineage");
        assertFalse(p.scheduler().requestPending(), "consumed");

        for (int f = 22; f < 26; f++) {
            RelocalizationPipeline.FrameStep s = p.step(vo.get(f), TestProfiles.observation(f, f * 0.1, 0, 0), 0.2);
            assertTrue(s.requestPending(), "frame " + f);
            assertTrue(s.retryGapBlocking(), "frame " + f);
            assertNull(s.retrieval(), "frame " + f);
        }
        RelocalizationPipeline.FrameStep again = p.step(vo.get(26), TestProfiles.observation(26, 2.6, 0, 0), 0.2);
        assertNotNull(again.retrieval(), "26 − 21 = 5 ≥ gap");
        assertEquals(0, again.retrieval().top().referenceId());
        assertEquals(AcceptanceGate.Verdict.RETAIN, again.decision().verdict(),
                "the rejected weak query at 21 lent no support: the chain starts here");
        assertEquals(1, again.track().supportCount());

        // Confirmation on the next frame accepts: the lineage now descends from reference 0, and a
        // reference stored afterwards records exactly that event (§22 provenance).
        RelocalizationPipeline.FrameStep acc = p.step(vo.get(27), TestProfiles.observation(27, 2.7, 0, 1), 0.2);
        assertTrue(acc.decision().accepted());
        assertEquals(0, p.aligner().lineage().anchorReferenceId());
        assertEquals(27, p.aligner().lineage().anchorFrameIndex());
        RelocalizationPipeline.FrameStep ins = p.step(vo.get(28), TestProfiles.observation(28, 2.8, 2, 50), 0.2);
        assertTrue(ins.insertion().inserted(), "" + ins.insertion().outcome());
        TrustedReference descendant = ins.insertion().reference();
        assertEquals(0, descendant.sourceAnchorId());
        assertEquals(27, descendant.sourceReanchorFrameIndex());
        assertTrue(descendant.descendsFromRelocalization());
        for (int id = 0; id < 4; id++) {
            assertFalse(p.memory().get(id).descendsFromRelocalization(), "seed " + id + " is root lineage");
        }
    }

    /** Test N end to end: a scheduled request escalated by instability keeps its origin in the log. */
    @Test
    public void instabilityEscalatesAScheduledRequestAndTheEvidenceKeepsBothCauses() {
        List<LocalPoseSample> vo = voTrack(80);
        RelocalizationPipeline p = seeded(vo);
        for (int f = 8; f <= 20; f++) {
            p.step(vo.get(f), null, 0.2);                       // armed at 20 by exposure
        }
        RelocalizationPipeline.FrameStep hot = p.step(vo.get(21), null, 5.0);
        assertTrue(hot.instabilityEvent());
        assertNull(hot.requestArmed(), "no second request");
        assertNotNull(hot.escalation());
        assertEquals(SearchScheduler.Cause.INSTABILITY, hot.escalation().cause());
        assertTrue(hot.escalation().changedCause());
        assertEquals(SearchScheduler.Cause.SCHEDULED_EXPOSURE, hot.pendingRequest().originCause());
        assertEquals(20, hot.pendingRequest().originFrame());
        assertEquals(SearchScheduler.Cause.INSTABILITY, hot.pendingRequest().cause());
        assertTrue(hot.output().lineage().instabilitySince());

        RelocalizationPipeline.FrameStep q = p.step(vo.get(22), TestProfiles.observation(22, 2.2, 2, 3), 0.2);
        assertNotNull(q.retrieval());
        MatchEvidence e = q.evidence();
        assertEquals(SearchScheduler.Cause.INSTABILITY, e.cause(), "the gate knows VO is suspect");
        assertEquals(SearchScheduler.Cause.SCHEDULED_EXPOSURE, e.originCause(), "the opportunity was scheduled");
        assertEquals(20, e.originFrame());
        assertTrue(e.escalated());
        assertEquals(SearchScheduler.Cause.SCHEDULED_EXPOSURE, q.attempt().originCause());
        assertEquals(1, q.attempt().escalationCount());
        assertNull(q.voAuxiliaryDisplacementErrorM(), "VO is suspect: no auxiliary comparison");
        assertEquals(InsertionDecision.Outcome.REJECTED_UNRESOLVED_INSTABILITY, q.insertion().outcome());
    }

    @Test
    public void instabilityBelowTauDoesNothingAndAtTauRequestsSearchImmediately() {
        List<LocalPoseSample> vo = voTrack(80);
        RelocalizationPipeline p = seeded(vo);
        RelocalizationPipeline.FrameStep quiet = p.step(vo.get(8), null, 0.99);
        assertFalse(quiet.instabilityEvent());
        assertNull(quiet.requestArmed());
        RelocalizationPipeline.FrameStep hot = p.step(vo.get(9), TestProfiles.observation(9, 0.9, 2, 3), 1.0);
        assertTrue(hot.instabilityEvent());
        assertNotNull(hot.requestArmed());
        assertEquals(SearchScheduler.Cause.INSTABILITY, hot.requestArmed().cause());
        assertEquals(SearchScheduler.Cause.INSTABILITY, hot.requestArmed().originCause(), "this one was created by instability");
        assertNotNull(hot.retrieval());
        assertEquals(2, hot.retrieval().top().referenceId());
        RelocalizationPipeline.FrameStep ok = p.step(vo.get(10), TestProfiles.observation(10, 1.0, 2, 4), 0.1);
        assertTrue(ok.decision().accepted());
        assertFalse(p.aligner().lineage().effectiveInstability());
        assertEquals(p.memory().get(2).baselineExposure(), p.aligner().lineage().effectiveExposure());
    }

    @Test
    public void hardLossLeavesThePositionAbsentKeepsTheHeadingAndAReferenceRestoresPositionOnly() {
        List<LocalPoseSample> vo = voTrack(80);
        RelocalizationPipeline p = seeded(vo);
        for (int f = 8; f < 12; f++) {
            p.step(vo.get(f), null, 0.2);
        }
        RelocalizationPipeline.FrameStep loss = p.step(TestSamples.loss(12, 1, 24.8), null, null);
        assertEquals(NavigationOutput.FrameEvent.HARD_LOSS, loss.output().event());
        assertTrue(loss.output().globalPosition().isEmpty());
        assertTrue(loss.output().headingValid(), "the heading channel never saw the visual failure");
        assertEquals(24.8, loss.output().headingDeg(), 0.0);
        assertFalse(loss.output().globalPoseValid());
        assertNotNull(loss.requestArmed());
        assertEquals(SearchScheduler.Cause.HARD_LOSS, loss.requestArmed().cause());
        assertEquals(1, loss.output().segmentId());

        RelocalizationPipeline.FrameStep q1 = p.step(TestSamples.ok(13, 1, 3, 0, 25.0),
                TestProfiles.observation(13, 1.3, 0, 0), 0.2);
        assertEquals(InsertionDecision.Outcome.REJECTED_GLOBAL_POSITION_INVALID, q1.insertion().outcome());
        assertEquals(AcceptanceGate.Verdict.RETAIN, q1.decision().verdict());
        assertTrue(q1.output().globalPosition().isEmpty());

        RelocalizationPipeline.FrameStep q2 = p.step(TestSamples.ok(14, 1, 6, 0, 25.2),
                TestProfiles.observation(14, 1.4, 0, 1), 0.2);
        assertTrue(q2.decision().accepted());
        assertTrue(q2.output().globalPositionValid());
        assertTrue(q2.output().globalPoseValid());
        assertPos(p.memory().get(0).positionGlobal(), q2.output().globalPosition().orElseThrow(), "restored");
        assertEquals(25.2, q2.output().headingDeg(), 0.0, "position restored, heading untouched");
        assertEquals(25.2, q2.output().globalPose().orElseThrow().yaw, 1e-9);
        assertNull(q2.reanchor().alignmentBefore(), "no alignment existed to jump from");
        assertNull(q2.reanchor().positionJumpM());
        assertEquals(1, q2.reanchor().segmentId());
        assertEquals(SearchScheduler.Cause.HARD_LOSS, q2.evidence().cause());
        assertEquals(1, p.aligner().hardLossEvents().size());
        assertEquals(1, p.aligner().reanchorEvents().size());
    }

    @Test
    public void framesWithoutOrWithInvalidObservationsNeverExecuteAndKeepTheRequestPending() {
        List<LocalPoseSample> vo = voTrack(80);
        RelocalizationPipeline p = seeded(vo);
        for (int f = 8; f <= 20; f++) {
            p.step(vo.get(f), null, 0.2);
        }
        RelocalizationPipeline.FrameStep invalid = p.step(vo.get(21),
                SkylineObservation.invalid(21, 2.1, "extraction refused"), 0.2);
        assertTrue(invalid.skylinePresent());
        assertFalse(invalid.skylineValid());
        assertNull(invalid.retrieval());
        assertTrue(invalid.requestPending());
        assertEquals(InsertionDecision.Outcome.REJECTED_SKYLINE_INVALID, invalid.insertion().outcome());
        assertEquals(0, p.scheduler().attempts().size());
        // A frame whose own metric increment is missing never executes either: its position is not
        // the query's position.
        RelocalizationPipeline.FrameStep dropout = p.step(TestSamples.noHeight(22, 0,
                vo.get(21).segmentPosition().eastM(), vo.get(21).segmentPosition().northM(), 28.0),
                TestProfiles.observation(22, 2.2, 0, 0), 0.2);
        assertNull(dropout.retrieval());
        assertNotNull(dropout.dropout());
        assertTrue(dropout.requestPending());
        assertFalse(dropout.output().globalPositionValid());
    }

    // ---------------------------------------------------------------- dual view (tests J, K, L)

    private static RelocalizationConfig strictFallback() {
        RelocalizationConfig c = config();
        c.fusionRule = RelocalizationConfig.FUSION_STRICT;
        c.temporalConfirmation = RelocalizationConfig.TEMPORAL_FALLBACK;
        c.regionGapReferences = 0;
        c.regionTrackMaxGap = 0;
        return c;
    }

    /** Four dual-view places on frames 1, 3, 5, 7. */
    private static RelocalizationPipeline dualSeeded(RelocalizationConfig c, List<LocalPoseSample> vo) {
        RelocalizationPipeline p = new RelocalizationPipeline(c);
        for (int f = 0; f <= 7; f++) {
            SkylineObservation obs = (f % 2 == 1) ? TestProfiles.dual(f, f * 0.1, (f - 1) / 2, 0, 0) : null;
            RelocalizationPipeline.FrameStep s = p.step(vo.get(f), obs, 0.2);
            if (obs != null) {
                assertTrue(s.insertion().inserted(), "frame " + f + ": " + s.insertion().outcome());
                assertTrue(s.insertion().reference().hasWest());
            }
        }
        for (int f = 8; f <= 20; f++) {
            p.step(vo.get(f), null, 0.2);
        }
        assertTrue(p.scheduler().requestPending());
        return p;
    }

    @Test
    public void oneAgreeingDualCaptureAcceptsWhileNorthOnlyAndInvalidWestQueriesAreRetainedHoweverLongTheyPersist() {
        List<LocalPoseSample> vo = voTrack(80);
        RelocalizationPipeline p = dualSeeded(strictFallback(), vo);

        RelocalizationPipeline.FrameStep q = p.step(vo.get(21), TestProfiles.dual(21, 2.1, 1, -5, 3), 0.2);
        assertTrue(q.westAvailable());
        assertTrue(q.westValid());
        assertEquals("complete", q.evidence().dualStatus());
        assertEquals(Boolean.TRUE, q.evidence().westAgreement());
        assertEquals(-5, q.evidence().winningLagSamples(), "North lag: logged");
        assertEquals(3, q.evidence().westLagSamples(), "West lag: logged");
        assertTrue(q.decision().accepted(), q.decision().reason().toString());
        assertEquals(AcceptanceGate.DualVerdict.AGREED, q.decision().dualVerdict());
        assertFalse(q.evidence().temporalRequired(), "one dual capture; no temporal chain was needed");
        assertPos(p.memory().get(1).positionGlobal(), q.output().globalPosition().orElseThrow(), "snapped");
        assertEquals(vo.get(21).headingDeg(), q.output().headingDeg(), 0.0, "heading untouched");
        AlignmentTransform after = p.aligner().alignment().orElseThrow();

        // Later, exposure re-arms; a run of North-only queries (missing or invalid West) naming one
        // region persists far beyond N_confirm — EXP-SKY-012 R5's alias along a straight leg — and
        // is retained every time: single-view evidence never accepts under a dual rule.
        for (int f = 22; f <= 45; f++) {
            p.step(vo.get(f), null, 0.2);
        }
        assertTrue(p.scheduler().requestPending());
        for (int f = 46; f <= 53; f++) {
            SkylineObservation obs = f % 2 == 0 ? TestProfiles.observation(f, f * 0.1, 2, 0)
                    : TestProfiles.westInvalid(f, f * 0.1, 2, 1);
            RelocalizationPipeline.FrameStep n = p.step(vo.get(f), obs, 0.2);
            assertEquals(f % 2 == 0 ? "west_unavailable" : "west_invalid", n.evidence().dualStatus());
            assertNull(n.evidence().westAgreement());
            assertEquals(AcceptanceGate.Verdict.RETAIN, n.decision().verdict(), "frame " + f);
            assertEquals(AcceptanceGate.Reason.WEST_INCOMPLETE, n.decision().reason(), "frame " + f);
            assertEquals(AcceptanceGate.DualVerdict.INCOMPLETE, n.decision().dualVerdict());
            assertFalse(n.evidence().temporalRequired(), "persistence was not consulted as a substitute");
            assertEquals(f - 45, n.track().supportCount(), "the chain is still counted and logged");
            assertNull(n.reanchor());
            assertSame(after, p.aligner().alignment().orElseThrow(), "the alignment did not move");
        }
        assertEquals(1, p.aligner().reanchorEvents().size(), "only the dual capture ever re-anchored");
    }

    /** Test L: nothing but an ACCEPT moves the alignment — not a reject, a retain, a disagreement, or a missing West. */
    @Test
    public void rejectRetainDisagreementAndMissingWestNeverMoveThePosition() {
        List<LocalPoseSample> vo = voTrack(80);
        RelocalizationConfig c = strictFallback();
        c.temporalConfirmation = RelocalizationConfig.TEMPORAL_DISABLED;   // strict alone decides
        RelocalizationPipeline p = dualSeeded(c, vo);
        AlignmentTransform before = p.aligner().alignment().orElseThrow();
        int epoch = p.aligner().alignmentEpochId();

        RelocalizationPipeline.FrameStep disagree = p.step(vo.get(21), TestProfiles.crossed(21, 2.1, 0, 2), 0.2);
        assertEquals(AcceptanceGate.Reason.VIEW_DISAGREEMENT, disagree.decision().reason());
        assertEquals(Boolean.FALSE, disagree.evidence().westAgreement());
        assertNull(disagree.reanchor());
        assertSame(before, p.aligner().alignment().orElseThrow());

        RelocalizationPipeline.FrameStep missing = p.step(vo.get(22), TestProfiles.observation(22, 2.2, 0, 0), 0.2);
        assertEquals(AcceptanceGate.Reason.WEST_INCOMPLETE, missing.decision().reason());
        assertNull(missing.reanchor());
        assertSame(before, p.aligner().alignment().orElseThrow());

        RelocalizationPipeline.FrameStep invalidWest = p.step(vo.get(23), TestProfiles.westInvalid(23, 2.3, 0, 0), 0.2);
        assertEquals(AcceptanceGate.Reason.WEST_INCOMPLETE, invalidWest.decision().reason());
        assertSame(before, p.aligner().alignment().orElseThrow());

        // Retained requests keep the request pending, so the next queries still execute (no gap).
        RelocalizationPipeline.FrameStep weak = p.step(vo.get(24), TestProfiles.dual(24, 2.4, 3, 40, 40), 0.2);
        assertEquals(AcceptanceGate.Verdict.REJECT, weak.decision().verdict());
        assertNull(weak.reanchor());
        assertSame(before, p.aligner().alignment().orElseThrow());
        assertEquals(epoch, p.aligner().alignmentEpochId(), "no epoch turned over");
        assertTrue(p.aligner().reanchorEvents().isEmpty());
        assertNull(p.aligner().lineage().anchorReferenceId(), "no descendant trust was created");
    }

    // ============================================================ no competing hypothesis (2026-09-08)

    /** Frames 0..7 storing only ONE place, on frame 1. */
    private static RelocalizationPipeline seededSingleton(RelocalizationConfig c,
                                                          List<LocalPoseSample> vo) {
        RelocalizationPipeline p = new RelocalizationPipeline(c);
        for (int f = 0; f <= 7; f++) {
            SkylineObservation obs = f == 1 ? TestProfiles.observation(f, 0.1, 1, 0) : null;
            p.step(vo.get(f), obs, 0.2);
        }
        assertEquals(1, p.memory().size());
        return p;
    }

    @Test
    public void aSingletonMemoryCannotAcceptHoweverPerfectTheMatch() {
        List<LocalPoseSample> vo = voTrack(80);
        RelocalizationConfig c = config();
        c.recentExclusionMode = RelocalizationConfig.MODE_COUNT;
        c.recentExclusionCount = 0;                          // the one reference IS eligible
        RelocalizationPipeline p = seededSingleton(c, vo);
        for (int f = 8; f <= 30; f++) {
            p.step(vo.get(f), null, 0.2);
        }
        assertTrue(p.scheduler().requestPending());
        AlignmentTransform before = p.aligner().alignment().orElseThrow();

        // An EXACT re-observation of the stored place: score 1.0, nothing to compete with it.
        for (int f = 31; f <= 36; f++) {
            RelocalizationPipeline.FrameStep s = p.step(vo.get(f), TestProfiles.observation(f, f * 0.1, 1, 0), 0.2);
            assertNotNull(s.retrieval());
            assertEquals(1.0, s.retrieval().top().score(), 1e-9, "a perfect match, frame " + f);
            assertNull(s.retrieval().competingRegionScore(), "and nothing outside its region");
            assertNull(s.retrieval().regionMargin());
            assertEquals(AcceptanceGate.Verdict.RETAIN, s.decision().verdict(), "frame " + f);
            assertEquals(AcceptanceGate.Reason.NO_COMPETING_HYPOTHESIS, s.decision().reason());
            assertNull(s.reanchor(), "no re-anchor, frame " + f);
            assertSame(before, p.aligner().alignment().orElseThrow(), "the pose never moved");
            assertEquals(0, s.track() == null ? 0 : s.track().supportCount(),
                    "and no confirmation chain is built out of unconfirmable queries");
        }
        assertTrue(p.aligner().reanchorEvents().isEmpty());
        assertEquals(SearchScheduler.AttemptOutcome.RETAINED_FOR_CONFIRMATION,
                p.scheduler().attempts().get(p.scheduler().attempts().size() - 1).outcome());
    }

    @Test
    public void recencyExclusionLeavingOneRegionCannotAccept() {
        // The exact §6 interaction: recent-reference exclusion is preserved, and when it leaves
        // only candidates inside one ambiguity region the evidence is insufficient. This costs an
        // early relocalization; that is the accepted trade.
        List<LocalPoseSample> vo = voTrack(80);
        RelocalizationConfig c = config();                    // regionGapReferences = 1, M = 1
        RelocalizationPipeline p = seeded(c, vo);
        assertEquals(4, p.memory().size());
        for (int f = 8; f <= 30; f++) {
            p.step(vo.get(f), null, 0.2);
        }
        AlignmentTransform before = p.aligner().alignment().orElseThrow();

        RelocalizationPipeline.FrameStep q = p.step(vo.get(31), TestProfiles.observation(31, 3.1, 1, 0), 0.2);
        assertEquals(List.of(3), q.retrieval().excludedRecentIds(), "recency is preserved, not weakened");
        assertEquals(3, q.retrieval().consideredCount(), "0, 1, 2 remain eligible");
        assertEquals(1, q.retrieval().top().referenceId());
        assertEquals(1, q.retrieval().regions().size(),
                "and gap 1 puts all three of them in the winner's region");
        assertNull(q.retrieval().regionMargin(), "so no competing hypothesis exists");
        assertEquals(AcceptanceGate.Verdict.RETAIN, q.decision().verdict());
        assertEquals(AcceptanceGate.Reason.NO_COMPETING_HYPOTHESIS, q.decision().reason());
        assertNull(q.reanchor());
        assertSame(before, p.aligner().alignment().orElseThrow());
    }

    @Test
    public void nullMarginCannotRestorePositionValidityAfterAHardLoss() {
        // The dangerous case: position is UNKNOWN, so any accept would look like pure gain. It is
        // still refused — an unopposed candidate is not evidence of where the aircraft is.
        List<LocalPoseSample> vo = voTrack(80);
        RelocalizationConfig c = config();
        c.recentExclusionMode = RelocalizationConfig.MODE_COUNT;
        c.recentExclusionCount = 0;
        RelocalizationPipeline p = seededSingleton(c, vo);
        for (int f = 8; f <= 20; f++) {
            p.step(vo.get(f), null, 0.2);
        }
        RelocalizationPipeline.FrameStep loss = p.step(TestSamples.loss(21, 1, 28.4), null, 0.2);
        assertEquals(NavigationOutput.FrameEvent.HARD_LOSS, loss.output().event());
        assertFalse(loss.output().globalPositionValid());
        assertEquals(SearchScheduler.Cause.HARD_LOSS, p.scheduler().pendingRequest().cause());

        for (int f = 22; f <= 27; f++) {
            LocalPoseSample s = TestSamples.ok(f, 1, (f - 21) * 2.0, 0.0, 28.4);
            RelocalizationPipeline.FrameStep step = p.step(s, TestProfiles.observation(f, f * 0.1, 1, 0), 0.2);
            assertEquals(1.0, step.retrieval().top().score(), 1e-9);
            assertEquals(AcceptanceGate.Reason.NO_COMPETING_HYPOTHESIS, step.decision().reason(),
                    "frame " + f);
            assertFalse(step.output().globalPositionValid(),
                    "position stays UNKNOWN: a null margin restores nothing, frame " + f);
            assertTrue(step.output().globalPosition().isEmpty());
            assertTrue(step.output().headingValid(), "while the heading channel is unaffected");
        }
        assertTrue(p.aligner().reanchorEvents().isEmpty());
        assertTrue(p.scheduler().requestPending(), "the request is still open, as a RETAIN implies");
    }
}
