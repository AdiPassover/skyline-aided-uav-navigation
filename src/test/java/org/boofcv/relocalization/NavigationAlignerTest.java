package org.boofcv.relocalization;

import org.boofcv.stitching.metric.HeadingReading;
import org.boofcv.stitching.metric.HeadingSemantics;
import org.boofcv.stitching.metric.HeadingStatus;
import org.boofcv.stitching.metric.HeightReading;
import org.boofcv.stitching.metric.HeightStatus;
import org.boofcv.stitching.metric.MetricNavigationState;
import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The alignment-layer semantics on the metric, externally heading-referenced VO contract
 * ({@code DEC-INT-001} amendment 2026-09-07; design §3 frame-convention gate, §4, §10): the
 * required known-answer tests, the split position/heading validity, hard loss with and without a
 * heading, translation dropout, the VO/INT segment invariant, and the equivalence against the real
 * {@link MetricNavigationState}. T1 — says nothing about relocalization on any imagery.
 */
public class NavigationAlignerTest {

    private static final double EPS = 1e-9;

    private static void assertPos(PlanarPosition expected, PlanarPosition actual, double eps, String what) {
        assertEquals(expected.eastM(), actual.eastM(), eps, what + " east");
        assertEquals(expected.northM(), actual.northM(), eps, what + " north");
    }

    private static TrustedReference ref(int id, int frameIndex, PlanarPosition position, double headingDeg,
                                        long baselineExposure) {
        return new TrustedReference(id, frameIndex, frameIndex * 0.1, TestDescriptors.sine(0.01 * id), null,
                position, headingDeg, HeadingStatus.FRESH, 0, 0, null, null, baselineExposure, false,
                TrustedReference.POSE_SCHEMA);
    }

    /** A scripted metric VO on one segment: 1 m steps along a slowly turning heading, metres ENU. */
    private static List<LocalPoseSample> voTrack(int frames) {
        List<LocalPoseSample> track = new ArrayList<>();
        double e = 0, n = 0;
        for (int i = 0; i < frames; i++) {
            double heading = 30.0 + 0.5 * i;
            if (i > 0) {
                e += Math.sin(Math.toRadians(heading));
                n += Math.cos(Math.toRadians(heading));
            }
            track.add(TestSamples.ok(i, 0, e, n, heading));
        }
        return track;
    }

    private static NavigationAligner driven(List<LocalPoseSample> vo, int upToInclusive) {
        NavigationAligner a = new NavigationAligner();
        for (int i = 0; i <= upToInclusive; i++) {
            a.observe(vo.get(i));
        }
        return a;
    }

    // ---------------------------------------------------------------- root convention

    @Test
    public void rootSegmentStartsWithIdentityAlignmentValidPositionAndValidHeading() {
        NavigationAligner a = new NavigationAligner();
        NavigationOutput o = a.observe(TestSamples.ok(0, 0, 0, 0, 100.0));
        assertEquals(NavigationOutput.FrameEvent.INIT, o.event());
        assertEquals(0, o.segmentId());
        assertEquals(0, o.alignmentEpochId());
        assertTrue(o.globalPositionValid());
        assertTrue(o.headingValid());
        assertTrue(o.globalPoseValid());
        assertTrue(a.alignment().orElseThrow().isIdentity(0.0));
        assertEquals(PlanarPosition.ORIGIN, o.localPosition());
        assertEquals(PlanarPosition.ORIGIN, o.globalPosition().orElseThrow());
        assertEquals(100.0, o.globalPose().orElseThrow().yaw, 0.0, "the heading is the VO's, verbatim");
        assertEquals(0, o.lineage().exposureAnchor());
        assertEquals(0, o.lineage().exposureSince());
        assertFalse(o.lineage().effectiveInstability());
        assertNull(o.lineage().anchorReferenceId());
    }

    /** Test B: metric segment position in, metric persistent position out — no other path exists. */
    @Test
    public void whileAlignmentIsIdentityGlobalEqualsLocalInMetres() {
        List<LocalPoseSample> vo = voTrack(30);
        NavigationAligner a = driven(vo, 29);
        NavigationOutput o = a.current();
        assertPos(vo.get(29).segmentPosition(), o.localPosition(), 0.0, "local");
        assertPos(vo.get(29).segmentPosition(), o.globalPosition().orElseThrow(), 0.0, "global");
        assertEquals(vo.get(29).headingDeg(), o.globalPose().orElseThrow().yaw, 0.0);
        assertEquals(29, o.lineage().exposureSince(), "one increment per usable frame after init");
    }

    // ---------------------------------------------------------------- 1. identity re-anchor

    @Test
    public void identityReanchorLeavesThePersistentPositionUnchanged() {
        List<LocalPoseSample> vo = voTrack(20);
        NavigationAligner a = driven(vo, 19);
        PlanarPosition before = a.current().globalPosition().orElseThrow();

        ReanchorEvent e = a.acceptReference(ref(0, 3, before, 77.0, 5));

        assertPos(before, a.current().globalPosition().orElseThrow(), EPS, "global after");
        assertTrue(a.alignment().orElseThrow().isIdentity(EPS));
        assertTrue(e.appliedDelta().isIdentity(EPS));
        assertEquals(0.0, e.positionJumpM(), EPS);
        assertEquals(NavigationOutput.FrameEvent.REANCHOR, a.current().event());
        assertEquals(1, a.alignmentEpochId(), "an epoch still turns over: the anchor changed");
        a.observe(TestSamples.ok(20, 0, vo.get(19).segmentPosition().eastM(),
                vo.get(19).segmentPosition().northM(), vo.get(19).headingDeg()));
        assertPos(vo.get(19).segmentPosition(), a.current().globalPosition().orElseThrow(), EPS, "next frame");
    }

    // ---------------------------------------------------------------- 2. known alignment recovery (test D)

    @Test
    public void aKnownSyntheticTranslationIsRecoveredAndNoRotationIsFitted() {
        AlignmentTransform truth = AlignmentTransform.of(100.0, -50.0);
        List<LocalPoseSample> vo = voTrack(40);
        NavigationAligner a = driven(vo, 39);

        PlanarPosition queryLocal = a.current().localPosition();
        PlanarPosition refGlobal = truth.apply(queryLocal);
        double headingBefore = a.current().headingDeg();
        // The reference was stored with a quite different heading: it must play no part.
        a.acceptReference(ref(0, 10, refGlobal, headingBefore + 137.0, 12));

        AlignmentTransform recovered = a.alignment().orElseThrow();
        assertEquals(truth.tE(), recovered.tE(), EPS);
        assertEquals(truth.tN(), recovered.tN(), EPS);
        assertEquals(headingBefore, a.current().headingDeg(), 0.0, "heading is the VO's, not the reference's");
    }

    // ---------------------------------------------------------------- 3. exact query mapping (tests C, D)

    @Test
    public void afterReanchorTheQueryPositionMapsExactlyOntoTheReferenceAndTheHeadingIsUntouched() {
        List<LocalPoseSample> vo = voTrack(40);
        NavigationAligner a = driven(vo, 39);
        PlanarPosition refGlobal = new PlanarPosition(-812.4, 355.9);
        double voHeading = a.current().headingDeg();

        ReanchorEvent e = a.acceptReference(ref(2, 7, refGlobal, 271.25, 33));

        assertPos(refGlobal, a.current().globalPosition().orElseThrow(), EPS, "query → reference");
        assertPos(refGlobal, e.globalAfter(), EPS, "event");
        assertPos(refGlobal, e.alignmentAfter().apply(e.queryLocalPosition()), EPS, "t_new + p_q");
        assertEquals(voHeading, a.current().globalPose().orElseThrow().yaw, 0.0,
                "the persistent heading is still the authoritative one, not 271.25");
        assertEquals(voHeading, e.headingDeg(), 0.0);

        // Every later frame composes through t_new and nothing else; heading keeps coming from VO.
        for (int i = 40; i < 60; i++) {
            LocalPoseSample s = TestSamples.ok(i, 0, vo.get(39).segmentPosition().eastM() + 0.5 * (i - 39),
                    vo.get(39).segmentPosition().northM(), 200.0 + i);
            NavigationOutput o = a.observe(s);
            assertPos(e.alignmentAfter().apply(o.localPosition()), o.globalPosition().orElseThrow(), EPS,
                    "frame " + i);
            assertEquals(s.headingDeg(), o.globalPose().orElseThrow().yaw, 0.0, "heading at " + i);
        }
    }

    // ---------------------------------------------------------------- 4. historical immutability

    @Test
    public void reanchoringDoesNotRewriteAlreadyEmittedOutputs() {
        List<LocalPoseSample> vo = voTrack(30);
        NavigationAligner a = new NavigationAligner();
        List<NavigationOutput> emitted = new ArrayList<>();
        List<PlanarPosition> emittedGlobal = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            NavigationOutput o = a.observe(vo.get(i));
            emitted.add(o);
            emittedGlobal.add(o.globalPosition().orElseThrow());
        }

        PlanarPosition farAway = new PlanarPosition(5000, -5000);
        AlignmentTransform tNew = a.acceptReference(ref(0, 2, farAway, 180.0, 1)).alignmentAfter();

        for (int i = 0; i < 29; i++) {
            NavigationOutput o = emitted.get(i);
            assertSame(emitted.get(i), o);
            assertPos(emittedGlobal.get(i), o.globalPosition().orElseThrow(), 0.0, "frame " + i + " kept");
            assertEquals(0, o.alignmentEpochId(), "frame " + i + " still belongs to epoch 0");
            assertTrue(tNew.apply(o.localPosition()).distanceTo(o.globalPosition().orElseThrow()) > 1000.0,
                    "history was not re-mapped through t_new at frame " + i);
        }
        assertEquals(29, a.current().frameIndex());
        assertEquals(1, a.current().alignmentEpochId());
    }

    // ---------------------------------------------------------------- 5. hard loss, heading healthy (test E)

    @Test
    public void hardLossWithHealthyHeadingInvalidatesPositionOnceKeepsHeadingAndAddsNoMotion() {
        List<LocalPoseSample> vo = voTrack(50);
        NavigationAligner a = driven(vo, 19);
        PlanarPosition lastGood = a.current().globalPosition().orElseThrow();
        long exposureBefore = a.lineage().effectiveExposure();

        // The metric VO opened segment 1 at this frame; the heading channel never saw the failure.
        NavigationOutput loss = a.observe(TestSamples.loss(20, 1, 44.0));

        assertEquals(NavigationOutput.FrameEvent.HARD_LOSS, loss.event());
        assertEquals(1, loss.segmentId(), "the VO's segment index is adopted");
        assertFalse(loss.globalPositionValid());
        assertTrue(loss.headingValid(), "the heading is the external channel's and is unaffected");
        assertEquals(44.0, loss.headingDeg(), 0.0);
        assertFalse(loss.globalPoseValid());
        assertTrue(loss.globalPosition().isEmpty(), "no persistent position may be handed out");
        assertTrue(loss.globalPose().isEmpty());
        assertTrue(a.alignment().isEmpty(), "t_e is UNKNOWN, not zero");
        assertEquals(PlanarPosition.ORIGIN, loss.localPosition(), "the new segment starts at its own origin");
        assertEquals(1, a.hardLossEvents().size());
        HardLossEvent h = a.hardLossEvents().get(0);
        assertEquals(0, h.closedSegmentId());
        assertEquals(1, h.newSegmentId());
        assertEquals(19, h.lastValidFrameIndex());
        assertTrue(h.headingKnownAcrossGap());
        assertPos(lastGood, h.lastValidGlobalPosition(), 0.0, "closed segment's last valid position kept");
        assertEquals(exposureBefore, loss.lineage().effectiveExposure(), "ledger untouched by loss");
        assertTrue(a.segment().unknownTranslationGapBefore());
        assertTrue(a.segment().headingKnownAcrossGap());

        // Validity changed exactly once (true → false at the loss frame); later frames stay invalid
        // in position and valid in heading.
        int changes = 0;
        boolean prev = loss.globalPositionValid();
        for (int i = 21; i < 30; i++) {
            NavigationOutput o = a.observe(TestSamples.ok(i, 1, 2.0 * (i - 20), 0.0, 44.0 + i));
            if (o.globalPositionValid() != prev) {
                changes++;
                prev = o.globalPositionValid();
            }
            assertTrue(o.globalPosition().isEmpty(), "frame " + i);
            assertTrue(o.headingValid(), "frame " + i);
            assertEquals(1, o.segmentId());
            assertEquals(new PlanarPosition(2.0 * (i - 20), 0.0), o.localPosition());
        }
        assertEquals(0, changes, "the single change was at the loss frame itself");
        assertTrue(a.hardLossEvents().get(0).lastValidFrameIndex() == 19 && !loss.globalPositionValid(),
                "true at 19, false from 20 on: exactly one transition");
        assertEquals(exposureBefore + 9, a.lineage().effectiveExposure(),
                "exposure keeps counting on the new segment");
    }

    // ---------------------------------------------------------------- 6. hard loss, heading unavailable (test F)

    @Test
    public void hardLossWithoutAuthoritativeHeadingFabricatesNeitherPositionNorHeading() {
        List<LocalPoseSample> vo = voTrack(10);
        NavigationAligner a = driven(vo, 9);
        NavigationOutput loss = a.observe(TestSamples.loss(10, 1, Double.NaN));
        assertFalse(loss.globalPositionValid());
        assertFalse(loss.headingValid(), "no heading channel sample: nothing is claimed");
        assertEquals(HeadingStatus.UNAVAILABLE, loss.headingStatus());
        assertTrue(Double.isNaN(loss.headingDeg()));
        assertTrue(loss.heading().isEmpty());
        assertTrue(loss.globalPose().isEmpty());
        assertTrue(loss.localPose().isEmpty(), "no NaN-yawed pose is handed out as if it were one");
        assertFalse(a.hardLossEvents().get(0).headingKnownAcrossGap());
        assertFalse(a.segment().headingKnownAcrossGap());

        // Frames without a heading produce no metric increment: the visual yaw (a number, 41.0 in
        // the sample) is NOT promoted, and the position stays unknown.
        NavigationOutput o = a.observe(TestSamples.noHeading(11, 1, 0.0, 0.0));
        assertEquals(NavigationOutput.FrameEvent.TRANSLATION_DROPOUT, o.event());
        assertFalse(o.headingValid());
        assertTrue(o.globalPose().isEmpty());
        assertEquals(41.0, TestSamples.noHeading(11, 1, 0.0, 0.0).visualYawDeg(), 0.0, "premise");
    }

    /** Test C: a NaN or unusable heading can never enter an output as a valid one. */
    @Test
    public void nanOrUnusableHeadingCannotContaminateAnOutput() {
        assertThrows(IllegalArgumentException.class, () -> new LocalPoseSample(0, 0.0, true, 0, false,
                true, PlanarPosition.ORIGIN, true, HeightStatus.FRESH, Double.NaN, HeadingStatus.FRESH, 0.0),
                "a usable status with no number is a contract violation");
        LocalPoseSample s = new LocalPoseSample(0, 0.0, true, 0, false, false, PlanarPosition.ORIGIN,
                true, HeightStatus.FRESH, 123.0, HeadingStatus.UNAVAILABLE, 0.0);
        assertTrue(Double.isNaN(s.headingDeg()), "an UNAVAILABLE status discards whatever number came with it");
        assertFalse(s.headingValid());
        NavigationAligner a = new NavigationAligner();
        NavigationOutput o = a.observe(s);
        assertTrue(o.globalPositionValid(), "the root position is known ...");
        assertFalse(o.headingValid(), "... but the heading is not");
        assertFalse(o.globalPoseValid());
        assertTrue(o.globalPose().isEmpty(), "position without heading is not a pose");
        assertTrue(o.globalPosition().isPresent());
        // STALE is held-and-used: valid, degraded — VO's own semantics, reused.
        NavigationOutput stale = a.observe(TestSamples.stale(1, 0, 1, 1, 90.0));
        assertTrue(stale.headingValid());
        assertEquals(HeadingStatus.STALE, stale.headingStatus());
    }

    // ---------------------------------------------------------------- 7. segment model invariant (test G)

    @Test
    public void voAndIntSegmentModelsMustAgreeOrTheLayerRefusesToContinue() {
        NavigationAligner a = new NavigationAligner();
        assertThrows(IllegalStateException.class, () -> a.observe(TestSamples.ok(0, 3, 0, 0, 10.0)),
                "the first sample must be VO segment 0");
        a.observe(TestSamples.ok(0, 0, 0, 0, 10.0));
        a.observe(TestSamples.ok(1, 0, 1, 0, 10.0));
        assertThrows(IllegalStateException.class, () -> a.observe(TestSamples.ok(2, 1, 2, 0, 10.0)),
                "the VO advanced its segment on a successful frame: divergence, refused");
        NavigationAligner b = new NavigationAligner();
        b.observe(TestSamples.ok(0, 0, 0, 0, 10.0));
        assertThrows(IllegalStateException.class, () -> b.observe(TestSamples.loss(1, 2, 10.0)),
                "a hard loss must open exactly the next VO segment");
        // The agreeing sequence is fine: loss → segment 1, loss → segment 2.
        NavigationAligner c = new NavigationAligner();
        c.observe(TestSamples.ok(0, 0, 0, 0, 10.0));
        c.observe(TestSamples.loss(1, 1, 10.0));
        c.observe(TestSamples.ok(2, 1, 1, 0, 10.0));
        c.observe(TestSamples.loss(3, 2, 10.0));
        assertEquals(2, c.current().segmentId());
        assertEquals(2, c.hardLossEvents().size());
        assertEquals(2, c.alignmentEpochId(), "one epoch per loss, none for ordinary frames");
    }

    // ---------------------------------------------------------------- 8. translation dropout

    @Test
    public void aTranslationDropoutInvalidatesThePositionOnceAndRefusesAReanchorOnThatFrame() {
        List<LocalPoseSample> vo = voTrack(10);
        NavigationAligner a = driven(vo, 9);
        PlanarPosition held = vo.get(9).segmentPosition();
        NavigationOutput d1 = a.observe(TestSamples.noHeight(10, 0, held.eastM(), held.northM(), 34.0));
        assertEquals(NavigationOutput.FrameEvent.TRANSLATION_DROPOUT, d1.event());
        assertFalse(d1.globalPositionValid(), "the segment position now hides an unobserved displacement");
        assertTrue(d1.headingValid(), "a height dropout says nothing about the heading");
        assertEquals(0, d1.segmentId(), "the VO segment continues");
        assertEquals(1, a.alignmentEpochId());
        assertEquals(1, a.translationDropoutEvents().size());
        assertEquals(HeightStatus.UNAVAILABLE, a.translationDropoutEvents().get(0).heightStatus());
        assertEquals(9, a.translationDropoutEvents().get(0).lastValidFrameIndex());
        assertThrows(IllegalStateException.class,
                () -> a.acceptReference(ref(0, 1, new PlanarPosition(1, 1), 34.0, 0)),
                "no alignment may be established on a frame whose displacement was not observed");

        NavigationOutput d2 = a.observe(TestSamples.noHeight(11, 0, held.eastM(), held.northM(), 34.0));
        assertEquals(1, a.alignmentEpochId(), "already unknown: no second epoch");
        assertEquals(1, a.translationDropoutEvents().size());
        assertEquals(2, a.unusableFrameCount());
        assertEquals(9, d2.lineage().effectiveExposure(), "unusable frames are not exposure");

        // A usable frame afterwards is still unknown until a reference is accepted.
        NavigationOutput back = a.observe(TestSamples.ok(12, 0, held.eastM() + 1, held.northM(), 34.0));
        assertFalse(back.globalPositionValid());
        a.acceptReference(ref(0, 1, new PlanarPosition(700, 800), 34.0, 3));
        assertTrue(a.current().globalPositionValid());
        assertPos(new PlanarPosition(700, 800), a.current().globalPosition().orElseThrow(), EPS, "restored");
    }

    // ---------------------------------------------------------------- 9. re-anchor after hard loss

    @Test
    public void aTrustedReferenceRestoresValidPositionForTheNewSegmentWithoutTouchingTheHeading() {
        List<LocalPoseSample> vo = voTrack(60);
        NavigationAligner a = driven(vo, 19);
        a.observe(TestSamples.loss(20, 1, 44.0));
        for (int i = 21; i <= 30; i++) {
            a.observe(TestSamples.ok(i, 1, i - 20.0, 0.5 * (i - 20), 44.0));
        }
        assertFalse(a.globalPositionValid());
        assertTrue(a.headingValid());

        PlanarPosition refGlobal = new PlanarPosition(240.0, -60.0);
        ReanchorEvent e = a.acceptReference(ref(5, 4, refGlobal, 88.0, 44));

        assertTrue(a.globalPositionValid());
        assertTrue(a.globalPoseValid());
        assertNull(e.globalBefore(), "there was no persistent position to jump from");
        assertNull(e.alignmentBefore());
        assertNull(e.appliedDelta());
        assertNull(e.positionJumpM());
        assertEquals(1, e.segmentId());
        assertPos(refGlobal, a.current().globalPosition().orElseThrow(), EPS, "snapped exactly");
        assertEquals(44.0, a.current().globalPose().orElseThrow().yaw, 0.0, "heading: VO's 44, not the reference's 88");
        assertEquals(44, a.lineage().effectiveExposure(), "baseline inherited from the reference");
        assertEquals(5, a.lineage().anchorReferenceId());
        assertEquals(30, a.lineage().anchorFrameIndex(), "the re-anchor frame establishes the lineage");

        AlignmentTransform tNew = e.alignmentAfter();
        for (int i = 31; i < 45; i++) {
            NavigationOutput o = a.observe(TestSamples.ok(i, 1, i - 20.0, 0.5 * (i - 20), 44.0 + 0.1 * i));
            assertPos(tNew.apply(o.localPosition()), o.globalPosition().orElseThrow(), EPS, "frame " + i);
            assertEquals(1, o.segmentId());
        }
    }

    // ---------------------------------------------------------------- 10. reject / no-op

    @Test
    public void aRejectedCandidateChangesNothing() {
        List<LocalPoseSample> vo = voTrack(20);
        NavigationAligner a = driven(vo, 19);
        a.noteInstability();
        NavigationOutput before = a.current();
        AlignmentTransform alignmentBefore = a.alignment().orElseThrow();
        AnchorLineage.Snapshot lineageBefore = a.lineage();

        NavigationAligner.RejectedCandidate r = a.reject(7, "region margin below threshold");

        assertSame(before, a.current(), "no new output was published");
        assertSame(alignmentBefore, a.alignment().orElseThrow());
        assertEquals(lineageBefore, a.lineage(), "exposure and instability untouched");
        assertEquals(0, a.alignmentEpochId());
        assertEquals(1, a.rejections().size());
        assertEquals(7, r.candidateReferenceId());
        assertThrows(IllegalArgumentException.class, () -> a.reject(null, " "));
    }

    // ---------------------------------------------------------------- 11. lineage transitions

    @Test
    public void lineageTransitionsAcrossIncrementsInstabilityLossAndAcceptance() {
        List<LocalPoseSample> vo = voTrack(200);
        NavigationAligner a = driven(vo, 100);
        assertEquals(100, a.lineage().exposureSince());

        a.noteInstability();
        assertTrue(a.current().lineage().instabilitySince(), "current output reflects it");
        for (int i = 101; i <= 150; i++) {
            a.observe(vo.get(i));
            assertTrue(a.lineage().effectiveInstability(), "sticky at " + i);
        }
        assertEquals(150, a.lineage().effectiveExposure(), "no decay");

        a.observe(TestSamples.loss(151, 1, 10.0));
        assertEquals(150, a.lineage().effectiveExposure(), "loss does not touch the ledger");
        assertTrue(a.lineage().effectiveInstability());
        a.observe(TestSamples.ok(152, 1, 1, 0, 10.0));
        assertEquals(151, a.lineage().effectiveExposure());

        a.acceptReference(ref(3, 40, new PlanarPosition(10, 10), 10.0, 70));
        assertEquals(70, a.lineage().exposureAnchor());
        assertEquals(0, a.lineage().exposureSince());
        assertEquals(70, a.lineage().effectiveExposure(), "70, not 0");
        assertFalse(a.lineage().instabilitySince(), "cleared only by the accepted anchor");
        assertFalse(a.lineage().effectiveInstability());
        for (int i = 153; i < 173; i++) {
            a.observe(TestSamples.ok(i, 1, i - 150.0, 0, 10.0));
        }
        assertEquals(90, a.lineage().effectiveExposure(), "70 + 20");
    }

    // ---------------------------------------------------------------- 12. the real metric state

    /**
     * Driven by the real {@link MetricNavigationState} with an authoritative heading: the layer's
     * local position is {@code segmentRelativePose()} verbatim, its persistent position is that plus
     * the translation, a restart is adopted as the VO's segment with the position absent while the
     * VO's <em>continuous</em> metric pose (the convenience trajectory that inserts nothing across
     * the gap) is never consumed, and re-anchoring afterwards is translation-only.
     */
    @Test
    public void followsTheRealMetricStateVerbatimAndNeverConsumesTheContinuousPose() {
        final double fWorking = 800.0;
        MetricNavigationState m = new MetricNavigationState(fWorking, true);
        NavigationAligner a = new NavigationAligner();

        m.observeOrigin(height(0.0), heading(0.0, 30.0));
        a.observe(LocalPoseSample.fromMetric(0, 0.0, true, m, new Pose3D()));
        for (int k = 1; k <= 20; k++) {
            m.observe(10.0, -4.0, Math.toRadians(0.2), height(k * 0.1), heading(k * 0.1, 30.0 + 0.4 * k));
            NavigationOutput o = a.observe(LocalPoseSample.fromMetric(k, k * 0.1, true, m, new Pose3D()));
            Pose3D seg = m.segmentRelativePose();
            assertEquals(seg.x, o.localPosition().eastM(), 0.0, "east verbatim at " + k);
            assertEquals(seg.y, o.localPosition().northM(), 0.0, "north verbatim at " + k);
            assertEquals(m.navHeadingDeg(), o.headingDeg(), 0.0, "heading verbatim at " + k);
            assertEquals(o.localPosition(), o.globalPosition().orElseThrow(), "root: t = 0");
        }
        Pose3D continuousBefore = m.metricPose();
        assertNotEquals(0.0, continuousBefore.x, "the aircraft moved");

        // A stitching restart, as MotionModelStitchingEstimator performs it: no increment,
        // beginNewSegment(true), observeOrigin at the restart frame.
        m.beginNewSegment(true);
        m.observeOrigin(height(2.1), heading(2.1, 38.0));
        NavigationOutput loss = a.observe(LocalPoseSample.fromMetric(21, 2.1, false, m, new Pose3D()));
        assertEquals(1, m.getSegmentIndex());
        assertEquals(1, loss.segmentId());
        assertTrue(m.isUnknownTranslationGapBeforeSegment());
        assertTrue(m.isHeadingKnownAcrossGap());
        assertEquals(PlanarPosition.ORIGIN, loss.localPosition(), "segment origin, not the continuous pose");
        assertTrue(loss.globalPosition().isEmpty());
        assertTrue(loss.headingValid());
        assertEquals(38.0, loss.headingDeg(), 0.0);
        assertEquals(continuousBefore.x, m.metricPose().x, 0.0,
                "the VO's continuous pose is numerically continuous across the gap (zero inserted) ...");
        assertFalse(loss.globalPositionValid(),
                "... and that continuity is NOT read as known zero motion: the position is UNKNOWN");

        for (int k = 22; k <= 25; k++) {
            m.observe(10.0, 0.0, 0.0, height(k * 0.1), heading(k * 0.1, 38.0));
            NavigationOutput o = a.observe(LocalPoseSample.fromMetric(k, k * 0.1, true, m, new Pose3D()));
            assertEquals(m.segmentRelativePose().x, o.localPosition().eastM(), 0.0);
            assertTrue(o.globalPosition().isEmpty());
        }
        PlanarPosition refPos = new PlanarPosition(1000.0, 2000.0);
        ReanchorEvent e = a.acceptReference(ref(0, 5, refPos, 300.0, 7));
        assertPos(refPos, a.current().globalPosition().orElseThrow(), 0.0, "snapped");
        assertEquals(38.0, a.current().globalPose().orElseThrow().yaw, 0.0, "heading stays the VO's");
        m.observe(0.0, -10.0, 0.0, height(2.6), heading(2.6, 38.0));      // image-up = North at heading 38
        NavigationOutput o = a.observe(LocalPoseSample.fromMetric(26, 2.6, true, m, new Pose3D()));
        assertPos(e.alignmentAfter().apply(o.localPosition()), o.globalPosition().orElseThrow(), 0.0, "t + p");
    }

    /** Test C: a metric state whose direction is the visual yaw is refused as an input. */
    @Test
    public void theVisualYawArmIsRefusedAsAnInput() {
        MetricNavigationState visual = new MetricNavigationState(800.0, false);
        visual.observeOrigin(height(0.0));
        assertThrows(IllegalStateException.class,
                () -> LocalPoseSample.fromMetric(0, 0.0, true, visual, new Pose3D()));
    }

    // ---------------------------------------------------------------- guards

    @Test
    public void oneBinaryActionPerFrameAndMonotoneFrames() {
        List<LocalPoseSample> vo = voTrack(10);
        NavigationAligner a = driven(vo, 9);
        a.acceptReference(ref(0, 1, new PlanarPosition(1, 1), 1.0, 0));
        assertThrows(IllegalStateException.class,
                () -> a.acceptReference(ref(1, 2, new PlanarPosition(2, 2), 2.0, 0)));
        assertThrows(IllegalArgumentException.class, () -> a.observe(vo.get(9)));
        assertThrows(IllegalStateException.class, () -> new NavigationAligner().current());
        assertNotEquals(0, a.alignmentEpochId());
    }

    private static HeightReading height(double t) {
        return new HeightReading(t, t, 0.0, 0.0, 80.0, 80.0, HeightStatus.FRESH);
    }

    private static HeadingReading heading(double t, double deg) {
        return new HeadingReading(t, t, 0.0, deg, 0.0, deg, HeadingSemantics.SIM_NADIR_CAMERA_HEADING,
                HeadingStatus.FRESH);
    }
}
