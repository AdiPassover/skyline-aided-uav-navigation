package org.boofcv.stitching;

import boofcv.struct.image.GrayF32;
import org.boofcv.relocalization.LocalPoseSample;
import org.boofcv.relocalization.NavigationAligner;
import org.boofcv.relocalization.NavigationOutput;
import org.boofcv.relocalization.PlanarPosition;
import org.boofcv.relocalization.SkylineDescriptor;
import org.boofcv.relocalization.TrustedReference;
import org.boofcv.stitching.ScriptedStitchingHarness.ScriptedStitching;
import org.boofcv.stitching.metric.HeadingSemantics;
import org.boofcv.stitching.metric.HeadingStatus;
import org.boofcv.stitching.metric.MetricNavigationState;
import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.boofcv.stitching.ScriptedStitchingHarness.DT;
import static org.boofcv.stitching.ScriptedStitchingHarness.FRAME;
import static org.boofcv.stitching.ScriptedStitchingHarness.headingEstimator;
import static org.boofcv.stitching.ScriptedStitchingHarness.translation;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * <b>A mosaic re-origin ("recenter") is tracker bookkeeping. A hard loss is a navigation event.</b>
 * They are not the same thing and this class is the code-level evidence, taken end to end: the real
 * {@link MotionModelStitchingEstimator} drives a real {@link MetricNavigationState}, whose readout
 * is handed to the real {@link NavigationAligner} through the one contract the layer consumes
 * ({@link LocalPoseSample#fromMetric}).
 *
 * <h2>Why it needs stating</h2>
 *
 * <p>Because the two events are adjacent in the estimator and one of them looks alarming from
 * outside. {@code reoriginMosaicCanvas()} calls {@code stitch.setOriginToCurrent()}, and BoofCV
 * welds an estimator reset to that — {@code setToFirst()} zeroes the accumulated transform. Read
 * carelessly, "the accumulation was zeroed" sounds exactly like a restart. It is not: the
 * accumulation is folded into the navigation anchor immediately beforehand and
 * {@code lastGoodFirstToCurrent} is reset in the same breath, so the next frame's increment is
 * measured against the new key frame and no motion is either lost or double counted. The metric
 * state is not touched by the re-origin path <em>at all</em> — no {@code beginNewSegment}, no
 * {@code observeOrigin}.
 *
 * <p>A hard loss is the opposite: the interval that failed was never estimated ({@code COMP-001}
 * §8), so {@code beginNewSegment(true)} terminates the segment and marks the displacement across
 * the boundary UNKNOWN, and nothing may bridge it.
 *
 * <p>The practical question this answers: a run record reporting <em>"12 recenters, 0 restarts,
 * 100 % usable VO"</em> is describing <b>one uninterrupted metric translation segment</b>. See
 * {@link #manyRecentersAreStillOneUninterruptedSegment()}.
 *
 * <p>Evidence tier: T1 (synthetic). {@code DEC-VO-009}, {@code DEC-VO-010}, {@code DEC-INT-001}.
 */
public class RecenterVersusRestartTest {

    private static final double MOUNT = 0.0;

    /** Drives the estimator and mirrors every frame into an aligner, as the capture path does. */
    private static final class Rig {
        final ScriptedStitching stitch;
        final MotionModelStitchingEstimator<GrayF32, ?> est;
        final NavigationAligner aligner = new NavigationAligner();
        final GrayF32 img = new GrayF32(FRAME, FRAME);
        final List<NavigationOutput> outputs = new ArrayList<>();
        int frame = 0;

        Rig(int recenterPeriod) {
            stitch = new ScriptedStitching(recenterPeriod);
            est = headingEstimator(stitch, HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null);
            est.submitHeightSample(0.0, 0.0);
            est.submitHeadingSample(0.0, MOUNT);
            boolean ok = est.processFrame(img, 0.0);
            outputs.add(observe(ok, 0.0));
        }

        /** One ordinary frame: 10 px east, height and heading healthy. */
        NavigationOutput step() {
            return step(false);
        }

        /** One frame, optionally with a synthetic hard loss injected onto it. */
        NavigationOutput step(boolean injectLoss) {
            frame++;
            double t = frame * DT;
            stitch.motion.advance(translation(10.0));
            est.submitHeightSample(t, 0.0);
            est.submitHeadingSample(t, MOUNT);
            if (injectLoss) {
                est.injectHardLossOnNextFrame();
            }
            boolean ok = est.processFrame(img, t);
            NavigationOutput o = observe(ok, t);
            outputs.add(o);
            return o;
        }

        private NavigationOutput observe(boolean voSuccess, double t) {
            MetricNavigationState m = est.getMetricNavigation();
            return aligner.observe(LocalPoseSample.fromMetric(frame, t, voSuccess, m,
                    est.getRigidPose()));
        }

        MetricNavigationState metric() {
            return est.getMetricNavigation();
        }
    }

    // ================================================================ B: the recenter contract

    @Test
    @DisplayName("B: an ordinary recenter changes no navigation state at all — segment, validity, XY, heading")
    void anOrdinaryRecenterIsInvisibleToNavigation() {
        Rig r = new Rig(7);                                   // a re-origin on every 7th frame
        for (int k = 0; k < 6; k++) {
            r.step();
        }
        // BEFORE: healthy, segment S, no gap.
        MetricNavigationState m = r.metric();
        int segmentBefore = m.getSegmentIndex();
        int epochBefore = r.outputs.get(r.outputs.size() - 1).alignmentEpochId();
        double eastBefore = m.eastM();
        double headingBefore = m.navHeadingDeg();
        assertEquals(0, segmentBefore);
        assertFalse(m.isUnknownTranslationGapBeforeSegment());
        assertTrue(r.outputs.get(r.outputs.size() - 1).globalPositionValid());

        // The frame the canvas re-origins on, and the two after it.
        NavigationOutput atRecenter = r.step();
        NavigationOutput after1 = r.step();
        NavigationOutput after2 = r.step();

        // AFTER: every single expectation of the recenter contract.
        assertEquals(segmentBefore, m.getSegmentIndex(),
                "a re-origin is bookkeeping in the mosaic, not an unestimated navigation interval");
        assertFalse(m.isUnknownTranslationGapBeforeSegment(),
                "no motion was lost, so nothing across the boundary is unknown");
        assertTrue(atRecenter.globalPositionValid(), "the persistent position survives a re-origin");
        assertTrue(after1.globalPositionValid());
        assertTrue(after2.globalPositionValid());
        assertEquals(NavigationOutput.FrameEvent.NONE, atRecenter.event(),
                "a re-origin is not a navigation event of any kind");
        assertEquals(epochBefore, after2.alignmentEpochId(),
                "no alignment epoch is opened by a re-origin: nothing re-anchored");
        assertEquals(segmentBefore, after2.segmentId());

        // Metric XY is continuous across it: three more 10 px steps at 0.1 m/px is exactly +3 m,
        // with no reset to a local origin and no jump.
        assertEquals(eastBefore + 3.0, m.eastM(), 1e-9,
                "accumulated metric XY does not reset when the mosaic origin moves");
        assertEquals(headingBefore, m.navHeadingDeg(), 0.0, "the heading channel is untouched");
        assertTrue(m.lastFrameUsable(), "the re-origin frame still produced its own increment");
        assertEquals(HeadingStatus.FRESH, atRecenter.headingStatus());
    }

    @Test
    @DisplayName("B: many recenters are still ONE uninterrupted segment — '12 recenters, 0 restarts' is coherent")
    void manyRecentersAreStillOneUninterruptedSegment() {
        // 12 re-origins over 60 frames, exactly the shape the figure-8 run records report.
        Rig r = new Rig(5);
        for (int k = 0; k < 60; k++) {
            r.step();
        }
        MetricNavigationState m = r.metric();
        assertEquals(12, r.stitch.frame / 5, "the schedule really did re-origin twelve times");
        assertEquals(0, m.getSegmentIndex(), "one segment, start to finish");
        assertFalse(m.isUnknownTranslationGapBeforeSegment());
        assertEquals(0, m.getUnusableFrameCount(), "100 % of frames produced a metric increment");
        assertEquals(60.0, m.eastM(), 1e-9, "60 steps of 10 px at 0.1 m/px, none lost to a re-origin");
        assertEquals(60.0, m.segmentRelativePose().x, 1e-9,
                "and the SEGMENT-relative position — what INT consumes — is the same number");
        for (NavigationOutput o : r.outputs) {
            assertTrue(o.globalPositionValid(), "position never became unknown at frame " + o.frameIndex());
        }
        assertEquals(0, r.aligner.hardLossEvents().size(), "no hard loss was ever reported");
    }

    @Test
    @DisplayName("B: a re-origin is bit-identical to no re-origin — the schedule cannot enter navigation")
    void theCanvasScheduleCannotEnterTheMetricTrack() {
        Rig none = new Rig(0);
        Rig often = new Rig(3);
        for (int k = 0; k < 40; k++) {
            none.step();
            often.step();
        }
        assertEquals(none.metric().eastM(), often.metric().eastM(), 0.0,
                "the canvas re-origin schedule is not a navigation input");
        assertEquals(none.metric().northM(), often.metric().northM(), 0.0);
        assertEquals(none.metric().getSegmentIndex(), often.metric().getSegmentIndex(), "0 both ways");
    }

    // ================================================================ B: the contrast

    @Test
    @DisplayName("B: a true hard loss does every one of the things a recenter does not")
    void aHardLossIsTheOppositeOfARecenter() {
        Rig r = new Rig(0);
        for (int k = 0; k < 10; k++) {
            r.step();
        }
        MetricNavigationState m = r.metric();
        assertEquals(0, m.getSegmentIndex());
        assertEquals(10.0, m.eastM(), 1e-9);
        assertTrue(r.outputs.get(r.outputs.size() - 1).globalPositionValid());

        NavigationOutput loss = r.step(true);

        assertEquals(1, m.getSegmentIndex(), "the segment terminated");
        assertTrue(m.isUnknownTranslationGapBeforeSegment(), "the displacement across it is UNKNOWN");
        assertFalse(loss.globalPositionValid(), "and therefore the persistent position is unknown");
        assertEquals(NavigationOutput.FrameEvent.HARD_LOSS, loss.event());
        assertEquals(0.0, m.segmentRelativePose().x, 0.0,
                "the new segment starts at its own origin, not at the old segment's position");
        assertEquals(1, r.aligner.hardLossEvents().size());
    }

    // ================================================================ C: synthetic == natural

    @Test
    @DisplayName("C: a synthetic hard loss and a natural one produce the identical downstream state")
    void syntheticAndNaturalHardLossAreTheSameEvent() {
        Rig synthetic = new Rig(0);
        Rig natural = new Rig(0);
        for (int k = 0; k < 8; k++) {
            synthetic.step();
            natural.step();
        }
        // Synthetic: armed by the caller. Natural: the stitcher itself fails on the same frame.
        synthetic.step(true);
        natural.stitch.forceFailure = true;
        natural.step();
        natural.stitch.forceFailure = false;

        for (int k = 0; k < 8; k++) {
            synthetic.step();
            natural.step();
        }

        MetricNavigationState s = synthetic.metric();
        MetricNavigationState n = natural.metric();
        assertEquals(n.getSegmentIndex(), s.getSegmentIndex(), "same segment index");
        assertEquals(n.isUnknownTranslationGapBeforeSegment(),
                s.isUnknownTranslationGapBeforeSegment(), "same gap semantics");
        assertEquals(n.isHeadingKnownAcrossGap(), s.isHeadingKnownAcrossGap(), "same heading survival");
        assertEquals(n.segmentRelativePose().x, s.segmentRelativePose().x, 1e-12,
                "same segment-relative position after recovery");
        assertEquals(n.segmentRelativePose().y, s.segmentRelativePose().y, 1e-12);
        assertEquals(n.getUnusableFrameCount(), s.getUnusableFrameCount());
        assertEquals(natural.aligner.hardLossEvents().size(),
                synthetic.aligner.hardLossEvents().size());
        for (int i = 0; i < natural.outputs.size(); i++) {
            assertEquals(natural.outputs.get(i).globalPositionValid(),
                    synthetic.outputs.get(i).globalPositionValid(),
                    "position validity diverged at frame " + i);
            assertEquals(natural.outputs.get(i).event(), synthetic.outputs.get(i).event(),
                    "event diverged at frame " + i);
        }
        assertEquals(1, synthetic.est.getSyntheticHardLossCount(),
                "and only the synthetic run is labelled as such");
        assertEquals(0, natural.est.getSyntheticHardLossCount());
    }

    @Test
    @DisplayName("C: injection is one-shot, deterministic, refuses a double-arm, and is off by default")
    void injectionIsOneShotAndDeterministic() {
        Rig r = new Rig(0);
        assertFalse(r.est.isHardLossInjectionArmed(), "nothing is armed unless a caller asks");
        r.est.injectHardLossOnNextFrame();
        assertTrue(r.est.isHardLossInjectionArmed());
        assertThrows(IllegalStateException.class, r.est::injectHardLossOnNextFrame,
                "a second arm would silently drop one of the two requested losses");
        r.step();                                             // consumes it
        assertFalse(r.est.isHardLossInjectionArmed(), "one-shot: it does not persist");
        assertEquals(1, r.metric().getSegmentIndex());

        // The very same script produces the very same losses every time.
        int[] lossFrames = {4, 5, 11};
        List<Integer> a = segmentsFor(lossFrames);
        List<Integer> b = segmentsFor(lossFrames);
        assertEquals(a, b, "same data + same config => the same losses at the same frames");
        assertEquals(3, a.get(a.size() - 1), "three injected losses, three segment increments");
    }

    private static List<Integer> segmentsFor(int[] lossFrames) {
        Rig r = new Rig(0);
        List<Integer> segments = new ArrayList<>();
        for (int k = 1; k <= 15; k++) {
            boolean inject = false;
            for (int f : lossFrames) {
                inject |= f == k;
            }
            r.step(inject);
            segments.add(r.metric().getSegmentIndex());
        }
        return segments;
    }

    @Test
    @DisplayName("C: two forced losses — segment indices and alignment epochs both behave")
    void twoForcedLossesBehave() {
        Rig r = new Rig(0);
        for (int k = 0; k < 5; k++) {
            r.step();
        }
        int epoch0 = r.outputs.get(r.outputs.size() - 1).alignmentEpochId();

        r.step(true);
        for (int k = 0; k < 5; k++) {
            r.step();
        }
        assertEquals(1, r.metric().getSegmentIndex());
        assertEquals(5.0, r.metric().segmentRelativePose().x, 1e-9,
                "five healthy frames since the first loss");

        r.step(true);
        for (int k = 0; k < 4; k++) {
            r.step();
        }
        assertEquals(2, r.metric().getSegmentIndex());
        assertEquals(4.0, r.metric().segmentRelativePose().x, 1e-9);
        assertTrue(r.metric().isUnknownTranslationGapBeforeSegment());
        assertEquals(2, r.aligner.hardLossEvents().size());
        // Each loss opens one epoch, because the alignment CHANGED — to "unknown". That is the
        // epoch id's job: it names which alignment was in force, and absent is a distinct answer
        // from the previous one. Contrast anOrdinaryRecenterIsInvisibleToNavigation, where the
        // alignment does not change and no epoch is opened.
        assertEquals(epoch0 + 2, r.outputs.get(r.outputs.size() - 1).alignmentEpochId(),
                "one epoch per hard loss — the alignment became UNKNOWN both times");
        assertFalse(r.outputs.get(r.outputs.size() - 1).globalPositionValid());
    }

    // ================================================================ C: the recovery contract

    @Test
    @DisplayName("C: across the gap the heading stays authoritative and NO translation is bridged")
    void headingSurvivesAndNothingBridgesTheGap() {
        Rig r = new Rig(0);
        for (int k = 0; k < 6; k++) {
            r.step();
        }
        double headingBefore = r.metric().navHeadingDeg();
        double eastBefore = r.metric().eastM();

        NavigationOutput loss = r.step(true);

        assertEquals(headingBefore, loss.headingDeg(), 0.0, "the sensor never saw the visual failure");
        assertTrue(loss.headingValid(), "so heading validity is untouched by a translation gap");
        assertTrue(r.metric().isHeadingKnownAcrossGap());
        assertFalse(loss.globalPositionValid());
        assertFalse(loss.globalPoseValid(), "the aggregate follows its weakest component");
        assertTrue(loss.globalPosition().isEmpty(), "absent, not degraded");
        assertTrue(loss.heading().isPresent());

        // No zero-motion insertion, and no continuous convenience pose used to bridge the gap.
        NavigationOutput first = r.step();
        assertEquals(1.0, r.metric().segmentRelativePose().x, 1e-9,
                "the new segment measures ONE step, from its own origin");
        NavigationOutput second = r.step();
        assertEquals(2.0, r.metric().segmentRelativePose().x, 1e-9, "then two — plain accumulation");
        assertNotEquals(eastBefore, r.metric().segmentRelativePose().x,
                "and it is not a continuation of the old segment's number");
        assertFalse(first.globalPositionValid(), "still UNKNOWN: only a trusted reference restores it");
        assertFalse(second.globalPositionValid());
        assertTrue(first.translationUsable(), "the frames themselves are healthy again");
    }

    @Test
    @DisplayName("C: a trusted reference after a synthetic loss restores POSITION only")
    void relocalizationAfterASyntheticLossRestoresPositionOnly() {
        Rig r = new Rig(0);
        for (int k = 0; k < 6; k++) {
            r.step();
        }
        r.step(true);
        NavigationOutput before = r.step();
        assertFalse(before.globalPositionValid());
        double headingBefore = before.headingDeg();

        TrustedReference ref = reference(new PlanarPosition(500.0, -250.0), headingBefore);
        r.aligner.acceptReference(ref);
        NavigationOutput after = r.aligner.current();

        assertTrue(after.globalPositionValid(), "recovered");
        assertEquals(500.0, after.globalPosition().orElseThrow().eastM(), 1e-9);
        assertEquals(-250.0, after.globalPosition().orElseThrow().northM(), 1e-9);
        assertEquals(headingBefore, after.headingDeg(), 0.0,
                "a discrete reference snap is a TRANSLATION; it never rewrites the heading");
        assertEquals(1, r.metric().getSegmentIndex(), "and it does not undo the segment boundary");
        assertTrue(r.metric().isUnknownTranslationGapBeforeSegment(),
                "the gap stays UNKNOWN as history — recovery does not make it estimated");

        // The next healthy frame moves on from the recovered position by its own increment.
        r.step();
        assertEquals(501.0, r.aligner.current().globalPosition().orElseThrow().eastM(), 1e-9,
                "one 10 px step at 0.1 m/px, added to the recovered position");
    }

    @Test
    @DisplayName("C: a rejected candidate after a synthetic loss leaves the position UNKNOWN")
    void aRejectedCandidateAfterALossRestoresNothing() {
        Rig r = new Rig(0);
        for (int k = 0; k < 6; k++) {
            r.step();
        }
        r.step(true);
        NavigationOutput before = r.step();
        assertFalse(before.globalPositionValid());

        r.aligner.reject(3, "weak_match");
        NavigationOutput after = r.aligner.current();

        assertSame(before, after, "a rejection changes no state whatsoever");
        assertFalse(after.globalPositionValid(), "still UNKNOWN");
        r.step();
        assertFalse(r.aligner.current().globalPositionValid(),
                "and no amount of healthy VO restores it on its own");
    }

    private static TrustedReference reference(PlanarPosition p, double headingDeg) {
        double[] profile = new double[256];
        for (int i = 0; i < profile.length; i++) {
            profile[i] = 0.5 + 0.2 * Math.sin(2 * Math.PI * 2 * i / (double) profile.length);
        }
        SkylineDescriptor d = SkylineDescriptor.of(profile, 256, 1e-6);
        return new TrustedReference(0, 0, 0.0, d, null, p, headingDeg, HeadingStatus.FRESH,
                0, 0, null, null, 0, false, TrustedReference.POSE_SCHEMA);
    }
}
