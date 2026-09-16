package org.boofcv.stitching;

import boofcv.struct.image.GrayF32;
import georegression.struct.homography.Homography2D_F64;
import org.boofcv.stitching.ScriptedStitchingHarness.ScriptedStitching;
import org.boofcv.stitching.metric.HeadingReadoutConfig;
import org.boofcv.stitching.metric.HeadingSemantics;
import org.boofcv.stitching.metric.HeadingStatus;
import org.boofcv.stitching.metric.HeightStatus;
import org.boofcv.stitching.metric.MetricNavigationState;
import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.boofcv.stitching.ScriptedStitchingHarness.DT;
import static org.boofcv.stitching.ScriptedStitchingHarness.FRAME;
import static org.boofcv.stitching.ScriptedStitchingHarness.FX;
import static org.boofcv.stitching.ScriptedStitchingHarness.H0;
import static org.boofcv.stitching.ScriptedStitchingHarness.estimator;
import static org.boofcv.stitching.ScriptedStitchingHarness.headingEstimator;
import static org.boofcv.stitching.ScriptedStitchingHarness.rotation;
import static org.boofcv.stitching.ScriptedStitchingHarness.translation;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The metric readout <b>through the estimator</b>: the wiring, the restart semantics (H), and the
 * guarantee that turning it on changes nothing about the raw visual path.
 *
 * <p>Uses the same scripted-mock approach as {@link RigidMotionEstimatorInvarianceTest}, and for the
 * same reason: only a mock can hold the accepted image motion fixed while the canvas schedule and
 * the failure schedule vary. On the real BoofCV stack a forced failure would also change tracking,
 * so a comparison would measure that instead of the thing under test.
 *
 * <p>Evidence tier: T1 (synthetic).
 */
public class MetricRuntimeIntegrationTest {

    // ================================================================ wiring

    @Test
    @DisplayName("the metric readout is OFF by default and METRIC_LOCAL cannot be selected without it")
    void metricIsOptInAndCannotBeFaked() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        var est = estimator(stitch, false);
        assertFalse(est.isMetricReadoutEnabled());
        assertEquals(NavigationSource.RIGID_MOTION, est.getNavigationSource());
        assertEquals(NavigationUnits.IMAGE_PIXELS, est.getNavigationUnits());
        assertThrows(IllegalStateException.class,
                () -> est.setNavigationSource(NavigationSource.METRIC_LOCAL),
                "a pixel-valued pose must never be relabelled as metres");
        assertThrows(IllegalStateException.class, () -> est.submitHeightSample(0.0, 1.0),
                "a discarded height sample would leave a caller believing height reached the pose");
        assertThrows(IllegalStateException.class, est::getMetricPose);
    }

    @Test
    @DisplayName("with the metric readout on, a frame without a timestamp is refused, not guessed")
    void timestamplessFramesAreRefusedWhenMetricIsOn() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        var est = estimator(stitch, true);
        assertThrows(IllegalStateException.class, () -> est.processFrame(new GrayF32(FRAME, FRAME)));
    }

    @Test
    @DisplayName("METRIC_LOCAL publishes metres and declares them; RIGID_MOTION publishes pixels")
    void thePublishedUnitIsDeclared() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        var est = estimator(stitch, true);
        GrayF32 img = new GrayF32(FRAME, FRAME);
        est.processFrame(img, 0.0);
        est.submitHeightSample(0.0, 0.0);
        for (int k = 1; k <= 10; k++) {
            stitch.motion.advance(translation(20.0));
            est.submitHeightSample(k * DT, 0.0);
            est.processFrame(img, k * DT);
        }
        assertEquals(NavigationUnits.IMAGE_PIXELS, est.getNavigationUnits());
        Pose3D px = est.getCurrentPose();
        assertEquals(200.0, px.x, 1e-9, "ten 20 px steps in first-frame pixels");

        est.setNavigationSource(NavigationSource.METRIC_LOCAL);
        assertEquals(NavigationUnits.METRES, est.getNavigationUnits());
        Pose3D m = est.getCurrentPose();
        assertEquals(20.0, m.x, 1e-9, "the same ten steps at 0.1 m/px");
        assertEquals(0.0, m.z, 0.0, "AGL does not populate z -- vertical ownership is unchanged");
        assertEquals(px.x * 0.1, m.x, 1e-9);
    }

    @Test
    @DisplayName("enabling the metric readout leaves the raw visual pose bit-identical")
    void theRawPathIsUntouched() {
        List<Pose3D> without = new ArrayList<>();
        List<Pose3D> with = new ArrayList<>();
        for (boolean metric : new boolean[]{false, true}) {
            ScriptedStitching stitch = new ScriptedStitching(13);
            var est = estimator(stitch, metric);
            GrayF32 img = new GrayF32(FRAME, FRAME);
            List<Pose3D> out = metric ? with : without;
            if (metric) {
                est.submitHeightSample(0.0, 0.0);
                est.processFrame(img, 0.0);
            } else {
                est.processFrame(img);
            }
            out.add(est.getRigidPose());
            for (int k = 1; k < 120; k++) {
                stitch.motion.advance(translation(3.0 + 0.05 * k));
                if (metric) {
                    est.submitHeightSample(k * DT, 3.0 * k);   // a large, varying height
                    est.processFrame(img, k * DT);
                } else {
                    est.processFrame(img);
                }
                out.add(est.getRigidPose());
            }
        }
        for (int k = 0; k < without.size(); k++) {
            assertEquals(without.get(k).x, with.get(k).x, 0.0, "raw x moved at frame " + k);
            assertEquals(without.get(k).y, with.get(k).y, 0.0, "raw y moved at frame " + k);
            assertEquals(without.get(k).yaw, with.get(k).yaw, 0.0, "raw yaw moved at frame " + k);
        }
    }

    // ================================================================ I: the UE / simulator arm

    @Test
    @DisplayName("I: the simulator arm follows the NADIR CAMERA heading and ignores body yaw entirely")
    void simulatorArmUsesNadirHeadingNotBodyYaw() {
        // The UE nadir camera is world_stabilized_nadir_independent_yaw: its heading is constant
        // while body yaw sweeps a full 360 deg with sd ~103 deg, and the ingest's own C6 check
        // records camera_body_yaw_is_fixed_offset = false. Feeding body yaw here would be wrong by
        // up to 180 deg, and no mounting offset could repair it because none exists.
        ScriptedStitching nadirStitch = new ScriptedStitching(0);
        ScriptedStitching bodyStitch = new ScriptedStitching(0);
        var nadir = headingEstimator(nadirStitch, HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null);
        var body = headingEstimator(bodyStitch, HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null);
        GrayF32 img = new GrayF32(FRAME, FRAME);

        double nadirHeading = 0.0;                 // constant, as in the committed UE datasets
        nadir.submitHeightSample(0.0, 0.0);
        nadir.submitHeadingSample(0.0, nadirHeading);
        nadir.processFrame(img, 0.0);
        body.submitHeightSample(0.0, 0.0);
        body.submitHeadingSample(0.0, 0.0);
        body.processFrame(img, 0.0);

        for (int k = 1; k <= 20; k++) {
            nadirStitch.motion.advance(translation(10.0));
            bodyStitch.motion.advance(translation(10.0));
            nadir.submitHeightSample(k * DT, 0.0);
            body.submitHeightSample(k * DT, 0.0);
            nadir.submitHeadingSample(k * DT, nadirHeading);
            // The airframe swings; a small step so the plausibility gate is not the thing under test.
            body.submitHeadingSample(k * DT, k * 8.0);
            nadir.processFrame(img, k * DT);
            body.processFrame(img, k * DT);
        }

        MetricNavigationState n = nadir.getMetricNavigation();
        assertEquals(20.0, n.eastM(), 1e-9, "20 steps of 10 px at 0.1 m/px, due East at heading 0");
        assertEquals(0.0, n.northM(), 1e-9);
        assertEquals(0.0, n.metricPose().yaw, 1e-12, "the pose reports the nadir camera's heading");
        assertEquals(HeadingSemantics.SIM_NADIR_CAMERA_HEADING,
                n.getReferenceHeading().semantics());
        assertEquals(0.0, n.getReferenceHeading().mountOffsetDeg(), 0.0,
                "no mounting correction is applied on this arm, ever");

        // Identical imagery, a different heading series: the track must differ, which is exactly
        // why substituting body yaw for camera heading is a silent 180-degree hazard.
        MetricNavigationState b = body.getMetricNavigation();
        assertNotEquals(n.northM(), b.northM(),
                "navigation follows the heading it was given -- so the two must never be confused");
        assertTrue(Math.abs(b.northM()) > 1.0, "the swinging series really did bend the track");
    }

    @Test
    @DisplayName("I: the production arm converts body heading through the DECLARED mounting calibration")
    void productionArmAppliesTheDeclaredMount() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        var est = headingEstimator(stitch, HeadingSemantics.FC_AHRS_BODY_COMPASS, 90.0);
        GrayF32 img = new GrayF32(FRAME, FRAME);
        est.submitHeightSample(0.0, 0.0);
        est.submitHeadingSample(0.0, 0.0);            // airframe pointing North
        est.processFrame(img, 0.0);
        for (int k = 1; k <= 10; k++) {
            stitch.motion.advance(translation(10.0));
            est.submitHeightSample(k * DT, 0.0);
            est.submitHeadingSample(k * DT, 0.0);
            est.processFrame(img, k * DT);
        }
        MetricNavigationState m = est.getMetricNavigation();
        assertEquals(90.0, m.navHeadingDeg(), 1e-12,
                "airframe North + a 90 deg mount is a camera pointing East");
        // Image-right at a camera heading of 90 deg is South.
        assertEquals(0.0, m.eastM(), 1e-9);
        assertEquals(-10.0, m.northM(), 1e-9, "10 steps of 10 px at 0.1 m/px, due South");
        assertEquals(0.0, m.getReferenceHeading().sourceHeadingDeg(), 0.0,
                "the raw airframe value is preserved for audit");
        assertEquals(90.0, m.getReferenceHeading().mountOffsetDeg(), 0.0);
    }

    @Test
    @DisplayName("I: a heading channel cannot be enabled without a metric one, or after the first frame")
    void headingWiringIsRefusedOutOfOrder() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        var bare = estimator(stitch, false);
        assertFalse(bare.isHeadingReadoutEnabled());
        assertThrows(IllegalStateException.class, () -> bare.submitHeadingSample(0.0, 10.0));
        assertThrows(IllegalStateException.class,
                () -> bare.enableHeadingReadout(new HeadingReadoutConfig(
                        HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null, 2.0, DT, 180.0)),
                "heading rotates a METRIC increment; without a height channel there is none");

        ScriptedStitching s2 = new ScriptedStitching(0);
        var late = estimator(s2, true);
        late.submitHeightSample(0.0, 0.0);
        late.processFrame(new GrayF32(FRAME, FRAME), 0.0);
        assertThrows(IllegalStateException.class,
                () -> late.enableHeadingReadout(new HeadingReadoutConfig(
                        HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null, 2.0, DT, 180.0)),
                "switching the navigation direction mid-run would splice two trajectories");
    }

    // ================================================================ J: hard loss WITH heading

    @Test
    @DisplayName("J: across a hard loss the translation gap stays UNKNOWN while the heading stays known")
    void hardLossKeepsHeadingAndLosesTranslation() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        var est = headingEstimator(stitch, HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null);
        GrayF32 img = new GrayF32(FRAME, FRAME);

        est.submitHeightSample(0.0, 0.0);
        est.submitHeadingSample(0.0, 0.0);
        est.processFrame(img, 0.0);
        for (int k = 1; k <= 10; k++) {                       // segment 0, heading North
            stitch.motion.advance(translation(10.0));
            est.submitHeightSample(k * DT, 0.0);
            est.submitHeadingSample(k * DT, 0.0);
            est.processFrame(img, k * DT);
        }
        MetricNavigationState m = est.getMetricNavigation();
        assertEquals(0, m.getSegmentIndex());
        assertEquals(10.0, m.eastM(), 1e-9);
        double eastBefore = m.eastM();
        double northBefore = m.northM();

        // The visual pipeline fails, and stays down for three seconds. The heading channel never
        // saw it: samples keep arriving at 10 Hz throughout, and the aircraft turns 45 deg over the
        // outage at a perfectly ordinary 15 deg/s -- well inside the plausibility gate, which is
        // itself part of what this test exercises.
        stitch.forceFailure = true;
        double tLoss = 4.1;
        for (int i = 1; i <= 30; i++) {
            double t = 1.0 + i * 0.1;
            est.submitHeightSample(t, 0.0);
            est.submitHeadingSample(t, 45.0 * i / 30.0);
        }
        assertFalse(est.processFrame(img, tLoss));
        stitch.forceFailure = false;

        assertEquals(1, m.getSegmentIndex(), "a restart opens a new segment");
        assertTrue(m.isUnknownTranslationGapBeforeSegment(),
                "the DISPLACEMENT across the failed interval was never estimated");
        assertTrue(m.isHeadingKnownAcrossGap(),
                "but the ORIENTATION was measured by a sensor that never saw the visual failure");
        assertEquals(45.0, m.navHeadingDeg(), 1e-12, "and it is available immediately after");
        assertEquals(45.0, m.segmentStartNavHeadingDeg(), 1e-12);
        assertEquals(0.0, m.segmentRelativeHeadingDeg(), 1e-12, "the segment starts at its own zero");
        assertEquals(eastBefore, m.eastM(), 0.0, "no displacement is synthesised across the gap");
        assertEquals(northBefore, m.northM(), 0.0);
        assertEquals(0.0, m.segmentRelativePose().x, 0.0, "and the segment pose contains no gap");
        assertEquals(0.0, m.segmentRelativePose().y, 0.0);

        // Post-restart motion is rotated by the AUTHORITATIVE heading, not by the visual yaw, which
        // never saw the turn: the same pixel motion now runs 45 deg off the pre-loss direction.
        for (int k = 1; k <= 10; k++) {
            stitch.motion.advance(translation(10.0));
            est.submitHeightSample(tLoss + k * DT, 0.0);
            est.submitHeadingSample(tLoss + k * DT, 45.0);
            est.processFrame(img, tLoss + k * DT);
        }
        double dEast = m.eastM() - eastBefore;
        double dNorth = m.northM() - northBefore;
        assertEquals(10.0, Math.hypot(dEast, dNorth), 1e-9, "10 steps of 10 px at 0.1 m/px");
        // Pre-loss the camera pointed North, so image-right ran due East (bearing 90 deg). After
        // the 45 deg turn the same image-right runs at bearing 135 deg -- the track rotated by
        // exactly the heading change the external channel measured through the outage.
        assertEquals(135.0, Math.toDegrees(Math.atan2(dEast, dNorth)), 1e-9);
        assertEquals(45.0, 135.0 - 90.0, 0.0, "i.e. a 45 deg rotation of the pre-loss leg");
        assertEquals(0.0, est.getRigidNavigation().headingRadUnwrapped(), 0.0,
                "the VISUAL yaw is still exactly zero -- it never observed the turn at all");
        assertEquals(0.0, m.segmentRelativeHeadingDeg(), 1e-12,
                "the segment has not turned SINCE its restart -- the 45 deg happened before it, "
                + "and the absolute heading is where that information lives");

        // A further 3 deg turn inside the new segment, to show segment-relative heading is a
        // difference of two measurements rather than an integrated quantity.
        for (int k = 11; k <= 20; k++) {
            stitch.motion.advance(translation(10.0));
            est.submitHeightSample(tLoss + k * DT, 0.0);
            est.submitHeadingSample(tLoss + k * DT, 45.0 + 0.3 * (k - 10));
            est.processFrame(img, tLoss + k * DT);
        }
        assertEquals(48.0, m.navHeadingDeg(), 1e-9);
        assertEquals(3.0, m.segmentRelativeHeadingDeg(), 1e-9, "48 - 45, both measured");
    }

    // ================================================================ K: hard loss WITHOUT heading

    @Test
    @DisplayName("K: with no authoritative heading across the loss, none is fabricated")
    void hardLossWithoutHeadingFabricatesNothing() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        var est = headingEstimator(stitch, HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null);
        GrayF32 img = new GrayF32(FRAME, FRAME);

        // No heading sample is ever submitted: the channel has no datum to assume one from.
        est.submitHeightSample(0.0, 0.0);
        est.processFrame(img, 0.0);
        for (int k = 1; k <= 5; k++) {
            stitch.motion.advance(translation(10.0));
            est.submitHeightSample(k * DT, 0.0);
            est.processFrame(img, k * DT);
        }
        MetricNavigationState m = est.getMetricNavigation();
        assertEquals(0.0, m.eastM(), 0.0, "no direction, so no metric displacement at all");
        assertTrue(Double.isNaN(m.navHeadingDeg()));
        assertEquals(HeadingStatus.UNAVAILABLE, m.currentHeadingStatus());
        assertEquals(5, m.getHeadingUnavailableFrameCount());

        stitch.forceFailure = true;
        est.submitHeightSample(6 * DT, 0.0);
        assertFalse(est.processFrame(img, 6 * DT));
        stitch.forceFailure = false;

        assertTrue(m.isUnknownTranslationGapBeforeSegment());
        assertFalse(m.isHeadingKnownAcrossGap(),
                "nothing anchored this segment's orientation, and nothing pretends otherwise");
        assertTrue(Double.isNaN(m.segmentStartNavHeadingDeg()));
        assertTrue(Double.isNaN(m.segmentRelativeHeadingDeg()));
        assertTrue(Double.isNaN(m.metricPose().yaw));
        // And the visual yaw was NOT silently promoted to fill the hole.
        assertNotEquals(0.0, 1.0);
        assertEquals(0.0, est.getRigidNavigation().headingRadUnwrapped(), 0.0);
    }

    // ================================================================ M: visual yaw stays a faithful diagnostic

    @Test
    @DisplayName("M: the raw visual path is bit-identical with the metric readout off, on, and heading-authoritative")
    void rawVisualPathIsUntouchedByEitherChannel() {
        List<Pose3D> plain = new ArrayList<>();
        List<Pose3D> metric = new ArrayList<>();
        List<Pose3D> heading = new ArrayList<>();
        List<Double> plainYaw = new ArrayList<>();
        List<Double> headingVisualYaw = new ArrayList<>();

        ScriptedStitching s1 = new ScriptedStitching(7);
        ScriptedStitching s2 = new ScriptedStitching(7);
        ScriptedStitching s3 = new ScriptedStitching(7);
        var a = estimator(s1, false);
        var b = estimator(s2, true);
        var c = headingEstimator(s3, HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null);
        GrayF32 img = new GrayF32(FRAME, FRAME);

        a.processFrame(img);
        b.submitHeightSample(0.0, 0.0);
        b.processFrame(img, 0.0);
        c.submitHeightSample(0.0, 0.0);
        c.submitHeadingSample(0.0, 137.0);
        c.processFrame(img, 0.0);

        for (int k = 1; k <= 60; k++) {
            Homography2D_F64 inc = new Homography2D_F64();
            translation(9.0).concat(rotation(0.6), inc);      // translate AND rotate every frame
            s1.motion.advance(inc);
            s2.motion.advance(inc);
            s3.motion.advance(inc);
            a.processFrame(img);
            b.submitHeightSample(k * DT, 0.02 * k);
            b.processFrame(img, k * DT);
            c.submitHeightSample(k * DT, 0.02 * k);
            c.submitHeadingSample(k * DT, 137.0 + 0.4 * k);
            c.processFrame(img, k * DT);
            plain.add(a.getRigidPose());
            metric.add(b.getRigidPose());
            heading.add(c.getRigidPose());
            plainYaw.add(a.getRigidNavigation().headingRadUnwrapped());
            headingVisualYaw.add(c.getMetricNavigation().headingRadUnwrapped());
        }

        for (int i = 0; i < plain.size(); i++) {
            assertEquals(plain.get(i).x, metric.get(i).x, 0.0, "raw x, frame " + i);
            assertEquals(plain.get(i).y, metric.get(i).y, 0.0, "raw y, frame " + i);
            assertEquals(plain.get(i).yaw, metric.get(i).yaw, 0.0, "raw yaw, frame " + i);
            assertEquals(plain.get(i).x, heading.get(i).x, 0.0, "raw x under heading, frame " + i);
            assertEquals(plain.get(i).y, heading.get(i).y, 0.0, "raw y under heading, frame " + i);
            assertEquals(plain.get(i).yaw, heading.get(i).yaw, 0.0,
                    "raw yaw under heading, frame " + i);
            // ...and the metric layer's DIAGNOSTIC visual yaw tracks the raw one exactly, which is
            // the property the height-unavailable defect used to break.
            assertEquals(plainYaw.get(i), headingVisualYaw.get(i), 0.0,
                    "diagnostic visual yaw, frame " + i);
        }
        assertNotEquals(0.0, plainYaw.get(plainYaw.size() - 1), "the run really did rotate");
        // The navigation pose meanwhile is on the external datum, not the visual one.
        assertEquals(137.0 + 0.4 * 60, c.getMetricNavigation().navHeadingDeg(), 1e-9);
    }

    @Test
    @DisplayName("M: the visual-vs-external disagreement is exposed and is a wrap-safe raw angle")
    void yawDisagreementIsExposed() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        var est = headingEstimator(stitch, HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null);
        GrayF32 img = new GrayF32(FRAME, FRAME);
        est.submitHeightSample(0.0, 0.0);
        est.submitHeadingSample(0.0, 359.0);
        est.processFrame(img, 0.0);
        // The camera sees -0.5 deg of content rotation (so the camera rotated +0.5 deg); the
        // external channel steps 359 -> 0.2, i.e. +1.2 deg. The difference straddles North.
        stitch.motion.advance(rotation(-0.5));
        est.submitHeightSample(DT, 0.0);
        est.submitHeadingSample(DT, 0.2);
        est.processFrame(img, DT);
        MetricNavigationState m = est.getMetricNavigation();
        assertEquals(0.5 - 1.2, m.getLastYawDisagreementDeg(), 1e-6,
                "computed across the branch cut, not through a 358-degree jump");
    }

    // ================================================================ H: restart at a new altitude

    @Test
    @DisplayName("H: two segments at different altitudes are both in metres, with no zero-motion bridge")
    void restartAtADifferentAltitudeStaysMetricAndFlagsTheGap() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        var est = estimator(stitch, true);
        est.setNavigationSource(NavigationSource.METRIC_LOCAL);
        GrayF32 img = new GrayF32(FRAME, FRAME);
        MetricNavigationState m = est.getMetricNavigation();

        // --- segment A: 10 frames of 20 px at h = 80 m (gsd 0.1) => 20 m -----------------------
        est.submitHeightSample(0.0, 0.0);
        est.processFrame(img, 0.0);
        for (int k = 1; k <= 10; k++) {
            stitch.motion.advance(translation(20.0));
            est.submitHeightSample(k * DT, 0.0);
            est.processFrame(img, k * DT);
        }
        assertEquals(20.0, m.eastM(), 1e-9);
        assertEquals(0, m.getSegmentIndex());
        double beforeGap = m.eastM();

        // --- the failed frame: a wild estimate lands in the accumulation, then the stitch fails ---
        stitch.motion.advance(new Homography2D_F64(1, 0, -900, 0, 1, 1300, 0, 0, 1));
        stitch.forceFailure = true;
        est.submitHeightSample(1.1, 160.0);                 // and the aircraft is now at 240 m AGL
        assertFalse(est.processFrame(img, 1.1));
        stitch.forceFailure = false;

        assertEquals(beforeGap, m.eastM(), 0.0,
                "the rejected estimate must not enter the metric track, and the metric layer must "
                + "not bridge the gap with a fabricated displacement");
        assertEquals(1, m.getSegmentIndex(), "the gap starts a new local segment");
        assertTrue(m.isUnknownTranslationGapBeforeSegment(),
                "the displacement across an unestimated interval is UNKNOWN and must be marked");
        assertEquals(0.0, m.segmentRelativePose().x, 0.0);

        // --- segment B: the same 20 px steps, now at 240 m => 0.3 m/px => 6 m each --------------
        for (int k = 12; k <= 21; k++) {
            stitch.motion.advance(translation(20.0));
            est.submitHeightSample(k * DT, 160.0);
            est.processFrame(img, k * DT);
        }
        assertEquals(0.3, m.getLastGsdMPerPx(), 1e-12, "the conversion scale followed the altitude");
        assertEquals(60.0, m.segmentRelativePose().x, 1e-9,
                "segment B is 10 x 20 px at 0.3 m/px = 60 METRES -- the same physical unit as A");
        assertEquals(beforeGap + 60.0, m.eastM(), 1e-9);

        // The physical-unit consistency claim, stated as the ratio it rests on: identical pixel
        // motion at 3x the height is 3x the ground distance, in both segments' own units.
        assertEquals(3.0, m.segmentRelativePose().x / 20.0, 1e-9);
    }

    @Test
    @DisplayName("H: a mosaic canvas re-origin is NOT a gap and does not start a segment")
    void recenterIsNotAGap() {
        ScriptedStitching stitch = new ScriptedStitching(7);
        var est = estimator(stitch, true);
        GrayF32 img = new GrayF32(FRAME, FRAME);
        est.submitHeightSample(0.0, 0.0);
        est.processFrame(img, 0.0);
        for (int k = 1; k < 60; k++) {
            stitch.motion.advance(translation(10.0));
            est.submitHeightSample(k * DT, 0.0);
            est.processFrame(img, k * DT);
        }
        MetricNavigationState m = est.getMetricNavigation();
        assertEquals(0, m.getSegmentIndex(),
                "a canvas re-origin loses no motion, so it is not an unestimated interval");
        assertEquals(59.0, m.eastM(), 1e-9, "59 steps of 10 px at 0.1 m/px");
    }

    // ================================================================ E: runtime stale/missing

    @Test
    @DisplayName("E: a height dropout degrades the metric pose visibly and recovers")
    void dropoutIsFlaggedAndRecovers() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        var est = estimator(stitch, true);
        GrayF32 img = new GrayF32(FRAME, FRAME);
        MetricNavigationState m = est.getMetricNavigation();

        est.submitHeightSample(0.0, 0.0);
        est.processFrame(img, 0.0);
        for (int k = 1; k <= 5; k++) {
            stitch.motion.advance(translation(10.0));
            est.submitHeightSample(k * DT, 0.0);
            est.processFrame(img, k * DT);
        }
        assertEquals(HeightStatus.FRESH, m.lastHeightStatus());

        // 4 s with no samples at all: HELD, then STALE past tau_stale = 2 s. The pose keeps being
        // produced -- flagged, never refused, and never using a fabricated scale.
        for (int k = 6; k <= 45; k++) {
            stitch.motion.advance(translation(10.0));
            est.processFrame(img, k * DT);
        }
        assertEquals(HeightStatus.STALE, m.lastHeightStatus());
        assertTrue(m.getDegradedFrameCount() > 0, "the degraded frames must be counted");
        assertEquals(0, m.getUnusableFrameCount(), "a stale height is still a usable one");
        assertEquals(45.0, m.eastM(), 1e-9, "45 steps of 10 px at the held 0.1 m/px");

        // Recovery costs exactly one frame, and that is the k-1 convention showing through rather
        // than a lag in the channel: the sample arriving at frame 46 is the REFERENCE for the
        // increment into frame 47, so frame 46's own increment is still converted with the stale
        // reading. The channel itself is fresh immediately.
        est.submitHeightSample(46 * DT, 0.0);
        stitch.motion.advance(translation(10.0));
        est.processFrame(img, 46 * DT);
        assertEquals(HeightStatus.FRESH, est.getHeightChannel().readAt(46 * DT).status(),
                "the channel recovers on the sample itself");
        assertEquals(HeightStatus.STALE, m.lastHeightStatus(),
                "frame 46's increment is still converted with frame 45's stale reading");

        stitch.motion.advance(translation(10.0));
        est.processFrame(img, 47 * DT);
        assertEquals(HeightStatus.FRESH, m.lastHeightStatus(),
                "frame 47 is the first increment converted with the recovered height");
    }

    @Test
    @DisplayName("E: with no samples at all, h_AGL is h0 by the datum's definition -- flagged HELD")
    void noSamplesMeansTheDatumItself() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        var est = estimator(stitch, true);
        GrayF32 img = new GrayF32(FRAME, FRAME);
        est.processFrame(img, 0.0);
        for (int k = 1; k <= 10; k++) {
            stitch.motion.advance(translation(20.0));
            est.processFrame(img, k * DT);
        }
        MetricNavigationState m = est.getMetricNavigation();
        assertEquals(HeightStatus.HELD, m.lastHeightStatus());
        assertEquals(H0, m.heightAglM(), 1e-12);
        assertEquals(20.0, m.eastM(), 1e-9,
                "a channel that has said nothing is a channel reading its own zero datum, which is "
                + "the FIXED arm -- not an unknown scale");
    }

    // ================================================================ reset

    @Test
    @DisplayName("reset() zeroes the metric track but keeps h0 and the samples already received")
    void resetKeepsMissionParameters() {
        ScriptedStitching stitch = new ScriptedStitching(0);
        var est = estimator(stitch, true);
        GrayF32 img = new GrayF32(FRAME, FRAME);
        est.submitHeightSample(0.0, 25.0);
        est.processFrame(img, 0.0);
        stitch.motion.advance(translation(50.0));
        est.processFrame(img, DT);
        assertNotEquals(0.0, est.getMetricNavigation().eastM());

        est.reset();
        assertEquals(0.0, est.getMetricNavigation().eastM(), 0.0);
        assertEquals(0, est.getMetricNavigation().getSegmentIndex());
        assertEquals(H0, est.getHeightChannel().h0M(), 0.0, "h0 is a mission parameter, not state");
        assertEquals(1, est.getHeightChannel().sampleCount(), "delivered samples are history");
    }
}
