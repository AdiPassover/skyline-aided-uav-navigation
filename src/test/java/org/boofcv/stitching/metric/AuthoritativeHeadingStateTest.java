package org.boofcv.stitching.metric;

import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The metric state under an <b>authoritative external heading</b>: known-answer integration at fixed
 * and varying heading, the orthogonality of the two external channels, the height-unavailable yaw
 * defect, and segment-relative heading.
 *
 * <p>Covers requirements A, B, C, D, L and N.
 *
 * <p><b>Geometry used throughout</b>, chosen so every expected value is exact: {@code f_working =
 * 800 px} and {@code h_AGL = 80 m}, hence {@code gsd = 0.1 m/px}. The rigid state's increment is
 * {@code dq = (dqX, dqY)} in image pixels with <b>y pointing DOWN</b>, so a displacement of the
 * imaged point "up the image" by 100 px is {@code dqY = -100}, i.e. 10 m along the camera's image-up
 * axis — the axis whose compass azimuth is {@code yaw_nav}.
 */
class AuthoritativeHeadingStateTest {

    private static final double F_WORKING = 800.0;
    private static final double H_AGL = 80.0;          // gsd = 0.1 m/px
    private static final double GSD = H_AGL / F_WORKING;
    private static final double UP_100PX = -100.0;     // 10 m along image-up at this gsd
    private static final double EPS = 1e-10;

    private static MetricNavigationState state() {
        return new MetricNavigationState(F_WORKING, true);
    }

    private static HeightReading h(double t, double aglM) {
        return new HeightReading(t, t, 0.0, aglM - H_AGL, H_AGL, aglM, HeightStatus.FRESH);
    }

    private static HeightReading h(double t) {
        return h(t, H_AGL);
    }

    private static HeightReading unusableHeight(double t) {
        return new HeightReading(t, t, 0.0, -200.0, H_AGL, -120.0, HeightStatus.UNAVAILABLE);
    }

    private static HeadingReading yaw(double t, double yawNavDeg) {
        return new HeadingReading(t, t, 0.0, yawNavDeg, 0.0, yawNavDeg,
                HeadingSemantics.SIM_NADIR_CAMERA_HEADING, HeadingStatus.FRESH);
    }

    private static HeadingReading noYaw(double t) {
        return new HeadingReading(t, t, 0.0, Double.NaN, 0.0, Double.NaN,
                HeadingSemantics.SIM_NADIR_CAMERA_HEADING, HeadingStatus.UNAVAILABLE);
    }

    // ================================================================ A: fixed-heading known answer

    @Test
    @DisplayName("A: at yaw_nav = 0 the camera's image-up axis is North, so up-the-image is North")
    void fixedHeadingNorth() {
        MetricNavigationState m = state();
        m.observeOrigin(h(0.0), yaw(0.0, 0.0));
        m.observe(0.0, UP_100PX, 0.0, h(0.1), yaw(0.1, 0.0));
        assertEquals(0.0, m.eastM(), EPS);
        assertEquals(10.0, m.northM(), EPS, "100 px up-image at 0.1 m/px, pointing North");
        assertEquals(0.0, m.metricPose().yaw, EPS);
    }

    @Test
    @DisplayName("A: at yaw_nav = 90 the same pixel motion is 10 m EAST -- direction comes from heading")
    void fixedHeadingEast() {
        MetricNavigationState m = state();
        m.observeOrigin(h(0.0), yaw(0.0, 90.0));
        m.observe(0.0, UP_100PX, 0.0, h(0.1), yaw(0.1, 90.0));
        assertEquals(10.0, m.eastM(), EPS);
        assertEquals(0.0, m.northM(), EPS);
        assertEquals(90.0, m.metricPose().yaw, EPS, "the pose reports the AUTHORITATIVE heading");
    }

    @Test
    @DisplayName("A: image-right at yaw_nav = 0 is East -- the two camera axes are 90 deg apart, CW")
    void imageRightIsNinetyDegreesClockwise() {
        MetricNavigationState m = state();
        m.observeOrigin(h(0.0), yaw(0.0, 0.0));
        m.observe(100.0, 0.0, 0.0, h(0.1), yaw(0.1, 0.0));
        assertEquals(10.0, m.eastM(), EPS, "image +x at heading North is East");
        assertEquals(0.0, m.northM(), EPS);
    }

    @Test
    @DisplayName("A: the composition is the SAME arithmetic DEC-VO-007 D3 always used, one symbol changed")
    void authoritativeFormMatchesTheVisualFormWhenTheAnglesMatch() {
        // The pre-existing update accumulated T in image axes and published east=+Tx, north=-Ty.
        // Expanding it shows that is already the clockwise-from-North form for a heading measured on
        // the image-up axis -- so feeding the SAME angle through both paths must agree exactly, and
        // replacing theta with psi is a substitution rather than a change of handedness.
        double[] angles = {0.0, 17.0, 90.0, 143.5, 270.0, 359.9};
        for (double a : angles) {
            MetricNavigationState visual = new MetricNavigationState(F_WORKING);
            MetricNavigationState external = state();
            visual.observeOrigin(h(0.0));
            external.observeOrigin(h(0.0), yaw(0.0, a));
            // Drive the visual state's own heading to exactly `a` with a pure rotation, then move.
            visual.observe(0.0, 0.0, Math.toRadians(a), h(0.05));
            visual.observe(37.0, -61.0, 0.0, h(0.1));
            external.observe(37.0, -61.0, 0.0, h(0.1), yaw(0.1, a));
            assertEquals(visual.eastM(), external.eastM(), 1e-9, "east at " + a + " deg");
            assertEquals(visual.northM(), external.northM(), 1e-9, "north at " + a + " deg");
        }
    }

    // ================================================================ B: varying-heading known answer

    @Test
    @DisplayName("B: four 10 m legs at N/E/S/W return to the origin -- and would NOT with visual yaw")
    void varyingHeadingClosesTheSquare() {
        MetricNavigationState m = state();
        m.observeOrigin(h(0.0), yaw(0.0, 0.0));
        // Each increment is rotated by the heading latched at k-1, so the reading handed to
        // observe() is the one the NEXT increment will use. Legs therefore run at 0, 90, 180, 270.
        double[] nextHeading = {90.0, 180.0, 270.0, 0.0};
        double t = 0.0;
        for (double psi : nextHeading) {
            // Identical camera-relative motion on every leg; only the external heading changes.
            m.observe(0.0, UP_100PX, 0.0, h(t += 0.1), yaw(t, psi));
        }
        assertEquals(0.0, m.eastM(), EPS, "the square closes in east");
        assertEquals(0.0, m.northM(), EPS, "the square closes in north");

        // The negative control that makes this test load-bearing: the visual yaw never moved
        // (every dTheta is 0), so substituting it would send all five legs North.
        assertEquals(0.0, m.headingRadUnwrapped(), 0.0, "no visual rotation was ever observed");
        MetricNavigationState visualOnly = new MetricNavigationState(F_WORKING);
        visualOnly.observeOrigin(h(0.0));
        for (int i = 0; i < 4; i++) {
            visualOnly.observe(0.0, UP_100PX, 0.0, h(0.1 * (i + 1)));
        }
        assertEquals(40.0, visualOnly.northM(), EPS,
                "with visual yaw the same input runs 40 m due North -- this is what the "
                + "authoritative heading replaces");
        assertNotEquals(visualOnly.northM(), m.northM(),
                "the two must not agree, or the substitution did nothing");
    }

    @Test
    @DisplayName("B: a heading that turns while the camera sees no rotation still curves the track")
    void headingTurnsWithoutAnyVisualRotation() {
        MetricNavigationState m = state();
        m.observeOrigin(h(0.0), yaw(0.0, 0.0));
        double t = 0.0;
        for (int i = 0; i < 90; i++) {
            m.observe(0.0, UP_100PX, 0.0, h(t += 0.1), yaw(t, i + 1.0));
        }
        // 90 one-degree steps of a 10 m leg approximates a quarter circle of radius 10/tan(0.5 deg).
        assertTrue(m.eastM() > 100.0, "the track must bend east; got " + m.eastM());
        assertTrue(m.northM() > 100.0, "and still make northing; got " + m.northM());
        assertEquals(0.0, m.headingRadUnwrapped(), 0.0,
                "and the visual yaw is untouched throughout -- it saw no rotation");
    }

    // ================================================================ C: orthogonality

    @Test
    @DisplayName("C: height changes SCALE and not direction; heading changes DIRECTION and not scale")
    void heightAndHeadingAreOrthogonal() {
        // Same pixel motion, double the height.
        MetricNavigationState low = state();
        MetricNavigationState high = state();
        low.observeOrigin(h(0.0, 80.0), yaw(0.0, 37.0));
        high.observeOrigin(h(0.0, 160.0), yaw(0.0, 37.0));
        low.observe(30.0, -40.0, 0.0, h(0.1, 80.0), yaw(0.1, 37.0));
        high.observe(30.0, -40.0, 0.0, h(0.1, 160.0), yaw(0.1, 37.0));
        assertEquals(2.0, high.eastM() / low.eastM(), 1e-12, "twice the height, twice the distance");
        assertEquals(2.0, high.northM() / low.northM(), 1e-12);
        assertEquals(Math.atan2(low.eastM(), low.northM()),
                     Math.atan2(high.eastM(), high.northM()), 1e-12,
                     "and the bearing is identical -- height carries no direction");

        // Same height, two headings 40 deg apart.
        MetricNavigationState a = state();
        MetricNavigationState b = state();
        a.observeOrigin(h(0.0), yaw(0.0, 10.0));
        b.observeOrigin(h(0.0), yaw(0.0, 50.0));
        a.observe(30.0, -40.0, 0.0, h(0.1), yaw(0.1, 10.0));
        b.observe(30.0, -40.0, 0.0, h(0.1), yaw(0.1, 50.0));
        assertEquals(Math.hypot(a.eastM(), a.northM()), Math.hypot(b.eastM(), b.northM()), 1e-12,
                "heading carries no scale -- the distance travelled is identical");
        double bearingA = Math.toDegrees(Math.atan2(a.eastM(), a.northM()));
        double bearingB = Math.toDegrees(Math.atan2(b.eastM(), b.northM()));
        assertEquals(40.0, CausalHeadingChannel.wrap180(bearingB - bearingA), 1e-10,
                "and the track rotates by exactly the heading difference");
    }

    @Test
    @DisplayName("C: the gsd used is still the REFERENCE frame's, and so is the heading")
    void bothChannelsUseTheReferenceFrame() {
        MetricNavigationState m = state();
        m.observeOrigin(h(0.0, 80.0), yaw(0.0, 0.0));
        // Frame 1's own reading is 160 m / 90 deg, but the increment INTO frame 1 must use frame 0's.
        m.observe(0.0, UP_100PX, 0.0, h(0.1, 160.0), yaw(0.1, 90.0));
        assertEquals(0.0, m.eastM(), EPS, "frame 0's heading (North) governed this increment");
        assertEquals(10.0, m.northM(), EPS, "and frame 0's height (0.1 m/px) scaled it");
        assertEquals(90.0, m.navHeadingDeg(), EPS, "while the POSE reports frame 1's own heading");
        // The next increment then uses frame 1's readings.
        m.observe(0.0, UP_100PX, 0.0, h(0.2, 160.0), yaw(0.2, 90.0));
        assertEquals(20.0, m.eastM(), EPS, "100 px at 0.2 m/px, pointing East");
        assertEquals(10.0, m.northM(), EPS);
    }

    // ================================================================ D: the height-unavailable yaw defect

    @Test
    @DisplayName("D: an UNAVAILABLE height stops the TRANSLATION and must not stop the visual yaw")
    void visualYawSurvivesAnUnavailableHeight() {
        // The 2026-09-06 defect: `visualHeadingRad += dTheta` sat inside the height-usable branch,
        // so an unusable height silently froze the metric layer's visual yaw while the raw rigid
        // state's kept advancing -- a divergence three documented contracts denied.
        MetricNavigationState broken = state();
        MetricNavigationState control = state();
        broken.observeOrigin(unusableHeight(0.0), yaw(0.0, 0.0));
        control.observeOrigin(h(0.0), yaw(0.0, 0.0));

        double dTheta = Math.toRadians(3.0);
        double t = 0.0;
        for (int i = 0; i < 10; i++) {
            t += 0.1;
            broken.observe(10.0, -20.0, dTheta, unusableHeight(t), yaw(t, 0.0));
            control.observe(10.0, -20.0, dTheta, h(t), yaw(t, 0.0));
        }
        assertEquals(Math.toRadians(30.0), broken.headingRadUnwrapped(), 1e-12,
                "the visual yaw integrates every accepted rotation, whatever the height is doing");
        assertEquals(control.headingRadUnwrapped(), broken.headingRadUnwrapped(), 0.0,
                "and it is bit-identical to the run whose height was fine");

        // The translation, correctly, did not move.
        assertEquals(0.0, broken.eastM(), 0.0);
        assertEquals(0.0, broken.northM(), 0.0);
        assertEquals(10, broken.getUnusableFrameCount());
        assertFalse(broken.lastFrameUsable());
        // ...while the control did.
        assertNotEquals(0.0, control.eastM());
    }

    @Test
    @DisplayName("D: the same holds with no heading channel at all -- height must never gate yaw")
    void visualYawSurvivesAnUnavailableHeightWithoutAHeadingChannel() {
        MetricNavigationState m = new MetricNavigationState(F_WORKING);
        m.observeOrigin(unusableHeight(0.0));
        m.observe(10.0, -20.0, Math.toRadians(7.0), unusableHeight(0.1));
        m.observe(10.0, -20.0, Math.toRadians(7.0), unusableHeight(0.2));
        assertEquals(Math.toRadians(14.0), m.headingRadUnwrapped(), 1e-12);
        assertEquals(14.0, m.metricPose().yaw, 1e-10,
                "and without a heading channel the pose still reports the visual yaw");
    }

    // ================================================================ heading unavailable

    @Test
    @DisplayName("no authoritative heading means no metric increment, and none is fabricated")
    void unavailableHeadingProducesNoTranslationAndNoInventedYaw() {
        MetricNavigationState m = state();
        m.observeOrigin(h(0.0), noYaw(0.0));
        m.observe(0.0, UP_100PX, Math.toRadians(5.0), h(0.1), noYaw(0.1));
        assertEquals(0.0, m.eastM(), 0.0, "no direction, so no metric displacement");
        assertEquals(0.0, m.northM(), 0.0);
        assertTrue(Double.isNaN(m.navHeadingDeg()), "and no heading is invented");
        assertTrue(Double.isNaN(m.metricPose().yaw));
        assertEquals(1, m.getHeadingUnavailableFrameCount());
        assertEquals(1, m.getUnusableFrameCount());
        assertEquals(Math.toRadians(5.0), m.headingRadUnwrapped(), 1e-12,
                "the visual yaw still advances -- it is a diagnostic, not a fallback");
        assertEquals(HeadingStatus.UNAVAILABLE, m.currentHeadingStatus());
    }

    @Test
    @DisplayName("once a heading has been measured, a STALE one is held and used, and flagged")
    void staleHeadingIsHeldAndFlagged() {
        MetricNavigationState m = state();
        HeadingReading stale = new HeadingReading(0.1, 0.0, 5.0, 45.0, 0.0, 45.0,
                HeadingSemantics.SIM_NADIR_CAMERA_HEADING, HeadingStatus.STALE);
        m.observeOrigin(h(0.0), stale);
        m.observe(0.0, UP_100PX, 0.0, h(0.1), stale);
        assertTrue(m.lastFrameUsable(), "STALE is still used");
        assertEquals(1, m.getDegradedFrameCount(), "and flagged, exactly once, on use");
        assertEquals(HeadingStatus.STALE, m.lastHeadingStatus());
        assertEquals(45.0, m.navHeadingDeg(), EPS);
    }

    // ================================================================ L: segment-relative heading

    @Test
    @DisplayName("L: segment-relative heading starts at zero and then equals the heading change since restart")
    void segmentRelativeHeading() {
        MetricNavigationState m = state();
        m.observeOrigin(h(0.0), yaw(0.0, 10.0));
        m.observe(0.0, UP_100PX, 0.0, h(0.1), yaw(0.1, 20.0));
        assertEquals(0, m.getSegmentIndex());
        assertEquals(10.0, m.segmentStartNavHeadingDeg(), EPS);
        assertEquals(10.0, m.segmentRelativeHeadingDeg(), EPS, "20 - 10");

        // Hard loss: the estimator restarts, and the heading channel never saw the visual failure.
        m.beginNewSegment(true);
        m.observeOrigin(h(1.0), yaw(1.0, 100.0));
        assertEquals(1, m.getSegmentIndex());
        assertTrue(m.isUnknownTranslationGapBeforeSegment());
        assertTrue(m.isHeadingKnownAcrossGap());
        assertEquals(100.0, m.segmentStartNavHeadingDeg(), EPS, "the new datum is the restart frame's");
        assertEquals(0.0, m.segmentRelativeHeadingDeg(), EPS, "and the segment starts at zero");
        assertEquals(0.0, m.segmentRelativePose().x, 0.0);
        assertEquals(0.0, m.segmentRelativePose().y, 0.0);

        m.observe(0.0, UP_100PX, 0.0, h(1.1), yaw(1.1, 130.0));
        assertEquals(30.0, m.segmentRelativeHeadingDeg(), EPS, "130 - 100");
        assertEquals(30.0, m.segmentRelativePose().yaw, EPS);
        assertEquals(130.0, m.metricPose().yaw, EPS, "the ABSOLUTE heading is what navigation uses");
    }

    @Test
    @DisplayName("L: segment-relative heading wraps across North rather than reporting ~360")
    void segmentRelativeHeadingWraps() {
        MetricNavigationState m = state();
        m.observeOrigin(h(0.0), yaw(0.0, 359.0));
        m.observe(0.0, UP_100PX, 0.0, h(0.1), yaw(0.1, 1.0));
        assertEquals(2.0, m.segmentRelativeHeadingDeg(), EPS,
                "359 -> 1 is a two-degree turn, not a 358-degree one");
    }

    @Test
    @DisplayName("L: with no heading across the gap, no segment orientation is claimed")
    void segmentHeadingUnknownWhenTheChannelIsDown() {
        MetricNavigationState m = state();
        m.observeOrigin(h(0.0), noYaw(0.0));
        m.beginNewSegment(true);
        m.observeOrigin(h(1.0), noYaw(1.0));
        assertTrue(m.isUnknownTranslationGapBeforeSegment());
        assertFalse(m.isHeadingKnownAcrossGap(), "nothing anchored this segment's orientation");
        assertTrue(Double.isNaN(m.segmentStartNavHeadingDeg()));
        assertTrue(Double.isNaN(m.segmentRelativeHeadingDeg()));
    }

    // ================================================================ the consistency signal

    @Test
    @DisplayName("the visual-vs-external disagreement is a wrap-safe raw angle, not a confidence")
    void yawDisagreementIsReported() {
        MetricNavigationState m = state();
        m.observeOrigin(h(0.0), yaw(0.0, 359.0));
        // The camera reports +3 deg of rotation while the external channel moves 359 -> 1, i.e. +2.
        m.observe(0.0, UP_100PX, Math.toRadians(3.0), h(0.1), yaw(0.1, 1.0));
        assertEquals(1.0, m.getLastYawDisagreementDeg(), 1e-10,
                "3 - 2, computed across the branch cut");
        // A frame with no external heading has no disagreement to report -- and reports NaN rather
        // than a zero that would read as perfect agreement.
        m.observe(0.0, UP_100PX, Math.toRadians(3.0), h(0.2), noYaw(0.2));
        assertTrue(Double.isNaN(m.getLastYawDisagreementDeg()));
    }

    // ================================================================ N: units, and arm consistency

    @Test
    @DisplayName("N: x/y are metres, yaw is degrees CW from North in [0,360), height is metres, z is 0")
    void unitsAreWhatTheyClaim() {
        MetricNavigationState m = state();
        m.observeOrigin(h(0.0, 80.0), yaw(0.0, 350.0));
        m.observe(0.0, -1600.0, 0.0, h(0.1, 80.0), yaw(0.1, 370.0 - 360.0));
        Pose3D p = m.metricPose();
        assertEquals(160.0 * Math.sin(Math.toRadians(350.0)), p.x, 1e-9, "east, metres");
        assertEquals(160.0 * Math.cos(Math.toRadians(350.0)), p.y, 1e-9, "north, metres");
        assertEquals(0.0, p.z, 0.0, "z is never estimated");
        assertEquals(10.0, p.yaw, EPS, "degrees, [0,360), clockwise from North");
        assertEquals(80.0, m.heightAglM(), 0.0, "metres AGL, separate from z");
        assertEquals(GSD, m.getLastGsdMPerPx(), 1e-15, "metres per pixel");
    }

    @Test
    @DisplayName("the two arms cannot be mixed by accident")
    void armMismatchIsRefused() {
        MetricNavigationState authoritative = state();
        assertThrows(IllegalArgumentException.class, () -> authoritative.observeOrigin(h(0.0)),
                "an authoritative state needs a heading every frame");
        MetricNavigationState visual = new MetricNavigationState(F_WORKING);
        assertThrows(IllegalArgumentException.class,
                () -> visual.observeOrigin(h(0.0), yaw(0.0, 30.0)),
                "a visual state must not silently ignore a heading it was handed");
        assertTrue(authoritative.isHeadingAuthoritative());
        assertFalse(visual.isHeadingAuthoritative());
    }
}
