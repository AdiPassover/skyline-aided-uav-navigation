package org.boofcv.stitching.metric;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The heading channel's own contract: causality, zero-order hold, wrap safety, the mounting
 * calibration, and the rate-plausibility gate.
 *
 * <p>Covers requirements E (causality), F (ZOH), G (wrap) and H (mount offset and its refusal).
 */
class CausalHeadingChannelTest {

    private static final double TAU = 2.0;
    private static final double INTERVAL = 0.01;      // 100 Hz, the measured real-channel rate
    private static final double MAX_RATE = 180.0;

    private static CausalHeadingChannel body(double mountDeg) {
        return new CausalHeadingChannel(HeadingSemantics.FC_AHRS_BODY_COMPASS, mountDeg,
                TAU, INTERVAL, MAX_RATE);
    }

    private static CausalHeadingChannel camera() {
        return new CausalHeadingChannel(HeadingSemantics.SIM_NADIR_CAMERA_HEADING, 0.0,
                TAU, INTERVAL, MAX_RATE);
    }

    // ================================================================ no datum before the first sample

    @Test
    @DisplayName("before the first sample there is NO heading -- an absolute azimuth has no datum to assume")
    void unavailableBeforeAnySample() {
        CausalHeadingChannel c = body(90.0);
        HeadingReading r = c.readAt(0.0);
        assertEquals(HeadingStatus.UNAVAILABLE, r.status());
        assertFalse(r.usable());
        assertTrue(Double.isNaN(r.yawNavDeg()));
        assertThrows(IllegalStateException.class, r::yawNavRad,
                "asking an unavailable reading for a number must refuse, not return 0");
        // This is the ONE place the heading channel deliberately differs from the height channel:
        // a takeoff-relative height reads zero at its own datum by construction, so h0 alone still
        // yields an AGL. Heading has no such datum.
        assertFalse(c.hasSample());
    }

    // ================================================================ H: the mounting calibration

    @Test
    @DisplayName("H: body heading + delta_mount is the navigation camera's heading")
    void mountOffsetIsApplied() {
        CausalHeadingChannel c = body(90.0);
        c.submit(0.0, 30.0);
        HeadingReading r = c.readAt(0.0);
        assertEquals(30.0, r.sourceHeadingDeg(), 0.0, "the source value is preserved verbatim");
        assertEquals(90.0, r.mountOffsetDeg(), 0.0);
        assertEquals(120.0, r.yawNavDeg(), 1e-12);
        assertEquals(HeadingSemantics.FC_AHRS_BODY_COMPASS, r.semantics());
    }

    @Test
    @DisplayName("H: the mount offset wraps across North rather than running past 360")
    void mountOffsetWrapsAcrossNorth() {
        CausalHeadingChannel c = body(90.0);
        c.submit(0.0, 300.0);
        assertEquals(30.0, c.readAt(0.0).yawNavDeg(), 1e-12, "300 + 90 = 390 -> 30, not 390");
    }

    @Test
    @DisplayName("H: a camera-heading arm refuses a mounting correction -- it is already in that frame")
    void cameraArmRefusesAMountOffset() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new CausalHeadingChannel(HeadingSemantics.SIM_NADIR_CAMERA_HEADING, 90.0,
                        TAU, INTERVAL, MAX_RATE));
        assertTrue(e.getMessage().contains("already carries the navigation camera's heading"));
    }

    @Test
    @DisplayName("H: the body arm REFUSES a config with no mounting calibration -- zero is not a default")
    void bodyArmRefusesAMissingCalibration() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new HeadingReadoutConfig(HeadingSemantics.FC_AHRS_BODY_COMPASS, null,
                        TAU, INTERVAL, MAX_RATE));
        // Zero looks entirely plausible and would silently reintroduce EXP-VO-013 R6's ~66 deg
        // error, so the absence must be refused rather than filled in.
        assertTrue(e.getMessage().contains("delta_mount_deg is REQUIRED"));
        assertTrue(e.getMessage().contains("AIRFRAME"));
    }

    @Test
    @DisplayName("H: the camera arm needs no calibration and defaults to exactly zero")
    void cameraArmNeedsNoCalibration() {
        HeadingReadoutConfig cfg = new HeadingReadoutConfig(
                HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null, TAU, INTERVAL, MAX_RATE);
        assertEquals(0.0, cfg.mountOffsetDeg(), 0.0);
        CausalHeadingChannel c = cfg.newHeadingChannel();
        c.submit(0.0, 217.5);
        assertEquals(217.5, c.readAt(0.0).yawNavDeg(), 1e-12, "used directly, unmodified");
    }

    @Test
    @DisplayName("H: the two semantics parse from config and refuse anything else")
    void semanticsAreDeclaredNotGuessed() {
        assertEquals(HeadingSemantics.FC_AHRS_BODY_COMPASS,
                HeadingSemantics.parse("fc_ahrs_body_compass"));
        assertEquals(HeadingSemantics.SIM_NADIR_CAMERA_HEADING,
                HeadingSemantics.parse("sim_nadir_camera_heading"));
        assertThrows(IllegalArgumentException.class, () -> HeadingSemantics.parse("body_yaw"));
        assertThrows(IllegalArgumentException.class, () -> HeadingSemantics.parse(null));
    }

    // ================================================================ F: zero-order hold, freshness

    @Test
    @DisplayName("F: ZOH -- a frame uses the most recent sample at or before it, never a later one")
    void zeroOrderHold() {
        // Samples are submitted as they become available, which is what the replay path does; the
        // channel is never handed a sample from the future of the frame being read.
        // Heading steps sized to a plausible turn rate (8 deg/s): the rate gate is live here too,
        // and a synthetic 200 deg/s step would be rejected exactly as a real glitch would.
        CausalHeadingChannel c = camera();
        c.submit(0.00, 10.0);
        assertEquals(10.0, c.readAt(0.00).yawNavDeg(), 1e-12);
        assertEquals(10.0, c.readAt(0.049).yawNavDeg(), 1e-12, "not yet 10.4 -- 0.05 has not arrived");
        assertTrue(c.submit(0.05, 10.4));
        assertEquals(10.4, c.readAt(0.05).yawNavDeg(), 1e-12, "at the boundary the sample is in");
        assertEquals(10.4, c.readAt(0.099).yawNavDeg(), 1e-12);
        assertTrue(c.submit(0.10, 10.8));
        assertEquals(10.8, c.readAt(0.10).yawNavDeg(), 1e-12);
        assertEquals(10.8, c.readAt(5.00).yawNavDeg(), 1e-12, "held, never extrapolated");
    }

    @Test
    @DisplayName("F: reading BACKWARDS in time is refused rather than answered")
    void backwardsReadIsRefused() {
        // The same guard CausalHeightChannel carries. A frame time earlier than the newest submitted
        // sample means the caller has gone backwards, and answering would hand it a value from its
        // own future. Unreachable through the replay path, which drains strictly forward.
        CausalHeadingChannel c = camera();
        c.submit(1.0, 42.0);
        assertEquals(HeadingStatus.UNAVAILABLE, c.readAt(0.5).status());
        assertEquals(42.0, c.readAt(1.0).yawNavDeg(), 1e-12);
    }

    @Test
    @DisplayName("F: FRESH -> HELD -> STALE follow the two declared thresholds and nothing else")
    void freshnessBoundaries() {
        CausalHeadingChannel c = camera();
        c.submit(0.0, 45.0);
        assertEquals(HeadingStatus.FRESH, c.readAt(0.015).status(), "age == 1.5 * interval");
        assertEquals(HeadingStatus.HELD, c.readAt(0.0151).status());
        assertEquals(HeadingStatus.HELD, c.readAt(TAU).status(), "age == tau_stale");
        assertEquals(HeadingStatus.STALE, c.readAt(TAU + 1e-9).status());
        // The value is identical in all three: tau_stale changes what is REPORTED, never what is
        // COMPUTED -- exactly the height channel's contract.
        assertEquals(45.0, c.readAt(0.015).yawNavDeg(), 0.0);
        assertEquals(45.0, c.readAt(TAU + 10.0).yawNavDeg(), 0.0);
        assertTrue(c.readAt(TAU + 10.0).usable(), "STALE is still used, and flagged");
        assertTrue(c.readAt(TAU + 10.0).degraded());
    }

    @Test
    @DisplayName("F: a 100 Hz channel against 10 Hz frames is FRESH at every frame")
    void realWorldRateIsAlwaysFresh() {
        // The measured shape of datasets/<id>/attitude.csv: 101.3 Hz against 10 Hz imagery, worst
        // observed sample age at a camera frame 0.0143 s.
        CausalHeadingChannel c = camera();
        for (int i = 0; i <= 1000; i++) {
            c.submit(i * 0.00987, 100.0 + 0.01 * i);
            if (i % 10 == 0) {
                assertEquals(HeadingStatus.FRESH, c.readAt(i * 0.00987 + 0.0143).status(),
                        "worst measured age must still classify FRESH");
            }
        }
    }

    // ================================================================ E: causality

    @Test
    @DisplayName("E: a sample submitted later cannot change an already-produced reading")
    void futureCannotChangeThePast() {
        CausalHeadingChannel c = camera();
        c.submit(0.0, 10.0);
        HeadingReading before = c.readAt(0.5);
        assertEquals(10.0, before.yawNavDeg(), 1e-12);
        // A sample that becomes available at 1.0 must be invisible to every frame before it. The
        // channel holds no array to index into, so this is structural rather than a policy: the only
        // way to observe the later value is to ask for a later frame.
        c.submit(1.0, 300.0);
        assertEquals(300.0, c.readAt(1.0).yawNavDeg(), 1e-12, "and the later frame does see it");
        assertEquals(10.0, before.yawNavDeg(), 1e-12,
                "the reading already produced is a value object and cannot be revised");
    }

    @Test
    @DisplayName("E: readAt is pure -- calling it twice returns the same answer and mutates nothing")
    void readAtIsPure() {
        CausalHeadingChannel c = camera();
        c.submit(0.0, 77.0);
        long n = c.sampleCount();
        assertEquals(c.readAt(0.4), c.readAt(0.4));
        assertEquals(n, c.sampleCount());
    }

    @Test
    @DisplayName("out-of-order samples are refused rather than silently reordered")
    void outOfOrderRefused() {
        CausalHeadingChannel c = camera();
        c.submit(1.0, 10.0);
        assertThrows(IllegalArgumentException.class, () -> c.submit(0.5, 20.0));
        assertThrows(IllegalArgumentException.class, () -> c.submit(1.5, Double.NaN));
    }

    // ================================================================ G: wrap safety

    @Test
    @DisplayName("G: 359 -> 0 -> 1 is a two-degree turn, not a 358-degree one")
    void wrapAcrossNorthIsNotAJump() {
        CausalHeadingChannel c = camera();
        c.submit(0.00, 359.0);
        double a = c.readAt(0.00).yawNavDeg();
        c.submit(0.01, 0.0);
        double b = c.readAt(0.01).yawNavDeg();
        c.submit(0.02, 1.0);
        double d = c.readAt(0.02).yawNavDeg();
        assertEquals(1.0, CausalHeadingChannel.wrap180(b - a), 1e-12);
        assertEquals(1.0, CausalHeadingChannel.wrap180(d - b), 1e-12);
        assertEquals(2.0, CausalHeadingChannel.wrap180(d - a), 1e-12,
                "the whole manoeuvre is two degrees");
        // And crucially the rate gate must not fire on it: a naive difference would read 358 deg
        // in 0.01 s = 35800 deg/s and reject a perfectly ordinary sample.
        assertEquals(3, c.sampleCount(), "no sample may be rejected by crossing North");
    }

    @Test
    @DisplayName("G: wrap360 and wrap180 are exact on the branch cuts")
    void wrapHelpersAreExact() {
        assertEquals(0.0, CausalHeadingChannel.wrap360(360.0), 0.0);
        assertEquals(0.0, CausalHeadingChannel.wrap360(720.0), 0.0);
        assertEquals(359.0, CausalHeadingChannel.wrap360(-1.0), 1e-12);
        assertEquals(180.0, CausalHeadingChannel.wrap180(180.0), 0.0);
        assertEquals(180.0, CausalHeadingChannel.wrap180(-180.0), 0.0, "the cut is (-180, 180]");
        assertEquals(-179.0, CausalHeadingChannel.wrap180(181.0), 1e-12);
    }

    // ================================================================ the rate-plausibility gate

    @Test
    @DisplayName("a physically impossible sample is rejected, the previous heading is held, and it is counted")
    void rateGateRejectsAGlitchAndHoldsTheLastValid() {
        // The shape measured on the committed real channel: 21 of 118,940 samples on amtown01-d
        // imply 274-5141 deg/s against a legitimate maximum of 26.7 deg/s.
        CausalHeadingChannel c = camera();
        c.submit(0.00, 100.0);
        c.submit(0.01, 100.2);
        assertTrue(c.submit(0.02, 100.4), "an ordinary 20 deg/s sample is accepted");
        assertFalse(c.submit(0.03, 160.0), "6000 deg/s is rejected");
        assertEquals(1, c.rejectedSampleCount());
        assertEquals(3, c.sampleCount(), "the rejected sample did not enter the latch");
        assertEquals(100.4, c.readAt(0.03).yawNavDeg(), 1e-12,
                "the previous valid heading is held -- the ordinary staleness policy takes over");
        assertTrue(c.submit(0.04, 100.6), "and the channel recovers on the next good sample");
        assertEquals(100.6, c.readAt(0.04).yawNavDeg(), 1e-12);
    }

    @Test
    @DisplayName("the gate is a gate, not a filter -- an accepted sample is never modified")
    void gateNeverAltersAnAcceptedValue() {
        CausalHeadingChannel c = camera();
        c.submit(0.0, 123.456789);
        assertEquals(123.456789, c.readAt(0.0).yawNavDeg(), 0.0);
        assertEquals(123.456789, c.readAt(0.0).sourceHeadingDeg(), 0.0);
    }

    @Test
    @DisplayName("a legitimate fast turn well under the ceiling is not rejected")
    void fastButPlausibleTurnSurvives() {
        CausalHeadingChannel c = camera();
        c.submit(0.00, 0.0);
        // 26.7 deg/s was the fastest genuine rate measured on either real flight.
        for (int i = 1; i <= 50; i++) {
            assertTrue(c.submit(i * 0.01, i * 0.267), "26.7 deg/s must never be rejected");
        }
        assertEquals(0, c.rejectedSampleCount());
    }

    @Test
    @DisplayName("constructor arguments are validated rather than defaulted")
    void constructorValidation() {
        assertThrows(IllegalArgumentException.class, () -> body(Double.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> new CausalHeadingChannel(null, 0.0, TAU, INTERVAL, MAX_RATE));
        assertThrows(IllegalArgumentException.class,
                () -> new CausalHeadingChannel(HeadingSemantics.SIM_NADIR_CAMERA_HEADING, 0.0,
                        0.0, INTERVAL, MAX_RATE));
        assertThrows(IllegalArgumentException.class,
                () -> new CausalHeadingChannel(HeadingSemantics.SIM_NADIR_CAMERA_HEADING, 0.0,
                        TAU, -1.0, MAX_RATE));
        assertThrows(IllegalArgumentException.class,
                () -> new CausalHeadingChannel(HeadingSemantics.SIM_NADIR_CAMERA_HEADING, 0.0,
                        TAU, INTERVAL, 0.0));
    }

    @Test
    @DisplayName("nothing about the channel reaches for the visual yaw")
    void noVisualFallbackExists() {
        // D4's heading analogue, enforced structurally: there is no constructor argument, field or
        // method through which a visually integrated yaw could enter this class, so it cannot be
        // promoted to authoritative by accident.
        for (var f : CausalHeadingChannel.class.getDeclaredFields()) {
            assertFalse(f.getName().toLowerCase().contains("visual"),
                    "CausalHeadingChannel." + f.getName() + " must not exist");
        }
        CausalHeadingChannel c = camera();
        assertNotEquals(null, c.semantics());
    }
}
