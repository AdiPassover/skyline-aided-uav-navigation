package org.boofcv.stitching.metric;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The runtime height channel's causal zero-order hold, against {@code DEC-VO-007} D2/D5/D6 and
 * against {@code evaluation/tools/exp_vo_012/baro.py}'s semantics.
 *
 * <p>Covers the {@code h0} contract (F), the ZOH and sample-rate behaviour (D), and the
 * stale/missing policy (E). The end-to-end causality proof (C) and the numerical agreement with the
 * Python resampler (G) live in {@link MetricNavigationCausalityTest} and
 * {@link MetricReadoutPythonParityTest}; this class pins the pieces they rest on.
 */
class CausalHeightChannelTest {

    private static final double EPS = 1e-12;

    // ------------------------------------------------------------------ F: the h0 contract

    @Test
    @DisplayName("F: h_AGL = h0 + relative, exactly, and h0 is a constructor argument")
    void heightIsH0PlusRelative() {
        CausalHeightChannel ch = new CausalHeightChannel(80.0, 2.0, 0.1);
        ch.submit(1.0, 12.5);
        HeightReading r = ch.readAt(1.0);
        assertEquals(80.0, r.h0M(), EPS);
        assertEquals(12.5, r.relativeHeightM(), EPS);
        assertEquals(92.5, r.heightAglM(), EPS, "h_AGL must be h0 + relative (LIT-VO-006 eq. 4)");
        assertEquals(80.0, ch.h0M(), EPS);
    }

    @Test
    @DisplayName("F: a negative relative height that cancels h0 is UNAVAILABLE, not a zero scale")
    void nonPositiveAglIsUnavailable() {
        CausalHeightChannel ch = new CausalHeightChannel(50.0, 2.0, 0.1);
        ch.submit(0.0, -50.0);
        HeightReading r = ch.readAt(0.0);
        assertEquals(HeightStatus.UNAVAILABLE, r.status(),
                "a camera at or below the imaged surface has no ground sampling distance");
        assertFalse(r.usable());
        assertThrows(IllegalStateException.class, () -> r.groundSamplingDistance(512.0),
                "an unusable reading must refuse to produce a scale rather than return one");
    }

    @Test
    @DisplayName("F: h0 must be supplied, positive and finite -- it is never estimated")
    void h0IsRequiredAndValidated() {
        assertThrows(IllegalArgumentException.class, () -> new CausalHeightChannel(0.0, 2.0, 0.1));
        assertThrows(IllegalArgumentException.class, () -> new CausalHeightChannel(-5.0, 2.0, 0.1));
        assertThrows(IllegalArgumentException.class, () -> new CausalHeightChannel(Double.NaN, 2.0, 0.1));
        assertThrows(IllegalArgumentException.class, () -> new CausalHeightChannel(80.0, 0.0, 0.1));
        assertThrows(IllegalArgumentException.class, () -> new CausalHeightChannel(80.0, 2.0, 0.0));
    }

    // ------------------------------------------------------------------ D: ZOH and sample rate

    @Test
    @DisplayName("D: a frame reads the most recent sample at or before it, and holds between samples")
    void zeroOrderHoldTakesTheMostRecentSample() {
        CausalHeightChannel ch = new CausalHeightChannel(80.0, 2.0, 1.0);
        ch.submit(0.0, 0.0);
        ch.submit(1.0, 5.0);

        assertEquals(80.0, ch.readAt(0.5).heightAglM(), EPS, "held at the 0.0 s sample");
        assertEquals(80.0, ch.readAt(0.999).heightAglM(), EPS);
        assertEquals(85.0, ch.readAt(1.0).heightAglM(), EPS, "a sample AT the frame time is usable");
        assertEquals(85.0, ch.readAt(1.4).heightAglM(), EPS, "held at the 1.0 s sample");
    }

    @Test
    @DisplayName("D: the sample-rate mismatch moves the FRESH/HELD flag but never the height")
    void declaredIntervalChangesFlagsNotValues() {
        CausalHeightChannel fast = new CausalHeightChannel(80.0, 2.0, 1.0);
        CausalHeightChannel slow = new CausalHeightChannel(80.0, 2.0, 0.1);
        for (CausalHeightChannel ch : new CausalHeightChannel[]{fast, slow}) {
            ch.submit(0.0, 0.0);
            ch.submit(1.0, 5.0);
        }
        // age = 0.4 s. Under a declared 1 s interval that is FRESH (0.4 <= 1.5); under 0.1 s it is
        // HELD (0.4 > 0.15). The VALUE is identical, which is the property EXP-VO-012 R6 relies on:
        // the freshness declaration is reporting, not computation.
        assertEquals(HeightStatus.FRESH, fast.readAt(1.4).status());
        assertEquals(HeightStatus.HELD, slow.readAt(1.4).status());
        assertEquals(fast.readAt(1.4).heightAglM(), slow.readAt(1.4).heightAglM(), 0.0);
    }

    @Test
    @DisplayName("D: the FRESH/HELD boundary is exactly 1.5x the declared interval, as in baro.py")
    void freshBoundaryIsOnePointFiveIntervals() {
        CausalHeightChannel ch = new CausalHeightChannel(80.0, 10.0, 0.1);
        ch.submit(0.0, 1.0);
        assertEquals(HeightStatus.FRESH, ch.readAt(0.15).status(), "age == 1.5*interval is FRESH");
        assertEquals(HeightStatus.HELD, ch.readAt(0.1500001).status());
    }

    // ------------------------------------------------------------------ E: stale, missing, recovery

    @Test
    @DisplayName("E: beyond tau_stale the height is STALE -- still held, and flagged")
    void beyondTauStaleIsFlaggedNotSubstituted() {
        CausalHeightChannel ch = new CausalHeightChannel(80.0, 2.0, 0.1);
        ch.submit(0.0, 7.0);

        assertEquals(HeightStatus.HELD, ch.readAt(1.9).status());
        assertEquals(HeightStatus.STALE, ch.readAt(2.5).status(), "age 2.5 s > tau_stale 2 s");
        assertEquals(87.0, ch.readAt(2.5).heightAglM(), EPS,
                "DEC-VO-007 D6: the value is HELD across the gap; tau_stale changes what is "
                + "REPORTED, not what is computed");
        assertTrue(ch.readAt(2.5).degraded());
        assertTrue(ch.readAt(2.5).usable(), "a stale height is still used -- flagged, not refused");
    }

    @Test
    @DisplayName("E: recovery -- a fresh sample after a dropout restores FRESH immediately")
    void recoveryAfterDropout() {
        CausalHeightChannel ch = new CausalHeightChannel(80.0, 2.0, 0.1);
        ch.submit(0.0, 1.0);
        assertEquals(HeightStatus.STALE, ch.readAt(9.0).status());
        ch.submit(9.5, 4.0);
        HeightReading r = ch.readAt(9.5);
        assertEquals(HeightStatus.FRESH, r.status());
        assertEquals(84.0, r.heightAglM(), EPS);
        assertFalse(r.degraded());
    }

    @Test
    @DisplayName("E: before any sample, the takeoff datum is zero BY DEFINITION -- h_AGL = h0, HELD")
    void beforeTheFirstSampleTheDatumIsZero() {
        CausalHeightChannel ch = new CausalHeightChannel(80.0, 2.0, 0.1);
        assertFalse(ch.hasSample());
        HeightReading r = ch.readAt(3.7);
        assertEquals(80.0, r.heightAglM(), EPS, "a takeoff-relative channel reads 0 at its datum");
        assertEquals(0.0, r.relativeHeightM(), EPS);
        assertEquals(HeightStatus.HELD, r.status(),
                "held by definition rather than measured -- and never FRESH, which would claim a "
                + "sample that does not exist");
        assertEquals(0.0, r.ageS(), EPS);
    }

    @Test
    @DisplayName("E: a frame BEFORE the first sample's availability does not borrow that sample")
    void aFrameBeforeTheFirstSampleDoesNotSeeIt() {
        // This is the one documented divergence from baro.py::_resample, which clips its index to 0
        // and therefore uses a sample from the frame's future when latency > 0. The runtime cannot.
        CausalHeightChannel ch = new CausalHeightChannel(80.0, 2.0, 0.1);
        ch.submit(5.0, 30.0);                     // taken at 4.0 s, published at 5.0 s (1 s latency)
        assertEquals(80.0, ch.readAt(2.0).heightAglM(), EPS,
                "a sample published at 5 s must not reach a frame at 2 s");
        assertEquals(110.0, ch.readAt(5.0).heightAglM(), EPS);
    }

    // ------------------------------------------------------------------ stream hygiene

    @Test
    @DisplayName("out-of-order samples are refused rather than silently reordered")
    void outOfOrderSamplesAreRefused() {
        CausalHeightChannel ch = new CausalHeightChannel(80.0, 2.0, 0.1);
        ch.submit(2.0, 1.0);
        assertThrows(IllegalArgumentException.class, () -> ch.submit(1.0, 2.0));
        assertThrows(IllegalArgumentException.class, () -> ch.submit(2.5, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> ch.readAt(Double.NaN));
    }

    @Test
    @DisplayName("readAt is pure -- reading twice cannot change the answer or the channel")
    void readAtIsPure() {
        CausalHeightChannel ch = new CausalHeightChannel(80.0, 2.0, 0.1);
        ch.submit(0.0, 3.0);
        HeightReading a = ch.readAt(1.0);
        HeightReading b = ch.readAt(1.0);
        assertEquals(a, b);
        assertEquals(1, ch.sampleCount());
    }
}
