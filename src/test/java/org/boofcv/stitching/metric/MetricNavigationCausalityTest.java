package org.boofcv.stitching.metric;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * <b>C: no already-produced metric pose can depend on a height sample from its own future.</b>
 *
 * <p>The offline resampler proves this by construction of its index arithmetic
 * ({@code baro.py::_resample}, gate G5). The runtime proves it more strongly, and this test is where
 * that is asserted: the channel is fed a stream, so a sample that has not been submitted does not
 * exist yet. Corrupting the future is therefore not merely ineffective — it is unreachable.
 *
 * <p>The second half of each test matters as much as the first: the corruption must actually change
 * the <i>later</i> poses, or the test would be passing because the corruption did nothing.
 */
class MetricNavigationCausalityTest {

    private static final double F_WORKING = 500.0;
    private static final double H0 = 60.0;
    private static final double DT = 0.1;
    private static final int FRAMES = 200;

    /** A fixed, arbitrary rigid increment sequence. Nothing about it is fitted or special. */
    private static double[] dq(int k) {
        return new double[]{4.0 + 0.3 * Math.sin(k * 0.21), -2.0 + 0.5 * Math.cos(k * 0.13)};
    }

    private static double dtheta(int k) {
        return 0.004 * Math.sin(k * 0.05);
    }

    /**
     * Integrates the whole sequence, submitting height samples strictly as they become available.
     *
     * @param relative one relative-height value per frame; the sample for frame {@code k} is
     *                 submitted at {@code k*DT}, i.e. immediately before that frame is processed
     */
    private static List<double[]> run(double[] relative) {
        CausalHeightChannel ch = new CausalHeightChannel(H0, 2.0, DT);
        MetricNavigationState m = new MetricNavigationState(F_WORKING);
        List<double[]> poses = new ArrayList<>();
        for (int k = 0; k < FRAMES; k++) {
            double t = k * DT;
            ch.submit(t, relative[k]);
            HeightReading r = ch.readAt(t);
            if (k == 0) {
                m.observeOrigin(r);
            } else {
                double[] d = dq(k);
                m.observe(d[0], d[1], dtheta(k), r);
            }
            poses.add(new double[]{m.eastM(), m.northM(), m.metricPose().yaw});
        }
        return poses;
    }

    @Test
    @DisplayName("C: corrupting every height after frame 120 leaves frames 0-120 bit-identical")
    void theFutureCannotReachThePast() {
        double[] clean = new double[FRAMES];
        for (int k = 0; k < FRAMES; k++) {
            clean[k] = 0.4 * k;
        }
        double[] poisoned = clean.clone();
        for (int k = 121; k < FRAMES; k++) {
            poisoned[k] += 400.0;               // a physically absurd jump, deliberately
        }

        List<double[]> a = run(clean);
        List<double[]> b = run(poisoned);

        for (int k = 0; k <= 120; k++) {
            assertEquals(a.get(k)[0], b.get(k)[0], 0.0, "east at frame " + k + " saw the future");
            assertEquals(a.get(k)[1], b.get(k)[1], 0.0, "north at frame " + k + " saw the future");
            assertEquals(a.get(k)[2], b.get(k)[2], 0.0, "yaw at frame " + k + " saw the future");
        }
        // And the corruption was real: the LAST pose must move, or the assertions above are vacuous.
        assertTrue(Math.hypot(a.get(FRAMES - 1)[0] - b.get(FRAMES - 1)[0],
                              a.get(FRAMES - 1)[1] - b.get(FRAMES - 1)[1]) > 1.0,
                "the corruption must change the later poses, or this test proves nothing");
    }

    @Test
    @DisplayName("C: frame 121 is the FIRST to move -- the k-1 convention, located exactly")
    void theFirstAffectedFrameIsTheOneAfterTheCorruptedReference() {
        // The height sampled at frame k is the REFERENCE for the increment into frame k+1
        // (LIT-VO-003 section 10.1). So corrupting the sample at frame 120 must leave frame 120's
        // own pose untouched and move frame 121's. Corrupting frame 121 instead would move 122's.
        // This pins the off-by-one in the direction the offline implementation fixed it.
        double[] clean = new double[FRAMES];
        double[] poisoned = clean.clone();
        poisoned[120] = 300.0;

        List<double[]> a = run(clean);
        List<double[]> b = run(poisoned);

        assertEquals(a.get(120)[0], b.get(120)[0], 0.0,
                "frame 120's own pose is integrated with frame 119's height, so it must not move");
        assertNotEquals(a.get(121)[0], b.get(121)[0],
                "frame 121's increment IS converted with frame 120's height, so it must move");
    }

    @Test
    @DisplayName("C: shifting the whole height series by one frame changes the answer")
    void theReferenceConventionIsNotAFreeChoice() {
        double[] rising = new double[FRAMES];
        for (int k = 0; k < FRAMES; k++) {
            rising[k] = 0.9 * k;             // 0 -> 179 m of climb, a x4 height change
        }
        double[] shifted = new double[FRAMES];
        System.arraycopy(rising, 1, shifted, 0, FRAMES - 1);
        shifted[FRAMES - 1] = rising[FRAMES - 1];

        double endA = run(rising).get(FRAMES - 1)[0];
        double endB = run(shifted).get(FRAMES - 1)[0];
        assertTrue(Math.abs(endA - endB) > 0.005 * Math.abs(endA), String.format(
                "using h_k instead of h_{k-1} must be measurably different: %.6f vs %.6f m",
                endA, endB));
    }

    @Test
    @DisplayName("C: a late-arriving sample is used from its availability time on, never earlier")
    void latencyDelaysWhenASampleTakesEffect() {
        CausalHeightChannel ch = new CausalHeightChannel(H0, 5.0, DT);
        ch.submit(0.0, 0.0);
        ch.submit(1.0, 40.0);                 // taken at 0.5 s, published at 1.0 s
        assertEquals(H0, ch.readAt(0.7).heightAglM(), 1e-12,
                "a sample published at 1.0 s must not be visible at 0.7 s, whenever it was taken");
        assertEquals(H0 + 40.0, ch.readAt(1.0).heightAglM(), 1e-12);
    }
}
