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
 * Known-answer tests for the metric integrator: {@code DEC-VO-007} D3 with every quantity chosen so
 * the expected metric displacement can be written down by hand.
 *
 * <p>Covers the fixed-height case (A), the varying-height case (B), the reference-frame convention
 * (C, arithmetically), the {@code f_working} divisor, and the restart/segment semantics (H) at the
 * integrator level. Nothing here uses imagery or an estimator, so a failure is unambiguously in the
 * conversion rather than in the vision.
 */
class MetricNavigationStateTest {

    private static final double EPS = 1e-12;

    private static HeightReading fresh(double tS, double relM, double h0M) {
        return new HeightReading(tS, tS, 0.0, relM, h0M, h0M + relM, HeightStatus.FRESH);
    }

    // ================================================================ A: fixed height, known answer

    @Test
    @DisplayName("A: pure translation at a known fixed height gives the exact metric displacement")
    void fixedHeightPureTranslation() {
        // h = 80 m, f_working = 800 px  =>  gsd = 0.1 m/px exactly.
        double h0 = 80.0, f = 800.0;
        MetricNavigationState m = new MetricNavigationState(f);
        m.observeOrigin(fresh(0.0, 0.0, h0));
        for (int k = 1; k <= 10; k++) {
            m.observe(30.0, -20.0, 0.0, fresh(k * 0.1, 0.0, h0));
        }
        // 10 increments of (30, -20) px at 0.1 m/px, no rotation: T = (30, -20) m in image
        // convention, i.e. east +30 m, north +20 m (north = -T_y).
        assertEquals(30.0, m.eastM(), 1e-10);
        assertEquals(20.0, m.northM(), 1e-10);
        assertEquals(0.1, m.getLastGsdMPerPx(), EPS);
        Pose3D p = m.metricPose();
        assertEquals(30.0, p.x, 1e-10, "Pose3D.x is EAST in metres");
        assertEquals(20.0, p.y, 1e-10, "Pose3D.y is NORTH in metres");
        assertEquals(0.0, p.z, 0.0, "z is never estimated -- DEC-VO-007 D9 is horizontal only");
    }

    @Test
    @DisplayName("A: translation and rotation compose -- the increment is rotated by the PREVIOUS heading")
    void fixedHeightTranslationAndRotationCompose() {
        double h0 = 100.0, f = 1000.0;                       // gsd = 0.1 m/px
        MetricNavigationState m = new MetricNavigationState(f);
        m.observeOrigin(fresh(0.0, 0.0, h0));

        // Increment 1: 100 px east, then a quarter turn. The turn must NOT rotate this increment.
        m.observe(100.0, 0.0, Math.PI / 2.0, fresh(0.1, 0.0, h0));
        assertEquals(10.0, m.eastM(), 1e-10, "the first increment is applied at heading 0");
        assertEquals(0.0, m.northM(), 1e-10);

        // Increment 2: another 100 px "east" in the image, now at heading +90 deg. In the image
        // convention R(pi/2) sends (100, 0) to (0, 100), i.e. T_y = +100 px = +10 m, so north = -10.
        m.observe(100.0, 0.0, 0.0, fresh(0.2, 0.0, h0));
        assertEquals(10.0, m.eastM(), 1e-10);
        assertEquals(-10.0, m.northM(), 1e-10, "the second increment is rotated by heading_1");
        assertEquals(90.0, m.metricPose().yaw, 1e-10);
    }

    @Test
    @DisplayName("A: yaw is the raw rigid yaw, untouched by any height")
    void yawIsUnaffectedByHeight() {
        MetricNavigationState flat = new MetricNavigationState(500.0);
        MetricNavigationState climbing = new MetricNavigationState(500.0);
        flat.observeOrigin(fresh(0.0, 0.0, 40.0));
        climbing.observeOrigin(fresh(0.0, 0.0, 40.0));
        for (int k = 1; k <= 20; k++) {
            double dth = 0.017 * k;
            flat.observe(3.0, 4.0, dth, fresh(k * 0.1, 0.0, 40.0));
            climbing.observe(3.0, 4.0, dth, fresh(k * 0.1, 5.0 * k, 40.0));
        }
        assertEquals(flat.headingRadUnwrapped(), climbing.headingRadUnwrapped(), 0.0,
                "DEC-VO-007 D7: height constrains scale and nothing else -- yaw is bit-identical");
        assertNotEquals(flat.eastM(), climbing.eastM(), "but the metric translation must differ");
    }

    // ================================================================ B: varying height

    @Test
    @DisplayName("B: a doubling height with halving pixel flow recovers a CONSTANT ground step")
    void varyingHeightRecoversConstantGroundDisplacement() {
        // The physics being reproduced (LIT-VO-003 section 2): over a plane, the image displacement
        // of a fixed ground step scales as f/h. Fly the same 8 m ground step at heights
        // 40, 50, ..., 120 m; the pixel flow must shrink as 1/h, and the metric readout must give
        // back 8 m every time. Removing the height scaling makes this test fail by construction.
        double f = 600.0;
        double groundStepM = 8.0;
        double h0 = 40.0;
        MetricNavigationState m = new MetricNavigationState(f);
        m.observeOrigin(fresh(0.0, 0.0, h0));

        double[] heights = {40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0};
        for (int k = 1; k < heights.length; k++) {
            // The increment into frame k is expressed in frame k-1's pixels, so its pixel size is
            // set by frame k-1's height. That is the convention under test.
            double dqPx = groundStepM * f / heights[k - 1];
            m.observe(dqPx, 0.0, 0.0, fresh(k * 0.1, heights[k] - h0, h0));
            assertEquals(groundStepM * k, m.eastM(), 1e-9,
                    "step " + k + " at h=" + heights[k - 1] + " m must still be " + groundStepM + " m");
        }
        assertEquals(groundStepM * (heights.length - 1), m.eastM(), 1e-9);
    }

    @Test
    @DisplayName("B: removing the height scaling breaks the varying-height case -- the control")
    void withoutHeightScalingTheVaryingHeightCaseIsWrong() {
        // The same sequence integrated at a FIXED first-frame height, i.e. what the runtime did
        // before this port. It must NOT recover the constant ground step; if it did, the previous
        // test would be passing for the wrong reason.
        double f = 600.0, groundStepM = 8.0, h0 = 40.0;
        MetricNavigationState fixed = new MetricNavigationState(f);
        fixed.observeOrigin(fresh(0.0, 0.0, h0));
        double[] heights = {40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0};
        for (int k = 1; k < heights.length; k++) {
            double dqPx = groundStepM * f / heights[k - 1];
            fixed.observe(dqPx, 0.0, 0.0, fresh(k * 0.1, 0.0, h0));   // channel reports nothing
        }
        double truth = groundStepM * (heights.length - 1);
        assertTrue(Math.abs(fixed.eastM() - truth) > 0.25 * truth,
                "a fixed first-frame height must be materially wrong over a 3x altitude change; "
                + "got " + fixed.eastM() + " m against " + truth + " m");
    }

    // ================================================================ C: the k-1 convention

    @Test
    @DisplayName("C: the increment into frame k is converted with frame k-1's height, not frame k's")
    void referenceFrameHeightIsUsed() {
        double f = 100.0, h0 = 10.0;
        MetricNavigationState m = new MetricNavigationState(f);
        m.observeOrigin(fresh(0.0, 0.0, h0));            // h_0 = 10 m -> gsd 0.1
        m.observe(50.0, 0.0, 0.0, fresh(0.1, 90.0, h0)); // h_1 = 100 m; must NOT be used here
        assertEquals(5.0, m.eastM(), EPS,
                "50 px at frame 0's gsd of 0.1 m/px = 5 m; frame 1's gsd (1.0) would give 50 m");
        assertEquals(10.0, m.getLastUsedHeight().heightAglM(), EPS);
        m.observe(50.0, 0.0, 0.0, fresh(0.2, 90.0, h0)); // now frame 1's 100 m IS the reference
        assertEquals(55.0, m.eastM(), EPS);
        assertEquals(100.0, m.getLastUsedHeight().heightAglM(), EPS);
    }

    // ================================================================ f_working

    @Test
    @DisplayName("forgetting the downsample factor doubles every metric distance")
    void fWorkingDividesByTheDownsampleFactor() {
        MetricReadoutConfig two = new MetricReadoutConfig(1470.0, 2, 80.0, 2.0, 0.1);
        MetricReadoutConfig one = new MetricReadoutConfig(1470.0, 1, 80.0, 2.0, 0.1);
        assertEquals(735.0, two.fWorkingPx(), EPS);
        assertEquals(1470.0, one.fWorkingPx(), EPS);
        assertEquals(735.5326538, new MetricReadoutConfig(1471.0653076171875, 2, 80.0, 2.0, 0.1)
                .fWorkingPx(), 1e-6, "LIT-VO-003 section 10.2's HKairport01 value");

        MetricNavigationState correct = new MetricNavigationState(two.fWorkingPx());
        MetricNavigationState wrong = new MetricNavigationState(one.fWorkingPx());
        for (MetricNavigationState m : new MetricNavigationState[]{correct, wrong}) {
            m.observeOrigin(fresh(0.0, 0.0, 80.0));
            m.observe(10.0, 0.0, 0.0, fresh(0.1, 0.0, 80.0));
        }
        assertEquals(2.0 * wrong.eastM(), correct.eastM(), 1e-12,
                "using fx_native unreduced at downsampleFactor 2 halves every distance");
        assertThrows(IllegalArgumentException.class,
                () -> new MetricReadoutConfig(1470.0, 0, 80.0, 2.0, 0.1));
        assertThrows(IllegalArgumentException.class, () -> new MetricNavigationState(0.0));
    }

    // ================================================================ E: unusable height

    @Test
    @DisplayName("E: an unusable reference height produces NO metric increment -- the pose holds")
    void unusableHeightProducesNoIncrement() {
        MetricNavigationState m = new MetricNavigationState(800.0);
        m.observeOrigin(fresh(0.0, 0.0, 80.0));
        m.observe(100.0, 0.0, 0.0, new HeightReading(0.1, 0.1, 0.0, -80.0, 80.0, 0.0,
                HeightStatus.UNAVAILABLE));
        double afterGoodFrame = m.eastM();
        assertEquals(10.0, afterGoodFrame, 1e-10);

        m.observe(100.0, 0.0, 0.0, fresh(0.2, 0.0, 80.0));   // reference is the UNAVAILABLE reading
        assertEquals(afterGoodFrame, m.eastM(), 0.0,
                "no usable scale means no metric displacement -- never a guessed one");
        assertFalse(m.lastFrameUsable());
        assertEquals(1, m.getUnusableFrameCount());
    }

    @Test
    @DisplayName("E: a STALE reference height IS used and the frame is counted as degraded")
    void staleHeightIsUsedAndFlagged() {
        MetricNavigationState m = new MetricNavigationState(800.0);
        HeightReading stale = new HeightReading(0.0, -5.0, 5.0, 0.0, 80.0, 80.0, HeightStatus.STALE);
        m.observeOrigin(stale);
        m.observe(80.0, 0.0, 0.0, fresh(0.1, 0.0, 80.0));
        assertEquals(8.0, m.eastM(), 1e-10, "DEC-VO-007 D6: held and used, not refused");
        assertTrue(m.lastFrameUsable());
        assertEquals(HeightStatus.STALE, m.lastHeightStatus());
        assertEquals(1, m.getDegradedFrameCount(),
                "counted once, when the reading was USED -- the origin frame produces no increment");
    }

    // ================================================================ H: restart / segments

    @Test
    @DisplayName("H: a restart starts a new metric segment and inserts NO displacement")
    void restartStartsASegmentWithoutBridgingTheGap() {
        MetricNavigationState m = new MetricNavigationState(800.0);
        m.observeOrigin(fresh(0.0, 0.0, 80.0));                    // segment A at 80 m, gsd 0.1
        m.observe(100.0, 0.0, 0.0, fresh(0.1, 0.0, 80.0));
        assertEquals(10.0, m.eastM(), 1e-10);
        assertEquals(0, m.getSegmentIndex());
        assertFalse(m.isUnknownTranslationGapBeforeSegment());

        double beforeGap = m.eastM();
        m.beginNewSegment(true);
        assertEquals(beforeGap, m.eastM(), 0.0,
                "the metric layer contributes exactly zero across an unestimated interval -- it "
                + "neither invents motion nor bridges the gap");
        assertEquals(1, m.getSegmentIndex());
        assertTrue(m.isUnknownTranslationGapBeforeSegment(), "the gap must be legible, not silent");
        assertEquals(0.0, m.segmentRelativePose().x, 0.0, "the new segment starts at its own origin");

        // Segment B at 240 m: three times the height, so the same pixel flow is three times the
        // ground distance. Both segments are still metres -- that is the whole point of the port.
        m.observeOrigin(fresh(0.2, 160.0, 80.0));
        m.observe(100.0, 0.0, 0.0, fresh(0.3, 160.0, 80.0));
        assertEquals(0.3, m.getLastGsdMPerPx(), 1e-12);
        assertEquals(30.0, m.segmentRelativePose().x, 1e-10,
                "100 px at 240 m through f=800 px is 30 m, in the same physical unit as segment A");
        assertEquals(beforeGap + 30.0, m.eastM(), 1e-10);
    }

    @Test
    @DisplayName("H: segment-relative pose contains no unestimated interval")
    void segmentRelativePoseExcludesTheGap() {
        MetricNavigationState m = new MetricNavigationState(800.0);
        m.observeOrigin(fresh(0.0, 0.0, 80.0));
        m.observe(100.0, 0.0, 0.0, fresh(0.1, 0.0, 80.0));
        m.beginNewSegment(true);
        m.observeOrigin(fresh(0.2, 0.0, 80.0));
        m.observe(50.0, 0.0, 0.0, fresh(0.3, 0.0, 80.0));
        assertEquals(15.0, m.eastM(), 1e-10, "the continuous track keeps accumulating");
        assertEquals(5.0, m.segmentRelativePose().x, 1e-10,
                "the segment-relative pose carries only what was actually estimated since the gap");
    }

    // ================================================================ D4: visual scale isolation

    @Test
    @DisplayName("D4: the integrator has no visual-scale input and cannot acquire one")
    void visualScaleHasNoPathIntoTheMetricState() {
        // DEC-VO-007 D4 enforced structurally: MetricNavigationState.observe takes dq, dtheta and a
        // height reading. There is no lambda parameter, no setter, and no accumulated-scale field,
        // so a caller cannot feed visual scale in even deliberately. The property is asserted here
        // by construction rather than by a value comparison, and the bit-identity form of the same
        // claim is MetricReadoutPythonParityTest, which reproduces a Python track whose generator
        // is gated by test_g4.
        for (var mth : MetricNavigationState.class.getMethods()) {
            String n = mth.getName().toLowerCase();
            assertFalse(n.contains("scale") && !n.contains("gsd"),
                    "no scale-bearing entry point may exist on the metric state: " + mth);
        }
    }
}
