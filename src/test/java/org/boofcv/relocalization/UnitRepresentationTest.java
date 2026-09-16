package org.boofcv.relocalization;

import org.boofcv.stitching.metric.HeadingReading;
import org.boofcv.stitching.metric.HeadingSemantics;
import org.boofcv.stitching.metric.HeadingStatus;
import org.boofcv.stitching.metric.HeightReading;
import org.boofcv.stitching.metric.HeightStatus;
import org.boofcv.stitching.metric.MetricNavigationState;
import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The horizontal-unit question ({@code DEC-INT-004}), pinned as arithmetic on the real
 * {@link MetricNavigationState}: the three candidate representations of one visual increment
 * stream —
 *
 * <pre>
 *   RAW         dq_k                         (pixels, altitude-dependent scale f/h)
 *   METRIC      dq_k * h_{k-1} / f_working   (metres; the current contract, DEC-VO-009)
 *   NORMALISED  dq_k * h_{k-1} / h_ref       (reference-height pixels; a proposed alternative)
 * </pre>
 *
 * <p>What the test proves: (1) NORMALISED is METRIC times the constant {@code f_working / h_ref}
 * on every frame, whatever the height does — the same geometry, the same drift, the same terrain
 * error, one unit label apart; (2) RAW is METRIC times a constant only while the height is
 * constant, and two segments restarted at different heights disagree on the scale of the same
 * physical displacement by exactly {@code h_2 / h_1}, which a translation-only alignment cannot
 * absorb; (3) the layer consumes metres and refuses a pixel-valued record. T1 — arithmetic only.
 */
public class UnitRepresentationTest {

    private static final double F_WORKING = 256.0;           // fx 512 native, downsampleFactor 2
    private static final double H0 = 45.0;

    private static HeightReading height(double t, double relativeM) {
        return new HeightReading(t, t, 0.0, relativeM, H0, H0 + relativeM, HeightStatus.FRESH);
    }

    private static HeadingReading heading(double t, double deg) {
        return new HeadingReading(t, t, 0.0, deg, 0.0, deg, HeadingSemantics.SIM_NADIR_CAMERA_HEADING,
                HeadingStatus.FRESH);
    }

    /** One increment in the doc-comment arithmetic of {@link MetricNavigationState}, with an arbitrary scale. */
    private static double[] rotate(double dqX, double dqY, double psiDeg, double scale) {
        double c = Math.cos(Math.toRadians(psiDeg)), s = Math.sin(Math.toRadians(psiDeg));
        return new double[]{(c * dqX - s * dqY) * scale, -(s * dqX + c * dqY) * scale};
    }

    /** A varying-height profile: the aircraft climbs 25 m over the flight, on undulating terrain. */
    private static double relativeHeight(int k) {
        return 25.0 * (1 - Math.cos(k / 40.0)) / 2 + 3.0 * Math.sin(k / 7.0);
    }

    @Test
    public void heightNormalisedIsMetricTimesOneConstantOnEveryFrameWhateverTheHeightDoes() {
        final double hRef = H0;                              // any positive constant; h0 is the natural one
        MetricNavigationState m = new MetricNavigationState(F_WORKING, true);
        m.observeOrigin(height(0.0, relativeHeight(0)), heading(0.0, 12.0));
        double normE = 0.0, normN = 0.0;
        double hPrev = H0 + relativeHeight(0), psiPrev = 12.0;
        double ratioMin = Double.POSITIVE_INFINITY, ratioMax = Double.NEGATIVE_INFINITY;
        for (int k = 1; k <= 400; k++) {
            double dqX = 6.0 + 2.0 * Math.sin(k / 9.0), dqY = -3.0 + Math.cos(k / 5.0);
            double psi = 12.0 + 0.3 * k;
            m.observe(dqX, dqY, 0.0, height(k * 0.1, relativeHeight(k)), heading(k * 0.1, psi));
            double[] dn = rotate(dqX, dqY, psiPrev, hPrev / hRef);           // NORMALISED, by its formula
            normE += dn[0];
            normN += dn[1];
            hPrev = H0 + relativeHeight(k);
            psiPrev = psi;

            double expectedE = m.eastM() * F_WORKING / hRef, expectedN = m.northM() * F_WORKING / hRef;
            assertEquals(expectedE, normE, 1e-9 * Math.max(1.0, Math.abs(normE)), "east, frame " + k);
            assertEquals(expectedN, normN, 1e-9 * Math.max(1.0, Math.abs(normN)), "north, frame " + k);
            if (Math.abs(m.eastM()) > 1e-6) {
                double r = normE / m.eastM();
                ratioMin = Math.min(ratioMin, r);
                ratioMax = Math.max(ratioMax, r);
            }
        }
        assertEquals(F_WORKING / hRef, ratioMin, 1e-9, "the factor is f_working / h_ref ...");
        assertEquals(F_WORKING / hRef, ratioMax, 1e-9, "... and it never moves, though the height moved 25 m");
        assertTrue(H0 + relativeHeight(400) - H0 > 20.0, "the height really varied");
    }

    @Test
    public void rawPixelsShareMetricGeometryOnlyWhileTheHeightIsConstant() {
        // Constant height: raw * gsd0 == metric, exactly.
        MetricNavigationState flat = new MetricNavigationState(F_WORKING, true);
        flat.observeOrigin(height(0.0, 0.0), heading(0.0, 0.0));
        double rawE = 0.0, rawN = 0.0;
        for (int k = 1; k <= 200; k++) {
            double dqX = 5.0, dqY = 1.0 * Math.sin(k / 11.0);
            flat.observe(dqX, dqY, 0.0, height(k * 0.1, 0.0), heading(k * 0.1, 0.0));
            double[] d = rotate(dqX, dqY, 0.0, 1.0);
            rawE += d[0];
            rawN += d[1];
            assertEquals(rawE * H0 / F_WORKING, flat.eastM(), 1e-9, "east " + k);
            assertEquals(rawN * H0 / F_WORKING, flat.northM(), 1e-9, "north " + k);
        }

        // Varying height: the raw/metric ratio is the instantaneous f/h, not a constant.
        MetricNavigationState climb = new MetricNavigationState(F_WORKING, true);
        climb.observeOrigin(height(0.0, 0.0), heading(0.0, 0.0));
        double prevMetricE = 0.0;
        double first = Double.NaN, last = Double.NaN;
        for (int k = 1; k <= 200; k++) {
            climb.observe(5.0, 0.0, 0.0, height(k * 0.1, 0.2 * k), heading(k * 0.1, 0.0));
            double dMetric = climb.eastM() - prevMetricE;                   // 5 px * h_{k-1} / f
            prevMetricE = climb.eastM();
            double ratio = 5.0 / dMetric;                                   // raw px per metre = f / h_{k-1}
            if (k == 1) first = ratio;
            last = ratio;
        }
        assertEquals(F_WORKING / H0, first, 1e-9);
        assertEquals(F_WORKING / (H0 + 0.2 * 199), last, 1e-9);
        assertNotEquals(first, last, "raw units drift with altitude: 5 px is fewer metres higher up");
    }

    /**
     * Two independently restarted segments at different heights: the same physical displacement is
     * the same number of metres in both, and a different number of raw pixels — by exactly
     * {@code h_2 / h_1}. INT's alignment is a translation ({@code DEC-INT-001}): it can relate the
     * two metric segments, and there is no translation that relates the two raw ones.
     */
    @Test
    public void restartedSegmentsAgreeInMetresAndDisagreeInRawPixelsByTheHeightRatio() {
        final double h1 = H0, h2 = 2.0 * H0;
        final double physicalEastM = 30.0;                   // the aircraft flies 30 m east in each segment
        MetricNavigationState m = new MetricNavigationState(F_WORKING, true);

        m.observeOrigin(height(0.0, h1 - H0), heading(0.0, 0.0));
        double rawPx1 = physicalEastM * F_WORKING / h1;      // what the camera sees at h1
        m.observe(rawPx1, 0.0, 0.0, height(0.1, h1 - H0), heading(0.1, 0.0));
        double seg1MetricE = m.segmentRelativePose().x;

        m.beginNewSegment(true);                             // a hard loss; the aircraft has climbed
        m.observeOrigin(height(5.0, h2 - H0), heading(5.0, 0.0));
        double rawPx2 = physicalEastM * F_WORKING / h2;      // the SAME 30 m, seen from twice the height
        m.observe(rawPx2, 0.0, 0.0, height(5.1, h2 - H0), heading(5.1, 0.0));
        double seg2MetricE = m.segmentRelativePose().x;

        assertEquals(physicalEastM, seg1MetricE, 1e-9);
        assertEquals(physicalEastM, seg2MetricE, 1e-9, "metres: one scale across the restart");
        assertEquals(h2 / h1, rawPx1 / rawPx2, 1e-12, "raw pixels: the segments disagree by h2/h1");

        // Translation-only alignment on metres: any t_e relates the segments' frames consistently.
        AlignmentTransform t = AlignmentTransform.of(100.0, -50.0);
        PlanarPosition p1 = t.apply(new PlanarPosition(seg1MetricE, 0.0));
        PlanarPosition p2 = t.apply(new PlanarPosition(seg2MetricE, 0.0));
        assertEquals(p1, p2, "the same physical displacement maps to the same persistent point");
        // On raw pixels no translation can do that: the two segments' displacements differ by a factor.
        assertNotEquals(t.apply(new PlanarPosition(rawPx1, 0.0)), t.apply(new PlanarPosition(rawPx2, 0.0)));
    }

    /** The layer's inputs are metres by construction, and its reference record refuses any other contract. */
    @Test
    public void theLayerConsumesMetresAndItsReferencesRefuseAPixelContract() {
        MetricNavigationState m = new MetricNavigationState(F_WORKING, true);
        m.observeOrigin(height(0.0, 0.0), heading(0.0, 90.0));
        m.observe(10.0, 0.0, 0.0, height(0.1, 0.0), heading(0.1, 90.0));   // 10 px right at heading 90 = 10 px South
        LocalPoseSample s = LocalPoseSample.fromMetric(1, 0.1, true, m, new Pose3D());
        assertEquals(m.segmentRelativePose().x, s.segmentPosition().eastM(), 0.0);
        assertEquals(m.segmentRelativePose().y, s.segmentPosition().northM(), 0.0);
        assertEquals(-10.0 * H0 / F_WORKING, s.segmentPosition().northM(), 1e-9, "metres, not pixels");
        assertTrue(TrustedReference.POSE_SCHEMA.contains("metres ENU"));
        assertTrue(SnapSafety.UNITS.startsWith("metres"));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new TrustedReference(0, 1, 0.1, TestProfiles.descriptor(0, 0), null,
                        new PlanarPosition(10.0, 0.0), 90.0, HeadingStatus.FRESH, 0, 0, null, null, 0, false,
                        "int-reference-pose/1: position first-frame pixels; yaw start-relative visual"));
    }
}
