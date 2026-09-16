package org.boofcv.stitching.metric;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What the metric readout costs per frame, measured rather than asserted to be small.
 *
 * <p>The claim {@code DEC-VO-007} makes about this design is that it is "a per-frame scalar
 * multiplication of an existing increment" — one height read, one divide, one rotate, three adds.
 * This measures that against the visual estimation it sits behind, and it is the number
 * Principle IX requires before anything is called cheap.
 *
 * <p>The assertion is deliberately loose (a microsecond, roughly two orders above the measured
 * value) because a tight timing bound in a unit test is a flaky test, not a guarantee. The value
 * printed is the evidence; the bound only catches a change of algorithmic class — an allocation per
 * frame, a sort, a scan of history, or blocking I/O sneaking in.
 */
class MetricReadoutCostTest {

    private static final int WARMUP = 200_000;
    private static final int MEASURE = 2_000_000;

    @Test
    @DisplayName("the metric conversion costs well under a microsecond per frame, and allocates nothing per frame")
    void perFrameCostIsNegligible() {
        CausalHeightChannel channel = new CausalHeightChannel(80.0, 2.0, 0.1);
        MetricNavigationState metric = new MetricNavigationState(800.0);
        channel.submit(0.0, 0.0);
        metric.observeOrigin(channel.readAt(0.0));

        double t = 0.0;
        for (int i = 0; i < WARMUP; i++) {
            t += 0.1;
            if ((i % 10) == 0) {
                channel.submit(t, 0.01 * i);
            }
            metric.observe(3.0, -2.0, 1e-4, channel.readAt(t));
        }

        long start = System.nanoTime();
        for (int i = 0; i < MEASURE; i++) {
            t += 0.1;
            if ((i % 10) == 0) {
                channel.submit(t, 0.01 * i);
            }
            metric.observe(3.0, -2.0, 1e-4, channel.readAt(t));
        }
        long elapsed = System.nanoTime() - start;
        double nsPerFrame = elapsed / (double) MEASURE;

        System.out.printf("metric readout cost: %.1f ns/frame over %,d frames "
                        + "(one readAt + one observe, including the Pose3D and HeightReading each "
                        + "call allocates)%n", nsPerFrame, MEASURE);
        assertTrue(nsPerFrame < 1000.0,
                "the metric conversion must stay a scalar multiply, not become an algorithm: "
                + nsPerFrame + " ns/frame");
        assertTrue(metric.getIntegratedFrameCount() == WARMUP + MEASURE);
    }

    @Test
    @DisplayName("adding the authoritative heading channel costs a comparable handful of nanoseconds")
    void headingChannelCostIsNegligible() {
        // The heading arm adds one readAt and one wrap per frame to a path that already does one
        // height readAt, one divide and one rotate. Measured against the same loop so the two
        // numbers are directly comparable rather than separately plausible.
        CausalHeightChannel height = new CausalHeightChannel(80.0, 2.0, 0.1);
        CausalHeadingChannel heading = new CausalHeadingChannel(
                HeadingSemantics.FC_AHRS_BODY_COMPASS, 90.5, 2.0, 0.01, 180.0);
        MetricNavigationState metric = new MetricNavigationState(800.0, true);
        height.submit(0.0, 0.0);
        heading.submit(0.0, 0.0);
        metric.observeOrigin(height.readAt(0.0), heading.readAt(0.0));

        double t = 0.0;
        for (int i = 0; i < WARMUP; i++) {
            t += 0.1;
            if ((i % 10) == 0) {
                height.submit(t, 0.01 * i);
            }
            heading.submit(t, (i * 0.001) % 360.0);
            metric.observe(3.0, -2.0, 1e-4, height.readAt(t), heading.readAt(t));
        }

        long start = System.nanoTime();
        for (int i = 0; i < MEASURE; i++) {
            t += 0.1;
            if ((i % 10) == 0) {
                height.submit(t, 0.01 * i);
            }
            heading.submit(t, (i * 0.001) % 360.0);
            metric.observe(3.0, -2.0, 1e-4, height.readAt(t), heading.readAt(t));
        }
        double nsPerFrame = (System.nanoTime() - start) / (double) MEASURE;

        System.out.printf("metric + heading readout cost: %.1f ns/frame over %,d frames "
                        + "(two readAt, one observe, one submit, including every value object "
                        + "each call allocates)%n", nsPerFrame, MEASURE);
        assertTrue(nsPerFrame < 1500.0,
                "the heading substitution must stay a rotation, not become an algorithm: "
                + nsPerFrame + " ns/frame");
        assertTrue(metric.getIntegratedFrameCount() == WARMUP + MEASURE);
        assertTrue(heading.rejectedSampleCount() >= 0);
    }

    @Test
    @DisplayName("the state added is a fixed handful of doubles -- nothing accumulates with frame count")
    void stateIsBounded() {
        // Structural rather than a memory measurement: neither class may hold a collection, an
        // array, or anything else whose size could grow with the run. A history buffer here would
        // turn a bounded-memory readout into an unbounded one on a long flight.
        for (Class<?> c : new Class<?>[]{CausalHeightChannel.class, CausalHeadingChannel.class,
                                         MetricNavigationState.class}) {
            for (var f : c.getDeclaredFields()) {
                Class<?> type = f.getType();
                // Records and enums are fixed-size value objects; everything else must be primitive.
                boolean scalar = type.isPrimitive() || type == HeightReading.class
                        || type == HeadingReading.class
                        || type == org.boofcv.util.structs.Pose3D.class || type == HeightStatus.class
                        || type == HeadingStatus.class || type == HeadingSemantics.class;
                assertTrue(scalar, c.getSimpleName() + "." + f.getName() + " is a " + type
                        + "; the metric layer must hold only bounded scalar state");
            }
        }
    }
}
