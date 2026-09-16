package org.boofcv.stitching.diagnostics;

import boofcv.struct.image.GrayF32;
import org.boofcv.evaluation.FrameSource;
import org.boofcv.stitching.InstrumentedStitching;
import org.boofcv.stitching.StitchingEstimator;
import org.boofcv.stitching.StitchingFactory;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.StringWriter;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Output-shape guarantees of the {@code EXP-VO-001} measurement harness: that absence survives the
 * trip to CSV, that every frame is accounted for, and that the harness cannot be pointed at
 * diagnostics belonging to a different estimator than the one producing the pose.
 *
 * <p>Evidence tier: T1 (synthetic). These check the recording layer, not the residual's meaning and
 * emphatically not its usefulness as an indicator.
 */
public class ResidualProbeRunnerTest {

    private static final int WORLD_SIZE = 512;
    private static final int FRAME_SIZE = 320;
    private static final int FRAME_COUNT = 20;

    /** Frames cropped from a fixed random-noise world, sliding by a constant offset per frame. */
    private static final class SyntheticFrames implements FrameSource {
        private final GrayF32 world;

        SyntheticFrames() {
            world = new GrayF32(WORLD_SIZE, WORLD_SIZE);
            Random random = new Random(4242);
            for (int y = 0; y < WORLD_SIZE; y++) {
                for (int x = 0; x < WORLD_SIZE; x++) {
                    world.set(x, y, random.nextInt(256));
                }
            }
        }

        @Override
        public int frameCount() {
            return FRAME_COUNT;
        }

        @Override
        public TimestampedFrame frame(int index) {
            int offsetX = 4 + index * 8;
            int offsetY = 4 + index * 5;
            GrayF32 frame = new GrayF32(FRAME_SIZE, FRAME_SIZE);
            for (int y = 0; y < FRAME_SIZE; y++) {
                for (int x = 0; x < FRAME_SIZE; x++) {
                    frame.set(x, y, world.get(x + offsetX, y + offsetY));
                }
            }
            return new TimestampedFrame(index, index * 0.1, frame);
        }
    }

    private record Probe(List<ResidualProbeRunner.Row> rows, String csv) {
    }

    private static Probe run() throws IOException {
        InstrumentedStitching<GrayF32> instrumented =
                StitchingFactory.builder().buildGrayInstrumented();
        StitchingEstimator<GrayF32> estimator = new StitchingEstimator<>(instrumented.stitch());
        estimator.setShrinkScale(0.5);
        estimator.setMinDistanceFromBorder(10);

        StringWriter csv = new StringWriter();
        List<ResidualProbeRunner.Row> rows =
                new ResidualProbeRunner(instrumented, estimator, new SyntheticFrames()).run(csv);
        return new Probe(rows, csv.toString());
    }

    private static int columnIndex(String header, String name) {
        String[] columns = header.split(",", -1);
        for (int i = 0; i < columns.length; i++) {
            if (columns[i].equals(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("no column named " + name);
    }

    @Test
    public void everyFrameProducesExactlyOneRow() throws IOException {
        Probe probe = run();
        String[] lines = probe.csv().split("\n", -1);

        assertEquals(FRAME_COUNT, probe.rows().size());
        // header + one row per frame + the trailing empty string after the final newline
        assertEquals(FRAME_COUNT + 2, lines.length);
        assertEquals(ResidualProbeRunner.HEADER, lines[0]);
    }

    @Test
    public void everyRowHasTheSameNumberOfFieldsAsTheHeader() throws IOException {
        Probe probe = run();
        String[] lines = probe.csv().split("\n", -1);
        int expected = ResidualProbeRunner.HEADER.split(",", -1).length;

        for (int i = 1; i <= FRAME_COUNT; i++) {
            assertEquals(expected, lines[i].split(",", -1).length,
                    "row " + i + " field count must match the header, so empty means absent "
                            + "rather than shifting every later column");
        }
    }

    @Test
    public void absentResidualsAreWrittenEmptyNeverZero() throws IOException {
        Probe probe = run();
        String[] lines = probe.csv().split("\n", -1);
        int meanColumn = columnIndex(lines[0], "residual_mean_sq_px");
        int countColumn = columnIndex(lines[0], "residual_inlier_count");

        // Frame 0 has no frame-to-keyframe motion yet: the run record must say so, not say zero.
        String[] first = lines[1].split(",", -1);
        assertEquals("", first[meanColumn],
                "the first frame's residual is absent; '0.0' would read as a perfect fit");
        assertEquals("", first[countColumn]);
        assertNull(probe.rows().get(0).residuals().meanSquaredPx());

        // ... and the sequence must actually produce measurements somewhere, or this test would
        // pass on a harness that reported absence for everything.
        int measured = 0;
        for (int i = 2; i <= FRAME_COUNT; i++) {
            if (!lines[i].split(",", -1)[meanColumn].isEmpty()) {
                measured++;
            }
        }
        assertTrue(measured > 0, "no frame carried a measured residual; the harness measured nothing");
    }

    @Test
    public void everyFrameCarriesEitherAMeasurementOrAnExplicitAbsence() throws IOException {
        Probe probe = run();
        for (ResidualProbeRunner.Row row : probe.rows()) {
            boolean measured = row.residuals().hasResiduals();
            boolean absent = row.residuals().meanSquaredPx() == null;
            assertTrue(measured ^ absent, "frame " + row.frameIndex()
                    + " must be exactly one of measured or absent");
            if (measured) {
                assertNotNull(row.residuals().inlierCount());
                assertNotNull(row.p90SquaredPx());
                assertNotNull(row.p95SquaredPx());
            } else {
                assertNull(row.p90SquaredPx(), "an absent residual array yields no percentile, not 0.0");
                assertNull(row.p95SquaredPx());
            }
        }
    }

    @Test
    public void probeComputedPercentilesAreOrderedWithinTheSummary() throws IOException {
        Probe probe = run();
        int checked = 0;
        for (ResidualProbeRunner.Row row : probe.rows()) {
            if (!row.residuals().hasResiduals()) {
                continue;
            }
            checked++;
            assertTrue(row.residuals().medianSquaredPx() <= row.p90SquaredPx(),
                    "median must not exceed p90 on frame " + row.frameIndex());
            assertTrue(row.p90SquaredPx() <= row.p95SquaredPx());
            assertTrue(row.p95SquaredPx() <= row.residuals().maxSquaredPx());
        }
        assertTrue(checked > 0, "nothing was checked");
    }

    @Test
    public void percentileIsNearestRankAndAbsentForAnAbsentArray() {
        double[] values = {5.0, 1.0, 4.0, 2.0, 3.0};

        assertEquals(3.0, ResidualProbeRunner.percentileSquaredPx(values, 0.5));
        assertEquals(5.0, ResidualProbeRunner.percentileSquaredPx(values, 0.90),
                "nearest-rank: every reported value must be one the estimator actually produced");
        assertEquals(1.0, ResidualProbeRunner.percentileSquaredPx(values, 0.0));

        assertNull(ResidualProbeRunner.percentileSquaredPx(null, 0.9));
        assertNull(ResidualProbeRunner.percentileSquaredPx(new double[0], 0.9),
                "an empty match set has no order statistic; absence is not zero");
    }

    @Test
    public void diagnosticsFromADifferentEstimatorAreRejected() {
        InstrumentedStitching<GrayF32> one = StitchingFactory.builder().buildGrayInstrumented();
        InstrumentedStitching<GrayF32> other = StitchingFactory.builder().buildGrayInstrumented();
        StitchingEstimator<GrayF32> estimator = new StitchingEstimator<>(other.stitch());

        assertThrows(IllegalArgumentException.class,
                () -> new ResidualProbeRunner(one, estimator, new SyntheticFrames()),
                "a residual read against the wrong estimator is the failure DEC-VO-001 exists to prevent");
    }

    @Test
    public void eventClassificationMarksTheFirstFrameAsInit() throws IOException {
        Probe probe = run();
        assertEquals("init", probe.rows().get(0).event());
        assertEquals(0, probe.rows().get(0).referenceId());
        for (ResidualProbeRunner.Row row : probe.rows().subList(1, probe.rows().size())) {
            assertFalse(row.event().equals("init"), "only frame 0 is init");
        }
    }
}
