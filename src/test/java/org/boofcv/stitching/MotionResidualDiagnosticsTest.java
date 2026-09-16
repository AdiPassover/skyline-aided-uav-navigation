package org.boofcv.stitching;

import boofcv.struct.image.GrayF32;
import org.boofcv.stitching.MotionResidualDiagnostics.Summary;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Semantics of the residual diagnostics themselves: that absence stays absent, that the reported
 * statistics are internally consistent, and that the residuals really are measured against the
 * model the estimator shipped rather than against stale or reset state.
 *
 * <p>Behavioural equivalence with the non-instrumented path is covered separately by
 * {@link StitchingFactoryInstrumentationEquivalenceTest}; this class assumes it.
 *
 * <p>Evidence tier: T1 (synthetic). These are known-answer and invariant checks on the extraction
 * layer. They say nothing about whether the residual is a <em>useful</em> failure predictor —
 * establishing that is CONF-lane work against real imagery.
 */
public class MotionResidualDiagnosticsTest {

    private static final int WORLD_SIZE = 512;
    private static final int FRAME_SIZE = 320;
    private static final int FRAME_COUNT = 20;

    private static GrayF32 syntheticTexturedWorld() {
        GrayF32 world = new GrayF32(WORLD_SIZE, WORLD_SIZE);
        Random random = new Random(4242);
        for (int y = 0; y < WORLD_SIZE; y++) {
            for (int x = 0; x < WORLD_SIZE; x++) {
                world.set(x, y, random.nextInt(256));
            }
        }
        return world;
    }

    private static GrayF32 frameAt(GrayF32 world, int index) {
        int offsetX = 4 + index * 8;
        int offsetY = 4 + index * 5;
        GrayF32 frame = new GrayF32(FRAME_SIZE, FRAME_SIZE);
        for (int y = 0; y < FRAME_SIZE; y++) {
            for (int x = 0; x < FRAME_SIZE; x++) {
                frame.set(x, y, world.get(x + offsetX, y + offsetY));
            }
        }
        return frame;
    }

    private record Captured(boolean success, Summary summary, double[] residuals) {
    }

    private static List<Captured> run(InstrumentedStitching<GrayF32> instrumented) {
        StitchingEstimator<GrayF32> estimator = new StitchingEstimator<>(instrumented.stitch());
        estimator.setShrinkScale(0.5);
        estimator.setMinDistanceFromBorder(10);

        GrayF32 world = syntheticTexturedWorld();
        List<Captured> captured = new ArrayList<>();
        for (int i = 0; i < FRAME_COUNT; i++) {
            boolean success = estimator.processFrame(frameAt(world, i));
            captured.add(new Captured(
                    success,
                    instrumented.diagnostics().summarize(),
                    instrumented.diagnostics().inlierSquaredResidualsPx()));
        }
        return captured;
    }

    @Test
    public void firstFrameReportsUnavailableRatherThanZero() {
        InstrumentedStitching<GrayF32> instrumented =
                StitchingFactory.builder().buildGrayInstrumented();

        StitchingEstimator<GrayF32> estimator = new StitchingEstimator<>(instrumented.stitch());
        estimator.processFrame(frameAt(syntheticTexturedWorld(), 0));

        Summary summary = instrumented.diagnostics().summarize();

        assertFalse(summary.hasResiduals(), "no motion has been estimated yet on the first frame");
        assertNull(summary.inlierCount(), "inlier count must be absent, not 0");
        assertNull(summary.meanSquaredPx(), "mean must be absent, not 0.0");
        assertNull(summary.rmsPx());
        assertNull(summary.medianSquaredPx());
        assertNull(summary.maxSquaredPx());
        assertNull(instrumented.diagnostics().inlierSquaredResidualsPx(),
                "absent residuals must be null, not an empty array");
    }

    @Test
    public void unavailableIsDistinguishableFromAMeasuredZero() {
        Summary absent = Summary.UNAVAILABLE;
        Summary perfectFit = new Summary(120, 0.0, 0.0, 0.0, 0.0);

        assertFalse(absent.hasResiduals());
        assertTrue(perfectFit.hasResiduals(),
                "a genuinely perfect fit is a measurement of zero, not an absence");
        assertNull(absent.inlierCount());
        assertEquals(120, perfectFit.inlierCount());
        assertFalse(absent.equals(perfectFit),
                "absence and a measured zero must not compare equal");
    }

    @Test
    public void resetClearsDiagnosticsRatherThanLeavingStaleValues() {
        InstrumentedStitching<GrayF32> instrumented =
                StitchingFactory.builder().buildGrayInstrumented();
        StitchingEstimator<GrayF32> estimator = new StitchingEstimator<>(instrumented.stitch());
        estimator.setShrinkScale(0.5);
        estimator.setMinDistanceFromBorder(10);

        GrayF32 world = syntheticTexturedWorld();
        for (int i = 0; i < 6; i++) {
            estimator.processFrame(frameAt(world, i));
        }

        estimator.reset();

        Summary afterReset = instrumented.diagnostics().summarize();
        assertFalse(afterReset.hasResiduals(),
                "after a reset the previous sequence's residuals must not still be reported");
        assertNull(instrumented.diagnostics().inlierSquaredResidualsPx());
    }

    @Test
    public void reportedStatisticsAreConsistentWithTheResidualArray() {
        List<Captured> captured = run(StitchingFactory.builder().buildGrayInstrumented());

        int framesWithResiduals = 0;
        for (int i = 0; i < captured.size(); i++) {
            Captured c = captured.get(i);
            Summary s = c.summary();
            if (!s.hasResiduals()) {
                assertNull(c.residuals(), "frame " + i + ": no summary implies no residual array");
                continue;
            }
            framesWithResiduals++;

            double[] residuals = c.residuals();
            assertNotNull(residuals, "frame " + i + ": summary present implies residuals present");
            assertEquals(s.inlierCount().intValue(), residuals.length,
                    "frame " + i + ": inlier count must be the size of the set measured over");

            double sum = 0.0;
            double max = Double.NEGATIVE_INFINITY;
            for (double r : residuals) {
                assertTrue(r >= 0.0, "frame " + i + ": a squared distance cannot be negative");
                sum += r;
                max = Math.max(max, r);
            }

            assertEquals(sum / residuals.length, s.meanSquaredPx(), 1e-12,
                    "frame " + i + ": mean");
            assertEquals(Math.sqrt(s.meanSquaredPx()), s.rmsPx(), 1e-12,
                    "frame " + i + ": rms must be the root of the mean squared residual");
            assertEquals(max, s.maxSquaredPx(), 1e-12, "frame " + i + ": max");

            boolean medianIsAMemberOfTheSet = false;
            for (double r : residuals) {
                if (r == s.medianSquaredPx()) {
                    medianIsAMemberOfTheSet = true;
                    break;
                }
            }
            assertTrue(medianIsAMemberOfTheSet,
                    "frame " + i + ": the reported median must be a value the estimator actually "
                            + "produced, not an interpolation");
        }

        assertTrue(framesWithResiduals > 0,
                "the synthetic sequence produced no residuals at all, so nothing was verified");
    }

    /**
     * The sharpest check that the residuals describe the shipped model and the current frame.
     *
     * <p>RANSAC admits a correspondence to its match set only if its squared reprojection error is
     * below the configured threshold. With model refinement off (the default), the shipped model is
     * the model RANSAC selected with, so every reported residual must respect that bound. A
     * residual computed against a stale model, against a keyframe reference that respawn had
     * already reset, or against a mismatched frame would have no reason to stay inside it.
     *
     * <p>The threshold is deliberately tightened well below the repository default of 3.0 so the
     * bound is a real constraint rather than one satisfied by accident.
     */
    @Test
    public void everyResidualRespectsTheConfiguredInlierThreshold() {
        double tightThreshold = 0.4;

        InstrumentedStitching<GrayF32> instrumented = StitchingFactory.builder()
                .inlierThresholdSq(tightThreshold)
                .buildGrayInstrumented();

        assertEquals(tightThreshold, instrumented.diagnostics().inlierThresholdSquaredPx(), 0.0,
                "the diagnostics must echo the configured threshold so magnitudes are interpretable");

        List<Captured> captured = run(instrumented);

        int checked = 0;
        for (int i = 0; i < captured.size(); i++) {
            double[] residuals = captured.get(i).residuals();
            if (residuals == null) {
                continue;
            }
            for (double r : residuals) {
                assertTrue(r < tightThreshold,
                        "frame " + i + ": residual " + r + " is not below the configured inlier "
                                + "threshold " + tightThreshold + "; the residual is not being "
                                + "measured against the model RANSAC selected these inliers with");
                checked++;
            }
        }
        assertTrue(checked > 0, "no residuals were checked, so the bound was never tested");
    }

    /**
     * Guards the specific failure mode the snapshot design exists to prevent.
     *
     * <p>BoofCV's respawn logic resets the keyframe reference immediately after the motion is
     * estimated. Residuals read after {@code process} returns would therefore map points onto
     * themselves through the identity transform on respawn frames and report an essentially perfect
     * fit. If every frame reports a residual of zero, that regression has been reintroduced.
     */
    @Test
    public void residualsAreNotUniformlyZero() {
        List<Captured> captured = run(StitchingFactory.builder().buildGrayInstrumented());

        boolean sawPositiveResidual = false;
        for (Captured c : captured) {
            if (c.summary().hasResiduals() && c.summary().maxSquaredPx() > 0.0) {
                sawPositiveResidual = true;
                break;
            }
        }

        assertTrue(sawPositiveResidual,
                "every frame reported a zero residual, which on real tracked features indicates "
                        + "the residual is being measured against a keyframe that respawn had "
                        + "already reset to the current frame -- see ResidualSnapshotTrackerKey");
    }

    @Test
    public void residualArrayIsACopyAndCannotMutateEstimatorState() {
        InstrumentedStitching<GrayF32> instrumented =
                StitchingFactory.builder().buildGrayInstrumented();
        StitchingEstimator<GrayF32> estimator = new StitchingEstimator<>(instrumented.stitch());
        estimator.setShrinkScale(0.5);

        GrayF32 world = syntheticTexturedWorld();
        for (int i = 0; i < 4; i++) {
            estimator.processFrame(frameAt(world, i));
        }

        double[] first = instrumented.diagnostics().inlierSquaredResidualsPx();
        assertNotNull(first, "expected residuals after four tracked frames");

        double original = first[0];
        first[0] = 999.0;

        double[] second = instrumented.diagnostics().inlierSquaredResidualsPx();
        assertNotNull(second);
        assertEquals(original, second[0], 0.0,
                "mutating a returned residual array must not affect the estimator's own values");
    }
}
