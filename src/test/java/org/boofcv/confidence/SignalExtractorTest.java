package org.boofcv.confidence;

import org.boofcv.evaluation.VoDiagnostics;
import org.boofcv.stitching.MotionResidualDiagnostics;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The pre-registered temporal state semantics, exactly as frozen in {@code EXP-CONF-001}
 * §Temporal signal state semantics (amendment A4). The Python re-scorer replays the same rules;
 * every arithmetic convention asserted here is one the agreement test holds both languages to.
 *
 * <p>Evidence tier: <b>T1 (analytical)</b>.
 */
public class SignalExtractorTest {

    private static final MotionResidualDiagnostics.Summary NO_RESIDUALS =
            MotionResidualDiagnostics.Summary.UNAVAILABLE;

    private static VoDiagnostics.Diagnostics counts(Integer track, Integer inlier) {
        return new VoDiagnostics.Diagnostics(track, inlier, null);
    }

    /** One ordinary valid frame with the given inlier count and log-scale increment. */
    private static SignalBlock feed(SignalExtractor x, int inliers, double incLogScale) {
        return x.extract(true, "none", counts(inliers + 20, inliers), NO_RESIDUALS, 3.0, 1.0, incLogScale);
    }

    @Test
    public void windowedSignalsAreAbsentUntilWarmup() {
        SignalExtractor x = new SignalExtractor(10, 5);
        for (int i = 0; i < 5; i++) {
            // Frames 0..4 push values 100..104; the buffer holds i entries when frame i computes,
            // so every one of these is below warm-up (m = 5).
            SignalBlock b = feed(x, 100 + i, 0.001);
            assertNull(b.relativeSupport(), "below warm-up at frame " + i);
            assertNull(b.incLogScaleDispersion(), "below warm-up at frame " + i);
        }
        // Frame 5 sees a 5-entry buffer: both windowed signals exist.
        SignalBlock b = feed(x, 200, 0.001);
        assertNotNull(b.relativeSupport());
        assertNotNull(b.incLogScaleDispersion());
    }

    @Test
    public void theReferenceIsStrictlyPastAndUsesTheFrozenMedianConvention() {
        SignalExtractor x = new SignalExtractor(10, 5);
        // Buffer after five valid frames: [100, 101, 102, 103, 104]; median = 102 (odd count).
        for (int i = 0; i < 5; i++) {
            feed(x, 100 + i, 0.0);
        }
        SignalBlock b = feed(x, 204, 0.0);
        // The current frame's own count (204) is NOT in its reference — strictly past.
        assertEquals(204.0 / 102.0, b.relativeSupport(), 0.0, "odd-count median, strictly past");

        // Buffer is now [100..104, 204] — six entries; median = (102 + 103) / 2.0 = 102.5.
        SignalBlock c = feed(x, 205, 0.0);
        assertEquals(205.0 / 102.5, c.relativeSupport(), 0.0, "even-count median = mean of middle two");
    }

    @Test
    public void dispersionMatchesTheTwoPassSampleSdConvention() {
        SignalExtractor x = new SignalExtractor(10, 5);
        double[] values = {0.001, 0.002, 0.003, 0.004, 0.005};
        for (double v : values) {
            feed(x, 100, v);
        }
        // Recompute exactly as frozen: left-to-right mean, then left-to-right squared deviations,
        // then sqrt(ssq / (n - 1)).
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        double mean = sum / values.length;
        double ssq = 0.0;
        for (double v : values) {
            double d = v - mean;
            ssq += d * d;
        }
        double expected = Math.sqrt(ssq / (values.length - 1));

        SignalBlock b = feed(x, 100, 0.001);
        assertEquals(expected, b.incLogScaleDispersion(), 0.0, "two-pass sample SD, bitwise");
    }

    @Test
    public void referenceChangesClearTheBuffersBeforeThisFrameComputes() {
        SignalExtractor x = new SignalExtractor(10, 5);
        for (int i = 0; i < 6; i++) {
            feed(x, 100, 0.001);
        }
        // Sanity: warmed up.
        assertNotNull(feed(x, 100, 0.001).relativeSupport());

        // A recenter moves the reference frame: the buffers are cleared FIRST, so this frame's
        // windowed signals are absent — a window never spans a reference_id boundary.
        SignalBlock atRecenter = x.extract(true, "recenter", counts(120, 100), NO_RESIDUALS, 3.0, 1.0, 0.001);
        assertNull(atRecenter.relativeSupport());
        assertNull(atRecenter.incLogScaleDispersion());

        // The recenter frame itself is valid and seeds the new segment's buffer; warm-up applies
        // again from there.
        for (int i = 0; i < 3; i++) {
            assertNull(feed(x, 100, 0.001).relativeSupport(), "still below warm-up after reset");
        }
        feed(x, 100, 0.001);
        assertNotNull(feed(x, 100, 0.001).relativeSupport(), "warmed up again after 5 valid frames");
    }

    @Test
    public void restartFramesResetAndAreNotPushed() {
        SignalExtractor x = new SignalExtractor(10, 5);
        for (int i = 0; i < 6; i++) {
            feed(x, 100, 0.001);
        }
        // The restart row's counts describe the re-initialised estimator (EXP-VO-001 R5a): the
        // buffers reset and the row is NOT pushed.
        x.extract(false, "restart", counts(2507, 2507), NO_RESIDUALS, 3.0, null, null);

        // Five valid frames must follow before a windowed signal exists — if the restart row had
        // been pushed, this would warm up one frame early.
        for (int i = 0; i < 4; i++) {
            assertNull(feed(x, 100, 0.001).relativeSupport(), "restart row must not seed the buffer");
        }
        feed(x, 100, 0.001);
        assertNotNull(feed(x, 100, 0.001).relativeSupport());
    }

    @Test
    public void initAndFailedFramesContributeNothing() {
        SignalExtractor x = new SignalExtractor(10, 5);
        // The init frame is never pushed (no motion was estimated against anything).
        x.extract(true, "init", counts(2507, 2507), NO_RESIDUALS, 3.0, null, null);
        // A failed ordinary frame is not pushed either.
        x.extract(false, "none", counts(40, 35), NO_RESIDUALS, 3.0, null, null);

        for (int i = 0; i < 4; i++) {
            assertNull(feed(x, 100, 0.001).relativeSupport());
        }
        feed(x, 100, 0.001);
        assertNotNull(feed(x, 100, 0.001).relativeSupport(),
                "exactly five valid pushes must be required, whatever preceded them");
    }

    @Test
    public void absentInputsAreSkippedNotZeroed() {
        SignalExtractor x = new SignalExtractor(10, 5);
        for (int i = 0; i < 5; i++) {
            // inlier count unavailable on these frames: nothing enters the support buffer.
            x.extract(true, "none", counts(null, null), NO_RESIDUALS, 3.0, 1.0, 0.001);
        }
        SignalBlock b = feed(x, 100, 0.001);
        assertNull(b.relativeSupport(), "an empty support buffer cannot produce a reference");
        // The log-scale buffer, by contrast, did fill — the two windows are independent.
        assertNotNull(b.incLogScaleDispersion());
    }

    @Test
    public void relativeSupportIsAbsentWhenTheCurrentCountIsAbsent() {
        SignalExtractor x = new SignalExtractor(10, 5);
        for (int i = 0; i < 6; i++) {
            feed(x, 100, 0.001);
        }
        SignalBlock b = x.extract(true, "none", counts(120, null), NO_RESIDUALS, 3.0, 1.0, 0.001);
        assertNull(b.relativeSupport(), "no numerator, no ratio — absence is not zero");
    }

    @Test
    public void aZeroMedianYieldsAbsenceNotInfinity() {
        SignalExtractor x = new SignalExtractor(10, 5);
        for (int i = 0; i < 5; i++) {
            feed(x, 0, 0.0);
        }
        SignalBlock b = feed(x, 10, 0.0);
        assertNull(b.relativeSupport(), "an undefined ratio is absent, never infinite");
    }

    @Test
    public void theWindowSlides() {
        SignalExtractor x = new SignalExtractor(10, 5);
        // Fill beyond W with ascending counts 100..114; the buffer must hold only the last 10.
        for (int i = 0; i < 15; i++) {
            feed(x, 100 + i, 0.0);
        }
        // Buffer = [105..114], median = (109 + 110) / 2.0 = 109.5.
        SignalBlock b = feed(x, 219, 0.0);
        assertEquals(219.0 / 109.5, b.relativeSupport(), 0.0, "only the most recent W frames count");
    }

    @Test
    public void degenerateWindowParametersAreRefused() {
        assertThrows(IllegalArgumentException.class, () -> new SignalExtractor(1, 1));
        assertThrows(IllegalArgumentException.class, () -> new SignalExtractor(10, 1));
        assertThrows(IllegalArgumentException.class, () -> new SignalExtractor(10, 11));
    }
}
