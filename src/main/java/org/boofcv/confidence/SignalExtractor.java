package org.boofcv.confidence;

import org.boofcv.evaluation.VoDiagnostics;
import org.boofcv.stitching.MotionResidualDiagnostics;

import javax.annotation.Nullable;
import java.util.ArrayDeque;
import java.util.Arrays;

/**
 * Builds the per-frame {@link SignalBlock} from the frozen VO's observational surfaces, and owns
 * the pre-registered temporal state ({@code EXP-CONF-001} §Temporal signal state semantics,
 * amendment A4).
 *
 * <p><b>This is the only class in {@code org.boofcv.confidence} permitted to name a VO type</b>
 * (tasks.md T020): it consumes {@link MotionResidualDiagnostics.Summary} and
 * {@link VoDiagnostics.Diagnostics}, both VO/evaluation-owned interfaces carrying no BoofCV type.
 * One recorded deviation from T021's letter: the coverage statistic is <i>computed</i> in
 * {@link VoDiagnostics} (where the BoofCV cast already lives, honouring this package's no-BoofCV
 * charter) and consumed here as a value.
 *
 * <h2>Temporal state semantics (frozen; the Python re-scorer replays these rules bit-for-bit)</h2>
 *
 * <ul>
 *   <li><b>Validity:</b> a frame's values enter the ring buffers iff the estimator succeeded on it
 *       and its event is neither {@code init} nor {@code restart} (a restart row's counts describe
 *       the re-initialised estimator, {@code EXP-VO-001} R5a). {@code recenter} rows are valid.</li>
 *   <li><b>Reset:</b> on a reference change ({@code restart} or {@code recenter}) the buffers are
 *       cleared <i>before</i> this frame's windowed signals are computed — windows never span a
 *       {@code reference_id} boundary.</li>
 *   <li><b>Strictly past:</b> the current frame's own values are pushed only <i>after</i> its
 *       windowed signals are computed, so the reference never contains the value it normalises.</li>
 *   <li><b>Warm-up:</b> each windowed signal is absent ({@code null}) until its buffer holds at
 *       least {@code m} entries. Never zero, never a default.</li>
 *   <li><b>Median</b> (relative support reference): sort a copy ascending; odd count — the middle
 *       value; even count — the mean of the two middle values, computed as
 *       {@code (a + b) / 2.0} in IEEE-754 double (exact for the integer counts involved).</li>
 *   <li><b>Sample SD</b> (scale dispersion): two-pass over the buffer in insertion order (oldest
 *       first) — {@code mean = (Σ x_i) / n} accumulated left-to-right, then
 *       {@code ssq = Σ (x_i − mean)²} left-to-right, then {@code sqrt(ssq / (n − 1))}.
 *       {@code sqrt} is exactly rounded in IEEE-754, so both languages agree bitwise.</li>
 * </ul>
 *
 * <p>Absent inputs contribute nothing: a valid frame whose {@code inlier_count} could not be read
 * pushes nothing into the support buffer (and likewise for {@code inc_log_scale}), per the
 * absence-is-not-zero rule.
 *
 * <p><b>Not thread-safe; one instance per run.</b> State is deliberately confined here so that
 * {@link ConfidenceScorer} stays a pure function and {@code DEC-CONF-001}'s offline re-scoring
 * remains possible from persisted rows alone.
 */
public final class SignalExtractor {

    /** Event names, mirrored from the run-record contract rather than imported. */
    public static final String EVENT_INIT = "init";
    public static final String EVENT_RESTART = "restart";
    public static final String EVENT_RECENTER = "recenter";

    private final int windowW;
    private final int warmupM;

    private final ArrayDeque<Integer> supportWindow;
    private final ArrayDeque<Double> logScaleWindow;

    /**
     * @param windowW window length {@code W} (frozen at 10 for the shipped calibration, but a
     *                parameter here because {@code W} is part of the calibration identity)
     * @param warmupM minimum buffered entries {@code m} before a windowed signal exists
     */
    public SignalExtractor(int windowW, int warmupM) {
        if (windowW < 2 || warmupM < 2 || warmupM > windowW) {
            throw new IllegalArgumentException(
                    "require 2 <= m <= W (sample SD needs n >= 2); got W=" + windowW + ", m=" + warmupM);
        }
        this.windowW = windowW;
        this.warmupM = warmupM;
        this.supportWindow = new ArrayDeque<>(windowW);
        this.logScaleWindow = new ArrayDeque<>(windowW);
    }

    public int windowW() {
        return windowW;
    }

    public int warmupM() {
        return warmupM;
    }

    /**
     * Extracts one frame's signal block and advances the temporal state.
     *
     * <p>Call exactly once per processed frame, in frame order, after {@code processFrame} and
     * before the next frame is submitted (FR-034 — the residual summary is only valid in that
     * window). The caller passes {@code incFlowPx}/{@code incLogScale} only for frames carrying a
     * genuine accepted increment ({@code success} and neither {@code init} nor {@code restart});
     * on other frames they must be {@code null}, because no increment exists to describe.
     *
     * @param estimatorSucceeded the estimator's own frame-update result
     * @param event              {@code init} | {@code none} | {@code recenter} | {@code restart}
     * @param diagnostics        track/inlier counts and coverage; may be entirely absent
     * @param residuals          the residual summary for this frame; may be
     *                           {@link MotionResidualDiagnostics.Summary#UNAVAILABLE}
     * @param inlierThresholdSqPx the configured RANSAC threshold, px² (configuration, FR-033)
     * @param incFlowPx          this frame's image-centre displacement, px, or {@code null}
     * @param incLogScale        this frame's {@code log √|det J|}, or {@code null}
     */
    public SignalBlock extract(boolean estimatorSucceeded, String event,
                               VoDiagnostics.Diagnostics diagnostics,
                               MotionResidualDiagnostics.Summary residuals,
                               @Nullable Double inlierThresholdSqPx,
                               @Nullable Double incFlowPx,
                               @Nullable Double incLogScale) {
        boolean referenceChanged = EVENT_RESTART.equals(event) || EVENT_RECENTER.equals(event);
        if (referenceChanged) {
            supportWindow.clear();
            logScaleWindow.clear();
        }

        Integer inlierCount = diagnostics.inlierCount();
        Double relativeSupport = computeRelativeSupport(inlierCount);
        Double dispersion = computeDispersion();

        SignalBlock block = new SignalBlock(
                diagnostics.trackCount(),
                inlierCount,
                residuals.inlierCount(),
                residuals.meanSquaredPx(),
                residuals.rmsPx(),
                residuals.medianSquaredPx(),
                residuals.maxSquaredPx(),
                inlierThresholdSqPx,
                diagnostics.inlierCoverage(),
                null,               // keyframe_age: not observable (FR-032)
                relativeSupport,
                incFlowPx,
                dispersion);

        boolean valid = estimatorSucceeded
                && !EVENT_INIT.equals(event) && !EVENT_RESTART.equals(event);
        if (valid) {
            if (inlierCount != null) {
                if (supportWindow.size() == windowW) {
                    supportWindow.removeFirst();
                }
                supportWindow.addLast(inlierCount);
            }
            if (incLogScale != null && !incLogScale.isNaN() && !incLogScale.isInfinite()) {
                if (logScaleWindow.size() == windowW) {
                    logScaleWindow.removeFirst();
                }
                logScaleWindow.addLast(incLogScale);
            }
        }
        return block;
    }

    @Nullable
    private Double computeRelativeSupport(@Nullable Integer inlierCount) {
        if (inlierCount == null || supportWindow.size() < warmupM) {
            return null;
        }
        double ref = median(supportWindow);
        if (ref == 0.0) {
            return null;      // an undefined ratio is absent, not infinite and not zero
        }
        return inlierCount / ref;
    }

    /** Median per the frozen convention; {@code (a + b) / 2.0} for even counts. */
    private static double median(ArrayDeque<Integer> window) {
        int n = window.size();
        int[] sorted = new int[n];
        int i = 0;
        for (int v : window) {
            sorted[i++] = v;
        }
        Arrays.sort(sorted);
        if ((n & 1) == 1) {
            return sorted[n / 2];
        }
        return (sorted[n / 2 - 1] + (double) sorted[n / 2]) / 2.0;
    }

    @Nullable
    private Double computeDispersion() {
        int n = logScaleWindow.size();
        if (n < warmupM) {
            return null;
        }
        double sum = 0.0;
        for (double v : logScaleWindow) {
            sum += v;
        }
        double mean = sum / n;
        double ssq = 0.0;
        for (double v : logScaleWindow) {
            double d = v - mean;
            ssq += d * d;
        }
        return Math.sqrt(ssq / (n - 1));
    }
}
