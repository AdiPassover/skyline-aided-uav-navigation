package org.boofcv.confidence;

import javax.annotation.Nullable;

/**
 * The raw per-frame diagnostics a confidence verdict is computed from (feature spec FR-004).
 *
 * <p>This is the feature's durable artifact. Verdicts are re-derivable; signals are not. They are
 * retained verbatim so that <em>any future calibration can be applied to past runs</em> — which is
 * why {@code DEC-CONF-001} names the persisted signal block, not the scorer, as this subsystem's
 * isolating interface.
 *
 * <h2>Absence is a value</h2>
 *
 * <p>Every field is independently nullable, and {@code null} means <b>not available</b>, never
 * zero. The distinction is load-bearing rather than fastidious: a missing residual read as
 * {@code 0.0} would look like a <i>perfect fit</i>, and a missing track count read as {@code 0}
 * would look like total tracking failure. Both would be confident statements about a frame nobody
 * measured. This follows the precedent {@code VoDiagnostics} already sets — "absence of this
 * diagnostic must never look like a measurement of zero".
 *
 * <h2>Signals are defined by meaning, not by their source</h2>
 *
 * <p>Nothing here names a BoofCV type or a BoofCV concept. A different front end could populate the
 * same block, which is the whole of Principle VII for this subsystem. The BoofCV-specific
 * extraction lives in one adapter class.
 *
 * @param trackCount           features the tracker is maintaining this frame
 * @param inlierCount          inliers counted through the tracker's per-track inlier flags
 * @param residualInlierCount  size of the estimator's RANSAC match set. {@code 0} is a real
 *                             measurement (estimation succeeded over an empty match set) and is
 *                             distinct from {@code null}
 * @param residualMeanSqPx     mean squared reprojection residual, input-image px²
 * @param residualRmsPx        RMS residual, input-image px (linear, not squared)
 * @param residualMedianSqPx   median squared residual, px²
 * @param residualMaxSqPx      largest squared residual in the match set, px²
 * @param inlierThresholdSqPx  configured RANSAC inlier threshold, px². Configuration, not a
 *                             measurement — carried so residual magnitudes stay interpretable
 *                             without re-deriving the estimator's configuration (FR-033)
 * @param inlierCoverage       spatial coverage of inliers over the frame, {@code [0,1]} — the
 *                             frozen 8×8 grid-occupancy definition ({@code EXP-CONF-001} P1)
 * @param keyframeAge          frames since the last keyframe change. <b>Always {@code null}
 *                             today</b> — see {@link #keyframeAge()}
 * @param relativeSupport      {@code inlier_count / running_ref}, where {@code running_ref} is the
 *                             median inlier count over the previous {@code W} valid frames of the
 *                             current reference segment ({@code EXP-CONF-001} A4 semantics; the
 *                             temporal state lives in {@link SignalExtractor}, never here — this
 *                             is the computed value, so a persisted row stays self-contained)
 * @param incFlowPx            per-frame image-centre displacement magnitude, input-image px — the
 *                             KLT operating-envelope variable, deliberately raw (amendment A3)
 * @param incLogScaleDispersion sample SD of {@code inc_log_scale} over the previous {@code W}
 *                             valid frames (A4 semantics; scene non-planarity indicator,
 *                             {@code EXP-VO-011} R2)
 */
public record SignalBlock(
        @Nullable Integer trackCount,
        @Nullable Integer inlierCount,
        @Nullable Integer residualInlierCount,
        @Nullable Double residualMeanSqPx,
        @Nullable Double residualRmsPx,
        @Nullable Double residualMedianSqPx,
        @Nullable Double residualMaxSqPx,
        @Nullable Double inlierThresholdSqPx,
        @Nullable Double inlierCoverage,
        @Nullable Integer keyframeAge,
        @Nullable Double relativeSupport,
        @Nullable Double incFlowPx,
        @Nullable Double incLogScaleDispersion) {

    /** Nothing could be read for this frame. Distinct from every field measuring zero. */
    public static final SignalBlock UNAVAILABLE =
            new SignalBlock(null, null, null, null, null, null, null, null, null, null,
                    null, null, null);

    /**
     * Canonical signal names, as used in the calibration configuration's {@code required_signals},
     * {@code rejection_rules} and {@code score_terms}, and as column names in the run record.
     *
     * <p>Kept as constants rather than string literals so that a typo is a compile error on this
     * side and a load-time refusal on the configuration side, rather than a silently ignored term
     * that changes a score.
     */
    public static final String TRACK_COUNT = "track_count";
    public static final String INLIER_COUNT = "inlier_count";
    public static final String INLIER_RATIO = "inlier_ratio";
    public static final String RESIDUAL_INLIER_COUNT = "residual_inlier_count";
    public static final String RESIDUAL_MEAN_SQ_PX = "residual_mean_sq_px";
    public static final String RESIDUAL_RMS_PX = "residual_rms_px";
    public static final String RESIDUAL_MEDIAN_SQ_PX = "residual_median_sq_px";
    public static final String RESIDUAL_MAX_SQ_PX = "residual_max_sq_px";
    public static final String INLIER_THRESHOLD_SQ_PX = "inlier_threshold_sq_px";
    public static final String INLIER_COVERAGE = "inlier_coverage";
    public static final String KEYFRAME_AGE = "keyframe_age";
    public static final String RELATIVE_SUPPORT = "relative_support";
    public static final String INC_FLOW_PX = "inc_flow_px";
    public static final String INC_LOG_SCALE_DISPERSION = "inc_log_scale_dispersion";

    /** Compact constructor validating the invariants in {@code data-model.md} §1.4. */
    public SignalBlock {
        requireNonNegative(trackCount, TRACK_COUNT);
        requireNonNegative(inlierCount, INLIER_COUNT);
        requireNonNegative(residualInlierCount, RESIDUAL_INLIER_COUNT);
        requireNonNegative(keyframeAge, KEYFRAME_AGE);

        requireFinite(residualMeanSqPx, RESIDUAL_MEAN_SQ_PX, true);
        requireFinite(residualRmsPx, RESIDUAL_RMS_PX, true);
        requireFinite(residualMedianSqPx, RESIDUAL_MEDIAN_SQ_PX, true);
        requireFinite(residualMaxSqPx, RESIDUAL_MAX_SQ_PX, true);
        requireFinite(inlierThresholdSqPx, INLIER_THRESHOLD_SQ_PX, true);
        requireFinite(inlierCoverage, INLIER_COVERAGE, true);
        requireFinite(relativeSupport, RELATIVE_SUPPORT, true);
        requireFinite(incFlowPx, INC_FLOW_PX, true);
        requireFinite(incLogScaleDispersion, INC_LOG_SCALE_DISPERSION, true);

        if (inlierCoverage != null && inlierCoverage > 1.0) {
            throw new IllegalArgumentException(INLIER_COVERAGE + " must be in [0,1], got " + inlierCoverage);
        }
        // A statistic without a population is incoherent: it would claim a mean over nothing.
        if (residualMeanSqPx != null && (residualInlierCount == null || residualInlierCount < 1)) {
            throw new IllegalArgumentException(
                    "residual statistics present but residual_inlier_count is " + residualInlierCount
                            + "; a mean over an empty or unknown match set is not a measurement");
        }
    }

    /**
     * Inlier ratio, <b>derived and never stored</b>.
     *
     * <p>The run-record contract already sets this precedent ("Ratio is derived, never stored").
     * Storing it would create a second place for the same fact to be wrong, and the two could then
     * disagree in a persisted file with no way to tell which was right.
     *
     * <p>Returns {@code null} when either operand is absent, or when {@code trackCount} is zero —
     * an undefined ratio is absent, not zero.
     */
    @Nullable
    public Double inlierRatio() {
        if (trackCount == null || inlierCount == null || trackCount == 0) {
            return null;
        }
        return (double) inlierCount / (double) trackCount;
    }

    /**
     * Keyframe age, currently <b>always absent</b>.
     *
     * <p>Keyframe state is not observable from outside BoofCV's motion wrapper: its {@code alg}
     * field is package-private and no method on its public surface reports it. The field is
     * nonetheless modelled, for two reasons — when the VO lane supplies it, no schema change is
     * needed; and, more importantly, its absence is <em>visible in every run</em>, so the confound
     * it represents cannot be quietly forgotten.
     */
    @Nullable
    @Override
    public Integer keyframeAge() {
        return keyframeAge;
    }

    /**
     * Looks up a signal by its canonical name, returning {@code null} when absent.
     *
     * <p>Used by the scorer so that a calibration can name signals in configuration. Throws on an
     * unknown name rather than returning {@code null}: a typo must not be indistinguishable from a
     * genuinely absent measurement.
     */
    @Nullable
    public Double signal(String name) {
        return switch (name) {
            case TRACK_COUNT -> toDouble(trackCount);
            case INLIER_COUNT -> toDouble(inlierCount);
            case INLIER_RATIO -> inlierRatio();
            case RESIDUAL_INLIER_COUNT -> toDouble(residualInlierCount);
            case RESIDUAL_MEAN_SQ_PX -> residualMeanSqPx;
            case RESIDUAL_RMS_PX -> residualRmsPx;
            case RESIDUAL_MEDIAN_SQ_PX -> residualMedianSqPx;
            case RESIDUAL_MAX_SQ_PX -> residualMaxSqPx;
            case INLIER_THRESHOLD_SQ_PX -> inlierThresholdSqPx;
            case INLIER_COVERAGE -> inlierCoverage;
            case KEYFRAME_AGE -> toDouble(keyframeAge);
            case RELATIVE_SUPPORT -> relativeSupport;
            case INC_FLOW_PX -> incFlowPx;
            case INC_LOG_SCALE_DISPERSION -> incLogScaleDispersion;
            default -> throw new IllegalArgumentException("Unknown signal name: '" + name + "'");
        };
    }

    /** Whether a signal by this name is present. Throws on an unknown name, as {@link #signal}. */
    public boolean hasSignal(String name) {
        return signal(name) != null;
    }

    /** Whether every field is absent. */
    public boolean isEmpty() {
        return equals(UNAVAILABLE);
    }

    @Nullable
    private static Double toDouble(@Nullable Integer v) {
        return v == null ? null : (double) v;
    }

    private static void requireNonNegative(@Nullable Integer v, String name) {
        if (v != null && v < 0) {
            throw new IllegalArgumentException(name + " must be >= 0, got " + v);
        }
    }

    private static void requireFinite(@Nullable Double v, String name, boolean nonNegative) {
        if (v == null) {
            return;
        }
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            // Matches the run record's encoding rule: NaN/Inf are invalid values, not a way to
            // spell absence. Absence has its own representation and it is null.
            throw new IllegalArgumentException(name + " must be finite, got " + v);
        }
        if (nonNegative && v < 0.0) {
            throw new IllegalArgumentException(name + " must be >= 0, got " + v);
        }
    }

    /** Builder, because ten nullable fields in a positional constructor is a transposition waiting to happen. */
    public static Builder builder() {
        return new Builder();
    }

    /** Mutable builder for {@link SignalBlock}; every field defaults to absent. */
    public static final class Builder {
        private Integer trackCount;
        private Integer inlierCount;
        private Integer residualInlierCount;
        private Double residualMeanSqPx;
        private Double residualRmsPx;
        private Double residualMedianSqPx;
        private Double residualMaxSqPx;
        private Double inlierThresholdSqPx;
        private Double inlierCoverage;
        private Integer keyframeAge;
        private Double relativeSupport;
        private Double incFlowPx;
        private Double incLogScaleDispersion;

        public Builder trackCount(@Nullable Integer v) { this.trackCount = v; return this; }
        public Builder inlierCount(@Nullable Integer v) { this.inlierCount = v; return this; }
        public Builder residualInlierCount(@Nullable Integer v) { this.residualInlierCount = v; return this; }
        public Builder residualMeanSqPx(@Nullable Double v) { this.residualMeanSqPx = v; return this; }
        public Builder residualRmsPx(@Nullable Double v) { this.residualRmsPx = v; return this; }
        public Builder residualMedianSqPx(@Nullable Double v) { this.residualMedianSqPx = v; return this; }
        public Builder residualMaxSqPx(@Nullable Double v) { this.residualMaxSqPx = v; return this; }
        public Builder inlierThresholdSqPx(@Nullable Double v) { this.inlierThresholdSqPx = v; return this; }
        public Builder inlierCoverage(@Nullable Double v) { this.inlierCoverage = v; return this; }
        public Builder keyframeAge(@Nullable Integer v) { this.keyframeAge = v; return this; }
        public Builder relativeSupport(@Nullable Double v) { this.relativeSupport = v; return this; }
        public Builder incFlowPx(@Nullable Double v) { this.incFlowPx = v; return this; }
        public Builder incLogScaleDispersion(@Nullable Double v) { this.incLogScaleDispersion = v; return this; }

        public SignalBlock build() {
            return new SignalBlock(trackCount, inlierCount, residualInlierCount, residualMeanSqPx,
                    residualRmsPx, residualMedianSqPx, residualMaxSqPx, inlierThresholdSqPx,
                    inlierCoverage, keyframeAge, relativeSupport, incFlowPx, incLogScaleDispersion);
        }
    }
}
