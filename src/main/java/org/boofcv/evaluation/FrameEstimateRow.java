package org.boofcv.evaluation;

/**
 * One {@code frames.csv} row, all 21 v1.0.0 columns (contracts/run-record.md).
 *
 * <p>Boxed types ({@link Double}, {@link Integer}) mark columns where the contract allows an
 * empty field: {@code null} here means "not available", which {@link RunRecordWriter} must
 * encode as an empty CSV field, never as zero. {@code estZ} is always {@code null} until height
 * estimation exists (FR-015); the homography and diagnostic columns are {@code null} only when
 * genuinely unavailable.
 */
public final class FrameEstimateRow {

    public final int frameIndex;
    public final double timestampS;
    public final double estX;
    public final double estY;
    public final Double estZ;
    /** Canonical sense and range: clockwise-positive degrees, [0, 360). Datum is the start frame. */
    public final double estYawDeg;
    public final boolean success;
    /** One of {@code none}, {@code init}, {@code recenter}, {@code restart}. */
    public final String event;
    public final int referenceId;

    // Row-major current-frame-to-mosaic homography.
    public final Double h00;
    public final Double h01;
    public final Double h02;
    public final Double h10;
    public final Double h11;
    public final Double h12;
    public final Double h20;
    public final Double h21;
    public final Double h22;

    public final Integer trackCount;
    public final Integer inlierCount;
    public final long processTimeNs;

    /**
     * The v1.1.0 confidence columns (contracts/run-record-v1.1.md), or {@code null} for a v1.0.0
     * row. {@code null} here means the run was captured without confidence — the writer then emits
     * the v1.0.0 layout byte-for-byte, so a capture-off run is indistinguishable from a
     * pre-confidence one.
     */
    public final Confidence confidence;

    /**
     * Columns 22–37 of a v1.1.0 row. Boxed types mark empty-allowed columns; {@code null} is
     * encoded as an empty field, never zero. {@code outcome}/{@code reason} are the lowercase wire
     * names; {@code score} is empty exactly when the outcome is {@code rejected} or
     * {@code not_produced}.
     */
    public static final class Confidence {
        public final Integer residualInlierCount;
        public final Double residualMeanSqPx;
        public final Double residualRmsPx;
        public final Double residualMedianSqPx;
        public final Double residualMaxSqPx;
        public final Double inlierThresholdSqPx;
        public final Double inlierCoverage;
        /** Always {@code null} today — specified-and-empty (contracts/run-record-v1.1.md). */
        public final Integer keyframeAge;
        public final Double relativeSupport;
        public final Double incFlowPx;
        /** This frame's raw {@code log √|det J|}; persisted so the dispersion is replayable offline. */
        public final Double incLogScale;
        public final Double incLogScaleDispersion;
        public final String outcome;
        public final String reason;
        public final Double score;
        public final long confidenceTimeNs;

        public Confidence(Integer residualInlierCount, Double residualMeanSqPx, Double residualRmsPx,
                          Double residualMedianSqPx, Double residualMaxSqPx, Double inlierThresholdSqPx,
                          Double inlierCoverage, Integer keyframeAge, Double relativeSupport,
                          Double incFlowPx, Double incLogScale, Double incLogScaleDispersion,
                          String outcome, String reason, Double score, long confidenceTimeNs) {
            this.residualInlierCount = residualInlierCount;
            this.residualMeanSqPx = residualMeanSqPx;
            this.residualRmsPx = residualRmsPx;
            this.residualMedianSqPx = residualMedianSqPx;
            this.residualMaxSqPx = residualMaxSqPx;
            this.inlierThresholdSqPx = inlierThresholdSqPx;
            this.inlierCoverage = inlierCoverage;
            this.keyframeAge = keyframeAge;
            this.relativeSupport = relativeSupport;
            this.incFlowPx = incFlowPx;
            this.incLogScale = incLogScale;
            this.incLogScaleDispersion = incLogScaleDispersion;
            this.outcome = outcome;
            this.reason = reason;
            this.score = score;
            this.confidenceTimeNs = confidenceTimeNs;
        }
    }

    public FrameEstimateRow(int frameIndex, double timestampS, double estX, double estY, Double estZ,
                             double estYawDeg, boolean success, String event, int referenceId,
                             Double h00, Double h01, Double h02,
                             Double h10, Double h11, Double h12,
                             Double h20, Double h21, Double h22,
                             Integer trackCount, Integer inlierCount, long processTimeNs) {
        this(frameIndex, timestampS, estX, estY, estZ, estYawDeg, success, event, referenceId,
                h00, h01, h02, h10, h11, h12, h20, h21, h22,
                trackCount, inlierCount, processTimeNs, null);
    }

    public FrameEstimateRow(int frameIndex, double timestampS, double estX, double estY, Double estZ,
                             double estYawDeg, boolean success, String event, int referenceId,
                             Double h00, Double h01, Double h02,
                             Double h10, Double h11, Double h12,
                             Double h20, Double h21, Double h22,
                             Integer trackCount, Integer inlierCount, long processTimeNs,
                             Confidence confidence) {
        this.frameIndex = frameIndex;
        this.timestampS = timestampS;
        this.estX = estX;
        this.estY = estY;
        this.estZ = estZ;
        this.estYawDeg = estYawDeg;
        this.success = success;
        this.event = event;
        this.referenceId = referenceId;
        this.h00 = h00;
        this.h01 = h01;
        this.h02 = h02;
        this.h10 = h10;
        this.h11 = h11;
        this.h12 = h12;
        this.h20 = h20;
        this.h21 = h21;
        this.h22 = h22;
        this.trackCount = trackCount;
        this.inlierCount = inlierCount;
        this.processTimeNs = processTimeNs;
        this.confidence = confidence;
    }
}
