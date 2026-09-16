package org.boofcv.evaluation;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Streams {@code frames.csv} row by row and writes {@code manifest.json} on close, per
 * contracts/run-record.md. Frames are never buffered for the whole run: each
 * {@link #writeFrame(FrameEstimateRow)} call writes and flushes immediately, so capture memory
 * use does not grow with run length.
 */
public final class RunRecordWriter implements Closeable {

    private static final String[] HEADER = {
            "frame_index", "timestamp_s", "est_x", "est_y", "est_z", "est_yaw_deg", "success", "event",
            "reference_id", "h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22",
            "track_count", "inlier_count", "process_time_ns"
    };

    /** Columns 22–37 of the v1.1.0 layout (contracts/run-record-v1.1.md), appended after column 21. */
    private static final String[] CONFIDENCE_HEADER = {
            "residual_inlier_count", "residual_mean_sq_px", "residual_rms_px", "residual_median_sq_px",
            "residual_max_sq_px", "inlier_threshold_sq_px", "inlier_coverage", "keyframe_age",
            "relative_support", "inc_flow_px", "inc_log_scale", "inc_log_scale_dispersion",
            "confidence_outcome", "confidence_reason", "confidence_score", "confidence_time_ns"
    };

    private final Path runDir;
    private final RunManifest manifest;
    private final int frameCount;
    private final BufferedWriter framesWriter;
    private final boolean confidenceColumns;

    private int processedCount = 0;
    private int lastFrameIndex = Integer.MIN_VALUE;
    private boolean completed = true;
    private boolean closed = false;

    /**
     * @param runDir     directory to write {@code manifest.json} and {@code frames.csv} into; created if absent
     * @param manifest   manifest fields known before capture starts; {@code frameCount},
     *                   {@code processedCount} and {@code completed} are overwritten on {@link #close()}
     * @param frameCount total frames in the dataset being processed, for the manifest's {@code frame_count}
     */
    public RunRecordWriter(Path runDir, RunManifest manifest, int frameCount) throws IOException {
        this(runDir, manifest, frameCount, false);
    }

    /**
     * @param confidenceColumns when true, the writer emits the v1.1.0 layout — the sixteen
     *                          confidence columns appended after {@code process_time_ns} — and
     *                          every row must carry a {@link FrameEstimateRow.Confidence}. When
     *                          false the output is byte-for-byte the v1.0.0 layout, so a
     *                          capture-off run is indistinguishable from a pre-confidence one.
     */
    public RunRecordWriter(Path runDir, RunManifest manifest, int frameCount,
                           boolean confidenceColumns) throws IOException {
        this.runDir = runDir;
        this.manifest = manifest;
        this.frameCount = frameCount;
        this.confidenceColumns = confidenceColumns;
        Files.createDirectories(runDir);
        this.framesWriter = Files.newBufferedWriter(runDir.resolve("frames.csv"), StandardCharsets.UTF_8);
        framesWriter.write(String.join(",", HEADER));
        if (confidenceColumns) {
            framesWriter.write(",");
            framesWriter.write(String.join(",", CONFIDENCE_HEADER));
        }
        framesWriter.write("\n");
    }

    /** Marks the run as interrupted; {@link #close()} will record {@code completed: false}. */
    public void markIncomplete() {
        this.completed = false;
    }

    public void writeFrame(FrameEstimateRow row) throws IOException {
        if (closed) {
            throw new IllegalStateException("RunRecordWriter is closed");
        }
        if (row.frameIndex <= lastFrameIndex) {
            throw new IllegalArgumentException(
                    "frame_index must be strictly increasing: " + row.frameIndex + " after " + lastFrameIndex);
        }
        lastFrameIndex = row.frameIndex;

        StringBuilder line = new StringBuilder();
        line.append(row.frameIndex).append(',');
        line.append(field(row.timestampS)).append(',');
        line.append(field(row.estX)).append(',');
        line.append(field(row.estY)).append(',');
        line.append(field(row.estZ)).append(',');
        line.append(field(row.estYawDeg)).append(',');
        line.append(row.success ? "true" : "false").append(',');
        line.append(row.event).append(',');
        line.append(row.referenceId).append(',');
        line.append(field(row.h00)).append(',');
        line.append(field(row.h01)).append(',');
        line.append(field(row.h02)).append(',');
        line.append(field(row.h10)).append(',');
        line.append(field(row.h11)).append(',');
        line.append(field(row.h12)).append(',');
        line.append(field(row.h20)).append(',');
        line.append(field(row.h21)).append(',');
        line.append(field(row.h22)).append(',');
        line.append(field(row.trackCount)).append(',');
        line.append(field(row.inlierCount)).append(',');
        if (row.processTimeNs < 0) {
            throw new IllegalArgumentException("process_time_ns must be >= 0, got " + row.processTimeNs);
        }
        line.append(row.processTimeNs);

        if (confidenceColumns) {
            FrameEstimateRow.Confidence c = row.confidence;
            if (c == null) {
                // SC-001: every frame of a confidence run carries a verdict; a missing one would be
                // indistinguishable from a lost record.
                throw new IllegalArgumentException(
                        "confidence columns enabled but frame " + row.frameIndex + " carries no confidence");
            }
            validateConfidence(row, c);
            line.append(',').append(field(c.residualInlierCount));
            line.append(',').append(field(c.residualMeanSqPx));
            line.append(',').append(field(c.residualRmsPx));
            line.append(',').append(field(c.residualMedianSqPx));
            line.append(',').append(field(c.residualMaxSqPx));
            line.append(',').append(field(c.inlierThresholdSqPx));
            line.append(',').append(field(c.inlierCoverage));
            line.append(',').append(field(c.keyframeAge));
            line.append(',').append(field(c.relativeSupport));
            line.append(',').append(field(c.incFlowPx));
            line.append(',').append(field(c.incLogScale));
            line.append(',').append(field(c.incLogScaleDispersion));
            line.append(',').append(c.outcome);
            line.append(',').append(c.reason);
            line.append(',').append(field(c.score));
            line.append(',').append(c.confidenceTimeNs);
        } else if (row.confidence != null) {
            throw new IllegalArgumentException(
                    "frame " + row.frameIndex + " carries confidence but the writer is in v1.0.0 mode");
        }

        framesWriter.write(line.toString());
        framesWriter.write("\n");
        framesWriter.flush();
        processedCount++;
    }

    /**
     * The v1.1.0 writer-side invariants (contracts/run-record-v1.1.md §Reader obligations,
     * enforced at the producer too so a malformed record is never written in the first place).
     */
    private static void validateConfidence(FrameEstimateRow row, FrameEstimateRow.Confidence c) {
        if (c.outcome == null || c.outcome.isBlank() || c.reason == null || c.reason.isBlank()) {
            throw new IllegalArgumentException(
                    "confidence_outcome and confidence_reason are required on frame " + row.frameIndex);
        }
        boolean scoreless = "rejected".equals(c.outcome) || "not_produced".equals(c.outcome);
        if (scoreless != (c.score == null)) {
            throw new IllegalArgumentException("confidence_score must be empty exactly when outcome is "
                    + "rejected/not_produced; frame " + row.frameIndex + " has outcome " + c.outcome
                    + " with score " + c.score);
        }
        if ("not_produced".equals(c.outcome) != !row.success) {
            throw new IllegalArgumentException("not_produced must coincide with success == false; frame "
                    + row.frameIndex + " has outcome " + c.outcome + " with success " + row.success);
        }
        if (c.residualMeanSqPx != null && (c.residualInlierCount == null || c.residualInlierCount < 1)) {
            throw new IllegalArgumentException("residual statistics present without a positive "
                    + "residual_inlier_count on frame " + row.frameIndex);
        }
        if (c.confidenceTimeNs < 0) {
            throw new IllegalArgumentException(
                    "confidence_time_ns must be >= 0, got " + c.confidenceTimeNs);
        }
    }

    /** Required, never-empty float field: NaN/Inf must never reach the file (contracts/run-record.md). */
    private static String field(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            throw new IllegalArgumentException("NaN/Inf is not a valid field value: " + value);
        }
        return Double.toString(value);
    }

    /** Optional float field: empty means "not available"; a present value must still not be NaN/Inf. */
    private static String field(Double value) {
        if (value == null) {
            return "";
        }
        return field(value.doubleValue());
    }

    /** Optional int field: empty means "not available", never zero. */
    private static String field(Integer value) {
        return value == null ? "" : value.toString();
    }

    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        framesWriter.close();

        manifest.frameCount = frameCount;
        manifest.processedCount = processedCount;
        manifest.completed = completed;

        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        mapper.writeValue(runDir.resolve("manifest.json").toFile(), manifest);
        closed = true;
    }
}
