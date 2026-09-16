package org.boofcv.stitching.diagnostics;

import boofcv.struct.image.GrayF32;
import org.boofcv.evaluation.FrameSource;
import org.boofcv.evaluation.VoDiagnostics;
import org.boofcv.stitching.InstrumentedStitching;
import org.boofcv.stitching.MotionResidualDiagnostics;
import org.boofcv.stitching.StitchingEstimator;
import org.boofcv.util.structs.Pose3D;

import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Drives the <em>instrumented</em> stitching estimator over a frame source and records, per frame,
 * the estimator-native residual diagnostics alongside the estimator state needed to interpret when
 * each measurement was produced. The measurement harness for experiment {@code EXP-VO-001}.
 *
 * <h2>What this is, and is not</h2>
 *
 * <p><b>Is:</b> an observational read of {@link MotionResidualDiagnostics} from the same estimator
 * instance that produces the pose written in the same row, plus the per-frame event classification
 * that says which estimator state the measurement was taken in.
 *
 * <p><b>Is not:</b> a scorer. No threshold is compared, no verdict is formed, no value is
 * calibrated, and no claim is made that any residual predicts anything. That belongs to the
 * estimator-health layer; this class must stay free of it.
 *
 * <h2>Relationship to {@code VoRunner}</h2>
 *
 * <p>The event classification below is a deliberate duplicate of {@code org.boofcv.evaluation
 * .VoRunner}'s, not a shared call: that class sits on the CONF lane's claimed capture-side surface
 * and is not VO's to modify. The duplication is made safe by measurement rather than by care —
 * when a {@code reference_run_dir} is configured, {@link ResidualProbeApp} checks this runner's
 * poses and events against the committed run record row by row, so any divergence from
 * {@code VoRunner} surfaces as a failed reproduction rather than as a quietly different number.
 *
 * <h2>Absence</h2>
 *
 * <p>Every residual column is written empty when the estimator reported no measurement, and empty
 * never means zero. Two distinct absences are preserved and stay distinguishable in the output: an
 * absent {@code residual_inlier_count} means no motion was estimated at all, whereas a
 * {@code residual_inlier_count} of {@code 0} alongside empty statistics means motion was estimated
 * over an empty match set.
 */
public final class ResidualProbeRunner {

    /**
     * Column header of the per-frame residual record this probe writes.
     *
     * <p>The five {@code residual_*} names and {@code inlier_threshold_sq_px} deliberately match
     * the names the CONF lane reserved in its run-record v1.1 contract, for the same quantities in
     * the same units, so a consumer reading both need not maintain a mapping. This file is
     * <em>not</em> that run record and does not claim to satisfy its contract.
     */
    public static final String HEADER = String.join(",",
            "frame_index", "timestamp_s",
            "est_x", "est_y", "est_yaw_deg",
            "success", "event", "reference_id",
            "track_count", "inlier_count",
            "residual_inlier_count",
            "residual_mean_sq_px", "residual_rms_px", "residual_median_sq_px",
            "residual_p90_sq_px", "residual_p95_sq_px", "residual_max_sq_px",
            "inlier_threshold_sq_px",
            "process_time_ns");

    /** One frame's measured row, retained in memory so the reproduction check can compare it. */
    public record Row(int frameIndex, double timestampS,
                      double estX, double estY, double estYawDeg,
                      boolean success, String event, int referenceId,
                      Integer trackCount, Integer inlierCount,
                      MotionResidualDiagnostics.Summary residuals,
                      Double p90SquaredPx, Double p95SquaredPx,
                      double inlierThresholdSquaredPx,
                      long processTimeNs) {
    }

    private final StitchingEstimator<GrayF32> estimator;
    private final MotionResidualDiagnostics diagnostics;
    private final FrameSource frameSource;
    private boolean recenterOccurred = false;

    public ResidualProbeRunner(InstrumentedStitching<GrayF32> instrumented,
                               StitchingEstimator<GrayF32> estimator,
                               FrameSource frameSource) {
        if (instrumented.stitch() != estimator.getStitch()) {
            throw new IllegalArgumentException(
                    "diagnostics must belong to the estimator that produces the pose (feature spec FR-002)");
        }
        this.estimator = estimator;
        this.diagnostics = instrumented.diagnostics();
        this.frameSource = frameSource;
        estimator.setRecenterListener(worldOldToNew -> recenterOccurred = true);
    }

    /** Processes every frame, writing one CSV row each and returning the rows for later checks. */
    public List<Row> run(Writer csv) throws IOException {
        csv.write(HEADER);
        csv.write("\n");

        List<Row> rows = new ArrayList<>();
        int referenceId = 0;
        int frameCount = frameSource.frameCount();

        for (int i = 0; i < frameCount; i++) {
            FrameSource.TimestampedFrame tf = frameSource.frame(i);

            recenterOccurred = false;
            long start = System.nanoTime();
            boolean success = estimator.processFrame(tf.image());
            long processTimeNs = System.nanoTime() - start;

            // Read immediately, before the next frame is submitted -- the values describe the most
            // recently processed frame only (MotionResidualDiagnostics, "Timing").
            MotionResidualDiagnostics.Summary summary = diagnostics.summarize();
            double[] perInlier = diagnostics.inlierSquaredResidualsPx();

            String event;
            if (i == 0) {
                event = "init";
            } else if (!success) {
                event = "restart";
                referenceId++;
            } else if (recenterOccurred) {
                event = "recenter";
                referenceId++;
            } else {
                event = "none";
            }

            Pose3D pose = estimator.getCurrentPose();
            // Reused, not reimplemented: the existing count diagnostics are already public and
            // already used by the run-record capture path, so both records report the same numbers.
            VoDiagnostics.Counts counts = VoDiagnostics.extract(estimator.getStitch().getMotion());

            Row row = new Row(
                    tf.frameIndex(), tf.timestampS(),
                    pose.x, pose.y, canonicalYaw(pose.yaw),
                    success, event, referenceId,
                    counts.trackCount(), counts.inlierCount(),
                    summary,
                    percentileSquaredPx(perInlier, 0.90),
                    percentileSquaredPx(perInlier, 0.95),
                    diagnostics.inlierThresholdSquaredPx(),
                    processTimeNs);

            rows.add(row);
            writeRow(csv, row);
        }
        return rows;
    }

    private static void writeRow(Writer csv, Row r) throws IOException {
        MotionResidualDiagnostics.Summary s = r.residuals();
        csv.write(String.join(",",
                Integer.toString(r.frameIndex()),
                Double.toString(r.timestampS()),
                Double.toString(r.estX()),
                Double.toString(r.estY()),
                Double.toString(r.estYawDeg()),
                Boolean.toString(r.success()),
                r.event(),
                Integer.toString(r.referenceId()),
                nullable(r.trackCount()),
                nullable(r.inlierCount()),
                nullable(s.inlierCount()),
                nullable(s.meanSquaredPx()),
                nullable(s.rmsPx()),
                nullable(s.medianSquaredPx()),
                nullable(r.p90SquaredPx()),
                nullable(r.p95SquaredPx()),
                nullable(s.maxSquaredPx()),
                Double.toString(r.inlierThresholdSquaredPx()),
                Long.toString(r.processTimeNs())));
        csv.write("\n");
    }

    /** Empty for absent. Never substitutes a zero, which would read as a measurement. */
    private static String nullable(Number value) {
        return value == null ? "" : value.toString();
    }

    /**
     * Order statistic of the per-inlier squared residuals, computed by this probe from the array
     * {@link MotionResidualDiagnostics#inlierSquaredResidualsPx()} already exposes.
     *
     * <p>Reported because {@code max} alone characterises the tail with a single, noisy sample.
     * Nearest-rank (no interpolation), so every reported value is one the estimator actually
     * produced — the same choice {@code Summary}'s lower median already makes.
     *
     * @return {@code null} when the residual array is absent or empty; absence is never zero.
     */
    static Double percentileSquaredPx(double[] residuals, double fraction) {
        if (residuals == null || residuals.length == 0) {
            return null;
        }
        double[] sorted = residuals.clone();
        Arrays.sort(sorted);
        int rank = (int) Math.ceil(fraction * sorted.length) - 1;
        if (rank < 0) {
            rank = 0;
        }
        if (rank >= sorted.length) {
            rank = sorted.length - 1;
        }
        return sorted[rank];
    }

    /**
     * Guards the [0, 360) invariant against the rounding boundary that can produce exactly 360.0.
     * Reimplemented rather than shared because {@code VoRunner.canonicalYaw} is package-private on
     * the CONF-claimed capture surface; the reproduction check is what keeps the two in step.
     */
    static double canonicalYaw(double yawDeg) {
        double normalized = ((yawDeg % 360.0) + 360.0) % 360.0;
        if (normalized >= 360.0) {
            normalized = 0.0;
        }
        return normalized;
    }
}
