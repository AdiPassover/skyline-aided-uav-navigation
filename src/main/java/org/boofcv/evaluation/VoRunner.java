package org.boofcv.evaluation;

import boofcv.struct.image.GrayF32;
import georegression.struct.homography.Homography2D_F64;
import org.boofcv.stitching.MotionModelStitchingEstimator;
import org.boofcv.util.structs.Pose3D;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Drives {@link MotionModelStitchingEstimator} over a {@link FrameSource}, unmodified (plan.md data flow).
 * Consumes the {@code processFrame} boolean success result — currently discarded at both existing
 * call sites — and classifies each frame's event for the run record.
 */
public final class VoRunner {

    private final MotionModelStitchingEstimator<GrayF32, ?> estimator;
    private final FrameSource frameSource;
    private boolean recenterOccurred = false;

    /** Frames on which a SYNTHETIC hard loss is to be injected (empty in every ordinary run). */
    private java.util.SortedSet<Integer> syntheticHardLossFrames = new java.util.TreeSet<>();
    /** Frames on which one actually fired, in order. */
    private final java.util.List<Integer> syntheticHardLossesFired = new java.util.ArrayList<>();
    /** Frames on which the ESTIMATOR itself failed, in order. */
    private final java.util.List<Integer> naturalHardLosses = new java.util.ArrayList<>();
    /** Frames on which the mosaic canvas was re-originned — bookkeeping, never a hard loss. */
    private final java.util.List<Integer> recenters = new java.util.ArrayList<>();

    /** When non-null, {@code logical_transform.csv} is written here (EXP-VO-004 diagnostic). */
    private Path logicalTransformCsv;

    /** When non-null, {@code refinement.csv} is written here (EXP-VO-010 diagnostic). */
    private Path refinementCsv;
    private org.boofcv.stitching.RefinementDiagnostics refinement;

    /** When non-null, {@code diagnostic_refit.csv} is written here (EXP-CONF-004 diagnostic). */
    private Path diagnosticRefitCsv;
    private org.boofcv.stitching.RefinementDiagnostics diagnosticRefit;

    /** When non-null, the alignment-layer sidecars are written into this directory (DEC-INT-001). */
    private Path alignmentSidecarDir;
    private org.boofcv.relocalization.NavigationAligner aligner;

    /** When non-null, the full relocalization pipeline is driven (DEC-INT-002, P1). */
    private org.boofcv.relocalization.RelocalizationPipeline pipeline;
    private org.boofcv.relocalization.SkylineProfileFile skylineProfiles;
    /** The observer-only refit probe delta_rot_refit is read from, when escalation is enabled. */
    private org.boofcv.stitching.RefinementDiagnostics instabilityProbe;

    // Confidence capture (contracts/run-record-v1.1.md). All three are set together or not at all.
    private org.boofcv.stitching.MotionResidualDiagnostics residualDiagnostics;
    private org.boofcv.confidence.SignalExtractor signalExtractor;
    private org.boofcv.confidence.ConfidenceScorer confidenceScorer;
    /** T022: frames where the tracker-flag inlier count and the match-set size disagreed. */
    private int inlierCountDisagreements = 0;

    /** When non-null, {@code metric_track.csv} is written here (DEC-VO-007 D3 runtime readout). */
    private Path metricTrackCsv;

    /** The recorded height stream replayed into the estimator, causally. Null when metric is off. */
    private HeightSampleStream heightStream;
    private HeadingSampleStream headingStream;
    @lombok.Getter private long rejectedHeadingSamples;

    /**
     * Header of the metric sidecar. {@code metric_*} is the runtime metric local pose in
     * <b>METRES</b> (east/north) and degrees; {@code h_used_m} is the AGL the increment into this
     * frame was actually converted with, i.e. the REFERENCE frame's, and {@code h_status} says how
     * old the sample behind it was ({@code DEC-VO-007} D6). {@code segment_index} advances at every
     * stitching restart and {@code gap_before_segment} marks the ones whose preceding displacement
     * was never estimated; {@code segment_*} is the pose relative to that segment's own origin, the
     * one containing no unestimated interval.
     *
     * <p>{@code metric_yaw_deg} is the AUTHORITATIVE navigation heading -- clockwise from North --
     * when a heading channel is configured, and the visually integrated yaw otherwise;
     * {@code heading_semantics} says which. {@code yaw_visual_deg} is always the visual diagnostic,
     * and {@code yaw_disagreement_deg} is the wrap-safe per-frame difference between the visual and
     * external rotation increments -- a raw angle, never a confidence.
     * {@code unknown_translation_gap} marks a segment whose preceding DISPLACEMENT was never
     * estimated; {@code heading_known_across_gap} says separately whether its orientation was
     * anchored by a sensor that never saw the visual failure.
     */
    static final String METRIC_SIDECAR_HEADER =
            "frame_index,timestamp_s,event,"
            + "metric_east_m,metric_north_m,metric_yaw_deg,"
            + "gsd_m_per_px,h_used_m,h_rel_m,h_sample_time_s,h_age_s,h_status,metric_frame_usable,"
            + "yaw_nav_deg,yaw_nav_source_deg,yaw_nav_mount_deg,yaw_nav_sample_time_s,"
            + "yaw_nav_age_s,yaw_nav_status,yaw_used_deg,yaw_used_status,"
            + "yaw_visual_deg,yaw_disagreement_deg,"
            + "segment_index,unknown_translation_gap,heading_known_across_gap,"
            + "segment_east_m,segment_north_m,segment_yaw_deg,"
            + "raw_x_px,raw_y_px,raw_yaw_deg";

    /**
     * Header of the refinement sidecar (EXP-VO-010). {@code m*} is RANSAC's winning
     * <b>minimal-sample</b> model and {@code s*} the model the estimator actually shipped, both in
     * homography form, row-major; they are identical when {@code refineEstimate} is false.
     * {@code inliers} is the size of the match set the refiner was (or would have been) given, and
     * the {@code d_*} columns are the readout quantities of each, so an analysis need not
     * re-decompose the matrices to see how far refinement moved the model.
     */
    static final String REFINEMENT_SIDECAR_HEADER =
            "frame_index,event,refine_enabled,inliers,"
            + "m00,m01,m02,m10,m11,m12,m20,m21,m22,"
            + "s00,s01,s02,s10,s11,s12,s20,s21,s22,"
            + "min_rot_deg,shp_rot_deg,d_rot_deg,"
            + "min_scale,shp_scale,d_log_scale,"
            + "min_tx_px,min_ty_px,shp_tx_px,shp_ty_px,d_trans_px,"
            + "min_aniso,shp_aniso";

    /**
     * Header of the sidecar. {@code g00..g22} are {@code G_k : L -> C_k}, row-major, homography form
     * (the FULL logical composition). The {@code rigid_*} and {@code inc_*} columns are the
     * DEC-VO-004 Alternative E readout and its observed-but-not-applied geometric state.
     */
    static final String LOGICAL_SIDECAR_HEADER =
            "frame_index,event,logical_x,logical_y,logical_yaw_deg,g00,g01,g02,g10,g11,g12,g20,g21,g22,"
            + "rigid_x,rigid_y,rigid_yaw_deg,rigid_accum_log_scale,rigid_accum_scale,"
            + "inc_scale,inc_log_scale,inc_anisotropy,inc_deformation,inc_stretch_axis_deg,"
            + "inc_perspective,inc_flow_px,inc_rotation_deg,inc_proper,improper_count";

    public VoRunner(MotionModelStitchingEstimator<GrayF32, ?> estimator, FrameSource frameSource) {
        this.estimator = estimator;
        this.frameSource = frameSource;
        estimator.setRecenterListener(worldOldToNew -> recenterOccurred = true);
    }

    /** Runs every frame in {@code frameSource} through the estimator, writing one row each. */
    /**
     * Enables the per-frame logical-transform sidecar. VO-owned diagnostic output, outside the
     * run-record contract: the file is written next to {@code frames.csv} and nothing in
     * {@code naveval} reads it.
     */
    public void setLogicalTransformCsv(Path logicalTransformCsv) {
        this.logicalTransformCsv = logicalTransformCsv;
    }

    /**
     * Enables the per-frame refinement sidecar (EXP-VO-010). VO-owned diagnostic output, outside the
     * run-record contract: written next to {@code frames.csv}, and nothing in {@code naveval} reads
     * it.
     *
     * @param diagnostics the probe attached to the very estimator being driven -- pairing them here
     *                    rather than looking one up means a sidecar cannot be written against a
     *                    different estimator than the one that produced the poses
     */
    public void setRefinementCsv(Path refinementCsv,
                                 org.boofcv.stitching.RefinementDiagnostics diagnostics) {
        this.refinementCsv = refinementCsv;
        this.refinement = diagnostics;
    }

    /**
     * Enables the per-frame diagnostic-refit sidecar (EXP-CONF-004). Observer-only: the probe's
     * diagnostic refiner computes what an all-inlier refit of the production match set would have
     * produced, while the estimator ships the minimal-sample model untouched.
     *
     * @param diagnostics the probe attached to the very estimator being driven, built with the
     *                    diagnostic refiner ({@code diagnosticRefitEnabled()})
     */
    public void setDiagnosticRefitCsv(Path diagnosticRefitCsv,
                                      org.boofcv.stitching.RefinementDiagnostics diagnostics) {
        if (!diagnostics.diagnosticRefitEnabled()) {
            throw new IllegalArgumentException(
                    "diagnostic_refit sidecar wired to a probe built without the diagnostic "
                    + "refiner; the sidecar would be empty. Construction bug, not a data state.");
        }
        this.diagnosticRefitCsv = diagnosticRefitCsv;
        this.diagnosticRefit = diagnostics;
    }

    /**
     * Enables the alignment-layer sidecars ({@code DEC-INT-001}): {@code alignment_frames.csv} and
     * {@code alignment_events.csv} in {@code directory}. Observational only — the
     * {@link org.boofcv.relocalization.NavigationAligner} reads the estimator's metric state
     * ({@code segmentRelativePose}, the authoritative heading, the VO's own validity flags) and the
     * {@code processFrame} boolean after each frame and feeds nothing back; {@code frames.csv} is
     * unaffected. Outside the run-record contract; nothing in {@code naveval} reads it.
     *
     * @throws IllegalStateException when the estimator has no metric readout with an authoritative
     *                               heading — the layer has no contract for a pixel / visual-yaw
     *                               pose ({@code DEC-INT-001} amendment 2026-09-07)
     */
    /**
     * <b>SYNTHETIC / TEST ONLY.</b> Declares frames on which a hard loss is forced, so the recovery
     * contract can be exercised deterministically ({@code MotionModelStitchingEstimator#injectHardLossOnNextFrame}).
     *
     * <p>The injected loss takes the estimator's own hard-loss branch, so downstream everything —
     * segment termination, the UNKNOWN translation gap, position invalidity, the scheduler's
     * {@code HARD_LOSS} request — is exactly what a tracker failure produces. The two are kept
     * distinguishable in the record: {@link #syntheticHardLossesFired()} and
     * {@link #naturalHardLosses()} are reported separately and the manifest carries both, so a
     * forced loss is never counted as observed estimator failure.
     *
     * @param frames dataset frame indices, each {@code > 0} (frame 0 integrates nothing)
     */
    public void setSyntheticHardLossFrames(java.util.SortedSet<Integer> frames) {
        this.syntheticHardLossFrames = frames == null ? new java.util.TreeSet<>()
                : new java.util.TreeSet<>(frames);
    }

    /** Frames on which a synthetic hard loss actually fired. */
    public java.util.List<Integer> syntheticHardLossesFired() {
        return java.util.Collections.unmodifiableList(syntheticHardLossesFired);
    }

    /** Frames on which the estimator itself failed to track — genuine, observed hard losses. */
    public java.util.List<Integer> naturalHardLosses() {
        return java.util.Collections.unmodifiableList(naturalHardLosses);
    }

    /** Frames on which the mosaic canvas was re-originned. Bookkeeping; never a navigation event. */
    public java.util.List<Integer> recenters() {
        return java.util.Collections.unmodifiableList(recenters);
    }

    public void setAlignmentSidecar(Path directory) {
        if (pipeline != null) {
            throw new IllegalStateException("setRelocalization already drives the alignment layer");
        }
        requireMetricHeadingContract();
        this.alignmentSidecarDir = directory;
        this.aligner = new org.boofcv.relocalization.NavigationAligner();
    }

    /**
     * Enables the full relocalization pipeline ({@code DEC-INT-002}, P1): the aligner plus
     * trusted-reference memory, exact dual-view retrieval, scheduler, region tracker and acceptance
     * gate, fed per frame with the metric VO handoff, the {@code processFrame} boolean, the skyline
     * observation for the frame (from {@code profiles}, when given) and {@code delta_rot_refit}
     * (from {@link #setInstabilityProbe}, when set). Sidecars go to {@code directory}.
     * Observational with respect to the estimator; nothing flows back into estimation.
     */
    public void setRelocalization(org.boofcv.relocalization.RelocalizationPipeline pipeline,
                                  @javax.annotation.Nullable org.boofcv.relocalization.SkylineProfileFile profiles,
                                  Path directory) {
        if (pipeline == null || directory == null) {
            throw new IllegalArgumentException("pipeline and directory are required");
        }
        if (aligner != null) {
            throw new IllegalStateException("setAlignmentSidecar already drives the alignment layer");
        }
        requireMetricHeadingContract();
        this.pipeline = pipeline;
        this.skylineProfiles = profiles;
        this.alignmentSidecarDir = directory;
    }

    /**
     * The relocalization layer consumes exactly one pose contract: the metric segment-relative
     * position (height owns the scale, {@code DEC-VO-009}) rotated by the authoritative external
     * heading ({@code DEC-VO-010}). Anything else — the pixel {@code RIGID_MOTION} readout, or a
     * metric state rotated by the visual yaw — is refused here, loudly, rather than consumed under a
     * contract that does not describe it. Which source <em>publishes</em> the pose is irrelevant:
     * the layer reads {@code getMetricNavigation()} directly and never {@code getCurrentPose()}.
     */
    private void requireMetricHeadingContract() {
        if (!estimator.isMetricReadoutEnabled()) {
            throw new IllegalStateException("the relocalization layer requires the metric readout "
                    + "(metric_readout.enabled with h0 and f_working): it consumes the metric "
                    + "segment-relative pose in metres and has no contract for the pixel readout");
        }
        if (!estimator.isHeadingReadoutEnabled()) {
            throw new IllegalStateException("the relocalization layer requires the authoritative "
                    + "external heading (heading_readout.enabled): without it the metric East/North "
                    + "axes follow the drifting visual yaw and a translation-only alignment is undefined");
        }
    }

    /**
     * The observer-only diagnostic refit probe ({@code EXP-CONF-004}) {@code delta_rot_refit} is
     * read from on every usable increment — the CONF-validated path, and the only source allowed.
     * Required when the relocalization config sets {@code instability_threshold_deg}.
     */
    public void setInstabilityProbe(org.boofcv.stitching.RefinementDiagnostics probe) {
        if (probe == null || !probe.diagnosticRefitEnabled()) {
            throw new IllegalArgumentException("delta_rot_refit needs a probe built with the "
                    + "diagnostic refiner (diagnostic_refit_sidecar); construction bug, not a data state");
        }
        this.instabilityProbe = probe;
    }

    /** The alignment layer driven by this run, or {@code null} when neither mode is on. */
    public org.boofcv.relocalization.NavigationAligner aligner() {
        return pipeline != null ? pipeline.aligner() : aligner;
    }

    /** The relocalization pipeline, or {@code null} when not enabled. */
    public org.boofcv.relocalization.RelocalizationPipeline pipeline() {
        return pipeline;
    }

    /**
     * Enables per-frame confidence capture ({@code DEC-CONF-002}, contracts/run-record-v1.1.md).
     *
     * <p>Observational only: the diagnostics handle must belong to the very estimator being driven
     * (the {@code InstrumentedStitching} pairing enforces that at the construction site), signals
     * are read in the FR-034 window — after each {@code processFrame}, before the next — and
     * nothing here feeds back into estimation. {@code VoRunnerConfidenceTest} asserts capture-on
     * and capture-off runs produce bit-identical poses, events and counts.
     *
     * @param diagnostics residual view of the driven estimator's own RANSAC
     * @param extractor   the temporal-state owner ({@code EXP-CONF-001} A4 semantics)
     * @param scorer      the pure verdict function under the active calibration
     */
    public void setConfidence(org.boofcv.stitching.MotionResidualDiagnostics diagnostics,
                              org.boofcv.confidence.SignalExtractor extractor,
                              org.boofcv.confidence.ConfidenceScorer scorer) {
        this.residualDiagnostics = diagnostics;
        this.signalExtractor = extractor;
        this.confidenceScorer = scorer;
    }

    /**
     * Frames on which the tracker-flag inlier count and the estimator's match-set size disagreed
     * (both are persisted per frame; neither is reconciled — data-model.md §1.2). Meaningful after
     * {@link #run}.
     */
    public int inlierCountDisagreements() {
        return inlierCountDisagreements;
    }

    /**
     * Header of the diagnostic-refit sidecar (EXP-CONF-004). {@code s*} is the model the frozen
     * estimator actually shipped (the minimal-sample winner; the sidecar requires
     * {@code refineEstimate = false}) and {@code r*} the observer-only all-inlier refit of the
     * same production match set, both in homography form, row-major, PROCESSED-frame coordinates
     * (the run's downsampled grid — the same convention as every persisted transform). Delta
     * columns use the {@code RIGID_MOTION} readout at the processed-frame centre, so
     * {@code delta_rotation_deg} is defined on exactly the quantity navigation integrates.
     * {@code delta_flow_direction_deg} is empty when either centre flow is below 1 processed px
     * (2 full-resolution px); refit-dependent columns are empty when {@code refit_ok} is 0 —
     * absence is explicit, never zeros.
     */
    static final String DIAGNOSTIC_REFIT_SIDECAR_HEADER =
            "frame_index,event,inliers,refit_ok,"
            + "s00,s01,s02,s10,s11,s12,s20,s21,s22,"
            + "r00,r01,r02,r10,r11,r12,r20,r21,r22,"
            + "shp_rot_deg,refit_rot_deg,delta_rotation_deg,delta_center_flow_px,"
            + "delta_flow_direction_deg,delta_log_scale,delta_perspective,"
            + "refit_time_ns,readout_time_ns";

    /** Centre-flow magnitude below which a flow direction is not defined (processed px). */
    private static final double DIR_MIN_FLOW_PROCESSED_PX = 1.0;

    /**
     * Supplies the external heading recording the run replays. The estimator must already have
     * {@code enableHeadingReadout(...)} called on it, for the same reason the height stream is
     * paired with {@code enableMetricReadout}: a recording cannot be replayed into an estimator that
     * is not consuming it.
     */
    public void setHeadingStream(HeadingSampleStream headingStream) {
        if (!estimator.isHeadingReadoutEnabled()) {
            throw new IllegalStateException(
                    "the estimator has no heading channel: call enableHeadingReadout(...) on it "
                    + "before supplying a heading recording");
        }
        this.headingStream = headingStream;
    }

    /**
     * Enables the runtime metric readout's per-frame sidecar and supplies the height recording it
     * replays ({@code DEC-VO-007} D3). The estimator must already have
     * {@code enableMetricReadout(...)} called on it; pairing them here means a sidecar cannot be
     * written against an estimator that is not producing a metric pose.
     */
    public void setMetricTrackCsv(Path metricTrackCsv, HeightSampleStream heightStream) {
        if (!estimator.isMetricReadoutEnabled()) {
            throw new IllegalStateException(
                    "metric_track.csv was requested but the estimator has no metric readout enabled");
        }
        this.metricTrackCsv = metricTrackCsv;
        this.heightStream = heightStream;
    }

    public void run(RunRecordWriter writer) throws IOException {
        BufferedWriter sidecar = null;
        BufferedWriter refineSidecar = null;
        BufferedWriter diagSidecar = null;
        org.boofcv.relocalization.AlignmentSidecarWriter alignmentSidecar = null;
        BufferedWriter metricSidecar = null;
        if (logicalTransformCsv != null) {
            sidecar = openSidecar(logicalTransformCsv, LOGICAL_SIDECAR_HEADER);
        }
        if (refinementCsv != null) {
            refineSidecar = openSidecar(refinementCsv, REFINEMENT_SIDECAR_HEADER);
        }
        if (diagnosticRefitCsv != null) {
            diagSidecar = openSidecar(diagnosticRefitCsv, DIAGNOSTIC_REFIT_SIDECAR_HEADER);
        }
        if (alignmentSidecarDir != null) {
            alignmentSidecar = new org.boofcv.relocalization.AlignmentSidecarWriter(alignmentSidecarDir);
            if (pipeline != null) {
                // Lets the sidecar report a selected reference's stored position and age.
                alignmentSidecar.setReferenceMemory(pipeline.memory());
            }
        }
        if (metricTrackCsv != null) {
            metricSidecar = openSidecar(metricTrackCsv, METRIC_SIDECAR_HEADER);
        }
        try {
            runFrames(writer, sidecar, refineSidecar, diagSidecar, alignmentSidecar, metricSidecar);
        } finally {
            if (sidecar != null) {
                sidecar.close();
            }
            if (refineSidecar != null) {
                refineSidecar.close();
            }
            if (diagSidecar != null) {
                diagSidecar.close();
            }
            if (alignmentSidecar != null) {
                alignmentSidecar.close();
            }
            if (metricSidecar != null) {
                metricSidecar.close();
            }
        }
    }

    private static BufferedWriter openSidecar(Path path, String header) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.UTF_8);
        w.write(header);
        w.newLine();
        return w;
    }

    private void runFrames(RunRecordWriter writer, BufferedWriter sidecar,
                           BufferedWriter refineSidecar, BufferedWriter diagSidecar,
                           org.boofcv.relocalization.AlignmentSidecarWriter alignmentSidecar,
                           BufferedWriter metricSidecar)
            throws IOException {
        int referenceId = 0;
        int hardLossesWritten = 0;
        int frameCount = frameSource.frameCount();
        Homography2D_F64 g = new Homography2D_F64();
        for (int i = 0; i < frameCount; i++) {
            FrameSource.TimestampedFrame tf = frameSource.frame(i);

            // The height stream is replayed as a stream: only samples that had become available at
            // or before this frame's time are delivered, and only once each. Nothing here can see a
            // sample from the frame's future, which is what makes the replay causally equivalent to
            // a live sensor (DEC-VO-007 D5).
            if (headingStream != null) {
                for (HeadingSampleStream.Sample sample : headingStream.drainUpTo(tf.timestampS())) {
                    if (!estimator.submitHeadingSample(sample.availabilityTimeS(),
                                                       sample.headingDeg())) {
                        rejectedHeadingSamples++;
                    }
                }
            }
            if (heightStream != null) {
                for (HeightSampleStream.Sample sample : heightStream.drainUpTo(tf.timestampS())) {
                    estimator.submitHeightSample(sample.availabilityTimeS(), sample.relativeHeightM());
                }
            }

            recenterOccurred = false;
            // SYNTHETIC hard loss: armed only from an explicit synthetic_hard_loss config block,
            // and recorded separately from an observed tracker failure below.
            boolean injectedHere = syntheticHardLossFrames.contains(i);
            if (injectedHere) {
                estimator.injectHardLossOnNextFrame();
            }
            long start = System.nanoTime();
            boolean success = estimator.processFrame(tf.image(), tf.timestampS());
            long processTimeNs = System.nanoTime() - start;

            String event;
            if (i == 0) {
                event = "init";
            } else if (!success) {
                // Restart: StitchingEstimator re-initializes from this frame internally
                // (resetStitchingAndRestart), which is itself a mosaic reference change.
                event = "restart";
                referenceId++;
                if (injectedHere) {
                    syntheticHardLossesFired.add(i);
                } else {
                    naturalHardLosses.add(i);
                }
            } else if (recenterOccurred) {
                event = "recenter";
                referenceId++;
                recenters.add(i);
            } else {
                event = "none";
            }

            Pose3D pose = estimator.getCurrentPose();
            Homography2D_F64 currToWorld =
                    estimator.getStitch().getWorldToCurr(new Homography2D_F64()).invert(null);

            VoDiagnostics.Counts diag;
            FrameEstimateRow.Confidence conf = null;
            if (confidenceScorer != null) {
                CapturedConfidence captured = captureConfidence(success, event, i == 0,
                        tf.image().getWidth(), tf.image().getHeight());
                conf = captured.confidence();
                diag = captured.counts();
            } else {
                diag = VoDiagnostics.extract(estimator.getStitch().getMotion());
            }

            FrameEstimateRow row = new FrameEstimateRow(
                    tf.frameIndex(), tf.timestampS(), pose.x, pose.y, null, canonicalYaw(pose.yaw),
                    success, event, referenceId,
                    currToWorld.getA11(), currToWorld.getA12(), currToWorld.getA13(),
                    currToWorld.getA21(), currToWorld.getA22(), currToWorld.getA23(),
                    currToWorld.getA31(), currToWorld.getA32(), currToWorld.getA33(),
                    diag.trackCount(), diag.inlierCount(), processTimeNs, conf);

            writer.writeFrame(row);

            if (sidecar != null) {
                Pose3D lp = estimator.getLogicalPose();
                estimator.getLogicalToCurrentAsHomography(g);
                Pose3D rp = estimator.getRigidPose();
                org.boofcv.stitching.RigidNavigationState<?> r = estimator.getRigidNavigation();
                sidecar.write(tf.frameIndex() + "," + event + "," + lp.x + "," + lp.y + ","
                        + canonicalYaw(lp.yaw) + ","
                        + g.a11 + "," + g.a12 + "," + g.a13 + ","
                        + g.a21 + "," + g.a22 + "," + g.a23 + ","
                        + g.a31 + "," + g.a32 + "," + g.a33 + ","
                        + rp.x + "," + rp.y + "," + canonicalYaw(rp.yaw) + ","
                        + r.getAccumulatedLogScale() + "," + r.accumulatedScale() + ","
                        + r.getIncrementScale() + "," + r.getIncrementLogScale() + ","
                        + r.getIncrementAnisotropy() + "," + r.getIncrementDeformation() + ","
                        + Math.toDegrees(r.getIncrementStretchAxisRad()) + ","
                        + r.getIncrementPerspective() + "," + r.getIncrementFlowPx() + ","
                        + r.getIncrementRotationDeg() + "," + (r.isIncrementProper() ? 1 : 0) + ","
                        + r.getImproperIncrementCount());
                sidecar.newLine();
            }

            if (alignmentSidecar != null && pipeline != null) {
                // P1: the full loop. Inputs are exactly what the run already produces — the
                // metric state's segment-relative pose, authoritative heading and validity flags
                // (DEC-VO-009/-010), the processFrame boolean, the observer-only refit probe's
                // delta_rot_refit, and the exported skyline profile for this frame. Nothing flows
                // back into the estimator, and getCurrentPose() is never read here.
                Double deltaRotRefit = instabilityProbe == null ? null
                        : diagnosticDeltaRotationDeg(instabilityProbe, tf.image().getWidth(),
                                tf.image().getHeight());
                org.boofcv.relocalization.SkylineObservation obs = skylineProfiles == null ? null
                        : skylineProfiles.lookup(tf.frameIndex(), tf.timestampS());
                org.boofcv.relocalization.LocalPoseSample sample =
                        org.boofcv.relocalization.LocalPoseSample.fromMetric(tf.frameIndex(),
                                tf.timestampS(), success, estimator.getMetricNavigation(),
                                estimator.getRigidPose());
                org.boofcv.relocalization.RelocalizationPipeline.FrameStep step =
                        pipeline.step(sample, obs, deltaRotRefit);
                java.util.List<org.boofcv.relocalization.HardLossEvent> losses =
                        pipeline.aligner().hardLossEvents();
                while (hardLossesWritten < losses.size()) {
                    writeLabelledHardLoss(alignmentSidecar, losses.get(hardLossesWritten++));
                }
                alignmentSidecar.writeStep(step);
            } else if (alignmentSidecar != null) {
                // P0: the alignment layer alone, on the same metric handoff.
                org.boofcv.relocalization.LocalPoseSample sample =
                        org.boofcv.relocalization.LocalPoseSample.fromMetric(tf.frameIndex(),
                                tf.timestampS(), success, estimator.getMetricNavigation(),
                                estimator.getRigidPose());
                int dropoutsBefore = aligner.translationDropoutEvents().size();
                org.boofcv.relocalization.NavigationOutput out = aligner.observe(sample);
                java.util.List<org.boofcv.relocalization.HardLossEvent> losses =
                        aligner.hardLossEvents();
                while (hardLossesWritten < losses.size()) {
                    writeLabelledHardLoss(alignmentSidecar, losses.get(hardLossesWritten++));
                }
                java.util.List<org.boofcv.relocalization.TranslationDropoutEvent> drops =
                        aligner.translationDropoutEvents();
                for (int d = dropoutsBefore; d < drops.size(); d++) {
                    alignmentSidecar.writeDropout(drops.get(d));
                }
                alignmentSidecar.writeFrame(out);
            }

            if (refineSidecar != null) {
                writeRefinementRow(refineSidecar, tf.frameIndex(), event,
                        tf.image().getWidth(), tf.image().getHeight());
            }

            if (diagSidecar != null) {
                writeDiagnosticRefitRow(diagSidecar, tf.frameIndex(), event,
                        tf.image().getWidth(), tf.image().getHeight());
            }

            if (metricSidecar != null) {
                writeMetricRow(metricSidecar, tf.frameIndex(), tf.timestampS(), event);
            }
        }
    }

    /**
     * Writes one hard loss with its source. A forced loss takes the identical downstream path, so
     * the label is the only thing keeping it distinguishable from observed estimator failure.
     */
    private void writeLabelledHardLoss(org.boofcv.relocalization.AlignmentSidecarWriter sidecar,
                                       org.boofcv.relocalization.HardLossEvent e)
            throws java.io.IOException {
        sidecar.writeHardLoss(e, syntheticHardLossFrames.contains(e.frameIndex())
                ? org.boofcv.relocalization.AlignmentSidecarWriter.LOSS_SYNTHETIC
                : org.boofcv.relocalization.AlignmentSidecarWriter.LOSS_NATURAL);
    }

    private record CapturedConfidence(FrameEstimateRow.Confidence confidence,
                                      VoDiagnostics.Counts counts) {
    }

    /**
     * Extracts, scores and packages one frame's confidence (contracts/run-record-v1.1.md).
     *
     * <p>Runs inside the FR-034 window — after {@code processFrame}, before the next frame. The
     * increment diagnostics ({@code inc_flow_px}, {@code inc_log_scale}) are passed only for
     * frames carrying a genuine accepted increment: not the init frame (no previous frame) and
     * not a restart frame (the estimator discarded the estimate and re-initialised). Signal reads
     * degrade to absent rather than aborting the frame (FR-014).
     *
     * <p>{@code confidence_time_ns} encloses extraction and scoring — everything this method does
     * — and excludes estimation, which {@code process_time_ns} already carries (FR-028).
     */
    private CapturedConfidence captureConfidence(boolean success, String event, boolean firstFrame,
                                                 int frameWidth, int frameHeight) {
        long confStart = System.nanoTime();

        VoDiagnostics.Diagnostics diagnostics;
        try {
            diagnostics = VoDiagnostics.extractWithCoverage(
                    estimator.getStitch().getMotion(), frameWidth, frameHeight);
        } catch (RuntimeException e) {
            diagnostics = VoDiagnostics.Diagnostics.UNAVAILABLE;   // FR-014: absent, never abort
        }

        org.boofcv.stitching.MotionResidualDiagnostics.Summary summary;
        Double threshold;
        try {
            summary = residualDiagnostics.summarize();
            threshold = residualDiagnostics.inlierThresholdSquaredPx();
        } catch (RuntimeException e) {
            summary = org.boofcv.stitching.MotionResidualDiagnostics.Summary.UNAVAILABLE;
            threshold = null;
        }

        boolean hasIncrement = success && !firstFrame && !"restart".equals(event);
        Double incFlowPx = null;
        Double incLogScale = null;
        if (hasIncrement) {
            org.boofcv.stitching.RigidNavigationState<?> r = estimator.getRigidNavigation();
            incFlowPx = r.getIncrementFlowPx();
            incLogScale = r.getIncrementLogScale();
        }

        org.boofcv.confidence.SignalBlock block = signalExtractor.extract(
                success, event, diagnostics, summary, threshold, incFlowPx, incLogScale);
        org.boofcv.confidence.ConfidenceResult result =
                confidenceScorer.score(block, success, firstFrame);

        if (block.inlierCount() != null && block.residualInlierCount() != null
                && !block.inlierCount().equals(block.residualInlierCount())) {
            inlierCountDisagreements++;    // recorded, never reconciled (data-model.md §1.2)
        }

        long confidenceTimeNs = System.nanoTime() - confStart;
        FrameEstimateRow.Confidence conf = new FrameEstimateRow.Confidence(
                block.residualInlierCount(), block.residualMeanSqPx(), block.residualRmsPx(),
                block.residualMedianSqPx(), block.residualMaxSqPx(), block.inlierThresholdSqPx(),
                block.inlierCoverage(), block.keyframeAge(), block.relativeSupport(),
                block.incFlowPx(), incLogScale, block.incLogScaleDispersion(),
                result.outcome().wireName(), result.reason().wireName(), result.score(),
                confidenceTimeNs);
        return new CapturedConfidence(conf, diagnostics.counts());
    }

    /**
     * One metric-sidecar row ({@code DEC-VO-007} D3 runtime readout).
     *
     * <p>{@code h_used_m} is the height of the reading the LAST increment was converted with, i.e.
     * the REFERENCE frame's -- not the reading taken at this frame, which is latched for the next
     * one. That asymmetry is the whole point of the convention, and logging the wrong one would make
     * the record disagree with the pose it is recording.
     */
    private void writeMetricRow(BufferedWriter out, int frameIndex, double timestampS, String event)
            throws IOException {
        org.boofcv.stitching.metric.MetricNavigationState m = estimator.getMetricNavigation();
        org.boofcv.util.structs.Pose3D metric = m.metricPose();
        org.boofcv.util.structs.Pose3D segment = m.segmentRelativePose();
        org.boofcv.util.structs.Pose3D raw = estimator.getRigidPose();
        org.boofcv.stitching.metric.HeightReading used = m.getLastUsedHeight();
        // The reading latched at THIS frame (observe() latches after converting), i.e. the one
        // navHeadingDeg() reports -- as against getLastUsedHeading(), the k-1 one the increment
        // into this frame was actually rotated by. Logging the wrong one would make the record
        // disagree with the pose it is recording, exactly as for height.
        org.boofcv.stitching.metric.HeadingReading now = m.getReferenceHeading();
        org.boofcv.stitching.metric.HeadingReading usedHdg = m.getLastUsedHeading();

        out.write(frameIndex + "," + timestampS + "," + event + ","
                + m.eastM() + "," + m.northM() + "," + num(metric.yaw) + ","
                + m.getLastGsdMPerPx() + ","
                + (used == null ? "" : Double.toString(used.heightAglM())) + ","
                + (used == null ? "" : Double.toString(used.relativeHeightM())) + ","
                + (used == null ? "" : Double.toString(used.sampleTimeS())) + ","
                + (used == null ? "" : Double.toString(used.ageS())) + ","
                + (used == null ? "" : used.status().name()) + ","
                + (m.lastFrameUsable() ? 1 : 0) + ","
                + num(m.navHeadingDeg()) + ","
                + (now == null ? "" : num(now.sourceHeadingDeg())) + ","
                + (now == null ? "" : num(now.mountOffsetDeg())) + ","
                + (now == null ? "" : Double.toString(now.sampleTimeS())) + ","
                + (now == null ? "" : Double.toString(now.ageS())) + ","
                + m.currentHeadingStatus().name() + ","
                + (usedHdg == null ? "" : num(usedHdg.yawNavDeg())) + ","
                + m.lastHeadingStatus().name() + ","
                + canonicalYaw(Math.toDegrees(m.headingRadUnwrapped())) + ","
                + num(m.getLastYawDisagreementDeg()) + ","
                + m.getSegmentIndex() + ","
                + (m.isUnknownTranslationGapBeforeSegment() ? 1 : 0) + ","
                + (m.isHeadingKnownAcrossGap() ? 1 : 0) + ","
                + segment.x + "," + segment.y + "," + num(segment.yaw) + ","
                + raw.x + "," + raw.y + "," + canonicalYaw(raw.yaw));
        out.newLine();
    }

    /**
     * One refinement-sidecar row (EXP-VO-010). Both models are decomposed through the same
     * {@code RigidMotionDecomposition} the {@code RIGID_MOTION} readout uses, at the image centre,
     * so {@code d_rot_deg} is exactly the quantity the navigation state integrates.
     *
     * <p>{@code d_rot_deg} is wrapped into {@code (-180, 180]} -- the two models can straddle the
     * branch cut when the rotation is near +/-180 deg, and an unwrapped difference would report a
     * 360 deg change where the geometry moved by a fraction of a degree.
     */
    private void writeRefinementRow(BufferedWriter out, int frameIndex, String event,
                                    int frameWidth, int frameHeight) throws IOException {
        if (!refinement.hasModels()) {
            return;   // absence is preserved rather than written as zeros (DEC-VO-001)
        }
        Homography2D_F64 m = refinement.minimalSampleModel();
        Homography2D_F64 sh = refinement.shippedModel();
        double cx = frameWidth / 2.0, cy = frameHeight / 2.0;

        double[] jm = org.boofcv.stitching.MotionModelSupport.HOMOGRAPHY.jacobian(m, cx, cy, null);
        double[] js = org.boofcv.stitching.MotionModelSupport.HOMOGRAPHY.jacobian(sh, cx, cy, null);
        org.boofcv.stitching.RigidMotionDecomposition dm =
                new org.boofcv.stitching.RigidMotionDecomposition().set(jm[0], jm[1], jm[2], jm[3]);
        org.boofcv.stitching.RigidMotionDecomposition ds =
                new org.boofcv.stitching.RigidMotionDecomposition().set(js[0], js[1], js[2], js[3]);

        georegression.struct.point.Point2D_F64 pm =
                org.boofcv.stitching.MotionModelSupport.HOMOGRAPHY.apply(m, cx, cy, null);
        georegression.struct.point.Point2D_F64 ps =
                org.boofcv.stitching.MotionModelSupport.HOMOGRAPHY.apply(sh, cx, cy, null);

        double dRot = wrapDegrees(Math.toDegrees(ds.getRotationRad() - dm.getRotationRad()));

        out.write(frameIndex + "," + event + "," + (refinement.refinementEnabled() ? 1 : 0) + ","
                + refinement.inlierCount() + ","
                + m.a11 + "," + m.a12 + "," + m.a13 + "," + m.a21 + "," + m.a22 + "," + m.a23 + ","
                + m.a31 + "," + m.a32 + "," + m.a33 + ","
                + sh.a11 + "," + sh.a12 + "," + sh.a13 + "," + sh.a21 + "," + sh.a22 + ","
                + sh.a23 + "," + sh.a31 + "," + sh.a32 + "," + sh.a33 + ","
                + Math.toDegrees(dm.getRotationRad()) + "," + Math.toDegrees(ds.getRotationRad())
                + "," + dRot + ","
                + dm.getUniformScale() + "," + ds.getUniformScale() + ","
                + (ds.logUniformScale() - dm.logUniformScale()) + ","
                + (pm.x - cx) + "," + (pm.y - cy) + "," + (ps.x - cx) + "," + (ps.y - cy) + ","
                + Math.hypot(ps.x - pm.x, ps.y - pm.y) + ","
                + dm.getAnisotropy() + "," + ds.getAnisotropy());
        out.newLine();
    }

    /**
     * One diagnostic-refit sidecar row (EXP-CONF-004): the shipped minimal-sample model against
     * the observer-only all-inlier refit of the same production match set, decomposed through the
     * same {@code RigidMotionDecomposition} at the processed-frame centre, so
     * {@code delta_rotation_deg} is defined on exactly the quantity navigation integrates.
     * {@code readout_time_ns} times the decomposition-and-delta arithmetic alone;
     * {@code refit_time_ns} is the probe's timing of the {@code fitModel} call alone.
     */
    private void writeDiagnosticRefitRow(BufferedWriter out, int frameIndex, String event,
                                         int frameWidth, int frameHeight) throws IOException {
        if (!diagnosticRefit.hasModels()) {
            return;   // absence is preserved rather than written as zeros (DEC-VO-001)
        }
        Homography2D_F64 sh = diagnosticRefit.shippedModel();
        boolean refitOk = diagnosticRefit.hasDiagnosticRefit();
        Homography2D_F64 r = refitOk ? diagnosticRefit.diagnosticRefitModel() : null;
        double cx = frameWidth / 2.0, cy = frameHeight / 2.0;

        StringBuilder row = new StringBuilder(512);
        row.append(frameIndex).append(',').append(event).append(',')
           .append(diagnosticRefit.inlierCount()).append(',').append(refitOk ? 1 : 0).append(',')
           .append(sh.a11).append(',').append(sh.a12).append(',').append(sh.a13).append(',')
           .append(sh.a21).append(',').append(sh.a22).append(',').append(sh.a23).append(',')
           .append(sh.a31).append(',').append(sh.a32).append(',').append(sh.a33).append(',');
        if (refitOk) {
            long readoutStart = System.nanoTime();
            double[] js = org.boofcv.stitching.MotionModelSupport.HOMOGRAPHY
                    .jacobian(sh, cx, cy, null);
            double[] jr = org.boofcv.stitching.MotionModelSupport.HOMOGRAPHY
                    .jacobian(r, cx, cy, null);
            org.boofcv.stitching.RigidMotionDecomposition ds =
                    new org.boofcv.stitching.RigidMotionDecomposition()
                            .set(js[0], js[1], js[2], js[3]);
            org.boofcv.stitching.RigidMotionDecomposition dr =
                    new org.boofcv.stitching.RigidMotionDecomposition()
                            .set(jr[0], jr[1], jr[2], jr[3]);
            georegression.struct.point.Point2D_F64 ps =
                    org.boofcv.stitching.MotionModelSupport.HOMOGRAPHY.apply(sh, cx, cy, null);
            georegression.struct.point.Point2D_F64 pr =
                    org.boofcv.stitching.MotionModelSupport.HOMOGRAPHY.apply(r, cx, cy, null);
            double fsx = ps.x - cx, fsy = ps.y - cy;
            double frx = pr.x - cx, fry = pr.y - cy;
            double magS = Math.hypot(fsx, fsy), magR = Math.hypot(frx, fry);
            double dRot = wrapDegrees(
                    Math.toDegrees(ds.getRotationRad() - dr.getRotationRad()));
            double dFlow = Math.hypot(ps.x - pr.x, ps.y - pr.y);
            Double dDir = null;
            if (magS >= DIR_MIN_FLOW_PROCESSED_PX && magR >= DIR_MIN_FLOW_PROCESSED_PX) {
                double cos = (fsx * frx + fsy * fry) / (magS * magR);
                dDir = Math.toDegrees(Math.acos(Math.max(-1.0, Math.min(1.0, cos))));
            }
            double dLogScale = ds.logUniformScale() - dr.logUniformScale();
            double dPersp = org.boofcv.stitching.MotionModelSupport.HOMOGRAPHY
                            .perspectiveMagnitude(sh, frameWidth, frameHeight)
                    - org.boofcv.stitching.MotionModelSupport.HOMOGRAPHY
                            .perspectiveMagnitude(r, frameWidth, frameHeight);
            long readoutTimeNs = System.nanoTime() - readoutStart;

            row.append(r.a11).append(',').append(r.a12).append(',').append(r.a13).append(',')
               .append(r.a21).append(',').append(r.a22).append(',').append(r.a23).append(',')
               .append(r.a31).append(',').append(r.a32).append(',').append(r.a33).append(',')
               .append(Math.toDegrees(ds.getRotationRad())).append(',')
               .append(Math.toDegrees(dr.getRotationRad())).append(',')
               .append(Math.abs(dRot)).append(',')
               .append(dFlow).append(',')
               .append(dDir != null ? Double.toString(dDir) : "").append(',')
               .append(Math.abs(dLogScale)).append(',')
               .append(Math.abs(dPersp)).append(',')
               .append(diagnosticRefit.diagnosticRefitTimeNs()).append(',')
               .append(readoutTimeNs);
        } else {
            // refit failed: shipped model persisted, every refit-dependent column empty
            row.append(",,,,,,,,,,,,,,,,")
               .append(diagnosticRefit.diagnosticRefitTimeNs()).append(',');
        }
        out.write(row.toString());
        out.newLine();
    }

    /**
     * {@code delta_rot_refit} for the frame just processed, from the observer-only refit probe:
     * {@code |rotation(H_shipped_minimal) − rotation(H_all_inlier_refit)|} in degrees, both read
     * through the {@code RIGID_MOTION} decomposition at the processed-frame centre — the same
     * quantity {@code diagnostic_refit.csv}'s {@code delta_rotation_deg} column carries
     * ({@code DEC-CONF-003}). {@code null} when the probe has no models or the refit failed this
     * frame: absence, never zero.
     */
    static Double diagnosticDeltaRotationDeg(org.boofcv.stitching.RefinementDiagnostics probe,
                                             int frameWidth, int frameHeight) {
        if (!probe.hasModels() || !probe.hasDiagnosticRefit()) {
            return null;
        }
        Homography2D_F64 sh = probe.shippedModel();
        Homography2D_F64 r = probe.diagnosticRefitModel();
        double cx = frameWidth / 2.0, cy = frameHeight / 2.0;
        double[] js = org.boofcv.stitching.MotionModelSupport.HOMOGRAPHY.jacobian(sh, cx, cy, null);
        double[] jr = org.boofcv.stitching.MotionModelSupport.HOMOGRAPHY.jacobian(r, cx, cy, null);
        org.boofcv.stitching.RigidMotionDecomposition ds =
                new org.boofcv.stitching.RigidMotionDecomposition().set(js[0], js[1], js[2], js[3]);
        org.boofcv.stitching.RigidMotionDecomposition dr =
                new org.boofcv.stitching.RigidMotionDecomposition().set(jr[0], jr[1], jr[2], jr[3]);
        return Math.abs(wrapDegrees(Math.toDegrees(ds.getRotationRad() - dr.getRotationRad())));
    }

    /**
     * A CSV cell for a possibly-absent number. {@code NaN} means "this quantity does not exist for
     * this frame" -- no authoritative heading has ever been available, say -- and an empty cell says
     * that where the token {@code NaN} would invite a reader to parse it as a value.
     */
    static String num(double v) {
        return Double.isNaN(v) ? "" : Double.toString(v);
    }

    /** Wraps an angle difference into {@code (-180, 180]}. */
    static double wrapDegrees(double deg) {
        double d = ((deg + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
        return d == -180.0 ? 180.0 : d;
    }

    /**
     * Guards the [0, 360) invariant against the floating-point rounding boundary that can produce
     * exactly 360.0 (research-log 2026-08-12 addendum, found in the Python normaliser). Pose3D
     * already produces clockwise-positive degrees; this applies no North offset — the heading
     * datum stays the start frame, a gauge freedom the Sim(2) alignment removes
     * (contracts/run-record.md).
     */
    static double canonicalYaw(double yawDeg) {
        double normalized = ((yawDeg % 360.0) + 360.0) % 360.0;
        if (normalized >= 360.0) {
            normalized = 0.0;
        }
        return normalized;
    }
}
