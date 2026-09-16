package org.boofcv.evaluation;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * {@code VoRunnerApp}'s config-file schema. Not a cross-boundary artifact (contracts/ only
 * covers dataset and run-record), so field choices here are local to this entry point.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class VoRunnerConfig {

    @JsonProperty("dataset_dir")
    public String datasetDir;

    @JsonProperty("output_dir")
    public String outputDir = "runs";

    @JsonProperty("run_id")
    public String runId; // null => generated from estimator_id + timestamp

    @JsonProperty("estimator_id")
    public String estimatorId = "stitching-vo";

    @JsonProperty("estimator_version")
    public String estimatorVersion = "unknown";

    @JsonProperty("is_target_hardware")
    public boolean isTargetHardware = false;

    // StitchingFactory.Builder parameters, using the contract's estimator_config JSON key names
    // (contracts/run-record.md), not necessarily the Java builder method names.

    @JsonProperty("shrinkScale")
    public double shrinkScale = 0.5;

    @JsonProperty("minDistanceFromBorder")
    public int minDistanceFromBorder = 10;

    @JsonProperty("detectorType")
    public String detectorType = "SHI_TOMASI";

    // NOTE (gap G17, research log 2026-08-14): this does NOT bound the number of tracks BoofCV's
    // KLT tracker actually maintains -- see StitchingFactory.Builder#buildGray. It only shapes the
    // detector's internal candidate ranking. Kept as-is; not fixed here (a tracking-behaviour
    // change needs literature-first treatment, not a config-plumbing pass).
    @JsonProperty("maxFeatures")
    public int maxFeatures = 300;

    @JsonProperty("detectorRadius")
    public int detectorRadius = 3;

    @JsonProperty("detectorThreshold")
    public double detectorThreshold = 1.0;

    @JsonProperty("kltPyramidLevels")
    public int kltPyramidLevels = 4;

    @JsonProperty("kltFeatureRadius")
    public int kltFeatureRadius = 3;

    @JsonProperty("ransacIterations")
    public int ransacIterations = 220;

    @JsonProperty("inlierThresholdSq")
    public double inlierThresholdSq = 3.0;

    @JsonProperty("outlierPrune")
    public int outlierPrune = 2;

    @JsonProperty("absoluteMinimumTracks")
    public int absoluteMinimumTracks = 30;

    @JsonProperty("respawnTrackFraction")
    public double respawnTrackFraction = 0.6;

    @JsonProperty("respawnCoverageFraction")
    public double respawnCoverageFraction = 0.5;

    @JsonProperty("refineEstimate")
    public boolean refineEstimate = false;

    @JsonProperty("maxJumpFraction")
    public double maxJumpFraction = 0.55;

    // Not a StitchingFactory/estimator parameter -- applied once per frame in
    // DirectoryFrameSource before the estimator ever sees the image. Default 1 = no-op, so
    // every existing config (A0, sim-square, golden) is unaffected. Added 2026-08-14 for
    // EXP-002, whose pre-registered Configuration specifies "1224 x 1024, an exact 2x2 box
    // downsample of the native 2448 x 2048" -- discovered missing (no downsample step existed
    // anywhere in the capture pipeline) while preparing Execution A, before any VO run, so
    // adding it is fulfilling the already-specified design rather than changing it after seeing
    // a result. Must exactly divide both the dataset's declared image_width and image_height,
    // or DirectoryFrameSource raises rather than silently resampling with interpolation.
    @JsonProperty("downsampleFactor")
    public int downsampleFactor = 1;

    // EXP-VO-002 / EXP-VO-009. Selects the 2D motion model estimated per frame:
    //   "homography"  8-DoF projective, BoofCV, the shipped default -- so every pre-existing
    //                 config is byte-for-byte unaffected
    //   "affine"      6-DoF, BoofCV, experimental (DEC-VO-002, LIT-VO-001)
    //   "similarity"  4-DoF, this repository's direct Umeyama fit, experimental
    //                 (DEC-VO-006, LIT-VO-005) -- the model LIT-VO-003 s2 shows is physically
    //                 exact for a nadir camera over a plane, estimated directly rather than
    //                 projected from a richer fit
    // The three are nested (similarity < affine < projective), so switching between them is a
    // model-class ablation, not a change of family. Nothing in production selects the latter two.
    @JsonProperty("motion_model")
    public String motionModel = "homography";

    // DEC-VO-003 / EXP-VO-004 / DEC-VO-008. Which computation publishes est_x/est_y/est_yaw_deg:
    //   "rigid_motion"  the DEFAULT since 2026-08-28 (DEC-VO-008) -- RigidNavigationState, the
    //                   per-frame rigid reduction; the readout every VO accuracy figure this
    //                   project quotes comes from
    //   "mosaic_legacy" the pre-DEC-VO-003 mosaic-derived readout, retained for reproducing
    //                   pre-2026-08-28 baselines and for diagnostics
    //   "logical_frame" full composition of the estimator's transform -- REJECTED as a navigation
    //                   representation, kept as the diagnostic that shows why
    // Keep this string in step with NavigationSource.DEFAULT; NavigationSourceDefaultTest asserts
    // they agree, and VoRunnerApp.parseNavigationSource routes null/blank to the same constant.
    //
    // A config that must reproduce a pre-2026-08-28 committed run record has to name
    // "mosaic_legacy" explicitly. The five committed configs that relied on the old implicit
    // default were made explicit when the default moved, so none of them changed meaning.
    @JsonProperty("navigation_source")
    public String navigationSource = "rigid_motion";

    // EXP-VO-004. When true, VoRunner additionally writes logical_transform.csv next to
    // frames.csv: the per-frame logical transform G_k : L -> C_k in homography form plus the
    // logical pose, regardless of which source publishes the pose. Not part of the run-record
    // contract (contracts/run-record.md); readers of frames.csv are unaffected.
    @JsonProperty("logical_transform_sidecar")
    public boolean logicalTransformSidecar = false;

    // DEC-CONF-002 / EXP-CONF-001. Path to a schema-2.0.0 calibration configuration
    // (evaluation/confidence_configs/*.json), resolved as given (like dataset_dir). When set,
    // VoRunner captures the per-frame signal block and verdict and the run record is written at
    // v1.1.0 (contracts/run-record-v1.1.md). Default null = NO confidence capture and a
    // byte-identical v1.0.0 record -- capture is opt-in, never a silent default (tasks.md T026).
    // Requires motion_model "homography" (the instrumented construction path exists for the
    // shipped model only) and is mutually exclusive with refinement_sidecar and --motion-timing
    // (each replaces the low-level estimator with its own observing subclass; stacking them is
    // untested and refused rather than silently combined).
    @JsonProperty("confidence_calibration")
    public String confidenceCalibration;

    // EXP-VO-010. When true, VoRunner additionally writes refinement.csv next to frames.csv: for
    // every accepted frame, RANSAC's winning MINIMAL-SAMPLE model, the model actually shipped, and
    // the inlier count the refiner was given. Observational only -- no model is re-estimated
    // (DEC-VO-001's provenance rule). With refineEstimate = false the two models are identical by
    // construction, which makes every such run a free correctness check on the probe itself.
    // Not part of the run-record contract (contracts/run-record.md); readers of frames.csv are
    // unaffected. Only "homography" and "affine" support it -- the similarity model cannot refine
    // (GenerateSimilarity2D implements ModelGenerator only), so the flag is rejected for it rather
    // than silently ignored.
    @JsonProperty("refinement_sidecar")
    public boolean refinementSidecar = false;

    // EXP-CONF-004. When true, VoRunner additionally writes diagnostic_refit.csv next to
    // frames.csv: for every frame with an accepted estimate, the SHIPPED (minimal-sample) model,
    // an OBSERVER-ONLY all-inlier refit of the estimator's own production match set, their
    // RIGID_MOTION-readout deltas (rotation, centre flow, flow direction where defined,
    // log-scale, perspective), the inlier count, and the refit's own wall time. The refined
    // model is NEVER shipped: the diagnostic refiner is a separate instance the estimator never
    // sees, so navigation, pose accumulation, restart/recenter logic, feature lifecycle and
    // reference selection are bit-identical to a run without the flag
    // (DiagnosticRefitEquivalenceTest). Requires refineEstimate = false -- the signal is
    // defined as "how far the shipped minimal hypothesis sits from the all-inlier estimate",
    // which stops existing once the refit itself ships. Mutually exclusive with
    // confidence_calibration for the same construction-path reason as refinement_sidecar; only
    // "homography" and "affine" support it.
    @JsonProperty("diagnostic_refit_sidecar")
    public boolean diagnosticRefitSidecar = false;

    // DEC-INT-001 (relocalization P0; amended 2026-09-07). When true, VoRunner additionally drives
    // the alignment layer (org.boofcv.relocalization.NavigationAligner) over the METRIC state --
    // segmentRelativePose (metres, East/North), the authoritative heading and the VO's own validity
    // flags (DEC-VO-009/-010) -- and the processFrame boolean, and writes alignment_frames.csv +
    // alignment_events.csv next to frames.csv: the segment-local position, the persistent position
    // (empty while the alignment is UNKNOWN after a hard loss or a translation dropout),
    // global_position_valid and heading_valid SEPARATELY, segment/epoch ids and the non-decaying
    // lineage ledger. Purely observational -- the estimator is untouched and frames.csv is
    // byte-identical with the flag off or on. REQUIRES metric_readout.enabled AND
    // heading_readout.enabled (VoRunnerApp.checkRelocalizationVsNavigationContract): the layer has
    // no contract for the pixel / visual-yaw readout and is refused without them. Whether
    // METRIC_LOCAL is also the PUBLISHED navigation source is irrelevant to it.
    @JsonProperty("alignment_sidecar")
    public boolean alignmentSidecar = false;

    // DEC-INT-002 / DEC-INT-003 (relocalization P1). Path to a RelocalizationConfig JSON. When
    // set, VoRunner drives the full RelocalizationPipeline (aligner + trusted-reference memory +
    // exact dual-view retrieval + scheduler + region tracker + acceptance gate) over the same
    // metric handoff, the processFrame boolean, the optional skyline feed below and the optional
    // delta_rot_refit signal, and writes alignment_frames.csv / alignment_events.csv /
    // relocalization_manifest.json next to frames.csv. Observational with respect to the
    // estimator: frames.csv is byte-identical with or without it. Implies alignment_sidecar and
    // its metric+heading requirement. If the relocalization config sets
    // instability_threshold_deg, diagnostic_refit_sidecar must be true too: delta_rot_refit is
    // taken from that observer-only probe (the CONF-validated path) and from nowhere else.
    @JsonProperty("relocalization_config")
    public String relocalizationConfig;

    // DEC-INT-002 / DEC-INT-003. Path to the canonical skyline_profiles.csv exported by the
    // validated Python SKY pipeline (evaluation/tools/int/export_skyline_profiles.py) -- North
    // profiles, plus synchronised West profiles when the capture carried them -- keyed by this
    // dataset's frame_index (paired by explicit map, index offset, or nearest capture time with
    // the residual recorded per row). Optional: without it the pipeline runs with no eligible
    // observation, so it schedules and logs but never retrieves, inserts or re-anchors. A feed
    // none of whose frames fall inside the dataset is refused as misaligned.
    @JsonProperty("skyline_profiles")
    public String skylineProfiles;

    // DEC-VO-007 D3, runtime port (2026-09-06). Optional and absent by default, so every config
    // written before that date is byte-for-byte unaffected: no height channel is built, no metric
    // state exists, the published pose stays in first-frame pixels, and no extra file is written.
    //
    // When enabled, VoRunner additionally writes metric_track.csv next to frames.csv (the runtime
    // metric local pose, plus which height sample each frame was converted with and how fresh it
    // was), and feeds the recorded height stream into the estimator causally, sample by sample.
    // Not part of the run-record contract (contracts/run-record.md); readers of frames.csv are
    // unaffected unless publish_as_navigation_source is also set, which changes est_x/est_y from
    // pixels to metres and is therefore an explicit, run-id-visible migration.
    @JsonProperty("metric_readout")
    public MetricReadoutSettings metricReadout;

    // Optional heading_readout block (DEC-VO-010), also absent by default.
    //
    // When enabled, an EXTERNAL heading channel owns the navigation direction: the metric pose's
    // yaw becomes the authoritative navigation heading (clockwise from North) and every metric
    // translation increment is rotated by it instead of by the visually integrated yaw, which is
    // retained and logged as a diagnostic. Requires metric_readout: heading rotates a metric
    // increment, and without a height channel there is none.
    @JsonProperty("heading_readout")
    public HeadingReadoutSettings headingReadout;

    // SYNTHETIC hard-loss injection (2026-09-08). Absent by default, so every existing config is
    // unaffected. Present only in TEST / EXPERIMENT configs: natural hard loss is rare and cannot
    // be scheduled by hand, yet the whole downstream recovery contract (segment termination,
    // UNKNOWN translation across the gap, position invalidity, heading survival, relocalization as
    // the only route back to a valid position) hangs off it.
    //
    // Each declared frame takes the estimator's own hard-loss branch on that frame -- the same
    // branch a tracker failure takes, not a parallel one -- and every resulting loss is labelled
    // loss_source = synthetic in the run manifest and the alignment sidecar, so a forced loss can
    // never be counted as observed estimator failure in a later analysis. Nothing outside this
    // block can arm it, and the two shipped applications do not read this config at all.
    @JsonProperty("synthetic_hard_loss")
    public SyntheticHardLossSettings syntheticHardLoss;

    /**
     * Declared synthetic hard losses. {@code acknowledge_synthetic} must be {@code true}: the
     * block is a deliberate departure from what the estimator observed, and it is spelled out
     * rather than inferred from the presence of a frame list.
     */
    public static class SyntheticHardLossSettings {

        /** Must be explicitly true; a frame list alone does not arm anything. */
        @JsonProperty("acknowledge_synthetic")
        public boolean acknowledgeSynthetic = false;

        /**
         * Frame indices, into this dataset, on which to force a hard loss. Frame 0 is refused: it
         * is the reference frame, which integrates nothing, so there is no segment to terminate.
         */
        @JsonProperty("frames")
        public int[] frames;

        /** Free text recorded in the manifest saying why these losses were injected. */
        @JsonProperty("note")
        public String note;

        public java.util.SortedSet<Integer> validated(int frameCount) {
            if (!acknowledgeSynthetic) {
                throw new IllegalArgumentException("synthetic_hard_loss needs "
                        + "\"acknowledge_synthetic\": true — a forced loss is not an observed "
                        + "estimator failure and the config must say so explicitly");
            }
            if (frames == null || frames.length == 0) {
                throw new IllegalArgumentException("synthetic_hard_loss.frames must list at least "
                        + "one frame index");
            }
            java.util.SortedSet<Integer> out = new java.util.TreeSet<>();
            for (int f : frames) {
                if (f <= 0) {
                    throw new IllegalArgumentException("synthetic_hard_loss.frames must be > 0 "
                            + "(frame 0 is the reference frame and integrates nothing, so it has "
                            + "no segment to terminate); got " + f);
                }
                if (f >= frameCount) {
                    throw new IllegalArgumentException("synthetic_hard_loss.frames has frame " + f
                            + " but the dataset has " + frameCount + " frames");
                }
                if (!out.add(f)) {
                    throw new IllegalArgumentException("synthetic_hard_loss.frames repeats frame "
                            + f + "; one loss per frame");
                }
            }
            return out;
        }
    }
}
