package org.boofcv.evaluation;

import boofcv.abst.feature.detect.interest.ConfigPointDetector;
import boofcv.abst.feature.detect.interest.PointDetectorTypes;
import boofcv.abst.sfm.d2.ImageMotion2D;
import boofcv.abst.tracker.PointTracker;
import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.factory.sfm.FactoryMotion2D;
import boofcv.factory.tracker.FactoryPointTracker;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.ImageType;
import com.fasterxml.jackson.databind.ObjectMapper;
import georegression.struct.affine.Affine2D_F64;
import georegression.struct.homography.Homography2D_F64;
import org.boofcv.stitching.MotionModelStitchingEstimator;
import org.boofcv.stitching.MotionModelSupport;
import org.boofcv.stitching.NavigationSource;
import org.boofcv.stitching.Sim2_F64;
import org.boofcv.stitching.StitchingFactory;
import org.boofcv.stitching.diagnostics.MotionTimingProbe;

import javax.annotation.Nullable;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Config-file-driven entry point that drives {@link StitchingEstimator} over a dataset directory
 * and writes a conforming run record (contracts/run-record.md, quickstart Scenario 3/4).
 *
 * <p>Usage: {@code --config <path/to/config.json>}
 */
public final class VoRunnerApp {

    public static void main(String[] args) throws IOException {
        String configPath = parseConfigArg(args);
        if (configPath == null) {
            System.err.println("Usage: VoRunnerApp --config <path/to/config.json> "
                    + "[--dataset-dir <path>] [--run-id <id>] [--motion-timing <path.csv>]");
            System.exit(1);
            return;
        }

        VoRunnerConfig config = new ObjectMapper().readValue(new java.io.File(configPath), VoRunnerConfig.class);

        // Overrides exist because the dataset imagery is not stored with the configs; pointing at a
        // local copy on the command line keeps the committed config free of machine-specific paths.
        String datasetOverride = parseArg(args, "--dataset-dir");
        if (datasetOverride != null) {
            config.datasetDir = datasetOverride;
        }
        String runIdOverride = parseArg(args, "--run-id");
        if (runIdOverride != null) {
            config.runId = runIdOverride;
        }
        run(config, System.out, parseArg(args, "--motion-timing"));
    }

    static String parseConfigArg(String[] args) {
        return parseArg(args, "--config");
    }

    static String parseArg(String[] args, String flag) {
        for (int i = 0; i < args.length - 1; i++) {
            if (flag.equals(args[i])) {
                return args[i + 1];
            }
        }
        return null;
    }

    /** Runs the configured VO capture and returns the run directory that was written. */
    public static Path run(VoRunnerConfig config, PrintStream log) throws IOException {
        return run(config, log, null);
    }

    /**
     * @param motionTimingCsv when non-null, per-frame <em>motion-estimation</em> time (KLT +
     *                        RANSAC, excluding mosaic rendering) is written here. EXP-VO-002.
     */
    public static Path run(VoRunnerConfig config, PrintStream log, String motionTimingCsv)
            throws IOException {
        Path datasetDir = Paths.get(config.datasetDir);
        DatasetDescriptor dataset = DatasetDescriptor.load(datasetDir);

        // DEC-CONF-002: confidence capture builds the residual-instrumented stack (the same
        // parameters through StitchingFactory.Builder; equivalence is bitwise-tested), so it is
        // mutually exclusive with the other two motion-stack observers.
        org.boofcv.confidence.CalibrationConfig calibration = null;
        org.boofcv.stitching.InstrumentedStitching<GrayF32> instrumented = null;
        if (config.diagnosticRefitSidecar && config.refineEstimate) {
            throw new IllegalArgumentException("diagnostic_refit_sidecar requires refineEstimate "
                    + "= false: the signal is the disagreement between the shipped minimal-sample "
                    + "model and the all-inlier refit, which stops existing once the refit itself "
                    + "ships. Use refinement_sidecar for refineEstimate = true runs.");
        }
        if (config.confidenceCalibration != null) {
            if (config.refinementSidecar || config.diagnosticRefitSidecar) {
                throw new IllegalArgumentException("confidence_calibration and the refinement/"
                        + "diagnostic-refit sidecars each replace the low-level estimator with an "
                        + "observing subclass; stacking them is untested and refused. Run them "
                        + "separately.");
            }
            if (motionTimingCsv != null) {
                throw new IllegalArgumentException("confidence_calibration and --motion-timing "
                        + "cannot be combined; the timing probe wraps a motion estimator the "
                        + "instrumented build path constructs internally. Run them separately.");
            }
            if (MotionModel.parse(config.motionModel) != MotionModel.HOMOGRAPHY) {
                throw new IllegalArgumentException("confidence_calibration requires motion_model "
                        + "\"homography\": the instrumented construction path (DEC-VO-001) exists "
                        + "for the shipped model only, and EXP-CONF-001's scope is the shipped "
                        + "configuration.");
            }
            checkConfidenceCalibrationVsRuntimeFeatures(config);
            calibration = org.boofcv.confidence.CalibrationConfig.load(
                    Paths.get(config.confidenceCalibration));
            instrumented = StitchingFactory.builder()
                    .detector(detectorConfig(config))
                    .kltLevels(config.kltPyramidLevels)
                    .kltRadius(config.kltFeatureRadius)
                    .motionMaxIterations(config.ransacIterations)
                    .inlierThresholdSq(config.inlierThresholdSq)
                    .outlierPrune(config.outlierPrune)
                    .motionMinFeatures(config.absoluteMinimumTracks)
                    .respawnTrackFraction(config.respawnTrackFraction)
                    .respawnCoverageFraction(config.respawnCoverageFraction)
                    .refineEstimate(config.refineEstimate)
                    .stitchOverlapThreshold(config.maxJumpFraction)
                    .buildGrayInstrumented();
        }

        // DEC-INT-002: the relocalization pipeline's instability escalation reads delta_rot_refit
        // from the observer-only diagnostic refit probe and nowhere else. It is refused, not
        // silently disabled, when the probe is not requested — a run that declared tau_I and never
        // received the signal would be the EXP-SKY-007 "lag-0 diagnostic" mistake all over again.
        checkRelocalizationVsNavigationContract(config);
        org.boofcv.relocalization.RelocalizationConfig relocalization = null;
        if (config.relocalizationConfig != null) {
            relocalization = org.boofcv.relocalization.RelocalizationConfig.load(
                    Paths.get(config.relocalizationConfig));
            if (relocalization.instabilityThresholdDeg != null && !config.diagnosticRefitSidecar) {
                throw new IllegalArgumentException("relocalization_config sets "
                        + "instability_threshold_deg, so delta_rot_refit must be produced: set "
                        + "diagnostic_refit_sidecar = true (the CONF-validated observer-only probe). "
                        + "Refused rather than run without the signal.");
            }
            if (config.confidenceCalibration != null) {
                throw new IllegalArgumentException("relocalization_config and "
                        + "confidence_calibration cannot be combined: the instrumented construction "
                        + "path has no refit probe. Run them separately.");
            }
        }

        // EXP-VO-010: when the refinement sidecar is asked for, the motion estimator is built with
        // the observing tracker key attached and handed in as an override, so there is still exactly
        // one estimator and one construction path. The timing probe, if also requested, wraps it.
        StitchingFactory.ProbedMotion<?> probed =
                (config.refinementSidecar || config.diagnosticRefitSidecar)
                        ? buildProbedMotion(config) : null;
        ImageMotion2D<GrayF32, ?> baseMotion =
                probed != null ? probed.motion() : (motionTimingCsv != null ? buildMotion(config) : null);

        MotionTimingProbe<GrayF32, ?> timingProbe =
                motionTimingCsv != null ? new MotionTimingProbe<>(baseMotion) : null;
        ImageMotion2D<GrayF32, ?> motionOverride = timingProbe != null ? timingProbe : baseMotion;
        MotionModelStitchingEstimator<GrayF32, ?> estimator;
        if (instrumented != null) {
            org.boofcv.stitching.StitchingEstimator<GrayF32> e =
                    new org.boofcv.stitching.StitchingEstimator<>(instrumented.stitch());
            e.setShrinkScale(config.shrinkScale);
            e.setMinDistanceFromBorder(config.minDistanceFromBorder);
            e.setNavigationSource(parseNavigationSource(config.navigationSource));
            estimator = e;
        } else {
            estimator = motionOverride != null
                    ? buildEstimator(config, motionOverride) : buildEstimator(config);
        }

        DirectoryFrameSource frameSource = new DirectoryFrameSource(dataset, config.downsampleFactor);

        String runId = config.runId != null ? config.runId
                : config.estimatorId + "-" + DateTimeFormatter.ISO_INSTANT.format(Instant.now());

        RunManifest manifest = new RunManifest();
        manifest.runId = runId;
        manifest.datasetId = dataset.datasetId;
        manifest.datasetRevision = dataset.datasetRevision;
        manifest.estimatorId = config.estimatorId;
        manifest.estimatorVersion = config.estimatorVersion;
        manifest.evaluatorCaptureVersion = config.estimatorVersion;
        manifest.estimatorConfig = appliedEstimatorConfig(config);
        manifest.environment = Environment.captureCurrent();
        manifest.environment.isTargetHardware = config.isTargetHardware;
        manifest.runTimestamp = Instant.now().toString();

        Map<String, Object> confidenceBlock = null;
        if (calibration != null) {
            manifest.schemaVersion = RunManifest.SCHEMA_VERSION_CONFIDENCE;
            Map<String, Object> captureContext = new LinkedHashMap<>();
            if (dataset.imageWidth != null && dataset.imageHeight != null) {
                // The processed grid: every per-frame pixel quantity, the coverage grid and the
                // residuals all live post-downsample (EXP-VO-001 R4's resolution warning).
                captureContext.put("image_width", dataset.imageWidth / config.downsampleFactor);
                captureContext.put("image_height", dataset.imageHeight / config.downsampleFactor);
            }
            captureContext.put("motion_model",
                    MotionModel.parse(config.motionModel).name().toLowerCase());

            // Amendment A3: refuse a capture whose context does not satisfy the calibration's
            // binding, before a single frame is processed.
            Map<String, Object> bindingContext = new LinkedHashMap<>(manifest.estimatorConfig);
            bindingContext.putAll(captureContext);
            calibration.checkBinding(bindingContext, "capture of dataset '" + dataset.datasetId + "'");

            confidenceBlock = new LinkedHashMap<>();
            confidenceBlock.put("calibration_id", calibration.calibrationId());
            confidenceBlock.put("calibration_digest", calibration.digest());
            confidenceBlock.put("calibration_validated", calibration.validated());
            confidenceBlock.put("probability_semantics", calibration.probabilitySemantics());
            confidenceBlock.put("residual_diagnostics_available", true);
            confidenceBlock.put("residual_source",
                    "org.boofcv.stitching.MotionResidualDiagnostics (DEC-VO-001)");
            confidenceBlock.put("signal_window_w", calibration.windowW());
            confidenceBlock.put("signal_warmup_m", calibration.warmupM());
            confidenceBlock.put("capture_context", captureContext);
            manifest.confidence = confidenceBlock;
        }

        Path runDir = Paths.get(config.outputDir).resolve(runId);
        Map<String, Object> skylinePairing = null;
        java.util.SortedSet<Integer> syntheticLossFrames = new java.util.TreeSet<>();
        VoRunner runnerForManifest = null;
        RunRecordWriter writer = new RunRecordWriter(runDir, manifest, dataset.frames.size(),
                calibration != null);
        try {
            VoRunner runner = new VoRunner(estimator, frameSource);
            runnerForManifest = runner;
            if (config.logicalTransformSidecar) {
                runner.setLogicalTransformCsv(runDir.resolve("logical_transform.csv"));
            }
            if (probed != null && config.refinementSidecar) {
                runner.setRefinementCsv(runDir.resolve("refinement.csv"), probed.diagnostics());
            }
            if (probed != null && config.diagnosticRefitSidecar) {
                runner.setDiagnosticRefitCsv(runDir.resolve("diagnostic_refit.csv"),
                        probed.diagnostics());
            }
            org.boofcv.relocalization.SkylineProfileFile profiles = null;
            if (relocalization != null) {
                if (config.skylineProfiles != null) {
                    profiles = org.boofcv.relocalization.SkylineProfileFile.load(
                            Paths.get(config.skylineProfiles), relocalization);
                    int frameCount = dataset.frames.size();
                    java.util.List<Integer> outside = profiles.framesOutside(frameCount);
                    if (profiles.matchedCount(frameCount) == 0) {
                        throw new IllegalArgumentException("skyline_profiles " + config.skylineProfiles
                                + ": none of its " + profiles.size() + " frames fall inside this "
                                + "dataset's " + frameCount + " frames — a misaligned feed, refused");
                    }
                    log.println("Skyline profiles (schema " + profiles.schemaVersion() + ", capture identity "
                            + profiles.pairingIdentity() + " via " + profiles.pairingMechanism()
                            + (profiles.exactCaptureIdentity() ? ""
                            : " — LEGACY/APPROXIMATE or declared pairing, not the simulator's own vo_frame_id")
                            + "): " + profiles.size() + " rows, " + profiles.validCount() + " North-valid, "
                            + profiles.westAvailableCount() + " with a West view ("
                            + profiles.westValidCount() + " valid), " + profiles.matchedCount(frameCount)
                            + " inside the dataset" + (outside.isEmpty() ? ""
                            : ", " + outside.size() + " OUTSIDE it (first " + outside.get(0) + ")")
                            + (profiles.syncRejectedCount() > 0 ? ", " + profiles.syncRejectedCount()
                            + " beyond the sync tolerance" : "")
                            + String.format(", max |sync_dt| %.4f s", profiles.maxAbsSyncDtS()));
                }
                skylinePairing = profiles == null ? null : Map.of(
                        "identity", profiles.pairingIdentity(), "mechanism", profiles.pairingMechanism(),
                        "schema_version", profiles.schemaVersion(), "rows", profiles.size(),
                        "north_valid", profiles.validCount(), "west_available", profiles.westAvailableCount(),
                        "west_valid", profiles.westValidCount(), "sync_rejected", profiles.syncRejectedCount(),
                        "max_abs_sync_dt_s", profiles.maxAbsSyncDtS());
                runner.setRelocalization(
                        new org.boofcv.relocalization.RelocalizationPipeline(relocalization),
                        profiles, runDir);
                if (relocalization.instabilityThresholdDeg != null) {
                    runner.setInstabilityProbe(probed.diagnostics());
                }
            } else if (config.alignmentSidecar) {
                runner.setAlignmentSidecar(runDir);
            }
            if (config.syntheticHardLoss != null) {
                syntheticLossFrames = config.syntheticHardLoss.validated(frameSource.frameCount());
                runner.setSyntheticHardLossFrames(syntheticLossFrames);
                log.println("*** SYNTHETIC HARD LOSS ARMED on frames " + syntheticLossFrames
                        + " — this run does NOT report observed estimator failure on those frames. "
                        + "Every such loss is labelled loss_source=synthetic in the sidecar and in "
                        + "the run manifest. Test / experiment configurations only.");
            }
            if (instrumented != null) {
                runner.setConfidence(instrumented.diagnostics(),
                        new org.boofcv.confidence.SignalExtractor(
                                calibration.windowW(), calibration.warmupM()),
                        new org.boofcv.confidence.ConfidenceScorer(calibration));
            }
            if (estimator.isMetricReadoutEnabled()) {
                MetricReadoutSettings m = config.metricReadout;
                if (m.heightCsv == null || m.heightCsv.isBlank()) {
                    throw new IllegalArgumentException(
                            "metric_readout.enabled requires height_csv -- the recorded "
                            + "takeoff-relative height stream to replay into the channel");
                }
                boolean oracle = m.isOracleArm();
                if (oracle) {
                    log.println("WARNING: metric_readout.height_semantics = oracle_agl_diagnostic. "
                            + "This run uses the renderer's TRUE height above the imaged surface, "
                            + "which NO SENSOR on the author's platform supplies (LIT-VO-006 s2). "
                            + "It is a diagnostic arm and must be labelled as one wherever reported.");
                }
                runner.setMetricTrackCsv(runDir.resolve("metric_track.csv"),
                        HeightSampleStream.fromCsv(Paths.get(m.heightCsv), m.heightTimeColumn,
                                m.heightRelativeColumn, m.heightLatencyS, m.heightDecimation,
                                oracle));
            }
            if (estimator.isHeadingReadoutEnabled()) {
                HeadingReadoutSettings h = config.headingReadout;
                if (h.headingCsv == null || h.headingCsv.isBlank()) {
                    throw new IllegalArgumentException(
                            "heading_readout.enabled requires heading_csv -- the recorded external "
                            + "heading stream to replay into the channel");
                }
                if (h.headingColumn == null || h.headingColumn.isBlank()) {
                    throw new IllegalArgumentException(
                            "heading_readout.enabled requires heading_column -- naming it is how "
                            + "an airframe column and a camera column stay distinguishable");
                }
                if (h.semantics() == org.boofcv.stitching.metric.HeadingSemantics.FC_AHRS_BODY_COMPASS) {
                    log.println("NOTE: heading_readout.heading_semantics = fc_ahrs_body_compass. "
                            + "The column is the AIRFRAME's heading; the navigation camera's is "
                            + "derived through the declared delta_mount_deg = " + h.deltaMountDeg
                            + " deg (" + h.deltaMountProvenance + "). That calibration is external "
                            + "and platform-specific. GPS-independence of this stream is NOT "
                            + "established by the available data.");
                }
                runner.setHeadingStream(HeadingSampleStream.fromCsv(Paths.get(h.headingCsv),
                        h.headingTimeColumn, h.headingColumn, h.headingLatencyS,
                        h.headingDecimation));
            }
            runner.run(writer);
            if (confidenceBlock != null) {
                // T022: both counts are persisted per frame; this is the disagreement tally.
                confidenceBlock.put("inlier_count_disagreements", runner.inlierCountDisagreements());
            }
        } catch (RuntimeException | IOException e) {
            writer.markIncomplete();
            throw e;
        } finally {
            writer.close();
        }

        if (timingProbe != null) {
            writeMotionTiming(Paths.get(motionTimingCsv), timingProbe, config);
            log.println("Motion-estimation timing written to " + motionTimingCsv);
        }

        if (relocalization != null) {
            writeRelocalizationManifest(runDir, config, relocalization, skylinePairing,
                    runnerForManifest == null ? java.util.List.of() : runnerForManifest.naturalHardLosses(),
                    runnerForManifest == null ? java.util.List.of() : runnerForManifest.syntheticHardLossesFired(),
                    runnerForManifest == null ? java.util.List.of() : runnerForManifest.recenters());
            log.println("Relocalization sidecars written to " + runDir);
        }
        if (runnerForManifest != null && !syntheticLossFrames.isEmpty()) {
            log.println("Synthetic hard losses fired on frames "
                    + runnerForManifest.syntheticHardLossesFired() + "; observed (natural) losses on "
                    + runnerForManifest.naturalHardLosses() + "; recenters (not losses): "
                    + runnerForManifest.recenters().size());
        }

        log.println("Run record written to " + runDir);
        return runDir;
    }

    /**
     * {@code relocalization_manifest.json}: the policy actually applied, so a sidecar can always
     * be traced to its configuration (Principle VI). Kept beside, not inside, {@code manifest.json},
     * whose schema is the shared run-record contract.
     */
    private static void writeRelocalizationManifest(Path runDir, VoRunnerConfig config,
                                                    org.boofcv.relocalization.RelocalizationConfig rc,
                                                    @javax.annotation.Nullable Map<String, Object> skylinePairing,
                                                    java.util.List<Integer> natural,
                                                    java.util.List<Integer> synthetic,
                                                    java.util.List<Integer> recenters)
            throws IOException {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("relocalization_config_path", config.relocalizationConfig);
        m.put("relocalization_config", new ObjectMapper().convertValue(rc, Map.class));
        m.put("skyline_profiles_path", config.skylineProfiles);
        m.put("skyline_pairing", skylinePairing == null ? "none (no skyline profiles)" : skylinePairing);
        m.put("instability_source", rc.instabilityThresholdDeg == null ? "disabled"
                : "diagnostic_refit probe (EXP-CONF-004), delta_rotation_deg, every usable increment");
        m.put("pose_rule", "discrete_reference (p_reloc = p_ref, position only; DEC-INT-002)");
        Map<String, Object> contract = new LinkedHashMap<>();
        contract.put("input", "MetricNavigationState.segmentRelativePose (x=east m, y=north m) + navHeadingDeg "
                + "(deg CW from North, NaN when unavailable) + lastFrameUsable + segment index/gap flags");
        contract.put("alignment", "translation t_e in metres ENU; p_global = p_segment + t_e; "
                + "re-anchor t_e = p_ref - p_query; heading is never part of the alignment (DEC-INT-001 amendment 2026-09-07)");
        contract.put("scale", "not an alignment unknown: the height channel's (DEC-VO-009)");
        contract.put("rotation", "not an alignment unknown: the authoritative external heading's (DEC-VO-010)");
        contract.put("validity", "global_position_valid and heading_valid are separate; global_pose_valid is their AND");
        contract.put("published_navigation_source", resolveNavigationSource(config).name().toLowerCase()
                + " (irrelevant to the layer, which reads the metric state directly)");
        contract.put("reference_pose_schema", org.boofcv.relocalization.TrustedReference.POSE_SCHEMA);
        m.put("pose_contract", contract);
        m.put("matcher", matcherBlock(rc));
        m.put("dual_view", dualViewBlock(rc));
        m.put("snap_safety_units", org.boofcv.relocalization.SnapSafety.UNITS);
        Map<String, Object> snap = new LinkedHashMap<>();
        snap.put("drift_correction",
                "global position VALID before the query: e_before is defined and the quantity is "
                + "delta_e = e_after - e_before");
        snap.put("hard_loss_recovery",
                "global position UNKNOWN before the query: there is NO navigation e_before across "
                + "the gap and none is fabricated; the quantity is the absolute error the re-anchor "
                + "restored");
        snap.put("column", "alignment_events.csv:recovery_case — the two cases are never pooled");
        m.put("snap_safety_cases", snap);
        m.put("hard_loss", hardLossBlock(config, natural, synthetic, recenters));
        m.put("evidence_tier_note", "sidecars are mechanism logs; no navigation-performance claim");
        new ObjectMapper().enable(com.fasterxml.jackson.databind.SerializationFeature.INDENT_OUTPUT)
                .writeValue(runDir.resolve("relocalization_manifest.json").toFile(), m);
    }

    /**
     * Which matcher scored, at what lag freedom, and in which role ({@code DEC-INT-007}): the
     * variant alone does not say whether a C1 run is the pre-registered C1-4 fallback or the legacy
     * C1-32 comparison, and a C0 run must say explicitly that no alignment search took place.
     */
    static Map<String, Object> matcherBlock(org.boofcv.relocalization.RelocalizationConfig rc) {
        Map<String, Object> mb = new LinkedHashMap<>();
        mb.put("variant", rc.matcher);
        mb.put("declared_max_lag_samples", rc.maxLagSamples);
        mb.put("effective_max_lag_samples", rc.effectiveMaxLagSamples());
        mb.put("alignment_search", rc.effectiveMaxLagSamples() == 0
                ? "none: pointwise NCC at lag 0" : "bounded horizontal shift, |lag| <= "
                + rc.effectiveMaxLagSamples() + " samples, overlap >= " + rc.minOverlapFrac);
        mb.put("role", rc.matcherRole());
        mb.put("lag_semantics", "any lag is evidence metadata only; never applied to a position (DEC-INT-002)");
        return mb;
    }

    /** The dual-view / region / threshold policy as configured — shared by the live and replay manifests. */
    static Map<String, Object> dualViewBlock(org.boofcv.relocalization.RelocalizationConfig rc) {
        Map<String, Object> dual = new LinkedHashMap<>();
        dual.put("fusion_rule", rc.fusionRule);
        dual.put("match_threshold", rc.matchThreshold);
        dual.put("margin_threshold", rc.marginThreshold);
        dual.put("west_match_threshold", rc.effectiveWestMatchThreshold());
        dual.put("west_margin_threshold", rc.effectiveWestMarginThreshold());
        dual.put("temporal_confirmation", rc.temporalConfirmation);
        dual.put("confirm_queries", rc.confirmQueries);
        dual.put("region_rule", rc.regionRule);
        dual.put("region_gap_references", rc.regionGapReferences);
        dual.put("region_track_max_gap", rc.regionTrackMaxGap);
        // Three questions, three radii (2026-09-08). The legacy single value is recorded as such,
        // so a run made under one radius is never read as though three had been chosen.
        Map<String, Object> radii = new LinkedHashMap<>();
        radii.put("ambiguity_region_radius_m", rc.effectiveAmbiguityRadiusM());
        radii.put("dual_agreement_radius_m", rc.effectiveDualAgreementRadiusM());
        radii.put("temporal_region_radius_m", rc.effectiveTemporalRadiusM());
        radii.put("source", rc.regionRadiiAreLegacyMapped()
                ? "LEGACY: one declared region_radius_m = " + rc.regionRadiusM
                        + " m mapped to all three; not three independent choices"
                : (rc.effectiveAmbiguityRadiusM() == null
                        ? "not applicable (region_rule = id_gap)" : "declared independently"));
        radii.put("legacy_region_radius_m", rc.regionRadiusM);
        radii.put("note", "A = which references may not count as competing hypotheses; "
                + "B = whether North and West named one place; C = whether a confirmation chain "
                + "continues. None is a recognition radius and none is a permitted snap distance.");
        dual.put("region_radii", radii);
        dual.put("region_radius_m", rc.regionRadiusM);
        dual.put("single_view_acceptance", rc.singleViewAcceptance());
        dual.put("no_competing_hypothesis", "RETAIN (NO_COMPETING_HYPOTHESIS): a null margin — no "
                + "scored reference outside the winner's ambiguity region — never accepts. Absence "
                + "of ambiguity evidence is not positive evidence (2026-09-08).");
        dual.put("policy_status", rc.policyStatus());
        dual.put("lag_semantics", "both views' C1 lags are logged and never applied to a position (DEC-INT-002)");
        return dual;
    }

    /**
     * Natural versus synthetic hard loss, and the recenters that are neither. A forced loss takes
     * the identical downstream path, so the record is the only thing that keeps it out of a later
     * count of observed estimator failure.
     */
    private static Map<String, Object> hardLossBlock(VoRunnerConfig config,
                                                     java.util.List<Integer> natural,
                                                     java.util.List<Integer> synthetic,
                                                     java.util.List<Integer> recenters) {
        Map<String, Object> h = new LinkedHashMap<>();
        h.put("natural_count", natural.size());
        h.put("natural_frames", natural);
        h.put("synthetic_count", synthetic.size());
        h.put("synthetic_frames", synthetic);
        h.put("recenter_count", recenters.size());
        h.put("recenter_frames", recenters);
        h.put("recenter_semantics", "a mosaic canvas re-origin is tracker bookkeeping: the metric "
                + "segment index is unchanged, no translation gap opens and the persistent position "
                + "stays valid. It is NOT a hard loss and is never counted as one.");
        if (config.syntheticHardLoss == null) {
            h.put("injection", "none (no synthetic_hard_loss block; every loss listed is observed)");
        } else {
            Map<String, Object> inj = new LinkedHashMap<>();
            inj.put("requested_frames", config.syntheticHardLoss.frames);
            inj.put("acknowledge_synthetic", config.syntheticHardLoss.acknowledgeSynthetic);
            inj.put("note", config.syntheticHardLoss.note);
            inj.put("mechanism", "MotionModelStitchingEstimator.injectHardLossOnNextFrame — the "
                    + "estimator's own hard-loss branch, taken on a declared frame; the stitcher is "
                    + "not asked to track that frame");
            inj.put("label", "alignment_events.csv:loss_source = synthetic on every resulting loss");
            h.put("injection", inj);
        }
        return h;
    }

    /**
     * Builds the motion estimator for {@code config}'s selected model, exactly as {@link #run}
     * does, exposed so any other driver over the same {@link VoRunnerConfig} -- e.g. the GUI
     * dataset-replay viewer -- shares this one construction path rather than a second,
     * potentially divergent copy of it.
     */
    public static MotionModelStitchingEstimator<GrayF32, ?> buildEstimator(VoRunnerConfig config) {
        return buildEstimator(config, null);
    }

    /**
     * @param motionOverride when non-null, wraps/replaces the motion estimator (EXP-VO-002's
     *                       timing probe). Must carry the same model {@code config} selects.
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    static MotionModelStitchingEstimator<GrayF32, ?> buildEstimator(
            VoRunnerConfig config, @Nullable ImageMotion2D<GrayF32, ?> motionOverride) {

        MotionModel model = MotionModel.parse(config.motionModel);
        StitchingFromMotion2D stitch;
        MotionModelSupport support;

        if (model == MotionModel.SIMILARITY) {
            ImageMotion2D<GrayF32, Sim2_F64> motion = motionOverride != null
                    ? (ImageMotion2D<GrayF32, Sim2_F64>) motionOverride
                    : similarityMotion(config);
            // Not FactoryMotion2D.createVideoStitch: it selects its StitchingTransform by testing
            // the transform type and would fall through to the homography variant (DEC-VO-006).
            stitch = StitchingFactory.similarityVideoStitch(config.maxJumpFraction, motion);
            support = MotionModelSupport.SIMILARITY;
        } else if (model == MotionModel.AFFINE) {
            ImageMotion2D<GrayF32, Affine2D_F64> motion = motionOverride != null
                    ? (ImageMotion2D<GrayF32, Affine2D_F64>) motionOverride
                    : affineMotion(config);
            stitch = FactoryMotion2D.createVideoStitch(
                    config.maxJumpFraction, motion, ImageType.single(GrayF32.class));
            support = MotionModelSupport.AFFINE;
        } else {
            ImageMotion2D<GrayF32, Homography2D_F64> motion = motionOverride != null
                    ? (ImageMotion2D<GrayF32, Homography2D_F64>) motionOverride
                    : homographyMotion(config);
            stitch = FactoryMotion2D.createVideoStitch(
                    config.maxJumpFraction, motion, ImageType.single(GrayF32.class));
            support = MotionModelSupport.HOMOGRAPHY;
        }

        MotionModelStitchingEstimator<GrayF32, ?> estimator =
                new MotionModelStitchingEstimator<>(stitch, support);
        estimator.setShrinkScale(config.shrinkScale);
        estimator.setMinDistanceFromBorder(config.minDistanceFromBorder);

        // DEC-VO-007 D3. Enabled BEFORE the navigation source is selected, because METRIC_LOCAL is
        // refused when no metric state exists -- which is the point: a config cannot ask for metres
        // without also declaring h0 and the focal length that make them metres.
        if (config.metricReadout != null && config.metricReadout.enabled) {
            estimator.enableMetricReadout(metricReadoutConfig(config));
        }
        // DEC-VO-010. After the metric readout, because heading rotates a metric increment; before
        // the first frame, because switching the navigation direction mid-run would splice two
        // differently-referenced trajectories.
        if (config.headingReadout != null && config.headingReadout.enabled) {
            if (!estimator.isMetricReadoutEnabled()) {
                throw new IllegalArgumentException(
                        "heading_readout.enabled requires metric_readout.enabled -- an "
                        + "authoritative heading rotates a METRIC translation increment, and "
                        + "without a height channel there is none to rotate");
            }
            estimator.enableHeadingReadout(config.headingReadout.toConfig());
        }
        estimator.setNavigationSource(resolveNavigationSource(config));
        return estimator;
    }

    /**
     * The declared metric-readout parameters, validated. {@code f_working} is derived here from
     * {@code fx_native_px} and the run's own {@code downsampleFactor} rather than being configured
     * directly, so the two cannot drift apart: the frames the estimator sees are decimated by
     * exactly that factor in {@link DirectoryFrameSource}, and the focal length must follow them
     * ({@code LIT-VO-003} eq. 9).
     */
    static org.boofcv.stitching.metric.MetricReadoutConfig metricReadoutConfig(VoRunnerConfig config) {
        MetricReadoutSettings m = config.metricReadout;
        if (m.fxNativePx == null) {
            throw new IllegalArgumentException(
                    "metric_readout.enabled requires fx_native_px (native-resolution horizontal "
                    + "focal length in pixels); f_working = fx_native_px / downsampleFactor");
        }
        if (m.h0AglM == null) {
            throw new IllegalArgumentException(
                    "metric_readout.enabled requires h0_agl_m -- the camera-to-imaged-surface height "
                    + "at the reference frame, supplied from outside the system (DEC-VO-007 D2). It "
                    + "must not be inferred from trajectory ground truth.");
        }
        return new org.boofcv.stitching.metric.MetricReadoutConfig(
                m.fxNativePx, config.downsampleFactor, m.h0AglM, m.tauStaleS,
                m.nominalSampleIntervalS);
    }

    /**
     * Refuses a {@code confidence_calibration} config that also asks for {@code metric_readout} or
     * {@code heading_readout}. A no-op when {@code confidence_calibration} is not set, so it is
     * safe to call standalone against any config, not only from inside {@link #run}'s
     * {@code confidenceCalibration != null} branch.
     *
     * <p>The instrumented estimator {@code confidence_calibration} builds
     * ({@code org.boofcv.stitching.StitchingEstimator} wired to {@code InstrumentedStitching}, in
     * {@link #run}) is constructed on a separate path from {@link #buildEstimator} and never calls
     * {@code enableMetricReadout}/{@code enableHeadingReadout}. Left unguarded, the two blocks would
     * not fail: {@code isMetricReadoutEnabled()}/{@code isHeadingReadoutEnabled()} would report
     * false, {@link VoRunner} would silently skip writing {@code metric_track.csv} and feeding the
     * height/heading streams, the pose would stay pixel-valued {@code RIGID_MOTION}, and (unless
     * {@code navigation_source} is explicitly {@code "metric_local"}, which does crash downstream in
     * {@code MotionModelStitchingEstimator.setNavigationSource}) {@link #appliedEstimatorConfig}
     * would still write a full {@code metric_readout}/{@code heading_readout} block into the
     * manifest -- including {@code navigation_units: "m"} when
     * {@code publish_as_navigation_source} is set -- describing a channel that never ran.
     *
     * <p>{@code publish_as_navigation_source} needs no separate check: it is a field of
     * {@code metric_readout} and only takes effect when {@code metric_readout.enabled} is true, so
     * the {@code metric_readout} guard below already covers it.
     *
     * <p>Combining the two is a real feature -- bit-equivalence, metric units and authoritative
     * heading all interacting with the instrumented estimator -- that needs its own design and
     * validation (INT's concern); this is a fail-fast guard, not that design.
     */
    static void checkConfidenceCalibrationVsRuntimeFeatures(VoRunnerConfig config) {
        if (config.confidenceCalibration == null) {
            return;
        }
        if (config.metricReadout != null && config.metricReadout.enabled) {
            throw new IllegalArgumentException("confidence_calibration and metric_readout are "
                    + "not yet jointly supported: the instrumented estimator confidence_calibration "
                    + "builds (DEC-CONF-002) does not enable the metric readout channel, so "
                    + "metric_readout would be silently inert instead of producing a metric pose. "
                    + "Run them separately until the combined estimator path is designed and "
                    + "validated.");
        }
        if (config.headingReadout != null && config.headingReadout.enabled) {
            throw new IllegalArgumentException("confidence_calibration and heading_readout are "
                    + "not yet jointly supported: the instrumented estimator confidence_calibration "
                    + "builds (DEC-CONF-002) does not enable the external heading channel, so "
                    + "heading_readout would be silently inert instead of producing an "
                    + "authoritative navigation heading. Run them separately until the combined "
                    + "estimator path is designed and validated.");
        }
    }

    /**
     * The relocalization layer's pose-contract guard (post-merge compatibility pass, 2026-09-07;
     * {@code DEC-INT-001} amendment). Same philosophy as
     * {@link #checkConfidenceCalibrationVsRuntimeFeatures}: a loud refusal beats plausible-looking
     * unit/frame corruption.
     *
     * <p>History: this started the pass as a temporary fail-fast refusing an active INT layer with
     * {@code METRIC_LOCAL} as the navigation source, because the layer was written against the
     * pixel / visual-yaw contract. The layer now consumes <em>only</em> the metric,
     * externally-heading-referenced state ({@code LocalPoseSample.fromMetric}), proven by the
     * {@code org.boofcv.relocalization} suite and {@code VoRunnerAlignmentSidecarTest}, so the
     * guard is inverted rather than removed:
     *
     * <ul>
     *   <li>INT active + {@code metric_readout.enabled} + {@code heading_readout.enabled} —
     *       <b>supported</b>, whether or not {@code publish_as_navigation_source} /
     *       {@code navigation_source: metric_local} is also set (the layer reads the metric state
     *       directly; {@code frames.csv} is unaffected either way).</li>
     *   <li>INT active without the metric readout — <b>refused</b>: there is no pixel contract.</li>
     *   <li>INT active with the metric readout but no heading channel — <b>refused</b>: the
     *       East/North axes would follow the visual yaw ({@code DEC-VO-009} arm), and a
     *       translation-only alignment is undefined there.</li>
     *   <li>{@code metric_readout} / {@code METRIC_LOCAL} without INT — untouched.</li>
     *   <li>INT + {@code confidence_calibration} — refused separately (no refit probe on the
     *       instrumented path), and that path is itself refused with metric/heading by the CONF×VO
     *       guard, which this guard neither bypasses nor weakens.</li>
     * </ul>
     */
    static void checkRelocalizationVsNavigationContract(VoRunnerConfig config) {
        boolean intActive = config.alignmentSidecar || config.relocalizationConfig != null;
        if (!intActive) {
            return;
        }
        boolean metric = config.metricReadout != null && config.metricReadout.enabled;
        boolean heading = config.headingReadout != null && config.headingReadout.enabled;
        if (!metric) {
            throw new IllegalArgumentException("alignment_sidecar / relocalization_config require "
                    + "metric_readout.enabled: the relocalization layer consumes the metric "
                    + "segment-relative pose (metres, East/North; DEC-VO-009) and has no contract for "
                    + "the pixel RIGID_MOTION readout (DEC-INT-001 amendment 2026-09-07). Refused "
                    + "rather than run with mixed units.");
        }
        if (!heading) {
            throw new IllegalArgumentException("alignment_sidecar / relocalization_config require "
                    + "heading_readout.enabled: the relocalization layer's alignment is a "
                    + "translation in a North-referenced frame, which needs the authoritative "
                    + "external heading (DEC-VO-010); a metric state rotated by the visual yaw is "
                    + "not North-referenced. Refused rather than run with an undefined frame.");
        }
    }

    /**
     * Which source publishes the pose, taking {@code metric_readout.publish_as_navigation_source}
     * into account.
     *
     * <p>The two settings are checked against each other rather than silently reconciled: asking for
     * {@code navigation_source: "metric_local"} without enabling the readout is an error, and
     * enabling {@code publish_as_navigation_source} alongside an explicit non-metric
     * {@code navigation_source} is an error too. Either would otherwise produce a run whose units
     * are not what its config appears to say.
     */
    static NavigationSource resolveNavigationSource(VoRunnerConfig config) {
        NavigationSource named = parseNavigationSource(config.navigationSource);
        boolean publishMetric = config.metricReadout != null && config.metricReadout.enabled
                && config.metricReadout.publishAsNavigationSource;
        if (!publishMetric) {
            return named;
        }
        if (named != NavigationSource.METRIC_LOCAL
                && config.navigationSource != null && !config.navigationSource.isBlank()
                && !"rigid_motion".equalsIgnoreCase(config.navigationSource)) {
            throw new IllegalArgumentException(
                    "metric_readout.publish_as_navigation_source is true but navigation_source is \""
                    + config.navigationSource + "\". The metric readout scales RIGID_MOTION's "
                    + "increments; it is not defined over the legacy or logical readouts.");
        }
        return NavigationSource.METRIC_LOCAL;
    }

    /**
     * The motion estimator with {@code EXP-VO-010}'s refinement probe attached.
     *
     * <p>Rejects the similarity model rather than silently producing an unprobed run: it has no
     * {@code ModelFitter}, so "before and after refinement" is not a thing that exists for it
     * ({@code EXP-VO-010} Phase 1 Q7).
     */
    static StitchingFactory.ProbedMotion<?> buildProbedMotion(VoRunnerConfig config) {
        MotionModel model = MotionModel.parse(config.motionModel);
        if (model == MotionModel.SIMILARITY) {
            throw new IllegalArgumentException(
                    "refinement_sidecar is not available for motion_model \"similarity\": the "
                    + "similarity generator implements ModelGenerator only, so RANSAC has no final "
                    + "model to refine (EXP-VO-010 Phase 1). Use \"homography\" or \"affine\".");
        }
        return StitchingFactory.createProbedMotion2D(
                model == MotionModel.AFFINE ? StitchingFactory.ProbeModel.AFFINE
                                            : StitchingFactory.ProbeModel.HOMOGRAPHY,
                config.ransacIterations, config.inlierThresholdSq, config.outlierPrune,
                config.absoluteMinimumTracks, config.respawnTrackFraction,
                config.respawnCoverageFraction, config.refineEstimate,
                config.diagnosticRefitSidecar, tracker(config));
    }

    /** The raw motion estimator for {@code config}'s model, before mosaic wrapping. */
    static ImageMotion2D<GrayF32, ?> buildMotion(VoRunnerConfig config) {
        switch (MotionModel.parse(config.motionModel)) {
            case AFFINE:
                return affineMotion(config);
            case SIMILARITY:
                return similarityMotion(config);
            default:
                return homographyMotion(config);
        }
    }

    private static ImageMotion2D<GrayF32, Homography2D_F64> homographyMotion(VoRunnerConfig config) {
        return FactoryMotion2D.createMotion2D(
                config.ransacIterations, config.inlierThresholdSq, config.outlierPrune,
                config.absoluteMinimumTracks, config.respawnTrackFraction,
                config.respawnCoverageFraction, config.refineEstimate,
                tracker(config), new Homography2D_F64());
    }

    private static ImageMotion2D<GrayF32, Affine2D_F64> affineMotion(VoRunnerConfig config) {
        return FactoryMotion2D.createMotion2D(
                config.ransacIterations, config.inlierThresholdSq, config.outlierPrune,
                config.absoluteMinimumTracks, config.respawnTrackFraction,
                config.respawnCoverageFraction, config.refineEstimate,
                tracker(config), new Affine2D_F64());
    }

    /**
     * The 4-DoF similarity arm (DEC-VO-006, EXP-VO-009). Built through StitchingFactory rather
     * than FactoryMotion2D, which implements only the homography, affine and SE(2) branches and
     * throws for anything else. Every parameter is the same value the other two arms receive; the
     * builder's own defaults are overridden from `config` exactly as they are above.
     */
    private static ImageMotion2D<GrayF32, Sim2_F64> similarityMotion(VoRunnerConfig config) {
        return StitchingFactory.builder()
                .detector(detectorConfig(config))
                .kltLevels(config.kltPyramidLevels)
                .kltRadius(config.kltFeatureRadius)
                .motionMaxIterations(config.ransacIterations)
                .inlierThresholdSq(config.inlierThresholdSq)
                .outlierPrune(config.outlierPrune)
                .motionMinFeatures(config.absoluteMinimumTracks)
                .respawnTrackFraction(config.respawnTrackFraction)
                .respawnCoverageFraction(config.respawnCoverageFraction)
                .buildGraySimilarityMotionOnly();
    }

    /** Identical detector configuration for every model -- the ablation holds this fixed. */
    private static ConfigPointDetector detectorConfig(VoRunnerConfig config) {
        ConfigPointDetector configDetector = new ConfigPointDetector();
        configDetector.type = PointDetectorTypes.valueOf(config.detectorType);
        configDetector.general.maxFeatures = config.maxFeatures;
        configDetector.general.radius = config.detectorRadius;
        configDetector.general.threshold = (float) config.detectorThreshold;
        return configDetector;
    }

    /** Identical tracker construction for every model -- the ablation holds this fixed. */
    private static PointTracker<GrayF32> tracker(VoRunnerConfig config) {
        return FactoryPointTracker.klt(config.kltPyramidLevels, detectorConfig(config),
                config.kltFeatureRadius, GrayF32.class, GrayF32.class);
    }

    /**
     * {@code navigation_source} config values, mapped onto {@link NavigationSource}.
     *
     * <p>An absent or blank value resolves to {@link NavigationSource#DEFAULT}, the same constant
     * {@code VoRunnerConfig}'s field initialiser uses. Before {@code DEC-VO-008} this branch
     * hardcoded {@code MOSAIC_LEGACY}, which was a second copy of the default and could have
     * survived the promotion silently — a config with an explicit {@code "navigation_source": null}
     * would have kept running legacy after the field default moved.
     */
    static NavigationSource parseNavigationSource(String s) {
        if (s == null || s.isBlank()) {
            return NavigationSource.DEFAULT;
        }
        if ("mosaic_legacy".equalsIgnoreCase(s)) {
            return NavigationSource.MOSAIC_LEGACY;
        }
        if ("logical_frame".equalsIgnoreCase(s)) {
            return NavigationSource.LOGICAL_FRAME;
        }
        if ("rigid_motion".equalsIgnoreCase(s)) {
            return NavigationSource.RIGID_MOTION;
        }
        if ("metric_local".equalsIgnoreCase(s)) {
            return NavigationSource.METRIC_LOCAL;
        }
        throw new IllegalArgumentException("Unknown navigation_source '" + s
                + "'; expected \"mosaic_legacy\", \"logical_frame\", \"rigid_motion\" or "
                + "\"metric_local\"");
    }

    /**
     * The three selectable motion models: the shipped 8-DoF projective homography, the 6-DoF
     * affine EXP-VO-002 compares against it (DEC-VO-002), and the 4-DoF similarity EXP-VO-009
     * adds (DEC-VO-006). They are nested -- similarity subset of affine subset of projective.
     */
    public enum MotionModel {
        HOMOGRAPHY, AFFINE, SIMILARITY;

        static MotionModel parse(String s) {
            if (s == null || s.isBlank() || "homography".equalsIgnoreCase(s)) return HOMOGRAPHY;
            if ("affine".equalsIgnoreCase(s)) return AFFINE;
            if ("similarity".equalsIgnoreCase(s)) return SIMILARITY;
            throw new IllegalArgumentException("Unknown motion_model '" + s
                    + "'; expected \"homography\", \"affine\" or \"similarity\"");
        }
    }

    /**
     * Writes per-frame motion-estimation time (KLT + RANSAC, excluding mosaic rendering).
     * EXP-VO-002; see {@link MotionTimingProbe} for exactly what the timer encloses.
     */
    private static void writeMotionTiming(Path out, MotionTimingProbe<GrayF32, ?> probe,
                                          VoRunnerConfig config) throws IOException {
        if (out.getParent() != null) {
            java.nio.file.Files.createDirectories(out.getParent());
        }
        List<Long> nanos = probe.frameNanos();
        StringBuilder sb = new StringBuilder("frame_index,motion_time_ns,motion_model\n");
        String model = MotionModel.parse(config.motionModel).name().toLowerCase();
        for (int i = 0; i < nanos.size(); i++) {
            sb.append(i).append(',').append(nanos.get(i)).append(',').append(model).append('\n');
        }
        java.nio.file.Files.writeString(out, sb.toString());
    }

    /**
     * Values as actually applied to the estimator, in contracts/run-record.md's example key
     * order — not the requested config verbatim, though today they are the same values, since
     * BoofCV applies every one of these parameters without clamping (COMP-001 §3.3).
     */
    private static Map<String, Object> appliedEstimatorConfig(VoRunnerConfig config) {
        Map<String, Object> applied = new LinkedHashMap<>();
        applied.put("shrinkScale", config.shrinkScale);
        applied.put("minDistanceFromBorder", config.minDistanceFromBorder);
        applied.put("detectorType", config.detectorType);
        applied.put("maxFeatures", config.maxFeatures);
        applied.put("detectorRadius", config.detectorRadius);
        applied.put("detectorThreshold", config.detectorThreshold);
        applied.put("kltPyramidLevels", config.kltPyramidLevels);
        applied.put("kltFeatureRadius", config.kltFeatureRadius);
        applied.put("ransacIterations", config.ransacIterations);
        applied.put("inlierThresholdSq", config.inlierThresholdSq);
        applied.put("outlierPrune", config.outlierPrune);
        applied.put("absoluteMinimumTracks", config.absoluteMinimumTracks);
        applied.put("respawnTrackFraction", config.respawnTrackFraction);
        applied.put("respawnCoverageFraction", config.respawnCoverageFraction);
        applied.put("refineEstimate", config.refineEstimate);
        applied.put("diagnosticRefitSidecar", config.diagnosticRefitSidecar);
        applied.put("maxJumpFraction", config.maxJumpFraction);
        applied.put("downsampleFactor", config.downsampleFactor);
        applied.put("motion_model", MotionModel.parse(config.motionModel).name().toLowerCase());
        applied.put("navigation_source", resolveNavigationSource(config).name().toLowerCase());
        // DEC-VO-007's declared parameters travel with the run, so a metric trajectory can never be
        // read without the h0 and focal length that made it metric (Principle VI).
        if (config.metricReadout != null && config.metricReadout.enabled) {
            Map<String, Object> metric = new LinkedHashMap<>();
            metric.put("enabled", true);
            metric.put("fx_native_px", config.metricReadout.fxNativePx);
            metric.put("f_working_px", metricReadoutConfig(config).fWorkingPx());
            metric.put("h0_agl_m", config.metricReadout.h0AglM);
            metric.put("tau_stale_s", config.metricReadout.tauStaleS);
            metric.put("nominal_sample_interval_s", config.metricReadout.nominalSampleIntervalS);
            metric.put("height_csv", config.metricReadout.heightCsv);
            metric.put("height_time_column", config.metricReadout.heightTimeColumn);
            metric.put("height_relative_column", config.metricReadout.heightRelativeColumn);
            metric.put("height_latency_s", config.metricReadout.heightLatencyS);
            metric.put("height_decimation", config.metricReadout.heightDecimation);
            metric.put("height_semantics", config.metricReadout.heightSemantics);
            // The ARM travels with the run, so an oracle trajectory can never be read as a
            // production-like one (DEC-VO-009 D4).
            metric.put("height_arm", config.metricReadout.isOracleArm() ? "oracle" : "baro");
            if (config.metricReadout.isOracleArm()) {
                metric.put("height_arm_caveat", "DIAGNOSTIC ONLY: the renderer's true height above "
                        + "the imaged surface. No sensor on the author's platform supplies it at "
                        + "survey height (LIT-VO-006 s2). Never report this arm unlabelled.");
            }
            metric.put("publish_as_navigation_source",
                    config.metricReadout.publishAsNavigationSource);
            metric.put("navigation_units", resolveNavigationSource(config).units().symbol());
            applied.put("metric_readout", metric);
        }
        // DEC-VO-010's declared parameters travel with the run for the same reason: a metric
        // trajectory rotated by an external heading can never be read without the semantics and the
        // mounting calibration that made it point where it points.
        if (config.headingReadout != null && config.headingReadout.enabled) {
            HeadingReadoutSettings h = config.headingReadout;
            Map<String, Object> heading = new LinkedHashMap<>();
            heading.put("enabled", true);
            heading.put("heading_semantics", h.semantics().configName());
            heading.put("heading_authority", "external");
            heading.put("delta_mount_deg", h.toConfig().mountOffsetDeg());
            heading.put("delta_mount_provenance", h.deltaMountProvenance);
            heading.put("tau_stale_s", h.tauStaleS);
            heading.put("nominal_sample_interval_s", h.nominalSampleIntervalS);
            heading.put("max_rate_deg_per_s", h.maxRateDegPerS);
            heading.put("heading_csv", h.headingCsv);
            heading.put("heading_time_column", h.headingTimeColumn);
            heading.put("heading_column", h.headingColumn);
            heading.put("heading_latency_s", h.headingLatencyS);
            heading.put("heading_decimation", h.headingDecimation);
            heading.put("visual_yaw_role", "diagnostic");
            if (h.semantics() == org.boofcv.stitching.metric.HeadingSemantics.FC_AHRS_BODY_COMPASS) {
                heading.put("heading_caveat", "The source column is the AIRFRAME's heading, "
                        + "converted to the navigation camera through a DECLARED platform "
                        + "calibration that is specific to this airframe/camera pair. "
                        + "GPS-independence of this fused-attitude stream is NOT established by "
                        + "the available data (rtk_info_yaw = 50 on every committed sample).");
            }
            applied.put("heading_readout", heading);
        }
        return applied;
    }
}
