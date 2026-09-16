package org.boofcv.stitching.diagnostics;

import boofcv.abst.feature.detect.interest.ConfigPointDetector;
import boofcv.abst.feature.detect.interest.PointDetectorTypes;
import boofcv.struct.image.GrayF32;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.boofcv.evaluation.DatasetDescriptor;
import org.boofcv.evaluation.DirectoryFrameSource;
import org.boofcv.evaluation.Environment;
import org.boofcv.evaluation.VoRunnerConfig;
import org.boofcv.stitching.InstrumentedStitching;
import org.boofcv.stitching.StitchingEstimator;
import org.boofcv.stitching.StitchingFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Config-file-driven entry point for the {@code EXP-VO-001} residual measurement.
 *
 * <p>Usage: {@code --config <probe-config.json> [--commit <sha>] [--dataset-dir <path>]}
 *
 * <p>Writes into {@code <output_dir>/<case_id>/}:
 *
 * <ul>
 *   <li>{@code residuals.csv} — one row per frame ({@link ResidualProbeRunner#HEADER});</li>
 *   <li>{@code manifest.json} — provenance: dataset, applied estimator configuration, environment,
 *       and the commit if one was supplied;</li>
 *   <li>{@code reproduction.json} — whether the instrumented path reproduced the committed
 *       reference run record exactly, or, when no reference was configured, an explicit record
 *       that the check was <em>not performed</em>.</li>
 * </ul>
 *
 * <p>The commit is recorded only when passed explicitly, and is otherwise {@code null}. A guess
 * would be worse than an absence: this app cannot tell whether the working tree it was compiled
 * from was clean.
 */
public final class ResidualProbeApp {

    public static void main(String[] args) throws IOException {
        String configPath = parseArg(args, "--config");
        if (configPath == null) {
            System.err.println("Usage: ResidualProbeApp --config <probe-config.json> "
                    + "[--commit <sha>] [--dataset-dir <path>]");
            System.exit(1);
            return;
        }
        run(configPath, parseArg(args, "--commit"), parseArg(args, "--dataset-dir"), System.out);
    }

    static String parseArg(String[] args, String name) {
        for (int i = 0; i < args.length - 1; i++) {
            if (name.equals(args[i])) {
                return args[i + 1];
            }
        }
        return null;
    }

    /**
     * Runs one probe case and returns the directory written.
     *
     * @param datasetDirOverride dataset root to use in place of the one the configs declare, or
     *                           {@code null}. Exists because dataset imagery is not stored with
     *                           the configs. Supplying it at the
     *                           command line rather than in the tracked config keeps a machine-
     *                           specific path out of version control; the path actually used is
     *                           recorded in {@code manifest.json} either way.
     */
    public static Path run(String probeConfigPath, String commit, String datasetDirOverride,
                           PrintStream log) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        ResidualProbeConfig probe = mapper.readValue(new File(probeConfigPath), ResidualProbeConfig.class);
        VoRunnerConfig vo = mapper.readValue(new File(probe.voRunnerConfig), VoRunnerConfig.class);

        String datasetDirText = datasetDirOverride != null ? datasetDirOverride
                : probe.datasetDirOverride != null ? probe.datasetDirOverride
                : vo.datasetDir;
        Path datasetDir = Paths.get(datasetDirText);
        DatasetDescriptor dataset = DatasetDescriptor.load(datasetDir);

        ConfigPointDetector configDetector = new ConfigPointDetector();
        configDetector.type = PointDetectorTypes.valueOf(vo.detectorType);
        configDetector.general.maxFeatures = vo.maxFeatures;
        configDetector.general.radius = vo.detectorRadius;
        configDetector.general.threshold = (float) vo.detectorThreshold;

        // buildGrayInstrumented, not buildGray: the diagnostics must come from the same estimator
        // instance that ships the pose (DEC-VO-001). Bit-identity with buildGray under identical
        // configuration is enforced by StitchingFactoryInstrumentationEquivalenceTest, and is
        // re-checked here at dataset scale by the reproduction check below.
        InstrumentedStitching<GrayF32> instrumented = StitchingFactory.builder()
                .detector(configDetector)
                .kltLevels(vo.kltPyramidLevels)
                .kltRadius(vo.kltFeatureRadius)
                .motionMaxIterations(vo.ransacIterations)
                .inlierThresholdSq(vo.inlierThresholdSq)
                .outlierPrune(vo.outlierPrune)
                .motionMinFeatures(vo.absoluteMinimumTracks)
                .respawnTrackFraction(vo.respawnTrackFraction)
                .respawnCoverageFraction(vo.respawnCoverageFraction)
                .refineEstimate(vo.refineEstimate)
                .stitchOverlapThreshold(vo.maxJumpFraction)
                .buildGrayInstrumented();

        StitchingEstimator<GrayF32> estimator = new StitchingEstimator<>(instrumented.stitch());
        estimator.setShrinkScale(vo.shrinkScale);
        estimator.setMinDistanceFromBorder(vo.minDistanceFromBorder);

        DirectoryFrameSource frames = new DirectoryFrameSource(dataset, vo.downsampleFactor);

        Path caseDir = Paths.get(probe.outputDir).resolve(probe.caseId);
        Files.createDirectories(caseDir);

        List<ResidualProbeRunner.Row> rows;
        Path csvPath = caseDir.resolve("residuals.csv");
        try (BufferedWriter csv = Files.newBufferedWriter(csvPath, StandardCharsets.UTF_8)) {
            rows = new ResidualProbeRunner(instrumented, estimator, frames).run(csv);
        }

        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("experiment_id", probe.experimentId);
        manifest.put("case_id", probe.caseId);
        manifest.put("dataset_id", dataset.datasetId);
        manifest.put("dataset_revision", dataset.datasetRevision);
        manifest.put("dataset_dir", datasetDir.toString().replace('\\', '/'));
        manifest.put("dataset_dir_overridden", datasetDirOverride != null);
        manifest.put("vo_runner_config", probe.voRunnerConfig);
        manifest.put("estimator_path", "StitchingFactory.Builder#buildGrayInstrumented (DEC-VO-001)");
        manifest.put("estimator_config", appliedEstimatorConfig(vo));
        manifest.put("residual_units", "squared Euclidean pixel error in the input-image pixel grid, "
                + "against the frame-to-keyframe homography (DistanceHomographySq)");
        manifest.put("frame_count", rows.size());
        manifest.put("software_commit", commit);
        manifest.put("environment", Environment.captureCurrent());
        manifest.put("run_timestamp", Instant.now().toString());
        writeJson(mapper, caseDir.resolve("manifest.json"), manifest);

        Map<String, Object> reproduction = probe.referenceRunDir == null
                ? notPerformed()
                : compareToReference(rows, Paths.get(probe.referenceRunDir));
        writeJson(mapper, caseDir.resolve("reproduction.json"), reproduction);

        log.println("Residual probe written to " + caseDir);
        log.println("Reproduction: " + reproduction.get("verdict"));
        return caseDir;
    }

    private static Map<String, Object> notPerformed() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("verdict", "not_performed");
        result.put("reason", "no reference_run_dir configured; absence of a check is not a passing check");
        return result;
    }

    /**
     * Compares this probe's per-frame poses and events against a committed run record captured by
     * the non-instrumented path.
     *
     * <p>Compared exactly, not within a tolerance. The run record writes doubles with
     * {@code Double.toString}, which round-trips, so an exact comparison is meaningful and a
     * tolerance would only hide a real difference. The claim being checked is bit-identity
     * (feature spec FR-007/FR-008); loosening it would void the claim.
     */
    static Map<String, Object> compareToReference(List<ResidualProbeRunner.Row> rows, Path referenceRunDir)
            throws IOException {
        Path framesCsv = referenceRunDir.resolve("frames.csv");
        List<String> lines = Files.readAllLines(framesCsv, StandardCharsets.UTF_8);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("reference_run_dir", referenceRunDir.toString().replace('\\', '/'));
        result.put("compared_columns", List.of(
                "est_x", "est_y", "est_yaw_deg", "success", "event", "reference_id",
                "track_count", "inlier_count"));
        result.put("comparison", "exact equality, no tolerance");

        String[] header = lines.get(0).split(",", -1);
        Map<String, Integer> col = new LinkedHashMap<>();
        for (int i = 0; i < header.length; i++) {
            col.put(header[i], i);
        }

        int referenceRows = lines.size() - 1;
        result.put("reference_frame_count", referenceRows);
        result.put("probe_frame_count", rows.size());
        if (referenceRows != rows.size()) {
            result.put("verdict", "frame_count_mismatch");
            return result;
        }

        Map<String, Integer> mismatches = new LinkedHashMap<>();
        List<Integer> firstMismatchFrames = new ArrayList<>();
        for (int i = 0; i < rows.size(); i++) {
            String[] f = lines.get(i + 1).split(",", -1);
            ResidualProbeRunner.Row r = rows.get(i);
            boolean any = false;
            any |= note(mismatches, "est_x", !eq(f[col.get("est_x")], r.estX()));
            any |= note(mismatches, "est_y", !eq(f[col.get("est_y")], r.estY()));
            any |= note(mismatches, "est_yaw_deg", !eq(f[col.get("est_yaw_deg")], r.estYawDeg()));
            any |= note(mismatches, "success",
                    !f[col.get("success")].equals(Boolean.toString(r.success())));
            any |= note(mismatches, "event", !f[col.get("event")].equals(r.event()));
            any |= note(mismatches, "reference_id",
                    !f[col.get("reference_id")].equals(Integer.toString(r.referenceId())));
            any |= note(mismatches, "track_count",
                    !f[col.get("track_count")].equals(str(r.trackCount())));
            any |= note(mismatches, "inlier_count",
                    !f[col.get("inlier_count")].equals(str(r.inlierCount())));
            if (any && firstMismatchFrames.size() < 20) {
                firstMismatchFrames.add(r.frameIndex());
            }
        }

        result.put("mismatch_counts", mismatches);
        result.put("first_mismatch_frames", firstMismatchFrames);
        result.put("verdict", mismatches.isEmpty() ? "identical" : "diverged");
        return result;
    }

    private static boolean note(Map<String, Integer> counts, String column, boolean mismatched) {
        if (mismatched) {
            counts.merge(column, 1, Integer::sum);
        }
        return mismatched;
    }

    /** Exact double comparison via the reference text's own parse, so round-tripping is exercised. */
    private static boolean eq(String referenceText, double probeValue) {
        return Double.parseDouble(referenceText) == probeValue;
    }

    private static String str(Integer value) {
        return value == null ? "" : value.toString();
    }

    /** Mirrors {@code VoRunnerApp}'s applied-config record so the two manifests are comparable. */
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
        applied.put("maxJumpFraction", config.maxJumpFraction);
        applied.put("downsampleFactor", config.downsampleFactor);
        return applied;
    }

    private static void writeJson(ObjectMapper mapper, Path path, Map<String, Object> value)
            throws IOException {
        mapper.writerWithDefaultPrettyPrinter().writeValue(path.toFile(), value);
    }
}
