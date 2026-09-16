package org.boofcv.evaluation;

import org.boofcv.confidence.CalibrationConfig;
import org.boofcv.confidence.ConfidenceResult;
import org.boofcv.confidence.ConfidenceScorer;
import org.boofcv.confidence.SignalBlock;
import org.boofcv.confidence.SignalExtractor;
import org.boofcv.stitching.MotionResidualDiagnostics.Summary;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Generates — and thereafter guards — the <b>Java-produced</b> fixture behind
 * {@code evaluation/tests/test_confidence_agreement.py} (contracts/agreement.md §The test).
 *
 * <p>The fixture is one scripted signal stream scored under both shipped calibrations, written
 * through the real {@link SignalExtractor} → {@link ConfidenceScorer} → {@link RunRecordWriter}
 * pipeline into {@code evaluation/tests/fixtures/confidence_agreement/}. A Python-authored fixture
 * would test Python against itself; this one carries verdicts and scores the Java implementation
 * actually produced. The stream covers every agreement case: nominal frames, warm-up absence,
 * absent residuals, an empty match set, an estimator failure with reset, the first frame,
 * exactly-on-threshold values, and the two-calibrations case (the two runs disagree, proving the
 * configuration is read).
 *
 * <p><b>Determinism, and the anti-churn rule.</b> Every value is fixed — timestamps, zero
 * process/confidence times, a synthetic environment — so regeneration is byte-identical. Unlike
 * {@code JavaProducedRunFixtureTest}, this test never overwrites an
 * existing fixture: if the committed fixture and the freshly-generated content differ, it
 * <b>fails</b>, and regeneration is a deliberate act (delete the directory, re-run, review the
 * diff, commit). An implementation change that moves a single persisted byte is thereby loud.
 *
 * <p>Evidence tier: <b>T1 (analytical)</b>.
 */
public class ConfidenceAgreementFixtureTest {

    private static final Path FIXTURE_DIR = Paths.get("evaluation/tests/fixtures/confidence_agreement");
    private static final Path BOOTSTRAP = Paths.get("evaluation/confidence_configs/bootstrap-unvalidated-v2.json");
    private static final Path CONTRAST = Paths.get("evaluation/confidence_configs/contrast-unvalidated-v2.json");

    /** One scripted frame of the fixture stream. */
    private record Scripted(boolean success, String event, Integer track, Integer inlier,
                            Summary summary, Double incFlowPx, Double incLogScale) {
    }

    /**
     * The scripted stream. Comments give the case each frame exists for; the values are arbitrary
     * but fixed forever — the fixture's identity is part of the agreement contract.
     */
    private static Scripted[] script() {
        Summary healthy = new Summary(180, 1.0, 1.0, 0.9, 2.5);
        return new Scripted[]{
                // f0 — first frame: not_established, everything windowed/residual absent.
                new Scripted(true, "init", 2507, 2507, Summary.UNAVAILABLE, null, null),
                // f1..f5 — nominal, but below warm-up: the contrast calibration (which needs
                // relative_support) must say signals_unavailable while bootstrap scores. This IS
                // the two-calibrations disagreement case, produced structurally.
                new Scripted(true, "none", 200, 180, healthy, 4.5, 8.0E-4),
                new Scripted(true, "none", 200, 180, healthy, 4.5, 8.1E-4),
                new Scripted(true, "none", 200, 180, healthy, 4.5, 8.2E-4),
                new Scripted(true, "none", 200, 180, healthy, 4.5, 8.3E-4),
                new Scripted(true, "none", 200, 180, healthy, 4.5, 8.4E-4),
                // f6 — nominal, warmed up: both calibrations score.
                new Scripted(true, "none", 200, 180, healthy, 4.5, 8.5E-4),
                // f7 — absent residuals: bootstrap says signals_unavailable, contrast scores.
                new Scripted(true, "none", 200, 180, Summary.UNAVAILABLE, 4.5, 8.6E-4),
                // f8 — empty match set: a REAL count of zero beside genuinely absent statistics.
                new Scripted(true, "none", 200, 180, new Summary(0, null, null, null, null), 4.5, 8.7E-4),
                // f9 — estimator failure: not_produced under both; the restart resets the windows,
                // and its counts describe the re-initialised estimator (EXP-VO-001 R5a).
                new Scripted(false, "restart", 2507, 2507, Summary.UNAVAILABLE, null, null),
                // f10..f14 — post-restart warm-up rebuild.
                new Scripted(true, "none", 200, 180, healthy, 4.5, 8.8E-4),
                new Scripted(true, "none", 200, 180, healthy, 4.5, 8.9E-4),
                new Scripted(true, "none", 200, 180, healthy, 4.5, 9.0E-4),
                new Scripted(true, "none", 200, 180, healthy, 4.5, 9.1E-4),
                new Scripted(true, "none", 200, 180, healthy, 4.5, 9.2E-4),
                // f15 — exactly-on-threshold: track_count == 30 (lt strict), inlier_ratio == 0.3
                // exactly (9/30), residual mean == 3.0 exactly (gt strict). No rule may fire in
                // either language; the low score lands the frame in DEGRADED instead.
                new Scripted(true, "none", 30, 9, new Summary(9, 3.0, 1.7320508075688772, 3.0, 3.0), 4.5, 9.3E-4),
        };
    }

    private static RunManifest manifest(String runId, CalibrationConfig calibration,
                                        Map<String, Object> estimatorConfig,
                                        Map<String, Object> captureContext) {
        RunManifest m = new RunManifest();
        m.schemaVersion = RunManifest.SCHEMA_VERSION_CONFIDENCE;
        m.runId = runId;
        m.datasetId = "confidence-agreement-fixture";
        m.datasetRevision = "v1";
        m.estimatorId = "scripted-signals";
        m.estimatorVersion = "fixture";
        m.evaluatorCaptureVersion = "fixture";
        m.estimatorConfig = estimatorConfig;
        m.environment = new Environment("fixture", "none", "none", "none", 0, false);
        m.runTimestamp = "1970-01-01T00:00:00Z";
        Map<String, Object> conf = new LinkedHashMap<>();
        conf.put("calibration_id", calibration.calibrationId());
        conf.put("calibration_digest", calibration.digest());
        conf.put("calibration_validated", calibration.validated());
        conf.put("probability_semantics", calibration.probabilitySemantics());
        conf.put("residual_diagnostics_available", true);
        conf.put("residual_source", "scripted (fixture; real interface types, scripted values)");
        conf.put("signal_window_w", calibration.windowW());
        conf.put("signal_warmup_m", calibration.warmupM());
        conf.put("capture_context", captureContext);
        m.confidence = conf;
        return m;
    }

    /** The estimator/capture context both shipped calibrations bind to (the MARS-LVIG dev context). */
    private static Map<String, Object> estimatorConfig() {
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("inlierThresholdSq", 3.0);
        c.put("absoluteMinimumTracks", 30);
        c.put("respawnTrackFraction", 0.6);
        c.put("respawnCoverageFraction", 0.5);
        c.put("refineEstimate", false);
        c.put("kltPyramidLevels", 4);
        c.put("kltFeatureRadius", 3);
        c.put("downsampleFactor", 2);
        c.put("motion_model", "homography");
        return c;
    }

    private static Map<String, Object> captureContext() {
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("image_width", 1224);
        c.put("image_height", 1024);
        c.put("motion_model", "homography");
        return c;
    }

    private static void generateRun(Path dir, CalibrationConfig calibration) throws IOException {
        Map<String, Object> estimatorConfig = estimatorConfig();
        Map<String, Object> captureContext = captureContext();
        Map<String, Object> bindingContext = new LinkedHashMap<>(estimatorConfig);
        bindingContext.putAll(captureContext);
        calibration.checkBinding(bindingContext, "the agreement fixture");

        SignalExtractor extractor = new SignalExtractor(calibration.windowW(), calibration.warmupM());
        ConfidenceScorer scorer = new ConfidenceScorer(calibration);
        Scripted[] frames = script();
        try (RunRecordWriter writer = new RunRecordWriter(dir,
                manifest(dir.getFileName().toString(), calibration, estimatorConfig, captureContext),
                frames.length, true)) {
            int referenceId = 0;
            for (int i = 0; i < frames.length; i++) {
                Scripted f = frames[i];
                if ("restart".equals(f.event()) || "recenter".equals(f.event())) {
                    referenceId++;
                }
                VoDiagnostics.Diagnostics diag =
                        new VoDiagnostics.Diagnostics(f.track(), f.inlier(), null);
                SignalBlock block = extractor.extract(f.success(), f.event(), diag, f.summary(),
                        3.0, f.incFlowPx(), f.incLogScale());
                ConfidenceResult result = scorer.score(block, f.success(), i == 0);

                FrameEstimateRow.Confidence conf = new FrameEstimateRow.Confidence(
                        block.residualInlierCount(), block.residualMeanSqPx(), block.residualRmsPx(),
                        block.residualMedianSqPx(), block.residualMaxSqPx(), block.inlierThresholdSqPx(),
                        block.inlierCoverage(), block.keyframeAge(), block.relativeSupport(),
                        block.incFlowPx(), f.incLogScale(), block.incLogScaleDispersion(),
                        result.outcome().wireName(), result.reason().wireName(), result.score(), 0L);
                writer.writeFrame(new FrameEstimateRow(i, i * 0.1, i * 1.0, i * -1.0, null, 0.0,
                        f.success(), f.event(), referenceId,
                        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                        f.track(), f.inlier(), 0L, conf));
            }
        }
    }

    private static String normalise(String s) {
        return s.replace("\r\n", "\n").replace("\r", "\n");
    }

    @Test
    public void fixtureExistsAndMatchesWhatTheCurrentImplementationProduces() throws IOException {
        CalibrationConfig bootstrap = CalibrationConfig.load(BOOTSTRAP);
        CalibrationConfig contrast = CalibrationConfig.load(CONTRAST);

        Path fresh = Files.createTempDirectory("confidence_agreement");
        generateRun(fresh.resolve("run_bootstrap"), bootstrap);
        generateRun(fresh.resolve("run_contrast"), contrast);

        if (!Files.isDirectory(FIXTURE_DIR)) {
            // First generation: write the fixture for review and commit. Deliberate, once.
            Files.createDirectories(FIXTURE_DIR);
            for (String run : new String[]{"run_bootstrap", "run_contrast"}) {
                Files.createDirectories(FIXTURE_DIR.resolve(run));
                for (String file : new String[]{"frames.csv", "manifest.json"}) {
                    Files.copy(fresh.resolve(run).resolve(file), FIXTURE_DIR.resolve(run).resolve(file));
                }
            }
        }

        // Guard: committed fixture == what the current implementation produces, byte-for-byte
        // (modulo line endings, which the repository's checkout settings may rewrite). On a
        // mismatch, FAIL — never overwrite. Regeneration is: delete the directory, re-run, review
        // the diff, commit.
        for (String run : new String[]{"run_bootstrap", "run_contrast"}) {
            for (String file : new String[]{"frames.csv", "manifest.json"}) {
                String committed = normalise(Files.readString(
                        FIXTURE_DIR.resolve(run).resolve(file), StandardCharsets.UTF_8));
                String generated = normalise(Files.readString(
                        fresh.resolve(run).resolve(file), StandardCharsets.UTF_8));
                assertEquals(generated, committed,
                        run + "/" + file + " diverged from the committed fixture. The scorer, "
                        + "extractor or writer changed a persisted byte; if intentional, delete "
                        + "evaluation/tests/fixtures/confidence_agreement/ and regenerate deliberately.");
            }
        }

        // Structural spot-checks so the fixture provably covers the contract's cases.
        var lines = Files.readAllLines(FIXTURE_DIR.resolve("run_bootstrap/frames.csv"));
        String[] header = lines.get(0).split(",", -1);
        assertEquals(37, header.length);
        assertTrue(lines.get(1).contains("not_established"), "first-frame case");
        assertTrue(lines.get(8).contains("signals_unavailable"), "absent-residuals case (bootstrap)");
        assertTrue(lines.get(10).contains("not_produced"), "estimator-failed case");
        assertTrue(lines.get(16).contains("degraded"), "on-threshold frame must be degraded, not rejected");

        var contrastLines = Files.readAllLines(FIXTURE_DIR.resolve("run_contrast/frames.csv"));
        assertTrue(contrastLines.get(2).contains("signals_unavailable"),
                "warm-up case under the contrast calibration");
        // The two-calibrations case: same frame, different verdict families.
        assertTrue(lines.get(2).contains("usable") && contrastLines.get(2).contains("rejected"),
                "the two calibrations must visibly disagree (FR-019)");
    }
}
