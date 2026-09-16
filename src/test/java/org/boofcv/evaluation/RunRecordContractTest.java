package org.boofcv.evaluation;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Golden-file round trip (contracts/run-record.md "Contract tests"): a record written by
 * {@link RunRecordWriter} must have the same structure — CSV header, manifest field set,
 * estimator_config key set — as the checked-in golden v1.0.0 fixture from T007, so writer and
 * reader cannot drift apart silently. Value-level recovery through the Python reader is T047.
 */
public class RunRecordContractTest {

    private static final Path GOLDEN_RUN = Paths.get("evaluation", "tests", "fixtures", "golden_run");

    @Test
    public void testFramesCsvHeaderMatchesGoldenFixture(@TempDir Path tempDir) throws IOException {
        Path writtenDir = writeSampleRecord(tempDir);

        List<String> goldenLines = Files.readAllLines(GOLDEN_RUN.resolve("frames.csv"));
        List<String> writtenLines = Files.readAllLines(writtenDir.resolve("frames.csv"));

        assertEquals(goldenLines.get(0), writtenLines.get(0),
                "frames.csv header must match the golden fixture exactly (contracts/run-record.md column order)");
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testManifestStructureMatchesGoldenFixture(@TempDir Path tempDir) throws IOException {
        Path writtenDir = writeSampleRecord(tempDir);

        ObjectMapper mapper = new ObjectMapper();
        Map<String, Object> golden = mapper.readValue(GOLDEN_RUN.resolve("manifest.json").toFile(), Map.class);
        Map<String, Object> written = mapper.readValue(writtenDir.resolve("manifest.json").toFile(), Map.class);

        assertEquals(golden.keySet(), written.keySet(),
                "manifest.json top-level fields must match contracts/run-record.md");

        Map<String, Object> goldenEnv = (Map<String, Object>) golden.get("environment");
        Map<String, Object> writtenEnv = (Map<String, Object>) written.get("environment");
        assertEquals(goldenEnv.keySet(), writtenEnv.keySet());

        Map<String, Object> goldenConfig = (Map<String, Object>) golden.get("estimator_config");
        Map<String, Object> writtenConfig = (Map<String, Object>) written.get("estimator_config");
        assertEquals(goldenConfig.keySet(), writtenConfig.keySet(),
                "estimator_config keys must match the golden fixture (COMP-001 §3.3 renamed parameters)");

        assertEquals("1.0.0", written.get("schema_version"));
    }

    private static Path writeSampleRecord(Path tempDir) throws IOException {
        RunManifest manifest = new RunManifest();
        manifest.runId = "contract-test-run";
        manifest.datasetId = "contract-test-dataset";
        manifest.datasetRevision = "rev1";
        manifest.estimatorId = "stitching-vo";
        manifest.estimatorVersion = "test-commit";
        manifest.evaluatorCaptureVersion = "test-commit";
        manifest.estimatorConfig = sampleEstimatorConfig();
        manifest.environment = new Environment("host", "os", "cpu", "19.0.2", 4096, false);
        manifest.runTimestamp = "2026-08-12T00:00:00Z";

        Path runDir = tempDir.resolve("run");
        try (RunRecordWriter writer = new RunRecordWriter(runDir, manifest, 1)) {
            writer.writeFrame(new FrameEstimateRow(
                    0, 0.0, 0.0, 0.0, null, 0.0, true, "init", 0,
                    0.5, 0.0, 100.0, 0.0, 0.5, 100.0, 0.0, 0.0, 1.0,
                    1, 1, 1000L));
        }
        return runDir;
    }

    private static Map<String, Object> sampleEstimatorConfig() {
        Map<String, Object> config = new LinkedHashMap<>();
        config.put("shrinkScale", 0.5);
        config.put("minDistanceFromBorder", 10);
        config.put("detectorType", "SHI_TOMASI");
        config.put("maxFeatures", 300);
        config.put("detectorRadius", 3);
        config.put("detectorThreshold", 1.0);
        config.put("kltPyramidLevels", 4);
        config.put("kltFeatureRadius", 3);
        config.put("ransacIterations", 220);
        config.put("inlierThresholdSq", 3.0);
        config.put("outlierPrune", 2);
        config.put("absoluteMinimumTracks", 30);
        config.put("respawnTrackFraction", 0.6);
        config.put("respawnCoverageFraction", 0.5);
        config.put("refineEstimate", false);
        config.put("maxJumpFraction", 0.55);
        return config;
    }
}
