package org.boofcv.evaluation;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import java.util.Map;

/**
 * {@code manifest.json} DTO per contracts/run-record.md. {@code frameCount} and
 * {@code processedCount} are only known once capture completes, so this object is
 * mutated by {@link RunRecordWriter} rather than being fully built up front.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonPropertyOrder({
        "schema_version", "run_id", "dataset_id", "dataset_revision",
        "estimator_id", "estimator_version", "estimator_config", "evaluator_capture_version",
        "environment", "run_timestamp", "frame_count", "processed_count", "completed", "confidence"
})
public class RunManifest {

    public static final String SCHEMA_VERSION = "1.0.0";

    /** Schema of a record carrying the confidence columns (contracts/run-record-v1.1.md). */
    public static final String SCHEMA_VERSION_CONFIDENCE = "1.1.0";

    @JsonProperty("schema_version")
    public String schemaVersion = SCHEMA_VERSION;

    /**
     * The confidence block (contracts/run-record-v1.1.md), or {@code null} for a run captured
     * without confidence. Absent ≠ {@code calibration_validated: false}: the first means no
     * verdict was computed, the second that one was, under an unvalidated calibration.
     */
    @com.fasterxml.jackson.annotation.JsonInclude(com.fasterxml.jackson.annotation.JsonInclude.Include.NON_NULL)
    @JsonProperty("confidence")
    public Map<String, Object> confidence;

    @JsonProperty("run_id")
    public String runId;

    @JsonProperty("dataset_id")
    public String datasetId;

    @JsonProperty("dataset_revision")
    public String datasetRevision;

    @JsonProperty("estimator_id")
    public String estimatorId;

    @JsonProperty("estimator_version")
    public String estimatorVersion;

    /** Values as actually applied, not requested (contracts/run-record.md field rules). */
    @JsonProperty("estimator_config")
    public Map<String, Object> estimatorConfig;

    @JsonProperty("evaluator_capture_version")
    public String evaluatorCaptureVersion;

    @JsonProperty("environment")
    public Environment environment;

    @JsonProperty("run_timestamp")
    public String runTimestamp;

    @JsonProperty("frame_count")
    public int frameCount;

    @JsonProperty("processed_count")
    public int processedCount;

    /** False when the run was interrupted (contracts/run-record.md field rules). */
    @JsonProperty("completed")
    public boolean completed;

    public RunManifest() {
        // Jackson
    }
}
