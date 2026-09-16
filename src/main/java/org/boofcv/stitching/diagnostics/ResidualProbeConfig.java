package org.boofcv.stitching.diagnostics;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Config schema for {@link ResidualProbeApp} — the {@code EXP-VO-001} measurement harness.
 *
 * <p>Deliberately carries <em>no estimator parameters of its own</em>. It points at an existing
 * {@code VoRunnerConfig} JSON, and the probe builds its estimator from that file. The reason is
 * provenance: the residuals measured by this probe must be attributable to a trajectory that has
 * already been characterised, and the only way to guarantee that without a second source of truth
 * is to read the very same configuration file the reference run was captured from. A probe-local
 * copy of the parameters could drift from it silently.
 *
 * <p>Not a cross-component contract. This is an experiment artifact; persistence of residuals
 * into the run record is a separate concern of the run-record writer.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ResidualProbeConfig {

    /** Research-record ID this probe run belongs to, e.g. {@code EXP-VO-001}. */
    @JsonProperty("experiment_id")
    public String experimentId;

    /** Short identifier for this case within the experiment, used as the output subdirectory. */
    @JsonProperty("case_id")
    public String caseId;

    /**
     * Path to the {@code VoRunnerConfig} JSON that supplies the dataset and every estimator
     * parameter. Resolved relative to the working directory, matching how {@code VoRunnerApp}
     * already resolves its own {@code dataset_dir}.
     */
    @JsonProperty("vo_runner_config")
    public String voRunnerConfig;

    /**
     * Directory holding an already-committed run record captured from {@link #voRunnerConfig} by
     * the <em>non</em>-instrumented path. When present, the probe re-derives that run's per-frame
     * pose/event columns and reports, in {@code reproduction.json}, whether the instrumented path
     * reproduced them exactly. Optional: {@code null} means no reference is available and the
     * check is reported as not performed rather than as passed.
     */
    @JsonProperty("reference_run_dir")
    public String referenceRunDir;

    /** Directory the probe writes {@code <case_id>/} into. */
    @JsonProperty("output_dir")
    public String outputDir = "evaluations/exp-vo-001";

    /**
     * Overrides the dataset root declared by {@link #voRunnerConfig}. Needed because dataset
     * imagery is not stored with the configs. Null means use the config's own path.
     */
    @JsonProperty("dataset_dir_override")
    public String datasetDirOverride;
}
