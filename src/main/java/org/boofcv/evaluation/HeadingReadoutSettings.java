package org.boofcv.evaluation;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.boofcv.stitching.metric.HeadingReadoutConfig;
import org.boofcv.stitching.metric.HeadingSemantics;

/**
 * The {@code heading_readout} block of a {@link VoRunnerConfig}: the authoritative external heading
 * that owns the navigation direction.
 *
 * <p>Optional and off by default. A config without it behaves exactly as {@code DEC-VO-009} shipped
 * — the metric pose is rotated by the visually integrated yaw and every committed run reproduces
 * byte-for-byte.
 *
 * <p>It requires {@code metric_readout} to be enabled: an authoritative heading rotates a metric
 * increment, and without a height channel there is none to rotate.
 */
@JsonIgnoreProperties(ignoreUnknown = false)
public class HeadingReadoutSettings {

    /** Master switch. */
    @JsonProperty("enabled")
    public boolean enabled = false;

    /**
     * <b>What the heading column means.</b> Required; there is no default, because defaulting it is
     * exactly how {@code EXP-VO-013} R6's ~66° error would return.
     *
     * <dl>
     *   <dt>{@code fc_ahrs_body_compass}</dt>
     *   <dd>The <b>airframe's</b> heading from a flight controller's fused attitude — MARS-LVIG's
     *       {@code attitude.csv:yaw_compass_deg}. Needs {@link #deltaMountDeg}.</dd>
     *   <dt>{@code sim_nadir_camera_heading}</dt>
     *   <dd>Already the <b>nadir camera's</b> heading — a UE capture's
     *       {@code groundtruth.csv:heading_deg}. Used directly; no mounting correction.</dd>
     * </dl>
     */
    @JsonProperty("heading_semantics")
    public String headingSemantics;

    /**
     * Declared camera-to-body mounting yaw, degrees, added to every sample on the
     * {@code fc_ahrs_body_compass} arm. <b>Required there, refused elsewhere, and never fitted.</b>
     *
     * <p>It is a platform calibration in the same sense {@code h0_agl_m} is a mission parameter: it
     * comes from mechanical design, a one-time calibration, or a vendor file. It must not be derived
     * per flight against trajectory ground truth at runtime.
     */
    @JsonProperty("delta_mount_deg")
    public Double deltaMountDeg;

    /** Free-text provenance of {@link #deltaMountDeg}, copied verbatim into the run manifest. */
    @JsonProperty("delta_mount_provenance")
    public String deltaMountProvenance;

    /** Declared hold limit, seconds. Beyond it a reading is STALE: still held, still used, flagged. */
    @JsonProperty("tau_stale_s")
    public double tauStaleS = 2.0;

    /**
     * The heading channel's nominal sampling interval, seconds. Sets the FRESH/HELD boundary at
     * {@code 1.5x} this. Default 0.01 s = 100 Hz, the measured rate of the only real channel here.
     */
    @JsonProperty("nominal_sample_interval_s")
    public double nominalSampleIntervalS = 0.01;

    /**
     * Plausibility ceiling on the implied yaw rate between consecutive samples, degrees per second.
     * A sample above it is rejected, counted, and the previous valid heading is held.
     */
    @JsonProperty("max_rate_deg_per_s")
    public double maxRateDegPerS = HeadingReadoutConfig.DEFAULT_MAX_RATE_DEG_PER_S;

    /** CSV holding the heading stream. Required when enabled. */
    @JsonProperty("heading_csv")
    public String headingCsv;

    /** Column carrying the sample instant, seconds, on the frame clock. */
    @JsonProperty("heading_time_column")
    public String headingTimeColumn = "timestamp_s";

    /** Column carrying the heading series. Its MEANING is declared by {@link #headingSemantics}. */
    @JsonProperty("heading_column")
    public String headingColumn;

    /** Transport/filtering latency, seconds: a sample taken at {@code t} is available at {@code t + latency}. */
    @JsonProperty("heading_latency_s")
    public double headingLatencyS = 0.0;

    /** Decimation of the heading stream: keep every Nth row. 1 = every row. */
    @JsonProperty("heading_decimation")
    public int headingDecimation = 1;

    /** The declared semantics, parsed and validated. */
    public HeadingSemantics semantics() {
        return HeadingSemantics.parse(headingSemantics);
    }

    /** Builds the runtime configuration, refusing an absent mounting calibration where one is needed. */
    public HeadingReadoutConfig toConfig() {
        return new HeadingReadoutConfig(semantics(), deltaMountDeg, tauStaleS,
                nominalSampleIntervalS, maxRateDegPerS);
    }
}
