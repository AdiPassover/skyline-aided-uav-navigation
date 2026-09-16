package org.boofcv.evaluation;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * {@code VoRunnerConfig}'s optional {@code metric_readout} block: everything the runtime metric
 * local pose needs, declared in a version-controlled file rather than hardcoded (working rule 10).
 *
 * <p><b>Absent by default.</b> A config without a {@code metric_readout} block runs exactly as it
 * did before 2026-09-06 — pixel pose, no height channel, no extra output file — so every committed
 * config and every committed run record is unaffected.
 *
 * <p>The four required values are the ones {@code DEC-VO-007} names and no others: {@code fx_native}
 * and the downsample factor give {@code f_working}; {@code h0_agl_m} is the external metre the
 * monocular scale ambiguity requires (D2); {@code tau_stale_s} is the declared hold limit (D6).
 * Nothing here is fitted, and {@code h0_agl_m} must not be taken from trajectory ground truth.
 *
 * <h2>Where the height stream comes from in a replay</h2>
 *
 * <p>{@code height_csv} names a CSV carrying a takeoff-relative height series — for the
 * {@code EXP-VO-014} datasets that is {@code datasets/&lt;id&gt;/terrain.csv}'s
 * {@code baro_relative_m}. {@code VoRunner} submits each sample to the channel at the moment it
 * becomes available and never earlier, so a replay is causally identical to a live stream. That is
 * the whole of the "runtime barometric stream" in an offline replay: the file is a recording of a
 * sensor, not a lookup table the estimator may index into.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class MetricReadoutSettings {

    /** Master switch. False (or an absent block) leaves the runtime exactly as it was. */
    @JsonProperty("enabled")
    public boolean enabled = false;

    /** Horizontal focal length in pixels at the camera's NATIVE resolution. Required when enabled. */
    @JsonProperty("fx_native_px")
    public Double fxNativePx;

    /**
     * Camera-to-imaged-surface height at the reference (first) frame, metres. Required when enabled.
     * External by {@code DEC-VO-007} D2 — mission setup, survey, clearance, rangefinder, or the
     * simulator's own setup value. <b>Never</b> inferred from trajectory ground truth.
     */
    @JsonProperty("h0_agl_m")
    public Double h0AglM;

    /** Declared height-hold limit, seconds ({@code DEC-VO-007} D6). Default matches {@code EXP-VO-012}. */
    @JsonProperty("tau_stale_s")
    public double tauStaleS = 2.0;

    /**
     * The height channel's nominal sampling interval, seconds. Sets the FRESH/HELD boundary at
     * {@code 1.5x} this, matching {@code baro.py}. Default 0.1 s = 10 Hz, the imagery's own rate in
     * every campaign so far.
     */
    @JsonProperty("nominal_sample_interval_s")
    public double nominalSampleIntervalS = 0.1;

    /** CSV holding the takeoff-relative height stream. Required when enabled. */
    @JsonProperty("height_csv")
    public String heightCsv;

    /** Column carrying the sample instant, seconds, on the frame clock. */
    @JsonProperty("height_time_column")
    public String heightTimeColumn = "timestamp_s";

    /** Column carrying the height series. Its MEANING is declared by {@link #heightSemantics}. */
    @JsonProperty("height_relative_column")
    public String heightRelativeColumn = "baro_relative_m";

    /**
     * <b>What the height column means.</b> Required to be one of two values, and the reason it
     * exists is {@code DEC-VO-009} D4: a simulator dataset carries both a production-like
     * takeoff-relative channel and an oracle AGL, in adjacent columns of the same file, and pointing
     * the config at the wrong one would silently produce a differently-armed run that still looked
     * ordinary.
     *
     * <dl>
     *   <dt>{@code takeoff_relative} (default)</dt>
     *   <dd><b>Production-like.</b> The column is height relative to the takeoff datum, zero at the
     *       datum by construction, and {@code h_AGL = h0 + column}. This is the only arm a real
     *       platform can fly.</dd>
     *   <dt>{@code oracle_agl_diagnostic}</dt>
     *   <dd><b>DIAGNOSTIC ONLY, and labelled as such wherever it is reported.</b> The column is the
     *       renderer's true height above the imaged surface. The runner subtracts the first sample
     *       to form a relative series, so the arithmetic is identical and only the array differs
     *       ({@code EXP-VO-012}'s oracle arm) -- but the run manifest records
     *       {@code height_arm: "oracle"} and the caveat, because <b>no sensor on the author's
     *       platform supplies this quantity at survey height</b> ({@code LIT-VO-006} section 2).
     *       Its value is that it measures what a rangefinder would be worth, and that it makes the
     *       terrain limitation a measured difference rather than an assertion.</dd>
     * </dl>
     */
    @JsonProperty("height_semantics")
    public String heightSemantics = "takeoff_relative";

    /** True when {@link #heightSemantics} selects the diagnostic oracle arm. */
    public boolean isOracleArm() {
        if ("takeoff_relative".equalsIgnoreCase(heightSemantics)) {
            return false;
        }
        if ("oracle_agl_diagnostic".equalsIgnoreCase(heightSemantics)) {
            return true;
        }
        throw new IllegalArgumentException("Unknown metric_readout.height_semantics '"
                + heightSemantics + "'; expected \"takeoff_relative\" (production-like) or "
                + "\"oracle_agl_diagnostic\" (DIAGNOSTIC ONLY -- no sensor supplies it)");
    }

    /**
     * Transport/filtering latency in seconds: a sample taken at {@code t} becomes available at
     * {@code t + latency}. Modelled here because only the caller knows it
     * ({@code CausalHeightChannel.submit} takes availability time, not sample time).
     */
    @JsonProperty("height_latency_s")
    public double heightLatencyS = 0.0;

    /**
     * Decimation of the height stream: keep every Nth row. 1 = every row. Exists to replay a height
     * channel slower than the imagery without editing the recording ({@code EXP-VO-012} R5).
     */
    @JsonProperty("height_decimation")
    public int heightDecimation = 1;

    /**
     * When true the run's published pose ({@code est_x}/{@code est_y} in {@code frames.csv}) becomes
     * the METRIC pose, in metres, and {@code navigation_source} is reported as {@code metric_local}.
     *
     * <p><b>Default false, and that is a deliberate migration decision.</b> The run-record contract
     * says {@code est_x} is "Estimator units. Not metres.", and every committed run record, every
     * {@code naveval} evaluation and every published VO figure was produced under it. Flipping this
     * on a config that reproduces a committed baseline would change that baseline's units without
     * changing its {@code run_id}. Turn it on for a run whose purpose is a metric pose, and say so
     * in the run id.
     */
    @JsonProperty("publish_as_navigation_source")
    public boolean publishAsNavigationSource = false;
}
