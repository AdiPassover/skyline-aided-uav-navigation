package org.boofcv.stitching.metric;

/**
 * Freshness of the height sample a metric increment was converted with ({@code DEC-VO-007} D6).
 *
 * <p>Four states, because the offline implementation this ports distinguishes exactly two flags and
 * a consumer needs to tell all four situations apart. The mapping to
 * {@code evaluation/tools/exp_vo_012/baro.py}'s {@code SampledHeight} is stated per constant and is
 * asserted by {@code CausalHeightChannelTest} — it is the contract the Python↔Java parity rests on.
 *
 * <p><b>None of these is "no height".</b> A channel is constructed with {@code h0}, so a height
 * always exists once the metric layer is enabled; what varies is how old the <i>relative</i> sample
 * behind it is. {@link #UNAVAILABLE} is the one state in which no metric pose may be produced, and
 * it is reachable only from a physically impossible height, never from mere staleness.
 */
public enum HeightStatus {

    /**
     * A sample arrived within one nominal sampling interval (×1.5, the same threshold
     * {@code baro.py::_resample} uses). Python equivalent: {@code valid=True, stale=False}.
     */
    FRESH,

    /**
     * A zero-order hold: the most recent sample is older than {@code 1.5 ×} the nominal interval but
     * still within {@code tau_stale}. The value is legitimate and the metric pose is <b>not</b>
     * degraded — it is simply held. Python equivalent: {@code valid=True, stale=True}.
     *
     * <p>Also the state of every frame before the channel's first sample, where the takeoff-relative
     * datum is {@code 0} by its own construction rather than by measurement — see
     * {@link CausalHeightChannel} <i>Before the first sample</i>.
     */
    HELD,

    /**
     * The most recent sample is older than the declared {@code tau_stale}. The height is still held
     * and the increment is still converted — that is {@code DEC-VO-007} D6 and the offline
     * {@code integrate_metric}, verbatim — but the frame is <b>flagged degraded</b> so a consumer
     * can see exactly which part of the trajectory used an out-of-date scale.
     * Python equivalent: {@code valid=False}.
     *
     * <p>{@code EXP-VO-012} R6 measured that {@code tau_stale} changes what is <i>reported</i>, not
     * what is <i>computed</i>: the endpoint error is identical for every {@code tau_stale}. Keep it
     * that way — tuning {@code tau_stale} must never move the trajectory.
     */
    STALE,

    /**
     * No usable height at all, so <b>no metric pose is produced for this frame</b>. Reached only
     * when the resulting AGL is non-positive or non-finite — the case the offline
     * {@code integrate_metric} answers by raising, which a flight-time runtime cannot do.
     *
     * <p>This is the deliberate runtime-only extension of D6 recorded in {@code DEC-VO-007}
     * <i>Amendment 2026-09-06</i>: a safe explicit invalidity, never a metrically-labelled pose
     * computed from an unknown scale.
     */
    UNAVAILABLE;

    /** True when a metric increment may be produced from this sample (everything but {@link #UNAVAILABLE}). */
    public boolean usable() {
        return this != UNAVAILABLE;
    }

    /**
     * True when the resulting metric pose must be reported as degraded — {@link #STALE} or
     * {@link #UNAVAILABLE}. Exactly the negation of {@code SampledHeight.valid} in Python.
     */
    public boolean degraded() {
        return this == STALE || this == UNAVAILABLE;
    }
}
