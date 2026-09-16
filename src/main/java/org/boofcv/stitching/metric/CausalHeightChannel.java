package org.boofcv.stitching.metric;

/**
 * The runtime height input: a <b>causal zero-order hold</b> over a takeoff-relative height stream,
 * plus the externally supplied {@code h0} that turns it into height above the imaged surface.
 *
 * <p>This is the runtime counterpart of {@code evaluation/tools/exp_vo_012/baro.py}'s resampler, and
 * it is deliberately the <i>whole</i> of the sensor-side logic: nothing here filters, fuses,
 * smooths, predicts or estimates. {@code DEC-VO-007} D5 says so explicitly, and says why — a ZOH is
 * resampling, not state estimation, and adopting one must not be mistaken for evidence that fusion
 * is needed. {@code EXP-VO-012} Phase 8 decided outcome A (direct constraint sufficient) from
 * measurement, not from convention.
 *
 * <h2>What it computes</h2>
 *
 * <pre>
 *   readAt(t):  s      = the most recent submitted sample with availabilityTime &lt;= t
 *               age    = t - s.availabilityTime
 *               h_AGL  = h0 + s.relativeHeight
 *               status = age &lt;= 1.5*interval ? FRESH
 *                      : age &lt;= tauStale     ? HELD
 *                      :                        STALE
 * </pre>
 *
 * <h2>Causality is structural, not a policy</h2>
 *
 * <p>The channel can only see samples that have been {@link #submit}ted, and a submitted sample
 * carries the time it became <i>available</i>. There is no array to index into and therefore no
 * future to consult — a stronger guarantee than the offline resampler's
 * {@code searchsorted(..., "right") - 1}, and what {@code MetricNavigationCausalityTest} pins.
 *
 * <h2>Before the first sample</h2>
 *
 * <p>A takeoff-relative channel reads <b>zero at its own datum by construction</b> — that is what
 * makes the absolute-accuracy term cancel ({@code LIT-VO-006} section 1) and it is a definition, not
 * a measurement. So before any sample has arrived the channel reports {@code relativeHeight = 0},
 * i.e. {@code h_AGL = h0}, marked {@link HeightStatus#HELD}.
 *
 * <p>This is the one place the runtime and {@code baro.py} can differ, and the difference is in the
 * runtime's favour: {@code _resample} clips its index to {@code 0} and uses {@code values[0]} — the
 * first sample, which under non-zero latency lies in that frame's <i>future</i>. Its own docstring
 * calls that branch "the honest description of what a real system does at startup"; this class
 * implements the description instead of approximating it with a future sample. For every
 * zero-latency arm the two are identical, because {@code baro.py} anchors its sample grid at the
 * first frame and its relative datum is exactly {@code 0} there. The residual difference is bounded
 * by one sample of channel noise over the first {@code latency} seconds, and is asserted rather than
 * assumed by {@code CausalHeightChannelTest}.
 *
 * <h2>What it will not do</h2>
 *
 * <ul>
 *   <li>It will never fall back to visual scale, at any age ({@code DEC-VO-007} D4/D6). The visual
 *       channel is not reachable from this class and cannot be made so without changing its
 *       constructor signature.</li>
 *   <li>It will never interpolate. Linear interpolation is offline-only ({@code DEC-VO-007} D5) and
 *       is not implemented here at all, so it cannot be selected by accident.</li>
 *   <li>It will never estimate {@code h0}. {@code h0} is a constructor argument ({@code DEC-VO-007}
 *       D2), and inferring it from trajectory ground truth is forbidden by that record.</li>
 * </ul>
 *
 * <p>Not thread-safe. {@link #submit} and {@link #readAt} are expected on the frame-processing
 * thread, which is where the estimator already runs.
 */
public final class CausalHeightChannel {

    /** {@code baro.py::_resample}: {@code stale = age > interval * 1.5}. */
    private static final double STALE_INTERVAL_FACTOR = 1.5;

    private final double h0M;
    private final double tauStaleS;
    private final double nominalIntervalS;

    private boolean hasSample;
    private double lastSampleTimeS;
    private double lastRelativeM;
    private long sampleCount;

    /**
     * @param h0M              camera-to-imaged-surface height at the navigation reference frame,
     *                         metres, strictly positive. A <b>mission-initialisation parameter</b>
     *                         supplied from outside ({@code DEC-VO-007} D2): mission setup, a
     *                         launch-site survey, a known clearance, a rangefinder reading at the
     *                         reference frame, or — in simulation only — the simulator's own ground
     *                         truth, which is legitimate because the simulator <i>is</i> the mission
     *                         setup. It must never be inferred from trajectory ground truth.
     * @param tauStaleS        declared hold limit, seconds ({@code DEC-VO-007} D6). Beyond it a
     *                         reading is {@link HeightStatus#STALE}: still held, still used, and
     *                         flagged. It must not be tuned against ground truth, and it must not
     *                         change the trajectory — only what is reported degraded
     *                         ({@code EXP-VO-012} R6).
     * @param nominalIntervalS the channel's nominal sampling interval, seconds; the
     *                         {@link HeightStatus#FRESH}/{@link HeightStatus#HELD} boundary sits at
     *                         {@code 1.5x} this, matching {@code baro.py}. Declare the sensor's own
     *                         rate here, not the camera's, unless they are the same.
     */
    public CausalHeightChannel(double h0M, double tauStaleS, double nominalIntervalS) {
        if (!(h0M > 0.0) || !Double.isFinite(h0M)) {
            throw new IllegalArgumentException(
                    "h0 must be a finite positive height above the imaged surface, in metres "
                    + "(DEC-VO-007 D2); got " + h0M);
        }
        if (!(tauStaleS > 0.0) || !Double.isFinite(tauStaleS)) {
            throw new IllegalArgumentException(
                    "tau_stale must be finite and positive; got " + tauStaleS);
        }
        if (!(nominalIntervalS > 0.0) || !Double.isFinite(nominalIntervalS)) {
            throw new IllegalArgumentException(
                    "nominal sample interval must be finite and positive; got " + nominalIntervalS);
        }
        this.h0M = h0M;
        this.tauStaleS = tauStaleS;
        this.nominalIntervalS = nominalIntervalS;
    }

    /**
     * Accepts one takeoff-relative height sample.
     *
     * @param availabilityTimeS when the value became usable by this consumer, seconds, on the same
     *                          clock the frame timestamps use. For a sensor with transport or
     *                          filtering latency this is the sample instant <i>plus</i> that
     *                          latency — the caller models the latency, because only the caller
     *                          knows it.
     * @param relativeHeightM   height relative to the takeoff datum, metres, positive up. Zero at
     *                          the datum by construction.
     * @throws IllegalArgumentException on a non-finite value, or on a sample older than the previous
     *                                  one — an out-of-order stream is a real fault, and silently
     *                                  reordering it would fabricate a history that never occurred
     */
    public void submit(double availabilityTimeS, double relativeHeightM) {
        if (!Double.isFinite(availabilityTimeS) || !Double.isFinite(relativeHeightM)) {
            throw new IllegalArgumentException("height sample must be finite; got t="
                    + availabilityTimeS + ", h_rel=" + relativeHeightM);
        }
        if (hasSample && availabilityTimeS < lastSampleTimeS) {
            throw new IllegalArgumentException("height samples must arrive in non-decreasing "
                    + "availability order; got " + availabilityTimeS + " after " + lastSampleTimeS);
        }
        hasSample = true;
        lastSampleTimeS = availabilityTimeS;
        lastRelativeM = relativeHeightM;
        sampleCount++;
    }

    /**
     * The height to convert a frame at {@code frameTimeS} with. Pure: calling it twice returns the
     * same answer, and it never mutates the channel.
     *
     * @throws IllegalArgumentException when {@code frameTimeS} is not finite
     */
    public HeightReading readAt(double frameTimeS) {
        if (!Double.isFinite(frameTimeS)) {
            throw new IllegalArgumentException("frame time must be finite; got " + frameTimeS);
        }
        double sampleTimeS;
        double relativeM;
        double ageS;
        HeightStatus status;

        if (!hasSample || frameTimeS < lastSampleTimeS) {
            // No sample has become available at or before this frame. The takeoff-relative datum is
            // zero by its own definition, so h_AGL = h0 -- held, not measured. See the class comment.
            sampleTimeS = frameTimeS;
            relativeM = 0.0;
            ageS = 0.0;
            status = HeightStatus.HELD;
        } else {
            sampleTimeS = lastSampleTimeS;
            relativeM = lastRelativeM;
            ageS = frameTimeS - lastSampleTimeS;
            if (ageS <= nominalIntervalS * STALE_INTERVAL_FACTOR) {
                status = HeightStatus.FRESH;
            } else if (ageS <= tauStaleS) {
                status = HeightStatus.HELD;
            } else {
                status = HeightStatus.STALE;
            }
        }

        double heightAglM = h0M + relativeM;
        if (!Double.isFinite(heightAglM) || heightAglM <= 0.0) {
            // The offline integrate_metric raises here ("non-positive camera height"). A flight-time
            // runtime cannot raise, so it refuses to produce a metric pose instead -- explicitly.
            status = HeightStatus.UNAVAILABLE;
        }
        return new HeightReading(frameTimeS, sampleTimeS, ageS, relativeM, h0M, heightAglM, status);
    }

    /** The externally supplied reference-frame height, metres. Never estimated ({@code DEC-VO-007} D2). */
    public double h0M() {
        return h0M;
    }

    /** The declared hold limit, seconds ({@code DEC-VO-007} D6). */
    public double tauStaleS() {
        return tauStaleS;
    }

    /** The declared nominal sampling interval, seconds. */
    public double nominalIntervalS() {
        return nominalIntervalS;
    }

    /** How many samples have been submitted. Diagnostic; used in no computation. */
    public long sampleCount() {
        return sampleCount;
    }

    /** True once at least one sample has been submitted. */
    public boolean hasSample() {
        return hasSample;
    }
}
