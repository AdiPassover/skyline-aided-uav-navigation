package org.boofcv.stitching.metric;

/**
 * The runtime heading input: a <b>causal zero-order hold</b> over an external heading stream, plus
 * the declared camera mounting calibration that turns it into the navigation camera's azimuth.
 *
 * <p>Structurally this is {@link CausalHeightChannel}'s twin, and deliberately so — same causality
 * guarantee, same four-state freshness model, same refusal to interpolate, filter, fuse or predict.
 * <b>No blending, no complementary filter, no Kalman, no confidence scalar.</b> The heading replaces
 * a term; it does not join an estimator.
 *
 * <h2>What it computes</h2>
 *
 * <pre>
 *   readAt(t):  s      = the most recent ACCEPTED sample with availabilityTime &lt;= t
 *               age    = t - s.availabilityTime
 *               yawNav = wrap360(s.heading + deltaMount)
 *               status = age &lt;= 1.5*interval ? FRESH
 *                      : age &lt;= tauStale     ? HELD
 *                      :                        STALE
 * </pre>
 *
 * <h2>Three differences from the height channel, each forced by the physics</h2>
 *
 * <ol>
 *   <li><b>Before the first sample the answer is {@link HeadingStatus#UNAVAILABLE}, not held.</b> A
 *       takeoff-relative height reads zero at its own datum <i>by definition</i>, so {@code h0}
 *       alone still yields an AGL. An absolute azimuth has no such datum, and inventing one — most
 *       temptingly by promoting the visually integrated yaw — is the single thing this class must
 *       never do.</li>
 *   <li><b>Every difference is wrap-safe.</b> Headings live on a circle; a naive subtraction across
 *       the 0°/360° branch cut is wrong by up to 360°, which is why {@code heading_metrics.py}
 *       already carries the same warning on the analysis side.</li>
 *   <li><b>A deterministic rate-plausibility gate.</b> Measured on the committed real-flight
 *       channel, 21 of 118 940 samples on {@code amtown01-d} and 2 of 63 003 on {@code amtown01-c}
 *       imply yaw rates of 274–5141 °/s, against a p99 of 12–16 °/s and a legitimate maximum of
 *       26.7 °/s; both HKairport windows have none. A single threshold rejects exactly those and
 *       nothing else. A rejected sample does not update the latch — the previous valid heading is
 *       held under the ordinary staleness policy — and it is counted so a run can report it. This is
 *       a gate, not a filter: it changes which samples are <i>accepted</i>, never what an accepted
 *       sample <i>says</i>.</li>
 * </ol>
 *
 * <p>Not thread-safe; it lives on the frame-processing thread with the estimator.
 */
public final class CausalHeadingChannel {

    /** Same FRESH/HELD boundary factor the height channel uses, for one reading convention. */
    private static final double STALE_INTERVAL_FACTOR = 1.5;

    private final HeadingSemantics semantics;
    private final double mountOffsetDeg;
    private final double tauStaleS;
    private final double nominalIntervalS;
    private final double maxRateDegPerS;

    private boolean hasSample;
    private double lastSampleTimeS;
    private double lastHeadingDeg;
    private double lastSubmittedTimeS = Double.NEGATIVE_INFINITY;
    private long sampleCount;
    private long rejectedSampleCount;

    /**
     * @param semantics        which arm this is; {@link HeadingSemantics#requiresMountCalibration()}
     *                         decides whether {@code mountOffsetDeg} is meaningful
     * @param mountOffsetDeg   declared camera-to-body mounting yaw, degrees, added to every sample.
     *                         A <b>calibration input</b>: it must come from mechanical design, a
     *                         one-time platform calibration, or a vendor file — never from a
     *                         runtime fit and never from trajectory ground truth. Must be exactly
     *                         {@code 0} for an arm whose source is already the camera's heading.
     * @param tauStaleS        declared hold limit, seconds; beyond it a reading is
     *                         {@link HeadingStatus#STALE} — still held, still used, flagged
     * @param nominalIntervalS the channel's nominal sampling interval, seconds; the FRESH/HELD
     *                         boundary sits at {@code 1.5x} this. Declare the sensor's rate, not the
     *                         camera's
     * @param maxRateDegPerS   plausibility ceiling on the implied yaw rate between consecutive
     *                         samples, degrees per second; a sample above it is rejected and counted
     */
    public CausalHeadingChannel(HeadingSemantics semantics, double mountOffsetDeg,
                                double tauStaleS, double nominalIntervalS, double maxRateDegPerS) {
        if (semantics == null) {
            throw new IllegalArgumentException("heading semantics must be declared, not defaulted");
        }
        if (!Double.isFinite(mountOffsetDeg)) {
            throw new IllegalArgumentException(
                    "delta_mount_deg must be a declared finite calibration in degrees; got "
                    + mountOffsetDeg);
        }
        if (!semantics.requiresMountCalibration() && mountOffsetDeg != 0.0) {
            throw new IllegalArgumentException(
                    semantics.configName() + " already carries the navigation camera's heading, so "
                    + "no mounting correction may be applied; got delta_mount_deg=" + mountOffsetDeg);
        }
        if (!(tauStaleS > 0.0) || !Double.isFinite(tauStaleS)) {
            throw new IllegalArgumentException("tau_stale must be finite and positive; got " + tauStaleS);
        }
        if (!(nominalIntervalS > 0.0) || !Double.isFinite(nominalIntervalS)) {
            throw new IllegalArgumentException(
                    "nominal sample interval must be finite and positive; got " + nominalIntervalS);
        }
        if (!(maxRateDegPerS > 0.0) || !Double.isFinite(maxRateDegPerS)) {
            throw new IllegalArgumentException(
                    "max heading rate must be finite and positive; got " + maxRateDegPerS);
        }
        this.semantics = semantics;
        this.mountOffsetDeg = mountOffsetDeg;
        this.tauStaleS = tauStaleS;
        this.nominalIntervalS = nominalIntervalS;
        this.maxRateDegPerS = maxRateDegPerS;
    }

    /**
     * Accepts one heading sample, subject to the rate gate.
     *
     * @param availabilityTimeS when the value became usable, seconds, on the frame clock (sample
     *                          instant plus any transport latency — the caller models the latency)
     * @param headingDeg        the channel's own heading, degrees clockwise from North. Its
     *                          <b>meaning</b> is {@link #semantics()}: airframe for the FC arm,
     *                          navigation camera for the simulator arm
     * @return true when the sample was accepted; false when the rate gate rejected it, in which case
     *         the previous valid heading is retained unchanged
     * @throws IllegalArgumentException on a non-finite value, or on a sample older than the previous
     *                                  <i>submitted</i> one — an out-of-order stream is a real fault
     */
    public boolean submit(double availabilityTimeS, double headingDeg) {
        if (!Double.isFinite(availabilityTimeS) || !Double.isFinite(headingDeg)) {
            throw new IllegalArgumentException("heading sample must be finite; got t="
                    + availabilityTimeS + ", heading=" + headingDeg);
        }
        if (availabilityTimeS < lastSubmittedTimeS) {
            throw new IllegalArgumentException("heading samples must arrive in non-decreasing "
                    + "availability order; got " + availabilityTimeS + " after " + lastSubmittedTimeS);
        }
        lastSubmittedTimeS = availabilityTimeS;

        if (hasSample) {
            double dt = availabilityTimeS - lastSampleTimeS;
            if (dt > 0.0) {
                double rate = Math.abs(wrap180(headingDeg - lastHeadingDeg)) / dt;
                if (rate > maxRateDegPerS) {
                    rejectedSampleCount++;
                    return false;
                }
            }
        }
        hasSample = true;
        lastSampleTimeS = availabilityTimeS;
        lastHeadingDeg = headingDeg;
        sampleCount++;
        return true;
    }

    /**
     * The navigation heading to rotate a frame at {@code frameTimeS} with. Pure: calling it twice
     * returns the same answer, and it never mutates the channel.
     *
     * @throws IllegalArgumentException when {@code frameTimeS} is not finite
     */
    public HeadingReading readAt(double frameTimeS) {
        if (!Double.isFinite(frameTimeS)) {
            throw new IllegalArgumentException("frame time must be finite; got " + frameTimeS);
        }
        if (!hasSample || frameTimeS < lastSampleTimeS) {
            // No accepted sample has become available at or before this frame. There is no datum an
            // absolute heading could be assumed from, so none is produced. See the class comment.
            return new HeadingReading(frameTimeS, frameTimeS, 0.0, Double.NaN, mountOffsetDeg,
                    Double.NaN, semantics, HeadingStatus.UNAVAILABLE);
        }
        double ageS = frameTimeS - lastSampleTimeS;
        HeadingStatus status;
        if (ageS <= nominalIntervalS * STALE_INTERVAL_FACTOR) {
            status = HeadingStatus.FRESH;
        } else if (ageS <= tauStaleS) {
            status = HeadingStatus.HELD;
        } else {
            status = HeadingStatus.STALE;
        }
        double yawNavDeg = wrap360(lastHeadingDeg + mountOffsetDeg);
        return new HeadingReading(frameTimeS, lastSampleTimeS, ageS, lastHeadingDeg, mountOffsetDeg,
                yawNavDeg, semantics, status);
    }

    /**
     * Normalises an angle to {@code [0, 360)}.
     *
     * <p>The in-range case returns the argument <b>bit-identically</b>. The obvious
     * {@code ((deg % 360) + 360) % 360} does not: for 123.456789 the intermediate 483.456789 is not
     * representable, and the round trip returns 123.45678900000001. Almost every heading a sensor
     * publishes is already in range, and a normalisation that perturbs it would put a
     * one-ulp-per-frame wobble into the channel whose whole purpose is to be the trustworthy one.
     */
    public static double wrap360(double deg) {
        if (deg >= 0.0 && deg < 360.0) {
            return deg;
        }
        double d = ((deg % 360.0) + 360.0) % 360.0;
        return d >= 360.0 ? 0.0 : d;
    }

    /**
     * Wraps an angle difference into {@code (-180, 180]} — the only safe way to subtract two
     * headings, and the operation every consumer of this channel needs.
     */
    public static double wrap180(double deg) {
        double d = ((deg + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
        return d == -180.0 ? 180.0 : d;
    }

    /** Which arm this channel is; recorded in the run manifest so the two can never be confused. */
    public HeadingSemantics semantics() {
        return semantics;
    }

    /** The declared camera-to-body mounting yaw applied to every sample, degrees. */
    public double mountOffsetDeg() {
        return mountOffsetDeg;
    }

    /** The declared hold limit, seconds. */
    public double tauStaleS() {
        return tauStaleS;
    }

    /** The declared nominal sampling interval, seconds. */
    public double nominalIntervalS() {
        return nominalIntervalS;
    }

    /** The declared plausibility ceiling on implied yaw rate, degrees per second. */
    public double maxRateDegPerS() {
        return maxRateDegPerS;
    }

    /** How many samples were accepted. Diagnostic; used in no computation. */
    public long sampleCount() {
        return sampleCount;
    }

    /** How many samples the rate gate rejected. Diagnostic, and worth reporting per run. */
    public long rejectedSampleCount() {
        return rejectedSampleCount;
    }

    /** True once at least one sample has been accepted. */
    public boolean hasSample() {
        return hasSample;
    }
}
