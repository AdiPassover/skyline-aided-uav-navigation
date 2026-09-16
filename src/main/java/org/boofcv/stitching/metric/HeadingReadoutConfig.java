package org.boofcv.stitching.metric;

/**
 * The declared parameters of the authoritative heading channel — the heading counterpart of
 * {@link MetricReadoutConfig}.
 *
 * <p>Every field is <b>declared, never fitted</b>. In particular {@code deltaMountDeg} is a platform
 * calibration: it must come from mechanical design, a one-time calibration, or a vendor file, and a
 * configuration that omits it on an arm that needs it is <b>refused rather than defaulted to
 * zero</b> — zero looks entirely plausible and would silently reintroduce {@code EXP-VO-013} R6's
 * ~66° error.
 *
 * @param semantics           which arm; see {@link HeadingSemantics}
 * @param deltaMountDeg       declared camera-to-body mounting yaw, degrees, or {@code null} to mean
 *                            "not supplied" (accepted only when the arm does not need one)
 * @param tauStaleS           declared hold limit, seconds
 * @param nominalIntervalS    the channel's nominal sampling interval, seconds
 * @param maxRateDegPerS      plausibility ceiling on the implied yaw rate, degrees per second
 */
public record HeadingReadoutConfig(HeadingSemantics semantics, Double deltaMountDeg,
                                   double tauStaleS, double nominalIntervalS,
                                   double maxRateDegPerS) {

    /**
     * A ceiling comfortably above any airframe yaw rate a survey UAV produces and comfortably below
     * the glitches measured on the committed real channel (274–5141 °/s against a legitimate maximum
     * of 26.7 °/s). Chosen as a round number an order of magnitude clear of both, not tuned.
     */
    public static final double DEFAULT_MAX_RATE_DEG_PER_S = 180.0;

    public HeadingReadoutConfig {
        if (semantics == null) {
            throw new IllegalArgumentException(
                    "heading_readout.heading_semantics must be declared: \"fc_ahrs_body_compass\" "
                    + "or \"sim_nadir_camera_heading\"");
        }
        if (semantics.requiresMountCalibration() && deltaMountDeg == null) {
            throw new IllegalArgumentException(
                    "heading_readout.delta_mount_deg is REQUIRED for " + semantics.configName()
                    + ": that column is the AIRFRAME's heading, and the navigation camera's differs "
                    + "from it by a fixed mounting yaw this repository does not know for your "
                    + "platform. Supply the calibration (mechanical design, a one-time platform "
                    + "calibration, or a vendor file) -- it must not be fitted at runtime and must "
                    + "not be inferred from trajectory ground truth (EXP-VO-013 R6)");
        }
        if (!semantics.requiresMountCalibration() && deltaMountDeg != null && deltaMountDeg != 0.0) {
            throw new IllegalArgumentException(
                    semantics.configName() + " already carries the navigation camera's heading, so "
                    + "no mounting correction may be applied; got delta_mount_deg=" + deltaMountDeg);
        }
    }

    /** The mounting yaw actually applied, degrees; exactly {@code 0} when the arm needs none. */
    public double mountOffsetDeg() {
        return deltaMountDeg == null ? 0.0 : deltaMountDeg;
    }

    /** Builds the channel these parameters describe. */
    public CausalHeadingChannel newHeadingChannel() {
        return new CausalHeadingChannel(semantics, mountOffsetDeg(), tauStaleS, nominalIntervalS,
                maxRateDegPerS);
    }
}
