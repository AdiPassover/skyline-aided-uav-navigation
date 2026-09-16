package org.boofcv.stitching.metric;

/**
 * One causally sampled navigation heading, with everything needed to audit how it was produced.
 *
 * <pre>
 *   yawNavDeg = wrap360(sourceHeadingDeg + mountOffsetDeg)
 * </pre>
 *
 * <p>Both terms are kept rather than only the result, because the two arms differ precisely in the
 * second one: {@link HeadingSemantics#FC_AHRS_BODY_COMPASS} carries a declared platform calibration
 * there, and {@link HeadingSemantics#SIM_NADIR_CAMERA_HEADING} carries exactly zero. A log that
 * recorded only {@code yawNavDeg} could not tell the two apart after the fact.
 *
 * @param frameTimeS        the frame this reading was taken for, seconds
 * @param sampleTimeS       when the underlying sample became available, seconds ({@code frameTimeS}
 *                          when no sample exists)
 * @param ageS              {@code frameTimeS - sampleTimeS}, seconds
 * @param sourceHeadingDeg  the raw channel value, degrees clockwise from North; {@code NaN} when
 *                          {@link HeadingStatus#UNAVAILABLE}
 * @param mountOffsetDeg    the declared camera-to-body mounting yaw applied, degrees; exactly
 *                          {@code 0} for an arm whose source is already the camera's heading
 * @param yawNavDeg         the navigation heading, degrees clockwise from North in {@code [0, 360)};
 *                          {@code NaN} when {@link HeadingStatus#UNAVAILABLE}
 * @param semantics         which arm produced this
 * @param status            freshness
 */
public record HeadingReading(double frameTimeS, double sampleTimeS, double ageS,
                             double sourceHeadingDeg, double mountOffsetDeg, double yawNavDeg,
                             HeadingSemantics semantics, HeadingStatus status) {

    /** True when a navigation heading may be produced from this reading. */
    public boolean usable() {
        return status.usable();
    }

    /** True when the navigation pose derived from this reading must be reported degraded. */
    public boolean degraded() {
        return status.degraded();
    }

    /**
     * The navigation heading in radians, for the rotation that carries a camera-frame displacement
     * into ENU.
     *
     * @throws IllegalStateException when this reading is not usable — returning a number here would
     *                               be exactly the fabricated absolute heading the contract forbids
     */
    public double yawNavRad() {
        if (!usable()) {
            throw new IllegalStateException(
                    "no navigation heading for an " + status + " reading at t=" + frameTimeS + " s");
        }
        return Math.toRadians(yawNavDeg);
    }
}
