package org.boofcv.stitching.metric;

/**
 * The declared parameters the metric readout needs, and the complete list of them
 * ({@code DEC-VO-007} <i>Swappability</i>: "a causal height source, the stale/degraded flag of D6,
 * and {@code f_working} from configuration — nothing else").
 *
 * <p>All five are <b>declared</b>, never estimated and never fitted. That is the property that makes
 * {@code DEC-VO-007} C1 a design with no free parameters: {@code h0} and {@code tauStale} are the
 * two the physics forces, and the three optical values are camera facts.
 *
 * @param fxNativePx       horizontal focal length in pixels at the camera's <b>native</b> resolution
 * @param downsampleFactor integer decimation applied to frames before the estimator sees them
 *                         (1 = none); the same value {@code VoRunnerConfig.downsampleFactor} feeds
 *                         to {@code DirectoryFrameSource}
 * @param h0AglM           camera-to-imaged-surface height at the navigation reference frame, metres
 *                         ({@code DEC-VO-007} D2) — external, never inferred from trajectory truth
 * @param tauStaleS        declared height-hold limit, seconds ({@code DEC-VO-007} D6)
 * @param nominalHeightIntervalS the height channel's nominal sampling interval, seconds
 */
public record MetricReadoutConfig(double fxNativePx, int downsampleFactor, double h0AglM,
                                  double tauStaleS, double nominalHeightIntervalS) {

    public MetricReadoutConfig {
        if (!(fxNativePx > 0.0) || !Double.isFinite(fxNativePx)) {
            throw new IllegalArgumentException(
                    "fx_native_px must be a finite positive focal length in NATIVE-resolution "
                    + "pixels; got " + fxNativePx);
        }
        if (downsampleFactor < 1) {
            throw new IllegalArgumentException("downsampleFactor must be >= 1; got " + downsampleFactor);
        }
    }

    /**
     * {@code f_working = fx_native / downsampleFactor} — {@code LIT-VO-003} eq. (9).
     *
     * <p><b>Forgetting the divisor halves every metric distance.</b> At the reference configuration's
     * {@code downsampleFactor: 2} on 2448x2048 imagery, {@code f_working} is 735.53
     * ({@code HKairport01}); using {@code fx_native} unreduced doubles the ground sampling distance
     * and therefore doubles every reported displacement. {@code LIT-VO-003} section 10.2 names the
     * trap and {@code MetricNavigationStateTest} turns it into a failing condition.
     */
    public double fWorkingPx() {
        return fxNativePx / (double) downsampleFactor;
    }

    /** A channel built from these parameters. One per run; it holds the ZOH state. */
    public CausalHeightChannel newHeightChannel() {
        return new CausalHeightChannel(h0AglM, tauStaleS, nominalHeightIntervalS);
    }
}
