package org.boofcv.stitching.metric;

/**
 * The height a single video frame was (or would be) converted with — the complete causal answer
 * {@link CausalHeightChannel} gives for one frame time.
 *
 * <p>Immutable and self-describing on purpose: every field a consumer or a log needs to reconstruct
 * <i>why</i> a particular ground sampling distance was used is carried here, so nothing downstream
 * has to re-derive it and get a different answer.
 *
 * <pre>
 *   heightAglM = h0M + relativeHeightM          (LIT-VO-006 eq. 4, flat-terrain assumption)
 * </pre>
 *
 * <p><b>The flat-terrain assumption is in that line and is not observable from anything here.</b>
 * The true relation carries a {@code −e(x_c(t))} terrain term that no input to this system can see
 * ({@code DEC-VO-007} <i>Rationale</i>; measured at 30.1 % of path on a rendered ramp,
 * {@code COMP-001} §7). A reading is therefore a statement about height above the <i>takeoff
 * datum</i> re-labelled as height above the imaged surface, and it is only correct where those
 * coincide.
 *
 * @param frameTimeS       the video frame time this reading answers for, seconds, on the frame clock
 * @param sampleTimeS      availability time of the relative-height sample actually used, seconds
 * @param ageS             {@code frameTimeS − sampleTimeS}; 0 before the first sample
 * @param relativeHeightM  takeoff-relative height as reported by the channel, metres, zero at the datum
 * @param h0M              the externally supplied camera-to-imaged-surface height at the reference
 *                         frame ({@code DEC-VO-007} D2) — never estimated here
 * @param heightAglM       {@code h0M + relativeHeightM}, metres
 * @param status           freshness/validity of the sample behind it
 */
public record HeightReading(double frameTimeS, double sampleTimeS, double ageS,
                            double relativeHeightM, double h0M, double heightAglM,
                            HeightStatus status) {

    /** True when a metric increment may be produced from this reading. */
    public boolean usable() {
        return status.usable();
    }

    /** True when the metric pose derived from this reading must be reported degraded. */
    public boolean degraded() {
        return status.degraded();
    }

    /**
     * Ground sampling distance in metres per pixel: {@code heightAglM / fWorkingPx}
     * ({@code DEC-VO-007} D3).
     *
     * <p>{@code fWorkingPx} is the <b>working-resolution</b> focal length,
     * {@code fx_native / downsampleFactor} — see {@link MetricReadoutConfig#fWorkingPx()}. Passing
     * {@code fx_native} unreduced halves every metric distance at {@code downsampleFactor: 2}, which
     * is the trap {@code LIT-VO-003} §10.2 names and {@code test_g3_forgetting_the_downsample_factor}
     * turns into a failing condition.
     *
     * @throws IllegalStateException when this reading is not usable — an unusable reading has no
     *                               ground sampling distance, and returning one would be exactly the
     *                               silent unknown scale {@code DEC-VO-007} D6 forbids
     */
    public double groundSamplingDistance(double fWorkingPx) {
        if (!usable()) {
            throw new IllegalStateException(
                    "no ground sampling distance for an " + status + " height reading at t="
                    + frameTimeS + " s");
        }
        return heightAglM / fWorkingPx;
    }
}
