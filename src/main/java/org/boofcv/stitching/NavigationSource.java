package org.boofcv.stitching;

/**
 * Which computation produces the published pose ({@code DEC-VO-003}, {@code DEC-VO-008}).
 *
 * <p>Three readouts are implemented and all three remain selectable. They are <b>not</b> numerically
 * equivalent, and the differences are large on real flight data — so the choice is explicit, and
 * {@link #DEFAULT} names the one a caller gets when it does not choose.
 *
 * <p><b>They are not three architectures.</b> {@code EXP-VO-008} recomposed all three offline from a
 * single recorded transform sequence and found them to be the same computation at three
 * <i>granularities</i> of rigid reduction — {@code MOSAIC_LEGACY} at the canvas re-origin interval
 * (~146 frames), {@code RIGID_MOTION} at 1 frame, {@code LOGICAL_FRAME} at infinity — with
 * normalised ATE monotone increasing in the interval on both real sequences. The substantive choice
 * is where on that axis to sit, and per-frame reduction is the measured optimum.
 */
public enum NavigationSource {

    /**
     * The pre-{@code DEC-VO-003} computation: pose read from the current frame's corners projected
     * into the mosaic canvas, re-anchored across canvas re-origins by an SE(2) accumulation of
     * centre displacement and one edge angle.
     *
     * <p><b>Retained for reproduction and diagnostics, not endorsed</b> ({@code DEC-VO-008}). It is
     * architecturally coupled to the render buffer — that is the documented defect, measured at a
     * <b>280 px</b> pose shift under a canvas intervention that moves {@code RIGID_MOTION} by
     * ≤ 8.8 × 10⁻¹³ px — and it discards scale, shear and perspective at every re-origin. Selecting
     * it explicitly reproduces {@code EXP-002}, {@code EXP-003} and {@code EXP-VO-002} exactly;
     * {@code runs/hkairport01-a-run-v1} reproduces bit-identically over 1,800 rows.
     *
     * <p>One measured caution against reading its one numerical win as quality: on {@code AMtown01}
     * under the affine model it beats {@code RIGID_MOTION}, and {@code EXP-VO-008} attributes that
     * to an <b>accidental cancellation</b> — the legacy top-edge angle is
     * {@code polar + atan2(q, p)}, the fitted affine carries a systematically positive shear, and
     * that offset happens to cancel part of a negative heading deficit. A camera mounted a quarter
     * turn round would make the identical code 1.7× <i>worse</i> than polar.
     */
    MOSAIC_LEGACY,

    /**
     * The logical navigation frame: pose from {@link LogicalNavigationState}, composed from the
     * motion estimator alone against a fixed origin, with reference resets folded exactly.
     *
     * <p><b>Rejected as a navigation representation</b>, and kept as the diagnostic that shows why:
     * it composes the estimator's transform <i>in full</i>, so it faithfully propagates the
     * non-rigid terms the estimator gets wrong. Worse than both other readouts on every window and
     * every motion model measured (2.55–21.70 % normalised ATE), with the homography's Sim(2)
     * alignment collapsing to a scale of 4.9 × 10⁻⁶ on {@code AMtown01} ({@code EXP-VO-006},
     * {@code EXP-VO-007}, {@code EXP-VO-008}).
     */
    LOGICAL_FRAME,

    /**
     * <b>The default</b> ({@link #DEFAULT}, {@code DEC-VO-008}; {@code DEC-VO-004} Alternative E).
     * Navigation from the <b>rigid part of each frame's image motion only</b> — in-plane rotation
     * and the translation of the image centre — with uniform scale, anisotropy and perspective
     * observed as diagnostics and never integrated into {@code x}/{@code y}/{@code yaw}. See
     * {@link RigidNavigationState}.
     *
     * <p>Promoted on architecture and on the completed evidence base, <b>not</b> on a claim that it
     * wins every numerical arm — it does not, and {@code MOSAIC_LEGACY}'s javadoc records the one
     * arm it loses and why that win is not a reason to prefer legacy. What it is: 0.99–2.43 %
     * normalised ATE across two real-flight sequences and three motion models, the only readout
     * independent of the render buffer, and the measured optimum of a monotone granularity sweep on
     * both flights.
     *
     * <p><b>Non-metric by construction.</b> {@code x} and {@code y} are in first-frame pixels and
     * {@code z} is never estimated. Converting to metres needs an external height and is the
     * separate, opt-in {@link #METRIC_LOCAL} layer ({@code DEC-VO-007}) that this selector neither
     * performs nor requires.
     */
    RIGID_MOTION,

    /**
     * <b>The metric local navigation pose</b> ({@code DEC-VO-007} D3/D9): {@link #RIGID_MOTION}'s
     * per-frame increments, each multiplied by the ground sampling distance of a causally sampled
     * external height, integrated online. <b>{@code x} and {@code y} are METRES</b> (east, north);
     * {@code yaw} is unchanged and {@code z} is still never estimated.
     *
     * <p><b>It is not a drop-in unit change and must never be treated as one.</b> Selecting it moves
     * the published pose from first-frame pixels to metres, and every threshold, gain and target a
     * consumer holds is in the units of whatever it was tuned against — see
     * {@code NavigationUnits} and {@code MissionStrategy}'s unit declaration. This is why it is a
     * distinct enum value rather than a flag on {@code RIGID_MOTION}: a caller must ask for metres
     * by name.
     *
     * <p><b>It cannot be selected without the configuration that makes it true.</b>
     * {@code MotionModelStitchingEstimator.setNavigationSource} rejects it unless the metric readout
     * has been enabled with an external {@code h0}, {@code f_working} and a height channel. There is
     * no path by which a pixel-valued pose is relabelled as metres.
     *
     * <p><b>What it is worth, and what it is not.</b> The conversion is exact to 0.030-0.083 % of
     * path against known height ({@code LIT-VO-003} section 10.4), and on UE-rendered imagery under
     * a x3.15 altitude change a takeoff-relative altitude removes 99.20 % of the 27.32 pp of metric
     * scale error a fixed first-frame height incurs, with zero fitted parameters
     * ({@code EXP-VO-014}). It assumes approximately constant terrain, it inherits
     * {@code RIGID_MOTION}'s unconstrained heading in full ({@code DEC-VO-007} D7), and it is local:
     * not latitude/longitude, not a global origin, not north-referenced, not terrain-aware AGL.
     */
    METRIC_LOCAL;

    /**
     * The readout a caller gets when it does not choose one — the single source of truth behind
     * {@code MotionModelStitchingEstimator}'s field initialiser and {@code VoRunnerConfig}'s
     * {@code navigation_source} default, so the two cannot drift apart.
     *
     * <p>Changed from {@link #MOSAIC_LEGACY} to {@link #RIGID_MOTION} on 2026-08-28 by
     * {@code DEC-VO-008}. Configurations that must reproduce a pre-2026-08-28 baseline name
     * {@code MOSAIC_LEGACY} explicitly; the five committed configs that relied on the old implicit
     * default were made explicit in the same change.
     */
    public static final NavigationSource DEFAULT = RIGID_MOTION;

    /**
     * The length unit of the {@code x}/{@code y} this source publishes.
     *
     * <p>Three of the four readouts are the same computation at different granularities of rigid
     * reduction and all three are in first-frame pixels; only {@link #METRIC_LOCAL} multiplies by a
     * ground sampling distance. Consumers that hold a distance constant compare against this rather
     * than assuming ({@link NavigationUnits}).
     */
    public NavigationUnits units() {
        return this == METRIC_LOCAL ? NavigationUnits.METRES : NavigationUnits.IMAGE_PIXELS;
    }
}
