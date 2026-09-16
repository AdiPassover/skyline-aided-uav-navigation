package org.boofcv.stitching.metric;

import lombok.Getter;
import org.boofcv.util.structs.Pose3D;

/**
 * The <b>metric local navigation state</b>: {@code RIGID_MOTION}'s image-space increments converted
 * to metres by a causally sampled external height, and rotated into the navigation frame by a
 * causally sampled external heading.
 *
 * <p>Three channels, three responsibilities, and no overlap between them:
 *
 * <pre>
 *   VO      owns the visual image displacement   dq_k          (pixels, frame k-1's axes)
 *   height  owns the scale                       gsd = h/f     (metres per pixel)
 *   heading owns the direction                   yaw_nav        (degrees CW from North)
 * </pre>
 *
 * <p>Per accepted frame {@code k}:
 *
 * <pre>
 *   h_{k-1}   = h0 + h_baro(t_{k-1})                     the REFERENCE frame's height
 *   gsd_{k-1} = h_{k-1} / f_working                      f_working = fx_native / downsampleFactor
 *   psi_{k-1} = yaw_nav(t_{k-1})                         the REFERENCE frame's navigation heading
 *   east     += ( dq_x*cos(psi) - dq_y*sin(psi) ) * gsd
 *   north    -= ( dq_x*sin(psi) + dq_y*cos(psi) ) * gsd
 * </pre>
 *
 * <p><b>That is the same arithmetic {@code DEC-VO-007} D3 always specified, with one symbol
 * changed.</b> The pre-existing update accumulated {@code T} in image axes and published
 * {@code east = +T_x, north = -T_y}; expanding it shows that composition is already the
 * clockwise-from-North form for a heading measured on the image-up axis, so replacing the visually
 * integrated {@code theta} with an external {@code psi} is a substitution and not a change of
 * handedness. {@code MetricNavigationStateTest} asserts exactly that equivalence rather than
 * assuming it.
 *
 * <h2>Visual yaw is retained, and demoted</h2>
 *
 * <p>{@link #headingRadUnwrapped()} still integrates {@code dtheta} from the rigid state, frame by
 * frame, bit-identically. It is a <b>diagnostic</b>: it no longer determines translation direction
 * when a heading channel is configured, and {@link #getLastYawDisagreementDeg()} exposes the
 * wrap-safe difference between the visual and external rotation increments as a free per-frame
 * consistency signal. It is a raw angle, deliberately — not a confidence, not a weight, and not an
 * input to anything.
 *
 * <p><b>Fixed here (defect found 2026-09-06):</b> that integration used to sit <i>inside</i> the
 * branch where the reference height was usable, so an {@link HeightStatus#UNAVAILABLE} height
 * silently froze the metric visual yaw while the raw rigid yaw kept advancing — a divergence three
 * documented contracts denied. Height availability now decides whether a <i>translation</i> can be
 * produced and nothing else.
 *
 * <h2>Segments, and the gap</h2>
 *
 * <p>A stitching failure restarts the estimator from the failing frame; the motion across the failed
 * interval is never estimated ({@code COMP-001} §8). This class contributes exactly zero metres
 * across it and makes it legible: {@link #getSegmentIndex()} advances and
 * {@link #isUnknownTranslationGapBeforeSegment()} is set.
 *
 * <p><b>The flag says translation, and that is a narrowing, not a rewording.</b> Before an external
 * heading channel existed, rotation across the gap was unobserved too — measured at −0.046°, +0.160°
 * and −0.426° at the three real hard-loss events in the corpus, against the zero the estimator
 * inserts. With a valid heading channel that rotation is measured by a sensor that never saw the
 * visual failure, so after a restart the yaw is <i>known</i> while the displacement is not.
 * {@link #isHeadingKnownAcrossGap()} says whether that held for this segment; when it did not, no
 * heading is fabricated.
 *
 * <p>Not thread-safe; it lives on the frame-processing thread with the estimator.
 */
public final class MetricNavigationState {

    private final double fWorkingPx;

    /** True when an external heading channel owns the navigation direction. */
    @Getter private final boolean headingAuthoritative;

    // --- integrated metric state, image convention internally (y down), metres ---
    private double txM;
    private double tyM;

    /** The visually integrated yaw. DIAGNOSTIC once a heading channel is configured. */
    private double visualHeadingRad;

    // --- segment bookkeeping ---
    @Getter private int segmentIndex;
    /**
     * True when the DISPLACEMENT across the boundary that opened this segment was never estimated.
     * Says nothing about rotation — see {@link #isHeadingKnownAcrossGap()}.
     */
    @Getter private boolean unknownTranslationGapBeforeSegment;
    /**
     * True when an authoritative navigation heading was available at this segment's first frame, so
     * the segment's orientation is anchored by a sensor rather than assumed. Always false when no
     * heading channel is configured.
     */
    @Getter private boolean headingKnownAcrossGap;
    private double segmentTxM;
    private double segmentTyM;
    private double segmentVisualHeadingRad;
    private double segmentNavHeadingDeg = Double.NaN;
    private boolean awaitingSegmentDatum = true;

    // --- causal height bookkeeping ---
    /** The height reading latched at frame {@code k-1}; the one the next increment is scaled by. */
    @Getter private HeightReading referenceHeight;
    /** The height reading actually used for the most recent increment. Null before the first. */
    @Getter private HeightReading lastUsedHeight;
    @Getter private double lastGsdMPerPx;

    // --- causal heading bookkeeping ---
    /** The heading reading latched at frame {@code k-1}; the one the next increment is rotated by. */
    @Getter private HeadingReading referenceHeading;
    /** The heading reading actually used for the most recent increment. Null before the first. */
    @Getter private HeadingReading lastUsedHeading;
    /** The most recent reading taken at this frame's own time — what {@link #navHeadingDeg()} reports. */
    private HeadingReading currentHeading;
    private double navHeadingDeg = Double.NaN;

    // --- diagnostics ---
    /** Frames whose increment was produced from a degraded (STALE) height or heading. */
    @Getter private long degradedFrameCount;
    /** Frames for which no metric increment could be produced at all. */
    @Getter private long unusableFrameCount;
    @Getter private long integratedFrameCount;
    /** Frames on which no authoritative heading was available to rotate the increment. */
    @Getter private long headingUnavailableFrameCount;
    /** Wrap-safe {@code visual - external} rotation increment for the last frame, degrees. */
    @Getter private double lastYawDisagreementDeg = Double.NaN;
    private boolean lastFrameUsable = true;

    private boolean observed;
    private Pose3D pose = new Pose3D();

    /**
     * A state whose direction comes from the visually integrated yaw — the {@code DEC-VO-009}
     * behaviour, retained so a run without a heading channel is unchanged.
     *
     * @param fWorkingPx working-resolution focal length in pixels, i.e.
     *                   {@link MetricReadoutConfig#fWorkingPx()}. Strictly positive and finite.
     */
    public MetricNavigationState(double fWorkingPx) {
        this(fWorkingPx, false);
    }

    /**
     * @param headingAuthoritative when true, every increment is rotated by the externally supplied
     *                             navigation heading and the visual yaw becomes diagnostic only
     */
    public MetricNavigationState(double fWorkingPx, boolean headingAuthoritative) {
        if (!(fWorkingPx > 0.0) || !Double.isFinite(fWorkingPx)) {
            throw new IllegalArgumentException(
                    "f_working must be a finite positive focal length in WORKING-resolution pixels "
                    + "(fx_native / downsampleFactor); got " + fWorkingPx);
        }
        this.fWorkingPx = fWorkingPx;
        this.headingAuthoritative = headingAuthoritative;
        reset();
    }

    /** Returns to the state of a fresh run: metric origin, segment 0, no gap, no latched readings. */
    public void reset() {
        txM = 0.0;
        tyM = 0.0;
        visualHeadingRad = 0.0;
        segmentIndex = 0;
        unknownTranslationGapBeforeSegment = false;
        headingKnownAcrossGap = false;
        segmentTxM = 0.0;
        segmentTyM = 0.0;
        segmentVisualHeadingRad = 0.0;
        segmentNavHeadingDeg = Double.NaN;
        awaitingSegmentDatum = true;
        referenceHeight = null;
        lastUsedHeight = null;
        lastGsdMPerPx = Double.NaN;
        referenceHeading = null;
        lastUsedHeading = null;
        currentHeading = null;
        navHeadingDeg = Double.NaN;
        degradedFrameCount = 0;
        unusableFrameCount = 0;
        integratedFrameCount = 0;
        headingUnavailableFrameCount = 0;
        lastYawDisagreementDeg = Double.NaN;
        lastFrameUsable = true;
        observed = false;
        pose = buildPose();
    }

    /** Reference frame with no heading channel configured. */
    public void observeOrigin(HeightReading heightAtThisFrame) {
        observeOrigin(heightAtThisFrame, null);
    }

    /**
     * Records the reference frame: the metric origin, and the first latch of both channels.
     * Integrates no motion.
     */
    public void observeOrigin(HeightReading heightAtThisFrame, HeadingReading headingAtThisFrame) {
        requireHeight(heightAtThisFrame);
        requireHeadingArm(headingAtThisFrame);
        referenceHeight = heightAtThisFrame;
        referenceHeading = headingAtThisFrame;
        adoptCurrentHeading(headingAtThisFrame);
        // A reference frame produces no increment, so there is no visual-vs-external comparison to
        // report for it. Carrying the previous frame's forward would put a stale number on exactly
        // the row a reader inspects most closely -- the restart.
        lastYawDisagreementDeg = Double.NaN;
        if (awaitingSegmentDatum) {
            // The segment's orientation datum is the heading at its FIRST frame, latched once.
            segmentNavHeadingDeg = navHeadingDeg;
            segmentVisualHeadingRad = visualHeadingRad;
            headingKnownAcrossGap = headingAtThisFrame != null && headingAtThisFrame.usable();
            awaitingSegmentDatum = false;
        }
        // Counted on USE, never on latch: the reference frame produces no increment.
        lastFrameUsable = heightAtThisFrame.usable();
        observed = true;
        pose = buildPose();
    }

    /** One accepted frame with no heading channel configured. */
    public void observe(double dqX, double dqY, double dThetaRad, HeightReading heightAtThisFrame) {
        observe(dqX, dqY, dThetaRad, heightAtThisFrame, null);
    }

    /**
     * Integrates one <b>accepted</b> frame's rigid increment, scaled by the <i>previously latched</i>
     * height and rotated by the <i>previously latched</i> navigation heading — both belonging to
     * frame {@code k-1}, because {@code dq_k} is expressed in frame {@code k-1}'s axes. The
     * {@code k-1} convention is structural: this method converts with the readings latched by the
     * previous call and only then latches the new ones, so there is no ordering a caller can get
     * wrong.
     *
     * @param dqX                {@code (q_k - c).x} in reference-frame pixels, image right
     * @param dqY                {@code (q_k - c).y} in reference-frame pixels, image <b>down</b>
     * @param dThetaRad          the polar rotation of the same inverse transform, radians
     * @param heightAtThisFrame  the causal height reading at frame {@code k}'s own time
     * @param headingAtThisFrame the causal heading reading at frame {@code k}'s own time; must be
     *                           non-null exactly when a heading channel is configured
     */
    public void observe(double dqX, double dqY, double dThetaRad,
                        HeightReading heightAtThisFrame, HeadingReading headingAtThisFrame) {
        requireHeight(heightAtThisFrame);
        requireHeadingArm(headingAtThisFrame);
        if (referenceHeight == null) {
            // No reference latched yet: this frame IS the origin. Cannot happen through the
            // estimator, which always calls observeOrigin first; guarded so a direct caller cannot
            // silently convert an increment with no reference.
            observeOrigin(heightAtThisFrame, headingAtThisFrame);
            return;
        }

        HeightReading refHeight = referenceHeight;
        HeadingReading refHeading = referenceHeading;

        // ---- the direction this increment is rotated by, taken at frame k-1 ----
        boolean haveDirection;
        double rotRad;
        if (headingAuthoritative) {
            haveDirection = refHeading != null && refHeading.usable();
            rotRad = haveDirection ? refHeading.yawNavRad() : Double.NaN;
        } else {
            haveDirection = true;
            rotRad = visualHeadingRad;             // theta_{k-1}, before the increment below
        }

        // ---- the visual yaw is a DIAGNOSTIC and advances unconditionally ----
        // It must not depend on the height channel: that coupling was the 2026-09-06 defect.
        visualHeadingRad += dThetaRad;

        // ---- free per-frame consistency signal, wrap-safe; a raw angle, never a confidence ----
        if (headingAuthoritative && refHeading != null && refHeading.usable()
                && headingAtThisFrame.usable()) {
            double externalIncrementDeg = CausalHeadingChannel.wrap180(
                    headingAtThisFrame.yawNavDeg() - refHeading.yawNavDeg());
            lastYawDisagreementDeg = CausalHeadingChannel.wrap180(
                    Math.toDegrees(dThetaRad) - externalIncrementDeg);
        } else {
            lastYawDisagreementDeg = Double.NaN;
        }

        boolean haveScale = refHeight.usable();
        if (haveScale && haveDirection) {
            double gsd = refHeight.groundSamplingDistance(fWorkingPx);
            double px = dqX * gsd;
            double py = dqY * gsd;
            double cos = Math.cos(rotRad);
            double sin = Math.sin(rotRad);
            txM += cos * px - sin * py;
            tyM += sin * px + cos * py;

            lastGsdMPerPx = gsd;
            lastUsedHeight = refHeight;
            lastUsedHeading = refHeading;
            integratedFrameCount++;
            lastFrameUsable = true;
            if (refHeight.degraded() || (refHeading != null && refHeading.degraded())) {
                degradedFrameCount++;
            }
        } else {
            // No usable scale, or no usable direction. The pose holds; nothing metric is invented.
            lastFrameUsable = false;
            unusableFrameCount++;
            if (!haveDirection) {
                headingUnavailableFrameCount++;
            }
        }

        referenceHeight = heightAtThisFrame;
        referenceHeading = headingAtThisFrame;
        adoptCurrentHeading(headingAtThisFrame);
        observed = true;
        pose = buildPose();
    }

    /**
     * Starts a new local metric segment, because the estimator restarted and the <b>displacement</b>
     * across the failed interval was never estimated.
     *
     * <p>Nothing about the integrated track changes: no displacement is added, none is removed, and
     * the continuous pose keeps accumulating so a consumer tracking one trajectory is not teleported
     * mid-flight. What changes is that the gap becomes legible. The heading datum for the new segment
     * is latched at its first frame by the {@code observeOrigin} that follows.
     *
     * @param translationAcrossGapUnknown true for a stitching restart; false for a boundary that
     *                                    loses nothing, such as a mosaic canvas re-origin
     */
    public void beginNewSegment(boolean translationAcrossGapUnknown) {
        segmentIndex++;
        unknownTranslationGapBeforeSegment = translationAcrossGapUnknown;
        segmentTxM = txM;
        segmentTyM = tyM;
        // Both latches are dropped: after a restart the estimator's accumulation is identity and the
        // next increment is measured from the restart frame, whose own readings are latched when
        // that frame is observed.
        referenceHeight = null;
        referenceHeading = null;
        awaitingSegmentDatum = true;
    }

    /**
     * The metric local navigation pose. <b>{@code x} is east in METRES, {@code y} is north in
     * METRES</b>, {@code z} is always {@code 0}, and {@code yaw} is degrees in {@code [0, 360)}.
     *
     * <p>{@code yaw} is the <b>authoritative navigation heading</b> — clockwise from <b>North</b> —
     * when a heading channel is configured, and {@code NaN} when none has ever been available.
     * Without a heading channel it is the visually integrated yaw on the start-frame datum, exactly
     * as before. {@link #isHeadingAuthoritative()} distinguishes the two, and the run manifest
     * records which arm ran.
     *
     * <p><b>{@code z} is not the AGL, deliberately</b> ({@code DEC-VO-007} D9): height reaches this
     * class as an input to the horizontal conversion and acquires no vertical authority. Read
     * {@link #heightAglM()} instead.
     */
    public Pose3D metricPose() {
        return pose;
    }

    /**
     * The metric pose relative to the current segment's origin — the pose with no unestimated
     * interval inside it. Metres, same axes as {@link #metricPose()}; {@code yaw} is
     * {@link #segmentRelativeHeadingDeg()}.
     */
    public Pose3D segmentRelativePose() {
        return new Pose3D(txM - segmentTxM, -(tyM - segmentTyM), 0.0, segmentRelativeHeadingDeg());
    }

    /**
     * Rotational displacement since this segment began, degrees in {@code [0, 360)}.
     *
     * <p>With an authoritative heading this is {@code wrap360(yaw_nav(t) - yaw_nav(t_restart))} — a
     * difference of two sensor measurements, so it contains no unestimated interval even though the
     * visual track does. {@code NaN} when either end is unknown. Without a heading channel it is the
     * visually integrated difference, as before.
     */
    public double segmentRelativeHeadingDeg() {
        if (headingAuthoritative) {
            if (Double.isNaN(navHeadingDeg) || Double.isNaN(segmentNavHeadingDeg)) {
                return Double.NaN;
            }
            return CausalHeadingChannel.wrap360(navHeadingDeg - segmentNavHeadingDeg);
        }
        return CausalHeadingChannel.wrap360(
                Math.toDegrees(visualHeadingRad - segmentVisualHeadingRad));
    }

    /** The authoritative navigation heading at the current frame, degrees CW from North, or NaN. */
    public double navHeadingDeg() {
        return navHeadingDeg;
    }

    /** The authoritative heading this segment started at, degrees CW from North, or NaN. */
    public double segmentStartNavHeadingDeg() {
        return segmentNavHeadingDeg;
    }

    /** Freshness of the heading reading at the current frame, or {@link HeadingStatus#UNAVAILABLE}. */
    public HeadingStatus currentHeadingStatus() {
        return currentHeading == null ? HeadingStatus.UNAVAILABLE : currentHeading.status();
    }

    /** Freshness of the heading reading the last increment was rotated with. */
    public HeadingStatus lastHeadingStatus() {
        return lastUsedHeading == null ? HeadingStatus.UNAVAILABLE : lastUsedHeading.status();
    }

    /** Metric east displacement from the run origin, metres. */
    public double eastM() {
        return txM;
    }

    /** Metric north displacement from the run origin, metres. */
    public double northM() {
        return -tyM;
    }

    /**
     * The <b>visually integrated</b> heading in radians, unwrapped. Diagnostic once a heading channel
     * is configured; identical by construction to the raw rigid state's, on every frame, whatever
     * the height channel is doing.
     */
    public double headingRadUnwrapped() {
        return visualHeadingRad;
    }

    /** The AGL of the height reading most recently <i>used</i> for a conversion, metres, or NaN. */
    public double heightAglM() {
        return lastUsedHeight == null ? Double.NaN : lastUsedHeight.heightAglM();
    }

    /** Freshness of the height reading the last increment was scaled by. */
    public HeightStatus lastHeightStatus() {
        return lastUsedHeight == null ? HeightStatus.UNAVAILABLE : lastUsedHeight.status();
    }

    /** False when the most recent frame produced no metric increment, for want of scale or direction. */
    public boolean lastFrameUsable() {
        return lastFrameUsable;
    }

    /** False until the reference frame has been observed. */
    public boolean hasObservation() {
        return observed;
    }

    /** The working-resolution focal length this state converts with, pixels. */
    public double fWorkingPx() {
        return fWorkingPx;
    }

    private void adoptCurrentHeading(HeadingReading reading) {
        currentHeading = reading;
        if (reading != null && reading.usable()) {
            navHeadingDeg = reading.yawNavDeg();
        }
        // An UNAVAILABLE reading leaves navHeadingDeg alone: once an absolute heading has been
        // measured it is held under the declared staleness policy, and if none ever has it stays NaN.
    }

    private void requireHeight(HeightReading reading) {
        if (reading == null) {
            throw new IllegalArgumentException("a height reading is required for every frame");
        }
    }

    private void requireHeadingArm(HeadingReading reading) {
        if (headingAuthoritative && reading == null) {
            throw new IllegalArgumentException(
                    "a heading reading is required for every frame when the heading channel is "
                    + "authoritative");
        }
        if (!headingAuthoritative && reading != null) {
            throw new IllegalArgumentException(
                    "this state integrates visual yaw; passing a heading reading would be silently "
                    + "ignored. Construct it with headingAuthoritative=true instead");
        }
    }

    private Pose3D buildPose() {
        double yawDeg = headingAuthoritative
                ? navHeadingDeg
                : CausalHeadingChannel.wrap360(Math.toDegrees(visualHeadingRad));
        return new Pose3D(txM, -tyM, 0.0, yawDeg);
    }
}
