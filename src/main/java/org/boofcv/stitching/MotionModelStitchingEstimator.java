package org.boofcv.stitching;

import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.struct.image.ImageBase;
import georegression.struct.InvertibleTransform;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.point.Point2D_F64;
import georegression.struct.shapes.Quadrilateral_F64;
import lombok.Getter;
import lombok.Setter;
import org.boofcv.estimation.OpticalMotionEstimator;
import org.boofcv.stitching.metric.CausalHeadingChannel;
import org.boofcv.stitching.metric.CausalHeightChannel;
import org.boofcv.stitching.metric.HeadingReading;
import org.boofcv.stitching.metric.HeadingReadoutConfig;
import org.boofcv.stitching.metric.MetricNavigationState;
import org.boofcv.stitching.metric.MetricReadoutConfig;
import org.boofcv.util.structs.Pose3D;

/**
 * The stitching VO's pose estimator, parameterised by the 2D motion model.
 *
 * <p><b>A logical navigation state is maintained independently of the mosaic raster</b>
 * ({@code DEC-VO-003}): {@link LogicalNavigationState} composes the motion estimator's own
 * accumulated transform against a fixed logical origin, with reference resets folded exactly. It is
 * updated on every frame and is always available via {@link #getLogicalPose()}.
 *
 * <p><b>Which state publishes the pose is selectable, and defaults to
 * {@link NavigationSource#DEFAULT}</b> — {@code RIGID_MOTION} since {@code DEC-VO-008}. The legacy
 * mosaic-derived computation and the full logical composition both remain selectable, the first for
 * reproducing pre-2026-08-28 baselines and the second as the diagnostic that shows why composing the
 * full transform is unsuitable. See {@link NavigationSource} for what each one is and what it is
 * worth.
 *
 * <h2>Coordinate frames</h2>
 *
 * <pre>
 *   C_k  camera image frame at video frame k    (input pixels; origin top-left, x right, y down)
 *   E_j  epoch reference frame j                (BoofCV's "first"; reset by canvas re-origin)
 *   L    logical navigation frame = C_0         (FIXED for the run — the pose frame)
 *   M    mosaic canvas frame                    (raster pixels; REDEFINED at every canvas re-origin)
 *
 *   motion estimator ──► F_k : E_j → C_k ──► G_k = A_j ∘ F_k : L → C_k ──► pose + logical footprint
 *                                       └──► StitchingFromMotion2D ──► mosaic in M  (rendering only)
 * </pre>
 *
 * <p>Three events are distinct and must not be conflated:
 *
 * <ul>
 *   <li><b>estimator keyframe change</b> — SmartRespawn's tracking-robustness respawn. BoofCV's
 *       {@code changeKeyFrame()} preserves the accumulated transform exactly, so navigation needs no
 *       compensation and takes none.</li>
 *   <li><b>mosaic canvas re-origin</b> — {@code setOriginToCurrent()}, which re-renders the canvas
 *       <i>and</i> zeroes the estimator's accumulation. Navigation folds the accumulation into its
 *       anchor immediately beforehand, which keeps the logical transform exactly continuous.</li>
 *   <li><b>logical navigation coordinate change</b> — never happens after initialisation. The
 *       logical origin is fixed at the first frame for the life of the run.</li>
 * </ul>
 *
 * <p><b>The shipped production path is unchanged.</b> {@link StitchingEstimator} remains the
 * homography-typed subclass with its exact constructor and generic shape. Tracking, respawn, restart
 * and the canvas re-origin schedule are bit-identical to before {@code DEC-VO-003} — verified on real
 * data: the reference-event schedules and per-frame track counts match the committed baselines
 * exactly. Only the pose <i>derivation</i> can differ, and only when the logical source is selected.
 *
 * <p><b>Not addressed here</b>, deliberately: absolute scale, relative altitude, any conversion of
 * footprint scale to height, IMU/compass yaw fusion, and confidence. The logical transform makes the
 * scale component readable, but reading it is separate future work.
 *
 * @param <T>  image type
 * @param <IT> 2D motion model type
 */
public class MotionModelStitchingEstimator<T extends ImageBase<T>, IT extends InvertibleTransform<IT>>
        implements OpticalMotionEstimator<T> {

    @Getter
    private final StitchingFromMotion2D<T, IT> stitch;

    private final MotionModelSupport<IT> motionModel;

    /** The navigation state. Depends only on the motion model, never on the mosaic. */
    @Getter
    private final LogicalNavigationState<IT> navigation;

    /**
     * The rigid-only navigation state ({@code DEC-VO-004} Alternative E). Maintained on every
     * accepted frame regardless of which source publishes the pose, so its diagnostics — uniform
     * scale, anisotropy, perspective — are always available for inspection.
     */
    @Getter
    private final RigidNavigationState<IT> rigidNavigation;

    /**
     * Render-side only: how far the frame is shrunk when drawn into the mosaic canvas. Since
     * {@code DEC-VO-003} this no longer participates in the pose computation — its former second job
     * of normalising pose units was part of the navigation/canvas coupling that decision removed.
     */
    @Getter @Setter
    private double shrinkScale = 0.5;

    /** Render-side only: canvas margin at which the mosaic is re-originned. Not a navigation input. */
    @Getter @Setter
    private int minDistanceFromBorder = 10;

    /**
     * Notified when the mosaic canvas is re-originned.
     *
     * <p>Deliberately typed to {@code Homography2D_F64} rather than to {@code IT}: consumers are GUI
     * history transforms that are homography-based regardless of the estimator's model, and BoofCV
     * supplies an exact conversion. An affine transform's homography form is exact — its last row is
     * {@code (0, 0, 1)} — so no information is lost and every existing listener keeps working under
     * both models.
     */
    public interface RecenterListener {
        void onRecenter(Homography2D_F64 worldOldToNew);
    }

    @Setter
    private RecenterListener recenterListener;

    // ------------------------------------------------------- SYNTHETIC hard-loss injection (test)
    /** Armed by {@link #injectHardLossOnNextFrame()}; consumed by the next {@link #processFrame}. */
    private boolean hardLossInjectionArmed;
    /** How many hard losses this estimator produced because a caller asked for one. */
    @Getter
    private int syntheticHardLossCount;

    /**
     * <b>SYNTHETIC / TEST ONLY.</b> Makes the <em>next</em> processed frame take the hard-loss
     * branch exactly as if the stitcher had failed to track on it.
     *
     * <p>Natural hard loss is rare and effectively impossible to schedule by hand in a simulator,
     * yet the whole of the downstream recovery contract — segment termination, UNKNOWN translation
     * across the gap, position invalidity, heading survival, relocalization-only recovery — hangs
     * off it. This arms that one branch deterministically so those semantics can be exercised at a
     * declared frame, without damaging imagery to provoke a tracker failure that would then also
     * change what the tracker sees.
     *
     * <p>It enters the <b>same</b> code path, not a parallel one: the flag only forces
     * {@code success = false} in {@link #processFrame}, after which
     * {@code resetStitchingAndRestart} and {@code MetricNavigationState.beginNewSegment(true)} run
     * unchanged. The single observable difference from a natural loss is that the stitcher's
     * {@code process} is not invoked on the injected frame at all (a natural loss invokes it and it
     * returns false), so no tracking result is computed and then silently discarded.
     *
     * <p><b>Nothing in production arms this.</b> The only caller is the evaluation runner, and only
     * when a {@code VoRunnerConfig} carries an explicit {@code synthetic_hard_loss} block; that
     * block is recorded in the run manifest and every resulting loss is labelled
     * {@code loss_source = synthetic}, so a synthetic loss can never be mistaken for observed
     * estimator failure in later analysis. The one-shot flag is cleared whether or not it fired.
     *
     * @throws IllegalStateException if a loss is already armed (a double-arm would silently drop one)
     */
    public void injectHardLossOnNextFrame() {
        if (hardLossInjectionArmed) {
            throw new IllegalStateException("a synthetic hard loss is already armed for the next "
                    + "frame; arming twice would silently drop one of the two requested losses");
        }
        hardLossInjectionArmed = true;
    }

    /** Whether a synthetic hard loss is armed for the next processed frame. */
    public boolean isHardLossInjectionArmed() {
        return hardLossInjectionArmed;
    }

    /**
     * Which computation publishes the pose. Defaults to {@link NavigationSource#DEFAULT} —
     * {@code RIGID_MOTION} since {@code DEC-VO-008}. Set it explicitly to
     * {@link NavigationSource#MOSAIC_LEGACY} to reproduce a pre-2026-08-28 baseline, or to
     * {@link NavigationSource#METRIC_LOCAL} — which requires {@link #enableMetricReadout} first —
     * to publish metres.
     */
    @Getter
    private NavigationSource navigationSource = NavigationSource.DEFAULT;

    /**
     * The metric local navigation state ({@code DEC-VO-007} D3), or {@code null} when the metric
     * readout is not enabled — which is the default and what every pre-2026-09-06 configuration
     * gets.
     */
    @Getter
    private MetricNavigationState metricNavigation;

    /**
     * The causal height channel feeding {@link #metricNavigation}, or {@code null} when the metric
     * readout is not enabled.
     */
    @Getter
    private CausalHeightChannel heightChannel;

    /**
     * The causal heading channel that owns the navigation direction, or {@code null} when no
     * external heading is configured — in which case the metric readout falls back to the visually
     * integrated yaw, exactly as {@code DEC-VO-009} shipped it.
     */
    @Getter
    private CausalHeadingChannel headingChannel;

    /** Working-resolution focal length, retained so the heading arm can rebuild the metric state. */
    private double fWorkingPx = Double.NaN;

    private boolean gotFirstImage = false;

    // --- MOSAIC_LEGACY state (pre-DEC-VO-003, retained verbatim for baseline reproduction) ---
    private final IT startToWorld;
    private Pose3D legacyPose = new Pose3D();
    private Pose3D legacyCentrePosition = new Pose3D();
    private Pose3D legacyOffsetFromCentre = new Pose3D();
    private Point2D_F64 legacyReferenceCentre = new Point2D_F64();

    // Cache the frame dimensions
    private int frameWidth;
    private int frameHeight;

    /**
     * {@code F_k} as it stood after the last <em>successful</em> frame. BoofCV's
     * {@code ImageMotionPointTrackerKey.process} writes {@code worldToCurr} before
     * {@code StitchingFromMotion2D.checkLargeMotion} can reject the frame, so on a restart the
     * estimator's accumulation already contains the rejected estimate. The legacy readout never
     * published it (its offset is only updated on success); the logical fold must likewise use
     * this last-good copy rather than the live value. Measured on {@code HKairport01} A
     * (homography): folding the live value at the single restart moved the logical position by
     * 288 px in one frame and accounted for most of the {@code DEC-VO-003} position regression
     * ({@code EXP-VO-004} R0).
     */
    private final IT lastGoodFirstToCurrent;

    public MotionModelStitchingEstimator(StitchingFromMotion2D<T, IT> stitch,
                                         MotionModelSupport<IT> motionModel) {
        if (stitch == null) {
            throw new IllegalArgumentException("StitchingFromMotion2D cannot be null");
        }
        if (motionModel == null) {
            throw new IllegalArgumentException("MotionModelSupport cannot be null");
        }
        this.stitch = stitch;
        this.motionModel = motionModel;
        this.navigation = new LogicalNavigationState<>(
                motionModel, stitch.getMotion().getFirstToCurrent().createInstance());
        this.rigidNavigation = new RigidNavigationState<>(
                motionModel, stitch.getMotion().getFirstToCurrent().createInstance());
        this.startToWorld = motionModel.shrinkTransform(1.0, 0.0, 0.0);
        this.lastGoodFirstToCurrent = stitch.getMotion().getFirstToCurrent().createInstance();
        this.lastGoodFirstToCurrent.reset();
    }

    /** The motion model this estimator was built with, e.g. {@code "homography"}, {@code "affine"}. */
    public String motionModelId() {
        return motionModel.id();
    }

    @Override
    public void init() {
        reset();
    }

    /**
     * Processes a frame with no timestamp.
     *
     * <p><b>Rejected outright when the metric readout is enabled</b>, rather than falling back to a
     * guessed or held frame time. The height channel is a causal zero-order hold over a clock
     * ({@code DEC-VO-007} D5); sampling it without knowing which instant to sample at would silently
     * convert an increment with an arbitrary height, which is precisely the unknown scale D6
     * forbids. Call {@link #processFrame(ImageBase, double)} instead.
     */
    @Override
    public boolean processFrame(T frame) {
        if (metricNavigation != null) {
            throw new IllegalStateException(
                    "the metric readout is enabled, so every frame needs a timestamp on the height "
                    + "channel's clock: call processFrame(frame, timestampS) (DEC-VO-007 D5)");
        }
        return process(frame, Double.NaN);
    }

    /**
     * Processes a frame captured at {@code timestampS}.
     *
     * @param timestampS the frame's capture time in seconds, on the <b>same clock</b> the height
     *                   samples submitted to {@link #submitHeightSample} carry. Only used by the
     *                   metric readout; ignored when it is not enabled.
     */
    @Override
    public boolean processFrame(T frame, double timestampS) {
        return process(frame, timestampS);
    }

    private boolean process(T frame, double timestampS) {
        if (frame == null) {
            return false;
        }

        this.frameWidth = frame.width;
        this.frameHeight = frame.height;

        if (!gotFirstImage) {
            initFrame(frame);
            if (metricNavigation != null) {
                // The reference frame: latch its height, integrate nothing. This is also where h0's
                // meaning is fixed -- it is the camera-to-imaged-surface height at THIS frame.
                metricNavigation.observeOrigin(heightChannel.readAt(timestampS),
                        headingAt(timestampS));
            }
            return true;
        }

        // One-shot synthetic loss (test/experiment only, never armed in production): short-circuit
        // so the stitcher is not asked to track a frame whose result would be discarded. Everything
        // after this line is the natural hard-loss path, unmodified.
        boolean injected = hardLossInjectionArmed;
        hardLossInjectionArmed = false;
        if (injected) {
            syntheticHardLossCount++;
        }
        boolean success = !injected && stitch.process(frame);
        if (!success) {
            // Re-stitch by resetting the engine, starting from this frame. The accumulated motion is
            // folded into the navigation anchor first, so logical coordinates survive the restart.
            resetStitchingAndRestart(frame);
            if (metricNavigation != null) {
                // The motion over the failed interval was never estimated (COMP-001 section 8) and
                // this layer does not invent it: no increment is integrated here, and the gap is
                // marked as a new segment whose preceding displacement is UNKNOWN.
                metricNavigation.beginNewSegment(true);
                metricNavigation.observeOrigin(heightChannel.readAt(timestampS),
                        headingAt(timestampS));
            }
            return false;
        }

        // Navigation: from the motion model alone. Always maintained, so the logical state is
        // available for inspection even when the legacy source is the one publishing the pose.
        navigation.observe(stitch.getMotion().getFirstToCurrent(), frame.width, frame.height);
        // Rigid path: increment against the last ACCEPTED accumulation, so a rejected estimate can
        // never enter it (EXP-VO-004 R0), and identity after any reset, so the canvas schedule
        // cannot enter it either.
        rigidNavigation.observe(lastGoodFirstToCurrent, stitch.getMotion().getFirstToCurrent(),
                frame.width, frame.height);
        lastGoodFirstToCurrent.setTo(stitch.getMotion().getFirstToCurrent());

        if (metricNavigation != null) {
            // DEC-VO-007 D3, one multiply: the SAME increment the rigid state just integrated,
            // scaled by the REFERENCE frame's ground sampling distance. The reading passed here is
            // this frame's; MetricNavigationState converts with the one it latched last time, which
            // is what makes the k-1 convention structural rather than a caller's responsibility.
            metricNavigation.observe(rigidNavigation.getIncrementDqX(),
                    rigidNavigation.getIncrementDqY(),
                    rigidNavigation.getIncrementRotationRad(),
                    heightChannel.readAt(timestampS),
                    headingAt(timestampS));
        }

        Quadrilateral_F64 canvasCorners = stitch.getImageCorners(frame.width, frame.height, null);
        if (navigationSource == NavigationSource.MOSAIC_LEGACY) {
            legacyUpdateOffsetFromCentre(canvasCorners);
        }

        if (nearCanvasBorder(canvasCorners)) {
            reoriginMosaicCanvas();
        }

        if (navigationSource == NavigationSource.MOSAIC_LEGACY) {
            legacyUpdatePose();
        }
        return true;
    }

    @Override
    public void reset() {
        stitch.reset();
        navigation.reset();
        rigidNavigation.reset();
        if (metricNavigation != null) {
            // Neither CHANNEL is reset: h0, delta_mount and tau_stale are mission parameters, and the
            // samples already submitted are history, not state to be rewound.
            metricNavigation.reset();
        }
        lastGoodFirstToCurrent.reset();
        gotFirstImage = false;
        hardLossInjectionArmed = false;
        syntheticHardLossCount = 0;
        startToWorld.reset();
        legacyCentrePosition = new Pose3D();
        legacyOffsetFromCentre = new Pose3D();
        legacyPose = new Pose3D();
    }

    /**
     * The published pose, in the units {@link #getNavigationUnits()} declares.
     *
     * <p><b>Read those units before comparing this against any stored distance.</b> Under
     * {@link NavigationSource#METRIC_LOCAL} the {@code x}/{@code y} are metres; under the other
     * three they are first-frame pixels. The two differ by the ground sampling distance, and
     * {@code Pose3D} carries no unit of its own.
     */
    @Override
    public Pose3D getCurrentPose() {
        switch (navigationSource) {
            case LOGICAL_FRAME:
                return navigation.pose();
            case RIGID_MOTION:
                return rigidNavigation.pose();
            case METRIC_LOCAL:
                return metricNavigation.metricPose();
            default:
                return legacyPose;
        }
    }

    /** The length unit of {@link #getCurrentPose()}'s {@code x}/{@code y}, for consumers that hold distances. */
    public NavigationUnits getNavigationUnits() {
        return navigationSource.units();
    }

    /**
     * Selects which computation publishes the pose.
     *
     * @throws IllegalStateException when {@link NavigationSource#METRIC_LOCAL} is requested without
     *                               {@link #enableMetricReadout} having been called. There is no
     *                               path by which a pixel-valued pose is relabelled as metres.
     */
    public void setNavigationSource(NavigationSource navigationSource) {
        if (navigationSource == NavigationSource.METRIC_LOCAL && metricNavigation == null) {
            throw new IllegalStateException(
                    "METRIC_LOCAL needs the metric readout enabled first: it requires an external "
                    + "h0, a working-resolution focal length and a causal height channel "
                    + "(DEC-VO-007 D2/D3). Call enableMetricReadout(...).");
        }
        this.navigationSource = navigationSource;
    }

    /**
     * Turns on the metric local navigation readout ({@code DEC-VO-007} D3). Off by default, so every
     * pre-2026-09-06 configuration is byte-for-byte unaffected.
     *
     * <p>It changes nothing about the estimator: no transform, no tracking, no RANSAC, no
     * acceptance, no state of {@link #getRigidNavigation()}. It adds one multiply per accepted frame
     * on an increment the rigid readout already produces, and it does <b>not</b> change which source
     * publishes the pose — call {@link #setNavigationSource} for that, deliberately as a second,
     * separate act.
     *
     * <p>From this call on, {@link #processFrame(ImageBase)} is refused and every frame must arrive
     * through {@link #processFrame(ImageBase, double)} with a timestamp on the height channel's
     * clock.
     *
     * @param config the declared optical and height parameters; a fresh channel is built from it
     * @return the channel to feed height samples into, also available from {@link #getHeightChannel()}
     */
    public CausalHeightChannel enableMetricReadout(MetricReadoutConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("a MetricReadoutConfig is required");
        }
        if (gotFirstImage) {
            throw new IllegalStateException(
                    "enable the metric readout before the first frame: enabling it mid-run would "
                    + "silently start the metric origin somewhere other than the run's own");
        }
        this.heightChannel = config.newHeightChannel();
        this.fWorkingPx = config.fWorkingPx();
        this.metricNavigation = new MetricNavigationState(fWorkingPx, headingChannel != null);
        return this.heightChannel;
    }

    /**
     * Makes an <b>external heading channel authoritative</b> for the navigation direction. The
     * visually integrated yaw is retained and keeps being computed and logged, but it no longer
     * determines where a metric translation increment points.
     *
     * <p>This is a substitution, not a fusion: there is no blend, no complementary filter, no
     * Kalman state and no confidence weight anywhere in the path. Height owns scale, heading owns
     * direction, and VO owns the image displacement.
     *
     * <p>Must be called before the first frame, and the metric readout must already be enabled —
     * heading has nothing to rotate without it.
     *
     * @param config the declared semantics, mounting calibration and staleness parameters
     * @return the channel to feed heading samples into, also available from
     *         {@link #getHeadingChannel()}
     */
    public CausalHeadingChannel enableHeadingReadout(HeadingReadoutConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("a HeadingReadoutConfig is required");
        }
        if (metricNavigation == null) {
            throw new IllegalStateException(
                    "enable the metric readout first: an authoritative heading rotates a METRIC "
                    + "translation increment, and without a height channel there is none");
        }
        if (gotFirstImage) {
            throw new IllegalStateException(
                    "enable the heading readout before the first frame: switching the navigation "
                    + "direction mid-run would splice two differently-referenced trajectories");
        }
        this.headingChannel = config.newHeadingChannel();
        this.metricNavigation = new MetricNavigationState(fWorkingPx, true);
        return this.headingChannel;
    }

    /**
     * Feeds one external heading sample into the causal channel. See
     * {@link CausalHeadingChannel#submit}.
     *
     * @return true when accepted; false when the rate-plausibility gate rejected it and the previous
     *         valid heading was retained
     * @throws IllegalStateException when no heading channel is configured
     */
    public boolean submitHeadingSample(double availabilityTimeS, double headingDeg) {
        if (headingChannel == null) {
            throw new IllegalStateException(
                    "no heading channel: call enableHeadingReadout(...) before submitting samples");
        }
        return headingChannel.submit(availabilityTimeS, headingDeg);
    }

    /** True when an external heading channel owns the navigation direction. */
    public boolean isHeadingReadoutEnabled() {
        return headingChannel != null;
    }

    /** The heading reading for a frame time, or {@code null} when no heading channel is configured. */
    private HeadingReading headingAt(double timestampS) {
        return headingChannel == null ? null : headingChannel.readAt(timestampS);
    }

    /**
     * Feeds one takeoff-relative height sample into the causal channel. See
     * {@link CausalHeightChannel#submit}.
     *
     * @throws IllegalStateException when the metric readout is not enabled — silently discarding a
     *                               sensor sample would leave a caller believing height was reaching
     *                               the pose when it was not
     */
    public void submitHeightSample(double availabilityTimeS, double relativeHeightM) {
        if (heightChannel == null) {
            throw new IllegalStateException(
                    "no height channel: call enableMetricReadout(...) before submitting samples");
        }
        heightChannel.submit(availabilityTimeS, relativeHeightM);
    }

    /**
     * The metric local pose regardless of which source is publishing, for diagnostics and logging.
     * <b>Metres.</b>
     *
     * @throws IllegalStateException when the metric readout is not enabled
     */
    public Pose3D getMetricPose() {
        if (metricNavigation == null) {
            throw new IllegalStateException("the metric readout is not enabled");
        }
        return metricNavigation.metricPose();
    }

    /** True when the metric readout is enabled. */
    public boolean isMetricReadoutEnabled() {
        return metricNavigation != null;
    }

    /** The rigid-frame pose, regardless of which source is publishing. For diagnostics. */
    public Pose3D getRigidPose() {
        return rigidNavigation.pose();
    }

    /** The logical-frame pose, regardless of which source is publishing. For diagnostics. */
    public Pose3D getLogicalPose() {
        return navigation.pose();
    }

    /**
     * The camera's image rectangle expressed in the fixed logical frame — centre, orientation, size
     * and deformation, with no dependence on the mosaic raster. Returns internal storage.
     */
    public Quadrilateral_F64 getLogicalFootprint() {
        return navigation.logicalFootprint();
    }

    /**
     * {@code G_k : L → C_k} in homography form (exact for both models), for diagnostics that
     * persist the logical transform per frame ({@code EXP-VO-004}). Allocates when {@code out} is
     * null.
     */
    public Homography2D_F64 getLogicalToCurrentAsHomography(Homography2D_F64 out) {
        return motionModel.asHomography(navigation.logicalToCurrent(), out);
    }

    private void initFrame(T frame) {
        configureCanvasFor(frame.width, frame.height);
        stitch.process(frame);
        rigidNavigation.reset();
        rigidNavigation.observeOrigin();

        // The logical origin is this frame: anchor = identity, and the estimator's accumulation is
        // identity here too, so the first pose is exactly zero.
        navigation.observe(stitch.getMotion().getFirstToCurrent(), frame.width, frame.height);
        lastGoodFirstToCurrent.setTo(stitch.getMotion().getFirstToCurrent());

        IT frameToCanvas = motionModel.shrinkTransform(
                shrinkScale, frame.width * (1.0 - shrinkScale) / 2.0,
                frame.height * (1.0 - shrinkScale) / 2.0);
        startToWorld.setTo(frameToCanvas);
        legacyReferenceCentre = centroid(stitch.getImageCorners(frame.width, frame.height, null));
        legacyCentrePosition = new Pose3D(0, 0, 0, 0);
        legacyOffsetFromCentre = new Pose3D();
        legacyUpdatePose();

        gotFirstImage = true;
    }

    /**
     * Places the frame, shrunk and centred, into a canvas of the same pixel dimensions. Pure
     * rendering setup: it defines the mosaic frame {@code M} and nothing else.
     */
    private void configureCanvasFor(int width, int height) {
        double tx = width * (1.0 - shrinkScale) / 2.0;
        double ty = height * (1.0 - shrinkScale) / 2.0;
        IT canvasToFrame = motionModel.shrinkTransform(shrinkScale, tx, ty).invert(null);
        stitch.configure(width, height, canvasToFrame);
    }

    /**
     * Recovery after a stitching failure. {@code stitch.reset()} zeroes the estimator's accumulated
     * transform, so it is folded into the navigation anchor first.
     *
     * <p><b>Unchanged pre-existing limitation</b> ({@code COMP-001} §8): the motion between the last
     * good frame and the restart frame is not estimated and is implicitly assumed zero. That error
     * is inherent to the restart and is not addressed by {@code DEC-VO-003}.
     *
     * <p><b>Fixed 2026-08-25 ({@code EXP-VO-004} R0):</b> the fold uses {@link #lastGoodFirstToCurrent},
     * not the live accumulation, because on a {@code checkLargeMotion} failure the live value already
     * contains the rejected estimate. This makes the logical path discard the rejected frame exactly
     * as the legacy path always did.
     */
    private void resetStitchingAndRestart(T frame) {
        navigation.foldEpoch(lastGoodFirstToCurrent);

        IT canvasOldToCurr = stitch.getWorldToCurr();
        IT startToCurrBeforeReset = canvasOldToCurr.concat(startToWorld, null);
        legacyCentrePosition = legacyCentrePosition.plus(legacyOffsetFromCentre.x,
                legacyOffsetFromCentre.y, legacyOffsetFromCentre.z, legacyOffsetFromCentre.yaw);
        legacyOffsetFromCentre = new Pose3D();

        stitch.reset();
        configureCanvasFor(frame.width, frame.height);
        stitch.process(frame);

        // Accumulation is identity again; the anchor now carries everything up to the restart.
        navigation.observe(stitch.getMotion().getFirstToCurrent(), frame.width, frame.height);
        // The accumulation is identity again, so the next frame's rigid increment is measured
        // against identity. The motion between the last good frame and the restart frame is not
        // estimated (COMP-001 §8) and is therefore not integrated by either path.
        lastGoodFirstToCurrent.reset();

        IT frameToCanvas = motionModel.shrinkTransform(
                shrinkScale, frame.width * (1.0 - shrinkScale) / 2.0,
                frame.height * (1.0 - shrinkScale) / 2.0);
        frameToCanvas.concat(startToCurrBeforeReset, startToWorld);
        legacyReferenceCentre = centroid(stitch.getImageCorners(frame.width, frame.height, null));
        legacyUpdatePose();
    }

    /**
     * Re-origins the finite mosaic canvas so the current frame is drawn at its centre again.
     *
     * <p>This is a <b>rendering</b> operation, but BoofCV welds an estimator reset to it:
     * {@code setOriginToCurrent()} calls {@code motion.setToFirst()}, which is
     * {@code changeKeyFrame(); resetTransforms()}, and {@code resetTransforms()} zeroes the
     * accumulated transform. The accumulation is therefore folded into the navigation anchor
     * immediately beforehand, which makes the logical transform exactly continuous across the event.
     */
    private void reoriginMosaicCanvas() {
        // Captured before the mutation: setOriginToCurrent() rewrites worldToCurr in place.
        Homography2D_F64 canvasOldToCurrH = stitch.getWorldToCurr(new Homography2D_F64());
        IT canvasOldToCurr = stitch.getWorldToCurr().createInstance();
        canvasOldToCurr.setTo(stitch.getWorldToCurr());
        IT startToCurrBeforeReorigin = canvasOldToCurr.concat(startToWorld, null);
        legacyCentrePosition = legacyCentrePosition.plus(legacyOffsetFromCentre.x,
                legacyOffsetFromCentre.y, legacyOffsetFromCentre.z, legacyOffsetFromCentre.yaw);
        legacyOffsetFromCentre = new Pose3D();

        navigation.foldEpoch(stitch.getMotion().getFirstToCurrent());

        stitch.setOriginToCurrent();
        // setOriginToCurrent() zeroes the estimator's accumulation, so the next frame's rigid
        // increment must be measured against identity. This is the whole of the rigid path's
        // re-origin handling: no anchor, nothing to fold, nothing to compensate.
        lastGoodFirstToCurrent.reset();

        if (recenterListener != null) {
            recenterListener.onRecenter(canvasOldToCurrH);
        }

        IT frameToCanvas = motionModel.shrinkTransform(
                shrinkScale, frameWidth * (1.0 - shrinkScale) / 2.0,
                frameHeight * (1.0 - shrinkScale) / 2.0);
        frameToCanvas.concat(startToCurrBeforeReorigin, startToWorld);
        legacyReferenceCentre = centroid(stitch.getImageCorners(frameWidth, frameHeight, null));
    }

    // ---------------------------------------------------------------- MOSAIC_LEGACY computation
    // Retained verbatim from before DEC-VO-003 so the committed baselines reproduce exactly. This
    // is the mosaic-coupled path the decision documents as architecturally wrong; it is kept, not
    // endorsed. See NavigationSource.

    private void legacyUpdateOffsetFromCentre(Quadrilateral_F64 canvasCorners) {
        Point2D_F64 centre = centroid(canvasCorners);
        double rotation = edgeAngleDegrees(canvasCorners);

        double relativeX = (centre.x - legacyReferenceCentre.x) / shrinkScale;
        double relativeY = (centre.y - legacyReferenceCentre.y) / shrinkScale;

        Pose3D ans = new Pose3D(legacyCentrePosition.x + relativeX,
                legacyCentrePosition.y - relativeY, 0.0, rotation);
        Pose3D rotated = ans.rotate(legacyCentrePosition.x, legacyCentrePosition.y,
                360.0 - legacyCentrePosition.yaw);
        legacyOffsetFromCentre = rotated.minus(legacyCentrePosition);
    }

    private void legacyUpdatePose() {
        legacyPose = legacyCentrePosition.plus(legacyOffsetFromCentre.x, legacyOffsetFromCentre.y,
                legacyOffsetFromCentre.z, legacyOffsetFromCentre.yaw);
    }

    private static Point2D_F64 centroid(Quadrilateral_F64 q) {
        return new Point2D_F64((q.a.x + q.b.x + q.c.x + q.d.x) / 4.0,
                               (q.a.y + q.b.y + q.c.y + q.d.y) / 4.0);
    }

    private static double edgeAngleDegrees(Quadrilateral_F64 q) {
        double angle = Math.toDegrees(Math.atan2(q.b.y - q.a.y, q.b.x - q.a.x));
        return (angle + 360.0) % 360.0;
    }

    private boolean nearCanvasBorder(Quadrilateral_F64 canvasCorners) {
        return nearCanvasBorder(canvasCorners.a) || nearCanvasBorder(canvasCorners.b) ||
               nearCanvasBorder(canvasCorners.c) || nearCanvasBorder(canvasCorners.d);
    }

    private boolean nearCanvasBorder(Point2D_F64 p) {
        int stitchWidth = stitch.getStitchedImage().width;
        int stitchHeight = stitch.getStitchedImage().height;

        return p.x < minDistanceFromBorder || p.y < minDistanceFromBorder ||
               p.x >= stitchWidth - minDistanceFromBorder ||
               p.y >= stitchHeight - minDistanceFromBorder;
    }
}
