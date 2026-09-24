package org.boofcv.stitching;

import boofcv.abst.feature.detect.interest.ConfigPointDetector;
import boofcv.abst.feature.detect.interest.PointDetectorTypes;
import boofcv.abst.sfm.d2.ImageMotion2D;
import boofcv.abst.sfm.d2.PlToGrayMotion2D;
import boofcv.abst.sfm.d2.WrapImageMotionPtkSmartRespawn;
import boofcv.abst.tracker.PointTracker;
import boofcv.alg.distort.ImageDistort;
import boofcv.alg.geo.robust.DistanceAffine2DSq;
import boofcv.alg.geo.robust.DistanceHomographySq;
import boofcv.alg.geo.robust.GenerateAffine2D;
import boofcv.alg.geo.robust.GenerateHomographyLinear;
import boofcv.alg.interpolate.InterpolatePixel;
import boofcv.alg.interpolate.InterpolationType;
import boofcv.alg.sfm.d2.ImageMotionPointTrackerKey;
import boofcv.alg.sfm.d2.ImageMotionPtkSmartRespawn;
import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.factory.distort.FactoryDistort;
import boofcv.factory.interpolate.FactoryInterpolation;
import boofcv.factory.sfm.FactoryMotion2D;
import boofcv.factory.tracker.FactoryPointTracker;
import boofcv.struct.border.BorderType;
import boofcv.struct.geo.AssociatedPair;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.ImageBase;
import boofcv.struct.image.ImageType;
import boofcv.struct.image.Planar;
import georegression.fitting.affine.ModelManagerAffine2D_F64;
import georegression.fitting.homography.ModelManagerHomography2D_F64;
import georegression.struct.InvertibleTransform;
import georegression.struct.affine.Affine2D_F64;
import georegression.struct.homography.Homography2D_F64;
import org.ddogleg.fitting.modelset.DistanceFromModel;
import org.ddogleg.fitting.modelset.ModelFitter;
import org.ddogleg.fitting.modelset.ModelGenerator;
import org.ddogleg.fitting.modelset.ModelManager;
import org.ddogleg.fitting.modelset.ModelMatcherPost;
import org.ddogleg.fitting.modelset.ransac.Ransac;
import org.ddogleg.struct.Factory;

import javax.annotation.Nullable;

/**
 * Factory and Builder for creating configured instances of {@link StitchingFromMotion2D}.
 *
 * <p>Portions of this file ({@code createInstrumentedMotion2D}, {@code createSimilarityMotion2D},
 * {@code createProbedMotion2D}) are adapted from BoofCV 0.44's {@code FactoryMotion2D}, Copyright
 * (c) 2021 Peter Abeles, licensed under the Apache License 2.0. See {@code THIRD_PARTY_NOTICES.md}
 * and {@code LICENSES/Apache-2.0.txt}.
 */
public class StitchingFactory {

    public static ConfigPointDetector defaultConfigDetector() {
        ConfigPointDetector configDetector = new ConfigPointDetector();
        configDetector.type = PointDetectorTypes.SHI_TOMASI;
        configDetector.general.maxFeatures = 300;
        configDetector.general.radius = 3;
        configDetector.general.threshold = 1;
        return configDetector;
    }

    public static StitchingFromMotion2D<Planar<GrayF32>, Homography2D_F64> defaultPlanar() {
        return defaultPlanar(defaultConfigDetector());
    }

    // NOTE (research log 2026-08-14, gap G17): this call uses the deprecated
    // FactoryPointTracker.klt(numLevels, configDetect, featureRadius, ...) overload, which builds
    // a default ConfigPKlt and does NOT copy configDetector.general.maxFeatures into it. BoofCV's
    // PointTrackerKltPyramid.spawnTracks() therefore caps every respawn at its own
    // ConfigPKlt.maximumTracks (default ConfigLength.relative(0.002, 50): 0.2% of the frame's
    // pixel count, minimum 50) -- NOT at configDetector.general.maxFeatures. `maxFeatures` in
    // this class and in VoRunnerConfig only bounds the detector's own internal candidate list; it
    // is not the effective per-frame track cap. Verified empirically (see COMP-001 s3.1): at
    // 640x512 the observed ceiling is exactly 655 = floor(0.002 * 327680); at 2448x2048 (MARS-LVIG
    // HKairport01, EXP-002) it is exactly 10027 = floor(0.002 * 5013504). Left unchanged
    // deliberately -- fixing it changes VO tracking behaviour, which needs literature-first
    // treatment as a component change, not a documentation pass.
    public static StitchingFromMotion2D<Planar<GrayF32>, Homography2D_F64> defaultPlanar(ConfigPointDetector configDetector) {
        PointTracker<GrayF32> tracker = FactoryPointTracker.klt(4, configDetector, 3, GrayF32.class, GrayF32.class);

        ImageMotion2D<GrayF32, Homography2D_F64> motion2D =
                FactoryMotion2D.createMotion2D(220, 3, 2, 30, 0.6, 0.5, false, tracker, new Homography2D_F64());

        ImageMotion2D<Planar<GrayF32>, Homography2D_F64> motion2DColor =
                new PlToGrayMotion2D<>(motion2D, GrayF32.class);

        return FactoryMotion2D.createVideoStitch(0.55, motion2DColor, ImageType.pl(3, GrayF32.class));
    }

    public static StitchingFromMotion2D<GrayF32, Homography2D_F64> defaultGray() {
        return defaultGray(defaultConfigDetector());
    }

    public static StitchingFromMotion2D<GrayF32, Homography2D_F64> defaultGray(ConfigPointDetector configDetector) {
        PointTracker<GrayF32> tracker = FactoryPointTracker.klt(4, configDetector, 3, GrayF32.class, GrayF32.class);

        ImageMotion2D<GrayF32, Homography2D_F64> motion2D =
                FactoryMotion2D.createMotion2D(220, 3, 2, 30, 0.6, 0.5, false, tracker, new Homography2D_F64());

        return FactoryMotion2D.createVideoStitch(0.55, motion2D, ImageType.single(GrayF32.class));
    }

    /**
     * Creates an advanced configuration Builder for the stitching engine.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * The RANSAC seed BoofCV's {@code FactoryMotion2D.createMotion2D} hardcodes (0.44,
     * {@code FactoryMotion2D:106}). Replicated verbatim so the instrumented stack draws the
     * identical random sample sequence as the factory-built one. If a BoofCV upgrade changes this
     * constant, {@code StitchingFactoryInstrumentationEquivalenceTest} fails.
     */
    private static final long BOOFCV_MOTION2D_RANSAC_SEED = 123123;

    /**
     * Builds the same motion-estimation stack as
     * {@code FactoryMotion2D.createMotion2D(..., new Homography2D_F64())}, but retains a handle on
     * the low-level estimator so its reprojection residuals can be observed.
     *
     * <h2>Why this exists rather than calling the factory</h2>
     *
     * <p>{@code createMotion2D} returns a {@code WrapImageMotionPtkSmartRespawn} whose {@code alg}
     * field is package-private with no accessor, and whose declared {@code ImageMotion2D} return
     * type exposes neither the {@code ModelMatcher} nor the frame-to-keyframe model. The chain
     * {@code Wrap… → getMotion() → getModelMatcher()} exists and is public at every step
     * <em>except</em> the first, so the only way to reach it is to hold the reference from
     * construction. Nothing here reaches into non-public BoofCV state, and no reflection is used.
     *
     * <h2>Equivalence</h2>
     *
     * <p>This is a verbatim replication of {@code FactoryMotion2D.createMotion2D}'s
     * {@code Homography2D_F64} branch as it stands in BoofCV 0.44 — same model manager, same
     * generator and distance factories, same {@code Ransac} construction with the same fixed seed,
     * same {@code ImageMotionPtkSmartRespawn} wrapping, same {@code WrapImageMotionPtkSmartRespawn}
     * return. Only the {@code Homography2D_F64} branch is replicated, because that is the only
     * model this repository uses; the affine and SE(2) branches are deliberately not carried over.
     *
     * <p>The single difference is that the low-level estimator is a
     * {@link ResidualSnapshotTrackerKey} rather than a plain {@code ImageMotionPointTrackerKey}.
     * That subclass overrides {@code process} only to read values out after
     * {@code super.process} has run, and mutates no estimator state.
     *
     * <p>Equivalence is not asserted by this argument alone —
     * {@code StitchingFactoryInstrumentationEquivalenceTest} drives both stacks over the same
     * frames and fails on any bitwise difference in motion or pose. Treat that test as the
     * load-bearing check when upgrading BoofCV.
     */
    private static <I extends ImageBase<I>> Instrumented<I> createInstrumentedMotion2D(
            int ransacIterations, double inlierThreshold, int outlierPrune,
            int absoluteMinimumTracks, double respawnTrackFraction, double respawnCoverageFraction,
            boolean refineEstimate, PointTracker<I> tracker) {

        ModelManager<Homography2D_F64> manager = new ModelManagerHomography2D_F64();

        @Nullable ModelFitter<Homography2D_F64, AssociatedPair> modelRefiner =
                refineEstimate ? new GenerateHomographyLinear(true) : null;

        Factory<ModelGenerator<Homography2D_F64, AssociatedPair>> fitter =
                () -> new GenerateHomographyLinear(true);
        Factory<DistanceFromModel<Homography2D_F64, AssociatedPair>> distance =
                DistanceHomographySq::new;

        ModelMatcherPost<Homography2D_F64, AssociatedPair> modelMatcher = new Ransac<>(
                BOOFCV_MOTION2D_RANSAC_SEED, ransacIterations, inlierThreshold,
                manager, AssociatedPair.class);
        modelMatcher.setModel(fitter, distance);

        ResidualSnapshotTrackerKey<I> lowlevel = new ResidualSnapshotTrackerKey<>(
                tracker, modelMatcher, modelRefiner, new Homography2D_F64(), outlierPrune,
                inlierThreshold);

        ImageMotionPtkSmartRespawn<I, Homography2D_F64> smartRespawn =
                new ImageMotionPtkSmartRespawn<>(lowlevel,
                        absoluteMinimumTracks, respawnTrackFraction, respawnCoverageFraction);

        return new Instrumented<>(new WrapImageMotionPtkSmartRespawn<>(smartRespawn), lowlevel);
    }

    /** Internal pairing of the built motion estimator with its residual view. */
    private record Instrumented<I extends ImageBase<I>>(
            ImageMotion2D<I, Homography2D_F64> motion,
            MotionResidualDiagnostics diagnostics) {
    }

    /**
     * Builds the motion-estimation stack for the 4-DoF {@link Sim2_F64} model
     * ({@code DEC-VO-006}, {@code EXP-VO-009}).
     *
     * <h2>Why this exists rather than calling the factory</h2>
     *
     * <p>{@code FactoryMotion2D.createMotion2D} dispatches on the runtime type of its
     * {@code motionModel} argument and implements exactly three branches
     * ({@code Homography2D_F64}, {@code Affine2D_F64}, {@code Se2_F64}); anything else throws
     * {@code RuntimeException("Unknown model type")}. Since BoofCV's own similarity struct is not an
     * {@code InvertibleTransform} at all, there is no way to reach the pipeline for this model
     * except to assemble it here.
     *
     * <h2>Equivalence</h2>
     *
     * <p>This is a verbatim replication of {@code createMotion2D}'s structure with the similarity
     * model's four components substituted — the same {@code Ransac} class with the same hardcoded
     * seed, the same iteration count, the same inlier threshold, the same
     * {@code ImageMotionPointTrackerKey}, the same {@code ImageMotionPtkSmartRespawn} with the same
     * three respawn parameters, the same {@code WrapImageMotionPtkSmartRespawn}. The four
     * substituted components are {@link ModelManagerSim2_F64}, {@link GenerateSimilarity2D},
     * {@link DistanceSimilarity2DSq} and — in {@link #similarityVideoStitch} —
     * {@link SimilarityStitchingTransform}, exactly parallel to the four the factory's affine branch
     * substitutes ({@code LIT-VO-001}).
     *
     * <p><b>{@code modelRefiner} is deliberately not wired.</b> The factory supplies one only when
     * {@code refineEstimate} is true, and this repository always passes false, so a refiner would
     * change behaviour relative to the other two arms. {@code LIT-VO-005} §4 argues that
     * {@code refineEstimate} may matter more than the model class does — which is exactly why it is
     * held fixed here rather than quietly enabled for the new arm.
     */
    private static <I extends ImageBase<I>> ImageMotion2D<I, Sim2_F64> createSimilarityMotion2D(
            int ransacIterations, double inlierThreshold, int outlierPrune,
            int absoluteMinimumTracks, double respawnTrackFraction, double respawnCoverageFraction,
            PointTracker<I> tracker) {

        ModelManager<Sim2_F64> manager = new ModelManagerSim2_F64();
        Factory<ModelGenerator<Sim2_F64, AssociatedPair>> fitter = GenerateSimilarity2D::new;
        Factory<DistanceFromModel<Sim2_F64, AssociatedPair>> distance = DistanceSimilarity2DSq::new;

        ModelMatcherPost<Sim2_F64, AssociatedPair> modelMatcher = new Ransac<>(
                BOOFCV_MOTION2D_RANSAC_SEED, ransacIterations, inlierThreshold,
                manager, AssociatedPair.class);
        modelMatcher.setModel(fitter, distance);

        ImageMotionPointTrackerKey<I, Sim2_F64> lowlevel = new ImageMotionPointTrackerKey<>(
                tracker, modelMatcher, null, new Sim2_F64(), outlierPrune);

        ImageMotionPtkSmartRespawn<I, Sim2_F64> smartRespawn =
                new ImageMotionPtkSmartRespawn<>(lowlevel,
                        absoluteMinimumTracks, respawnTrackFraction, respawnCoverageFraction);

        return new WrapImageMotionPtkSmartRespawn<>(smartRespawn);
    }

    /** A motion estimator paired with the refinement view of its own low-level tracker key. */
    public record ProbedMotion<IT extends InvertibleTransform<IT>>(
            ImageMotion2D<GrayF32, IT> motion,
            RefinementDiagnostics diagnostics) {
    }

    /**
     * The homography or affine motion stack, built exactly as
     * {@code FactoryMotion2D.createMotion2D} builds it, but retaining a handle on the low-level
     * estimator so the model can be observed <b>before and after</b> RANSAC's optional final
     * refinement ({@code EXP-VO-010} Phase 4).
     *
     * <h2>Why this exists rather than calling the factory</h2>
     *
     * <p>Same reason as {@link #createInstrumentedMotion2D}: {@code createMotion2D} returns a
     * {@code WrapImageMotionPtkSmartRespawn} whose {@code alg} field is package-private with no
     * accessor, so the only way to reach the tracker key is to hold the reference from
     * construction. Nothing here reaches into non-public BoofCV state and no reflection is used.
     *
     * <h2>Equivalence</h2>
     *
     * <p>A verbatim replication of {@code createMotion2D}'s {@code Homography2D_F64} and
     * {@code Affine2D_F64} branches as they stand in BoofCV 0.44 — same model manager, same
     * generator and distance factories, same {@code Ransac} construction with the same hardcoded
     * seed, <b>same refiner construction rule</b> (a second instance of the generator class, exactly
     * as the factory does, supplied only when {@code refineEstimate} is true), same
     * {@code ImageMotionPtkSmartRespawn} wrapping, same {@code WrapImageMotionPtkSmartRespawn}
     * return. The single difference is that the low-level estimator is a
     * {@link RefinementSnapshotTrackerKey}, which overrides {@code process} only to read values out
     * after {@code super.process} has run.
     *
     * <p>Equivalence is not asserted by this argument alone —
     * {@code RefinementProbeEquivalenceTest} drives probed and unprobed stacks over the same frames,
     * for both models and both refinement settings, and fails on any bitwise difference.
     *
     * @param model whether to build the projective or the affine stack; the similarity model cannot
     *              refine and is rejected ({@code GenerateSimilarity2D} implements
     *              {@code ModelGenerator} only — {@code EXP-VO-010} Phase 1 Q7)
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    public static ProbedMotion<?> createProbedMotion2D(
            ProbeModel model, int ransacIterations, double inlierThreshold, int outlierPrune,
            int absoluteMinimumTracks, double respawnTrackFraction, double respawnCoverageFraction,
            boolean refineEstimate, PointTracker<GrayF32> tracker) {
        return createProbedMotion2D(model, ransacIterations, inlierThreshold, outlierPrune,
                absoluteMinimumTracks, respawnTrackFraction, respawnCoverageFraction,
                refineEstimate, false, tracker);
    }

    /**
     * As {@link #createProbedMotion2D(ProbeModel, int, double, int, int, double, double, boolean,
     * PointTracker)}, optionally attaching {@code EXP-CONF-004}'s <b>diagnostic-only refiner</b>:
     * a second, observer-owned instance of the same generator class the refinement rule would
     * construct, which the probe applies to the production match set as a pure function after the
     * estimator has shipped its model. It is never handed to the estimator, so with
     * {@code refineEstimate = false} the shipped model remains the minimal-sample winner —
     * {@code DiagnosticRefitEquivalenceTest} asserts bit-identity with the unobserved stack.
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    public static ProbedMotion<?> createProbedMotion2D(
            ProbeModel model, int ransacIterations, double inlierThreshold, int outlierPrune,
            int absoluteMinimumTracks, double respawnTrackFraction, double respawnCoverageFraction,
            boolean refineEstimate, boolean diagnosticRefit, PointTracker<GrayF32> tracker) {

        ModelManager manager;
        Factory fitterFactory;
        Factory distanceFactory;
        ModelFitter refiner;
        ModelFitter diagnosticRefiner;
        InvertibleTransform prototype;
        MotionModelSupport support;

        if (model == ProbeModel.AFFINE) {
            manager = new ModelManagerAffine2D_F64();
            fitterFactory = (Factory<ModelGenerator<Affine2D_F64, AssociatedPair>>) GenerateAffine2D::new;
            distanceFactory = (Factory<DistanceFromModel<Affine2D_F64, AssociatedPair>>) DistanceAffine2DSq::new;
            refiner = refineEstimate ? new GenerateAffine2D() : null;
            diagnosticRefiner = diagnosticRefit ? new GenerateAffine2D() : null;
            prototype = new Affine2D_F64();
            support = MotionModelSupport.AFFINE;
        } else {
            manager = new ModelManagerHomography2D_F64();
            fitterFactory = (Factory<ModelGenerator<Homography2D_F64, AssociatedPair>>)
                    () -> new GenerateHomographyLinear(true);
            distanceFactory = (Factory<DistanceFromModel<Homography2D_F64, AssociatedPair>>)
                    DistanceHomographySq::new;
            refiner = refineEstimate ? new GenerateHomographyLinear(true) : null;
            diagnosticRefiner = diagnosticRefit ? new GenerateHomographyLinear(true) : null;
            prototype = new Homography2D_F64();
            support = MotionModelSupport.HOMOGRAPHY;
        }

        ModelMatcherPost modelMatcher = new Ransac<>(
                BOOFCV_MOTION2D_RANSAC_SEED, ransacIterations, inlierThreshold, manager,
                AssociatedPair.class);
        modelMatcher.setModel(fitterFactory, distanceFactory);

        RefinementSnapshotTrackerKey lowlevel = new RefinementSnapshotTrackerKey<>(
                tracker, modelMatcher, refiner, prototype, outlierPrune, support,
                diagnosticRefiner);

        ImageMotionPtkSmartRespawn smartRespawn = new ImageMotionPtkSmartRespawn<>(
                lowlevel, absoluteMinimumTracks, respawnTrackFraction, respawnCoverageFraction);

        return new ProbedMotion<>(new WrapImageMotionPtkSmartRespawn<>(smartRespawn), lowlevel);
    }

    /** The two models that support refinement, and therefore the two the probe covers. */
    public enum ProbeModel { HOMOGRAPHY, AFFINE }

    /**
     * The similarity-model equivalent of {@code FactoryMotion2D.createVideoStitch} for
     * {@link GrayF32}.
     *
     * <p>Needed because {@code createVideoStitch} picks its {@code StitchingTransform} by testing
     * the transform type and falls through to the homography variant for anything it does not
     * recognise, which would fail on a {@code Sim2_F64}. Every other line is transliterated from
     * BoofCV 0.44: the same bilinear interpolation with {@code BorderType.EXTENDED} over
     * {@code [0, 255]}, the same {@code FactoryDistort.distort(false, interp, imageType)}, the same
     * {@code setRenderAll(false)}, the same {@code StitchingFromMotion2D} constructor arguments.
     *
     * <p>Public because {@code VoRunnerApp} must build the same stack for the config-driven capture
     * path, and a second copy of this wiring is exactly what {@code DEC-VO-002} rejected.
     */
    public static StitchingFromMotion2D<GrayF32, Sim2_F64> similarityVideoStitch(
            double maxJumpFraction, ImageMotion2D<GrayF32, Sim2_F64> motion2D) {
        InterpolatePixel<GrayF32> interp = FactoryInterpolation.createPixelS(
                0, 255, InterpolationType.BILINEAR, BorderType.EXTENDED, GrayF32.class);
        ImageDistort<GrayF32, GrayF32> distorter =
                FactoryDistort.distort(false, interp, ImageType.single(GrayF32.class));
        distorter.setRenderAll(false);
        return new StitchingFromMotion2D<>(
                motion2D, distorter, new SimilarityStitchingTransform(), maxJumpFraction);
    }

    public static class Builder {
        private ConfigPointDetector configDetector = defaultConfigDetector();
        private int kltPyramidLevels = 4;
        private int kltTemplateRadius = 3;

        // Motion2D parameters
        private int motionMaxIterations = 220;
        private double inlierThresholdSq = 3.0;
        private int outlierPrune = 2;
        private int motionMinFeatures = 30;
        private double respawnTrackFraction = 0.6;
        private double respawnCoverageFraction = 0.5;
        private boolean refineEstimate = false;

        // Stitching parameters
        private double stitchOverlapThreshold = 0.55;

        public Builder detector(ConfigPointDetector detector) {
            this.configDetector = detector;
            return this;
        }

        public Builder kltLevels(int levels) {
            this.kltPyramidLevels = levels;
            return this;
        }

        public Builder kltRadius(int radius) {
            this.kltTemplateRadius = radius;
            return this;
        }

        public Builder motionMaxIterations(int maxIterations) {
            this.motionMaxIterations = maxIterations;
            return this;
        }

        public Builder inlierThresholdSq(double inlierThresholdSq) {
            this.inlierThresholdSq = inlierThresholdSq;
            return this;
        }

        public Builder outlierPrune(int outlierPrune) {
            this.outlierPrune = outlierPrune;
            return this;
        }

        public Builder motionMinFeatures(int minFeatures) {
            this.motionMinFeatures = minFeatures;
            return this;
        }

        public Builder respawnTrackFraction(double respawnTrackFraction) {
            this.respawnTrackFraction = respawnTrackFraction;
            return this;
        }

        public Builder respawnCoverageFraction(double respawnCoverageFraction) {
            this.respawnCoverageFraction = respawnCoverageFraction;
            return this;
        }

        public Builder refineEstimate(boolean refineEstimate) {
            this.refineEstimate = refineEstimate;
            return this;
        }

        public Builder stitchOverlapThreshold(double overlapThreshold) {
            this.stitchOverlapThreshold = overlapThreshold;
            return this;
        }

        public StitchingFromMotion2D<Planar<GrayF32>, Homography2D_F64> buildPlanar() {
            PointTracker<GrayF32> tracker = FactoryPointTracker.klt(
                    kltPyramidLevels, configDetector, kltTemplateRadius, GrayF32.class, GrayF32.class
            );

            ImageMotion2D<GrayF32, Homography2D_F64> motion2D = FactoryMotion2D.createMotion2D(
                    motionMaxIterations, inlierThresholdSq, outlierPrune, motionMinFeatures,
                    respawnTrackFraction, respawnCoverageFraction, refineEstimate, tracker, new Homography2D_F64()
            );

            ImageMotion2D<Planar<GrayF32>, Homography2D_F64> motion2DColor =
                    new PlToGrayMotion2D<>(motion2D, GrayF32.class);

            return FactoryMotion2D.createVideoStitch(
                    stitchOverlapThreshold, motion2DColor, ImageType.pl(3, GrayF32.class)
            );
        }

        // NOTE (research log 2026-08-14, gap G17): buildGray/buildPlanar both call the deprecated
        // FactoryPointTracker.klt(numLevels, configDetect, featureRadius, ...) overload, which
        // does NOT copy configDetector.general.maxFeatures into the tracker's own
        // ConfigPKlt.maximumTracks. BoofCV's PointTrackerKltPyramid.spawnTracks() caps every
        // respawn at ConfigPKlt.maximumTracks instead (default 0.2% of frame pixels, min 50), so
        // `maxFeatures`/`detectorRadius`/`detectorThreshold` set here only shape the detector's
        // internal candidate ranking, not the number of tracks actually maintained. Confirmed
        // empirically: observed track_count ceilings are exactly floor(0.002 * width * height)
        // (655 at 640x512; 10027 at 2448x2048, EXP-002's MARS-LVIG imagery). See COMP-001 s3.1.
        public StitchingFromMotion2D<GrayF32, Homography2D_F64> buildGray() {
            PointTracker<GrayF32> tracker = FactoryPointTracker.klt(
                    kltPyramidLevels, configDetector, kltTemplateRadius, GrayF32.class, GrayF32.class
            );

            ImageMotion2D<GrayF32, Homography2D_F64> motion2D = FactoryMotion2D.createMotion2D(
                    motionMaxIterations, inlierThresholdSq, outlierPrune, motionMinFeatures,
                    respawnTrackFraction, respawnCoverageFraction, refineEstimate, tracker, new Homography2D_F64()
            );

            return FactoryMotion2D.createVideoStitch(
                    stitchOverlapThreshold, motion2D, ImageType.single(GrayF32.class)
            );
        }

        /**
         * The motion estimator alone, with no mosaic attached — the navigation-only path
         * ({@code DEC-VO-003}).
         *
         * <p>Since navigation state is derived from the motion model rather than from the stitched
         * raster, a caller that does not need a mosaic can drive this directly and never allocate,
         * render into, or re-origin a canvas. Same parameters, same tracker construction and same
         * hardcoded RANSAC seed as {@link #buildGray()}; the only difference is the absence of
         * {@code StitchingFromMotion2D}.
         *
         * <p><b>One behavioural consequence to be aware of</b>, and the reason this is not simply a
         * faster equivalent: {@code StitchingFromMotion2D.process} additionally runs
         * {@code checkLargeMotion}, an area-change fault check, and its canvas re-origin calls
         * {@code setToFirst()} which re-anchors tracks. A motion-only run therefore has neither the
         * area fault check nor those periodic track re-anchorings, so its estimates diverge from a
         * mosaic-backed run over time. See {@code MotionOnlyNavigationEstimator}.
         */
        public ImageMotion2D<GrayF32, Homography2D_F64> buildGrayMotionOnly() {
            PointTracker<GrayF32> tracker = FactoryPointTracker.klt(
                    kltPyramidLevels, configDetector, kltTemplateRadius, GrayF32.class, GrayF32.class
            );
            return FactoryMotion2D.createMotion2D(
                    motionMaxIterations, inlierThresholdSq, outlierPrune, motionMinFeatures,
                    respawnTrackFraction, respawnCoverageFraction, refineEstimate, tracker,
                    new Homography2D_F64());
        }

        /**
         * As {@link #buildGrayMotionOnly()}, with the 4-DoF similarity model
         * ({@code DEC-VO-006}) — the physically exact model for a nadir camera over a plane
         * ({@code LIT-VO-003} §2), estimated directly rather than projected from a richer fit.
         */
        public ImageMotion2D<GrayF32, Sim2_F64> buildGraySimilarityMotionOnly() {
            PointTracker<GrayF32> tracker = FactoryPointTracker.klt(
                    kltPyramidLevels, configDetector, kltTemplateRadius, GrayF32.class, GrayF32.class
            );
            return createSimilarityMotion2D(
                    motionMaxIterations, inlierThresholdSq, outlierPrune, motionMinFeatures,
                    respawnTrackFraction, respawnCoverageFraction, tracker);
        }

        /**
         * <b>Experimental — {@code EXP-VO-009} only.</b> As {@link #buildGray()}, but with the 4-DoF
         * {@link Sim2_F64} similarity model ({@code DEC-VO-006}) in place of the shipped 8-DoF
         * {@code Homography2D_F64}. Nothing in production calls this.
         *
         * <p>The tracker is built by the identical call, and every RANSAC/respawn/stitching argument
         * is the same value {@link #buildGray()} and {@link #buildGrayAffine()} pass. The model
         * substitutes four components — {@link ModelManagerSim2_F64}, {@link GenerateSimilarity2D}
         * (minimal sample <b>3</b>, matching affine so the two arms are RANSAC-sample-paired),
         * {@link DistanceSimilarity2DSq}, and {@link SimilarityStitchingTransform} — structurally
         * parallel to the four the affine arm substitutes.
         *
         * <p>Critically for interpretability, {@link DistanceSimilarity2DSq} returns squared
         * Euclidean pixel reprojection error just as {@code DistanceAffine2DSq} and
         * {@code DistanceHomographySq} do, so {@link #inlierThresholdSq(double)} means the same
         * thing in all three arms.
         *
         * <p><b>Note this arm carries no refiner even if {@link #refineEstimate(boolean)} is set.</b>
         * The other two arms would gain one; keeping it absent holds the estimator's
         * minimal-sample behaviour fixed across arms, which is the variable {@code LIT-VO-005} §4
         * identifies as potentially dominant and {@code EXP-VO-009} therefore refuses to move.
         */
        public StitchingFromMotion2D<GrayF32, Sim2_F64> buildGraySimilarity() {
            return similarityVideoStitch(stitchOverlapThreshold, buildGraySimilarityMotionOnly());
        }

        /** As {@link #buildGrayMotionOnly()}, with the 6-DoF affine model ({@code DEC-VO-002}). */
        public ImageMotion2D<GrayF32, Affine2D_F64> buildGrayAffineMotionOnly() {
            PointTracker<GrayF32> tracker = FactoryPointTracker.klt(
                    kltPyramidLevels, configDetector, kltTemplateRadius, GrayF32.class, GrayF32.class
            );
            return FactoryMotion2D.createMotion2D(
                    motionMaxIterations, inlierThresholdSq, outlierPrune, motionMinFeatures,
                    respawnTrackFraction, respawnCoverageFraction, refineEstimate, tracker,
                    new Affine2D_F64());
        }

        /**
         * <b>Experimental — {@code EXP-VO-002} only.</b> As {@link #buildGray()}, but with BoofCV's
         * 6-DoF {@code Affine2D_F64} motion model in place of the shipped 8-DoF
         * {@code Homography2D_F64}. Nothing in production calls this.
         *
         * <p>Every argument to {@code FactoryMotion2D.createMotion2D} is the same value
         * {@link #buildGray()} passes, and the tracker is built by the identical call. Verified
         * against BoofCV 0.44 source ({@code LIT-VO-001}), the factory's affine branch differs from
         * its homography branch in exactly four places, all forced by the model rather than chosen:
         * {@code ModelManagerAffine2D_F64}, {@code GenerateAffine2D} (minimal sample <b>3</b>, vs 4
         * for the DLT), {@code DistanceAffine2DSq}, and — inside {@code createVideoStitch} —
         * {@code FactoryStitchingTransform.createAffine_F64()}. Everything else (the {@code Ransac}
         * class, its hardcoded seed 123123, the iteration count, the inlier threshold,
         * {@code outlierPrune}, {@code ImageMotionPointTrackerKey},
         * {@code ImageMotionPtkSmartRespawn} and all three respawn parameters,
         * {@code maxJumpFraction}) is shared code reached with identical arguments.
         *
         * <p>Critically for interpretability, {@code DistanceAffine2DSq} and
         * {@code DistanceHomographySq} both return squared Euclidean pixel reprojection error, so
         * {@link #inlierThresholdSq(double)} means the same thing in both arms.
         */
        public StitchingFromMotion2D<GrayF32, Affine2D_F64> buildGrayAffine() {
            PointTracker<GrayF32> tracker = FactoryPointTracker.klt(
                    kltPyramidLevels, configDetector, kltTemplateRadius, GrayF32.class, GrayF32.class
            );

            ImageMotion2D<GrayF32, Affine2D_F64> motion2D = FactoryMotion2D.createMotion2D(
                    motionMaxIterations, inlierThresholdSq, outlierPrune, motionMinFeatures,
                    respawnTrackFraction, respawnCoverageFraction, refineEstimate, tracker,
                    new Affine2D_F64()
            );

            return FactoryMotion2D.createVideoStitch(
                    stitchOverlapThreshold, motion2D, ImageType.single(GrayF32.class)
            );
        }

        /**
         * As {@link #buildPlanar()}, but additionally exposing the reprojection residuals of the
         * very motion estimator inside the returned stitcher.
         *
         * <p>Behaviourally identical to {@link #buildPlanar()} under identical configuration: same
         * tracker, same RANSAC seed and parameters, same respawn logic, same poses. The diagnostics
         * are observational — no second image pyramid, no second RANSAC or homography estimation
         * (feature spec FR-009). Callers that do not want diagnostics should keep using
         * {@link #buildPlanar()}, which is untouched.
         */
        public InstrumentedStitching<Planar<GrayF32>> buildPlanarInstrumented() {
            PointTracker<GrayF32> tracker = FactoryPointTracker.klt(
                    kltPyramidLevels, configDetector, kltTemplateRadius, GrayF32.class, GrayF32.class
            );

            Instrumented<GrayF32> instrumented = createInstrumentedMotion2D(
                    motionMaxIterations, inlierThresholdSq, outlierPrune, motionMinFeatures,
                    respawnTrackFraction, respawnCoverageFraction, refineEstimate, tracker
            );

            ImageMotion2D<Planar<GrayF32>, Homography2D_F64> motion2DColor =
                    new PlToGrayMotion2D<>(instrumented.motion(), GrayF32.class);

            StitchingFromMotion2D<Planar<GrayF32>, Homography2D_F64> stitch =
                    FactoryMotion2D.createVideoStitch(
                            stitchOverlapThreshold, motion2DColor, ImageType.pl(3, GrayF32.class)
                    );

            return new InstrumentedStitching<>(stitch, instrumented.diagnostics());
        }

        /**
         * As {@link #buildGray()}, but additionally exposing the reprojection residuals of the very
         * motion estimator inside the returned stitcher.
         *
         * <p>Behaviourally identical to {@link #buildGray()} under identical configuration — see
         * {@link #buildPlanarInstrumented()}.
         */
        public InstrumentedStitching<GrayF32> buildGrayInstrumented() {
            PointTracker<GrayF32> tracker = FactoryPointTracker.klt(
                    kltPyramidLevels, configDetector, kltTemplateRadius, GrayF32.class, GrayF32.class
            );

            Instrumented<GrayF32> instrumented = createInstrumentedMotion2D(
                    motionMaxIterations, inlierThresholdSq, outlierPrune, motionMinFeatures,
                    respawnTrackFraction, respawnCoverageFraction, refineEstimate, tracker
            );

            StitchingFromMotion2D<GrayF32, Homography2D_F64> stitch =
                    FactoryMotion2D.createVideoStitch(
                            stitchOverlapThreshold, instrumented.motion(),
                            ImageType.single(GrayF32.class)
                    );

            return new InstrumentedStitching<>(stitch, instrumented.diagnostics());
        }
    }
}
