package org.boofcv.stitching;

import boofcv.abst.feature.detect.interest.ConfigPointDetector;
import boofcv.struct.image.GrayF32;
import org.boofcv.evaluation.VoDiagnostics;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins gap G17 (research log 2026-08-14): {@code FactoryPointTracker.klt}'s deprecated overload,
 * which {@link StitchingFactory.Builder#buildGray()}/{@code buildPlanar()} both call, does NOT
 * propagate {@code configDetector.general.maxFeatures} into the KLT tracker's own
 * {@code ConfigPKlt.maximumTracks}. BoofCV's {@code PointTrackerKltPyramid.spawnTracks()} caps
 * every respawn at {@code ConfigPKlt.maximumTracks} instead (default
 * {@code ConfigLength.relative(0.002, 50)}: 0.2% of the frame's pixel count, minimum 50) — not at
 * the configured {@code maxFeatures}.
 *
 * <p>Found while diagnosing an EXP-002 A0 plumbing run where {@code track_count} reached 10,027
 * against a configured {@code maxFeatures = 300} at 2448 x 2448x2048 resolution — exactly
 * {@code floor(0.002 * 2448 * 2048)}. The 640x512 baseline used elsewhere in this repository
 * shows the same relationship: {@code floor(0.002 * 640 * 512) = 655}, matching the ceiling
 * observed in {@code runs/sim-square-run-v1/frames.csv}.
 *
 * <p>This is a characterisation test of existing (unintended) behaviour, not a specification of
 * desired behaviour. If it starts failing after a {@code StitchingFactory} change, that change
 * fixed the underlying issue and this test — together with {@code COMP-001} section 3.1 and
 * {@code VoRunnerConfig}'s {@code maxFeatures} javadoc — should be updated to match, not silently
 * adjusted to keep passing.
 */
public class StitchingFactoryTrackCapTest {

    private static final int WORLD_SIZE = 400;
    private static final int FRAME_SIZE = 300; // area 90,000 -> BoofCV's own ceiling is 180

    private static GrayF32 syntheticTexturedWorld() {
        GrayF32 world = new GrayF32(WORLD_SIZE, WORLD_SIZE);
        Random random = new Random(17);
        for (int y = 0; y < WORLD_SIZE; y++) {
            for (int x = 0; x < WORLD_SIZE; x++) {
                world.set(x, y, random.nextInt(256));
            }
        }
        return world;
    }

    private static GrayF32 crop(GrayF32 world, int offsetX, int offsetY) {
        GrayF32 frame = new GrayF32(FRAME_SIZE, FRAME_SIZE);
        for (int y = 0; y < FRAME_SIZE; y++) {
            for (int x = 0; x < FRAME_SIZE; x++) {
                frame.set(x, y, world.get(x + offsetX, y + offsetY));
            }
        }
        return frame;
    }

    @Test
    public void observedTrackCeilingIsBoofcvDefaultNotConfiguredMaxFeatures() {
        // Deliberately far below the expected BoofCV-default ceiling of floor(0.002*300*300)=180,
        // so exceeding it is unambiguous evidence maxFeatures is not the effective cap.
        int configuredMaxFeatures = 50;

        ConfigPointDetector configDetector = StitchingFactory.defaultConfigDetector();
        configDetector.general.maxFeatures = configuredMaxFeatures;

        StitchingEstimator<GrayF32> estimator = new StitchingEstimator<>(
                StitchingFactory.builder().detector(configDetector).buildGray());
        estimator.setShrinkScale(0.5);
        estimator.setMinDistanceFromBorder(10);

        GrayF32 world = syntheticTexturedWorld();
        int maxObservedTrackCount = 0;
        for (int frameIdx = 0; frameIdx < 6; frameIdx++) {
            GrayF32 frame = crop(world, frameIdx * 3, frameIdx * 2);
            estimator.processFrame(frame);
            VoDiagnostics.Counts counts =
                    VoDiagnostics.extract(estimator.getStitch().getMotion());
            if (counts.trackCount() != null) {
                maxObservedTrackCount = Math.max(maxObservedTrackCount, counts.trackCount());
            }
        }

        int boofcvDefaultCeiling = (int) (FRAME_SIZE * FRAME_SIZE * 0.002); // ConfigPKlt default

        assertTrue(maxObservedTrackCount > configuredMaxFeatures,
                "expected the observed track count (" + maxObservedTrackCount + ") to exceed the "
                        + "configured maxFeatures (" + configuredMaxFeatures + "); if this now "
                        + "fails, maxFeatures has started being respected and G17's documentation "
                        + "(COMP-001 s3.1, VoRunnerConfig, StitchingFactory) should be updated");
        assertTrue(maxObservedTrackCount <= boofcvDefaultCeiling,
                "expected the observed track count (" + maxObservedTrackCount + ") to stay at or "
                        + "below BoofCV's own ConfigPKlt.maximumTracks default ("
                        + boofcvDefaultCeiling + " = floor(0.002 * " + FRAME_SIZE + " * "
                        + FRAME_SIZE + "))");
    }
}
