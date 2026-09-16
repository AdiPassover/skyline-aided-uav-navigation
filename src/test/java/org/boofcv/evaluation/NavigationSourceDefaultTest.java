package org.boofcv.evaluation;

import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.struct.image.GrayF32;
import georegression.struct.homography.Homography2D_F64;
import org.boofcv.stitching.MotionModelStitchingEstimator;
import org.boofcv.stitching.MotionModelSupport;
import org.boofcv.stitching.NavigationSource;
import org.boofcv.stitching.StitchingFactory;
import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins {@code DEC-VO-008}: the software default navigation source is {@code RIGID_MOTION}, the
 * other two remain selectable, and there is exactly <b>one</b> place the default is written down.
 *
 * <p>Written because the promotion is the kind of change that can be half-made. Before it there
 * were three copies of "the default is legacy" — {@code MotionModelStitchingEstimator}'s field
 * initialiser, {@code VoRunnerConfig}'s string, and {@code VoRunnerApp.parseNavigationSource}'s
 * null/blank branch — and moving two of the three would have left a config carrying an explicit
 * {@code "navigation_source": null} still running the old readout, silently, with the run
 * manifest recording it correctly and nobody looking. All three now resolve through
 * {@link NavigationSource#DEFAULT}, and this test asserts they agree.
 *
 * <p>It also asserts the two things the promotion promised <i>not</i> to break: an explicit
 * {@code mosaic_legacy} still selects the legacy computation, and the default estimator does not
 * silently enable anything metric — {@code z} stays unestimated, because a height source is a
 * separate opt-in layer ({@code DEC-VO-007}) that this selector neither performs nor requires.
 *
 * <p>Evidence tier: T1 (synthetic). Says nothing about which readout navigates better — that is
 * {@code EXP-VO-006}/{@code EXP-VO-007}/{@code EXP-VO-008}, on real data.
 */
public class NavigationSourceDefaultTest {

    private static final int WORLD = 512;
    private static final int FRAME = 320;
    private static final int FRAMES = 40;

    // ---------------------------------------------------------------- the default, in three places

    @Test
    public void theDeclaredDefaultIsRigidMotion() {
        assertSame(NavigationSource.RIGID_MOTION, NavigationSource.DEFAULT,
                "DEC-VO-008 promoted RIGID_MOTION to the software default");
    }

    @Test
    public void freshEstimatorUsesTheDeclaredDefault() {
        assertEquals(NavigationSource.DEFAULT, freshEstimator().getNavigationSource());
    }

    @Test
    public void freshConfigResolvesToTheDeclaredDefault() {
        assertEquals(NavigationSource.DEFAULT,
                VoRunnerApp.parseNavigationSource(new VoRunnerConfig().navigationSource));
    }

    @Test
    public void absentOrBlankNavigationSourceResolvesToTheDeclaredDefault() {
        assertEquals(NavigationSource.DEFAULT, VoRunnerApp.parseNavigationSource(null));
        assertEquals(NavigationSource.DEFAULT, VoRunnerApp.parseNavigationSource(""));
        assertEquals(NavigationSource.DEFAULT, VoRunnerApp.parseNavigationSource("   "));
    }

    // ---------------------------------------------------------------- what stayed selectable

    @Test
    public void allThreeSourcesRemainSelectableByName() {
        assertEquals(NavigationSource.MOSAIC_LEGACY, VoRunnerApp.parseNavigationSource("mosaic_legacy"));
        assertEquals(NavigationSource.LOGICAL_FRAME, VoRunnerApp.parseNavigationSource("logical_frame"));
        assertEquals(NavigationSource.RIGID_MOTION, VoRunnerApp.parseNavigationSource("rigid_motion"));
        assertThrows(IllegalArgumentException.class,
                () -> VoRunnerApp.parseNavigationSource("mosaic-legacy"));
    }

    /**
     * The promotion changed which computation publishes, and this asserts it actually did — an
     * estimator left at the default and one pinned to {@code MOSAIC_LEGACY} must disagree on real
     * tracked imagery. Without it, every other assertion here would pass just as happily if the
     * selector had quietly stopped being wired to the readout at all.
     */
    @Test
    public void theDefaultPublishesADifferentTrajectoryFromExplicitLegacy() {
        GrayF32 world = texturedWorld();
        List<Pose3D> byDefault = run(freshEstimator(), world);
        List<Pose3D> legacy = run(pinned(NavigationSource.MOSAIC_LEGACY), world);

        assertEquals(byDefault.size(), legacy.size());
        double maxGap = 0;
        for (int i = 0; i < byDefault.size(); i++) {
            maxGap = Math.max(maxGap, Math.hypot(byDefault.get(i).x - legacy.get(i).x,
                                                 byDefault.get(i).y - legacy.get(i).y));
        }
        assertTrue(maxGap > 1e-6,
                "the default and explicit legacy produced identical poses (" + maxGap
                        + " px) -- the navigation-source selector is not wired to the readout");
    }

    /** The default must be the rigid readout itself, not merely something that differs from legacy. */
    @Test
    public void theDefaultPublishesTheRigidPoseExactly() {
        GrayF32 world = texturedWorld();
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est = freshEstimator();
        for (int i = 0; i < FRAMES; i++) {
            est.processFrame(frameAt(world, i));
            Pose3D published = est.getCurrentPose();
            Pose3D rigid = est.getRigidPose();
            assertEquals(rigid.x, published.x, 0.0, "frame " + i);
            assertEquals(rigid.y, published.y, 0.0, "frame " + i);
            assertEquals(rigid.yaw, published.yaw, 0.0, "frame " + i);
        }
    }

    @Test
    public void explicitLegacyStillPublishesTheLegacyComputation() {
        GrayF32 world = texturedWorld();
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est =
                pinned(NavigationSource.MOSAIC_LEGACY);
        boolean diverged = false;
        for (int i = 0; i < FRAMES; i++) {
            est.processFrame(frameAt(world, i));
            // The legacy pose has no getter, so it is identified negatively: it is the published
            // pose, and it is NOT the rigid one -- which is still maintained on every frame.
            if (Math.hypot(est.getCurrentPose().x - est.getRigidPose().x,
                           est.getCurrentPose().y - est.getRigidPose().y) > 1e-6) {
                diverged = true;
            }
        }
        assertTrue(diverged, "explicit mosaic_legacy published the rigid pose");
        assertNotEquals(NavigationSource.DEFAULT, est.getNavigationSource());
    }

    // ---------------------------------------------------------------- what the promotion must NOT do

    /**
     * {@code DEC-VO-007}'s metric layer is opt-in and lives outside the estimator entirely.
     * Promoting {@code RIGID_MOTION} must not have dragged any of it in: {@code z} is still never
     * estimated, so no barometer or height source can have become implicitly required.
     */
    @Test
    public void theDefaultEnablesNothingMetric() {
        GrayF32 world = texturedWorld();
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est = freshEstimator();
        for (int i = 0; i < FRAMES; i++) {
            est.processFrame(frameAt(world, i));
            assertEquals(0.0, est.getCurrentPose().z, 0.0, "z must never be estimated, frame " + i);
        }
    }

    // ---------------------------------------------------------------- helpers

    private static MotionModelStitchingEstimator<GrayF32, Homography2D_F64> freshEstimator() {
        StitchingFromMotion2D<GrayF32, Homography2D_F64> stitch = StitchingFactory.builder().buildGray();
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est =
                new MotionModelStitchingEstimator<>(stitch, MotionModelSupport.HOMOGRAPHY);
        est.setShrinkScale(0.5);
        est.setMinDistanceFromBorder(10);
        return est;                      // deliberately no setNavigationSource -- that is the point
    }

    private static MotionModelStitchingEstimator<GrayF32, Homography2D_F64> pinned(NavigationSource s) {
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est = freshEstimator();
        est.setNavigationSource(s);
        return est;
    }

    private static List<Pose3D> run(MotionModelStitchingEstimator<GrayF32, Homography2D_F64> est,
                                    GrayF32 world) {
        List<Pose3D> out = new ArrayList<>(FRAMES);
        for (int i = 0; i < FRAMES; i++) {
            est.processFrame(frameAt(world, i));
            out.add(est.getCurrentPose());
        }
        return out;
    }

    /** Deterministic high-frequency texture, so KLT has something to hold on to. */
    private static GrayF32 texturedWorld() {
        GrayF32 img = new GrayF32(WORLD, WORLD);
        Random rng = new Random(7);
        for (int y = 0; y < WORLD; y++) {
            for (int x = 0; x < WORLD; x++) {
                img.set(x, y, (float) (rng.nextDouble() * 255.0));
            }
        }
        return img;
    }

    /** A diagonal pan across the world, one pixel per axis per frame. */
    private static GrayF32 frameAt(GrayF32 world, int index) {
        GrayF32 out = new GrayF32(FRAME, FRAME);
        int ox = 40 + index, oy = 40 + index;
        for (int y = 0; y < FRAME; y++) {
            for (int x = 0; x < FRAME; x++) {
                out.set(x, y, world.get(Math.min(ox + x, WORLD - 1), Math.min(oy + y, WORLD - 1)));
            }
        }
        return out;
    }
}
