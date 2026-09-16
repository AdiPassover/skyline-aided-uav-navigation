package org.boofcv.stitching;

import org.boofcv.stitching.diagnostics.ScaleBiasMonteCarlo;
import org.boofcv.stitching.diagnostics.ScaleBiasMonteCarlo.Condition;
import org.boofcv.stitching.diagnostics.ScaleBiasMonteCarlo.Model;
import org.boofcv.stitching.diagnostics.ScaleBiasMonteCarlo.Relief;
import org.boofcv.stitching.diagnostics.ScaleBiasMonteCarlo.Result;
import org.boofcv.stitching.diagnostics.ScaleBiasMonteCarlo.Spread;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Known-answer gates for {@code EXP-VO-011}'s attribution harness. Every one of these must hold
 * before any mechanism arm is believed: an instrument that cannot return zero when the answer is
 * zero cannot be trusted to return {@code 3e-4} when it is not.
 */
class ScaleBiasMonteCarloTest {

    private static final int RANSAC_ITERATIONS = 220;
    private static final double THRESHOLD_SQ = 3.0;
    private static final double H = 80.0;
    private static final double TRAVEL = 0.30;

    private static Condition cond(String name, Model m, Relief r, double spread, double k1,
                                  boolean correct, double dH, double travel, double ang,
                                  double tilt, double sigma, Spread sp) {
        return cond(name, m, r, spread, k1, correct, dH, travel, ang, tilt, sigma, sp, 10);
    }

    private static Condition cond(String name, Model m, Relief r, double spread, double k1,
                                  boolean correct, double dH, double travel, double ang,
                                  double tilt, double sigma, Spread sp, int baseline) {
        return new Condition(name, m, r, spread, k1, 0.0, correct, H, dH, travel, ang, 0.0,
                tilt, 0.0, sigma, baseline, 600, sp);
    }

    private static Result run(Condition c, int trials) {
        return ScaleBiasMonteCarlo.flightGeometry()
                .measure(c, trials, 424242L, RANSAC_ITERATIONS, THRESHOLD_SQ);
    }

    @Test
    @DisplayName("planar, pinhole, noiseless, nadir: every model returns exactly zero log-scale bias")
    void idealCaseIsExactlyZero() {
        for (Model m : Model.values()) {
            Result r = run(cond("gate", m, Relief.PLANAR, 0, 0, false, 0, TRAVEL, 0, 0, 0,
                    Spread.DISTRIBUTED), 40);
            assertEquals(0, r.failures(), m + ": no trial may fail on the ideal case");
            assertTrue(Math.abs(r.biasLogScale()) < 1e-9,
                    m + ": ideal-case bias " + r.biasLogScale() + " exceeds 1e-9");
        }
    }

    @Test
    @DisplayName("a known height change is recovered exactly, so the instrument reads real scale")
    void knownHeightChangeIsRecovered() {
        // 0.03 m of climb at 80 m is log(80.03/80) = +3.75e-4 -- deliberately the same order as the
        // real per-frame bias this record has to explain, so the instrument is shown to resolve it.
        for (Model m : Model.values()) {
            Result r = run(cond("gate", m, Relief.PLANAR, 0, 0, false, 0.03, TRAVEL, 0, 0, 0,
                    Spread.DISTRIBUTED), 40);
            assertTrue(r.truthLogScale() > 3.7e-4 && r.truthLogScale() < 3.8e-4,
                    "truth should be +3.75e-4, was " + r.truthLogScale());
            assertTrue(Math.abs(r.biasLogScale()) < 1e-9,
                    m + ": climb recovered with bias " + r.biasLogScale());
        }
    }

    @Test
    @DisplayName("the sign convention is the pipeline's: climbing reads positive")
    void climbReadsPositive() {
        Result up = run(cond("up", Model.AFFINE, Relief.PLANAR, 0, 0, false, +1.0, TRAVEL, 0, 0, 0,
                Spread.DISTRIBUTED), 10);
        Result down = run(cond("down", Model.AFFINE, Relief.PLANAR, 0, 0, false, -1.0, TRAVEL, 0, 0,
                0, Spread.DISTRIBUTED), 10);
        assertTrue(up.truthLogScale() > 0, "a climb must read positive");
        assertTrue(down.truthLogScale() < 0, "a descent must read negative");
    }

    @Test
    @DisplayName("correcting the distortion that was applied restores the ideal answer exactly")
    void distortionCorrectionIsExact() {
        Result r = run(cond("corrected", Model.AFFINE, Relief.PLANAR, 0, -0.05, true, 0, TRAVEL, 0,
                0, 0, Spread.DISTRIBUTED), 30);
        assertTrue(Math.abs(r.biasLogScale()) < 1e-9,
                "rectified arm should be exact, bias was " + r.biasLogScale());
    }

    @Test
    @DisplayName("relief is generated at the requested in-footprint depth spread")
    void reliefSpreadIsCalibrated() {
        for (Relief kind : new Relief[]{Relief.SLOPE, Relief.RANDOM_FIELD, Relief.STRUCTURES}) {
            Result r = run(cond("relief", Model.AFFINE, kind, 22.0, 0, false, 0, TRAVEL, 0, 0, 0,
                    Spread.DISTRIBUTED), 20);
            assertTrue(r.actualSpreadM() > 14.0 && r.actualSpreadM() < 32.0,
                    kind + ": requested 22 m p95-p5, generated " + r.actualSpreadM());
        }
        Result flat = run(cond("flat", Model.AFFINE, Relief.PLANAR, 22.0, 0, false, 0, TRAVEL, 0, 0,
                0, Spread.DISTRIBUTED), 5);
        assertEquals(0.0, flat.actualSpreadM(), 1e-12, "PLANAR must be exactly flat");
    }

    @Test
    @DisplayName("distortion under pure translation flips sign with the travel direction")
    void distortionBiasIsOddInTravelDirection() {
        // The pre-registered derivation says the apparent scale goes as (centroid . travel), so
        // reversing the travel must reverse the sign. Measured on the clustered arm, where the
        // centroid offset -- and therefore the effect -- is largest.
        Result fwd = run(cond("d0", Model.AFFINE, Relief.PLANAR, 0, -0.05, false, 0, TRAVEL, 0, 0, 0,
                Spread.CLUSTERED), 60);
        Result back = run(cond("d180", Model.AFFINE, Relief.PLANAR, 0, -0.05, false, 0, TRAVEL, 180,
                0, 0, Spread.CLUSTERED), 60);
        assertTrue(Math.abs(fwd.biasLogScale()) > 1e-7, "the effect must be resolvable at all");
        assertTrue(fwd.biasLogScale() * back.biasLogScale() < 0,
                "distortion bias should reverse with travel direction: "
                        + fwd.biasLogScale() + " vs " + back.biasLogScale());
    }

    @Test
    @DisplayName("errors-in-variables noise biases the scale POSITIVE, as attenuation predicts")
    void noiseBiasesScalePositive() {
        // Attenuation of the fitted linear block shrinks D and therefore magnifies D^-1, so the
        // increment reads positive: the estimator reports a phantom climb. Large sigma so the sign
        // is unambiguous; the magnitude sweep is the experiment, this is the direction gate.
        Result r = run(cond("eiv", Model.AFFINE, Relief.PLANAR, 0, 0, false, 0, TRAVEL, 0, 0, 4.0,
                Spread.DISTRIBUTED), 60);
        assertTrue(r.biasLogScale() > 0,
                "attenuation must read as a phantom climb, was " + r.biasLogScale());
    }

    @Test
    @DisplayName("the similarity model reports structurally degenerate diagnostics")
    void similarityDiagnosticsAreDegenerate() {
        Result r = run(cond("sim", Model.SIMILARITY, Relief.RANDOM_FIELD, 22.0, 0, false, 0, TRAVEL,
                0, 0, 0, Spread.DISTRIBUTED), 15);
        assertEquals(1.0, r.meanAnisotropy(), 1e-9, "a similarity cannot be anisotropic");
        assertEquals(0.0, r.meanPerspective(), 1e-12, "a similarity has no perspective row");
    }

    @Test
    @DisplayName("an odd mechanism grows with the keyframe baseline, and vanishes at baseline 0")
    void oddMechanismGrowsWithKeyframeBaseline() {
        // The whole reason the harness models a keyframe rather than a consecutive pair: an
        // odd-in-image-position mechanism acts on a correspondence field displaced by the WHOLE
        // epoch. Measured on the clustered arm, where the effect is largest and unambiguous.
        double small = Math.abs(run(cond("b1", Model.AFFINE, Relief.PLANAR, 0, -0.05, false, 0,
                TRAVEL, 0, 0, 0, Spread.CLUSTERED, 1), 40).biasLogScale());
        double large = Math.abs(run(cond("b40", Model.AFFINE, Relief.PLANAR, 0, -0.05, false, 0,
                TRAVEL, 0, 0, 0, Spread.CLUSTERED, 40), 40).biasLogScale());
        assertTrue(large > small,
                "distortion bias should grow with the keyframe baseline: " + small + " -> " + large);
    }

    @Test
    @DisplayName("a symmetric lattice leaves an odd mechanism unresolvable; an asymmetric one does not")
    void latticeSeparatesTheOddMechanism() {
        // The odd-symmetry derivation says the apparent scale goes as (centroid . travel). On a
        // lattice whose centroid sits at the image centre that inner product is zero, so strong
        // distortion must leave no bias distinguishable from zero; confining the same points to one
        // quadrant moves the centroid 479 px and the effect appears. That contrast is what makes the
        // distortion arm a measurement rather than a scatter of Monte Carlo noise.
        //
        // The point SET is deterministic but RANSAC's minimal-sample draw is not, so the assertion
        // is against each arm's own standard error rather than against zero -- which is also exactly
        // how the record scores every cell of the grid.
        Result sym = run(cond("lat", Model.AFFINE, Relief.PLANAR, 0, -0.10, false, 0, TRAVEL, 0, 0,
                0, Spread.LATTICE), 120);
        Result asym = run(cond("latq", Model.AFFINE, Relief.PLANAR, 0, -0.10, false, 0, TRAVEL, 0, 0,
                0, Spread.LATTICE_QUADRANT), 120);
        assertTrue(Math.abs(sym.biasLogScale()) < 3 * sym.seLogScale(),
                "symmetric lattice must be indistinguishable from zero, got "
                        + sym.biasLogScale() + " against SE " + sym.seLogScale());
        assertTrue(Math.abs(asym.biasLogScale()) > 3 * asym.seLogScale(),
                "one-quadrant lattice must resolve the effect, got "
                        + asym.biasLogScale() + " against SE " + asym.seLogScale());
        assertTrue(Math.abs(asym.biasLogScale()) > 5 * Math.abs(sym.biasLogScale()),
                "the asymmetric arm must dominate the symmetric one: "
                        + sym.biasLogScale() + " vs " + asym.biasLogScale());
    }

    @Test
    @DisplayName("corner distortion is reported in pixels, so k1 is physically legible")
    void cornerDistortionIsReported() {
        ScaleBiasMonteCarlo mc = ScaleBiasMonteCarlo.flightGeometry();
        double px = mc.cornerDistortionPx(-0.02, 0);
        // r_corner/f = hypot(612, 512)/735.53 = 1.0847; |k1| r^3 f = 0.02 * 1.276 * 735.53
        assertTrue(px > 15 && px < 25, "expected ~19 px of corner barrel at k1=-0.02, got " + px);
    }
}
