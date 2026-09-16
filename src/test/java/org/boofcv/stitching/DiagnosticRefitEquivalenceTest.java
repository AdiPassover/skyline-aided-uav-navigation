package org.boofcv.stitching;

import boofcv.struct.image.GrayF32;
import georegression.struct.homography.Homography2D_F64;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@code EXP-CONF-004}'s standard-of-evidence test: attaching the observer-only diagnostic
 * refiner changes <b>nothing</b> the estimator ships — capture-off and capture-on runs are
 * bitwise identical in success/recenter events, poses and world transforms — while the probe
 * itself reports a genuine all-inlier refit that differs from the shipped minimal-sample model.
 *
 * <p>Same harness and same zero-delta standard as {@code RefinementSemanticsTest}: both stacks
 * are deterministic (BoofCV's hardcoded RANSAC seed), so any difference is a real behavioural
 * change, and the frozen configuration ({@code refineEstimate = false}) must remain exactly the
 * minimal/unrefined transform. The full-scale companion check compares the diagnostic runs'
 * persisted outputs byte-for-byte against the committed frozen run records (EXP-CONF-004
 * Phase 1).
 */
class DiagnosticRefitEquivalenceTest {

    @Test
    void diagnosticRefitObserverChangesNothingShipped() {
        GrayF32 world = EquivalenceHarness.syntheticTexturedWorld();

        StitchingFactory.ProbedMotion<?> plain =
                EquivalenceHarness.probed(StitchingFactory.ProbeModel.HOMOGRAPHY, false);
        StitchingFactory.ProbedMotion<?> observed =
                EquivalenceHarness.probedWithDiagnosticRefit(
                        StitchingFactory.ProbeModel.HOMOGRAPHY, false);

        List<EquivalenceHarness.Observation> a = EquivalenceHarness.run(
                plain.motion(), StitchingFactory.ProbeModel.HOMOGRAPHY, world);
        List<EquivalenceHarness.Observation> b = EquivalenceHarness.run(
                observed.motion(), StitchingFactory.ProbeModel.HOMOGRAPHY, world);

        assertEquals(a.size(), b.size());
        for (int i = 0; i < a.size(); i++) {
            EquivalenceHarness.assertIdentical("frame " + i, a.get(i), b.get(i));
        }
    }

    @Test
    void diagnosticRefitIsComputedAndIsNotTheShippedModel() {
        GrayF32 world = EquivalenceHarness.syntheticTexturedWorld();
        StitchingFactory.ProbedMotion<?> observed =
                EquivalenceHarness.probedWithDiagnosticRefit(
                        StitchingFactory.ProbeModel.HOMOGRAPHY, false);

        // Drive frames through the estimator stack; read the probe after the last frame — the
        // snapshot semantics guarantee the values belong to that frame.
        EquivalenceHarness.run(observed.motion(), StitchingFactory.ProbeModel.HOMOGRAPHY, world);
        RefinementDiagnostics d = observed.diagnostics();

        assertTrue(d.diagnosticRefitEnabled(), "probe must carry the diagnostic refiner");
        assertTrue(d.hasModels(), "last frame should have an accepted estimate");
        assertTrue(d.hasDiagnosticRefit(), "diagnostic refit should have run on the last frame");
        assertTrue(d.diagnosticRefitTimeNs() > 0, "refit wall time must be measured");

        // With refineEstimate = false the shipped model IS the minimal-sample winner...
        Homography2D_F64 shipped = d.shippedModel();
        Homography2D_F64 minimal = d.minimalSampleModel();
        assertEquals(minimal.a11, shipped.a11, 0.0, "frozen configuration must ship the minimal model");
        assertEquals(minimal.a13, shipped.a13, 0.0, "frozen configuration must ship the minimal model");

        // ...and the all-inlier refit of hundreds of correspondences is a different estimate.
        Homography2D_F64 refit = d.diagnosticRefitModel();
        boolean differs = shipped.a11 != refit.a11 || shipped.a12 != refit.a12
                || shipped.a13 != refit.a13 || shipped.a21 != refit.a21
                || shipped.a22 != refit.a22 || shipped.a23 != refit.a23;
        assertTrue(differs, "an all-inlier refit over a large noisy match set should not "
                + "reproduce the minimal-sample hypothesis bit-for-bit");
        assertFalse(d.refinementEnabled(), "the estimator's own refiner must remain absent");
    }
}
