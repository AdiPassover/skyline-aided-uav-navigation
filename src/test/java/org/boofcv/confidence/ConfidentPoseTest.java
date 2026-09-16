package org.boofcv.confidence;

import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A pose cannot be obtained without its verdict (feature spec FR-007).
 *
 * <p>Constitution Principle X's "low-confidence estimates MUST NOT be silently consumed" is made
 * structural here rather than advisory. What is asserted is precisely that the verdict is
 * unavoidable at the interface — <em>not</em> that misuse is impossible, which no type system can
 * guarantee and which would be a false claim.
 *
 * <p>Evidence tier: <b>T1 (analytical)</b>.
 */
public class ConfidentPoseTest {

    private static final SignalBlock SIGNALS = SignalBlock.builder()
            .trackCount(200).inlierCount(180).build();

    private static ConfidenceResult verdict(ConfidenceOutcome outcome, ConfidenceReason reason, Double score) {
        return new ConfidenceResult(outcome, reason, score, SIGNALS, "test-unvalidated", "sha256:test");
    }

    @Test
    public void everyAccessorYieldingAPoseAlsoCarriesTheVerdict() throws Exception {
        // The structural claim, asserted structurally: no public method returns a bare Pose3D.
        // A future refactor adding a convenience getPose() would fail here, which is the point --
        // FR-007 is a property of the API surface, not of any one call site.
        for (Method m : ConfidentPose.class.getDeclaredMethods()) {
            if (!Modifier.isPublic(m.getModifiers())) {
                continue;
            }
            assertFalse(Pose3D.class.equals(m.getReturnType()),
                    "public method " + m.getName() + " returns a bare Pose3D, "
                            + "which would let a caller take the pose without its verdict (FR-007)");
        }
    }

    @Test
    public void poseIsReachableOnlyThroughAnOptionalAlongsideItsVerdict() {
        Pose3D pose = new Pose3D(1.0, 2.0, 0.0, 30.0);
        ConfidentPose cp = ConfidentPose.of(pose, verdict(ConfidenceOutcome.USABLE, ConfidenceReason.OK, 0.9));

        assertSame(pose, cp.pose().orElseThrow());
        assertEquals(ConfidenceOutcome.USABLE, cp.confidence().outcome());
        assertSame(pose, cp.consumablePose().orElseThrow());
    }

    @Test
    public void degradedPoseIsStillConsumableButCarriesItsLowScore() {
        // DEGRADED is a low number attached to a real estimate, not a refusal. Erasing the pose
        // here would conflate it with REJECTED, which is the distinction Principle VIII asks for.
        Pose3D pose = new Pose3D(1.0, 2.0, 0.0, 30.0);
        ConfidentPose cp = ConfidentPose.of(pose,
                verdict(ConfidenceOutcome.DEGRADED, ConfidenceReason.LOW_SCORE, 0.2));

        assertTrue(cp.consumablePose().isPresent());
        assertEquals(0.2, cp.confidence().score(), 0.0);
    }

    @Test
    public void rejectedPoseExistsForInspectionButIsNotConsumable() {
        // A rejected estimate is still evidence about the frame. Deleting it would destroy that
        // evidence, which is the opposite of what a rejection is for -- so pose() is present while
        // consumablePose() is empty.
        Pose3D pose = new Pose3D(1.0, 2.0, 0.0, 30.0);
        ConfidentPose cp = ConfidentPose.of(pose,
                verdict(ConfidenceOutcome.REJECTED, ConfidenceReason.INSUFFICIENT_FEATURES, null));

        assertTrue(cp.pose().isPresent(), "a rejected estimate is retained for inspection");
        assertEquals(Optional.empty(), cp.consumablePose(),
                "a rejected estimate must not be handed out as a measurement");
    }

    @Test
    public void notProducedCarriesNoPoseAtAll() {
        ConfidentPose cp = ConfidentPose.notProduced(
                verdict(ConfidenceOutcome.NOT_PRODUCED, ConfidenceReason.ESTIMATOR_FAILED, null));

        assertEquals(Optional.empty(), cp.pose());
        assertEquals(Optional.empty(), cp.consumablePose());
        assertEquals(ConfidenceReason.ESTIMATOR_FAILED, cp.confidence().reason());
    }

    @Test
    public void poseAndVerdictCannotDisagreeAboutWhetherAnEstimateExists() {
        Pose3D pose = new Pose3D(1.0, 2.0, 0.0, 30.0);

        // Supplying a pose alongside a NOT_PRODUCED verdict, or omitting one alongside a verdict
        // that says an estimate exists, is a contradiction. Resolved loudly, never silently.
        assertThrows(IllegalArgumentException.class, () -> ConfidentPose.of(pose,
                verdict(ConfidenceOutcome.NOT_PRODUCED, ConfidenceReason.ESTIMATOR_FAILED, null)));

        assertThrows(IllegalArgumentException.class, () -> ConfidentPose.notProduced(
                verdict(ConfidenceOutcome.USABLE, ConfidenceReason.OK, 0.9)));
    }
}
