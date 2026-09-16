package org.boofcv.confidence;

import org.boofcv.util.structs.Pose3D;

import javax.annotation.Nullable;

import java.util.Objects;
import java.util.Optional;

/**
 * A pose bound to its confidence verdict (feature spec FR-007).
 *
 * <p>Constitution Principle X requires that "low-confidence estimates MUST NOT be silently consumed
 * as reliable measurements". This type makes that structural rather than advisory: there is no
 * accessor yielding a bare {@link Pose3D} without its {@link ConfidenceResult}, so a caller cannot
 * take the pose while ignoring the verdict.
 *
 * <h2>What this does and does not claim</h2>
 *
 * <p><b>Does claim:</b> the verdict is unavoidable at the interface. Every path to the pose passes
 * through a structure carrying it.
 *
 * <p><b>Does not claim:</b> that low-confidence estimates cannot be misused. Nothing in a type
 * system can compel a caller to <em>act</em> on a verdict it has been handed, and the stronger
 * claim would simply be false. Stated here rather than left ambiguous, because the difference
 * matters for what the thesis may assert.
 *
 * <p>{@link #pose()} returns an {@link Optional} rather than a nullable value, so the case where no
 * pose exists ({@link ConfidenceOutcome#NOT_PRODUCED}) has to be handled at the call site instead of
 * being dereferenced by accident.
 */
public final class ConfidentPose {

    @Nullable
    private final Pose3D pose;
    private final ConfidenceResult confidence;

    private ConfidentPose(@Nullable Pose3D pose, ConfidenceResult confidence) {
        this.pose = pose;
        this.confidence = Objects.requireNonNull(confidence, "confidence");
    }

    /**
     * A frame that produced a pose, with its verdict.
     *
     * @throws IllegalArgumentException if the verdict says no pose was produced — the two would
     *                                  then disagree about whether an estimate exists, and a
     *                                  disagreement silently resolved is worse than a loud one
     */
    public static ConfidentPose of(Pose3D pose, ConfidenceResult confidence) {
        Objects.requireNonNull(pose, "pose");
        Objects.requireNonNull(confidence, "confidence");
        if (confidence.outcome() == ConfidenceOutcome.NOT_PRODUCED) {
            throw new IllegalArgumentException(
                    "A pose was supplied but the verdict is NOT_PRODUCED; use notProduced() instead");
        }
        return new ConfidentPose(pose, confidence);
    }

    /**
     * A frame that produced no pose.
     *
     * @throws IllegalArgumentException if the verdict does not say {@code NOT_PRODUCED}
     */
    public static ConfidentPose notProduced(ConfidenceResult confidence) {
        Objects.requireNonNull(confidence, "confidence");
        if (confidence.outcome() != ConfidenceOutcome.NOT_PRODUCED) {
            throw new IllegalArgumentException(
                    "No pose was supplied but the verdict is " + confidence.outcome()
                            + "; use of(pose, confidence) instead");
        }
        return new ConfidentPose(null, confidence);
    }

    /**
     * The pose, if one was produced.
     *
     * <p>Empty when the estimator produced no update for the frame. Note this is <em>not</em> empty
     * for {@link ConfidenceOutcome#REJECTED}: a rejected estimate exists and can be inspected — it
     * simply must not be consumed as a measurement. Erasing it would destroy evidence about the
     * frame, which is the opposite of what a rejection is for.
     */
    public Optional<Pose3D> pose() {
        return Optional.ofNullable(pose);
    }

    /** The verdict. Never null. */
    public ConfidenceResult confidence() {
        return confidence;
    }

    /**
     * The pose only when the verdict says it may be consumed as a measurement.
     *
     * <p>Empty for {@link ConfidenceOutcome#REJECTED} and {@link ConfidenceOutcome#NOT_PRODUCED}.
     * This is the accessor a consumer that does not want to reason about outcomes should use.
     */
    public Optional<Pose3D> consumablePose() {
        return confidence.poseConsumable() ? pose() : Optional.empty();
    }

    @Override
    public String toString() {
        return "ConfidentPose[" + (pose == null ? "no pose" : pose.toString())
                + ", " + confidence.outcome() + "/" + confidence.reason()
                + ", score=" + confidence.score() + "]";
    }
}
