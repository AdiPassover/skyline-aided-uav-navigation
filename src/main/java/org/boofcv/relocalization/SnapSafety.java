package org.boofcv.relocalization;

import java.util.Optional;

/**
 * The snap-safety quantities (design §14): whether a discrete snap onto a stored reference position
 * would improve or worsen the position error, given ground truth.
 *
 * <pre>
 *   e_before = ‖p_truth − p_global_before‖          the persistent/VO position before the snap
 *   e_after  = ‖p_truth − p_reference‖              the position the query would be snapped onto
 *   Δe       = e_after − e_before                    positive = the snap made it worse
 * </pre>
 *
 * <p>A correct place match with {@code Δe > 0} is a <em>harmful accepted correction</em> and must be
 * reported separately from a wrong place match.
 *
 * <h2>One coordinate convention — {@link #UNITS}</h2>
 *
 * <p>Every input is a {@link PlanarPosition}: metres, East/North, in the <b>persistent frame rooted
 * at the run's first frame</b>. Dataset ground truth ({@code groundtruth.csv}: {@code east_m},
 * {@code north_m} in the source's own ENU frame — for a UE capture the world's ENU mapping, not
 * re-origined) is brought into that frame by {@link #groundTruthToPersistent}: a subtraction of the
 * ground-truth position at the run's first frame. No rotation is involved because both frames share
 * the North datum ({@code DEC-VO-010} D2) and no scale is involved because both are metres. The P0
 * version of this class compared pixel-valued persistent poses with metric truth and left the
 * conversion to the caller; that mixed convention no longer exists.
 *
 * <p>Heading is not part of the snap-safety quantity: a re-anchor is translation-only, so the
 * heading error before and after a snap is the same number, the VO's own.
 *
 * <p>Pure functions; nothing here reads a file or runs an experiment.
 */
public final class SnapSafety {

    /** The single convention every argument to this class is in. */
    public static final String UNITS =
            "metres, ENU (east, north), persistent frame rooted at the run's first frame";

    private SnapSafety() {
    }

    /**
     * A ground-truth position expressed in the persistent frame: {@code gt − gt(first frame)}.
     *
     * @param groundTruthAtRootFrame the ground-truth position at the run's first frame
     * @param groundTruth            the ground-truth position at the frame of interest
     */
    public static PlanarPosition groundTruthToPersistent(PlanarPosition groundTruthAtRootFrame,
                                                          PlanarPosition groundTruth) {
        if (groundTruthAtRootFrame == null || groundTruth == null) {
            throw new IllegalArgumentException("both positions are required");
        }
        return groundTruth.minus(groundTruthAtRootFrame);
    }

    /** {@code ‖truth − estimate‖} in metres. */
    public static double positionErrorM(PlanarPosition truthPersistent, PlanarPosition estimatePersistent) {
        if (truthPersistent == null || estimatePersistent == null) {
            throw new IllegalArgumentException("both positions are required");
        }
        return truthPersistent.distanceTo(estimatePersistent);
    }

    /**
     * @param truthPersistent the true position at the query frame, persistent frame
     * @param globalBefore    the VO-derived persistent position before the snap
     * @param referenceSnap   the reference position the query would be snapped onto
     */
    public static Delta evaluate(PlanarPosition truthPersistent, PlanarPosition globalBefore,
                                 PlanarPosition referenceSnap) {
        if (truthPersistent == null || globalBefore == null || referenceSnap == null) {
            throw new IllegalArgumentException("all three positions are required");
        }
        double eBefore = truthPersistent.distanceTo(globalBefore);
        double eAfter = truthPersistent.distanceTo(referenceSnap);
        return new Delta(eBefore, eAfter, eAfter - eBefore);
    }

    /**
     * The quantity for an applied re-anchor, from its own record. Empty when the re-anchor followed
     * a loss (no persistent position existed before it, so there is no {@code e_before}).
     */
    public static Optional<Delta> evaluate(ReanchorEvent reanchor, PlanarPosition truthPersistent) {
        if (reanchor == null) {
            throw new IllegalArgumentException("reanchor is required");
        }
        if (reanchor.globalBefore() == null) {
            return Optional.empty();
        }
        return Optional.of(evaluate(truthPersistent, reanchor.globalBefore(), reanchor.globalAfter()));
    }

    /**
     * @param positionErrorBeforeM {@code e_before}, metres
     * @param positionErrorAfterM  {@code e_after}, metres
     * @param positionDeltaM       {@code Δe}, metres; positive = the snap made it worse
     */
    public record Delta(double positionErrorBeforeM, double positionErrorAfterM, double positionDeltaM) {

        /** The design's harmful-correction criterion. */
        public boolean harmful() {
            return positionDeltaM > 0.0;
        }
    }
}
