package org.boofcv.relocalization;

import javax.annotation.Nullable;

/**
 * One query–reference comparison with its alignment parameters reported — the Java mirror of
 * {@code hsreloc.matchers.base.MatchResult} for the two variants INT uses ({@code DEC-INT-002}).
 *
 * <p>{@code score} is on the NCC scale in {@code [-1, 1]}, or {@code -∞} when no alignment cleared
 * the overlap floor ({@code accepted == false}). {@code shiftSamples} is in profile samples,
 * positive meaning the reference is sampled further right. <b>The shift carries no pose
 * semantics</b>: skyline observations are north-facing, so a lag is not a relative heading, and
 * whether it carries translation information is the SKY lane's open investigation. It is preserved
 * and logged, never converted.
 *
 * @param variant            {@code c0_frozen_ncc} or {@code c1_bounded_lag_ncc}
 * @param score              NCC of the best alignment, or {@code -∞}
 * @param shiftSamples       the winning lag (0 for C0)
 * @param overlap            samples compared at the winning lag
 * @param overlapFraction    {@code overlap / n}
 * @param accepted           false when no alignment cleared the overlap floor
 * @param scoreAtZeroLag     the C1 score at lag 0 when that lag was searched, else {@code null}
 * @param alignmentsSearched alignments that cleared the floor and were scored
 * @param refusalReason      why nothing was scored, when {@code accepted} is false
 */
public record MatchResult(String variant, double score, int shiftSamples, int overlap,
                          double overlapFraction, boolean accepted, @Nullable Double scoreAtZeroLag,
                          int alignmentsSearched, @Nullable String refusalReason) {
}
