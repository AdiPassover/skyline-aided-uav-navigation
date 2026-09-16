package org.boofcv.relocalization;

import javax.annotation.Nullable;

import java.util.List;

/**
 * The outcome of one exact exhaustive retrieval (design §7, §9.2) — per view, plus the dual
 * region-level evidence the acceptance gate reads ({@code DEC-INT-003}).
 *
 * <p>{@link #north} is always scored; {@link #west} exists only when the query carried a valid West
 * view and at least one reference had an admissible West alignment; {@link #fused} exists only under
 * a fusing rule ({@code weakest_view} / {@code mean_score}) with both views present. The
 * {@link #primary()} ranking — the one the tracker and the gate act on — is the fused one when it
 * exists and the North one otherwise, so a query without West is scored honestly on North alone.
 *
 * <p>Regions are formed by the configured {@link RegionRule}: the top region is the set of
 * candidates in the ball around the top-1, and the <b>competing score is the best score of any
 * scored reference outside that ball</b> — not merely among the top-K — so a dense memory whose
 * top-K is one place does not hide its real competitor ({@code EXP-SKY-010}/{@code -011}
 * region-level margin).
 *
 * <p>Scores are raw NCC values and lags are raw sample shifts. They are logged; they are never
 * converted into a position, a weight or a probability.
 *
 * @param queryFrameIndex      the query frame
 * @param queryTimestampS      its timestamp
 * @param matcherVariant       which matcher scored (both views)
 * @param fusionRule           the configured rule
 * @param databaseSize         references in memory at query time
 * @param consideredCount      references not excluded as recent
 * @param excludedRecentIds    ids excluded by the recent-reference rule, ascending
 * @param north                the North ranking
 * @param west                 the West ranking, or {@code null}
 * @param fused                the fused ranking, or {@code null}
 * @param westQueryAvailable   the query carried a West view
 * @param westQueryValid       that view was valid
 * @param westReferencesScored references whose West descriptor was scored
 * @param dual                 the region-level dual evidence, or {@code null} when refused
 * @param refusedReason        why no retrieval ran (invalid query, empty database, nothing
 *                             admissible), else {@code null}
 */
public record RetrievalResult(int queryFrameIndex, double queryTimestampS, String matcherVariant,
                              String fusionRule, int databaseSize, int consideredCount,
                              List<Integer> excludedRecentIds, @Nullable ViewRanking north,
                              @Nullable ViewRanking west, @Nullable ViewRanking fused,
                              boolean westQueryAvailable, boolean westQueryValid,
                              int westReferencesScored, @Nullable DualRegionEvidence dual,
                              @Nullable String refusedReason) {

    public RetrievalResult {
        excludedRecentIds = List.copyOf(excludedRecentIds);
        if ((refusedReason == null) == (north == null)) {
            throw new IllegalArgumentException("a North ranking is present exactly when retrieval ran");
        }
    }

    /** The ranking the tracker and the gate act on: fused when it exists, else North. */
    @Nullable
    public ViewRanking primary() {
        return fused != null ? fused : north;
    }

    /** True when retrieval ran and the primary ranking has at least one candidate. */
    public boolean hasCandidates() {
        ViewRanking p = primary();
        return refusedReason == null && p != null && !p.candidates().isEmpty();
    }

    /** The primary ranking's best candidate, or {@code null}. */
    @Nullable
    public Candidate top() {
        ViewRanking p = primary();
        return p == null ? null : p.top();
    }

    /** The primary ranking's best region, or {@code null}. */
    @Nullable
    public Region topRegion() {
        ViewRanking p = primary();
        return p == null ? null : p.topRegion();
    }

    /** The primary ranking's candidates (empty when refused). */
    public List<Candidate> candidates() {
        ViewRanking p = primary();
        return p == null ? List.of() : p.candidates();
    }

    /** The primary ranking's regions (empty when refused). */
    public List<Region> regions() {
        ViewRanking p = primary();
        return p == null ? List.of() : p.regions();
    }

    @Nullable
    public Double topRegionScore() {
        ViewRanking p = primary();
        return p == null ? null : p.topRegionScore();
    }

    @Nullable
    public Double competingRegionScore() {
        ViewRanking p = primary();
        return p == null ? null : p.competingRegionScore();
    }

    @Nullable
    public Double regionMargin() {
        ViewRanking p = primary();
        return p == null ? null : p.regionMargin();
    }

    /**
     * One view's (or the fused) ranking.
     *
     * @param view                 {@code north}, {@code west} or {@code fused}
     * @param candidates           the top-K, descending score; ties broken by ascending id
     * @param regions              regions among the candidates, descending best score; region 0 is
     *                             the ball around the top-1
     * @param topRegionScore       the top-1's score, or {@code null} when no candidate
     * @param competingRegionScore best score of any scored reference outside the top region, or
     *                             {@code null} when every scored reference lies inside it
     * @param regionMargin         {@code topRegionScore − competingRegionScore}, or {@code null}
     * @param refusedAlignmentIds  ids for which no alignment cleared the overlap floor (C1), ascending
     * @param scoredCount          references that received a finite score in this view
     */
    public record ViewRanking(String view, List<Candidate> candidates, List<Region> regions,
                              @Nullable Double topRegionScore, @Nullable Double competingRegionScore,
                              @Nullable Double regionMargin, List<Integer> refusedAlignmentIds,
                              int scoredCount) {
        public ViewRanking {
            candidates = List.copyOf(candidates);
            regions = List.copyOf(regions);
            refusedAlignmentIds = List.copyOf(refusedAlignmentIds);
        }

        @Nullable
        public Candidate top() {
            return candidates.isEmpty() ? null : candidates.get(0);
        }

        @Nullable
        public Region topRegion() {
            return regions.isEmpty() ? null : regions.get(0);
        }

        public boolean hasCandidates() {
            return !candidates.isEmpty();
        }

        /** The candidate entry for a reference id, or {@code null} when not in the top-K. */
        @Nullable
        public Candidate candidate(int referenceId) {
            for (Candidate c : candidates) {
                if (c.referenceId() == referenceId) {
                    return c;
                }
            }
            return null;
        }
    }

    /**
     * One retrieved reference in one ranking.
     *
     * @param rank            1-based position in the ranked list
     * @param referenceId     the reference
     * @param score           raw NCC at the winning alignment (or the fused value)
     * @param lagSamples      the winning lag (0 for C0 / fused) — preserved, never a position
     * @param overlapFraction fraction of samples compared at that lag
     * @param scoreAtZeroLag  the score at lag 0 when searched (C1), else {@code null}
     * @param frameIndex      the reference's frame
     * @param timestampS      its timestamp
     * @param segmentId       the segment it was created in
     * @param regionId        the region it was grouped into (0 = the top region)
     */
    public record Candidate(int rank, int referenceId, double score, int lagSamples,
                            double overlapFraction, @Nullable Double scoreAtZeroLag, int frameIndex,
                            double timestampS, int segmentId, int regionId) {
    }

    /**
     * A region among the candidates.
     *
     * @param regionId        0 for the top region
     * @param bestReferenceId the region's best-scoring member
     * @param bestScore       its score
     * @param memberIds       all member reference ids, ascending
     * @param bestPosition    the best member's stored persistent position (metres, ENU)
     */
    public record Region(int regionId, int bestReferenceId, double bestScore,
                         List<Integer> memberIds, PlanarPosition bestPosition) {
        public Region {
            memberIds = List.copyOf(memberIds);
            if (bestPosition == null) {
                throw new IllegalArgumentException("bestPosition is required");
            }
        }
    }

    /** Whether, and why not, the West view contributed to this retrieval. */
    public enum DualStatus {
        /** The configured rule consults North only. */
        NOT_USED,
        /** The query carried no West view. */
        WEST_UNAVAILABLE,
        /** The query's West view was invalid. */
        WEST_INVALID,
        /** The West view was valid but no reference had an admissible West alignment. */
        WEST_NO_CANDIDATE,
        /** Both views produced a ranking; {@code agreement} is defined. */
        COMPLETE;

        public String wireName() {
            return name().toLowerCase();
        }

        public boolean complete() {
            return this == COMPLETE;
        }
    }

    /**
     * Region-level dual evidence for the primary candidate ({@code EXP-SKY-011}'s
     * {@code DualRegionEvidence}, without the position-truth fields an online system does not have).
     * All scores raw; no probability. {@code agreement} is {@code null} unless both rankings
     * exist — <b>a missing West is never an agreement</b>.
     *
     * @param candidateReferenceId the primary ranking's top-1
     * @param candidateRegionId    always 0 (the primary top region)
     * @param northScore           the candidate's North score, or {@code null} when it had none
     * @param westScore            the candidate's West score, or {@code null}
     * @param northMargin          the North ranking's region-level margin
     * @param westMargin           the West ranking's region-level margin
     * @param fusedScore           the fused ranking's top score, when a fused ranking exists
     * @param fusedMargin          its region-level margin
     * @param northTopReferenceId  the North ranking's top-1
     * @param westTopReferenceId   the West ranking's top-1, or {@code null}
     * @param agreement            the two top-1s name one region under the region rule, or {@code null}
     * @param topSeparationM       distance between the two top-1s' stored positions, or {@code null}
     * @param status               why {@code agreement} is or is not defined
     */
    public record DualRegionEvidence(int candidateReferenceId, int candidateRegionId,
                                     @Nullable Double northScore, @Nullable Double westScore,
                                     @Nullable Double northMargin, @Nullable Double westMargin,
                                     @Nullable Double fusedScore, @Nullable Double fusedMargin,
                                     int northTopReferenceId, @Nullable Integer westTopReferenceId,
                                     @Nullable Boolean agreement, @Nullable Double topSeparationM,
                                     DualStatus status) {
        public DualRegionEvidence {
            if (status == null) {
                throw new IllegalArgumentException("status is required");
            }
            if ((agreement != null) != status.complete()) {
                throw new IllegalArgumentException("agreement is defined exactly when the dual evidence is complete");
            }
        }
    }
}
