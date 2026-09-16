package org.boofcv.relocalization;

import javax.annotation.Nullable;

import java.util.List;

/**
 * Region-level temporal persistence (design §7, §8): does the same region keep receiving the
 * strongest support over consecutive eligible queries? It tracks the retrieval's <em>primary</em>
 * ranking (fused when a fusing rule produced one, North otherwise).
 *
 * <p>Region identity across queries is the configured {@link RegionRule}: by reference-id
 * proximity ({@code region_track_max_gap}), or by stored position within
 * {@code temporal_region_radius_m}. Both are symmetric, so forward traversal (ids rising), reverse
 * traversal (ids falling) and near-stationary observation (ids repeating) all count; the tracker
 * records which it saw but does not require any.
 *
 * <h2>The tracked region is a MOVING ball, and that is a per-step bound, not a total one</h2>
 *
 * <p>Each confirmation replaces the tracked representative with the new query's winner, so identity
 * means "within the radius of the <em>previous</em> query", never "within the radius of where the
 * chain began". A chain therefore <b>walks</b>: with a 60 m radius and captures 41 m apart, four
 * consecutive queries are one unbroken support-4 chain whose ends are 124 m apart. Under
 * {@code id_gap} the same holds with ids.
 *
 * <p>This is recorded rather than corrected, deliberately. Whether "same candidate region over
 * time" should mean a bounded step (what this does) or a bounded total extent depends on how far a
 * skyline match can move while still naming one place, which is exactly what the SKY matcher study
 * is measuring; picking a rule now would be picking it blind. It is also currently latent under the
 * starting policy, where a dual rule supplies the confirmation and temporal support is not consulted
 * ({@code temporal_confirmation: fallback}, {@link AcceptanceGate}) — it becomes live under
 * {@code north_only} or {@code required}. Pinned by
 * {@code RegionRadiiSeparationTest#theTemporalRegionIsAMovingBallAndTheChainHasNoTotalBound}.
 *
 * <p>Support counts <b>consecutive eligible queries that pass the match/margin gate</b> and name
 * the same region — {@code EXP-SKY-011}'s {@code temporal_confirmation} definition, where every
 * query in the window must pass. A query that fails the gate (weak or ambiguous) breaks the chain
 * rather than lending its region support it did not earn; so does a query with no candidate.
 *
 * <p>Since {@code EXP-SKY-011} temporal confirmation is a <em>configurable</em> requirement
 * ({@code temporal_confirmation}: required / fallback / disabled) — the {@link AcceptanceGate}
 * decides whether the support count is consulted; this class always keeps it, so switching the
 * requirement never changes the candidate state. VO progression is <b>not</b> consulted here:
 * the pipeline logs a VO-motion auxiliary alongside, never as a rejection cue (§7).
 *
 * <p>No probability semantics; support is an integer count of consecutive confirmations.
 */
public final class RegionTracker {

    /** The region currently being tracked across queries. */
    public record Track(int bestReferenceId, List<Integer> memberIds, PlanarPosition bestPosition,
                        int supportCount, int firstQueryFrame, int lastQueryFrame,
                        double lastTopScore, Traversal lastTraversal) {
        public Track {
            memberIds = List.copyOf(memberIds);
        }

        /** True when the reference is a member of the tracked region. */
        public boolean covers(int referenceId) {
            return bestReferenceId == referenceId || memberIds.contains(referenceId);
        }
    }

    /** How the best reference id moved between the last two confirmations. */
    public enum Traversal {
        FIRST, FORWARD, REVERSE, STATIONARY;

        public String wireName() {
            return name().toLowerCase();
        }
    }

    private final RegionRule rule;
    @Nullable private Track current;

    public RegionTracker(RelocalizationConfig config) {
        this(RegionRule.fromConfig(config));
    }

    /** Chronological identity with one id gap for both purposes (the P0/P1 test constructor). */
    public RegionTracker(int regionTrackMaxGap) {
        this(RegionRule.idGap(regionTrackMaxGap, regionTrackMaxGap));
    }

    RegionTracker(RegionRule rule) {
        if (rule == null) {
            throw new IllegalArgumentException("rule is required");
        }
        this.rule = rule;
    }

    /** {@link #observe(RetrievalResult, boolean)} for a query that passes the gate. */
    @Nullable
    public Track observe(RetrievalResult result) {
        return observe(result, true);
    }

    /**
     * Updates the track with one retrieval. A retrieval with no candidates, or one whose query
     * did not pass the match/margin gate, breaks the chain (the region was not supported on an
     * eligible query), so the track is dropped.
     *
     * @param queryPasses whether the primary top region cleared {@code tau_match} and
     *                    {@code tau_margin} ({@link AcceptanceGate#primaryPassesGate})
     * @return the current track after this observation, or {@code null} when nothing is tracked
     */
    @Nullable
    public Track observe(RetrievalResult result, boolean queryPasses) {
        if (result == null) {
            throw new IllegalArgumentException("result is required");
        }
        RetrievalResult.Region top = result.topRegion();
        if (top == null || !queryPasses) {
            current = null;
            return null;
        }
        if (current != null && rule.sameTemporalRegion(asRegion(current), top)) {
            Traversal t = top.bestReferenceId() > current.bestReferenceId() ? Traversal.FORWARD
                    : top.bestReferenceId() < current.bestReferenceId() ? Traversal.REVERSE
                    : Traversal.STATIONARY;
            current = new Track(top.bestReferenceId(), top.memberIds(), top.bestPosition(),
                    current.supportCount() + 1, current.firstQueryFrame(), result.queryFrameIndex(),
                    top.bestScore(), t);
        } else {
            current = new Track(top.bestReferenceId(), top.memberIds(), top.bestPosition(), 1,
                    result.queryFrameIndex(), result.queryFrameIndex(), top.bestScore(),
                    Traversal.FIRST);
        }
        return current;
    }

    private static RetrievalResult.Region asRegion(Track t) {
        return new RetrievalResult.Region(0, t.bestReferenceId(), t.lastTopScore(), t.memberIds(),
                t.bestPosition());
    }

    /** Symmetric id-proximity under the chronological rule: any member of one within the gap of any member of the other. */
    boolean sameRegion(List<Integer> a, List<Integer> b) {
        if (rule.positional()) {
            throw new IllegalStateException("id-based identity is not defined under the positional region rule");
        }
        PlanarPosition p = PlanarPosition.ORIGIN;
        return rule.sameTemporalRegion(new RetrievalResult.Region(0, a.get(0), 0.0, a, p),
                new RetrievalResult.Region(0, b.get(0), 0.0, b, p));
    }

    /** Drops the track — after an acceptance, so confirmation starts afresh. */
    public void reset() {
        current = null;
    }

    @Nullable
    public Track current() {
        return current;
    }
}
