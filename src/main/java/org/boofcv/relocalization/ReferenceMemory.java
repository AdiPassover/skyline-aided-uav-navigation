package org.boofcv.relocalization;

import javax.annotation.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Sparse online trusted-reference memory with the design's starting insertion rule, one explicit
 * recent-reference exclusion mode, and exact exhaustive top-K retrieval per view (design §6, §7;
 * {@code DEC-INT-003} for the dual view).
 *
 * <h2>Insertion (§6.2)</h2>
 *
 * <pre>
 *   trusted_insert = north_skyline_valid AND global_position_valid AND (I_eff = false)
 *                    AND (E_eff ≤ search_exposure_bound)
 *                    AND (novel_view OR max_spacing_reached)
 * </pre>
 *
 * <p>Novelty is {@code 1 − NCC} (pointwise, C0) of the North view to the <em>last trusted
 * reference</em>; max-spacing is the fallback for slowly changing skylines. The West view, when
 * valid, is stored beside North and takes no part in the insertion decision. The stored pose is the
 * persistent position in metres ENU and the VO's authoritative heading ({@link TrustedReference}).
 *
 * <h2>Retrieval (§6.3, §7)</h2>
 *
 * <p>Exact and exhaustive, per view: every non-excluded reference is scored with the configured
 * {@link SkylineMatcher} on its North descriptor, and — when the query carries a valid West view
 * and a fusing rule is configured — every non-excluded reference that has a West descriptor is
 * scored on it, as a second independent bank. Recent references are excluded under exactly one
 * mechanism. A fused ranking ({@code weakest_view} = {@code min(s_N, s_W)}; {@code mean_score} =
 * {@code (s_N + s_W)/2}, {@code EXP-SKY-011} verbatim) is formed over references scored in both
 * banks; a reference with no admissible alignment in either view loses, never gains. No ANN, no
 * index.
 *
 * <p>This class decides nothing about acceptance and moves no position. The {@link AcceptanceGate}
 * and the {@link NavigationAligner} act on what it returns.
 */
public final class ReferenceMemory {

    private final RelocalizationConfig config;
    private final SkylineMatcher matcher;
    private final RegionRule regionRule;
    private final List<TrustedReference> references = new ArrayList<>();
    private final List<InsertionDecision> decisions = new ArrayList<>();
    private final List<RetrievalResult> retrievals = new ArrayList<>();

    public ReferenceMemory(RelocalizationConfig config) {
        this(config, SkylineMatcher.fromConfig(config));
    }

    public ReferenceMemory(RelocalizationConfig config, SkylineMatcher matcher) {
        if (config == null || matcher == null) {
            throw new IllegalArgumentException("config and matcher are required");
        }
        this.config = config.validate();
        this.matcher = matcher;
        this.regionRule = RegionRule.fromConfig(config);
    }

    /**
     * Offers an observation for insertion, evaluated against the navigation state at that frame.
     *
     * @param observation the skyline observation on the current frame
     * @param navigation  the aligner's output for the same frame (its persistent position,
     *                    heading and lineage are what the rule reads and what is stored)
     */
    public InsertionDecision offer(SkylineObservation observation, NavigationOutput navigation) {
        if (observation == null || navigation == null) {
            throw new IllegalArgumentException("observation and navigation are required");
        }
        if (observation.frameIndex() != navigation.frameIndex()) {
            throw new IllegalArgumentException("observation frame " + observation.frameIndex()
                    + " != navigation frame " + navigation.frameIndex());
        }
        int frame = observation.frameIndex();
        double ts = observation.timestampS();

        if (!observation.northValid()) {
            return record(new InsertionDecision(frame, ts,
                    InsertionDecision.Outcome.REJECTED_SKYLINE_INVALID, null, null, null));
        }
        if (!navigation.globalPositionValid()) {
            return record(new InsertionDecision(frame, ts,
                    InsertionDecision.Outcome.REJECTED_GLOBAL_POSITION_INVALID, null, null, null));
        }
        AnchorLineage.Snapshot lineage = navigation.lineage();
        if (lineage.effectiveInstability()) {
            return record(new InsertionDecision(frame, ts,
                    InsertionDecision.Outcome.REJECTED_UNRESOLVED_INSTABILITY, null, null, null));
        }
        if (lineage.effectiveExposure() > config.searchExposureBound) {
            return record(new InsertionDecision(frame, ts,
                    InsertionDecision.Outcome.REJECTED_EXPOSURE_BEYOND_SEARCH_BOUND, null, null, null));
        }

        Double novelty = null;
        Integer framesSinceLast = null;
        boolean allowed;
        if (references.isEmpty()) {
            allowed = true;
        } else {
            TrustedReference last = references.get(references.size() - 1);
            novelty = observation.northDescriptor().distance(last.northDescriptor());
            framesSinceLast = frame - last.frameIndex();
            boolean novel = novelty >= config.noveltyMinDistance;
            boolean spacing = framesSinceLast >= config.maxSpacingFrames;
            allowed = novel || spacing;
        }
        if (!allowed) {
            return record(new InsertionDecision(frame, ts,
                    InsertionDecision.Outcome.REJECTED_NOT_NOVEL, null, novelty, framesSinceLast));
        }

        PlanarPosition global = navigation.globalPosition().orElseThrow();
        Integer anchorId = lineage.anchorReferenceId();
        TrustedReference ref = new TrustedReference(references.size(), frame, ts,
                observation.northDescriptor(),
                observation.westValid() ? observation.westDescriptor() : null,
                global, navigation.headingDeg(), navigation.headingStatus(),
                navigation.segmentId(), navigation.alignmentEpochId(),
                anchorId, anchorId == null ? null : lineage.anchorFrameIndex(),
                lineage.effectiveExposure(), lineage.effectiveInstability(),
                TrustedReference.POSE_SCHEMA);
        references.add(ref);
        return record(new InsertionDecision(frame, ts, InsertionDecision.Outcome.INSERTED, ref,
                novelty, framesSinceLast));
    }

    /**
     * Exact exhaustive retrieval for a query, per view, with recent-reference exclusion in the one
     * configured mode.
     *
     * @param query the query observation; one whose North view is invalid yields a refused result
     */
    public RetrievalResult retrieve(SkylineObservation query) {
        if (query == null) {
            throw new IllegalArgumentException("query is required");
        }
        int frame = query.frameIndex();
        double ts = query.timestampS();
        String variant = matcher.variant();
        boolean westAvailable = query.westAvailable();
        boolean westValid = query.westValid();

        if (!query.northValid()) {
            return record(refused(frame, ts, variant, references.size(), 0, List.of(), westAvailable,
                    westValid, "query invalid: " + query.invalidReason()));
        }
        if (references.isEmpty()) {
            return record(refused(frame, ts, variant, 0, 0, List.of(), westAvailable, westValid,
                    "empty database"));
        }

        List<Integer> excluded = new ArrayList<>();
        List<Scored> north = new ArrayList<>();
        List<Integer> northRefused = new ArrayList<>();
        List<Scored> west = new ArrayList<>();
        List<Integer> westRefused = new ArrayList<>();
        boolean scoreWest = config.fusesWest() && westValid;
        int westScoredRefs = 0;
        int recentStart = config.excludeByCount()
                ? Math.max(0, references.size() - config.recentExclusionCount)
                : references.size();
        int considered = 0;
        for (int i = 0; i < references.size(); i++) {
            TrustedReference r = references.get(i);
            boolean recent = config.excludeByCount()
                    ? i >= recentStart
                    : r.timestampS() > ts - config.recentExclusionSeconds;
            if (recent) {
                excluded.add(r.id());
                continue;
            }
            considered++;
            MatchResult mn = matcher.match(query.northDescriptor(), r.northDescriptor());
            if (mn.accepted()) {
                north.add(new Scored(r, mn));
            } else {
                northRefused.add(r.id());
            }
            if (scoreWest && r.hasWest()) {
                westScoredRefs++;
                MatchResult mw = matcher.match(query.westDescriptor(), r.westDescriptor());
                if (mw.accepted()) {
                    west.add(new Scored(r, mw));
                } else {
                    westRefused.add(r.id());
                }
            }
        }
        if (north.isEmpty()) {
            String why = considered == 0 ? "every reference excluded as recent"
                    : "no reference had an admissible North alignment";
            return record(refused(frame, ts, variant, references.size(), considered, excluded,
                    westAvailable, westValid, why));
        }

        RetrievalResult.ViewRanking northRanking = rank(SkylineView.NORTH, north, northRefused);
        RetrievalResult.ViewRanking westRanking = west.isEmpty() ? null
                : rank(SkylineView.WEST, west, westRefused);
        RetrievalResult.ViewRanking fusedRanking = null;
        if (config.usesFusedRanking() && westRanking != null) {
            List<Scored> fused = fuse(north, west);
            if (!fused.isEmpty()) {
                fusedRanking = rank("fused", fused, List.of());
            }
        }

        RetrievalResult.DualStatus status;
        if (!config.fusesWest()) {
            status = RetrievalResult.DualStatus.NOT_USED;
        } else if (!westAvailable) {
            status = RetrievalResult.DualStatus.WEST_UNAVAILABLE;
        } else if (!westValid) {
            status = RetrievalResult.DualStatus.WEST_INVALID;
        } else if (westRanking == null) {
            status = RetrievalResult.DualStatus.WEST_NO_CANDIDATE;
        } else {
            status = RetrievalResult.DualStatus.COMPLETE;
        }
        RetrievalResult.ViewRanking primary = fusedRanking != null ? fusedRanking : northRanking;
        RetrievalResult.DualRegionEvidence dual = dualEvidence(primary, northRanking, westRanking,
                fusedRanking, north, west, status);

        return record(new RetrievalResult(frame, ts, variant, config.fusionRule, references.size(),
                considered, excluded, northRanking, westRanking, fusedRanking, westAvailable,
                westValid, westScoredRefs, dual, null));
    }

    private static RetrievalResult refused(int frame, double ts, String variant, int size,
                                           int considered, List<Integer> excluded,
                                           boolean westAvailable, boolean westValid, String why) {
        return new RetrievalResult(frame, ts, variant, "", size, considered, excluded, null, null,
                null, westAvailable, westValid, 0, null, why);
    }

    /** {@code EXP-SKY-011 fuse_scores}: min or mean over references scored in both views. */
    private List<Scored> fuse(List<Scored> north, List<Scored> west) {
        Map<Integer, Scored> byId = new HashMap<>();
        for (Scored s : west) {
            byId.put(s.reference.id(), s);
        }
        List<Scored> out = new ArrayList<>();
        boolean mean = RelocalizationConfig.FUSION_MEAN.equals(config.fusionRule);
        for (Scored n : north) {
            Scored w = byId.get(n.reference.id());
            if (w == null) {
                continue;                                   // -inf in one view: the reference loses
            }
            double a = n.match.score(), b = w.match.score();
            double f = mean ? 0.5 * (a + b) : Math.min(a, b);
            out.add(new Scored(n.reference, new MatchResult("fused", f, 0, 0, 1.0, true, null, 0, null)));
        }
        return out;
    }

    /**
     * Sorts one bank, takes the top-K, forms the top region as the ball around the top-1 under the
     * region rule, and reads the competing score over <em>every</em> scored reference outside it.
     */
    private RetrievalResult.ViewRanking rank(String view, List<Scored> scored, List<Integer> refused) {
        List<Scored> sorted = new ArrayList<>(scored);
        sorted.sort(Comparator.comparingDouble((Scored s) -> -s.match.score())
                .thenComparingInt(s -> s.reference.id()));
        List<Scored> top = sorted.subList(0, Math.min(config.topK, sorted.size()));
        Scored best = sorted.get(0);

        // Region 0: candidates in the ball around the top-1. Competing: best outside the ball,
        // over all scored references — a dense memory's top-K may be one place entirely.
        List<Scored> region0 = new ArrayList<>();
        List<Scored> rest = new ArrayList<>();
        for (Scored s : top) {
            if (regionRule.inAmbiguityRegionOf(best.reference, s.reference)) {
                region0.add(s);
            } else {
                rest.add(s);
            }
        }
        Double competing = null;
        for (Scored s : sorted) {
            if (!regionRule.inAmbiguityRegionOf(best.reference, s.reference)) {
                competing = s.match.score();                // sorted descending: the first is the best
                break;
            }
        }

        List<List<Scored>> groups = new ArrayList<>();
        groups.add(region0);
        for (Scored s : rest) {                              // greedy grouping of the remaining candidates
            boolean placed = false;
            for (int g = 1; g < groups.size(); g++) {
                if (regionRule.inAmbiguityRegionOf(groups.get(g).get(0).reference, s.reference)) {
                    groups.get(g).add(s);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                List<Scored> g = new ArrayList<>();
                g.add(s);
                groups.add(g);
            }
        }

        Map<Integer, Integer> regionOf = new HashMap<>();
        List<RetrievalResult.Region> regions = new ArrayList<>();
        for (int g = 0; g < groups.size(); g++) {
            List<Scored> members = groups.get(g);
            Scored gBest = members.get(0);                   // groups are formed in score order
            List<Integer> ids = new ArrayList<>();
            for (Scored s : members) {
                ids.add(s.reference.id());
                regionOf.put(s.reference.id(), g);
            }
            Collections.sort(ids);
            regions.add(new RetrievalResult.Region(g, gBest.reference.id(), gBest.match.score(), ids,
                    gBest.reference.positionGlobal()));
        }

        List<RetrievalResult.Candidate> candidates = new ArrayList<>();
        for (int i = 0; i < top.size(); i++) {
            Scored s = top.get(i);
            candidates.add(new RetrievalResult.Candidate(i + 1, s.reference.id(), s.match.score(),
                    s.match.shiftSamples(), s.match.overlapFraction(), s.match.scoreAtZeroLag(),
                    s.reference.frameIndex(), s.reference.timestampS(), s.reference.segmentId(),
                    regionOf.get(s.reference.id())));
        }
        double topScore = best.match.score();
        Double margin = competing == null ? null : topScore - competing;
        List<Integer> refusedSorted = new ArrayList<>(refused);
        Collections.sort(refusedSorted);
        return new RetrievalResult.ViewRanking(view, candidates, regions, topScore, competing, margin,
                refusedSorted, scored.size());
    }

    private RetrievalResult.DualRegionEvidence dualEvidence(
            RetrievalResult.ViewRanking primary, RetrievalResult.ViewRanking north,
            @Nullable RetrievalResult.ViewRanking west, @Nullable RetrievalResult.ViewRanking fused,
            List<Scored> northScored, List<Scored> westScored, RetrievalResult.DualStatus status) {
        int candidate = primary.top().referenceId();
        Double northScore = scoreOf(northScored, candidate);
        Double westScore = scoreOf(westScored, candidate);
        Integer westTop = west == null ? null : west.top().referenceId();
        Boolean agreement = null;
        Double separation = null;
        if (status.complete()) {
            TrustedReference n = references.get(north.top().referenceId());
            TrustedReference w = references.get(westTop);
            agreement = regionRule.sameDualPlace(n, w);
            separation = n.positionGlobal().distanceTo(w.positionGlobal());
        }
        return new RetrievalResult.DualRegionEvidence(candidate, 0, northScore, westScore,
                north.regionMargin(), west == null ? null : west.regionMargin(),
                fused == null ? null : fused.topRegionScore(), fused == null ? null : fused.regionMargin(),
                north.top().referenceId(), westTop, agreement, separation, status);
    }

    @Nullable
    private static Double scoreOf(List<Scored> scored, int referenceId) {
        for (Scored s : scored) {
            if (s.reference.id() == referenceId) {
                return s.match.score();
            }
        }
        return null;
    }

    public int size() {
        return references.size();
    }

    public List<TrustedReference> references() {
        return Collections.unmodifiableList(references);
    }

    @Nullable
    public TrustedReference get(int id) {
        return id >= 0 && id < references.size() ? references.get(id) : null;
    }

    public List<InsertionDecision> insertionDecisions() {
        return Collections.unmodifiableList(decisions);
    }

    public List<RetrievalResult> retrievals() {
        return Collections.unmodifiableList(retrievals);
    }

    public RelocalizationConfig config() {
        return config;
    }

    public SkylineMatcher matcher() {
        return matcher;
    }

    RegionRule regionRule() {
        return regionRule;
    }

    private InsertionDecision record(InsertionDecision d) {
        decisions.add(d);
        return d;
    }

    private RetrievalResult record(RetrievalResult r) {
        retrievals.add(r);
        return r;
    }

    private record Scored(TrustedReference reference, MatchResult match) {
    }
}
