package org.boofcv.relocalization;

import javax.annotation.Nullable;

/**
 * The acceptance gate (design §8) on the post-{@code EXP-SKY-011} evidence model
 * ({@code DEC-INT-003}), and nothing else:
 *
 * <ol>
 *   <li>the primary ranking's best region is a strong enough match ({@code tau_match});</li>
 *   <li>a reference <em>outside its region</em> was scored at all, and it is sufficiently separated
 *       from that one ({@code tau_margin}, region-level) — see below;</li>
 *   <li>the dual-view condition of the configured {@code fusion_rule} — for
 *       {@code strict_agreement}: the West view passes its own gate and its top-1 names the same
 *       region as North's; for the fusing rules: the fused ranking is the primary one, so (1) and
 *       (2) were evaluated on it; for {@code north_only}: none;</li>
 *   <li>temporal region persistence over {@code N_confirm} consecutive eligible queries
 *       ({@link RegionTracker}), <b>when the configured {@code temporal_confirmation} requires
 *       it</b> — see below.</li>
 * </ol>
 *
 * <p>Trusted-reference eligibility is a hard provenance rule, not a fifth score: the candidate must
 * be a reference the memory created under its insertion policy. If any condition fails the
 * navigation position stays exactly where the VO placed it; an ambiguous, disagreeing or
 * not-yet-confirmed candidate is <em>retained</em> for a later query but does not move the
 * trajectory. <b>Two views that both pass their gate and name different regions are never
 * accepted</b> — that disagreement is the signal the second view exists to provide. A missing or
 * invalid West is incomplete evidence, never agreement. No posterior, no entropy, no
 * score-to-probability mapping.
 *
 * <h2>No competing hypothesis never accepts</h2>
 *
 * <p>A margin exists only when some scored reference lay <em>outside</em> the winner's ambiguity
 * region. When none did — a memory holding one reference, a recent-exclusion window that leaves one
 * old candidate, or an ambiguity radius wide enough to swallow the whole database — the margin is
 * {@code null}, and this gate <b>retains</b> ({@code NO_COMPETING_HYPOTHESIS}). It does not treat
 * the missing test as a passed one.
 *
 * <p>The reason is a safety asymmetry, not a statistical one. <em>"No competitor was available"</em>
 * and <em>"the winner is demonstrably better than the alternatives"</em> are different claims, and
 * only the second justifies moving a navigation position onto a stored pose. Until 2026-09-08 this
 * class read {@code margin == null} as satisfying condition (2); on the flat figure-8 development
 * runs that was every single retrieval, so the margin was never once evaluated and every accepted
 * snap was authorised by condition (1) alone. The cost of the rule is a lost early relocalization
 * on a nearly empty memory; that is the intended trade — a false negative is recoverable and a
 * false discrete correction is not. {@link #primaryPassesGate} refuses a null margin for the same
 * reason, so such a query cannot lend temporal support to its region either.
 *
 * <h2>Single-view evidence never accepts under a dual rule (EXP-SKY-012)</h2>
 *
 * <p>Under {@code strict_agreement}, {@code weakest_view} or {@code mean_score} the dual condition
 * is the confirmation, and <b>no amount of North-only persistence substitutes for it</b>: a query
 * whose West is unavailable or invalid ({@code WEST_INCOMPLETE}), or — under strict agreement —
 * valid but below its own gate ({@code WEST_REFUSED}), is retained whatever the tracker's support
 * says. {@code EXP-SKY-012} R5 measured North-only temporal confirmation at k = 3 still accepting
 * 30 of 87 hard-city negatives at the frozen gate (1 of 87 on the dense memory at 0.95 / 0.20), and
 * R2 that West's protective action on the flat city is <em>refusal</em>, which a rule that falls
 * back to North on refusal would ignore; {@code EXP-SKY-011} R6 measured the same on the mountains
 * dense memory (12 / 8 absent-place acceptances at k = 2 / 3). The tracker still runs and its
 * support is logged on every decision, so the value a North-only fallback would have had stays
 * measurable offline.
 *
 * <p>The three {@code temporal_confirmation} modes therefore mean: {@code required} — persistence
 * on every acceptance, in addition to the dual condition; {@code fallback} — persistence is the
 * confirmation only where no dual condition exists ({@code north_only}), and under a dual rule the
 * dual condition alone accepts; {@code disabled} — never, which under {@code north_only} is the
 * single-capture baseline (a comparison arm, not a policy). The starting experimental policy
 * ({@code DEC-INT-003}, amended on {@code EXP-SKY-012}) is {@code weakest_view} + {@code fallback};
 * every value stays a config selection.
 */
public final class AcceptanceGate {

    public enum Verdict {
        ACCEPT, RETAIN, REJECT;

        public String wireName() {
            return name().toLowerCase();
        }
    }

    public enum Reason {
        ACCEPTED, NO_CANDIDATE, WEAK_MATCH, AMBIGUOUS_REGION, UNCONFIRMED, NOT_TRUSTED,
        /** Both views pass their gates and name different regions: retained, never accepted. */
        VIEW_DISAGREEMENT,
        /** A dual rule is configured and the West view is unavailable or invalid: single-view evidence, retained. */
        WEST_INCOMPLETE,
        /** Strict agreement: the West view is valid but below its own gate — a refusal, retained (EXP-SKY-012 R2). */
        WEST_REFUSED,
        /**
         * No reference outside the winner's ambiguity region was scored at all, so no margin exists:
         * the absence of a competitor is not evidence that the winner beat one. Retained.
         */
        NO_COMPETING_HYPOTHESIS;

        public String wireName() {
            return name().toLowerCase();
        }
    }

    /** What the dual-view condition concluded for this decision. */
    public enum DualVerdict {
        NOT_USED, INCOMPLETE, WEST_BELOW_GATE, DISAGREE, AGREED;

        public String wireName() {
            return name().toLowerCase();
        }
    }

    /**
     * @param verdict              the binary action ({@code ACCEPT}) or why not
     * @param reason               the first failing condition, or {@code ACCEPTED}
     * @param candidateReferenceId the primary top region's best reference, when there was one
     * @param topScore             primary top-region score, when there was one
     * @param margin               primary region-level margin, when a competing reference existed
     * @param supportCount         the tracked region's consecutive support
     * @param fusionRule           the configured rule
     * @param dualVerdict          the dual-view condition's outcome
     * @param temporalRequired     whether condition (4) was consulted for this decision
     * @param westScore            the West top-1 score when a West ranking existed
     * @param westMargin           the West region-level margin
     * @param agreement            the two views' top-1s named one region, or {@code null}
     */
    public record Decision(Verdict verdict, Reason reason, @Nullable Integer candidateReferenceId,
                           @Nullable Double topScore, @Nullable Double margin, int supportCount,
                           String fusionRule, DualVerdict dualVerdict, boolean temporalRequired,
                           @Nullable Double westScore, @Nullable Double westMargin,
                           @Nullable Boolean agreement) {
        public boolean accepted() {
            return verdict == Verdict.ACCEPT;
        }
    }

    private final RelocalizationConfig config;

    public AcceptanceGate(RelocalizationConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config is required");
        }
        this.config = config.validate();
    }

    /**
     * Conditions (1) and (2) alone — the frozen {@code accept-v2-margin} shape on the primary
     * ranking. What the {@link RegionTracker} needs to know before it counts a query as support.
     */
    public boolean primaryPassesGate(RetrievalResult result) {
        if (result == null) {
            throw new IllegalArgumentException("result is required");
        }
        RetrievalResult.Region top = result.topRegion();
        if (top == null || top.bestScore() < config.matchThreshold) {
            return false;
        }
        Double margin = result.regionMargin();
        // A null margin does not pass. A query with no competing hypothesis is not a query whose
        // region was confirmed, so it may not lend temporal support to that region either.
        return margin != null && margin >= config.marginThreshold;
    }

    /**
     * @param result the retrieval on this query
     * @param track  the region tracker's state after observing {@code result}, or {@code null}
     * @param memory the memory the candidate must belong to (provenance rule)
     */
    public Decision evaluate(RetrievalResult result, @Nullable RegionTracker.Track track,
                             ReferenceMemory memory) {
        if (result == null || memory == null) {
            throw new IllegalArgumentException("result and memory are required");
        }
        String rule = config.fusionRule;
        RetrievalResult.Region top = result.topRegion();
        if (top == null) {
            return new Decision(Verdict.REJECT, Reason.NO_CANDIDATE, null, null, null, 0, rule,
                    DualVerdict.NOT_USED, false, null, null, null);
        }
        int candidate = top.bestReferenceId();
        double score = top.bestScore();
        Double margin = result.regionMargin();
        int support = track == null ? 0 : track.supportCount();
        RetrievalResult.DualRegionEvidence d = result.dual();
        Double westScore = null;
        Double westMargin = null;
        Boolean agreement = null;
        if (d != null && result.west() != null) {
            westScore = result.west().topRegionScore();
            westMargin = d.westMargin();
            agreement = d.agreement();
        }

        // (1) strong match, provenance, (2) region-level margin — on the primary ranking.
        if (score < config.matchThreshold) {
            return new Decision(Verdict.REJECT, Reason.WEAK_MATCH, candidate, score, margin, support,
                    rule, DualVerdict.NOT_USED, false, westScore, westMargin, agreement);
        }
        if (memory.get(candidate) == null) {
            return new Decision(Verdict.REJECT, Reason.NOT_TRUSTED, candidate, score, margin, support,
                    rule, DualVerdict.NOT_USED, false, westScore, westMargin, agreement);
        }
        if (margin == null) {
            // Nothing outside the winner's ambiguity region was scored, so there is no margin to
            // test. "No competitor was available" is not "the winner is demonstrably better than
            // the alternatives", and a discrete pose correction is not a place to treat absence of
            // ambiguity evidence as positive evidence. RETAIN, not REJECT: the candidate itself is
            // not disproven — the EVIDENCE SET is incomplete, exactly as for an incomplete West —
            // so it stays available to a later query whose memory does contain a competitor.
            return new Decision(Verdict.RETAIN, Reason.NO_COMPETING_HYPOTHESIS, candidate, score,
                    null, support, rule, DualVerdict.NOT_USED, false, westScore, westMargin,
                    agreement);
        }
        if (margin < config.marginThreshold) {
            return new Decision(Verdict.RETAIN, Reason.AMBIGUOUS_REGION, candidate, score, margin,
                    support, rule, DualVerdict.NOT_USED, false, westScore, westMargin, agreement);
        }

        // (3) the dual-view condition.
        DualVerdict dual;
        if (RelocalizationConfig.FUSION_NORTH_ONLY.equals(rule)) {
            dual = DualVerdict.NOT_USED;
        } else if (RelocalizationConfig.FUSION_STRICT.equals(rule)) {
            if (d == null || !d.status().complete() || westScore == null) {
                dual = DualVerdict.INCOMPLETE;
            } else {
                boolean westPasses = westScore >= config.effectiveWestMatchThreshold()
                        && (westMargin == null || westMargin >= config.effectiveWestMarginThreshold());
                if (!westPasses) {
                    dual = DualVerdict.WEST_BELOW_GATE;
                } else if (!Boolean.TRUE.equals(agreement)) {
                    return new Decision(Verdict.RETAIN, Reason.VIEW_DISAGREEMENT, candidate, score,
                            margin, support, rule, DualVerdict.DISAGREE, false, westScore, westMargin,
                            agreement);
                } else {
                    dual = DualVerdict.AGREED;
                }
            }
        } else {
            // weakest_view / mean_score: the primary ranking IS the fused one when both views exist.
            dual = result.fused() != null ? DualVerdict.AGREED : DualVerdict.INCOMPLETE;
        }

        // Under a dual rule, the dual condition is the confirmation. Single-view evidence — a
        // missing / invalid West, or a West that refused under strict agreement — is retained no
        // matter how long North persists (EXP-SKY-012 R2/R5, EXP-SKY-011 R6).
        boolean dualRule = !RelocalizationConfig.FUSION_NORTH_ONLY.equals(rule);
        if (dualRule && dual != DualVerdict.AGREED) {
            Reason why = dual == DualVerdict.WEST_BELOW_GATE ? Reason.WEST_REFUSED : Reason.WEST_INCOMPLETE;
            return new Decision(Verdict.RETAIN, why, candidate, score, margin, support, rule, dual,
                    false, westScore, westMargin, agreement);
        }

        // (4) temporal persistence, as configured: required — always; fallback — only where no
        // dual condition exists (north_only); disabled — never.
        boolean temporalRequired = switch (config.temporalConfirmation) {
            case RelocalizationConfig.TEMPORAL_REQUIRED -> true;
            case RelocalizationConfig.TEMPORAL_DISABLED -> false;
            default -> !dualRule;                               // fallback
        };
        if (temporalRequired
                && (track == null || !track.covers(candidate) || support < config.confirmQueries)) {
            return new Decision(Verdict.RETAIN, Reason.UNCONFIRMED, candidate, score, margin,
                    support, rule, dual, true, westScore, westMargin, agreement);
        }
        return new Decision(Verdict.ACCEPT, Reason.ACCEPTED, candidate, score, margin, support, rule,
                dual, temporalRequired, westScore, westMargin, agreement);
    }
}
