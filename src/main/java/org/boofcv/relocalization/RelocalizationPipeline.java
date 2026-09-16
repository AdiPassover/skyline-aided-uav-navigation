package org.boofcv.relocalization;

import javax.annotation.Nullable;

/**
 * The per-frame orchestration of the relocalization-assisted navigation loop (design §5–§10),
 * with every decision delegated to one small component and nothing decided here:
 *
 * <pre>
 *   VO handoff (LocalPoseSample) ──► aligner ──► instability? (tau_I) ──► scheduler.onFrame
 *                                                                              │
 *              eligible North skyline & executable & increment usable? ────────┘
 *                     │ yes
 *                     ▼
 *   memory.retrieve (North bank, West bank, fused) ──► tracker.observe ──► gate.evaluate
 *        ──► ACCEPT: aligner.acceptReference (translation only)
 *            RETAIN: keep request, wait for next
 *            REJECT: aligner.reject (no-op)
 *                     ▼
 *   memory.offer (insertion — after retrieval, so a frame never matches itself)
 * </pre>
 *
 * <p>Order matters and is fixed: the aligner sees the frame first; instability is decided on this
 * frame's {@code delta_rot_refit} and noted before scheduling; retrieval runs only when the
 * scheduler says the request is executable on an eligible observation <em>and</em> the frame's own
 * metric increment exists (the query position must be the query's position); insertion runs last,
 * on the frame's final navigation output. Synchronous by design (§11).
 *
 * <p>What is logged but never acted on: both views' C1 lags ({@code DEC-INT-002}), and the
 * VO-motion auxiliary comparison between consecutive confirmations, which is computed only when VO
 * is healthy and the search was not caused by suspect VO, and is never a rejection cue (§7).
 */
public final class RelocalizationPipeline {

    /** Everything that happened on one frame, for the sidecar and for tests. */
    public record FrameStep(NavigationOutput output, boolean instabilityEvent,
                            @Nullable Double deltaRotRefitDeg,
                            @Nullable TranslationDropoutEvent dropout,
                            @Nullable SearchScheduler.Request requestArmed,
                            @Nullable SearchScheduler.Escalation escalation,
                            @Nullable SearchScheduler.Request pendingRequest,
                            boolean requestPending, boolean retryGapBlocking,
                            boolean skylinePresent, boolean skylineValid,
                            boolean westAvailable, boolean westValid, @Nullable Double syncDtS,
                            @Nullable RetrievalResult retrieval,
                            @Nullable RegionTracker.Track track,
                            @Nullable AcceptanceGate.Decision decision,
                            @Nullable MatchEvidence evidence,
                            @Nullable SearchScheduler.Attempt attempt,
                            @Nullable ReanchorEvent reanchor,
                            @Nullable NavigationAligner.RejectedCandidate rejection,
                            @Nullable InsertionDecision insertion,
                            @Nullable Double voAuxiliaryDisplacementErrorM) {
    }

    private final RelocalizationConfig config;
    private final NavigationAligner aligner;
    private final ReferenceMemory memory;
    private final SearchScheduler scheduler;
    private final RegionTracker tracker;
    private final AcceptanceGate gate;

    // VO-auxiliary consistency between consecutive confirmations — logged, never a gate.
    @Nullable private PlanarPosition previousConfirmationLocal;
    @Nullable private PlanarPosition previousConfirmationReferencePosition;
    private int previousConfirmationSegment = -1;

    public RelocalizationPipeline(RelocalizationConfig config) {
        this(config, RelocalizedPoseRule.DISCRETE_REFERENCE);
    }

    public RelocalizationPipeline(RelocalizationConfig config, RelocalizedPoseRule poseRule) {
        if (config == null) {
            throw new IllegalArgumentException("config is required");
        }
        this.config = config.validate();
        this.aligner = new NavigationAligner(poseRule);
        this.memory = new ReferenceMemory(config);
        this.scheduler = new SearchScheduler(config);
        this.tracker = new RegionTracker(config);
        this.gate = new AcceptanceGate(config);
    }

    /**
     * @param sample            the VO handoff for this frame
     * @param skyline           the skyline observation for this frame, or {@code null} when none
     * @param deltaRotRefitDeg  this frame's {@code delta_rot_refit}, or {@code null} when the
     *                          diagnostic refit produced none (or is not enabled)
     */
    public FrameStep step(LocalPoseSample sample, @Nullable SkylineObservation skyline,
                          @Nullable Double deltaRotRefitDeg) {
        if (sample == null) {
            throw new IllegalArgumentException("sample is required");
        }
        int frameIndex = sample.frameIndex();
        double timestampS = sample.timestampS();
        if (skyline != null && skyline.frameIndex() != frameIndex) {
            throw new IllegalArgumentException("skyline observation frame " + skyline.frameIndex()
                    + " != frame " + frameIndex);
        }
        int dropoutsBefore = aligner.translationDropoutEvents().size();
        NavigationOutput out = aligner.observe(sample);
        TranslationDropoutEvent dropout = aligner.translationDropoutEvents().size() > dropoutsBefore
                ? aligner.translationDropoutEvents().get(dropoutsBefore) : null;

        // Instability: a validated high delta_rot_refit on a usable increment marks the ledger and
        // requests search now. A low value is simply "no event" — it cannot delay anything.
        boolean instability = deltaRotRefitDeg != null && config.instabilityThresholdDeg != null
                && sample.voSuccess() && out.event() == NavigationOutput.FrameEvent.NONE
                && deltaRotRefitDeg >= config.instabilityThresholdDeg;
        if (instability) {
            aligner.noteInstability();
            out = aligner.current();
        }

        SearchScheduler.Request armed = scheduler.onFrame(out, instability);
        SearchScheduler.Escalation escalation = scheduler.escalationThisFrame();

        boolean present = skyline != null;
        boolean valid = present && skyline.northValid();
        boolean westAvailable = present && skyline.westAvailable();
        boolean westValid = present && skyline.westValid();
        RetrievalResult retrieval = null;
        RegionTracker.Track track = null;
        AcceptanceGate.Decision decision = null;
        MatchEvidence evidence = null;
        SearchScheduler.Attempt attempt = null;
        ReanchorEvent reanchor = null;
        NavigationAligner.RejectedCandidate rejection = null;
        Double voAux = null;

        if (valid && out.translationUsable() && scheduler.executable(frameIndex, true)) {
            SearchScheduler.Request req = scheduler.pendingRequest();
            retrieval = memory.retrieve(skyline);
            track = tracker.observe(retrieval, gate.primaryPassesGate(retrieval));
            decision = gate.evaluate(retrieval, track, memory);

            if (retrieval.hasCandidates()) {
                RetrievalResult.Region top = retrieval.topRegion();
                RetrievalResult.Candidate best = retrieval.primary().candidate(top.bestReferenceId());
                RetrievalResult.Candidate northBest = retrieval.north().candidate(top.bestReferenceId());
                RetrievalResult.Candidate westBest = retrieval.west() == null ? null
                        : retrieval.west().candidate(top.bestReferenceId());
                RetrievalResult.DualRegionEvidence d = retrieval.dual();
                voAux = voAuxiliary(out, top.bestReferenceId(), track, req.cause());
                evidence = new MatchEvidence(frameIndex, timestampS, out.localPosition(),
                        out.headingDeg(), req.cause(), req.originCause(), req.originFrame(),
                        req.escalated(), retrieval.matcherVariant(), retrieval.fusionRule(),
                        top.regionId(), top.bestReferenceId(),
                        decision.accepted() ? top.bestReferenceId() : null, top.bestScore(),
                        retrieval.competingRegionScore(), retrieval.regionMargin(),
                        northBest == null ? best.lagSamples() : northBest.lagSamples(),
                        northBest == null ? best.overlapFraction() : northBest.overlapFraction(),
                        northBest == null ? best.scoreAtZeroLag() : northBest.scoreAtZeroLag(),
                        d == null ? null : d.northScore(), d == null ? null : d.northMargin(),
                        d == null ? null : d.westScore(), d == null ? null : d.westMargin(),
                        westBest == null ? null : westBest.lagSamples(),
                        d == null ? null : d.agreement(),
                        d == null ? RetrievalResult.DualStatus.NOT_USED.wireName() : d.status().wireName(),
                        decision.dualVerdict().wireName(),
                        track == null ? 0 : track.supportCount(), decision.temporalRequired(),
                        decision.accepted());
            }

            switch (decision.verdict()) {
                case ACCEPT -> {
                    TrustedReference ref = memory.get(decision.candidateReferenceId());
                    reanchor = aligner.acceptReference(ref, evidence);
                    out = aligner.current();
                    attempt = scheduler.recordAttempt(frameIndex, timestampS,
                            SearchScheduler.AttemptOutcome.ACCEPTED, ref.id(),
                            "score=" + decision.topScore() + ";margin=" + decision.margin()
                                    + ";support=" + decision.supportCount()
                                    + ";dual=" + decision.dualVerdict().wireName());
                    tracker.reset();
                    previousConfirmationLocal = null;
                    previousConfirmationReferencePosition = null;
                }
                case RETAIN -> attempt = scheduler.recordAttempt(frameIndex, timestampS,
                        SearchScheduler.AttemptOutcome.RETAINED_FOR_CONFIRMATION,
                        decision.candidateReferenceId(), decision.reason().wireName()
                                + ";support=" + decision.supportCount()
                                + ";dual=" + decision.dualVerdict().wireName());
                case REJECT -> {
                    if (decision.candidateReferenceId() != null) {
                        rejection = aligner.reject(decision.candidateReferenceId(),
                                decision.reason().wireName());
                    }
                    attempt = scheduler.recordAttempt(frameIndex, timestampS,
                            retrieval.refusedReason() != null
                                    ? SearchScheduler.AttemptOutcome.REFUSED
                                    : SearchScheduler.AttemptOutcome.REJECTED,
                            decision.candidateReferenceId(),
                            retrieval.refusedReason() != null ? retrieval.refusedReason()
                                    : decision.reason().wireName());
                }
            }
        }

        InsertionDecision insertion = null;
        if (present) {
            insertion = memory.offer(skyline, aligner.current());
        }

        return new FrameStep(aligner.current(), instability, deltaRotRefitDeg, dropout, armed,
                escalation, scheduler.pendingRequest(), scheduler.requestPending(),
                scheduler.retryGapBlocking(frameIndex),
                present, valid, westAvailable, westValid, present ? skyline.syncDtS() : null,
                retrieval, track, decision, evidence, attempt, reanchor, rejection, insertion, voAux);
    }

    /**
     * {@code |Δlocal| − |Δreference|} between this confirmation and the previous one of the same
     * tracked region, metres. Auxiliary only: computed when VO is not suspect (search not caused
     * by instability / hard loss / dropout, no unresolved instability) and the two confirmations
     * lie in one segment; never used by the gate.
     */
    @Nullable
    private Double voAuxiliary(NavigationOutput out, int bestReferenceId,
                               @Nullable RegionTracker.Track track, SearchScheduler.Cause cause) {
        TrustedReference ref = memory.get(bestReferenceId);
        Double result = null;
        boolean voHealthy = !cause.voIsSuspect() && !out.lineage().effectiveInstability();
        if (voHealthy && ref != null && track != null && track.supportCount() >= 2
                && previousConfirmationLocal != null && previousConfirmationReferencePosition != null
                && previousConfirmationSegment == out.segmentId()) {
            double dLocal = out.localPosition().distanceTo(previousConfirmationLocal);
            double dRef = ref.positionGlobal().distanceTo(previousConfirmationReferencePosition);
            result = Math.abs(dLocal - dRef);
        }
        previousConfirmationLocal = out.localPosition();
        previousConfirmationReferencePosition = ref == null ? null : ref.positionGlobal();
        previousConfirmationSegment = out.segmentId();
        return result;
    }

    public NavigationAligner aligner() {
        return aligner;
    }

    public ReferenceMemory memory() {
        return memory;
    }

    public SearchScheduler scheduler() {
        return scheduler;
    }

    public RegionTracker tracker() {
        return tracker;
    }

    public AcceptanceGate gate() {
        return gate;
    }

    public RelocalizationConfig config() {
        return config;
    }
}
