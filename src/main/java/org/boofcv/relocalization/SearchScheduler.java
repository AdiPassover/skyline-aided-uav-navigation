package org.boofcv.relocalization;

import javax.annotation.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Exposure-driven search scheduling with pending-request, retry and <b>attribution</b> semantics
 * (design §5, §13).
 *
 * <pre>
 *   scheduled_due     = (E_eff ≥ N_search) OR (time_since_anchor ≥ T_search_max)
 *   search_requested  = scheduled_due OR instability_event OR hard_loss OR translation_dropout OR confirmation_needed
 *   search_executable = search_requested AND skyline_query_valid AND retry gap clear
 * </pre>
 *
 * <p>Rules pinned by test:
 *
 * <ul>
 *   <li>A request is <b>sticky</b>: it stays pending across frames with no eligible skyline
 *       observation until an attempt runs on one.</li>
 *   <li>There is at most <b>one</b> pending request. A benign trigger while one is pending is
 *       counted as suppressed, not queued.</li>
 *   <li>A <b>suspect-VO</b> trigger (hard loss, translation dropout, instability) while one is
 *       pending <b>escalates</b> it: the request keeps its origin (cause, frame, time) and records
 *       an {@link Escalation} — whether it changed the effective cause, and whether a retry gap was
 *       blocking at that moment (so the override actually changed execution). A scheduled request
 *       later hit by instability is therefore never logged as if the instability had created the
 *       search opportunity, and there is still exactly one request.</li>
 *   <li>After an unsuccessful attempt a <b>minimum retry gap</b> applies before the next attempt
 *       may execute. This is operational backoff only — it never lowers exposure or clears
 *       instability (the ledger lives in {@link AnchorLineage} and this class cannot reach it).</li>
 *   <li>A new suspect-VO trigger <b>overrides</b> the retry gap.</li>
 *   <li>A candidate retained for confirmation keeps the request pending and makes the next eligible
 *       observation executable regardless of the retry gap — temporal confirmation is over
 *       <em>consecutive</em> eligible queries.</li>
 *   <li>The instability input is a boolean event decided upstream against {@code tau_I}; a low
 *       {@code delta_rot_refit} value is simply the absence of that event and cannot postpone or
 *       suppress anything here ({@code DEC-CONF-003} constraint B).</li>
 * </ul>
 */
public final class SearchScheduler {

    /** Why a search was requested or escalated. */
    public enum Cause {
        SCHEDULED_EXPOSURE, SCHEDULED_TIME, INSTABILITY, HARD_LOSS, TRANSLATION_DROPOUT, CONFIRMATION;

        public String wireName() {
            return name().toLowerCase();
        }

        /** Searches caused by suspect VO: VO motion may not be a rejection cue for them (§7). */
        public boolean voIsSuspect() {
            return this == INSTABILITY || this == HARD_LOSS || this == TRANSLATION_DROPOUT;
        }
    }

    /** What an attempt concluded. */
    public enum AttemptOutcome {
        ACCEPTED, RETAINED_FOR_CONFIRMATION, REJECTED, REFUSED;

        public String wireName() {
            return name().toLowerCase();
        }
    }

    /**
     * A suspect-VO trigger that arrived while a request was pending.
     *
     * @param frameIndex       the frame it arrived on
     * @param timestampS       its timestamp
     * @param cause            the trigger
     * @param changedCause     the pending request's effective cause was benign and became this one
     * @param overrodeRetryGap a retry gap was blocking execution at that moment and this lifted it
     */
    public record Escalation(int frameIndex, double timestampS, Cause cause, boolean changedCause,
                             boolean overrodeRetryGap) {
    }

    /**
     * The one pending search request: its origin, and every escalation it received since.
     *
     * @param originFrame      the frame the request was armed on
     * @param originTimestampS its timestamp
     * @param originCause      the trigger that armed it
     * @param escalations      suspect-VO triggers received while pending, in order
     */
    public record Request(int originFrame, double originTimestampS, Cause originCause,
                          List<Escalation> escalations) {
        public Request {
            escalations = List.copyOf(escalations);
        }

        /** The effective cause: the last cause-changing escalation's, else the origin's. */
        public Cause cause() {
            for (int i = escalations.size() - 1; i >= 0; i--) {
                if (escalations.get(i).changedCause()) {
                    return escalations.get(i).cause();
                }
            }
            return originCause;
        }

        public boolean escalated() {
            return !escalations.isEmpty();
        }

        /** Alias of {@link #originFrame()} for readers of the request's arming frame. */
        public int requestedFrame() {
            return originFrame;
        }

        public double requestedTimestampS() {
            return originTimestampS;
        }

        Request withEscalation(Escalation e) {
            List<Escalation> next = new ArrayList<>(escalations);
            next.add(e);
            return new Request(originFrame, originTimestampS, originCause, next);
        }
    }

    /**
     * One executed attempt, with the request's attribution copied onto it.
     *
     * @param cause              the effective cause at execution
     * @param originCause        the cause that armed the request
     * @param originFrame        the frame that armed it
     * @param escalationCount    escalations the request had received
     * @param underRetryOverride a suspect-VO override was in force at execution
     */
    public record Attempt(int frameIndex, double timestampS, Cause cause, Cause originCause,
                          int originFrame, int escalationCount, boolean underRetryOverride,
                          AttemptOutcome outcome, @Nullable Integer candidateReferenceId,
                          String detail) {
    }

    private final RelocalizationConfig config;

    @Nullable private Request pending;
    @Nullable private Escalation escalationThisFrame;
    private boolean overrideRetryGap;
    private boolean confirmationNeeded;
    private int lastAttemptFrame = -1;
    private int suppressedDuplicates;
    private int escalations;

    private final List<Request> requests = new ArrayList<>();
    private final List<Escalation> allEscalations = new ArrayList<>();
    private final List<Attempt> attempts = new ArrayList<>();

    public SearchScheduler(RelocalizationConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config is required");
        }
        this.config = config.validate();
    }

    /**
     * Evaluates the triggers for one frame. Call once per frame, after the aligner has observed it
     * and after any instability event has been decided.
     *
     * @param out              the aligner's output for the frame
     * @param instabilityEvent whether {@code delta_rot_refit} crossed {@code tau_I} on this frame
     * @return the request newly armed on this frame, or {@code null} (an escalation of an existing
     *         request is reported by {@link #escalationThisFrame()}, not here)
     */
    @Nullable
    public Request onFrame(NavigationOutput out, boolean instabilityEvent) {
        escalationThisFrame = null;
        Request armed = null;
        if (out.event() == NavigationOutput.FrameEvent.HARD_LOSS) {
            armed = request(out, Cause.HARD_LOSS);
        } else if (out.event() == NavigationOutput.FrameEvent.TRANSLATION_DROPOUT
                && out.alignmentEpochId() != lastDropoutEpoch) {
            // Requested once per known → unknown transition, not on every unusable frame.
            lastDropoutEpoch = out.alignmentEpochId();
            armed = request(out, Cause.TRANSLATION_DROPOUT);
        }
        if (instabilityEvent) {
            Request r = request(out, Cause.INSTABILITY);
            armed = armed != null ? armed : r;
        }
        AnchorLineage.Snapshot lineage = out.lineage();
        if (lineage.effectiveExposure() >= config.searchExposureBound) {
            Request r = request(out, Cause.SCHEDULED_EXPOSURE);
            armed = armed != null ? armed : r;
        } else if (config.searchTimeMaxS != null
                && out.timestampS() - lineage.anchorTimestampS() >= config.searchTimeMaxS) {
            Request r = request(out, Cause.SCHEDULED_TIME);
            armed = armed != null ? armed : r;
        }
        if (confirmationNeeded) {
            Request r = request(out, Cause.CONFIRMATION);
            armed = armed != null ? armed : r;
        }
        return armed;
    }

    private int lastDropoutEpoch = -1;

    /**
     * Arms a request, or — when one is already pending — escalates it (suspect-VO trigger) or
     * suppresses the duplicate (benign trigger). Escalation keeps the origin and appends an
     * {@link Escalation}; the gate must later know VO is suspect (§7), and the evaluation must
     * later know the search opportunity was not created by the escalating trigger.
     */
    @Nullable
    private Request request(NavigationOutput out, Cause cause) {
        if (pending != null) {
            if (cause.voIsSuspect()) {
                boolean gapWasBlocking = retryGapBlocking(out.frameIndex());
                boolean changed = !pending.cause().voIsSuspect();
                Escalation e = new Escalation(out.frameIndex(), out.timestampS(), cause, changed,
                        gapWasBlocking);
                pending = pending.withEscalation(e);
                escalationThisFrame = e;
                allEscalations.add(e);
                escalations++;
                overrideRetryGap = true;
            } else {
                suppressedDuplicates++;
            }
            return null;
        }
        if (cause.voIsSuspect()) {
            overrideRetryGap = true;
        }
        pending = new Request(out.frameIndex(), out.timestampS(), cause, List.of());
        requests.add(pending);
        return pending;
    }

    /** Whether a search is currently requested. */
    public boolean requestPending() {
        return pending != null;
    }

    @Nullable
    public Request pendingRequest() {
        return pending;
    }

    /** The escalation recorded on the most recent {@link #onFrame}, or {@code null}. */
    @Nullable
    public Escalation escalationThisFrame() {
        return escalationThisFrame;
    }

    /** Whether the retry gap alone is holding an otherwise-executable request back on this frame. */
    public boolean retryGapBlocking(int frameIndex) {
        return pending != null && !overrideRetryGap && !confirmationNeeded && lastAttemptFrame >= 0
                && frameIndex - lastAttemptFrame < config.minRetryGapFrames;
    }

    /** {@code search_executable}: a pending request, an eligible observation, and no retry hold. */
    public boolean executable(int frameIndex, boolean skylineValid) {
        return pending != null && skylineValid && !retryGapBlocking(frameIndex);
    }

    /**
     * Records an executed attempt and resolves the request accordingly: accepted or unsuccessful
     * attempts consume the request (the deadline will re-arm it, subject to the retry gap);
     * a retained candidate keeps it pending and flags confirmation.
     */
    public Attempt recordAttempt(int frameIndex, double timestampS, AttemptOutcome outcome,
                                 @Nullable Integer candidateReferenceId, String detail) {
        if (pending == null) {
            throw new IllegalStateException("no pending request to attempt");
        }
        Attempt a = new Attempt(frameIndex, timestampS, pending.cause(), pending.originCause(),
                pending.originFrame(), pending.escalations().size(), overrideRetryGap, outcome,
                candidateReferenceId, detail == null ? "" : detail);
        attempts.add(a);
        lastAttemptFrame = frameIndex;
        overrideRetryGap = false;
        if (outcome == AttemptOutcome.RETAINED_FOR_CONFIRMATION) {
            confirmationNeeded = true;
        } else {
            confirmationNeeded = false;
            pending = null;
        }
        return a;
    }

    public int suppressedDuplicates() {
        return suppressedDuplicates;
    }

    /** Suspect-VO triggers received while a request was pending (cause-changing or not). */
    public int escalations() {
        return escalations;
    }

    public int lastAttemptFrame() {
        return lastAttemptFrame;
    }

    public boolean confirmationNeeded() {
        return confirmationNeeded;
    }

    /** Every request as originally armed (escalations are on the pending request and in {@link #escalationEvents()}). */
    public List<Request> requests() {
        return Collections.unmodifiableList(requests);
    }

    public List<Escalation> escalationEvents() {
        return Collections.unmodifiableList(allEscalations);
    }

    public List<Attempt> attempts() {
        return Collections.unmodifiableList(attempts);
    }
}
