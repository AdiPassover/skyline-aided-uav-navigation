package org.boofcv.relocalization;

import javax.annotation.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The alignment layer between the metric VO and the persistent navigation frame
 * ({@code DEC-INT-001} as amended 2026-09-07; design §3, §4, §10).
 *
 * <pre>
 *   p_global(k) = p_segment(k) + t_e          heading_global(k) = psi_nav(k)   (the VO's, untouched)
 * </pre>
 *
 * <p>{@code p_segment} is the VO's segment-relative metric position ({@link LocalPoseSample});
 * {@code t_e} is the current epoch's {@link AlignmentTransform} — a translation, or UNKNOWN. The
 * estimator is never touched: this class only <em>reads</em> what the metric state publishes.
 *
 * <h2>State transitions</h2>
 *
 * <ul>
 *   <li><b>First frame</b> — root segment (VO segment 0), {@code t_0 = 0}, position valid,
 *       {@code E_anchor = 0}, {@code I_anchor = false} (design §3 root convention).</li>
 *   <li><b>Usable increment</b> — {@code E_since += 1}; position composed through {@code t_e} if
 *       known.</li>
 *   <li><b>Hard loss</b> ({@code voSuccess == false}) — the VO opened a new segment; this layer
 *       adopts its index (and refuses to continue if the two ever disagree), {@code t_e} becomes
 *       UNKNOWN, the persistent position is absent until a trusted reference is accepted. No
 *       zero-motion bridge is inserted anywhere. The heading is not affected: it is the external
 *       channel's, and its own validity travels with every output. The ledger is untouched
 *       (design §4).</li>
 *   <li><b>Translation dropout</b> ({@code voSuccess} but no metric increment produced) — the
 *       segment position now hides an unobserved displacement, so {@code t_e} becomes UNKNOWN
 *       exactly as after a loss; the VO segment continues.</li>
 *   <li><b>Accepted reference</b> — {@code t_e ← p_reloc − p_segment} for the current segment,
 *       where {@code p_reloc} comes from the {@link RelocalizedPoseRule} (today: the stored
 *       reference position); the lineage inherits the reference's stored baseline; a
 *       {@link ReanchorEvent} is emitted with the applied delta. <b>The heading is not changed and
 *       cannot be.</b> Outputs already emitted are not revisited.</li>
 *   <li><b>Rejected / ambiguous candidate</b> — nothing moves. {@link #reject} exists so the no-op
 *       is explicit and logged.</li>
 *   <li><b>Instability event</b> — {@code I_since ← true}, sticky. The threshold lives in the
 *       {@link RelocalizationPipeline}; this class holds no threshold.</li>
 * </ul>
 *
 * <p>Navigation action is binary: the persistent position is either {@code p_segment + t_e} for the
 * current alignment or absent. Nothing here interpolates, blends, or rewrites history.
 *
 * <p>Not thread-safe; call from the frame loop.
 */
public final class NavigationAligner {

    private final RelocalizedPoseRule poseRule;

    private boolean started = false;

    private LocalSegment segment;
    private int alignmentEpochId = 0;

    /** {@code t_e}, or {@code null} while UNKNOWN. */
    @Nullable private AlignmentTransform alignment;

    private AnchorLineage lineage;

    private NavigationOutput current;
    private final List<HardLossEvent> hardLosses = new ArrayList<>();
    private final List<TranslationDropoutEvent> dropouts = new ArrayList<>();
    private final List<ReanchorEvent> reanchors = new ArrayList<>();
    private final List<RejectedCandidate> rejections = new ArrayList<>();
    private long unusableFrames = 0;

    /** Last frame with a valid persistent position, for the loss/dropout records. */
    private int lastValidGlobalFrame = -1;
    @Nullable private PlanarPosition lastValidGlobalPosition;

    /** The MVP: snaps onto the stored reference position. */
    public NavigationAligner() {
        this(RelocalizedPoseRule.DISCRETE_REFERENCE);
    }

    public NavigationAligner(RelocalizedPoseRule poseRule) {
        if (poseRule == null) {
            throw new IllegalArgumentException("poseRule is required");
        }
        this.poseRule = poseRule;
    }

    /**
     * Feeds one frame of the VO handoff.
     *
     * @return the output for this frame, also available via {@link #current()}
     * @throws IllegalStateException when the VO's segment index and this layer's segment disagree
     *                               — the two segment models must never drift silently
     */
    public NavigationOutput observe(LocalPoseSample s) {
        if (s == null) {
            throw new IllegalArgumentException("sample is required");
        }
        if (started && s.frameIndex() <= current.frameIndex()) {
            throw new IllegalArgumentException("frameIndex must be strictly increasing: "
                    + s.frameIndex() + " after " + current.frameIndex());
        }

        if (!started) {
            if (s.voSegmentIndex() != 0) {
                throw new IllegalStateException("the alignment layer must start with the run: the "
                        + "first sample is VO segment " + s.voSegmentIndex() + ", not 0");
            }
            segment = LocalSegment.root(s);
            alignment = AlignmentTransform.IDENTITY;
            lineage = new AnchorLineage(s.frameIndex(), s.timestampS());
            started = true;
            return publish(s, NavigationOutput.FrameEvent.INIT, null);
        }

        if (!s.voSuccess()) {
            int expected = segment.segmentId() + 1;
            if (s.voSegmentIndex() != expected) {
                throw new IllegalStateException("VO and INT segment models diverged at the hard loss "
                        + "on frame " + s.frameIndex() + ": the VO reports segment "
                        + s.voSegmentIndex() + ", this layer expected " + expected);
            }
            HardLossEvent loss = new HardLossEvent(s.frameIndex(), s.timestampS(), segment.segmentId(),
                    expected, lastValidGlobalPosition, lastValidGlobalFrame, s.headingKnownAcrossGap());
            hardLosses.add(loss);
            segment = new LocalSegment(expected, s.frameIndex(), s.timestampS(),
                    s.unknownTranslationGapBeforeSegment(), s.headingKnownAcrossGap());
            alignment = null;
            alignmentEpochId++;
            return publish(s, NavigationOutput.FrameEvent.HARD_LOSS, null);
        }

        if (s.voSegmentIndex() != segment.segmentId()) {
            throw new IllegalStateException("VO and INT segment models diverged on frame "
                    + s.frameIndex() + ": the VO reports segment " + s.voSegmentIndex()
                    + " on a successful frame while this layer is in segment " + segment.segmentId());
        }

        if (!s.translationUsable()) {
            unusableFrames++;
            if (alignment != null) {
                dropouts.add(new TranslationDropoutEvent(s.frameIndex(), s.timestampS(),
                        segment.segmentId(), s.heightStatus(), s.headingStatus(),
                        lastValidGlobalPosition, lastValidGlobalFrame));
                alignment = null;
                alignmentEpochId++;
            }
            return publish(s, NavigationOutput.FrameEvent.TRANSLATION_DROPOUT, null);
        }

        lineage.incrementExposure();
        return publish(s, NavigationOutput.FrameEvent.NONE, null);
    }

    /** A validated instability event on the current frame: {@code I_since} becomes sticky. */
    public void noteInstability() {
        requireStarted();
        lineage.noteInstability();
        // The ledger changed; re-publish so current() reflects it on this frame.
        if (current.event() != NavigationOutput.FrameEvent.REANCHOR) {
            current = withLineage(current, lineage.snapshot());
        }
    }

    /** {@link #acceptReference(TrustedReference, MatchEvidence)} without evidence. */
    public ReanchorEvent acceptReference(TrustedReference reference) {
        return acceptReference(reference, null);
    }

    /**
     * Accepts a trusted reference for the current frame: the alignment is replaced so that the
     * current segment position maps exactly onto the relocalized position the
     * {@link RelocalizedPoseRule} returns for that reference. The heading is untouched.
     *
     * <p>Whether the reference <em>should</em> be accepted is decided before calling this — by the
     * {@link AcceptanceGate}, or by a test. This method only applies the binary action. It refuses
     * to run before the first frame, refuses a second action on the same frame, and refuses a frame
     * whose own metric increment was not produced (its segment position is not the query's true
     * position).
     */
    public ReanchorEvent acceptReference(TrustedReference reference, @Nullable MatchEvidence evidence) {
        requireStarted();
        if (reference == null) {
            throw new IllegalArgumentException("reference is required");
        }
        if (current.event() == NavigationOutput.FrameEvent.REANCHOR) {
            throw new IllegalStateException("frame " + current.frameIndex()
                    + " was already re-anchored; one binary action per frame");
        }
        if (!current.translationUsable()) {
            throw new IllegalStateException("frame " + current.frameIndex() + " produced no metric "
                    + "increment (" + current.event().wireName() + "); its segment position is not "
                    + "the query's position, so no alignment may be established on it");
        }

        PlanarPosition queryLocal = current.localPosition();
        AlignmentTransform before = alignment;
        PlanarPosition globalBefore = current.globalPositionOrNull();
        AnchorLineage.Snapshot lineageBefore = lineage.snapshot();

        PlanarPosition relocalized = poseRule.relocalizedPosition(reference, evidence, queryLocal);
        if (relocalized == null) {
            throw new IllegalStateException("the pose rule returned no position");
        }
        AlignmentTransform after = AlignmentTransform.reanchor(relocalized, queryLocal);
        AlignmentTransform delta = before == null ? null : after.deltaFrom(before);

        alignment = after;
        alignmentEpochId++;
        lineage.inherit(reference, current.frameIndex(), current.timestampS());

        PlanarPosition globalAfter = after.apply(queryLocal);
        ReanchorEvent event = new ReanchorEvent(current.frameIndex(), current.timestampS(),
                reference.id(), reference.positionGlobal(), relocalized, queryLocal,
                current.headingDeg(), globalBefore, globalAfter, before, after, delta,
                lineageBefore, lineage.snapshot(), segment.segmentId(), alignmentEpochId,
                current.frameIndex() - reference.frameIndex(),
                current.timestampS() - reference.timestampS(), evidence, poseRule.id());
        reanchors.add(event);

        current = new NavigationOutput(current.frameIndex(), current.timestampS(), current.voSuccess(),
                queryLocal, current.translationUsable(), current.headingDeg(), current.headingStatus(),
                globalAfter, segment.segmentId(), alignmentEpochId, lineage.snapshot(),
                NavigationOutput.FrameEvent.REANCHOR, event);
        lastValidGlobalFrame = current.frameIndex();
        lastValidGlobalPosition = globalAfter;
        return event;
    }

    /**
     * Records that a candidate was not accepted on the current frame. Deliberately changes no
     * state: the position stays exactly where the VO placed it and the ledger is untouched (design
     * §4 table, §8).
     */
    public RejectedCandidate reject(@Nullable Integer candidateReferenceId, String reason) {
        requireStarted();
        if (reason == null || reason.isBlank()) {
            throw new IllegalArgumentException("a rejection needs a reason");
        }
        RejectedCandidate r = new RejectedCandidate(current.frameIndex(), current.timestampS(),
                candidateReferenceId, reason);
        rejections.add(r);
        return r;
    }

    /** The output for the most recent frame, reflecting any re-anchor applied on it. */
    public NavigationOutput current() {
        requireStarted();
        return current;
    }

    /** The current alignment {@code t_e}, empty while UNKNOWN. */
    public java.util.Optional<AlignmentTransform> alignment() {
        return java.util.Optional.ofNullable(alignment);
    }

    public LocalSegment segment() {
        requireStarted();
        return segment;
    }

    public AnchorLineage.Snapshot lineage() {
        requireStarted();
        return lineage.snapshot();
    }

    /** The persistent position is known on the current frame. */
    public boolean globalPositionValid() {
        return started && alignment != null;
    }

    /** The authoritative heading is valid on the current frame. */
    public boolean headingValid() {
        return started && current.headingValid();
    }

    /** {@code globalPositionValid AND headingValid}. */
    public boolean globalPoseValid() {
        return globalPositionValid() && headingValid();
    }

    public boolean started() {
        return started;
    }

    public int alignmentEpochId() {
        return alignmentEpochId;
    }

    public RelocalizedPoseRule poseRule() {
        return poseRule;
    }

    public List<HardLossEvent> hardLossEvents() {
        return Collections.unmodifiableList(hardLosses);
    }

    public List<TranslationDropoutEvent> translationDropoutEvents() {
        return Collections.unmodifiableList(dropouts);
    }

    /** Successful VO frames on which no metric increment existed (counted, whether or not they changed state). */
    public long unusableFrameCount() {
        return unusableFrames;
    }

    public List<ReanchorEvent> reanchorEvents() {
        return Collections.unmodifiableList(reanchors);
    }

    public List<RejectedCandidate> rejections() {
        return Collections.unmodifiableList(rejections);
    }

    private NavigationOutput publish(LocalPoseSample s, NavigationOutput.FrameEvent event,
                                     @Nullable ReanchorEvent reanchor) {
        PlanarPosition local = s.segmentPosition();
        PlanarPosition global = alignment == null ? null : alignment.apply(local);
        current = new NavigationOutput(s.frameIndex(), s.timestampS(), s.voSuccess(), local,
                s.translationUsable(), s.headingDeg(), s.headingStatus(), global,
                segment.segmentId(), alignmentEpochId, lineage.snapshot(), event, reanchor);
        if (global != null) {
            lastValidGlobalFrame = s.frameIndex();
            lastValidGlobalPosition = global;
        }
        return current;
    }

    private static NavigationOutput withLineage(NavigationOutput o, AnchorLineage.Snapshot lineage) {
        return new NavigationOutput(o.frameIndex(), o.timestampS(), o.voSuccess(), o.localPosition(),
                o.translationUsable(), o.headingDeg(), o.headingStatus(), o.globalPositionOrNull(),
                o.segmentId(), o.alignmentEpochId(), lineage, o.event(), o.reanchorEvent());
    }

    private void requireStarted() {
        if (!started) {
            throw new IllegalStateException("no frame observed yet");
        }
    }

    /** A candidate that was not accepted; recorded so the no-op is visible in the log. */
    public record RejectedCandidate(int frameIndex, double timestampS,
                                    @Nullable Integer candidateReferenceId, String reason) {
    }
}
