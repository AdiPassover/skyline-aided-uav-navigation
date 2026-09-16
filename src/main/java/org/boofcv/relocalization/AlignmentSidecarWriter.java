package org.boofcv.relocalization;

import javax.annotation.Nullable;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Structured observability for the alignment layer and the relocalization loop (design §10, §14,
 * §15), as two CSV sidecars written next to {@code frames.csv} in the same opt-in style as
 * {@code logical_transform.csv}. Not part of the run-record contract; nothing in {@code naveval}
 * reads them yet.
 *
 * <ul>
 *   <li>{@code alignment_frames.csv} — one row per frame: the segment-relative position (metres,
 *       ENU), the authoritative heading and its status, <b>position validity and heading validity
 *       separately</b> plus their conjunction, the persistent position (empty when invalid), VO
 *       segment and alignment epoch, the anchor and its establishing re-anchor, the full ledger,
 *       the frame event, the scheduler's state with the pending request's origin and effective
 *       cause, and the skyline availability of both views.</li>
 *   <li>{@code alignment_events.csv} — one row per event: hard loss, translation dropout,
 *       instability, search request, search escalation (origin vs escalating cause, whether it
 *       changed the cause and whether it lifted a retry gap), retrieval (both views' top-K
 *       ids/scores/lags, region-level margins, dual status and agreement), attempt (with
 *       attribution), candidate rejection, reference insertion or refusal (with pose schema and
 *       lineage provenance), re-anchor (translation delta, jump, snap-safety inputs, evidence).</li>
 * </ul>
 *
 * <p>Absence is written as an empty field, never as zero. No column is a confidence. Lags are
 * logged as evidence and carry no position meaning ({@code DEC-INT-002}). All lengths are metres.
 */
public final class AlignmentSidecarWriter implements Closeable {

    public static final String FRAMES_FILE = "alignment_frames.csv";
    public static final String EVENTS_FILE = "alignment_events.csv";

    static final String FRAMES_HEADER =
            "frame_index,timestamp_s,vo_success,translation_usable,"
            + "local_east_m,local_north_m,heading_deg,heading_status,heading_valid,"
            + "global_position_valid,global_pose_valid,global_east_m,global_north_m,"
            + "segment_id,alignment_epoch_id,anchor_reference_id,anchor_reanchor_frame,"
            + "e_anchor,e_since,e_eff,i_anchor,i_since,i_eff,time_since_anchor_s,event,"
            + "delta_rot_refit_deg,instability_event,request_pending,request_origin_cause,"
            + "request_cause,retry_gap_blocking,"
            + "skyline_present,north_valid,west_available,west_valid,sync_dt_s";

    static final String EVENTS_HEADER =
            "frame_index,timestamp_s,kind,segment_id,alignment_epoch_id,reference_id,"
            + "reason,detail,"
            + "global_before_east_m,global_before_north_m,global_after_east_m,global_after_north_m,"
            + "local_east_m,local_north_m,heading_deg,"
            + "delta_east_m,delta_north_m,position_jump_m,"
            + "query_reference_frame_gap,query_reference_time_gap_s,"
            + "e_eff_before,e_eff_after,i_eff_before,i_eff_after,"
            + "top_k_ids,top_k_scores,top_k_lags,west_top_k_ids,west_top_k_scores,west_top_k_lags,"
            + "excluded_recent_ids,database_size,considered_count,"
            + "top_region_score,competing_region_score,region_margin,"
            + "north_score,north_margin,west_score,west_margin,west_lag_samples,agreement,"
            + "dual_status,dual_verdict,fusion_rule,temporal_required,"
            + "cause,origin_cause,origin_frame,escalation_cause,changed_cause,overrode_retry_gap,"
            + "matcher,lag_samples,overlap_fraction,support_count,verdict,vo_aux_displacement_error_m,"
            + "pose_schema,source_anchor_id,source_reanchor_frame,heading_status,"
            // Snap-safety block (2026-09-08), appended so existing column offsets are unchanged.
            + "loss_source,recovery_case,global_position_valid_before,heading_valid,"
            + "competitor_exists,scored_north,scored_west,"
            + "north_top_reference_id,west_top_reference_id,top_separation_m,"
            + "selected_reference_east_m,selected_reference_north_m,estimated_query_to_reference_m,"
            + "reference_age_frames,reference_age_s";

    /**
     * The two questions an accepted correction answers are not the same question, and a single
     * snap-safety statistic that pools them is meaningless ({@code EXP-INT-001}).
     *
     * <p><b>Drift correction</b> — the persistent position was VALID before the query, so the
     * correction has a defensible {@code e_before} and the quantity of interest is
     * {@code Δe = e_after − e_before}: did the snap improve or worsen an estimate that already
     * existed? <b>Hard-loss recovery</b> — the position was UNKNOWN, so there is no navigation
     * {@code e_before} at all and none is manufactured; the quantity of interest is the absolute
     * error the re-anchor restored. {@code SnapSafety.evaluate} already returns empty for the
     * second case rather than inventing a baseline, and this column names which case each row is so
     * an offline analysis cannot pool them by accident.
     */
    public static final String CASE_DRIFT_CORRECTION = "drift_correction";
    /** See {@link #CASE_DRIFT_CORRECTION}. */
    public static final String CASE_HARD_LOSS_RECOVERY = "hard_loss_recovery";

    private final BufferedWriter frames;
    private final BufferedWriter events;

    /**
     * The memory whose references this writer may look up, for the stored position of a selected
     * reference and its age. Optional: without it those columns are blank rather than guessed.
     */
    @Nullable private ReferenceMemory memory;

    /** Whether the persistent position was valid immediately BEFORE the current frame's decision. */
    @Nullable private Boolean positionValidBeforeDecision;

    public AlignmentSidecarWriter(Path directory) throws IOException {
        Files.createDirectories(directory);
        frames = Files.newBufferedWriter(directory.resolve(FRAMES_FILE), StandardCharsets.UTF_8);
        frames.write(FRAMES_HEADER);
        frames.newLine();
        events = Files.newBufferedWriter(directory.resolve(EVENTS_FILE), StandardCharsets.UTF_8);
        events.write(EVENTS_HEADER);
        events.newLine();
    }

    /** A frame from the alignment-only path: no scheduler state, no skyline. */
    public void writeFrame(NavigationOutput o) throws IOException {
        writeFrame(o, null, false, false, null, false, false, false, false, false, null);
    }

    /**
     * Lets the writer report a selected reference's stored position and age. Optional; without it
     * those snap-safety columns stay blank rather than being guessed.
     */
    public void setReferenceMemory(@Nullable ReferenceMemory memory) {
        this.memory = memory;
    }

    /** A frame from the full pipeline, with every event that happened on it. */
    public void writeStep(RelocalizationPipeline.FrameStep s) throws IOException {
        NavigationOutput at = s.output();
        // The state BEFORE this frame's decision. On an accepted frame `at` is already the
        // post-re-anchor output, and the re-anchor's own record is the only place that still knows
        // whether a position existed beforehand — globalBefore() is null exactly when it did not.
        positionValidBeforeDecision = s.reanchor() != null
                ? s.reanchor().globalBefore() != null
                : at.globalPositionValid();
        if (s.dropout() != null) {
            writeDropout(s.dropout());
        }
        if (s.instabilityEvent()) {
            EventRow row = new EventRow(at.frameIndex(), at.timestampS(), "instability");
            row.segmentId = at.segmentId();
            row.epochId = at.alignmentEpochId();
            row.detail = "delta_rot_refit_deg=" + s.deltaRotRefitDeg();
            write(row);
        }
        if (s.requestArmed() != null) {
            writeRequest(s.requestArmed(), at);
        }
        if (s.escalation() != null) {
            writeEscalation(s.escalation(), at);
        }
        if (s.retrieval() != null) {
            writeRetrieval(s.retrieval(), at, s.voAuxiliaryDisplacementErrorM());
        }
        if (s.rejection() != null) {
            writeRejection(s.rejection(), at);
        }
        if (s.attempt() != null) {
            writeAttempt(s.attempt(), s.decision(), s.evidence(), at);
        }
        if (s.reanchor() != null) {
            writeReanchor(s.reanchor());
        }
        if (s.insertion() != null) {
            writeInsertion(s.insertion(), at);
        }
        writeFrame(at, s.deltaRotRefitDeg(), s.instabilityEvent(), s.requestPending(),
                s.pendingRequest(), s.retryGapBlocking(), s.skylinePresent(), s.skylineValid(),
                s.westAvailable(), s.westValid(), s.syncDtS());
    }

    private void writeFrame(NavigationOutput o, @Nullable Double deltaRotRefit, boolean instability,
                            boolean requestPending, @Nullable SearchScheduler.Request pending,
                            boolean retryBlocking, boolean skylinePresent, boolean northValid,
                            boolean westAvailable, boolean westValid, @Nullable Double syncDtS)
            throws IOException {
        PlanarPosition l = o.localPosition();
        PlanarPosition g = o.globalPositionOrNull();
        AnchorLineage.Snapshot s = o.lineage();
        StringBuilder b = new StringBuilder(400);
        b.append(o.frameIndex()).append(',').append(o.timestampS()).append(',')
         .append(o.voSuccess()).append(',').append(o.translationUsable()).append(',')
         .append(l.eastM()).append(',').append(l.northM()).append(',')
         .append(num(o.headingDeg())).append(',').append(o.headingStatus().name()).append(',')
         .append(o.headingValid()).append(',')
         .append(o.globalPositionValid()).append(',').append(o.globalPoseValid()).append(',')
         .append(g == null ? "" : Double.toString(g.eastM())).append(',')
         .append(g == null ? "" : Double.toString(g.northM())).append(',')
         .append(o.segmentId()).append(',').append(o.alignmentEpochId()).append(',')
         .append(field(s.anchorReferenceId())).append(',')
         .append(s.anchorReferenceId() == null ? "" : Integer.toString(s.anchorFrameIndex())).append(',')
         .append(s.exposureAnchor()).append(',').append(s.exposureSince()).append(',')
         .append(s.effectiveExposure()).append(',')
         .append(s.instabilityAnchor()).append(',').append(s.instabilitySince()).append(',')
         .append(s.effectiveInstability()).append(',')
         .append(o.timestampS() - s.anchorTimestampS()).append(',')
         .append(o.event().wireName()).append(',')
         .append(field(deltaRotRefit)).append(',').append(instability).append(',')
         .append(requestPending).append(',')
         .append(pending == null ? "" : pending.originCause().wireName()).append(',')
         .append(pending == null ? "" : pending.cause().wireName()).append(',')
         .append(retryBlocking).append(',')
         .append(skylinePresent).append(',').append(northValid).append(',')
         .append(westAvailable).append(',').append(westValid).append(',')
         .append(field(syncDtS));
        frames.write(b.toString());
        frames.newLine();
        frames.flush();
    }

    public void writeHardLoss(HardLossEvent e) throws IOException {
        writeHardLoss(e, LOSS_NATURAL);
    }

    /** A hard loss the estimator itself reported. */
    public static final String LOSS_NATURAL = "natural";
    /** A hard loss a test or experiment forced onto a declared frame. */
    public static final String LOSS_SYNTHETIC = "synthetic";

    /**
     * @param lossSource {@link #LOSS_NATURAL} or {@link #LOSS_SYNTHETIC} — never blank. A forced
     *                   loss takes the same downstream path as an observed one, so the record is
     *                   the only thing that keeps them apart in a later analysis.
     */
    public void writeHardLoss(HardLossEvent e, String lossSource) throws IOException {
        EventRow row = new EventRow(e.frameIndex(), e.timestampS(), "hard_loss");
        row.segmentId = e.closedSegmentId();
        row.detail = "new_segment_id=" + e.newSegmentId() + ";last_valid_frame=" + e.lastValidFrameIndex()
                + ";heading_known_across_gap=" + e.headingKnownAcrossGap();
        row.before = e.lastValidGlobalPosition();
        row.lossSource = lossSource;
        row.globalPositionValidBefore = e.lastValidGlobalPosition() != null;
        row.headingValid = e.headingKnownAcrossGap();
        write(row);
    }

    public void writeDropout(TranslationDropoutEvent e) throws IOException {
        EventRow row = new EventRow(e.frameIndex(), e.timestampS(), "translation_dropout");
        row.segmentId = e.segmentId();
        row.detail = "height_status=" + e.heightStatus().name() + ";heading_status="
                + e.headingStatus().name() + ";last_valid_frame=" + e.lastValidFrameIndex();
        row.before = e.lastValidGlobalPosition();
        row.headingStatus = e.headingStatus().name();
        write(row);
    }

    public void writeRequest(SearchScheduler.Request r, NavigationOutput at) throws IOException {
        EventRow row = new EventRow(r.originFrame(), r.originTimestampS(), "search_requested");
        row.segmentId = at.segmentId();
        row.epochId = at.alignmentEpochId();
        row.cause = r.originCause().wireName();
        row.originCause = r.originCause().wireName();
        row.originFrame = r.originFrame();
        write(row);
    }

    public void writeEscalation(SearchScheduler.Escalation e, NavigationOutput at) throws IOException {
        EventRow row = new EventRow(e.frameIndex(), e.timestampS(), "search_escalated");
        row.segmentId = at.segmentId();
        row.epochId = at.alignmentEpochId();
        row.escalationCause = e.cause().wireName();
        row.changedCause = e.changedCause();
        row.overrodeRetryGap = e.overrodeRetryGap();
        write(row);
    }

    public void writeAttempt(SearchScheduler.Attempt a, @Nullable AcceptanceGate.Decision d,
                             @Nullable MatchEvidence e, NavigationOutput at) throws IOException {
        EventRow row = new EventRow(a.frameIndex(), a.timestampS(), "attempt");
        row.segmentId = at.segmentId();
        row.epochId = at.alignmentEpochId();
        row.cause = a.cause().wireName();
        row.originCause = a.originCause().wireName();
        row.originFrame = a.originFrame();
        row.overrodeRetryGap = a.underRetryOverride();
        row.reason = a.outcome().wireName();
        row.detail = a.detail() + ";escalations=" + a.escalationCount();
        row.referenceId = a.candidateReferenceId();
        row.local = at.localPosition();
        row.headingDeg = at.headingDeg();
        if (d != null) {
            row.verdict = d.verdict().wireName() + ":" + d.reason().wireName();
            row.topRegionScore = d.topScore();
            row.regionMargin = d.margin();
            row.supportCount = d.supportCount();
            row.dualVerdict = d.dualVerdict().wireName();
            row.fusionRule = d.fusionRule();
            row.temporalRequired = d.temporalRequired();
            row.westScore = d.westScore();
            row.westMargin = d.westMargin();
            row.agreement = d.agreement();
        }
        if (e != null) {
            fillEvidence(row, e);
        }
        write(row);
    }

    public void writeReanchor(ReanchorEvent e) throws IOException {
        EventRow row = new EventRow(e.frameIndex(), e.timestampS(), "reanchor");
        row.segmentId = e.segmentId();
        row.epochId = e.alignmentEpochId();
        row.referenceId = e.referenceId();
        row.detail = "pose_rule=" + e.poseRuleId() + ";translation_only=true";
        row.before = e.globalBefore();
        row.after = e.globalAfter();
        row.local = e.queryLocalPosition();
        row.headingDeg = e.headingDeg();
        AlignmentTransform d = e.appliedDelta();
        if (d != null) {
            row.deltaE = d.tE();
            row.deltaN = d.tN();
        }
        row.positionJump = e.positionJumpM();
        row.frameGap = e.queryReferenceFrameGap();
        row.timeGap = e.queryReferenceTimeGapS();
        row.exposureBefore = e.lineageBefore().effectiveExposure();
        row.exposureAfter = e.lineageAfter().effectiveExposure();
        row.instabilityBefore = e.lineageBefore().effectiveInstability();
        row.instabilityAfter = e.lineageAfter().effectiveInstability();
        if (e.evidence() != null) {
            fillEvidence(row, e.evidence());
        }
        // e_before exists exactly when a position existed to be corrected. globalBefore() being
        // null IS the hard-loss-recovery case, and no baseline is fabricated across the gap.
        row.globalPositionValidBefore = e.globalBefore() != null;
        row.headingValid = Double.isFinite(e.headingDeg());
        row.recoveryCase = e.globalBefore() != null ? CASE_DRIFT_CORRECTION : CASE_HARD_LOSS_RECOVERY;
        row.selectedRefE = e.referencePosition().eastM();
        row.selectedRefN = e.referencePosition().northM();
        row.referenceAgeFrames = e.queryReferenceFrameGap();
        row.referenceAgeS = e.queryReferenceTimeGapS();
        if (e.globalBefore() != null) {
            row.estimatedQueryToReferenceM = e.globalBefore().distanceTo(e.referencePosition());
        }
        if (e.evidence() != null) {
            row.competitorExists = e.evidence().competingRegionScore() != null;
        }
        write(row);
    }

    private static void fillEvidence(EventRow row, MatchEvidence ev) {
        row.cause = ev.cause().wireName();
        row.originCause = ev.originCause().wireName();
        row.originFrame = ev.originFrame();
        row.matcher = ev.matcherVariant();
        row.fusionRule = ev.fusionRule();
        row.lagSamples = ev.winningLagSamples();
        row.overlapFraction = ev.overlapFraction();
        row.supportCount = ev.temporalSupportCount();
        row.temporalRequired = ev.temporalRequired();
        row.topRegionScore = ev.topRegionScore();
        row.competingRegionScore = ev.competingRegionScore();
        row.regionMargin = ev.regionMargin();
        row.northScore = ev.northScore();
        row.northMargin = ev.northMargin();
        row.westScore = ev.westScore();
        row.westMargin = ev.westMargin();
        row.westLag = ev.westLagSamples();
        row.agreement = ev.westAgreement();
        row.dualStatus = ev.dualStatus();
        row.dualVerdict = ev.dualVerdict();
    }

    public void writeInsertion(InsertionDecision d, NavigationOutput at) throws IOException {
        EventRow row = new EventRow(d.frameIndex(), d.timestampS(),
                d.inserted() ? "reference_inserted" : "reference_rejected");
        row.segmentId = at.segmentId();
        row.epochId = at.alignmentEpochId();
        row.reason = d.inserted() ? null : d.outcome().wireName();
        row.detail = (d.noveltyDistance() == null ? "" : "novelty=" + d.noveltyDistance())
                + (d.framesSinceLast() == null ? "" : ";frames_since_last=" + d.framesSinceLast());
        TrustedReference ref = d.reference();
        if (ref != null) {
            row.referenceId = ref.id();
            row.after = ref.positionGlobal();
            row.headingDeg = ref.headingDeg();
            row.headingStatus = ref.headingStatus().name();
            row.exposureBefore = ref.baselineExposure();
            row.instabilityBefore = ref.baselineInstability();
            row.poseSchema = ref.poseSchema();
            row.sourceAnchorId = ref.sourceAnchorId();
            row.sourceReanchorFrame = ref.sourceReanchorFrameIndex();
            row.detail += ";west_stored=" + ref.hasWest();
        }
        write(row);
    }

    public void writeRetrieval(RetrievalResult r, NavigationOutput at) throws IOException {
        writeRetrieval(r, at, null);
    }

    public void writeRetrieval(RetrievalResult r, NavigationOutput at, @Nullable Double voAux)
            throws IOException {
        EventRow row = new EventRow(r.queryFrameIndex(), r.queryTimestampS(), "retrieval");
        row.segmentId = at.segmentId();
        row.epochId = at.alignmentEpochId();
        row.referenceId = r.top() == null ? null : r.top().referenceId();
        row.reason = r.refusedReason();
        row.local = at.localPosition();
        row.headingDeg = at.headingDeg();
        row.matcher = r.matcherVariant();
        row.fusionRule = r.fusionRule();
        if (r.north() != null) {
            String[] cols = columns(r.north().candidates());
            row.topIds = cols[0];
            row.topScores = cols[1];
            row.topLags = cols[2];
            row.northMargin = r.north().regionMargin();
        }
        if (r.west() != null) {
            String[] cols = columns(r.west().candidates());
            row.westTopIds = cols[0];
            row.westTopScores = cols[1];
            row.westTopLags = cols[2];
            row.westMargin = r.west().regionMargin();
        }
        row.excludedIds = join(r.excludedRecentIds());
        row.databaseSize = r.databaseSize();
        row.consideredCount = r.consideredCount();
        row.topRegionScore = r.topRegionScore();
        row.competingRegionScore = r.competingRegionScore();
        row.regionMargin = r.regionMargin();
        if (r.top() != null) {
            row.lagSamples = r.top().lagSamples();
            row.overlapFraction = r.top().overlapFraction();
        }
        RetrievalResult.DualRegionEvidence d = r.dual();
        if (d != null) {
            row.northScore = d.northScore();
            row.westScore = d.westScore();
            row.agreement = d.agreement();
            row.dualStatus = d.status().wireName();
            if (r.west() != null && r.west().top() != null) {
                RetrievalResult.Candidate w = r.west().candidate(d.candidateReferenceId());
                row.westLag = w == null ? null : w.lagSamples();
            }
        }
        row.voAux = voAux;
        fillSnapSafety(row, at, r, r.top() == null ? null : r.top().referenceId());
        write(row);
    }

    /**
     * The snap-safety block: what an offline analysis needs in order to ask whether an accepted
     * correction was safe, and to keep the two cases of §14 apart.
     *
     * <p>Ground truth is deliberately absent — this runs online and has none. Everything here is
     * either the estimator's own state or the memory's, joined offline against the dataset's
     * ground truth by frame index.
     */
    private void fillSnapSafety(EventRow row, NavigationOutput at, @Nullable RetrievalResult r,
                                @Nullable Integer selectedReferenceId) {
        Boolean validBefore = positionValidBeforeDecision != null
                ? positionValidBeforeDecision : at.globalPositionValid();
        row.globalPositionValidBefore = validBefore;
        row.headingValid = at.headingValid();
        row.recoveryCase = validBefore ? CASE_DRIFT_CORRECTION : CASE_HARD_LOSS_RECOVERY;
        if (r != null) {
            row.competitorExists = r.competingRegionScore() != null;
            row.scoredNorth = r.north() == null ? null : r.north().scoredCount();
            row.scoredWest = r.west() == null ? null : r.west().scoredCount();
            RetrievalResult.DualRegionEvidence d = r.dual();
            if (d != null) {
                row.northTopReferenceId = d.northTopReferenceId();
                row.westTopReferenceId = d.westTopReferenceId();
                row.topSeparationM = d.topSeparationM();
            }
        }
        if (memory != null && selectedReferenceId != null) {
            TrustedReference ref = memory.get(selectedReferenceId);
            if (ref != null) {
                row.selectedRefE = ref.positionGlobal().eastM();
                row.selectedRefN = ref.positionGlobal().northM();
                row.referenceAgeFrames = at.frameIndex() - ref.frameIndex();
                row.referenceAgeS = at.timestampS() - ref.timestampS();
                // Only meaningful while a persistent position exists: across an unknown gap the
                // query has no persistent position to measure from, and none is invented.
                if (Boolean.TRUE.equals(validBefore) && row.before != null) {
                    row.estimatedQueryToReferenceM = row.before.distanceTo(ref.positionGlobal());
                } else if (Boolean.TRUE.equals(validBefore) && at.globalPositionValid()) {
                    row.estimatedQueryToReferenceM =
                            at.globalPosition().orElseThrow().distanceTo(ref.positionGlobal());
                }
            }
        }
    }

    private static String[] columns(List<RetrievalResult.Candidate> candidates) {
        StringBuilder ids = new StringBuilder();
        StringBuilder scores = new StringBuilder();
        StringBuilder lags = new StringBuilder();
        for (RetrievalResult.Candidate c : candidates) {
            if (ids.length() > 0) {
                ids.append(';');
                scores.append(';');
                lags.append(';');
            }
            ids.append(c.referenceId());
            scores.append(c.score());
            lags.append(c.lagSamples());
        }
        return new String[]{ids.toString(), scores.toString(), lags.toString()};
    }

    public void writeRejection(NavigationAligner.RejectedCandidate c, NavigationOutput at)
            throws IOException {
        EventRow row = new EventRow(c.frameIndex(), c.timestampS(), "candidate_rejected");
        row.segmentId = at.segmentId();
        row.epochId = at.alignmentEpochId();
        row.referenceId = c.candidateReferenceId();
        row.reason = c.reason();
        write(row);
    }

    /** One events row, every optional column absent (empty) unless set by name. */
    private static final class EventRow {
        final int frame;
        final double ts;
        final String kind;
        Integer segmentId, epochId, referenceId, frameGap, databaseSize, consideredCount;
        Integer lagSamples, westLag, supportCount, originFrame, sourceAnchorId, sourceReanchorFrame;
        String reason, detail, topIds, topScores, topLags, westTopIds, westTopScores, westTopLags;
        String excludedIds, cause, originCause, escalationCause, matcher, verdict, fusionRule;
        String dualStatus, dualVerdict, poseSchema, headingStatus;
        PlanarPosition before, after, local;
        Double deltaE, deltaN, positionJump, timeGap, headingDeg = Double.NaN;
        Double topRegionScore, competingRegionScore, regionMargin, overlapFraction, voAux;
        Double northScore, northMargin, westScore, westMargin;
        Long exposureBefore, exposureAfter;
        Boolean instabilityBefore, instabilityAfter, agreement, changedCause, overrodeRetryGap, temporalRequired;
        // Snap-safety block.
        String lossSource, recoveryCase;
        Boolean globalPositionValidBefore, headingValid, competitorExists;
        Integer scoredNorth, scoredWest, northTopReferenceId, westTopReferenceId, referenceAgeFrames;
        Double topSeparationM, selectedRefE, selectedRefN, estimatedQueryToReferenceM, referenceAgeS;

        EventRow(int frame, double ts, String kind) {
            this.frame = frame;
            this.ts = ts;
            this.kind = kind;
        }
    }

    /** Column order is {@link #EVENTS_HEADER}'s; keep the two in step. */
    private void write(EventRow r) throws IOException {
        StringBuilder b = new StringBuilder(900);
        b.append(r.frame).append(',').append(r.ts).append(',').append(r.kind).append(',')
         .append(field(r.segmentId)).append(',').append(field(r.epochId)).append(',')
         .append(field(r.referenceId)).append(',')
         .append(text(r.reason)).append(',').append(text(r.detail)).append(',')
         .append(pos(r.before)).append(',').append(pos(r.after)).append(',')
         .append(pos(r.local)).append(',').append(num(r.headingDeg == null ? Double.NaN : r.headingDeg)).append(',')
         .append(field(r.deltaE)).append(',').append(field(r.deltaN)).append(',')
         .append(field(r.positionJump)).append(',')
         .append(field(r.frameGap)).append(',').append(field(r.timeGap)).append(',')
         .append(field(r.exposureBefore)).append(',').append(field(r.exposureAfter)).append(',')
         .append(flag(r.instabilityBefore)).append(',').append(flag(r.instabilityAfter)).append(',')
         .append(text(r.topIds)).append(',').append(text(r.topScores)).append(',')
         .append(text(r.topLags)).append(',')
         .append(text(r.westTopIds)).append(',').append(text(r.westTopScores)).append(',')
         .append(text(r.westTopLags)).append(',')
         .append(text(r.excludedIds)).append(',')
         .append(field(r.databaseSize)).append(',').append(field(r.consideredCount)).append(',')
         .append(field(r.topRegionScore)).append(',').append(field(r.competingRegionScore)).append(',')
         .append(field(r.regionMargin)).append(',')
         .append(field(r.northScore)).append(',').append(field(r.northMargin)).append(',')
         .append(field(r.westScore)).append(',').append(field(r.westMargin)).append(',')
         .append(field(r.westLag)).append(',').append(flag(r.agreement)).append(',')
         .append(text(r.dualStatus)).append(',').append(text(r.dualVerdict)).append(',')
         .append(text(r.fusionRule)).append(',').append(flag(r.temporalRequired)).append(',')
         .append(text(r.cause)).append(',').append(text(r.originCause)).append(',')
         .append(field(r.originFrame)).append(',').append(text(r.escalationCause)).append(',')
         .append(flag(r.changedCause)).append(',').append(flag(r.overrodeRetryGap)).append(',')
         .append(text(r.matcher)).append(',')
         .append(field(r.lagSamples)).append(',').append(field(r.overlapFraction)).append(',')
         .append(field(r.supportCount)).append(',').append(text(r.verdict)).append(',')
         .append(field(r.voAux)).append(',')
         .append(text(r.poseSchema)).append(',').append(field(r.sourceAnchorId)).append(',')
         .append(field(r.sourceReanchorFrame)).append(',').append(text(r.headingStatus)).append(',')
         .append(text(r.lossSource)).append(',').append(text(r.recoveryCase)).append(',')
         .append(flag(r.globalPositionValidBefore)).append(',').append(flag(r.headingValid)).append(',')
         .append(flag(r.competitorExists)).append(',')
         .append(field(r.scoredNorth)).append(',').append(field(r.scoredWest)).append(',')
         .append(field(r.northTopReferenceId)).append(',').append(field(r.westTopReferenceId)).append(',')
         .append(field(r.topSeparationM)).append(',')
         .append(field(r.selectedRefE)).append(',').append(field(r.selectedRefN)).append(',')
         .append(field(r.estimatedQueryToReferenceM)).append(',')
         .append(field(r.referenceAgeFrames)).append(',').append(field(r.referenceAgeS));
        events.write(b.toString());
        events.newLine();
        events.flush();
    }

    private static String text(@Nullable String s) {
        return s == null ? "" : s.replace(',', ';').replace('\n', ' ');
    }

    private static String flag(@Nullable Boolean b) {
        return b == null ? "" : b.toString();
    }

    private static String pos(@Nullable PlanarPosition p) {
        return p == null ? "," : p.eastM() + "," + p.northM();
    }

    private static String field(@Nullable Number v) {
        return v == null ? "" : v.toString();
    }

    /** {@code NaN} means "no such quantity for this frame"; written as an empty cell, never as a token. */
    private static String num(double v) {
        return Double.isNaN(v) ? "" : Double.toString(v);
    }

    private static String join(List<Integer> ids) {
        StringBuilder b = new StringBuilder();
        for (int id : ids) {
            if (b.length() > 0) {
                b.append(';');
            }
            b.append(id);
        }
        return b.toString();
    }

    @Override
    public void close() throws IOException {
        frames.close();
        events.close();
    }
}
