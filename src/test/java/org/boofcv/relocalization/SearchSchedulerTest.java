package org.boofcv.relocalization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Exposure deadline, pending-request, retry-gap, override and <b>attribution</b> semantics
 * (design §5, §13; test N). T1.
 */
public class SearchSchedulerTest {

    private static RelocalizationConfig config(long deadline, Double timeMax, int retryGap) {
        RelocalizationConfig c = new RelocalizationConfig();
        c.searchExposureBound = deadline;
        c.searchTimeMaxS = timeMax;
        c.minRetryGapFrames = retryGap;
        return c;
    }

    /** Drives an aligner so the scheduler sees genuine outputs and lineage snapshots. */
    private static final class World {
        final NavigationAligner aligner = new NavigationAligner();
        final SearchScheduler scheduler;
        int frame = -1;
        int segment = 0;

        World(RelocalizationConfig c) {
            scheduler = new SearchScheduler(c);
        }

        SearchScheduler.Request step(boolean success, boolean instability) {
            frame++;
            NavigationOutput out;
            if (success) {
                out = aligner.observe(TestSamples.ok(frame, segment, frame, 0, 10.0));
            } else {
                segment++;
                out = aligner.observe(TestSamples.loss(frame, segment, 10.0));
            }
            return scheduler.onFrame(out, instability);
        }

        SearchScheduler.Request step() {
            return step(true, false);
        }

        SearchScheduler.Request dropout() {
            frame++;
            NavigationOutput out = aligner.observe(TestSamples.noHeight(frame, segment, frame - 1, 0, 10.0));
            return scheduler.onFrame(out, false);
        }
    }

    @Test
    public void exposureDeadlineArmsExactlyWhenEffReachesNSearch() {
        World w = new World(config(5, null, 30));
        w.step();                                   // frame 0: init, E = 0
        for (int i = 1; i < 5; i++) {
            assertNull(w.step(), "E = " + i + " is below the deadline");
            assertFalse(w.scheduler.requestPending());
        }
        SearchScheduler.Request r = w.step();       // E = 5
        assertNotNull(r);
        assertEquals(SearchScheduler.Cause.SCHEDULED_EXPOSURE, r.cause());
        assertEquals(SearchScheduler.Cause.SCHEDULED_EXPOSURE, r.originCause());
        assertEquals(5, r.originFrame());
        assertFalse(r.escalated());
        assertTrue(w.scheduler.requestPending());
    }

    @Test
    public void elapsedTimeMaximumArmsWhenExposureIsStillBelowTheDeadline() {
        World w = new World(config(1000, 0.45, 30));   // 0.1 s per frame → due at frame 5
        w.step();
        for (int i = 1; i < 5; i++) {
            assertNull(w.step());
        }
        SearchScheduler.Request r = w.step();
        assertNotNull(r);
        assertEquals(SearchScheduler.Cause.SCHEDULED_TIME, r.cause());
    }

    @Test
    public void aRequestIsStickyAcrossIneligibleFramesAndNeverDuplicated() {
        World w = new World(config(3, null, 30));
        for (int i = 0; i <= 3; i++) {
            w.step();
        }
        assertTrue(w.scheduler.requestPending());
        for (int i = 0; i < 20; i++) {
            assertNull(w.step(), "already pending: no second request");
            assertTrue(w.scheduler.requestPending());
            assertFalse(w.scheduler.executable(w.frame, false), "no eligible skyline → not executable");
        }
        assertEquals(1, w.scheduler.requests().size());
        assertEquals(20, w.scheduler.suppressedDuplicates());
        assertTrue(w.scheduler.executable(w.frame, true));
    }

    @Test
    public void lowInstabilityNeverDelaysTheBaselineAndHighInstabilityRequestsNow() {
        World w = new World(config(50, null, 30));
        w.step();
        w.step();
        SearchScheduler.Request r = w.step(true, true);      // high delta_rot_refit at E = 2
        assertNotNull(r);
        assertEquals(SearchScheduler.Cause.INSTABILITY, r.cause());
        assertTrue(w.scheduler.executable(w.frame, true));

        // A low value is the absence of the event: the deadline still fires on its own schedule.
        World quiet = new World(config(4, null, 30));
        quiet.step();
        for (int i = 1; i < 4; i++) {
            assertNull(quiet.step(true, false));
        }
        assertNotNull(quiet.step(true, false));
    }

    @Test
    public void hardLossRequestsSearchImmediately() {
        World w = new World(config(50, null, 30));
        w.step();
        w.step();
        SearchScheduler.Request r = w.step(false, false);
        assertNotNull(r);
        assertEquals(SearchScheduler.Cause.HARD_LOSS, r.cause());
        assertTrue(r.cause().voIsSuspect());
    }

    @Test
    public void aTranslationDropoutRequestsSearchOncePerTransitionAndIsSuspect() {
        World w = new World(config(50, null, 30));
        w.step();
        w.step();
        SearchScheduler.Request r = w.dropout();
        assertNotNull(r);
        assertEquals(SearchScheduler.Cause.TRANSLATION_DROPOUT, r.cause());
        assertTrue(r.cause().voIsSuspect());
        assertNull(w.dropout(), "a second unusable frame while already unknown arms nothing new");
        assertNull(w.scheduler.escalationThisFrame());
        assertEquals(1, w.scheduler.requests().size());
        assertEquals(0, w.scheduler.escalations());
    }

    @Test
    public void anUnsuccessfulAttemptConsumesTheRequestAndTheRetryGapHoldsTheNextOne() {
        World w = new World(config(3, null, 5));
        for (int i = 0; i <= 3; i++) {
            w.step();
        }
        assertTrue(w.scheduler.executable(w.frame, true));
        w.scheduler.recordAttempt(w.frame, w.frame * 0.1, SearchScheduler.AttemptOutcome.REJECTED, 7, "weak");
        assertFalse(w.scheduler.requestPending(), "consumed");

        // The deadline re-arms immediately (exposure did not decrease), but execution waits.
        SearchScheduler.Request again = w.step();               // frame 4
        assertNotNull(again);
        assertEquals(SearchScheduler.Cause.SCHEDULED_EXPOSURE, again.cause());
        for (int i = 0; i < 4; i++) {                            // frames 4, 5, 6, 7: 1..4 < 5
            assertTrue(w.scheduler.retryGapBlocking(w.frame), "frame " + w.frame);
            assertFalse(w.scheduler.executable(w.frame, true));
            w.step();
        }
        // frame 8: 8 − 3 = 5 ≥ gap
        assertEquals(8, w.frame);
        assertFalse(w.scheduler.retryGapBlocking(w.frame));
        assertTrue(w.scheduler.executable(w.frame, true));
        assertEquals(8, w.aligner.lineage().effectiveExposure(), "exposure was never lowered by scheduling");
    }

    @Test
    public void aNewHardLossOrInstabilityOverridesTheRetryGap() {
        World w = new World(config(3, null, 50));
        for (int i = 0; i <= 3; i++) {
            w.step();
        }
        w.scheduler.recordAttempt(w.frame, w.frame * 0.1, SearchScheduler.AttemptOutcome.REJECTED, null, "x");
        w.step();
        assertTrue(w.scheduler.retryGapBlocking(w.frame));
        w.step(true, true);                                 // new instability event
        assertFalse(w.scheduler.retryGapBlocking(w.frame));
        assertTrue(w.scheduler.executable(w.frame, true));
        w.scheduler.recordAttempt(w.frame, w.frame * 0.1, SearchScheduler.AttemptOutcome.REJECTED, null, "x");
        w.step();
        assertTrue(w.scheduler.retryGapBlocking(w.frame));
        w.step(false, false);                               // hard loss
        assertFalse(w.scheduler.retryGapBlocking(w.frame));
        assertEquals(SearchScheduler.Cause.HARD_LOSS, w.scheduler.pendingRequest().cause(),
                "a suspect-VO trigger escalates a pending benign request rather than being a duplicate");
        assertEquals(2, w.scheduler.escalations(), "the instability at frame 5 escalated one too");
        assertTrue(w.scheduler.pendingRequest().cause().voIsSuspect());
    }

    /** Test N: one sticky request, and its origin survives the escalation. */
    @Test
    public void aScheduledRequestEscalatedByInstabilityKeepsItsOriginAndStaysOneRequest() {
        World w = new World(config(3, null, 50));
        for (int i = 0; i <= 3; i++) {
            w.step();                                        // armed at frame 3 by exposure
        }
        assertNull(w.step(true, true), "an escalation arms nothing new");   // frame 4: instability
        SearchScheduler.Escalation e = w.scheduler.escalationThisFrame();
        assertNotNull(e);
        assertEquals(4, e.frameIndex());
        assertEquals(SearchScheduler.Cause.INSTABILITY, e.cause());
        assertTrue(e.changedCause(), "SCHEDULED_EXPOSURE → INSTABILITY");
        assertFalse(e.overrodeRetryGap(), "no attempt had run, so no gap was blocking");

        SearchScheduler.Request p = w.scheduler.pendingRequest();
        assertEquals(SearchScheduler.Cause.SCHEDULED_EXPOSURE, p.originCause(), "the search opportunity was scheduled");
        assertEquals(3, p.originFrame());
        assertEquals(SearchScheduler.Cause.INSTABILITY, p.cause(), "the effective cause is the escalation's");
        assertTrue(p.escalated());
        assertEquals(1, p.escalations().size());
        assertEquals(1, w.scheduler.requests().size(), "still exactly one request");
        assertEquals(1, w.scheduler.escalationEvents().size());

        SearchScheduler.Attempt a = w.scheduler.recordAttempt(w.frame, w.frame * 0.1,
                SearchScheduler.AttemptOutcome.REJECTED, 2, "x");
        assertEquals(SearchScheduler.Cause.INSTABILITY, a.cause());
        assertEquals(SearchScheduler.Cause.SCHEDULED_EXPOSURE, a.originCause());
        assertEquals(3, a.originFrame());
        assertEquals(1, a.escalationCount());
        assertTrue(a.underRetryOverride(), "the suspect trigger's override was in force at execution");
        assertNull(w.scheduler.pendingRequest());
    }

    /** Whether an escalation actually changed execution is recorded: did it lift a retry gap? */
    @Test
    public void anEscalationRecordsWhetherItLiftedARetryGap() {
        World w = new World(config(3, null, 50));
        for (int i = 0; i <= 3; i++) {
            w.step();
        }
        w.scheduler.recordAttempt(w.frame, w.frame * 0.1, SearchScheduler.AttemptOutcome.REJECTED, null, "x");
        w.step();                                            // frame 4: re-armed, gap blocking
        assertTrue(w.scheduler.retryGapBlocking(w.frame));
        w.step(true, true);                                  // frame 5: instability
        SearchScheduler.Escalation e = w.scheduler.escalationThisFrame();
        assertNotNull(e);
        assertTrue(e.overrodeRetryGap(), "the gap was blocking; this escalation lifted it");
        assertTrue(e.changedCause());
        assertTrue(w.scheduler.executable(w.frame, true));
        assertEquals(4, w.scheduler.pendingRequest().originFrame(), "origin: the re-armed scheduled request");
    }

    @Test
    public void aSuspectTriggerOnAnAlreadySuspectRequestIsRecordedWithoutChangingTheCause() {
        World w = new World(config(50, null, 30));
        w.step();
        w.step();
        w.step(true, true);                                  // INSTABILITY armed at frame 2
        assertNull(w.step(false, false));                    // frame 3: hard loss while pending
        SearchScheduler.Escalation e = w.scheduler.escalationThisFrame();
        assertNotNull(e);
        assertEquals(SearchScheduler.Cause.HARD_LOSS, e.cause());
        assertFalse(e.changedCause(), "already suspect: the effective cause is unchanged");
        assertEquals(SearchScheduler.Cause.INSTABILITY, w.scheduler.pendingRequest().cause());
        assertEquals(SearchScheduler.Cause.INSTABILITY, w.scheduler.pendingRequest().originCause());
        assertEquals(1, w.scheduler.escalations());
        assertEquals(1, w.scheduler.requests().size());
    }

    @Test
    public void aRetainedCandidateKeepsTheRequestPendingAndBypassesTheRetryGap() {
        World w = new World(config(3, null, 50));
        for (int i = 0; i <= 3; i++) {
            w.step();
        }
        w.scheduler.recordAttempt(w.frame, w.frame * 0.1,
                SearchScheduler.AttemptOutcome.RETAINED_FOR_CONFIRMATION, 2, "unconfirmed");
        assertTrue(w.scheduler.requestPending());
        assertTrue(w.scheduler.confirmationNeeded());
        w.step();
        assertFalse(w.scheduler.retryGapBlocking(w.frame));
        assertTrue(w.scheduler.executable(w.frame, true), "the next eligible query confirms");
        w.scheduler.recordAttempt(w.frame, w.frame * 0.1, SearchScheduler.AttemptOutcome.ACCEPTED, 2, "ok");
        assertFalse(w.scheduler.requestPending());
        assertFalse(w.scheduler.confirmationNeeded());
    }

    @Test
    public void recordingAnAttemptWithoutARequestIsAnError() {
        World w = new World(config(100, null, 5));
        w.step();
        assertThrows(IllegalStateException.class, () -> w.scheduler.recordAttempt(0, 0.0,
                SearchScheduler.AttemptOutcome.REJECTED, null, ""));
    }
}
