package org.boofcv.relocalization;

import javax.annotation.Nullable;

/**
 * One skyline observation at one physical location and instant: the <b>North</b> view (required —
 * the primary view every SKY record was produced with) and, when the capture carried one, the
 * synchronised <b>West</b> view ({@code EXP-SKY-011}'s dual-direction model; {@code DEC-INT-003}).
 *
 * <p>Three West states are distinct and never conflated: {@link #westAvailable()} false — no West
 * view was captured (every North-only session, including all pre-2026-09-06 data); available but
 * {@link #westValid()} false — captured, extraction refused or degenerate; valid. A missing or
 * invalid West is never an agreement, never a disagreement, and never a copy of North.
 *
 * <p>Nothing here opens an image: profiles arrive from the validated Python SKY pipeline through the
 * exported profile file ({@link SkylineProfileFile}; {@code DEC-INT-002}), where the two views of one
 * row were paired by the same capture callback ({@code sim_observation_id_raw}) by the exporter.
 *
 * @param frameIndex the VO frame the observation is keyed to
 * @param timestampS the VO frame's timestamp
 * @param north      the North view
 * @param west       the West view, or {@code null} when none was captured
 * @param syncDtS    the exporter's time residual between the skyline capture and the VO frame it
 *                   was paired to (skyline − VO, seconds), or {@code null} when keyed by index
 */
public record SkylineObservation(int frameIndex, double timestampS, SkylineView north,
                                 @Nullable SkylineView west, @Nullable Double syncDtS) {

    public SkylineObservation {
        if (north == null || !SkylineView.NORTH.equals(north.view())) {
            throw new IllegalArgumentException("a North view is required");
        }
        if (west != null && !SkylineView.WEST.equals(west.view())) {
            throw new IllegalArgumentException("the second view must be the West view");
        }
        if (west != null && north.descriptor() != null && west.descriptor() == north.descriptor()) {
            throw new IllegalArgumentException("the West view must not be the North descriptor");
        }
    }

    // ---------------------------------------------------------------- North-only factories

    public static SkylineObservation valid(int frameIndex, double timestampS, SkylineDescriptor descriptor) {
        return new SkylineObservation(frameIndex, timestampS,
                SkylineView.valid(SkylineView.NORTH, descriptor, null), null, null);
    }

    public static SkylineObservation valid(int frameIndex, double timestampS, SkylineDescriptor descriptor,
                                           String observationId) {
        return new SkylineObservation(frameIndex, timestampS,
                SkylineView.valid(SkylineView.NORTH, descriptor, observationId), null, null);
    }

    public static SkylineObservation invalid(int frameIndex, double timestampS, String reason) {
        return new SkylineObservation(frameIndex, timestampS,
                SkylineView.invalid(SkylineView.NORTH, reason, null), null, null);
    }

    public static SkylineObservation invalid(int frameIndex, double timestampS, String reason,
                                             String observationId) {
        return new SkylineObservation(frameIndex, timestampS,
                SkylineView.invalid(SkylineView.NORTH, reason, observationId), null, null);
    }

    /** A synchronised North + West observation. */
    public static SkylineObservation dual(int frameIndex, double timestampS, SkylineView north,
                                          SkylineView west) {
        if (west == null) {
            throw new IllegalArgumentException("use a North-only factory when there is no West view");
        }
        return new SkylineObservation(frameIndex, timestampS, north, west, null);
    }

    // ---------------------------------------------------------------- validity

    /** Eligible as a query or an insertion candidate: the North view is valid. */
    public boolean valid() {
        return north.valid();
    }

    public boolean northValid() {
        return north.valid();
    }

    /** A West view was captured for this observation (valid or not). */
    public boolean westAvailable() {
        return west != null;
    }

    public boolean westValid() {
        return west != null && west.valid();
    }

    @Nullable
    public SkylineDescriptor northDescriptor() {
        return north.descriptor();
    }

    @Nullable
    public SkylineDescriptor westDescriptor() {
        return west == null ? null : west.descriptor();
    }

    /** The North view's SKY-side observation id, when known. */
    @Nullable
    public String observationId() {
        return north.observationId();
    }

    /** Why the North view is invalid, or {@code null}. */
    @Nullable
    public String invalidReason() {
        return north.invalidReason();
    }

    /** Why the West view is unusable: "unavailable", its reason, or {@code null} when valid. */
    @Nullable
    public String westUnusableReason() {
        if (west == null) {
            return "unavailable";
        }
        return west.invalidReason();
    }
}
