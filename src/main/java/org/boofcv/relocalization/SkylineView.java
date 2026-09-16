package org.boofcv.relocalization;

import javax.annotation.Nullable;

/**
 * One world-locked skyline view of an observation — {@code north} or {@code west} — either a usable
 * descriptor or an explicit reason it is not. Validity is a hard fact, never a probability
 * (design §6.1).
 *
 * @param view          {@code "north"} or {@code "west"}
 * @param descriptor    the descriptor, or {@code null} when the view is invalid
 * @param invalidReason why it is invalid, or {@code null} when it is valid
 * @param observationId the SKY-side observation id of this view, when known
 */
public record SkylineView(String view, @Nullable SkylineDescriptor descriptor,
                          @Nullable String invalidReason, @Nullable String observationId) {

    public static final String NORTH = "north";
    public static final String WEST = "west";

    public SkylineView {
        if (!NORTH.equals(view) && !WEST.equals(view)) {
            throw new IllegalArgumentException("view must be north or west, got " + view);
        }
        if ((descriptor == null) == (invalidReason == null)) {
            throw new IllegalArgumentException("exactly one of descriptor / invalidReason must be present");
        }
    }

    public static SkylineView valid(String view, SkylineDescriptor descriptor, @Nullable String observationId) {
        return new SkylineView(view, descriptor, null, observationId);
    }

    public static SkylineView invalid(String view, String reason, @Nullable String observationId) {
        return new SkylineView(view, null, reason, observationId);
    }

    public boolean valid() {
        return descriptor != null;
    }
}
