package org.boofcv.relocalization;

import java.util.Arrays;

/**
 * The retrieval descriptor: a fixed-length, mean-removed skyline height profile, scored by
 * normalised cross-correlation — the SKY lane's frozen {@code C0} primitive
 * ({@code skyline.descriptors.height_profile} + {@code profile_distance}, {@code COMP-SKY-001},
 * {@code DEC-SKY-007} naming C0 the confirmatory baseline).
 *
 * <p>The Python primitive is mirrored exactly so a Java-side retrieval result is comparable with
 * the SKY lane's evaluations: elevation as a fraction of image height, resampled to
 * {@code n_samples} (256 in every SKY record), mean removed, and
 *
 * <pre>
 *   NCC(p, q) = ⟨p − p̄, q − q̄⟩ / (‖p − p̄‖ · ‖q − q̄‖ + 1e-12)
 * </pre>
 *
 * <p>NCC lies in {@code [-1, 1]}; 1 is identical shape. Amplitude is discarded by the norm
 * division. <b>No shift search</b> — this is C0, not the bounded-lag C1-32 ({@code DEC-SKY-007});
 * C1 is a later, separately declared variant and is not implemented here.
 *
 * <p>A profile with no structure (sample standard deviation below the configured floor) is
 * <b>degenerate</b> and refused at construction ({@link #of} returns {@code null}), matching
 * {@code hsreloc.retrieval.profile.is_degenerate}: a featureless query must resolve to an explicit
 * refusal rather than to whichever reference wins a comparison between two flat signals.
 *
 * <p>Immutable. The mean-removed profile is stored; the norm is cached so scoring is a dot product.
 */
public final class SkylineDescriptor {

    private final double[] centred;
    private final double norm;

    private SkylineDescriptor(double[] centred, double norm) {
        this.centred = centred;
        this.norm = norm;
    }

    /**
     * Builds a descriptor from a profile already resampled to the configured length.
     *
     * @param profile  elevation samples (larger = skyline higher in the frame), any constant
     *                 offset is removed here
     * @param expectedLength the configured descriptor length; a mismatch is a construction error,
     *                 never silently resampled
     * @param degenerateStdFloor profiles whose sample SD is below this are refused
     * @return the descriptor, or {@code null} when the profile is degenerate
     */
    public static SkylineDescriptor of(double[] profile, int expectedLength, double degenerateStdFloor) {
        if (profile == null) {
            throw new IllegalArgumentException("profile is required");
        }
        if (profile.length != expectedLength) {
            throw new IllegalArgumentException("profile length " + profile.length
                    + " != configured descriptor length " + expectedLength);
        }
        if (expectedLength < 2) {
            throw new IllegalArgumentException("descriptor length must be >= 2");
        }
        double mean = 0.0;
        for (double v : profile) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("profile contains a non-finite sample");
            }
            mean += v;
        }
        mean /= profile.length;
        double[] centred = new double[profile.length];
        double sumSq = 0.0;
        for (int i = 0; i < profile.length; i++) {
            centred[i] = profile[i] - mean;
            sumSq += centred[i] * centred[i];
        }
        // numpy.std: population SD (ddof = 0), which is what is_degenerate compares to its floor.
        double std = Math.sqrt(sumSq / profile.length);
        if (std < degenerateStdFloor) {
            return null;
        }
        return new SkylineDescriptor(centred, Math.sqrt(sumSq));
    }

    /** Normalised cross-correlation with {@code other}, in {@code [-1, 1]}. */
    public double ncc(SkylineDescriptor other) {
        if (other == null) {
            throw new IllegalArgumentException("other is required");
        }
        if (other.centred.length != centred.length) {
            throw new IllegalArgumentException("descriptor lengths differ: " + centred.length
                    + " vs " + other.centred.length);
        }
        double dot = 0.0;
        for (int i = 0; i < centred.length; i++) {
            dot += centred[i] * other.centred[i];
        }
        return dot / (norm * other.norm + 1e-12);
    }

    /** {@code 1 − NCC}, in {@code [0, 2]}; 0 is identical shape. The novelty metric. */
    public double distance(SkylineDescriptor other) {
        return 1.0 - ncc(other);
    }

    public int length() {
        return centred.length;
    }

    /** A copy of the mean-removed profile. */
    public double[] centredProfile() {
        return Arrays.copyOf(centred, centred.length);
    }

    /** Internal storage, read-only, for the matchers' window scoring. Never mutate. */
    double[] centred() {
        return centred;
    }

    /**
     * The frozen primitive on an arbitrary aligned window: {@code a[aFrom..aFrom+len)} against
     * {@code b[bFrom..bFrom+len)}, each window's own mean removed
     * ({@code hsreloc.matchers.base.BaseMatcher._ncc} → {@code profile_distance}).
     */
    static double windowNcc(double[] a, int aFrom, double[] b, int bFrom, int len) {
        double ma = 0.0, mb = 0.0;
        for (int i = 0; i < len; i++) {
            ma += a[aFrom + i];
            mb += b[bFrom + i];
        }
        ma /= len;
        mb /= len;
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < len; i++) {
            double x = a[aFrom + i] - ma, y = b[bFrom + i] - mb;
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
    }
}
