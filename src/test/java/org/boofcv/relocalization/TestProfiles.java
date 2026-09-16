package org.boofcv.relocalization;

/**
 * Aperiodic synthetic profiles for the P1 tests — the {@code hsreloc/tests/test_matchers.py}
 * signal family, sampled from a continuous function so a shifted copy is exact rather than
 * resampled. Four families with disjoint frequency sets stand in for four distinct places; the
 * periodic {@link TestDescriptors} sines are deliberately <em>not</em> used where a lag search is
 * involved, because a lag can align phase-shifted copies of a periodic signal. Families 4–7 are a
 * second, independent set standing in for the same four places seen from the West.
 */
final class TestProfiles {

    static final int N = 256;

    private TestProfiles() {
    }

    static double place(int family, double t) {
        return switch (family) {
            case 0 -> Math.sin(2 * Math.PI * t / 37.0) + 0.5 * Math.sin(2 * Math.PI * t / 13.0 + 1.0)
                    + 0.25 * Math.sin(2 * Math.PI * t / 71.0 - 0.4);
            case 1 -> Math.sin(2 * Math.PI * t / 23.0 + 0.7) + 0.6 * Math.sin(2 * Math.PI * t / 53.0)
                    + 0.2 * Math.sin(2 * Math.PI * t / 9.0 - 1.1);
            case 2 -> Math.sin(2 * Math.PI * t / 29.0 - 0.3) + 0.4 * Math.sin(2 * Math.PI * t / 17.0 + 2.0)
                    + 0.3 * Math.sin(2 * Math.PI * t / 61.0 + 0.9);
            case 3 -> Math.sin(2 * Math.PI * t / 41.0 + 1.4) + 0.7 * Math.sin(2 * Math.PI * t / 11.0)
                    + 0.2 * Math.sin(2 * Math.PI * t / 83.0 - 0.6);
            case 4 -> Math.sin(2 * Math.PI * t / 19.0 + 0.2) + 0.5 * Math.sin(2 * Math.PI * t / 47.0 - 0.8)
                    + 0.3 * Math.sin(2 * Math.PI * t / 7.0 + 1.7);
            case 5 -> Math.sin(2 * Math.PI * t / 31.0 - 1.2) + 0.6 * Math.sin(2 * Math.PI * t / 67.0 + 0.3)
                    + 0.2 * Math.sin(2 * Math.PI * t / 15.0);
            case 6 -> Math.sin(2 * Math.PI * t / 43.0 + 2.2) + 0.4 * Math.sin(2 * Math.PI * t / 21.0 - 0.5)
                    + 0.3 * Math.sin(2 * Math.PI * t / 89.0 + 1.1);
            case 7 -> Math.sin(2 * Math.PI * t / 27.0 - 0.9) + 0.7 * Math.sin(2 * Math.PI * t / 59.0 + 1.9)
                    + 0.2 * Math.sin(2 * Math.PI * t / 12.0 + 0.4);
            default -> throw new IllegalArgumentException("family " + family);
        };
    }

    /** Family {@code f} observed with a horizontal shift of {@code shift} samples. */
    static double[] profile(int family, double shift) {
        double[] p = new double[N];
        for (int i = 0; i < N; i++) {
            p[i] = place(family, i + shift);
        }
        return p;
    }

    static SkylineDescriptor descriptor(int family, double shift) {
        SkylineDescriptor d = SkylineDescriptor.of(profile(family, shift), N, 1e-9);
        if (d == null) {
            throw new AssertionError("test profile unexpectedly degenerate");
        }
        return d;
    }

    /** A North-only observation (West unavailable). */
    static SkylineObservation observation(int frame, double ts, int family, double shift) {
        return SkylineObservation.valid(frame, ts, descriptor(family, shift), "fam" + family + "_s" + shift);
    }

    /** A synchronised North + West observation of one place: North family {@code f}, West family {@code f + 4}. */
    static SkylineObservation dual(int frame, double ts, int family, double northShift, double westShift) {
        return SkylineObservation.dual(frame, ts,
                SkylineView.valid(SkylineView.NORTH, descriptor(family, northShift), "n" + family + "_s" + northShift),
                SkylineView.valid(SkylineView.WEST, descriptor(family + 4, westShift), "w" + family + "_s" + westShift));
    }

    /** North of place {@code f}, West of a DIFFERENT place {@code westFamily}: a disagreeing observation. */
    static SkylineObservation crossed(int frame, double ts, int family, int westFamily) {
        return SkylineObservation.dual(frame, ts,
                SkylineView.valid(SkylineView.NORTH, descriptor(family, 0), "n" + family),
                SkylineView.valid(SkylineView.WEST, descriptor(westFamily + 4, 0), "w" + westFamily));
    }

    /** North valid, West captured but invalid. */
    static SkylineObservation westInvalid(int frame, double ts, int family, double shift) {
        return SkylineObservation.dual(frame, ts,
                SkylineView.valid(SkylineView.NORTH, descriptor(family, shift), "n" + family),
                SkylineView.invalid(SkylineView.WEST, "extraction refused", "w" + family));
    }
}
