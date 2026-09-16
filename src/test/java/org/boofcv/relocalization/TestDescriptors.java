package org.boofcv.relocalization;

/** Synthetic profiles for the T1 tests: sinusoids whose NCC falls off with phase difference. */
final class TestDescriptors {

    static final int N = 256;
    static final double FLOOR = 1e-6;

    private TestDescriptors() {
    }

    static double[] sineProfile(double phase) {
        double[] p = new double[N];
        for (int i = 0; i < N; i++) {
            double x = (double) i / N;
            p[i] = 0.5 + 0.2 * Math.sin(2 * Math.PI * (2 * x + phase))
                    + 0.05 * Math.sin(2 * Math.PI * (7 * x + 3 * phase));
        }
        return p;
    }

    static SkylineDescriptor sine(double phase) {
        SkylineDescriptor d = SkylineDescriptor.of(sineProfile(phase), N, FLOOR);
        if (d == null) {
            throw new AssertionError("test profile unexpectedly degenerate");
        }
        return d;
    }
}
