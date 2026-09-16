package org.boofcv.relocalization;

/**
 * A profile matcher: an alignment followed by the frozen NCC primitive, exactly as the SKY lane's
 * {@code hsreloc.matchers} ladder defines it ({@code DEC-INT-002}; {@code LIT-SKY-005}).
 *
 * <p>Two variants exist here and no more: {@link #frozenNcc() C0}, the pointwise baseline every
 * SKY primary number was produced with, and {@link #boundedLagNcc C1}, the owner's post-close-out
 * operational selection ({@code DEC-SKY-007}). Both score with the same primitive, so their scores
 * sit on one scale and any difference is attributable to the lag freedom alone. Parity with the
 * Python implementation is asserted on a fixture produced by the Python code, not by re-derivation
 * ({@code C1ParityTest}).
 */
public interface SkylineMatcher {

    String variant();

    MatchResult match(SkylineDescriptor query, SkylineDescriptor reference);

    /** The matcher a config names. */
    static SkylineMatcher fromConfig(RelocalizationConfig c) {
        c.validate();
        return RelocalizationConfig.MATCHER_C0.equals(c.matcher)
                ? frozenNcc()
                : boundedLagNcc(c.maxLagSamples, c.minOverlapFrac);
    }

    /** C0: pointwise NCC, no lag, no scale, no warp. */
    static SkylineMatcher frozenNcc() {
        return new SkylineMatcher() {
            @Override
            public String variant() {
                return RelocalizationConfig.MATCHER_C0;
            }

            @Override
            public MatchResult match(SkylineDescriptor query, SkylineDescriptor reference) {
                int n = query.length();
                return new MatchResult(variant(), query.ncc(reference), 0, n, 1.0, true, null, 1, null);
            }
        };
    }

    /**
     * C1: {@code max over |δ| ≤ maxLag of NCC(q(x), r(x + δ))}, integer lags only, exhaustive over
     * the bound, comparing only the overlapping samples re-normalised on that window, refusing any
     * alignment whose overlap is below {@code minOverlapFrac}. Ties break to the least-transformed
     * alignment — smallest {@code |δ|}, then the negative one — as
     * {@code hsreloc.matchers.base.BaseMatcher._best} does.
     */
    static SkylineMatcher boundedLagNcc(int maxLagSamples, double minOverlapFrac) {
        if (maxLagSamples < 0) {
            throw new IllegalArgumentException("maxLagSamples must be >= 0");
        }
        if (!(minOverlapFrac > 0.0 && minOverlapFrac <= 1.0)) {
            throw new IllegalArgumentException("minOverlapFrac must be in (0, 1]");
        }
        return new SkylineMatcher() {
            @Override
            public String variant() {
                return RelocalizationConfig.MATCHER_C1;
            }

            @Override
            public MatchResult match(SkylineDescriptor query, SkylineDescriptor reference) {
                if (query.length() != reference.length()) {
                    throw new IllegalArgumentException("profiles must be the same length ("
                            + query.length() + " vs " + reference.length() + ")");
                }
                double[] q = query.centred();
                double[] r = reference.centred();
                int n = q.length;
                int floor = Math.max(2, (int) Math.ceil(minOverlapFrac * n));

                double bestScore = Double.NEGATIVE_INFINITY;
                int bestLag = 0, bestOverlap = 0;
                boolean found = false;
                Double atZero = null;
                int searched = 0;
                for (int lag = -maxLagSamples; lag <= maxLagSamples; lag++) {
                    int lo = Math.max(0, -lag);
                    int hi = Math.min(n - 1, n - 1 - lag);
                    int overlap = hi - lo + 1;
                    if (overlap < floor) {
                        continue;
                    }
                    searched++;
                    double s = SkylineDescriptor.windowNcc(q, lo, r, lo + lag, overlap);
                    if (lag == 0) {
                        atZero = s;
                    }
                    // Python: min over (-score, |shift|, shift) — strictly better score wins;
                    // on an exact tie the smaller |lag| wins, then the more negative lag. Lags are
                    // visited from -L upward, so "later wins only if strictly better or smaller
                    // in magnitude" reproduces that order exactly.
                    boolean better;
                    if (!found) {
                        better = true;
                    } else if (s != bestScore) {
                        better = s > bestScore;
                    } else if (Math.abs(lag) != Math.abs(bestLag)) {
                        better = Math.abs(lag) < Math.abs(bestLag);
                    } else {
                        better = lag < bestLag;
                    }
                    if (better) {
                        bestScore = s;
                        bestLag = lag;
                        bestOverlap = overlap;
                        found = true;
                    }
                }
                if (!found) {
                    return new MatchResult(variant(), Double.NEGATIVE_INFINITY, 0, 0, 0.0, false,
                            null, searched, "no lag in [-" + maxLagSamples + ", " + maxLagSamples
                            + "] keeps overlap >= " + String.format("%.2f", minOverlapFrac));
                }
                return new MatchResult(variant(), bestScore, bestLag, bestOverlap,
                        (double) bestOverlap / n, true, atZero, searched, null);
            }
        };
    }
}
