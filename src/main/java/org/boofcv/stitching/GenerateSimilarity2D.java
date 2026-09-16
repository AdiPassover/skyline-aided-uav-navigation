package org.boofcv.stitching;

import boofcv.struct.geo.AssociatedPair;
import org.ddogleg.fitting.modelset.ModelGenerator;

import java.util.List;

/**
 * Least-squares fit of a {@link Sim2_F64} to point correspondences — <b>Umeyama's closed form</b>
 * specialised to two dimensions ({@code LIT-VO-005} §3; Umeyama 1991, already {@code LIT-002};
 * Horn 1987 gives the same solution from quaternions).
 *
 * <h2>The solution</h2>
 *
 * <p>With centroids {@code p̄, q̄} and centred points {@code p̃ᵢ, q̃ᵢ}:
 *
 * <pre>
 *   a = Σ ⟨p̃ᵢ, q̃ᵢ⟩            b = Σ (p̃ᵢₓ q̃ᵢᵧ − p̃ᵢᵧ q̃ᵢₓ)          d = Σ ‖p̃ᵢ‖²
 *   (linear part) = (a/d, b/d)                    t = q̄ − (a/d, b/d) ⊗ p̄
 * </pre>
 *
 * <p>which is simultaneously (i) Umeyama's {@code R = USVᵀ}, {@code s = tr(DS)/σ_p²} written out for
 * 2×2 without forming an SVD, and (ii) the ordinary linear least-squares solution under the
 * substitution {@code (α, β) = (s cos θ, s sin θ)}, because the model is <em>linear</em> in those
 * coordinates ({@code LIT-VO-005} eq. 5). The equivalence matters: it is what makes this the
 * standard optimal fit for its model class, in the same sense that {@code GenerateAffine2D} and
 * {@code GenerateHomographyLinear} are for theirs — which is the fairness condition
 * {@code EXP-VO-009} needs.
 *
 * <h2>Why not BoofCV's {@code GenerateScaleTranslateRotate2D}</h2>
 *
 * <p>Because it does not fit a similarity. Its own Javadoc: <i>"First the affine transform is found
 * using the standard linear equation… NOTE: The found solution is not going to be optimal due to the
 * initial approximation using an affine transform."</i> It calls {@code GenerateAffine2D}, projects
 * the 2×2 block to {@code UVᵀ}, and copies the affine translation <b>unrefit</b>. That is — up to
 * the scale convention — precisely what {@code RIGID_MOTION} already does at readout time, so an
 * experiment built on it would compare a readout against itself. See {@code DEC-VO-006}
 * Alternative A.
 *
 * <h2>Minimum sample size: 3, not 2</h2>
 *
 * <p>A similarity has 4 DoF and is determined exactly by <b>2</b> correspondences. This generator
 * reports <b>3</b>, deliberately, so that it is <em>sample-matched to the affine arm</em>:
 * ddogleg's {@code Ransac} takes its sample size from {@code getMinimumPoints()} and derives one
 * RNG per trial independently of that size, so two arms with equal sample size and an equal track
 * list draw <em>identical</em> index sequences. Holding it at 3 removes a variable rather than
 * adding one ({@code DEC-VO-006} <i>Rationale</i>). BoofCV's own similarity generator also uses 3.
 *
 * <p>The cost is stated rather than hidden: at three points the affine fit is exactly determined and
 * averages nothing, while this fit is over-determined by two equations and does — and that averaging
 * is part of the model-class effect under test, quantified in closed form as
 * {@code (2 + κ + 1/κ)/4} by {@code LIT-VO-005} eq. (8). The 2-point variant is measured on
 * synthetic data by {@code EXP-VO-009} rather than left as an unexamined confound.
 *
 * <h2>Degeneracy</h2>
 *
 * <p>Returns {@code false} — never a NaN model — when the sample has no spatial extent
 * ({@code d = 0}, all points coincident) or when the cross-covariance vanishes ({@code a = b = 0}),
 * which would leave the rotation undefined and the scale zero. RANSAC treats a {@code false} as a
 * skipped hypothesis, which is the correct handling: a degenerate minimal sample carries no
 * information and must not be allowed to win by accident.
 */
public class GenerateSimilarity2D implements ModelGenerator<Sim2_F64, AssociatedPair> {

    @Override
    public boolean generate(List<AssociatedPair> dataSet, Sim2_F64 output) {
        final int n = dataSet.size();
        if (n < 2) {
            return false;
        }

        double p1x = 0, p1y = 0, p2x = 0, p2y = 0;
        for (int i = 0; i < n; i++) {
            AssociatedPair pair = dataSet.get(i);
            p1x += pair.p1.x;
            p1y += pair.p1.y;
            p2x += pair.p2.x;
            p2y += pair.p2.y;
        }
        p1x /= n;
        p1y /= n;
        p2x /= n;
        p2y /= n;

        double a = 0, b = 0, d = 0;
        for (int i = 0; i < n; i++) {
            AssociatedPair pair = dataSet.get(i);
            double ux = pair.p1.x - p1x, uy = pair.p1.y - p1y;
            double vx = pair.p2.x - p2x, vy = pair.p2.y - p2y;
            a += ux * vx + uy * vy;
            b += ux * vy - uy * vx;
            d += ux * ux + uy * uy;
        }

        if (d == 0.0 || (a == 0.0 && b == 0.0)) {
            return false;   // no spatial extent, or no recoverable rotation/scale
        }

        double la = a / d;
        double lb = b / d;
        if (!Double.isFinite(la) || !Double.isFinite(lb)) {
            return false;
        }

        // t = q̄ − L p̄, with L the linear block just found
        output.setTo(la, lb,
                p2x - (la * p1x - lb * p1y),
                p2y - (lb * p1x + la * p1y));
        return true;
    }

    @Override
    public int getMinimumPoints() {
        return 3;
    }
}
