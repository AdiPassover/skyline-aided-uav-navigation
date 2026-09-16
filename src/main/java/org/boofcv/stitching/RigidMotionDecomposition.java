package org.boofcv.stitching;

import lombok.Getter;

/**
 * Polar/SVD decomposition of a 2×2 image-motion Jacobian into the parts {@code DEC-VO-004}
 * Alternative E treats differently: a <b>rigid rotation</b> that navigation uses, a <b>uniform
 * scale</b> that is observed but not applied, and <b>anisotropy/shear</b> that are diagnostics only.
 *
 * <h2>The decomposition</h2>
 *
 * <p>Any {@code J ∈ ℝ^{2×2}} with {@code det J > 0} factors as {@code J = R P} with {@code R} a
 * proper rotation and {@code P} symmetric positive-definite (polar decomposition). Equivalently,
 * from the SVD {@code J = U Σ Vᵀ} with {@code Σ = diag(σ₁ ≥ σ₂ > 0)}: {@code R = U Vᵀ} and
 * {@code P = V Σ Vᵀ}. {@code R} is the <b>closest proper rotation to {@code J} in the Frobenius
 * norm</b> (Higham's nearest-orthogonal-matrix result), which is what makes it the defensible
 * "in-plane rotation of this transform" rather than one edge's angle — the quantity
 * {@code COMP-001} §3.6's {@code getRotation()} used, and the one {@code EXP-VO-004} R6 measured to
 * cost 30–80° of yaw error once the transform is anisotropic.
 *
 * <p>This class computes {@code R}'s angle in closed form rather than through an SVD routine. For a
 * 2×2 matrix the polar rotation is
 *
 * <pre>
 *   θ = atan2(a₂₁ − a₁₂, a₁₁ + a₂₂)
 * </pre>
 *
 * which is exact, allocation-free and branch-free. (Derivation: write {@code J = R P}; since
 * {@code P} is symmetric, the antisymmetric part of {@code RᵀJ} vanishes, and expanding
 * {@code R(θ)ᵀ J} gives {@code sin θ (a₁₁ + a₂₂) = cos θ (a₂₁ − a₁₂)}.) The singular values follow
 * from the two Frobenius invariants without forming {@code JᵀJ}'s eigenvectors.
 *
 * <h2>Why {@code √|det J|} is the uniform-scale observable</h2>
 *
 * <p>{@code det J = σ₁σ₂} is the <b>area</b> scale, so {@code √|det J| = √(σ₁σ₂)} is the
 * <b>geometric mean of the singular values</b> — the linear scale of the isotropic transform with
 * the same area change. Three properties make it the right scalar for this system's semantics, and
 * they are the reason it is used rather than assumed:
 *
 * <ol>
 *   <li><b>It is exact under the intended physical model.</b> For a nadir camera over a plane,
 *       pure translation plus a height change induces an exact similarity with linear scale
 *       {@code h₀/h_k} ({@code LIT-VO-003} §2), for which {@code σ₁ = σ₂} and {@code √det = h₀/h_k}
 *       identically. Any other symmetric function of the singular values agrees here too, so this
 *       case does not by itself select {@code √det}.</li>
 *   <li><b>It is multiplicative under composition</b>: {@code det(AB) = det A · det B}, so the
 *       accumulated scale is the product of the per-frame increments and {@code log √det} is
 *       additive. Neither {@code (σ₁+σ₂)/2} nor {@code σ₁} has that property, and without it a
 *       per-frame increment could not be integrated into a per-run diagnostic at all.</li>
 *   <li><b>It is the projection onto the similarity group in the natural metric</b>: the scalar
 *       {@code s} minimising {@code ‖P − sI‖_F} is {@code (σ₁+σ₂)/2}, but the scalar minimising the
 *       <em>logarithmic</em> (scale-invariant) distance {@code (log σ₁ − log s)² + (log σ₂ − log s)²}
 *       is {@code √(σ₁σ₂)}. Scale here is a multiplicative quantity whose errors compound
 *       multiplicatively ({@code EXP-VO-004} R2), so the log metric is the appropriate one.</li>
 * </ol>
 *
 * <p><b>And the case where it is not enough is reported rather than hidden.</b> A single scalar
 * describes {@code P} only when {@code σ₁ ≈ σ₂}; {@link #anisotropy()} says by how much that fails,
 * and {@code LIT-VO-003} §4 gives the physical ceiling ({@code 1/cos θ} — 1.04 at the 15.5° maximum
 * tilt measured on {@code HKairport01}). Above that ceiling the "uniform scale" is a summary of a
 * transform that is not uniform, and the diagnostic is what says so.
 *
 * <h2>Reflections</h2>
 *
 * <p>{@code det J ≤ 0} is a mirrored or singular estimate — a failure, not a scale
 * ({@code LIT-VO-003} §5, and the same reasoning that makes {@code DEC-003} reject reflections in
 * the evaluator). {@link #properRotation()} is false and the scale/anisotropy readings are marked
 * degenerate rather than fabricated.
 */
public final class RigidMotionDecomposition {

    /** Rotation angle of the polar factor, radians, in {@code (−π, π]}. */
    @Getter
    private double rotationRad;

    /** {@code √|det J|} — the isotropic-equivalent linear scale. See the class comment. */
    @Getter
    private double uniformScale;

    /** {@code σ₁/σ₂ ≥ 1}. 1 for a pure similarity; {@code 1/cos θ} under a tilt of θ. */
    @Getter
    private double anisotropy;

    /**
     * Relative departure of the stretch {@code P} from the nearest uniform scaling, in the same
     * log-metric that selects {@code √det}: {@code √((log σ₁ − log s)² + (log σ₂ − log s)²)} with
     * {@code s = √(σ₁σ₂)}, which reduces to {@code |log(σ₁/σ₂)| / √2}.
     *
     * <p>It is therefore a monotone function of {@link #anisotropy()} and carries no independent
     * information for a 2×2 map — reported because it is the magnitude in the metric the scale
     * observable is chosen under, and because {@code 0} is unambiguous where a ratio's {@code 1}
     * is not. The genuinely independent part of {@code P} is its axis, {@link #stretchAxisRad()}.
     */
    @Getter
    private double deformationMagnitude;

    /**
     * Orientation of the major stretch axis (the eigenvector of {@code P} for {@code σ₁}), radians
     * in {@code [0, π)}. Meaningless when {@link #anisotropy()} is 1; carried because a *systematic*
     * stretch axis distinguishes a tilt (axis fixed relative to the airframe) from tracking noise
     * (axis uniform), which the along/across-flow split in {@code EXP-VO-004} R2 probed indirectly.
     */
    @Getter
    private double stretchAxisRad;

    /** False when {@code det J ≤ 0} — a mirrored or singular estimate. Scale is then not meaningful. */
    @Getter
    private boolean properRotation;

    /**
     * Decomposes the 2×2 Jacobian {@code [a11, a12, a21, a22]} (row-major).
     *
     * @return this, for chaining
     */
    public RigidMotionDecomposition set(double a11, double a12, double a21, double a22) {
        double det = a11 * a22 - a12 * a21;
        properRotation = det > 0.0;

        // Polar rotation, closed form (see class comment).
        rotationRad = Math.atan2(a21 - a12, a11 + a22);

        // Singular values without forming an eigen-decomposition: for a 2x2 matrix,
        //   sigma1 + sigma2 = sqrt((a11+a22)^2 + (a21-a12)^2)   [ = |J| in the "rotation" part ]
        //   sigma1 - sigma2 = sqrt((a11-a22)^2 + (a21+a12)^2)   [ = the "stretch" part ]
        double sum = Math.hypot(a11 + a22, a21 - a12);
        double diff = Math.hypot(a11 - a22, a21 + a12);
        double sigma1 = 0.5 * (sum + diff);
        double sigma2 = 0.5 * Math.abs(sum - diff);   // abs: sum < diff exactly when det < 0

        uniformScale = Math.sqrt(Math.abs(det));
        anisotropy = sigma2 > 0.0 ? sigma1 / sigma2 : Double.POSITIVE_INFINITY;
        deformationMagnitude = sigma2 > 0.0 ? Math.abs(Math.log(anisotropy)) / Math.sqrt(2.0)
                                            : Double.POSITIVE_INFINITY;
        // Major axis of P = V diag(sigma) V^T. With the 2x2 SVD's two auxiliary angles
        // a2 = atan2(a21 - a12, a11 + a22) (= the polar rotation, the angle of U) and
        // a1 = atan2(a21 + a12, a11 - a22), V sits at (a1 - a2)/2, so that is P's major axis.
        // The orientation (rather than the direction) is what is meaningful, hence mod pi.
        // Verified against a reconstructed R*(V S V^T) in RigidMotionDecompositionTest.
        stretchAxisRad = 0.5 * (Math.atan2(a21 + a12, a11 - a22) - rotationRad);
        stretchAxisRad = ((stretchAxisRad % Math.PI) + Math.PI) % Math.PI;
        if (stretchAxisRad < 0) {
            stretchAxisRad += Math.PI;
        }
        return this;
    }

    /** Rotation angle of the polar factor in degrees, normalised to {@code [0, 360)}. */
    public double rotationDegrees() {
        return (Math.toDegrees(rotationRad) + 360.0) % 360.0;
    }

    /** Natural log of {@link #uniformScale()} — the additive form, for accumulating increments. */
    public double logUniformScale() {
        return Math.log(uniformScale);
    }
}
