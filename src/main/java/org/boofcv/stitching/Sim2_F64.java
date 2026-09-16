package org.boofcv.stitching;

import georegression.struct.InvertibleTransform;

import javax.annotation.Nullable;

/**
 * A 2-D similarity transform — uniform scale, rotation and translation, 4 DoF
 * ({@code DEC-VO-006}, {@code LIT-VO-005} §1):
 *
 * <pre>
 *   x' = s R(θ) x + t
 * </pre>
 *
 * <h2>Why this type exists rather than BoofCV's</h2>
 *
 * <p>BoofCV 0.44 has {@code boofcv.struct.geo.ScaleTranslateRotate2D}, which holds the same four
 * numbers — but it is a plain struct, <b>not</b> a {@link InvertibleTransform}, so it has none of
 * {@code concat}, {@code invert}, {@code createInstance} or {@code reset}. Those are exactly the
 * operations {@code ImageMotionPointTrackerKey} and {@code StitchingFromMotion2D} call on the motion
 * model every frame, so BoofCV's struct cannot be driven through the motion pipeline at all. Every
 * other class in that pipeline is already generic over {@code IT extends InvertibleTransform<IT>}
 * and needs no change ({@code LIT-VO-005} §5).
 *
 * <h2>Storage: the linear parameterisation, not (scale, angle)</h2>
 *
 * <p>The linear part is held as {@code (a, b) = (s cos θ, s sin θ)} rather than as {@code (s, θ)}.
 * Three reasons, all of them about correctness rather than speed:
 *
 * <ol>
 *   <li><b>Composition and inversion become exact complex arithmetic</b> with no trigonometry:
 *       composing two similarities multiplies {@code a + ib}, and inverting reciprocates it. A
 *       {@code (s, θ)} representation would call {@code sin}/{@code cos} on every one of the
 *       ~12,000 compositions per flight and accumulate their rounding.</li>
 *   <li><b>No angle wrapping exists to get wrong.</b> {@code θ} is recovered on demand by
 *       {@code atan2}, always in {@code (−π, π]}, and the transform itself never stores a
 *       representation that could drift past a branch cut.</li>
 *   <li><b>It is the parameterisation the estimator is linear in</b> ({@code LIT-VO-005} eq. 5), so
 *       {@link GenerateSimilarity2D} writes its least-squares solution in without converting.</li>
 * </ol>
 *
 * <p>The corresponding matrix, and therefore the constant Jacobian everywhere in the image, is
 *
 * <pre>
 *   [ a  −b  tx ]
 *   [ b   a  ty ]
 *   [ 0   0   1 ]
 * </pre>
 *
 * <p>so {@code det = a² + b² = s²} and the singular values are both {@code s}: a similarity has
 * anisotropy exactly 1 and no perspective, which is why {@link RigidMotionDecomposition} reads a
 * fitted similarity back as its own {@code (θ, s)} with no special case ({@code LIT-VO-005} §1).
 *
 * <h2>Group law</h2>
 *
 * <p>Sim(2) is closed under composition and inversion, so the estimator's per-frame
 * {@code worldToKey ∘ keyToCurr} and {@code currToWorld} are exact within the family — no
 * projection, no residual leaked into the accumulation. Following georegression's convention,
 * {@code this.concat(second, result)} means {@code result(p) = second(this(p))}.
 *
 * <p><b>Degeneracy is a failure, not a value.</b> {@code s = 0} (equivalently {@code a = b = 0}) has
 * no inverse; {@link #invert} throws rather than emitting infinities into the accumulation, and
 * {@link GenerateSimilarity2D} refuses to produce such a model in the first place.
 */
public class Sim2_F64 implements InvertibleTransform<Sim2_F64> {

    private static final long serialVersionUID = 1L;

    /** {@code s·cos θ} — the (1,1) and (2,2) entries of the linear block. */
    public double a;
    /** {@code s·sin θ} — the (2,1) entry; the (1,2) entry is {@code −b}. */
    public double b;
    /** Translation, applied after the linear part. */
    public double tx, ty;

    /** The identity. */
    public Sim2_F64() {
        reset();
    }

    /** Directly from the linear parameterisation {@code (a, b) = (s cos θ, s sin θ)}. */
    public Sim2_F64(double a, double b, double tx, double ty) {
        this.a = a;
        this.b = b;
        this.tx = tx;
        this.ty = ty;
    }

    /** From scale, rotation (radians) and translation — the human-readable form. */
    public static Sim2_F64 of(double scale, double thetaRad, double tx, double ty) {
        return new Sim2_F64(scale * Math.cos(thetaRad), scale * Math.sin(thetaRad), tx, ty);
    }

    public Sim2_F64 setTo(double a, double b, double tx, double ty) {
        this.a = a;
        this.b = b;
        this.tx = tx;
        this.ty = ty;
        return this;
    }

    /** Uniform scale {@code s = √(a² + b²)}. */
    public double scale() {
        return Math.hypot(a, b);
    }

    /** Rotation in radians, {@code atan2(b, a)} — always in {@code (−π, π]}. */
    public double thetaRad() {
        return Math.atan2(b, a);
    }

    @Override
    public int getDimension() {
        return 2;
    }

    @Override
    public Sim2_F64 createInstance() {
        return new Sim2_F64();
    }

    @Override
    public Sim2_F64 setTo(Sim2_F64 target) {
        return setTo(target.a, target.b, target.tx, target.ty);
    }

    /**
     * {@code result(p) = second(this(p))}, per georegression's convention.
     *
     * <p>{@code second(this(p)) = s₂R₂(s₁R₁p + t₁) + t₂ = (s₁s₂)R(θ₁+θ₂)p + (s₂R₂t₁ + t₂)}, i.e.
     * the linear parts multiply as complex numbers and {@code second}'s linear part is applied to
     * {@code this}'s translation.
     */
    @Override
    public Sim2_F64 concat(Sim2_F64 second, @Nullable Sim2_F64 result) {
        if (result == null) result = new Sim2_F64();
        double na = second.a * a - second.b * b;
        double nb = second.a * b + second.b * a;
        double ntx = second.a * tx - second.b * ty + second.tx;
        double nty = second.b * tx + second.a * ty + second.ty;
        return result.setTo(na, nb, ntx, nty);
    }

    /**
     * {@code p ↦ (1/s)R(−θ)(p − t)}.
     *
     * @throws IllegalStateException when the scale is zero — a collapsed model has no inverse, and
     *                               silently returning infinities would corrupt the accumulation
     *                               rather than fail it.
     */
    @Override
    public Sim2_F64 invert(@Nullable Sim2_F64 inverse) {
        double det = a * a + b * b;
        if (det == 0.0) {
            throw new IllegalStateException("Sim2_F64 with zero scale has no inverse");
        }
        if (inverse == null) inverse = new Sim2_F64();
        double ia = a / det;
        double ib = -b / det;
        // inverse translation = -(1/s)R(-θ) t, i.e. the inverted linear part applied to -t
        double itx = -(ia * tx - ib * ty);
        double ity = -(ib * tx + ia * ty);
        return inverse.setTo(ia, ib, itx, ity);
    }

    @Override
    public void reset() {
        a = 1.0;
        b = 0.0;
        tx = 0.0;
        ty = 0.0;
    }

    @Override
    public String toString() {
        return String.format("Sim2_F64[s=%.9f theta=%.9f rad tx=%.6f ty=%.6f]",
                scale(), thetaRad(), tx, ty);
    }
}
