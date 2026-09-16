package org.boofcv.stitching;

import georegression.struct.InvertibleTransform;
import georegression.struct.affine.Affine2D_F64;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.point.Point2D_F64;
import georegression.transform.affine.AffinePointOps_F64;
import georegression.transform.homography.HomographyPointOps_F64;

import javax.annotation.Nullable;

/**
 * The only model-specific operation {@link MotionModelStitchingEstimator} needs: building the
 * scale-and-centre transform that places the first frame into the mosaic.
 *
 * <p>Everything else the estimator does to a motion model — {@code invert}, {@code concat},
 * {@code setTo}, {@code reset} — is already declared on {@link InvertibleTransform}, and the pose
 * itself is read from {@code getImageCorners()}, which is a {@code Quadrilateral_F64} regardless of
 * model (see {@code COMP-001} §3.6). So this one factory method is the entire model-specific
 * surface, which is what makes an affine-vs-homography comparison a genuine model-only ablation
 * rather than a comparison of two pose implementations.
 *
 * <p>All three constants below produce the <em>same geometric transform</em> — uniform scale
 * {@code s} about the origin followed by translation {@code (tx, ty)} — expressed in their
 * respective model. A scale-and-translate is exactly representable in all three, so no
 * approximation is introduced at initialisation for any arm.
 *
 * <p>Introduced by {@code DEC-VO-002} for {@code EXP-VO-002}, extended by {@code DEC-VO-006} for
 * {@code EXP-VO-009}. {@link #HOMOGRAPHY} (8 DoF, projective) is the shipped production path;
 * {@link #AFFINE} (6 DoF) and {@link #SIMILARITY} (4 DoF) are selectable experimental alternatives
 * that nothing in production selects. The three are nested — similarity ⊂ affine ⊂ projective —
 * which is what makes a comparison between them a model-class ablation rather than a change of
 * family ({@code LIT-VO-001}, {@code LIT-VO-005} §1).
 *
 * @param <IT> the 2D motion model
 */
public interface MotionModelSupport<IT extends InvertibleTransform<IT>> {

    /**
     * @param scale uniform scale factor applied about the origin
     * @param tx    translation in x, applied after the scale
     * @param ty    translation in y, applied after the scale
     * @return a fresh transform of this model's type representing that scale-then-translate
     */
    IT shrinkTransform(double scale, double tx, double ty);

    /**
     * Maps a point through {@code transform}.
     *
     * <p>Exists here because {@link InvertibleTransform} declares {@code concat}/{@code invert}/
     * {@code setTo}/{@code reset}/{@code createInstance} but <b>no point-application method</b>, so
     * applying a transform to a pixel is the one further model-specific operation
     * {@code LogicalNavigationState} needs ({@code DEC-VO-003}). Keeping it on this interface means
     * the navigation layer stays model-agnostic rather than branching on the transform type.
     *
     * @param out optional storage; a new instance is allocated when null
     */
    Point2D_F64 apply(IT transform, double x, double y, @Nullable Point2D_F64 out);

    /**
     * The 2×2 Jacobian of the point map at {@code (x, y)}, row-major
     * {@code [∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]}.
     *
     * <p>This is the local linearisation {@code DEC-VO-004}'s rigid readout decomposes: its polar
     * rotation is the in-plane rotation increment, {@code √|det|} the isotropic-equivalent scale,
     * and its singular-value ratio the anisotropy ({@code LIT-VO-003} §5–§6).
     *
     * <p>For an affine transform the Jacobian is the constant linear block and {@code (x, y)} is
     * ignored. For a homography it varies across the image whenever the perspective row is
     * nonzero, which is exactly why the evaluation point must be stated: this codebase always
     * evaluates at the <b>image centre</b>, the standard approximation to the principal point.
     *
     * @param out storage of length ≥ 4; a new array is allocated when null
     */
    double[] jacobian(IT transform, double x, double y, @Nullable double[] out);

    /**
     * Magnitude of the projective (perspective) part, as the maximum image-edge displacement it
     * induces: {@code max(|h₃₁|·width, |h₃₂|·height)} on the {@code h₃₃ = 1} normalisation.
     * <b>Exactly 0 for the affine and similarity models</b>, neither of which has a perspective
     * row.
     *
     * <p>Diagnostic only. It measures how far the transform departs from the affine family, and
     * therefore how meaningless a single global scale would be for it ({@code LIT-VO-003} §6).
     */
    double perspectiveMagnitude(IT transform, double width, double height);

    /**
     * The transform in homography form, exactly. An affine or similarity transform embeds
     * losslessly (last row {@code (0, 0, 1)}); a homography is copied. Used by diagnostics that persist the logical
     * transform in one model-independent layout ({@code EXP-VO-004}).
     *
     * @param out optional storage; a new instance is allocated when null
     */
    Homography2D_F64 asHomography(IT transform, @Nullable Homography2D_F64 out);

    /** Short stable identifier, written into run-record provenance. */
    String id();

    /** The 8-DoF projective model — the shipped default (BoofCV {@code Homography2D_F64}). */
    MotionModelSupport<Homography2D_F64> HOMOGRAPHY = new MotionModelSupport<>() {
        @Override
        public Homography2D_F64 shrinkTransform(double scale, double tx, double ty) {
            return new Homography2D_F64(
                    scale, 0, tx,
                    0, scale, ty,
                    0, 0, 1);
        }

        @Override
        public Point2D_F64 apply(Homography2D_F64 transform, double x, double y,
                                 @Nullable Point2D_F64 out) {
            return HomographyPointOps_F64.transform(transform, x, y, out);
        }

        @Override
        public double[] jacobian(Homography2D_F64 h, double x, double y, @Nullable double[] out) {
            if (out == null) out = new double[4];
            double w = h.a31 * x + h.a32 * y + h.a33;
            double u = (h.a11 * x + h.a12 * y + h.a13) / w;
            double v = (h.a21 * x + h.a22 * y + h.a23) / w;
            // J = ([h1; h2]_{:,1:2} - x' h3_{1:2}^T) / w   (LIT-VO-003 eq. 5)
            out[0] = (h.a11 - u * h.a31) / w;
            out[1] = (h.a12 - u * h.a32) / w;
            out[2] = (h.a21 - v * h.a31) / w;
            out[3] = (h.a22 - v * h.a32) / w;
            return out;
        }

        @Override
        public double perspectiveMagnitude(Homography2D_F64 h, double width, double height) {
            double n = h.a33;
            if (n == 0.0) return Double.POSITIVE_INFINITY;
            return Math.max(Math.abs(h.a31 / n) * width, Math.abs(h.a32 / n) * height);
        }

        @Override
        public Homography2D_F64 asHomography(Homography2D_F64 transform,
                                             @Nullable Homography2D_F64 out) {
            if (out == null) out = new Homography2D_F64();
            out.setTo(transform);
            return out;
        }

        @Override
        public String id() {
            return "homography";
        }
    };

    /**
     * The 6-DoF affine model — experimental, selected only by {@code EXP-VO-002}.
     *
     * <p>Note the constructor argument order: {@code Affine2D_F64(a11, a12, a21, a22, tx, ty)}
     * stores the linear block and the translation separately, so the diagonal scale entries are the
     * 1st and 4th arguments, not the 1st and 5th as the row-major homography constructor above
     * might suggest.
     */
    MotionModelSupport<Affine2D_F64> AFFINE = new MotionModelSupport<>() {
        @Override
        public Affine2D_F64 shrinkTransform(double scale, double tx, double ty) {
            return new Affine2D_F64(scale, 0, 0, scale, tx, ty);
        }

        @Override
        public Point2D_F64 apply(Affine2D_F64 transform, double x, double y,
                                 @Nullable Point2D_F64 out) {
            return AffinePointOps_F64.transform(transform, x, y, out);
        }

        @Override
        public double[] jacobian(Affine2D_F64 t, double x, double y, @Nullable double[] out) {
            if (out == null) out = new double[4];
            // Constant across the image: an affine map's Jacobian IS its linear block.
            out[0] = t.a11;
            out[1] = t.a12;
            out[2] = t.a21;
            out[3] = t.a22;
            return out;
        }

        @Override
        public double perspectiveMagnitude(Affine2D_F64 t, double width, double height) {
            return 0.0;   // exact, by construction: the affine family has no perspective row
        }

        @Override
        public Homography2D_F64 asHomography(Affine2D_F64 t, @Nullable Homography2D_F64 out) {
            if (out == null) out = new Homography2D_F64();
            out.setTo(t.a11, t.a12, t.tx,
                      t.a21, t.a22, t.ty,
                      0, 0, 1);
            return out;
        }

        @Override
        public String id() {
            return "affine";
        }
    };

    /**
     * The 4-DoF similarity model — experimental, selected only by {@code EXP-VO-009}
     * ({@code DEC-VO-006}).
     *
     * <p>This is the model {@code LIT-VO-003} §2 shows is <em>physically exact</em> for a nadir
     * camera translating over a plane: uniform scale {@code h₀/h_k}, in-plane rotation, translation,
     * and no perspective. The other two constants above approximate it from above, with 6 and 8
     * degrees of freedom; this one estimates it directly ({@code LIT-VO-005}).
     *
     * <p><b>Every method below is exact, and three of them are exactly degenerate — which is the
     * point.</b> A similarity's Jacobian is its linear block {@code [a, −b; b, a]} everywhere in the
     * image, so {@link RigidMotionDecomposition} recovers the fitted {@code (θ, s)} identically and
     * reports anisotropy {@code 1}, deformation {@code 0} and perspective {@code 0} <em>by
     * computation, not by special-casing</em>. The diagnostics are therefore structurally
     * not-applicable rather than fabricated, and the {@code RIGID_MOTION} readout applied to a
     * similarity is the identity readout.
     */
    MotionModelSupport<Sim2_F64> SIMILARITY = new MotionModelSupport<>() {
        @Override
        public Sim2_F64 shrinkTransform(double scale, double tx, double ty) {
            // Exact: a uniform scale about the origin followed by a translation IS a similarity,
            // with rotation zero — so this arm's initialisation introduces no approximation either,
            // exactly as DEC-VO-002 established for the affine and homography arms.
            return new Sim2_F64(scale, 0.0, tx, ty);
        }

        @Override
        public Point2D_F64 apply(Sim2_F64 t, double x, double y, @Nullable Point2D_F64 out) {
            if (out == null) out = new Point2D_F64();
            out.setTo(t.a * x - t.b * y + t.tx,
                      t.b * x + t.a * y + t.ty);
            return out;
        }

        @Override
        public double[] jacobian(Sim2_F64 t, double x, double y, @Nullable double[] out) {
            if (out == null) out = new double[4];
            // Constant across the image, and equal to s·R(θ): the singular values are both s, so
            // the polar decomposition of this returns (θ, s) exactly and anisotropy exactly 1.
            out[0] = t.a;
            out[1] = -t.b;
            out[2] = t.b;
            out[3] = t.a;
            return out;
        }

        @Override
        public double perspectiveMagnitude(Sim2_F64 t, double width, double height) {
            return 0.0;   // exact, by construction: the similarity family has no perspective row
        }

        @Override
        public Homography2D_F64 asHomography(Sim2_F64 t, @Nullable Homography2D_F64 out) {
            if (out == null) out = new Homography2D_F64();
            out.setTo(t.a, -t.b, t.tx,
                      t.b, t.a, t.ty,
                      0, 0, 1);
            return out;
        }

        @Override
        public String id() {
            return "similarity";
        }
    };
}
