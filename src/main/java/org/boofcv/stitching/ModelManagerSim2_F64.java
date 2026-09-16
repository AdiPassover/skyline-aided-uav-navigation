package org.boofcv.stitching;

import org.ddogleg.fitting.modelset.ModelManager;

/**
 * ddogleg allocation/copy manager for {@link Sim2_F64}, the direct analogue of georegression's
 * {@code ModelManagerAffine2D_F64} and {@code ModelManagerHomography2D_F64}. {@code Ransac} uses it
 * for nothing but creating and copying model instances, so this class carries no behaviour that
 * could differentiate the similarity arm from the other two ({@code DEC-VO-006}).
 */
public class ModelManagerSim2_F64 implements ModelManager<Sim2_F64> {

    @Override
    public Sim2_F64 createModelInstance() {
        return new Sim2_F64();
    }

    @Override
    public void copyModel(Sim2_F64 src, Sim2_F64 dst) {
        dst.setTo(src);
    }
}
