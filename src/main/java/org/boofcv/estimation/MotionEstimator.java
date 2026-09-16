package org.boofcv.estimation;

import org.boofcv.util.structs.Pose3D;

public interface MotionEstimator {

    Pose3D getCurrentPose();

}
