import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import java.awt.geom.Point2D;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class Pose3DTest {

	private static final double EPS = 1e-6;

	@Test
	public void testRelativeDelta() {
		Pose3D p1 = new Pose3D(1, 3, 0, 270);
		Pose3D p2 = new Pose3D(0, -2, 180, 180);
		Pose3D p3 = new Pose3D(-5,7,-40,45);
		Pose3D p4 = new Pose3D(-1,1,0,0);

		Point2D.Double delta1 = p1.diffFromAbsolutePoint(p2);
		assertEquals(-5.0, delta1.x, EPS);
		assertEquals(1.0, delta1.y, EPS);

		Point2D.Double delta2 = p2.diffFromAbsolutePoint(p1);
		assertEquals(-1.0, delta2.x, EPS);
		assertEquals(-5.0, delta2.y, EPS);

		Point2D.Double delta3 = p1.diffFromAbsolutePoint(p3);
		assertEquals(4.0, delta3.x, EPS);
		assertEquals(6.0, delta3.y, EPS);

		Point2D.Double delta4 = p3.diffFromAbsolutePoint(p1);
		assertEquals(10.0/Math.sqrt(2), delta4.x, EPS);
		assertEquals(2.0/Math.sqrt(2), delta4.y, EPS);

		Point2D.Double delta5 = p3.diffFromAbsolutePoint(p4);
		assertEquals(10.0/Math.sqrt(2), delta5.x, EPS);
		assertEquals(-2.0/Math.sqrt(2), delta5.y, EPS);
	}

	@Test
	public void testRotate() {
		Pose3D p, ans;

		p = new Pose3D(1,1,0,0);
		double deg = Math.random()*360;
		ans = p.rotate(1,1,deg);
		assertEquals(p.x,ans.x,EPS);
		assertEquals(p.y,ans.y,EPS);
		assertEquals(p.z, ans.z, EPS);
		assertEquals(360-deg, ans.yaw, EPS);

		p = new Pose3D(-5,27.3);
		ans = p.rotate(0,0,90);
		assertEquals(-27.3,ans.x,EPS);
		assertEquals(-5,ans.y,EPS);
		assertEquals(270,ans.yaw,EPS);

		p = Pose3D.orientation(20,25,70);
		ans = p.rotate(15,25,45);
		assertEquals(18.5355339,ans.x,EPS);
		assertEquals(28.5355339,ans.y,EPS);
		assertEquals(25,ans.yaw,EPS);
	}

	@Test
	public void testGetAngleWithXAxisDeg() {
		Pose3D p1 = new Pose3D(Math.random()*900-450, Math.random()*900-450, 0, 0);
		assertEquals(90, p1.getAngleWithXAxisDeg(), EPS);

		Pose3D p2 = new Pose3D(Math.random()*900-450, Math.random()*900-450, 0, 90);
		assertEquals(0, p2.getAngleWithXAxisDeg(), EPS);

		Pose3D p3 = new Pose3D(Math.random()*900-450, Math.random()*900-450, 0, 180);
		assertEquals(270, p3.getAngleWithXAxisDeg(), EPS);
	}

	@Test
	public void testGetDesiredYawTo() {
		Pose3D pose = new Pose3D(0, 0, 0, 0);

		assertEquals(0.0, pose.getDesiredYawTo(0, 10), 1e-6, "Yaw to North");
		assertEquals(90.0, pose.getDesiredYawTo(10, 0), 1e-6, "Yaw to East");
		assertEquals(180.0, pose.getDesiredYawTo(0, -10), 1e-6, "Yaw to South");
		assertEquals(270.0, pose.getDesiredYawTo(-10, 0), 1e-6, "Yaw to West");
		assertEquals(45.0, pose.getDesiredYawTo(10, 10), 1e-6, "Yaw to North-East");
		assertEquals(135.0, pose.getDesiredYawTo(10, -10), 1e-6, "Yaw to South-East");
		assertEquals(225.0, pose.getDesiredYawTo(-10, -10), 1e-6, "Yaw to South-West");
		assertEquals(315.0, pose.getDesiredYawTo(-10, 10), 1e-6, "Yaw to North-West");

		Pose3D offsetPose = new Pose3D(5, 5, 30, 45);
		assertEquals(90.0, offsetPose.getDesiredYawTo(10, 5), 1e-6, "Yaw from offset position to East");

		Pose3D samePointPose = new Pose3D(1, 1, 0, 123.45);
		assertEquals(samePointPose.yaw, samePointPose.getDesiredYawTo(1, 1), 1e-6, "Yaw to same point");

		Pose3D pose2 = new Pose3D(2, 3, 0, 0);
		assertEquals(56.30993247402023, pose2.getDesiredYawTo(5, 5), 1e-6, "Yaw from (2, 3) to (5, 5)");
		Pose3D pose3 = new Pose3D(-3, -4, 0, 0);
		assertEquals(39.80557109226521, pose3.getDesiredYawTo(2, 2), 1e-6, "Yaw from (-3, -4) to (2, 2)");
	}

}
