package org.boofcv.util.structs;

import javax.annotation.CheckReturnValue;
import java.awt.geom.Point2D;
import java.io.Serializable;

/**
 * Stores a static position with (x, y, z, yaw). <p>
 * <ul>
 *     <li> x : positive is right. </li>
 *     <li> y : positive is forward. </li>
 *     <li> z : positive is up. </li>
 *     <li> yaw : rotation in xy plane, in degrees 0-360 where 0 is pointing up (forward IRL) and clockwise is positive. </li>
 * </ul>
 */
public class Pose3D implements Serializable {

	public final double x, y, z, yaw;
	public final static double DEFAULT_HEIGHT = 0, DEFAULT_ROTATION = 0;

	public Pose3D(double x, double y, double z, double yaw) {
		this.x = x;
		this.y = y;
		this.z = z;
		this.yaw = (yaw + 360) % 360;
	}
	public Pose3D(Pose3D p) { this(p.x, p.y, p.z, p.yaw); }
	public Pose3D() { this(0, 0, 0, 0); }
	public Pose3D(double x, double y) { this(x, y, DEFAULT_HEIGHT, DEFAULT_ROTATION); }
	public static Pose3D orientation(double x, double y, double yaw) { return new Pose3D(x, y, DEFAULT_HEIGHT, yaw); }

	public double getAngleWithXAxisDeg() {
		double ans = 360 - yaw;
		ans += 90;
		return ans % 360;
	}

	public double distance(Pose3D p) {
		return Math.sqrt((x - p.x)*(x - p.x) + (y - p.y)*(y - p.y) + (z - p.z)*(z - p.z));
	}

	public double distance2D(Pose3D p) {
		return Math.sqrt((x - p.x)*(x - p.x) + (y - p.y)*(y - p.y));
	}

//	/**
//	 * Returns the dx,dy from another (x,y) relative to the current orientation
//	 * @return (dx, dy)
//	 */
//	public Point2D.Double relativeDelta(double x, double y) {
//		double alpha = Math.toRadians(360-this.yaw);
//
//		double dx = x - this.x;
//		double dy = y - this.y;
//
//		double relativeDx = dx * Math.cos(alpha) + dy * Math.sin(alpha);
//		double relativeDy = -dx * Math.sin(alpha) + dy * Math.cos(alpha);
//
//		return new Point2D.Double(relativeDx, relativeDy);
//	}

	public double distanceFromLine(double slope, double intercept) {
		return Math.abs(slope*x - y + intercept) / Math.sqrt(slope*slope + 1);
	}

	public Pose3D plus(double x1, double y1, double z1, double yaw1) {
		return new Pose3D(x + x1, y + y1, z + z1, yaw + yaw1);
	}
	public Pose3D plus(double x1, double y1) {
		return new Pose3D(x + x1, y + y1, z, yaw);
	}

	public Pose3D minus(Pose3D p) {
		return new Pose3D(x - p.x, y - p.y, z - p.z, yaw - p.yaw);
	}

	public Pose3D errorFromTarget(Pose3D target) {
		double dx = target.x - this.x;
		double dy = target.y - this.y;
		double dz = target.z - this.z;
		double dYaw = target.yaw - this.yaw;
		if (Math.abs(dYaw) > 180) dYaw = dYaw - Math.signum(dYaw) * 360;
		return new Pose3D(dx, dy, dz, dYaw);
	}


	/**
	 * Rotates the point around the specified center point anticlockwise
	 * @param cx x-coordinate of the center point it is rotated around
	 * @param cy y-coordinate of the center point it is rotated around
	 * @param angleDeg Angle in degrees from the x-axis it is rotated around
	 */
	@CheckReturnValue public Pose3D rotate(double cx, double cy, double angleDeg) {
		double angleRad = Math.toRadians(angleDeg);
		double translatedX = x - cx;
		double translatedY = y - cy;

		double rotatedX = translatedX * Math.cos(angleRad) - translatedY * Math.sin(angleRad);
		double rotatedY = translatedX * Math.sin(angleRad) + translatedY * Math.cos(angleRad);

		return new Pose3D(rotatedX+cx, rotatedY+cy, z, yaw -angleDeg);
	}
	@CheckReturnValue public Pose3D rotate(double angleDeg) { return rotate(0, 0, angleDeg); }

	public static Point2D.Double diffFromRelativePoint(Pose3D home, Pose3D other) {
		Pose3D targetAbs = new Pose3D(home.x + other.x, home.y + other.y, home.z + other.z, home.yaw + other.yaw);
		return diffFromAbsolutePoint(home, targetAbs);
	}
	public Point2D.Double diffFromRelativePoint(Pose3D other) { return diffFromRelativePoint(this, other); }

	public static Point2D.Double diffFromAbsolutePoint(Pose3D home, Pose3D target) {
		Pose3D ans = target.rotate(home.x, home.y, home.yaw);
		return new Point2D.Double(ans.x - home.x, ans.y - home.y);
	}
	public Point2D.Double diffFromAbsolutePoint(Pose3D target) { return diffFromAbsolutePoint(this, target); }

	public double getDesiredYawTo(double x1, double y1) {
		if (x1 == x && y1 == y) return yaw;

		double dx = x1 - x;
		double dy = y1 - y;
		double angleRad = Math.atan2(dx, dy);
		double angleDeg = Math.toDegrees(angleRad);

		return (angleDeg + 360) % 360;
	}
	public double getDesiredYawTo(Pose3D pose) { return getDesiredYawTo(pose.x, pose.y); }

	public Pose3D withXYof(Pose3D other) { return new Pose3D(other.x, other.y, z, yaw); }
	public Pose3D withZof(double desiredZ) { return new Pose3D(x, y, desiredZ, yaw); }
	public Pose3D withYawOf(double desiredYaw) { return new Pose3D(x, y, z, desiredYaw); }
	public Pose3D withForwardOf(double dy) { return new Pose3D(x, y + dy, z, yaw); }


	@Override
	public String toString() { return String.format("(x=%.2f, y=%.2f, z=%.2f, yaw=%.2f°)", x, y, z, yaw); }

}
