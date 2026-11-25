#!/usr/bin/env python3
import json
import math
import os
import select
from pathlib import Path
import socket
import sys
from xml.etree import ElementTree

# roslaunch loads the same URDF as Pinocchio, resolving its relative mesh paths for RViz.
if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--description":
    path = Path(sys.argv[2]).resolve()
    robot = ElementTree.parse(path)
    for mesh in robot.iter("mesh"):
        mesh.set("filename", (path.parent / mesh.get("filename")).as_uri())
    print(ElementTree.tostring(robot.getroot(), encoding="unicode"))
    sys.exit(0)

ROS2 = os.environ.get("ROS_VERSION") == "2"
if ROS2:
    import rclpy
    from builtin_interfaces.msg import Duration
    LIFETIME = Duration(nanosec=350000000)
else:
    import rospy
    LIFETIME = rospy.Duration(0.35)
import tf2_ros
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, Quaternion, TransformStamped, Vector3, Vector3Stamped
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray


LEFT = (0.15, 0.68, 1.0)
RIGHT = (1.0, 0.48, 0.15)
COM = (0.18, 1.0, 0.53)
ZMP = (1.0, 0.84, 0.12)
CONTACT = (0.95, 0.28, 0.68)
WBC = (0.30, 0.87, 1.0)
ESTIMATED = (0.69, 0.46, 1.0)
EXTERNAL = (1.0, 0.22, 0.14)
FORCE_SCALE = 0.001  # metres per newton


def point(values):
    return Point(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def vector(x, y, z):
    return Vector3(x=float(x), y=float(y), z=float(z))


def quaternion(values):
    return Quaternion(x=float(values[0]), y=float(values[1]), z=float(values[2]), w=float(values[3]))


def make_pose(values):
    return Pose(position=point(values), orientation=quaternion(values[3:]))


def transform(header, child, values):
    tf = TransformStamped(header=header, child_frame_id=child)
    tf.transform.translation = vector(*values[:3])
    tf.transform.rotation = quaternion(values[3:])
    return tf


def marker(header, ns, index, kind, color):
    msg = Marker(header=header, ns=ns, id=index, type=kind)
    msg.pose.orientation.w = 1.0
    msg.color = ColorRGBA(r=float(color[0]), g=float(color[1]), b=float(color[2]), a=1.0)
    msg.lifetime = LIFETIME
    return msg


def make_markers(frame, header):
    result = []
    for index, landing in enumerate(frame["landings"]):
        side = landing["foot"]
        msg = marker(header, "planned_" + side, index, Marker.LINE_STRIP,
                     LEFT if side == "left" else RIGHT)
        msg.pose = make_pose(landing["pose"])
        msg.scale.x = 0.002
        # WBC contact rectangle, expressed in the planned sole frame.
        msg.points = [point((x, y, 0)) for x, y in
                      [(0.10, 0.035), (0.10, -0.035), (-0.10, -0.035),
                       (-0.10, 0.035), (0.10, 0.035)]]
        result.append(msg)

    for field, color in [("com", COM), ("zmp", ZMP)]:
        points = frame[field]
        line = marker(header, "mpc_" + field, 0, Marker.LINE_STRIP, color)
        line.scale.x = 0.006
        line.points = [point(p) for p in points]
        result.append(line)
        dot = marker(header, "current_" + field, 0, Marker.SPHERE, color)
        dot.scale = vector(0.026, 0.026, 0.026)
        dot.pose.position = point(points[0])
        result.append(dot)

    for group, forces, color in [("contact", frame["contacts"], CONTACT),
                                  ("wbc", frame["wbc"].values(), WBC),
                                  ("estimated", frame["estimated"].values(), ESTIMATED),
                                  ("external", frame["external"], EXTERNAL)]:
        for index, item in enumerate(forces):
            force, pos = item["force"], item["point"]
            magnitude = math.sqrt(sum(f * f for f in force))
            if magnitude < 0.01:
                continue
            arrow = marker(header, group, index, Marker.ARROW, color)
            arrow.points = [point(pos), point([p + FORCE_SCALE*f for p, f in zip(pos, force)])]
            arrow.scale = vector(0.005, 0.012, min(0.028, magnitude * FORCE_SCALE * 0.3))
            arrow.text = "{} {:.1f} N".format(item.get("body", group), magnitude)
            result.append(arrow)

    label = marker(header, "phase", 0, Marker.TEXT_VIEW_FACING, (1, 1, 1))
    label.pose.position = point(frame["pelvis"])
    label.pose.position.y += 0.50
    label.pose.position.z += 0.32
    label.scale.z = 0.036
    label.text = "{phase}\nSupport:{support}\n{time:.2f}s".format(**frame)
    result.append(label)
    return result


TASKS = ("com", "left_foot", "right_foot")


class Diagnostics:
    def __init__(self, publisher):
        self.markers = publisher("markers", MarkerArray)
        self.foot_forces = {(source, side): publisher(source + "/" + side, Vector3Stamped)
                           for source in ("wbc", "estimated") for side in ("left", "right")}
        self.points = {(source, task): publisher(source + "/" + task, PointStamped)
                       for source in ("desired", "feedback") for task in TASKS}
        self.errors = {task: publisher("tracking_error/" + task, Vector3Stamped) for task in TASKS}
        self.desired_zmp = publisher("desired/zmp", PointStamped)
        self.previous = set()

    def show_points(self, header, source, positions):
        color = (1.0, 0.45, 0.06) if source == "desired" else (0.15, 0.9, 0.95)
        result = []
        for i, task in enumerate(TASKS):
            p = point(positions[task])
            self.points[source, task].publish(PointStamped(header=header, point=p))
            dot = marker(header, source + "_" + task, i, Marker.SPHERE, color)
            dot.scale = vector(0.032, 0.032, 0.032) if source == "desired" else vector(0.021, 0.021, 0.021)
            dot.color.a = 0.55 if source == "desired" else 1.0
            dot.pose.position = p
            result.append(dot)
        return result

    def tracking_errors(self, header, desired, feedback):
        for task in TASKS:
            difference = [a - d for a, d in zip(feedback[task], desired[task])]
            self.errors[task].publish(Vector3Stamped(header=header, vector=vector(*difference)))

    def controller(self, frame, header):
        for (source, side), pub in self.foot_forces.items():
            force = vector(*frame[source][side]["force"])
            pub.publish(Vector3Stamped(header=header, vector=force))
        if "zmp" in frame["desired"]:
            self.desired_zmp.publish(PointStamped(header=header, point=point(frame["desired"]["zmp"])))
        current = make_markers(frame, header)
        visible = {(m.ns, m.id) for m in current}
        current.extend(Marker(header=header, ns=ns, id=i, action=Marker.DELETE)
                       for ns, i in self.previous - visible)
        self.previous = visible
        return current + self.show_points(header, "desired", frame["desired"])


def latest_frame(stream):
    packet = None
    while True:
        try:
            packet = stream.recv(212992)
        except BlockingIOError:
            break
    if packet is None or b"null" in packet:
        return None
    return json.loads(packet)


def run(publisher, broadcaster, now, running, poll, log):
    joints = publisher("joint_states", JointState)
    command = publisher("command/joint_states", JointState)
    diagnostics = Diagnostics(publisher)
    contact_zmp = publisher("feedback/contact_zmp", PointStamped)
    pelvis = publisher("feedback/pelvis", PoseStamped)
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as stream:
        stream.bind("\0g1-wbc-rviz-" + str(os.getuid()))
        stream.setblocking(False)
        log("Waiting for MuJoCo pose from ./main --sim --rviz")
        while running():
            poll()
            if not select.select([stream], [], [], 0.1)[0]:
                continue
            frame = latest_frame(stream)
            if frame is None:
                continue
            header = Header(stamp=now(), frame_id="world")
            pelvis.publish(PoseStamped(header=header, pose=make_pose(frame["pelvis"])))
            transforms = [transform(header, "pelvis", frame["pelvis"])]
            transforms.extend(transform(header, "wbc_" + name, values)
                              for name, values in frame.get("frames", {}).items())
            broadcaster.sendTransform(transforms)
            joints.publish(JointState(header=header, name=list(frame["joints"]),
                                      position=list(frame["joints"].values()),
                                      velocity=[frame["velocity"][name] for name in frame["joints"]],
                                      effort=[frame["effort"][name] for name in frame["joints"]]
                                      if "effort" in frame else []))
            if "desired" not in frame:
                continue  # Startup carries measured pose before the first WBC solve.
            if "contact_zmp" in frame:
                contact_zmp.publish(PointStamped(header=header, point=point(frame["contact_zmp"])))
            torque = frame["torque_command"]
            command.publish(JointState(header=header, name=list(torque), effort=list(torque.values())))
            overlay = diagnostics.controller(frame, header)
            overlay.extend(diagnostics.show_points(header, "feedback", frame["feedback"]))
            diagnostics.markers.publish(MarkerArray(markers=overlay))
            diagnostics.tracking_errors(header, frame["desired"], frame["feedback"])


def main():
    if ROS2:
        rclpy.init()
        node = rclpy.create_node("bridge")
        try:
            run(lambda topic, kind: node.create_publisher(kind, topic, 1),
                tf2_ros.TransformBroadcaster(node), lambda: node.get_clock().now().to_msg(),
                rclpy.ok, lambda: rclpy.spin_once(node, timeout_sec=0.0),
                node.get_logger().info)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    else:
        rospy.init_node("bridge")
        run(lambda topic, kind: rospy.Publisher(topic, kind, queue_size=1),
            tf2_ros.TransformBroadcaster(), rospy.Time.now, lambda: not rospy.is_shutdown(),
            lambda: None, rospy.loginfo)


if __name__ == "__main__":
    main()
