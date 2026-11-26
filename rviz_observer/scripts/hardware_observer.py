#!/usr/bin/env python3
"""Read-only Unitree G1 body observer. No publisher uses a Unitree command topic."""
from functools import partial
import math
import os
import socket
from xml.etree import ElementTree as ET

import PyKDL as kdl
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, String
from unitree_hg.msg import HandState, LowCmd, LowState

from observer import (Diagnostics, Marker, MarkerArray, latest_frame, make_pose, marker, point,
                      quaternion, tf2_ros, transform, vector)


# PR-mode serial joint coordinates in SDK2 motor order. The six reserve slots
# and Dex3 hand motors do not belong to this body LowState.
def motor_names():
    legs = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    arms = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw")
    return ([side + "_" + joint + "_joint" for side in ("left", "right") for joint in legs]
            + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
            + [side + "_" + joint + "_joint" for side in ("left", "right") for joint in arms])


NAMES = motor_names()
# Dex3-1 IDL motor order differs between the two hands.
HAND_NAMES = {
    side: [f"{side}_hand_{finger}_{index}_joint" for finger in fingers
           for index in range(3 if finger == "thumb" else 2)]
    for side, fingers in (("left", ("thumb", "middle", "index")),
                          ("right", ("thumb", "index", "middle")))
}
MAX_SKEW_NS = 50_000_000
STALE_NS = 250_000_000


def triple(element, name):
    return [float(s) for s in element.get(name, "0 0 0").split()]


class BodyKinematics:
    """World or pelvis FK from one measured encoder sample and the controller URDF."""
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.inertias = {}
        for link in root.findall("link"):
            inertia = link.find("inertial")
            if inertia is not None:
                mass = float(inertia.find("mass").get("value"))
                origin = inertia.find("origin")
                offset = kdl.Vector(*triple(origin, "xyz")) if origin is not None else kdl.Vector()
                self.inertias[link.get("name")] = (mass, offset)
        self.mass = sum(m for m, _ in self.inertias.values())
        by_parent = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            frame = kdl.Frame()
            if origin is not None:
                frame = kdl.Frame(kdl.Rotation.RPY(*triple(origin, "rpy")), kdl.Vector(*triple(origin, "xyz")))
            axis = joint.find("axis")
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            name = joint.get("name")
            # Include the hand masses at the neutral finger pose.
            moving = joint.get("type") != "fixed" and name in NAMES
            rotation_axis = kdl.Vector(*triple(axis, "xyz")) if moving else None
            by_parent.setdefault(parent, []).append((parent, child, name, frame, rotation_axis))
        self.tree = []
        queue = ["pelvis"]
        for parent in queue:
            for entry in by_parent.get(parent, ()):
                self.tree.append(entry)
                queue.append(entry[1])
        assert set(NAMES) == {name for _, _, name, _, axis in self.tree if axis is not None}

    def frames(self, q, base):
        frames = {"pelvis": base}
        for parent, child, name, origin, axis in self.tree:
            pose = frames[parent] * origin
            if axis is not None:
                pose.M = pose.M * kdl.Rotation.Rot(axis, q[name])
            frames[child] = pose
        return frames

    def positions(self, frames):
        moment = kdl.Vector()
        for link, (mass, offset) in self.inertias.items():
            moment += mass * (frames[link] * offset)
        return {"com": list(moment / self.mass),
                "left_foot": list(frames["left_foot_link"].p),
                "right_foot": list(frames["right_foot_link"].p)}


def valid_pose(values):
    return (len(values) == 7 and all(math.isfinite(x) for x in values)
            and abs(sum(x*x for x in values[3:]) - 1.0) < 0.02)


class HardwareObserver(Node):
    def __init__(self):
        super().__init__("hardware_observer")
        urdf = self.declare_parameter("urdf", "").value
        self.hardware_mode = self.declare_parameter("hardware_mode", 13).value
        self.model = BodyKinematics(urdf)
        self.base_source = self.declare_parameter("base_source", "controller").value
        self.tf = tf2_ros.TransformBroadcaster(self)
        self.diag = Diagnostics(lambda topic, kind: self.create_publisher(kind, topic, 1))
        self.joints = self.create_publisher(JointState, "joint_states", 1)
        self.command = self.create_publisher(JointState, "command/joint_states", 1)
        self.imu = self.create_publisher(Imu, "feedback/imu", 1)
        self.status = self.create_publisher(String, "status", 1)
        self.pelvis = self.create_publisher(PoseStamped, "feedback/pelvis", 1)
        self.reference = self.create_publisher(PoseStamped, "reference/pelvis", 1)
        self.last_status = ""
        self.last_status_ns = -10**9
        self.state = self.cmd = self.odom = self.frame = None
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LowState, "/lowstate", self.on_state, qos)
        self.create_subscription(LowCmd, "/lowcmd", self.on_command, qos)
        self.hand_samples = {}
        for side in HAND_NAMES:
            for prefix in ("", "/lf"):
                topic = f"{prefix}/dex3/{side}/state"
                self.create_subscription(HandState, topic, partial(self.on_hand, side, prefix), qos)
        if self.base_source == "odometry":
            self.create_subscription(Odometry, "base_odometry", self.on_odometry, qos)
        if self.base_source == "sport":
            from unitree_go.msg import SportModeState
            self.create_subscription(SportModeState, "/sportmodestate", self.on_sport, qos)
            self.buffer = tf2_ros.Buffer()
            self.listener = tf2_ros.TransformListener(self.buffer, self)
        self.stream = None
        # The native WBC telemetry producer is built for Mode 13.
        if self.hardware_mode == 13:
            self.stream = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.stream.bind("\0g1-wbc-hardware-" + str(os.getuid()))
            self.stream.setblocking(False)
        self.create_timer(1 / 60, self.update)
        self.last_state_ns = self.last_cmd_ns = -1

    def on_state(self, msg):
        if self.state:
            last = self.state[1]
            if (last.tick, last.mode_machine, last.mode_pr) == (msg.tick, msg.mode_machine, msg.mode_pr):
                return
        self.state = self.get_clock().now(), msg

    def on_hand(self, side, prefix, msg):
        motors = msg.motor_state
        if len(motors) == len(HAND_NAMES[side]) and all(
                math.isfinite(value) for motor in motors for value in (motor.q, motor.dq, motor.tau_est)):
            self.hand_samples[side, prefix] = self.get_clock().now().nanoseconds, motors

    def append_hand(self, joints, side, now_ns):
        # Prefer the standard stream when both DDS rates are available.
        for prefix in ("", "/lf"):
            sample = self.hand_samples.get((side, prefix))
            if sample and 0 <= now_ns - sample[0] <= STALE_NS:
                motors = sample[1]
                joints.name.extend(HAND_NAMES[side])
                joints.position.extend(m.q for m in motors)
                joints.velocity.extend(m.dq for m in motors)
                joints.effort.extend(m.tau_est for m in motors)
                return "live"
        return "stale" if any((side, prefix) in self.hand_samples for prefix in ("", "/lf")) else "waiting"

    def on_command(self, msg):
        self.cmd = self.get_clock().now(), msg

    def on_odometry(self, msg):
        self.odom = None
        if msg.header.frame_id != "world" or msg.child_frame_id != "pelvis":
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        if valid_pose(values):
            stamp = msg.header.stamp
            self.odom = stamp.sec * 10**9 + stamp.nanosec, values

    def on_sport(self, msg):
        w, x, y, z = msg.imu_state.quaternion
        values = list(msg.position) + [x, y, z, w]
        self.odom = None
        if msg.error_code == 0 and valid_pose(values):
            self.odom = self.get_clock().now().nanoseconds, values

    def report(self, text, overlay=None):
        now = self.get_clock().now()
        changed = text != self.last_status
        if changed:
            self.get_logger().info(text)
            self.last_status = text
        heartbeat = now.nanoseconds - self.last_status_ns >= 200_000_000
        if changed or heartbeat:
            self.status.publish(String(data=text))
            self.last_status_ns = now.nanoseconds
        if overlay is None and not changed and not heartbeat:
            return
        header = Header(stamp=now.to_msg(), frame_id="pelvis")
        color = (0.75, 0.95, 0.78) if overlay is not None else (1.0, 0.55, 0.25)
        label = marker(header, "hardware_status", 0, Marker.TEXT_VIEW_FACING, color)
        label.pose.position.z = 0.83
        label.scale.z = 0.028
        label.text = text
        if overlay is None:
            overlay = []
        overlay.append(label)
        self.diag.markers.publish(MarkerArray(markers=overlay))

    def update(self):
        now = self.get_clock().now()
        incoming = latest_frame(self.stream) if self.stream is not None else None
        if incoming is not None:
            self.frame = incoming
        if not self.state or now.nanoseconds - self.state[0].nanoseconds > STALE_NS:
            self.report("Waiting for fresh /lowstate")
            return
        stamp, state = self.state
        if state.mode_machine != self.hardware_mode or state.mode_pr != 0:
            self.report(f"G1 reported mode_machine={state.mode_machine}, mode_pr={state.mode_pr}. "
                        f"The observer expects mode_machine={self.hardware_mode} and serial PR coordinates (0).")
            return
        if stamp.nanoseconds == self.last_state_ns:
            return
        self.last_state_ns = stamp.nanoseconds
        motors = state.motor_state[:len(NAMES)]
        angles = [m.q for m in motors]
        if not all(math.isfinite(x) for x in angles):
            self.report("Invalid encoder sample")
            return
        local = Header(stamp=stamp.to_msg(), frame_id="pelvis")
        joints = JointState(header=local, name=list(NAMES), position=angles,
                            velocity=[m.dq for m in motors], effort=[m.tau_est for m in motors])
        hands = [f"{side} {self.append_hand(joints, side, now.nanoseconds)}" for side in HAND_NAMES]
        self.joints.publish(joints)
        imu = state.imu_state
        w, x, y, z = imu.quaternion
        if valid_pose([0, 0, 0, x, y, z, w]):
            # A colocated level frame permits orientation inspection even without localization.
            self.tf.sendTransform(transform(local, "g1_attitude", [0, 0, 0, -x, -y, -z, w]))
            self.imu.publish(Imu(header=Header(stamp=local.stamp, frame_id="imu_in_pelvis"),
                                 orientation=quaternion([x, y, z, w]),
                                 angular_velocity=vector(*imu.gyroscope),
                                 linear_acceleration=vector(*imu.accelerometer)))
        frame = self.frame
        aligned = frame and abs(stamp.nanoseconds - frame["stamp_ns"]) <= MAX_SKEW_NS
        base = None
        if self.base_source in ("odometry", "sport"):
            if self.odom and abs(stamp.nanoseconds - self.odom[0]) <= MAX_SKEW_NS:
                base = self.odom[1]
        elif aligned and valid_pose(frame.get("estimated_base", [])):
            base = frame["estimated_base"]

        world = Header(stamp=local.stamp, frame_id="world")
        origin = kdl.Frame()
        header = local
        if base is not None:
            parent = "unitree_odom" if self.base_source == "sport" else "world"
            header = Header(stamp=local.stamp, frame_id=parent)
            self.tf.sendTransform(transform(header, "pelvis", base))
            origin = kdl.Frame(kdl.Rotation.Quaternion(*base[3:]), kdl.Vector(*base[:3]))
            if parent == "unitree_odom":
                # Only compare in WBC world when the experiment supplies the registration.
                try:
                    tf = self.buffer.lookup_transform("world", parent, stamp).transform
                    t, q = tf.translation, tf.rotation
                    origin = kdl.Frame(kdl.Rotation.Quaternion(q.x, q.y, q.z, q.w),
                                       kdl.Vector(t.x, t.y, t.z)) * origin
                    header = world
                except tf2_ros.TransformException:
                    pass
            pose_values = list(origin.p) + list(origin.M.GetQuaternion())
            self.pelvis.publish(PoseStamped(header=header, pose=make_pose(pose_values)))
        feedback = self.model.positions(self.model.frames(dict(zip(NAMES, angles)), origin))
        overlay = self.diag.show_points(header, "feedback", feedback)
        if aligned:
            # Preserve the controller sample timestamp rather than stamping references as encoder data.
            ref = Header(stamp=Time(nanoseconds=frame["stamp_ns"]).to_msg(), frame_id="world")
            self.reference.publish(PoseStamped(header=ref, pose=make_pose(frame["pelvis"])))
            overlay.extend(self.diag.controller(frame, ref))
            self.tf.sendTransform([transform(ref, "wbc_" + n, v) for n, v in frame["frames"].items()])
            if header.frame_id == "world":
                self.diag.tracking_errors(world, frame["desired"], feedback)
        if self.cmd and abs(self.cmd[0].nanoseconds - stamp.nanoseconds) <= MAX_SKEW_NS:
            use_reference = (aligned and header.frame_id == "world" and
                             abs(frame["stamp_ns"] - self.cmd[0].nanoseconds) <= MAX_SKEW_NS)
            overlay.extend(self.command_markers(local, frame["pelvis"] if use_reference else None))
        source = self.base_source if base is not None else "pelvis-relative"
        status = f"G1 mode {self.hardware_mode} | encoders live | base: {source}"
        if self.hardware_mode == 13:
            status += " | WBC live" if aligned else " | waiting for WBC telemetry"
        self.report(status + " | Dex3 " + ", ".join(hands), overlay)

    def command_markers(self, header, reference):
        stamp, cmd = self.cmd
        if cmd.mode_machine != self.hardware_mode or cmd.mode_pr != 0:
            return []
        motors = cmd.motor_cmd[:len(NAMES)]
        if not all(math.isfinite(m.q) for m in motors):
            return []
        header = Header(stamp=stamp.to_msg(), frame_id=header.frame_id)
        if stamp.nanoseconds != self.last_cmd_ns:
            self.command.publish(JointState(header=header, name=NAMES, position=[m.q for m in motors],
                                            velocity=[m.dq for m in motors], effort=[m.tau for m in motors]))
            self.last_cmd_ns = stamp.nanoseconds
        # Position references are only meaningful for a leg whose position servos are active.
        q = dict(zip(NAMES, [m.q for m in motors]))
        frames = self.model.frames(q, kdl.Frame())
        # A position command is observable only down a chain of enabled position servos.
        active = {name for name, m in zip(NAMES, motors) if m.mode and m.kp > 0}
        valid = {"pelvis"}
        skeleton = marker(header, "command_posture_pelvis_relative", 0, Marker.LINE_LIST, (1.0, 0.62, 0.15))
        if reference is not None:
            skeleton.header = Header(stamp=header.stamp, frame_id="world")
            skeleton.ns = "controller_command_posture_world"
            skeleton.pose = make_pose(reference)
        skeleton.scale.x = 0.009
        for parent, child, name, _, axis in self.model.tree:
            if parent not in valid or (axis is not None and name not in active):
                continue
            valid.add(child)
            if axis is not None or child in ("left_foot_link", "right_foot_link",
                                              "left_hand_palm_link", "right_hand_palm_link"):
                skeleton.points.extend([point(frames[parent].p), point(frames[child].p)])
        dots = [skeleton] if skeleton.points else []
        for i, side in enumerate(("left", "right")):
            if all(m.mode and m.kp > 0 for m in motors[6*i:6*i+6]):
                dot = marker(header, "commanded_foot_relative_to_pelvis", i, Marker.CUBE, (0.82, 0.44, 0.96))
                dot.pose.position = point(frames[side + "_foot_link"].p)
                dot.scale = vector(0.018, 0.018, 0.018)
                dots.append(dot)
        return dots


def main():
    rclpy.init()
    node = HardwareObserver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.stream is not None:
            node.stream.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
