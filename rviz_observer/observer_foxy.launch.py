import os
from pathlib import Path
from xml.etree import ElementTree as ET

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def start(context):
    root = Path(__file__).resolve().parent
    scripts = root / "scripts"
    models = root.parent / "g1_description"
    if not scripts.is_dir():
        scripts = Path(get_package_prefix("g1_wbc_observer")) / "lib/g1_wbc_observer"
        models = root / "g1_description"
    hardware = LaunchConfiguration("source").perform(context) == "robot"
    urdf = models / "g1_29dof_mode_13.urdf"
    xml = ET.parse(urdf).getroot()
    for mesh in xml.iter("mesh"):
        mesh.set("filename", (models / mesh.get("filename")).as_uri())
    if hardware:
        # Body LowState has no Dex3 encoders. Keep the palms, omit unobserved finger links.
        omitted = set()
        for joint in list(xml.findall("joint")):
            if "_hand_" in joint.get("name") and joint.get("type") == "revolute":
                omitted.add(joint.find("child").get("link"))
                xml.remove(joint)
        for link in list(xml.findall("link")):
            if link.get("name") in omitted:
                xml.remove(link)
    description = ParameterValue(ET.tostring(xml, encoding="unicode"), value_type=str)
    program = "hardware_observer.py" if hardware else "observer.py"
    params = {"urdf": str(urdf), "base_source": LaunchConfiguration("base_source")} if hardware else {}
    fixed = LaunchConfiguration("fixed_frame").perform(context) or ("pelvis" if hardware else "world")
    actions = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             namespace="g1_observer", parameters=[{"robot_description": description,
                                                    "publish_frequency": 120.0}]),
        Node(executable=str(scripts / program), name="bridge", namespace="g1_observer",
             parameters=[params] if params else [], output="screen",
             remappings=[("tf", "/tf"), ("tf_static", "/tf_static"),
                         ("/lowstate", LaunchConfiguration("state_topic")),
                         ("/lowcmd", LaunchConfiguration("command_topic"))] if hardware else [],
             on_exit=[Shutdown(reason="Observer stopped")]),
        Node(package="rviz2", executable="rviz2", namespace="g1_observer",
             arguments=["-d", str(root / "walking_foxy.rviz"), "-f", fixed], output="screen",
             condition=IfCondition(LaunchConfiguration("gui")),
             on_exit=[Shutdown(reason="RViz2 closed")]),
    ]
    if IfCondition(LaunchConfiguration("start_sim")).evaluate(context):
        if hardware:
            raise ValueError("start_sim requires source:=sim")
        path = LaunchConfiguration("sim_executable").perform(context)
        binary = Path(path).expanduser().resolve() if path else root.parent / "build" / "main"
        if not binary.is_file():
            raise ValueError("Build the MuJoCo main target and set sim_executable to its absolute path")
        # SDK2 and Foxy ship different Cyclone DDS ABIs. Keep the native executable
        # on its build RUNPATH while preserving non-ROS user library directories.
        prefixes = [Path(p).resolve() for key in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH")
                    for p in os.environ.get(key, "").split(os.pathsep) if p]
        library_dirs = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                        if p and not any(root in Path(p).resolve().parents for root in prefixes)]
        native_env = {"LD_LIBRARY_PATH": os.pathsep.join(library_dirs)}
        actions.append(ExecuteProcess(cmd=[str(binary), "--sim", "--rviz"], cwd=str(binary.parent),
                                      additional_env=native_env, output="screen", on_exit=[Shutdown(reason="MuJoCo closed")]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="sim", choices=["robot", "sim"]),
        DeclareLaunchArgument("base_source", default_value="controller",
                              choices=["controller", "odometry", "sport"]),
        DeclareLaunchArgument("state_topic", default_value="/lowstate"),
        DeclareLaunchArgument("command_topic", default_value="/lowcmd"),
        DeclareLaunchArgument("fixed_frame", default_value=""),
        DeclareLaunchArgument("start_sim", default_value="false"),
        DeclareLaunchArgument("sim_executable", default_value=""),
        DeclareLaunchArgument("gui", default_value="true"),
        OpaqueFunction(function=start),
    ])
