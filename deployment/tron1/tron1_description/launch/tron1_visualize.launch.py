import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _build_nodes(context):
    foot_type = LaunchConfiguration("foot_type").perform(context).strip().lower()
    with_arm = _as_bool(LaunchConfiguration("with_arm").perform(context))
    use_rviz = _as_bool(LaunchConfiguration("use_rviz").perform(context))
    use_joint_state_gui = _as_bool(LaunchConfiguration("use_joint_state_gui").perform(context))
    rviz_config = LaunchConfiguration("rviz_config").perform(context)

    valid_foot_types = {"flat", "point", "wheel"}
    if foot_type not in valid_foot_types:
        raise RuntimeError(
            f"Invalid foot_type '{foot_type}'. Valid options: {sorted(valid_foot_types)}"
        )

    pkg_share = get_package_share_directory("tron1_description")

    if with_arm:
        xacro_file = os.path.join(pkg_share, "xacro", "tron1_selectable.urdf.xacro")
        cmd = ["xacro", xacro_file, f"foot_type:={foot_type}", "with_arm:=true"]
        try:
            robot_description = subprocess.check_output(cmd, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"xacro failed: {' '.join(cmd)}") from exc
    else:
        urdf_file = os.path.join(pkg_share, "urdf", f"tron1_{foot_type}_foot.urdf")
        with open(urdf_file, "r", encoding="utf-8") as f:
            robot_description = f.read()

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
    ]

    nodes.append(
        Node(
            package="joint_state_publisher_gui" if use_joint_state_gui else "joint_state_publisher",
            executable="joint_state_publisher_gui" if use_joint_state_gui else "joint_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        )
    )

    if use_rviz:
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", rviz_config],
                output="screen",
            )
        )

    return nodes


def generate_launch_description():
    default_rviz = os.path.join(
        get_package_share_directory("tron1_description"),
        "config",
        "tron1_flat_with_arm.rviz",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "foot_type",
                default_value="flat",
                description="Tron1 foot type: flat | point | wheel",
            ),
            DeclareLaunchArgument(
                "with_arm",
                default_value="false",
                description="Attach Airbot arm to Tron1 base (true/false)",
            ),
            DeclareLaunchArgument(
                "use_joint_state_gui",
                default_value="true",
                description="Use joint_state_publisher_gui (true) or joint_state_publisher (false)",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Launch RViz2",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz,
                description="Path to RViz config file",
            ),
            OpaqueFunction(function=_build_nodes),
        ]
    )
