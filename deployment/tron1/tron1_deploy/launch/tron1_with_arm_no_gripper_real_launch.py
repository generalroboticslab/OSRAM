from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    use_keyboard_joy = LaunchConfiguration("use_keyboard_joy")
    robot_ip = LaunchConfiguration("robot_ip")
    vicon_server = LaunchConfiguration("vicon_server")
    urdf_file = PathJoinSubstitution([
        FindPackageShare("tron1_description"),
        "urdf",
        "tron1_with_arm_flat_foot_no_gripper.urdf",
    ])

    config_file = os.path.join(
        get_package_share_directory("tron1_deploy"),
        "config",
        "tron1_with_arm_no_gripper_real_deploy.yaml",
    )

    recorder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("recorder"),
            "launch",
            "joy_recorder.launch.py",
        )),
        launch_arguments={"config": config_file}.items(),
    )

    mppi_config_file = os.path.join(
        get_package_share_directory("tron1_deploy"),
        "config",
        "tron1_loco-manip_mppi_config.yaml",
    )

    joy_launch = Node(
        package="joy",
        executable="joy_node",
        name="joy_node",
        output="screen",
        parameters=[{
            "dev": "/dev/input/js0",
        }],
        condition=UnlessCondition(use_keyboard_joy),
    )

    keyboard_joy_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("keyboard_joy"),
            "launch",
            "keyboard_joy.launch.py",
        )),
        launch_arguments={
            "config": os.path.join(
                get_package_share_directory("keyboard_joy"),
                "config",
                "key_mappings.yaml",
            ),
        }.items(),
        condition=IfCondition(use_keyboard_joy),
    )

    joy_fsm_node = Node(
        package="fsm",
        executable="joy_fsm_node",
        output="screen",
        parameters=[{
            "joy_topic": "/joy",
            "key_controller_map": {
                "5": ["tron1_edamp_controller"],
                "3": ["tron1_standing_controller"],
                "2": ["tron1_loco_manipulation"],
                "1": ["tron1_squatting_controller"],
            },
        }],
    )

    vicon_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(
            get_package_share_directory("vrpn_mocap"),
            "launch",
            "client.launch.yaml"
        )),
        launch_arguments=[
            ('server', vicon_server),
            ('port', '3883')
        ]
    )

    tron1_loco_manip_mppi_node = Node(
        package="tron1_deploy",
        executable="tron1_loco-manip_mppi.py",
        name="tron1_mppi",
        parameters=[mppi_config_file],
        output="screen",
    )

    tron1_hardware_interface_node = Node(
        package="limx_dynamics_hardware_interfaces",
        executable="tron1_hardware_interface",
        parameters=[{
            "robot_type": "FLAT_FOOT",
            "robot_ip": robot_ip,
        }],
        output="screen",
    )

    airbot_play_hardware_interface_node = Node(
        package="airbot_play_hardware_interfaces",
        executable="airbot_play_arm_hardware_interface",
        parameters=[{
            "interface": "can0",
            "arm_control_mode": "mit",
        }],
        output="screen",
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": Command(["cat ", urdf_file])}],
        output="screen",
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[config_file],
        output="screen",
    )

    load_tron1_iekf_estimator = ExecuteProcess(
        cmd=[
            "ros2", "control", "load_controller", "--set-state", "active",
            "tron1_iekf_estimator",
        ],
        output="screen",
    )

    load_tron1_velocity_planner = ExecuteProcess(
        cmd=[
            "ros2", "control", "load_controller", "--set-state", "active",
            "tron1_velocity_planner",
        ],
        output="screen",
    )

    load_tron1_edamp_controller = ExecuteProcess(
        cmd=[
            "ros2", "control", "load_controller", "--set-state", "active",
            "tron1_edamp_controller",
        ],
        output="screen",
    )

    load_tron1_squatting_controller = ExecuteProcess(
        cmd=[
            "ros2", "control", "load_controller", "--set-state", "inactive",
            "tron1_squatting_controller",
        ],
        output="screen",
    )

    load_tron1_standing_controller = ExecuteProcess(
        cmd=[
            "ros2", "control", "load_controller", "--set-state", "inactive",
            "tron1_standing_controller",
        ],
        output="screen",
    )

    load_tron1_with_arm_planner = ExecuteProcess(
        cmd=[
            "ros2", "control", "load_controller", "--set-state", "active",
            "tron1_with_arm_planner",
        ],
        output="screen",
    )

    load_tron1_loco_manipulation = ExecuteProcess(
        cmd=[
            "ros2", "control", "load_controller", "--set-state", "inactive",
            "tron1_loco_manipulation",
        ],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_ip",
            description="IP address of the TRON1 hardware interface",
        ),
        DeclareLaunchArgument(
            "vicon_server",
            description="IP address or hostname of the VRPN server",
        ),
        DeclareLaunchArgument(
            "use_keyboard_joy",
            default_value="false",
            description="Use keyboard_joy key mappings instead of the hardware joy_node.",
        ),
        vicon_launch,
        tron1_hardware_interface_node,
        airbot_play_hardware_interface_node,
        keyboard_joy_launch,
        joy_launch,
        joy_fsm_node,
        recorder_launch,
        robot_state_publisher_node,
        RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=tron1_hardware_interface_node,
                on_start=[
                    TimerAction(
                        period=10.0,
                        actions=[ros2_control_node],
                    )
                ],
            )
        ),
        TimerAction(
            period=7.0,
            actions=[load_tron1_iekf_estimator],
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_iekf_estimator,
                on_exit=[load_tron1_edamp_controller],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_edamp_controller,
                on_exit=[load_tron1_standing_controller],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_edamp_controller,
                on_exit=[load_tron1_squatting_controller],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_standing_controller,
                on_exit=[load_tron1_loco_manipulation],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_loco_manipulation,
                on_exit=[load_tron1_velocity_planner],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_velocity_planner,
                on_exit=[load_tron1_with_arm_planner],
            )
        ),
        tron1_loco_manip_mppi_node,
    ])
