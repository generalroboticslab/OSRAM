from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import OnProcessStart, OnProcessExit

import os

def generate_launch_description():
    robot_ip = LaunchConfiguration("robot_ip")
    # Define the robot_description from xacro
    urdf_file = PathJoinSubstitution([
        FindPackageShare("tron1_description"),
        "urdf",
        "tron1_flat_foot.urdf"
    ])

    # Load ros2_control config YAML
    config_file = os.path.join(
        get_package_share_directory("tron1_deploy"),
        "config",
        "tron1_real_deploy.yaml"
    )

    joy_launch = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{
            'dev': '/dev/input/js0',
        }]
    )

    joy_fsm_node = Node(
        package='fsm',
        executable='joy_fsm_node',
        output='screen',
        parameters=[{
            'joy_topic': '/joy',
            'key_controller_map': {
                '5': ['tron1_edamp_controller'],
                '3': ['tron1_standing_controller'],
                '2': ['tron1_velocity_policy'],
                '1': ['tron1_squatting_controller']
            }
        }]
    )

    # launch the hardware interface node here
    tron1_hardware_interface_node = Node(
        package="limx_dynamics_hardware_interfaces",
        executable="tron1_hardware_interface",
        parameters=[{
            'robot_type': 'FLAT_FOOT',
            'robot_ip': robot_ip
        }]
    )

    # Start robot_state_publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['cat ', urdf_file])}],
        output='screen',
    )

    # Start ros2_control_node
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[config_file],
        output="screen"
    )

    load_tron1_iekf_estimator = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'active',
            'tron1_iekf_estimator'
        ],
        output='screen'
    )

    load_tron1_velocity_planner = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'active',
            'tron1_velocity_planner'
        ],
        output='screen'
    )

    load_tron1_edamp_controller = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'active',
            'tron1_edamp_controller'
        ],
        output='screen'
    )

    load_tron1_squatting_controller = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'inactive',
            'tron1_squatting_controller'
        ],
        output='screen',
    )

    load_tron1_standing_controller = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'inactive',
            'tron1_standing_controller'
        ],
        output='screen'
    )

    load_tron1_velocity_policy = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'inactive',
            'tron1_velocity_policy'
        ],
        output='screen'
    )


    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_ip",
            description="IP address of the TRON1 hardware interface",
        ),
        tron1_hardware_interface_node,
        joy_launch,
        joy_fsm_node,
        robot_state_publisher_node,

        # Wait for hardware interface to be ready before starting ros2_control
        RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=tron1_hardware_interface_node,
                on_start=[
                    TimerAction(
                        period=10.0,
                        actions=[ros2_control_node]
                    )
                ]
            )
        ),

        TimerAction(
            period=5.0,
            actions=[load_tron1_iekf_estimator]
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_iekf_estimator,
                on_exit=[load_tron1_edamp_controller]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_edamp_controller,
                on_exit=[load_tron1_squatting_controller]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_squatting_controller,
                on_exit=[load_tron1_standing_controller]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_standing_controller,
                on_exit=[load_tron1_velocity_policy]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_velocity_policy,
                on_exit=[load_tron1_velocity_planner]
            )
        ),
    ])
