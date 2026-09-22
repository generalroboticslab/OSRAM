from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import OnProcessStart, OnProcessExit

import os

def generate_launch_description():
    use_keyboard_joy = LaunchConfiguration("use_keyboard_joy")
    # Define the robot_description from the no-gripper URDF.
    robot_description_file = PathJoinSubstitution([
        FindPackageShare("tron1_description"),
        "urdf",
        "tron1_with_arm_flat_foot_no_gripper.urdf",
    ])

    # Load ros2_control config YAML
    config_file = os.path.join(
        get_package_share_directory("tron1_deploy"),
        "config",
        "tron1_with_arm_no_gripper_sim_deploy.yaml"
    )

    robot_type_config_file = os.path.join(
        get_package_share_directory("tron1_deploy"),
        "config",
        "tron1_robot_with_arm_no_gripper_type.yaml"
    )

    mppi_config_file = os.path.join(
        get_package_share_directory("tron1_deploy"),
        "config",
        "tron1_loco-manip_mppi_config.yaml"
    )

    joy_launch = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{
            'dev': '/dev/input/js0',
        }],
        condition=UnlessCondition(use_keyboard_joy),
    )

    keyboard_joy_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("keyboard_joy"),
            "launch",
            "keyboard_joy.launch.py"
        )),
        launch_arguments={
            'config': os.path.join(
                get_package_share_directory("keyboard_joy"),
                "config",
                "key_mappings.yaml"
            )
        }.items(),
        condition=IfCondition(use_keyboard_joy),
    )

    joy_fsm_node = Node(
        package='fsm',
        executable='joy_fsm_node',
        # name='joy_fsm_node',
        output='screen',
        parameters=[{
            'joy_topic': '/joy',
            'key_controller_map': {
                '5': ['tron1_edamp_controller'],
                '3': ['tron1_standing_controller'],
                '2': ['tron1_loco_manipulation'],
                '1': ['tron1_squatting_controller']
            }
        }]
    )

    tron1_loco_manip_mppi_node = Node(
        package="tron1_deploy",
        executable="tron1_loco-manip_mppi.py",
        name="tron1_mppi",
        parameters=[mppi_config_file],
        output="screen",
    )

    mujoco_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("mujoco_sim"),
            "launch",
            "sim_launch.py"
        )),
        launch_arguments={
            'sim_config': os.path.join(
                get_package_share_directory("tron1_deploy"),
                "config",
                "tron1_with_arm_no_gripper_default_sim.yaml",
            ),
        }.items()
    )

    # Start robot_state_publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': Command(['cat ', robot_description_file])
        }],
        output='screen',
    )

    # Start ros2_control_node
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[config_file],
        output="screen"
    )

    load_tron1_dummy_estimator = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'active',
            'tron1_dummy_estimator'
        ],
        output='screen',
    )

    load_tron1_iekf_estimator = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'active',
            'tron1_iekf_estimator'
        ],
        output='screen',
    )

    load_tron1_velocity_planner = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'active',
            'tron1_velocity_planner'
        ],
        output='screen',
    )

    load_tron1_edamp_controller = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'active',
            'tron1_edamp_controller'
        ],
        output='screen',
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
        output='screen',
    )

    load_tron1_with_arm_planner = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'active',
            'tron1_with_arm_planner'
        ],
        output='screen',
    )

    load_tron1_loco_manipulation = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'inactive',
            'tron1_loco_manipulation'
        ],
        output='screen',
    )


    return LaunchDescription([
        DeclareLaunchArgument(
            "use_keyboard_joy",
            default_value="false",
            description="Use keyboard_joy key mappings instead of the hardware joy_node.",
        ),
        keyboard_joy_launch,
        joy_launch,
        joy_fsm_node,
        mujoco_sim_launch,
        ros2_control_node,
        robot_state_publisher_node,
        load_tron1_dummy_estimator,
        load_tron1_iekf_estimator,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_dummy_estimator,
                on_exit=[load_tron1_edamp_controller]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_edamp_controller,
                on_exit=[load_tron1_standing_controller]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_standing_controller,
                on_exit=[load_tron1_loco_manipulation]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_loco_manipulation,
                on_exit=[load_tron1_velocity_planner]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_velocity_planner,
                on_exit=[load_tron1_with_arm_planner]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_tron1_edamp_controller,
                on_exit=[load_tron1_squatting_controller]
            )
        ),
        tron1_loco_manip_mppi_node,
    ])
