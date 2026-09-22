cmake_minimum_required(VERSION 3.8)
project(tron1_deploy)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tsid REQUIRED)
find_package(base_utils REQUIRED)
find_package(rl_utils REQUIRED)
find_package(base_planners REQUIRED)
find_package(base_humanoid_estimators REQUIRED)
find_package(base_humanoid_planners REQUIRED)
find_package(base_humanoid_controllers REQUIRED)
find_package(mujoco_sim REQUIRED)
find_package(humanoid_msgs REQUIRED)
find_package(manipulator_msgs REQUIRED)
find_package(end_effector_msgs REQUIRED)
find_package(control_toolbox REQUIRED)

# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

function(configure_tron1_controller_target target)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(${target} PUBLIC tsid::tsid)
  ament_target_dependencies(${target} PUBLIC
    rclcpp
    rclcpp_lifecycle
    controller_interface
    hardware_interface
    pluginlib
    realtime_tools
    sensor_msgs
    geometry_msgs
    tf2_ros
    tsid
    base_planners
    base_humanoid_estimators
    base_humanoid_planners
    base_humanoid_controllers
    base_utils
    rl_utils
    control_toolbox
  )
endfunction()

set(TRON1_CONTROLLER_PLUGIN_TARGETS
  tron1_velocity_policy
  tron1_velocity
  tron1_loco_manipulation
  tron1_loco_manipulation_no_gripper
  tron1_loco_manipulation_no_gripper_standing_policy
  tron1_iekf_estimator
  tron1_velocity_planner
  tron1_mppi_planner
  tron1_standing_policy
  tron1_with_arm_planner
)

add_library(tron1_velocity_policy SHARED
  "src/tron1_velocity_policy.cpp"
)
configure_tron1_controller_target(tron1_velocity_policy)

add_library(tron1_velocity SHARED
  "src/tron1_velocity.cpp"
)
configure_tron1_controller_target(tron1_velocity)

add_library(tron1_loco_manipulation SHARED
  "src/tron1_loco_manipulation.cpp"
)
configure_tron1_controller_target(tron1_loco_manipulation)

add_library(tron1_loco_manipulation_no_gripper SHARED
  "src/tron1_loco_manipulation_no_gripper.cpp"
)
configure_tron1_controller_target(tron1_loco_manipulation_no_gripper)

add_library(tron1_loco_manipulation_no_gripper_standing_policy SHARED
  "src/tron1_loco_manipulation_no_gripper_standing_policy.cpp"
)
configure_tron1_controller_target(tron1_loco_manipulation_no_gripper_standing_policy)

add_library(tron1_iekf_estimator SHARED
  "src/tron1_iekf_estimator.cpp"
)
configure_tron1_controller_target(tron1_iekf_estimator)

add_library(tron1_velocity_planner SHARED
  "src/tron1_velocity_planner.cpp"
)
configure_tron1_controller_target(tron1_velocity_planner)

add_library(tron1_mppi_planner SHARED
  "src/tron1_mppi_planner.cpp"
)
configure_tron1_controller_target(tron1_mppi_planner)

add_library(tron1_standing_policy SHARED
  "src/tron1_standing_policy.cpp"
)
configure_tron1_controller_target(tron1_standing_policy)

add_library(tron1_with_arm_planner SHARED
  "src/tron1_with_arm_planner.cpp"
)
configure_tron1_controller_target(tron1_with_arm_planner)

pluginlib_export_plugin_description_file(controller_interface tron1_deploy_config.xml)

install(TARGETS ${TRON1_CONTROLLER_PLUGIN_TARGETS}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
ament_export_libraries(${TRON1_CONTROLLER_PLUGIN_TARGETS})
ament_export_targets(
  export_${PROJECT_NAME} HAS_LIBRARY_TARGET
)

add_library(tron1_mujoco_sim_plugins SHARED
  plugins/tron1_sim_node.cpp
)

target_include_directories(tron1_mujoco_sim_plugins PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/plugins/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(tron1_mujoco_sim_plugins PUBLIC
  rclcpp
  pluginlib
  mujoco_sim
  humanoid_msgs
  manipulator_msgs
  end_effector_msgs
)

pluginlib_export_plugin_description_file(mujoco_sim tron1_mujoco_sim_plugins.xml)

install(TARGETS tron1_mujoco_sim_plugins
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
ament_export_libraries(tron1_mujoco_sim_plugins)

install(
        DIRECTORY include/
        DESTINATION include/${PROJECT_NAME}
)

install(
        DIRECTORY plugins/include/
        DESTINATION include
)

install(
    DIRECTORY launch config model
    DESTINATION share/${PROJECT_NAME}/
)

# python related
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  script/tron1_mppi.py
  script/tron1_loco-manip_mppi.py
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
