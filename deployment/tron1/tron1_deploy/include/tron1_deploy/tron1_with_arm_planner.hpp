#pragma once
#ifndef TRON1_WITH_ARM_PLANNER_HPP__
#define TRON1_WITH_ARM_PLANNER_HPP__

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <utility>

#include "base_humanoid_planners/base_humanoid_planner.hpp"
#include "base_planners/joystick_velocity_planner.hpp"
#include "base_planners/trajectory_planner.hpp"
#include "base_utils/file_utils.hpp"
#include "base_utils/pinocchio_utils.hpp"
#include "base_utils/ros2_control_utils.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace tron1_deploy
{

/*
  This planner is a planner for tron1 with arm, it includes
  a waypoint planner for tron1 arm end effector and listen to
  a velocity planner for tron1 base velocity planning
*/
class Tron1WithArmPlanner : public humanoid_planners::BaseHumanoidPlanner
{
  public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  protected:
  controller_interface::return_type update_and_write_commands(const rclcpp::Time& time,
                                                              const rclcpp::Duration& period) override;

  controller_interface::return_type update_reference_from_subscribers(const rclcpp::Time& time,
                                                                      const rclcpp::Duration& period) override;

  std::pair<base_planners::Waypoint, std::array<double, 3>> generate_sine_commands(
      const base_planners::Waypoint& world_reference) const;

  base_planners::Waypoint express_world_reference_in_base(
      const base_planners::Waypoint& world_reference) const;

  base_planners::Waypoint express_base_reference_in_world(
      const Eigen::Vector3d& position_b, const Eigen::Quaterniond& orientation_b) const;

  std::vector<hardware_interface::StateInterface> on_export_state_interfaces() override;

  // robot model and data
  pinocchio::Model model_;
  pinocchio::Data data_;
  std::vector<int> perm_{};
  std::string base_frame_name_{"base_Link"};
  pinocchio::FrameIndex base_frame_id_{0};
  pinocchio::FrameIndex ee_frame_id_{0};
  Eigen::VectorXd q_{};
  Eigen::VectorXd v_{};

  // planners
  std::vector<std::string> waypoint_names_ = {};
  base_planners::Waypoint ee_pose_des_;
  std::unique_ptr<base_planners::PoseTrajectory> sine_line_trajectory_{nullptr};

  // point pose design related
  Eigen::Vector3d point_pos_b_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond point_ori_b_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d meta_command_pos_b_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond meta_command_ori_b_{Eigen::Quaterniond::Identity()};
  bool has_meta_command_ = false;

  // Global sine trajectory command generator
  Eigen::Vector2d command_k_xy_{Eigen::Vector2d::Ones()};
  double command_vxy_max_ = 0.5;
  double command_k_yaw_ = 1.0;
  double command_wz_max_ = 0.5;
  Eigen::Matrix<double, 6, 1> command_k_arm_{Eigen::Matrix<double, 6, 1>::Ones()};
  Eigen::Matrix<double, 6, 1> command_xi_max_{
      (Eigen::Matrix<double, 6, 1>() << 0.15, 0.15, 0.15, 0.35, 0.35, 0.35).finished()};

  // velocity
  std::array<double, 3> velocity_cmd_{0.0, 0.0, 0.0};  // x, y, yaw rate

  // introspection
  bool debug_{false};
  double sine_trajectory_active_debug_{0.0};
  double sine_trajectory_elapsed_time_debug_{0.0};
  std::array<double, 3> ee_global_ref_position_debug_{0.0, 0.0, 0.0};
  std::array<double, 4> ee_global_ref_orientation_debug_{1.0, 0.0, 0.0, 0.0};
  std::array<double, 3> ee_global_real_position_debug_{0.0, 0.0, 0.0};
  std::array<double, 4> ee_global_real_orientation_debug_{1.0, 0.0, 0.0, 0.0};

  // meta-dynamics related
  bool use_meta_dynamics_ = false;
  std::string node_name_ = "";
  std::string publish_topic_name_ = "";
  std::string subscribe_topic_name_ = "";
  int horizon_ = 20;
  double dt_ = 0.0;
  int mode_button_ = 0;
  bool mode_button_pressed_prev_ = false;
  bool sine_line_trajectory_requested_ = false;
  bool sine_line_trajectory_started_ = false;
  double sine_line_trajectory_start_time_seconds_{0.0};
  rclcpp::Node::SharedPtr node_ptr_ = nullptr;
  rclcpp::executors::SingleThreadedExecutor executor_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscriber_ = nullptr;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr control_subscriber_ = nullptr;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr state_publisher_ = nullptr;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>> realtime_state_publisher_ =
      nullptr;
};

}  // namespace tron1_deploy

#endif  // TRON1_WITH_ARM_PLANNER_HPP__
