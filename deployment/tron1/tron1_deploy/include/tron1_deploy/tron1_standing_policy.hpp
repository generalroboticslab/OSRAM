#pragma once
#ifndef TRON1_STANDING_POLICY_HPP__
#define TRON1_STANDING_POLICY_HPP__

#include "Eigen/Dense"
#include "base_controllers/onnx_policy.hpp"
#include "base_humanoid_controllers/base_humanoid_controller.hpp"
#include "base_utils/file_utils.hpp"
#include "base_utils/ros2_control_utils.hpp"
#include "controller_interface/chainable_controller_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "rl_utils/rl_locomotion_utils.hpp"

namespace tron1_deploy
{
/*
    This controller is an RL velocity policy that outputs desired joint positions
*/

class Tron1StandingPolicy : public humanoid_controllers::BaseHumanoidController
{
  public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::return_type update_and_write_commands(const rclcpp::Time& time,
                                                              const rclcpp::Duration& period) override;

  protected:
  std::vector<hardware_interface::CommandInterface> on_export_reference_interfaces() override;

  // RL related
  std::vector<double> observations_{};  // input to policy
  std::vector<double> actions_{};       // output of policy
  std::shared_ptr<base_controllers::OnnxPolicy> velocity_policy_ptr_{nullptr};
  // For constructing observation
  Eigen::Vector3d base_lin_vel_{0.0, 0.0, 0.0};  // velocimeter readings, which means velocity in base frame
  Eigen::Vector3d base_ang_vel_{0.0, 0.0, 0.0};  // gyroscope readings, which means angular velocity in base frame
  Eigen::Quaterniond ori_{1.0, 0.0, 0.0, 0.0};   // orientation
  Eigen::Vector3d projected_gravity_{0.0, 0.0, -9.81};

  std::vector<double> vel_refs_{0.0, 0.0, 0.0};  // x, y, yaw rate command

  std::vector<double> joint_pos_{};
  std::vector<double> joint_vel_{};
  std::vector<double> joint_pos_des_{};  // desired joint positions after scaling and indices rematching
  std::vector<double> joint_pos_init_{};
  std::vector<double> joint_vel_init_{};
  std::vector<double> kp_gains_{};
  std::vector<double> kd_gains_{};
  std::vector<double> action_scales_{};
};
};  // namespace tron1_deploy

#endif  // TRON1_STANDING_POLICY_HPP__
