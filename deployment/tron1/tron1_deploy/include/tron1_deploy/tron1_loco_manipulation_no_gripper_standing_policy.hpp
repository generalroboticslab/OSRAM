#pragma once
#ifndef TRON1_LOCO_MANIPULATION_NO_GRIPPER_STANDING_POLICY_HPP__
#define TRON1_LOCO_MANIPULATION_NO_GRIPPER_STANDING_POLICY_HPP__

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "controller_interface/chainable_controller_interface.hpp"
#include "rclcpp/rclcpp.hpp"

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include "pinocchio/parsers/urdf.hpp"

#include "base_controllers/onnx_policy.hpp"
#include "base_humanoid_controllers/base_humanoid_controller.hpp"
#include "base_utils/file_utils.hpp"
#include "base_utils/pinocchio_utils.hpp"
#include "rl_utils/mdp/action/actions.hpp"
#include "rl_utils/mdp/observation/observations.hpp"

namespace tron1_deploy
{

class Tron1LocoManipulationNoGripperStandingPolicy : public humanoid_controllers::BaseHumanoidController
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

  private:
  void initialize_observation_manager();
  void initialize_action_manager();
  void update_robot_state();
  void apply_policy_actions();

  static constexpr size_t leg_joint_dim_ = 8;
  static constexpr size_t arm_joint_dim_ = 6;
  static constexpr size_t joint_dim_ = 14;
  static constexpr size_t observation_dim_ = 52;
  static constexpr size_t action_dim_ = 14;

  observation::ObservationManager observation_manager_;
  action::ActionManager action_manager_;

  std::vector<double> observations_{};
  std::vector<double> actions_{};
  std::shared_ptr<base_controllers::OnnxMLP> policy_ptr_{nullptr};

  Eigen::Vector3d base_pos_w_{0.0, 0.0, 0.0};
  Eigen::Vector3d base_lin_vel_{0.0, 0.0, 0.0};
  Eigen::Vector3d base_ang_vel_{0.0, 0.0, 0.0};
  Eigen::Quaterniond ori_{1.0, 0.0, 0.0, 0.0};

  pinocchio::Model model_;
  pinocchio::Data data_;
  std::vector<int> perm_{};
  Eigen::VectorXd q_{};
  Eigen::VectorXd v_{};

  std::vector<double> joint_pos_{};
  std::vector<double> joint_vel_{};
  std::vector<double> joint_pos_des_{};
  std::vector<double> joint_vel_des_{};
  std::vector<double> joint_tau_des_{};
  std::vector<double> joint_pos_init_{};
  std::vector<double> joint_vel_init_{};
  std::vector<double> kp_gains_{};
  std::vector<double> kd_gains_{};
  std::vector<double> action_scales_{};
  bool use_policy_arm_actions_{true};
};

} // namespace tron1_deploy

#endif // TRON1_LOCO_MANIPULATION_NO_GRIPPER_STANDING_POLICY_HPP__
