#pragma once
#ifndef TRON1_VELOCITY_PLANNER_HPP__
#define TRON1_VELOCITY_PLANNER_HPP__

#include "base_planners/joystick_velocity_planner.hpp"
#include "base_humanoid_planners/base_humanoid_planner.hpp"
#include "base_utils/ros2_control_utils.hpp"

namespace tron1_deploy
{
/*
  A mixture of joystick velocity planning and constant velocity planning to facilitate online
  finetuning and comparison.
*/

class Tron1VelocityPlanner : public base_planners::JoystickVelocityPlanner
{
  public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  protected:
  controller_interface::return_type update_and_write_commands(const rclcpp::Time& time,
                                                              const rclcpp::Duration& period) override;

  controller_interface::return_type update_reference_from_subscribers(const rclcpp::Time& time,
                                                                      const rclcpp::Duration& period) override;

  std::vector<hardware_interface::StateInterface> on_export_state_interfaces() override;

  int mode_button_{4}; //
  bool mode_{false}; // false for joystick control, true for constant velocity control

  double constant_velocity_x_{}; // constant velocity in x direction when mode_ is true
  double constant_velocity_y_{}; // constant velocity in y direction when mode_ is true
  double constant_yaw_rate_{}; // constant yaw rate when mode_ is true
};

} // namespace tron1_deploy

#endif // TRON1_VELOCITY_PLANNER_HPP__
