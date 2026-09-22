#pragma once
#ifndef TRON1_MPPI_PLANNER_HPP__
#define TRON1_MPPI_PLANNER_HPP__

#include <Eigen/Dense>
#include <vector>

#include "base_humanoid_planners/base_humanoid_planner.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "control_toolbox/low_pass_filter.hpp"

namespace tron1_deploy  // controller as chained interfaces from estimators
{
class Tron1MPPIPlanner : public humanoid_planners::BaseHumanoidPlanner
{
  public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_cleanup(const rclcpp_lifecycle::State& previous_state) override;

  protected:
  std::vector<hardware_interface::StateInterface> on_export_state_interfaces() override;

  bool on_set_chained_mode(bool chained_mode) override;

  controller_interface::return_type update_and_write_commands(const rclcpp::Time& time,
                                                              const rclcpp::Duration& period) override;

  controller_interface::return_type update_reference_from_subscribers(const rclcpp::Time& time,
                                                                      const rclcpp::Duration& period) override;

  // state
  std::array<double, 3> velocity_est_{0.0, 0.0, 0.0};  // vx, vy, yaw rate
  // filter
  std::vector<double> sampling_frequency_{50.0};  // Hz
  std::vector<double> damping_frequency_{1.0};    // Hz
  std::vector<double> damping_intensity_{0.0};    // dB
  std::array<std::shared_ptr<control_toolbox::LowPassFilter<double>>, 3> lp_filters_;

  // reference
  std::array<double, 3> velocity_cmd_{0.0, 0.0, 0.0};  // command after steer, vx, vy, yaw rate
  std::array<double, 3> velocity_des_{0.0, 0.0, 0.0};  // command before steer, vx, vy, yaw rate

  // interfaces related
  std::string node_name_ = "";
  std::string publish_topic_name_ = "";
  std::string subscribe_topic_name_ = "";

  // planning related
  int horizon_ = 20;
  double dt_ = 0.02;

  // topics related
  rclcpp::Node::SharedPtr node_ptr_ = nullptr;
  rclcpp::executors::SingleThreadedExecutor executor_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr control_subscriber_ = nullptr;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr state_publisher_ = nullptr;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>> realtime_state_publisher_ =
      nullptr;
};

};  // namespace tron1_deploy

#endif  // TRON1_MPPI_PLANNER_HPP__
