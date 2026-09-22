#pragma once
#ifndef TRON1_LOCO_MANIPULATION_NO_GRIPPER_HPP__
#define TRON1_LOCO_MANIPULATION_NO_GRIPPER_HPP__

#include "Eigen/Dense"
#include "controller_interface/chainable_controller_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "control_toolbox/low_pass_filter.hpp"

#include "base_controllers/onnx_policy.hpp"
#include "base_planners/waypoint_planner.hpp"
#include "base_humanoid_controllers/base_humanoid_controller.hpp"
#include "base_utils/file_utils.hpp"
#include "base_utils/ros2_control_utils.hpp"
#include "base_utils/pinocchio_utils.hpp"
#include "rl_utils/mdp/observation/observations.hpp"
#include "rl_utils/mdp/action/actions.hpp"

namespace tron1_deploy
{

class Tron1LocoManipulationNoGripper : public humanoid_controllers::BaseHumanoidController
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

  void initialize_observation_managers();

  void initialize_action_managers();
  // RL related
  std::string interface_name_;
  std::string base_frame_name_ = "base_Link";
  std::string ee_frame_name_ = "ee_site";
  std::string policy_type_ = "base"; // "base", "rnn", "residual"
  std::string steer_type_ = "none"; // "none", "closed", "open_full", "open_ref"
  int history_length_ = 1;
  int residual_history_length_ = 1;
  observation::ObservationManager observation_manager_;
  action::ActionManager action_manager_;
  observation::ObservationManager residual_observation_manager_;
  action::ActionManager residual_action_manager_;
  observation::ObservationManager open_full_observation_manager_;
  observation::ObservationManager open_ref_observation_manager_;

  std::vector<double> observations_{};  // input to policy
  std::vector<double> actions_{};       // output of policy
  std::vector<double> residual_observations_{};
  std::vector<double> residual_actions_;
  std::shared_ptr<base_controllers::OnnxMLP> base_policy_ptr_{nullptr};
  std::shared_ptr<base_controllers::OnnxRNN> rnn_policy_ptr_{nullptr};
  std::shared_ptr<base_controllers::OnnxRNN> residual_policy_ptr_{nullptr};
  // for constructing obervations
  Eigen::Vector3d base_lin_vel_{0.0, 0.0, 0.0};  // velocimeter readings, which means velocity in base frame
  Eigen::Vector3d base_ang_vel_{0.0, 0.0, 0.0};  // gyroscope readings, which means angular velocity in base frame
  Eigen::Quaterniond ori_{1.0, 0.0, 0.0, 0.0};   // orientation
  Eigen::Vector3d ee_pos_{0.0, 0.0, 0.0};     // end-effector position in base frame
  Eigen::Quaterniond ee_ori_{1.0, 0.0, 0.0, 0.0};    // end-effector orientation wxyz
  std::vector<double> vel_refs_;
  std::vector<double> ee_pose_ref_;

  // pinocchio related for forward kinematics
  pinocchio::Model model_;
  pinocchio::Data data_;
  std::vector<int> perm_{};
  pinocchio::FrameIndex base_frame_id_{0};
  pinocchio::FrameIndex ee_frame_id_{0};

  std::vector<double> joint_pos_{};
  std::vector<double> joint_vel_{};
  std::vector<double> joint_pos_des_{};  // desired joint positions after scaling and indices rematching
  std::vector<double> joint_pos_init_{};
  std::vector<double> joint_vel_init_{};
  std::vector<double> kp_gains_{};
  std::vector<double> kd_gains_{};
  std::vector<double> action_scales_{};
  std::vector<double> residual_action_scales_{};

  // steer related
  int horizon_ = 20;
  double dt_ = 0.02;
  std::string node_name_ = "";
  std::string subscribe_topic_name_ = "";
  std::string publish_topic_name_ = "";
  // topics related
  rclcpp::Node::SharedPtr node_ptr_ = nullptr;
  rclcpp::executors::SingleThreadedExecutor executor_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr control_subscriber_ = nullptr;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr state_publisher_ = nullptr;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>> realtime_state_publisher_ =
      nullptr;
  // TODO: whether position estimation requires filtering?
  // std::vector<double> sampling_frequency_{50.0};  // Hz
  // std::vector<double> damping_frequency_{1.0};    // Hz
  // std::vector<double> damping_intensity_{0.0};    // dB
  // std::array<std::shared_ptr<control_toolbox::LowPassFilter<double>>, 3> lp_filters_;

  // debug
  std::array<double, 3> ee_pos_debug_{0.0, 0.0, 0.0};
  std::array<double, 4> ee_ori_debug_{1.0, 0.0, 0.0, 0.0};
  std::array<double, 3> ee_pos_ref_debug_{0.0, 0.0, 0.0};
  std::array<double, 4> ee_ori_ref_debug_{1.0, 0.0, 0.0, 0.0};
};


} // namespace tron1_deploy


#endif // TRON1_LOCO_MANIPULATION_NO_GRIPPER_HPP__
