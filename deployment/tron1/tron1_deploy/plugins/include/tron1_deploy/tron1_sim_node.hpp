#pragma once
#ifndef TRON1_SIM_NODE_HPP
#define TRON1_SIM_NODE_HPP

#include <rmw/types.h>

#include <rclcpp/rclcpp.hpp>

#include "array_safety.h"
#include "humanoid_msgs/msg/humanoid_est.hpp"
#include "humanoid_msgs/msg/low_cmd.hpp"
#include "humanoid_msgs/msg/low_state.hpp"
#include "manipulator_msgs/msg/manipulator_est.hpp"
#include "manipulator_msgs/msg/low_cmd.hpp"
#include "manipulator_msgs/msg/low_state.hpp"
#include "end_effector_msgs/msg/gripper_state.hpp"
#include "end_effector_msgs/msg/gripper_cmd.hpp"

#include "mujoco_sim_node_base.hpp"
#include "simulate.h"

using namespace std::chrono_literals;

namespace mujoco_sim
{

namespace mj = mujoco;

class Tron1SimNode : public MujocoSimNodeBase
{
  public:
  // constructor of the node
  Tron1SimNode();

  void reset_params();

  virtual void load_ros2_params() override;

  protected:
  // robot_type
  enum RobotType
  {
    POINT_FOOT = 0,
    FLAT_FOOT = 1,
    WHEEL_FOOT = 2,
    POINT_FOOT_WITH_ARM = 3,
    FLAT_FOOT_WITH_ARM = 4,
    WHEEL_FOOT_WITH_ARM = 5,
    POINT_FOOT_WITH_ARM_NO_GRIPPER = 6,
    FLAT_FOOT_WITH_ARM_NO_GRIPPER = 7,
    WHEEL_FOOT_WITH_ARM_NO_GRIPPER = 8
  } robot_type_;
  RobotType parse_robot_type(const std::string& robot_type_str);

  // type related helpers
  bool has_arm() const;
  bool has_gripper() const;
  size_t leg_joint_count() const;
  size_t arm_start_index() const;
  size_t total_joint_count() const;

  // for legs
  void build_low_state_msg();

  void callback_low_state();
  void callback_low_cmd(const humanoid_msgs::msg::LowCmd::SharedPtr msg);

  std::vector<rclcpp::TimerBase::SharedPtr> timers_;
  rclcpp::Subscription<humanoid_msgs::msg::LowCmd>::SharedPtr cmd_sub_ptr_;
  rclcpp::Publisher<humanoid_msgs::msg::LowState>::SharedPtr low_state_pub_ptr_;

  // for arm and gripper
  void callback_manipulator_low_cmd(const manipulator_msgs::msg::LowCmd::SharedPtr msg);
  void callback_gripper_low_cmd(const end_effector_msgs::msg::GripperCmd::SharedPtr msg);
  void load_gripper_gains_from_xml();

  rclcpp::Subscription<manipulator_msgs::msg::LowCmd>::SharedPtr manipulator_cmd_sub_ptr_;
  rclcpp::Publisher<manipulator_msgs::msg::LowState>::SharedPtr manipulator_state_pub_ptr_;
  rclcpp::Subscription<end_effector_msgs::msg::GripperCmd>::SharedPtr gripper_cmd_sub_ptr_;
  rclcpp::Publisher<end_effector_msgs::msg::GripperState>::SharedPtr gripper_state_pub_ptr_;

  // sensor readings
  // L -> R
  std::vector<float> joint_pos_;  // abad, hip, knee, maybe ankle or wheel
  std::vector<float> joint_vel_;
  std::vector<float> joint_torque_;
  std::array<float, 4> quat_; // w, x, y, z
  std::array<float, 3> gyro_;
  std::array<float, 3> accelerom_;
  std::array<bool, 2> contact_;

  // motor commands
  std::vector<uint8_t> mode_;
  std::vector<float> cmd_torque_;
  std::vector<float> cmd_pos_;
  std::vector<float> cmd_vel_;
  // motor params
  std::vector<float> cmd_kp_;
  std::vector<float> cmd_kd_;
  float gripper_kp_from_xml_{0.0f};
  float gripper_kd_from_xml_{0.0f};
  bool gripper_gains_from_xml_loaded_{false};

  // msg
  humanoid_msgs::msg::LowState low_state_msg_;
  manipulator_msgs::msg::LowState manipulator_low_state_msg_;
  end_effector_msgs::msg::GripperState gripper_state_msg_;
};

class Tron1SimGroundTruth : public Tron1SimNode
{
  public:
  Tron1SimGroundTruth();

  void reset_params();

  void load_ros2_params() override;

  protected:
  void build_ground_truth_msg();

  void ground_truth_callback();
  rclcpp::Publisher<humanoid_msgs::msg::HumanoidEst>::SharedPtr ground_truth_pub_ptr_;

  // manipulator and grippers
  rclcpp::Publisher<manipulator_msgs::msg::ManipulatorEst>::SharedPtr manipulator_ground_truth_pub_ptr_;

  // ground truth
  humanoid_msgs::msg::HumanoidEst ground_truth_msg_;
  std::array<float, 3> pos_truth_{0.0, 0.0, 0.0};        // x, y, z in the world frame
  std::array<float, 4> ori_truth_{1.0, 0.0, 0.0, 0.0};   // w, x, y, z
  std::array<float, 3> lin_vel_truth_{0.0, 0.0, 0.0};    // vx, vy, vz in the world frame
  std::array<float, 3> lin_accel_truth_{0.0, 0.0, 0.0};  // ax, ay, az  in the world frame
  std::array<float, 3> ang_vel_truth_{0.0, 0.0, 0.0};    // wx, wy, wz in the world frame
  std::array<float, 3> ang_acc_truth_{0.0, 0.0, 0.0};    // alphax, alphay, alphaz in the world frame
  std::array<std::array<float, 6>, 2> contact_wrenches_{{
      {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
      {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
  }};  // fx, fy, fz, mx, my, mz in the world frame
  // manipulator ground truth
  manipulator_msgs::msg::ManipulatorEst manipulator_ground_truth_msg_;
};

}  // namespace mujoco_sim

#endif
