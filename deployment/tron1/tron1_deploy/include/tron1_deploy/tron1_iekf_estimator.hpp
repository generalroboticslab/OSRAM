#pragma once
#ifndef TRON1_IEKF_ESTIMATOR_HPP
#define TRON1_IEKF_ESTIMATOR_HPP

#include <mutex>
#include <vector>

#include "base_humanoid_estimators/base_humanoid_estimator.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "base_estimators/IEKF.hpp"
#include "base_utils/file_utils.hpp"
#include "base_utils/ros2_control_utils.hpp"
#include "base_utils/pinocchio_utils.hpp"

namespace tron1_deploy
{
using namespace Eigen;
class Tron1IEKFEstimator : public humanoid_estimators::BaseHumanoidEstimator
{
  public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  protected:
  std::vector<hardware_interface::StateInterface> on_export_state_interfaces() override;

  controller_interface::return_type update_and_write_commands(const rclcpp::Time& time,
                                                              const rclcpp::Duration& period) override;

  controller_interface::return_type update_reference_from_subscribers(const rclcpp::Time& time,
                                                                      const rclcpp::Duration& period) override;

  void get_params();

  robot_IEKF* tron1;

  std::string package_name_ = "";
  std::string urdf_path_ = "";

  std::vector<int> perm;
  std::vector<int> contact_perm;

  pinocchio::Model pin_model_;
  pinocchio::Data pin_data_;
  std::shared_ptr<robot_store> robot_store_;
  std::shared_ptr<robot_params> robot_params_;

  Matrix3d R_body_imu_ = Matrix3d::Identity(); // calibrated transformation from body to imu

  double time_init = 0;
  int discrete_time = 0; // discrete time of the estimation
  int msg_num = 0;

  // vicon pos
  bool use_vicon_pos_ = false;
  std::string node_name_ = "";
  std::string tracker_name_ = "";
  rclcpp::Node::SharedPtr node_ptr_ = nullptr;
  rclcpp::executors::SingleThreadedExecutor executor_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr PoseEst_subscriber_ = nullptr;
  std::mutex vicon_pose_mutex_;
  Eigen::Quaterniond latest_vicon_orientation_w_{1.0, 0.0, 0.0, 0.0};
  bool latest_vicon_orientation_received_ = false;
  double yaw_offset_w_from_imu_ = 0.0;
  bool yaw_offset_initialized_ = false;

  // debug
  std::array<double, 2> contact_mob_{0.0, 0.0};
};


}; // namespace tron1_deploy


#endif  // TRON1_IEKF_ESTIMATOR_HPP
