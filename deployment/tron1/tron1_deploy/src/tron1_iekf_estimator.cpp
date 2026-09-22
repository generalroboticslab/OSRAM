#include "tron1_deploy/tron1_iekf_estimator.hpp"

#include <cmath>
#include <sstream>

namespace tron1_deploy
{
using namespace pinocchio_utils;

namespace
{
double yaw_from_quaternion(const Eigen::Quaterniond& q)
{
  const Eigen::Matrix3d R = q.normalized().toRotationMatrix();
  return std::atan2(R(1, 0), R(0, 0));
}

double wrap_to_pi(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}
}  // namespace

controller_interface::CallbackReturn Tron1IEKFEstimator::on_init()
{
  robot_store_ = std::make_shared<robot_store>();
  robot_params_ = std::make_shared<robot_params>();

  BaseHumanoidEstimator::on_init();
  get_params();

  // initialize the IEKF estimator
  pinocchio::urdf::buildModel(urdf_path_, pinocchio::JointModelFreeFlyer(), pin_model_);
  pin_data_ = pinocchio::Data(pin_model_);

  perm = build_joint_reorder_map(robot_params_->joint_names_, pin_model_);
  contact_perm = build_contact_reorder_map(robot_params_->contact_names_, pin_model_);

  tron1 = new robot_IEKF(pin_model_, pin_data_, robot_store_, robot_params_);

  Quaterniond q_body_imu;
  q_body_imu.w() = robot_params_->q_body_imu_[0];
  q_body_imu.x() = robot_params_->q_body_imu_[1];
  q_body_imu.y() = robot_params_->q_body_imu_[2];
  q_body_imu.z() = robot_params_->q_body_imu_[3];
  R_body_imu_ = q_body_imu.normalized().toRotationMatrix();

  debug_ = auto_declare<bool>("debug", false);
  if(debug_) {
    std::ostringstream pin_joint_order;
    pin_joint_order << "Pinocchio joint order:";
    for (pinocchio::JointIndex joint_id = 0; joint_id < static_cast<pinocchio::JointIndex>(pin_model_.names.size());
         ++joint_id)
    {
      pin_joint_order << "\n  [" << joint_id << "] " << pin_model_.names[joint_id]
                      << " idx_q=" << pin_model_.idx_qs[joint_id]
                      << " idx_v=" << pin_model_.idx_vs[joint_id]
                      << " nq=" << pin_model_.nqs[joint_id]
                      << " nv=" << pin_model_.nvs[joint_id];
    }
    RCLCPP_INFO(this->get_node()->get_logger(), "%s", pin_joint_order.str().c_str());

    size_t num_interfaces = state_interfaces_.size();
    REGISTER_ROS2_CONTROL_INTROSPECTION("contact_l", &contact_mob_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("contact_r", &contact_mob_[1]);
  }

  if (use_vicon_pos_){
    node_name_ = auto_declare<std::string>("node_name", "");
    tracker_name_ = auto_declare<std::string>("tracker_name", "");
    std::string subscribe_topic_name_pos = "/vrpn_mocap/" + tracker_name_ + "/pose";
    if (node_name_.empty() || tracker_name_.empty())
    {
      RCLCPP_ERROR(this->get_node()->get_logger(),
                  "Tron1IEKFEstimator: 'node_name' or 'tracker_name' parameter is empty.");
      return controller_interface::CallbackReturn::FAILURE;
    }
    auto topics = this->get_node()->get_topic_names_and_types();
    if (topics.find(subscribe_topic_name_pos) != topics.end())
    {
      // do nothing
    }
    else
    {
      RCLCPP_ERROR(this->get_node()->get_logger(), "Tron1IEKFEstimator: velocity topic '%s' does not exist.",
                  subscribe_topic_name_pos.c_str());
      return controller_interface::CallbackReturn::FAILURE;
    }
    node_ptr_ = std::make_shared<rclcpp::Node>(node_name_);
    auto qos = rclcpp::QoS(rclcpp::KeepLast(2), rmw_qos_profile_sensor_data);
    PoseEst_subscriber_ = node_ptr_->create_subscription<geometry_msgs::msg::PoseStamped>(
    subscribe_topic_name_pos, qos,
    [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
      Eigen::Quaterniond q_vicon(
          msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
      if (q_vicon.norm() > 1.0e-9)
      {
        q_vicon.normalize();
        if (q_vicon.w() < 0.0)
        {
          q_vicon.coeffs() *= -1.0;
        }
        std::lock_guard<std::mutex> lock(vicon_pose_mutex_);
        latest_vicon_orientation_w_ = q_vicon;
        latest_vicon_orientation_received_ = true;
      }
      pos_[0] = msg->pose.position.x;
      pos_[1] = msg->pose.position.y;
      pos_[2] = msg->pose.position.z;
    });
    executor_.add_node(node_ptr_);
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration Tron1IEKFEstimator::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::NONE;

  return config;
}

controller_interface::InterfaceConfiguration Tron1IEKFEstimator::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  auto imu_interface = this->get_imu_state_interface_configuration();
  config.names.insert(config.names.end(), imu_interface.names.begin(), imu_interface.names.end());
  auto joint_interface = this->get_joint_state_interface_configuration();
  config.names.insert(config.names.end(), joint_interface.names.begin(), joint_interface.names.end());
  auto contact_interface = this->get_contact_state_interface_configuration();
  config.names.insert(config.names.end(), contact_interface.names.begin(), contact_interface.names.end());

  return config;
}

controller_interface::CallbackReturn Tron1IEKFEstimator::on_configure(const rclcpp_lifecycle::State& previous_state)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1IEKFEstimator::on_activate(const rclcpp_lifecycle::State& previous_state)
{
  (void)previous_state;
  // reset the estimator
  discrete_time = 0;
  if (tron1) {
    delete tron1;
    tron1 = new robot_IEKF(pin_model_, pin_data_, robot_store_, robot_params_);
  }
  robot_store_->imu_time_ = 0.0;

  yaw_offset_initialized_ = false;
  yaw_offset_w_from_imu_ = 0.0;
  if (use_vicon_pos_)
  {
    Eigen::Quaterniond q_vicon;
    bool has_vicon_orientation = false;
    {
      std::lock_guard<std::mutex> lock(vicon_pose_mutex_);
      q_vicon = latest_vicon_orientation_w_;
      has_vicon_orientation = latest_vicon_orientation_received_;
    }

    if (has_vicon_orientation)
    {
      std::array<double, 3> imu_accel;
      std::array<double, 3> imu_gyro;
      std::array<double, 4> imu_orientation;
      read_imu_from_state_interfaces(imu_accel, imu_gyro, imu_orientation);
      Eigen::Quaterniond q_imu_raw(
          imu_orientation[0], imu_orientation[1], imu_orientation[2], imu_orientation[3]);
      if (q_imu_raw.norm() > 1.0e-9)
      {
        q_imu_raw.normalize();
        const double yaw_vicon = yaw_from_quaternion(q_vicon);
        const double yaw_imu = yaw_from_quaternion(q_imu_raw);
        yaw_offset_w_from_imu_ = wrap_to_pi(yaw_vicon - yaw_imu);
        yaw_offset_initialized_ = true;
        RCLCPP_INFO(this->get_node()->get_logger(),
                    "IEKF yaw alignment from Vicon: yaw_vicon=%.3f yaw_imu=%.3f offset=%.3f",
                    yaw_vicon, yaw_imu, yaw_offset_w_from_imu_);
      }
    }
    else
    {
      RCLCPP_WARN(this->get_node()->get_logger(),
                  "IEKF using Vicon position but no Vicon orientation has been received; yaw offset stays zero.");
    }
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1IEKFEstimator::on_deactivate(const rclcpp_lifecycle::State& previous_state)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> Tron1IEKFEstimator::on_export_state_interfaces()
{
  return BaseHumanoidEstimator::on_export_state_interfaces();
}

controller_interface::return_type Tron1IEKFEstimator::update_and_write_commands(const rclcpp::Time& time,
                                                                                  const rclcpp::Duration& period)
{
  // get measurements
  if (discrete_time == 0) {
    robot_store_->imu_time_ = 0.0;
  }
  robot_store_->imu_time_ += period.seconds();
  // std::cout << "current time: " << robot_store_->imu_time_ << " seconds" << std::endl;
  std::array<double, 3> imu_accel, imu_gyro;
  std::array<double, 4> imu_orientation;
  read_imu_from_state_interfaces(imu_accel, imu_gyro, imu_orientation);
  robot_store_->accel_b_ = R_body_imu_ * Vector3d(imu_accel[0], imu_accel[1], imu_accel[2]);
  robot_store_->omega_b_ = R_body_imu_ * Vector3d(imu_gyro[0], imu_gyro[1], imu_gyro[2]);
  // std::cout << "Duration since last update: " << period.seconds() << " seconds" << std::endl;
  VectorXd joint_position = VectorXd::Zero(joint_names_.size()); // [2 foot * 4 joints]
  VectorXd joint_velocity = VectorXd::Zero(joint_names_.size()); // [2 foot * 4 joints]
  VectorXd joint_effort = VectorXd::Zero(joint_names_.size());
  // VectorXd joint_position = VectorXd::Zero(robot_params_->dim_legs_ * robot_params_->num_legs_); // [2 foot * 4 joints]
  // VectorXd joint_velocity = VectorXd::Zero(robot_params_->dim_legs_ * robot_params_->num_legs_); // [2 foot * 4 joints]
  // VectorXd joint_effort = VectorXd::Zero(robot_params_->dim_legs_ * robot_params_->num_legs_);
  // if (joint_position.size() != joint_names_.size())
  // {
  //   RCLCPP_ERROR(this->get_node()->get_logger(), "Size of joint position vector does not match number of joint names");
  //   return controller_interface::return_type::ERROR;
  // }
  std::vector<double> joint_position_vec, joint_velocity_vec, joint_effort_vec;
  read_joint_states_from_state_interfaces(joint_position_vec, joint_velocity_vec, joint_effort_vec);
  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    joint_position(i) = joint_position_vec[i];
    joint_velocity(i) = joint_velocity_vec[i];
    joint_effort(i) = joint_effort_vec[i];
    joint_pos_[i] = joint_position_vec[i];
    joint_vel_[i] = joint_velocity_vec[i];
    joint_tau_[i] = joint_effort_vec[i];
  }
  // std::cout<< "joint positions: " << joint_position.transpose() << std::endl;
  robot_store_->joint_states_position_ = reorder_joint_to_pinocchio(joint_position, perm);
  robot_store_->joint_states_velocity_ = reorder_joint_to_pinocchio(joint_velocity, perm);
  robot_store_->joint_states_effort_ = reorder_joint_to_pinocchio(joint_effort, perm);
  // std::cout << "joint positions after reorder: " << robot_store_->joint_states_position_.transpose() << std::endl;
  std::array<double, 2> contact_array;
  read_contact_states_from_state_interfaces(contact_array);
  // std::cout << "contact states: " << contact_array[0] << ", " << contact_array[1] << std::endl;
  VectorXd contact = VectorXd::Zero(robot_params_->num_legs_);
  for (int i=0; i < robot_params_->num_legs_; ++i) {
    contact(i) = contact_array[i];
  }
  robot_store_->contact_ = reorder_contact_to_pinocchio(contact, contact_perm);

  // wxyz
  Eigen::Quaterniond q_imu_raw(imu_orientation[0], imu_orientation[1], imu_orientation[2], imu_orientation[3]);
  if (q_imu_raw.norm() < 1.0e-9)
  {
    q_imu_raw = Eigen::Quaterniond::Identity();
  }
  else
  {
    q_imu_raw.normalize();
  }

  Eigen::Quaterniond q_for_iekf = q_imu_raw;
  if (use_vicon_pos_ && yaw_offset_initialized_)
  {
    const Eigen::Quaterniond q_yaw_offset(
        Eigen::AngleAxisd(yaw_offset_w_from_imu_, Eigen::Vector3d::UnitZ()));
    q_for_iekf = (q_yaw_offset * q_imu_raw).normalized();
  }
  robot_store_->quaternion_ = q_for_iekf;
  // std::cout << q_for_iekf.w() << " " << q_for_iekf.x() << " " << q_for_iekf.y() << " " << q_for_iekf.z() << std::endl;

  // call init or update
  if (discrete_time == 0){
    tron1->initialize();
  }
  else{
    tron1->update(discrete_time);
  }
  discrete_time++;

  // pub estimated states
  auto pos = tron1->x_est_.segment<3>(0);
  auto vel = tron1->x_est_.segment<3>(3);
  auto q = tron1->x_est_.segment<4>(6);
  auto omega = robot_store_->omega_b_;

  // unwinding
  double dot = q(0) * ori_[0] + q(1) * ori_[1] + q(2) * ori_[2] + q(3) * ori_[3];
  if (dot < 0.0f) {
    q = -q;
  }

  ori_[0] = q(0); // wxyz
  ori_[1] = q(1);
  ori_[2] = q(2);
  ori_[3] = q(3);

  Quaterniond quat(q(0), q(1), q(2), q(3)); // w, x, y, z
  Matrix3d R = quat.toRotationMatrix();

  Vector3d omega_w = R * omega;
  Vector3d accel_w = R * robot_store_->accel_b_;

  for (size_t i = 0; i < 3; ++i)
  {
    if (!use_vicon_pos_){
      pos_[i] = pos(i);
    }
    lin_vel_[i] = vel(i);
    ang_vel_[i] = omega_w(i);
    lin_acc_[i] = accel_w(i);
  }

  // vicon_pos
  if (use_vicon_pos_){
    executor_.spin_some(std::chrono::milliseconds(1));
  }

  // debug
  if (debug_)
  {
    contact_mob_[0] = tron1->contact_(0);
    contact_mob_[1] = tron1->contact_(1);
  }
  return controller_interface::return_type::OK;
}

controller_interface::return_type Tron1IEKFEstimator::update_reference_from_subscribers(const rclcpp::Time& time,
                                                                                          const rclcpp::Duration& period)
{
  return controller_interface::return_type::OK;
}

void Tron1IEKFEstimator::get_params()
{
  // whether use vicon pos
  use_vicon_pos_ = auto_declare<bool>("use_vicon_pos", false);

  // filter params
  package_name_ = auto_declare<std::string>("package_name", "tron1_description");
  urdf_path_ = auto_declare<std::string>("urdf_path", "urdf/tron1_flat_foot.urdf");
  urdf_path_ = file_utils::resolve_package_path(package_name_) + "/" + urdf_path_;

  robot_params_->log_name_ = auto_declare<std::string>("log_name", "tron1_iekf_log");
  robot_params_->urdf_path_ = urdf_path_;
  robot_params_->visualize_ = auto_declare<int>("visualize", 0);

  robot_params_->rate_ = auto_declare<int>("rate", 0); // in hz
  robot_params_->N_ = auto_declare<int>("N", 0);
  robot_params_->using_lo_p_ = auto_declare<int>("using_lo_p", 0);
  robot_params_->using_lo_v_ = auto_declare<int>("using_lo_v", 0);
  robot_params_->using_mob_ = auto_declare<int>("using_mob", 0);

  robot_params_->num_legs_ = auto_declare<int>("num_legs", 0);
  robot_params_->dim_legs_ = auto_declare<int>("dim_legs", 0);
  robot_params_->contact_names_ = auto_declare<std::vector<std::string>>("contact_names", {"lf", "rf"});
  robot_params_->contact_theshold_ = auto_declare<double>("contact_effort_threshold", 0.0);

  // Priors
  robot_params_->p_init_std_ = auto_declare<std::vector<double>>("p_init_std", {0.0, 0.0, 0.0});
  robot_params_->v_init_std_ = auto_declare<std::vector<double>>("v_init_std", {0.0, 0.0, 0.0});
  robot_params_->orientation_init_std_ = auto_declare<double>("orientation_init_std", 0.0);
  robot_params_->foot_init_std_ = auto_declare<double>("foot_init_std", 0.0);
  robot_params_->accel_bias_init_std_ = auto_declare<double>("accel_bias_init_std", 0.0);
  robot_params_->angular_bias_init_std_ = auto_declare<double>("angular_bias_init_std", 0.0);

  // Process and Measurement Noise
  robot_params_->p_process_std_ = auto_declare<std::vector<double>>("p_process_std", {0.0, 0.0, 0.0});
  robot_params_->accel_input_std_ = auto_declare<std::vector<double>>("accel_input_std", {0.0, 0.0, 0.0});
  robot_params_->gyro_input_std_ = auto_declare<std::vector<double>>("gyro_input_std", {0.0, 0.0, 0.0});
  robot_params_->accel_bias_process_std_ = auto_declare<std::vector<double>>("accel_bias_process_std", {0.0, 0.0, 0.0});
  robot_params_->angular_bias_process_std_ = auto_declare<std::vector<double>>("angular_bias_process_std", {0.0, 0.0, 0.0});
  robot_params_->foot_slide_std_ = auto_declare<std::vector<double>>("foot_slide_std", {0.0, 0.0});
  robot_params_->foot_swing_std_ = auto_declare<std::vector<double>>("foot_swing_std", {0.0, 0.0});
  robot_params_->joint_position_std_ = auto_declare<std::vector<double>>("joint_position_std", std::vector<double>(joint_names_.size(), 0.0));
  robot_params_->joint_velocity_std_ = auto_declare<std::vector<double>>("joint_velocity_std", std::vector<double>(joint_names_.size(), 0.0));

  // IMU Extrinsics
  robot_params_->q_body_imu_ = auto_declare<std::vector<double>>("q_body_imu", {1.0, 0.0, 0.0, 0.0});
  robot_params_->p_body_imu_ = auto_declare<std::vector<double>>("p_body_imu", {0.0, 0.0, 0.0});
  robot_params_->joint_names_ = joint_names_;

  if (robot_params_->num_legs_ != 2) {
    RCLCPP_ERROR(this->get_node()->get_logger(), "\num_legs should be 2 for tron1!");
  }
}


}; // namespace tron1_deploy

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(tron1_deploy::Tron1IEKFEstimator, controller_interface::ChainableControllerInterface);
