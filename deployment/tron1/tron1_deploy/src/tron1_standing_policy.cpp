#include "tron1_deploy/tron1_standing_policy.hpp"

namespace tron1_deploy
{

controller_interface::CallbackReturn Tron1StandingPolicy::on_init()
{
  controller_interface::CallbackReturn state = BaseHumanoidController::on_init();

  // resize vectors
  observations_.resize(36, 0.0);
  actions_.resize(joint_names_.size(), 0.0);
  vel_refs_.resize(3, 0.0);
  joint_pos_.resize(joint_names_.size(), 0.0);
  joint_vel_.resize(joint_names_.size(), 0.0);
  joint_pos_des_.resize(joint_names_.size(), 0.0);
  joint_pos_init_.resize(joint_names_.size(), 0.0);
  joint_vel_init_.resize(joint_names_.size(), 0.0);
  kp_gains_.resize(joint_names_.size(), 0.0);
  kd_gains_.resize(joint_names_.size(), 0.0);
  action_scales_.resize(joint_names_.size(), 0.0);

  // init policy
  std::string package_name = auto_declare<std::string>("package_name", "tron1_deploy");
  std::string policy_path = auto_declare<std::string>("policy_path", "");
  std::string resolved_policy_path = file_utils::resolve_file_path(policy_path, package_name);
  velocity_policy_ptr_ =
      std::make_shared<base_controllers::OnnxPolicy>(resolved_policy_path, observations_.size(), actions_.size());

  // init pd gains and targets from parameters if exist
  joint_pos_init_ = auto_declare<std::vector<double>>("joint_pos_ref", joint_pos_init_);
  joint_vel_init_ = auto_declare<std::vector<double>>("joint_vel_ref", joint_vel_init_);
  kp_gains_ = auto_declare<std::vector<double>>("kp_gains", kp_gains_);
  kd_gains_ = auto_declare<std::vector<double>>("kd_gains", kd_gains_);
  action_scales_ = auto_declare<std::vector<double>>("action_scales", action_scales_);

  // debug
  if (debug_)
  {
    REGISTER_ROS2_CONTROL_INTROSPECTION("lin_vel_x_local", &base_lin_vel_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("lin_vel_y_local", &base_lin_vel_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ang_vel_yaw_local", &base_ang_vel_[2]);
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration Tron1StandingPolicy::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config =
    humanoid_controllers::BaseHumanoidController::get_joint_command_interface_configuration();

  return config;
}

controller_interface::InterfaceConfiguration Tron1StandingPolicy::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // state estimation
  if (estimator_name_ == "")
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1StandingPolicy::state_interface_configuration() failed because estimator_name is not set.");
  }
  else
  {
    // Joint interfaces
    for (const auto& joint : joint_names_)
    {
      for (const auto& iface : joint_state_interface_types_)
      {
        config.names.push_back(estimator_name_ + "/" + joint + "_" + iface + "_est");
      }
    }

    // Contact and force wrench sensors
    for (const auto& foot : foot_names_)
    {
      for (const auto& sensor : foot_sensor_names_)
      {
        config.names.push_back(estimator_name_ + "/" + foot + "_" + sensor + "_est");
      }
    }

    // Position (x, y, z)
    config.names.push_back(estimator_name_ + "/" + pos_name_ + "_x_est");
    config.names.push_back(estimator_name_ + "/" + pos_name_ + "_y_est");
    config.names.push_back(estimator_name_ + "/" + pos_name_ + "_z_est");

    // Orientation (w, x, y, z)
    config.names.push_back(estimator_name_ + "/" + ori_name_ + "_w_est");
    config.names.push_back(estimator_name_ + "/" + ori_name_ + "_x_est");
    config.names.push_back(estimator_name_ + "/" + ori_name_ + "_y_est");
    config.names.push_back(estimator_name_ + "/" + ori_name_ + "_z_est");

    // Linear velocity (x, y, z)
    config.names.push_back(estimator_name_ + "/" + lin_vel_name_ + "_x_est");
    config.names.push_back(estimator_name_ + "/" + lin_vel_name_ + "_y_est");
    config.names.push_back(estimator_name_ + "/" + lin_vel_name_ + "_z_est");

    // Angular velocity (x, y, z)
    config.names.push_back(estimator_name_ + "/" + ang_vel_name_ + "_x_est");
    config.names.push_back(estimator_name_ + "/" + ang_vel_name_ + "_y_est");
    config.names.push_back(estimator_name_ + "/" + ang_vel_name_ + "_z_est");
  }

  return config;
}

controller_interface::CallbackReturn Tron1StandingPolicy::on_configure(const rclcpp_lifecycle::State& previous_state)
{
  controller_interface::CallbackReturn state = BaseHumanoidController::on_configure(previous_state);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1StandingPolicy::on_activate(const rclcpp_lifecycle::State& previous_state)
{
  // initialize actions
  std::vector<std::string> joint_position_interfaces;
  joint_position_interfaces.reserve(joint_names_.size());
  for (const auto& joint_name : joint_names_)
  {
    joint_position_interfaces.emplace_back(estimator_name_ + "/" + joint_name + "_position_est");
  }
  base_utils::get_state_interface_values(state_interfaces_, joint_position_interfaces, joint_pos_);
  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    actions_[i] = (joint_pos_[i] - joint_pos_init_[i]) / action_scales_[i];
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1StandingPolicy::on_deactivate(const rclcpp_lifecycle::State& previous_state)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type Tron1StandingPolicy::update_and_write_commands(const rclcpp::Time& time,
                                                                                 const rclcpp::Duration& period)
{
  // gather observations
  // 0. orientation
  std::vector<double> orientation_components(4, 0.0);
  base_utils::get_state_interface_values(
      state_interfaces_,
      {estimator_name_ + "/" + ori_name_ + "_w_est", estimator_name_ + "/" + ori_name_ + "_x_est",
       estimator_name_ + "/" + ori_name_ + "_y_est", estimator_name_ + "/" + ori_name_ + "_z_est"},
      orientation_components);
  ori_ = Eigen::Quaterniond(orientation_components[0], orientation_components[1], orientation_components[2],
                            orientation_components[3]);
  ori_.normalize();

  // 1. base_lin_vel, should be velocimeter readings
  std::vector<double> lin_vel_components(3, 0.0);
  base_utils::get_state_interface_values(
      state_interfaces_,
      {estimator_name_ + "/" + lin_vel_name_ + "_x_est", estimator_name_ + "/" + lin_vel_name_ + "_y_est",
       estimator_name_ + "/" + lin_vel_name_ + "_z_est"},
      lin_vel_components);
  const Eigen::Vector3d base_lin_vel_global(lin_vel_components[0], lin_vel_components[1], lin_vel_components[2]);
  base_lin_vel_ = ori_.inverse() * base_lin_vel_global;

  // 2. base_ang_vel, should be gyroscope readings
  std::vector<double> ang_vel_components(3, 0.0);
  base_utils::get_state_interface_values(
      state_interfaces_,
      {estimator_name_ + "/" + ang_vel_name_ + "_x_est", estimator_name_ + "/" + ang_vel_name_ + "_y_est",
       estimator_name_ + "/" + ang_vel_name_ + "_z_est"},
      ang_vel_components);
  const Eigen::Vector3d base_ang_vel_global(ang_vel_components[0], ang_vel_components[1], ang_vel_components[2]);
  base_ang_vel_ = ori_.inverse() * base_ang_vel_global;

  // 3. projected gravity
  rl_utils::projected_gravity(ori_, projected_gravity_);

  // 4. joint_pos
  // 5. joint_vel
  std::vector<std::string> joint_position_interfaces;
  std::vector<std::string> joint_velocity_interfaces;
  joint_position_interfaces.reserve(joint_names_.size());
  joint_velocity_interfaces.reserve(joint_names_.size());
  for (const auto& joint_name : joint_names_)
  {
    joint_position_interfaces.emplace_back(estimator_name_ + "/" + joint_name + "_position_est");
    joint_velocity_interfaces.emplace_back(estimator_name_ + "/" + joint_name + "_velocity_est");
  }
  base_utils::get_state_interface_values(state_interfaces_, joint_position_interfaces, joint_pos_);
  base_utils::get_state_interface_values(state_interfaces_, joint_velocity_interfaces, joint_vel_);
  std::vector<double> obs_joint_pos = joint_pos_;  //
  std::vector<double> obs_joint_vel = joint_vel_;
  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    obs_joint_pos[i] -= joint_pos_init_[i];
    obs_joint_vel[i] -= joint_vel_init_[i];
  }

  // 6. actions
  // 7. commands, always zero

  // 8. concatenate as observations_
  observations_.clear();
  observations_.reserve(36);
  for (int i = 0; i < 3; ++i)
  {
    observations_.push_back(base_lin_vel_[i]);
  }
  for (int i = 0; i < 3; ++i)
  {
    observations_.push_back(base_ang_vel_[i]);
  }
  for (int i = 0; i < 3; ++i)
  {
    observations_.push_back(projected_gravity_[i]);
  }
  observations_.insert(observations_.end(), obs_joint_pos.begin(), obs_joint_pos.end());
  observations_.insert(observations_.end(), obs_joint_vel.begin(), obs_joint_vel.end());
  observations_.insert(observations_.end(), actions_.begin(), actions_.end());
  observations_.insert(observations_.end(), vel_refs_.begin(), vel_refs_.end());

  // inference policy
  actions_ = velocity_policy_ptr_->infer(observations_);

  // action to desired joint positions
  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    joint_pos_des_[i] = joint_pos_init_[i] + actions_[i] * action_scales_[i];
  }

  // apply cmd
  std::vector<std::string> position_interfaces;
  std::vector<std::string> kp_interfaces;
  std::vector<std::string> kd_interfaces;
  position_interfaces.reserve(joint_names_.size());
  kp_interfaces.reserve(joint_names_.size());
  kd_interfaces.reserve(joint_names_.size());
  for (const auto& joint_name : joint_names_)
  {
    position_interfaces.emplace_back(joint_name + "/position");
    kp_interfaces.emplace_back(joint_name + "/kp");
    kd_interfaces.emplace_back(joint_name + "/kd");
  }
  base_utils::set_command_interface_values(command_interfaces_, position_interfaces, joint_pos_des_);
  base_utils::set_command_interface_values(command_interfaces_, kp_interfaces, kp_gains_);
  base_utils::set_command_interface_values(command_interfaces_, kd_interfaces, kd_gains_);

  return controller_interface::return_type::OK;
}

std::vector<hardware_interface::CommandInterface> Tron1StandingPolicy::on_export_reference_interfaces()
{
  std::vector<hardware_interface::CommandInterface> reference_interfaces;
  std::string controller_name = this->get_node()->get_name();
  reference_interfaces_.resize(1, 0.0);
  // lin_x_vel, lin_y_vel, ang_z_vel
  reference_interfaces.emplace_back(controller_name, "dummy", &vel_refs_[0]);

  return reference_interfaces;
}
};  // namespace tron1_deploy

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(tron1_deploy::Tron1StandingPolicy, controller_interface::ChainableControllerInterface);
