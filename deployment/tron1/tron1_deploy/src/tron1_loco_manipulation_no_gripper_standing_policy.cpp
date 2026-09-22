#include "tron1_deploy/tron1_loco_manipulation_no_gripper_standing_policy.hpp"

namespace tron1_deploy
{

controller_interface::CallbackReturn Tron1LocoManipulationNoGripperStandingPolicy::on_init()
{
  auto ret = humanoid_controllers::BaseHumanoidController::on_init();
  if (ret != controller_interface::CallbackReturn::SUCCESS)
  {
    return ret;
  }

  if (joint_names_.size() != joint_dim_)
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1LocoManipulationNoGripperStandingPolicy::on_init() failed because joint_names size must be %zu, got %zu.",
                 joint_dim_, joint_names_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  joint_pos_.assign(joint_dim_, 0.0);
  joint_vel_.assign(joint_dim_, 0.0);
  joint_pos_des_.assign(joint_dim_, 0.0);
  joint_vel_des_.assign(joint_dim_, 0.0);
  joint_tau_des_.assign(joint_dim_, 0.0);
  joint_pos_init_.assign(joint_dim_, 0.0);
  joint_vel_init_.assign(joint_dim_, 0.0);
  kp_gains_.assign(joint_dim_, 0.0);
  kd_gains_.assign(joint_dim_, 0.0);
  action_scales_.assign(joint_dim_, 1.0);
  actions_.assign(action_dim_, 0.0);

  joint_pos_init_ = auto_declare<std::vector<double>>("joint_pos_ref", joint_pos_init_);
  joint_vel_init_ = auto_declare<std::vector<double>>("joint_vel_ref", joint_vel_init_);
  kp_gains_ = auto_declare<std::vector<double>>("kp_gains", kp_gains_);
  kd_gains_ = auto_declare<std::vector<double>>("kd_gains", kd_gains_);
  action_scales_ = auto_declare<std::vector<double>>("action_scales", action_scales_);
  use_policy_arm_actions_ = auto_declare<bool>("use_policy_arm_actions", use_policy_arm_actions_);

  if (joint_pos_init_.size() < joint_dim_ || joint_vel_init_.size() < joint_dim_ ||
      kp_gains_.size() < joint_dim_ || kd_gains_.size() < joint_dim_ ||
      action_scales_.size() < joint_dim_)
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1LocoManipulationNoGripperStandingPolicy::on_init() failed because joint_pos_ref, joint_vel_ref, kp_gains, kd_gains, and action_scales must all have at least %zu values.",
                 joint_dim_);
    return controller_interface::CallbackReturn::ERROR;
  }
  joint_pos_init_.resize(joint_dim_);
  joint_vel_init_.resize(joint_dim_);
  kp_gains_.resize(joint_dim_);
  kd_gains_.resize(joint_dim_);
  action_scales_.resize(joint_dim_);

  initialize_observation_manager();
  initialize_action_manager();
  observations_.assign(observation_manager_.total_observation_dim(), 0.0);
  actions_.assign(action_manager_.total_action_dim(), 0.0);

  if (observations_.size() != observation_dim_ || actions_.size() != action_dim_)
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1LocoManipulationNoGripperStandingPolicy manager dimensions mismatch: obs=%zu action=%zu.",
                 observations_.size(), actions_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  std::string description_package_name = auto_declare<std::string>("description_package_name", "tron1_description");
  std::string urdf_path = auto_declare<std::string>("urdf_path", "");
  const std::string resolved_urdf_path = file_utils::resolve_file_path(urdf_path, description_package_name);
  pinocchio::urdf::buildModel(resolved_urdf_path, pinocchio::JointModelFreeFlyer(), model_);
  data_ = pinocchio::Data(model_);
  q_ = pinocchio::neutral(model_);
  v_ = Eigen::VectorXd::Zero(model_.nv);
  perm_ = pinocchio_utils::build_joint_reorder_map(joint_names_, model_);

  std::string policy_package_name = auto_declare<std::string>("policy_package_name", "tron1_deploy");
  std::string policy_path = auto_declare<std::string>("policy_path", "");
  const std::string resolved_policy_path = file_utils::resolve_file_path(policy_path, policy_package_name);
  base_controllers::OnnxMLPCfg policy_cfg;
  policy_cfg.model_path = resolved_policy_path;
  policy_cfg.input_size = static_cast<int>(observations_.size());
  policy_cfg.output_size = static_cast<int>(actions_.size());
  policy_cfg.validate_model_io = false;
  policy_ptr_ = std::make_shared<base_controllers::OnnxMLP>(policy_cfg);

  if (debug_)
  {
    REGISTER_ROS2_CONTROL_INTROSPECTION("standing_base_lin_vel_x", &base_lin_vel_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("standing_base_lin_vel_y", &base_lin_vel_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("standing_base_ang_vel_z", &base_ang_vel_[2]);
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration Tron1LocoManipulationNoGripperStandingPolicy::command_interface_configuration() const
{
  return humanoid_controllers::BaseHumanoidController::get_joint_command_interface_configuration();
}

controller_interface::InterfaceConfiguration Tron1LocoManipulationNoGripperStandingPolicy::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  if (estimator_name_.empty())
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1LocoManipulationNoGripperStandingPolicy::state_interface_configuration() failed because estimator_name is not set.");
    return config;
  }

  auto joint_state_config = get_joint_state_interface_configuration();
  config.names.insert(config.names.end(), joint_state_config.names.begin(), joint_state_config.names.end());

  auto pos_config = get_global_pos_interface_configuration();
  config.names.insert(config.names.end(), pos_config.names.begin(), pos_config.names.end());

  auto ori_config = get_ori_interface_configuration();
  config.names.insert(config.names.end(), ori_config.names.begin(), ori_config.names.end());

  auto lin_vel_config = get_global_lin_vel_interface_configuration();
  config.names.insert(config.names.end(), lin_vel_config.names.begin(), lin_vel_config.names.end());

  auto ang_vel_config = get_global_ang_vel_interface_configuration();
  config.names.insert(config.names.end(), ang_vel_config.names.begin(), ang_vel_config.names.end());

  return config;
}

controller_interface::CallbackReturn Tron1LocoManipulationNoGripperStandingPolicy::on_configure(
    const rclcpp_lifecycle::State& previous_state)
{
  return humanoid_controllers::BaseHumanoidController::on_configure(previous_state);
}

controller_interface::CallbackReturn Tron1LocoManipulationNoGripperStandingPolicy::on_activate(
    const rclcpp_lifecycle::State& previous_state)
{
  auto ret = humanoid_controllers::BaseHumanoidController::on_activate(previous_state);
  if (ret != controller_interface::CallbackReturn::SUCCESS)
  {
    return ret;
  }

  (void)previous_state;
  read_joint_states_from_state_interfaces(joint_pos_, joint_vel_);
  actions_.assign(action_dim_, 0.0);
  for (size_t i = 0; i < joint_dim_; ++i)
  {
    const double scale = std::abs(action_scales_[i]) > 1.0e-9 ? action_scales_[i] : 1.0;
    actions_[i] = (joint_pos_[i] - joint_pos_init_[i]) / scale;
  }
  action_manager_.process_action(actions_);
  observation_manager_.reset();

  joint_pos_des_ = joint_pos_;
  joint_vel_des_.assign(joint_dim_, 0.0);
  joint_tau_des_.assign(joint_dim_, 0.0);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1LocoManipulationNoGripperStandingPolicy::on_deactivate(
    const rclcpp_lifecycle::State& previous_state)
{
  (void)previous_state;
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type Tron1LocoManipulationNoGripperStandingPolicy::update_and_write_commands(
    const rclcpp::Time& time, const rclcpp::Duration& period)
{
  (void)time;
  (void)period;

  update_robot_state();
  observations_ = observation_manager_.compute();
  actions_ = policy_ptr_->infer(observations_);
  if (actions_.size() != action_dim_)
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1LocoManipulationNoGripperStandingPolicy expected action size %zu, got %zu.",
                 action_dim_, actions_.size());
    return controller_interface::return_type::ERROR;
  }

  action_manager_.process_action(actions_);
  apply_policy_actions();

  write_joint_commands_to_command_interfaces(joint_pos_des_, joint_vel_des_, joint_tau_des_, kp_gains_, kd_gains_);
  return controller_interface::return_type::OK;
}

std::vector<hardware_interface::CommandInterface> Tron1LocoManipulationNoGripperStandingPolicy::on_export_reference_interfaces()
{
  std::vector<hardware_interface::CommandInterface> reference_interfaces;
  const std::string controller_name = this->get_node()->get_name();
  reference_interfaces_.resize(1, 0.0);
  reference_interfaces.emplace_back(controller_name, "dummy", &reference_interfaces_[0]);
  return reference_interfaces;
}

void Tron1LocoManipulationNoGripperStandingPolicy::update_robot_state()
{
  std::array<double, 4> orientation_components{1.0, 0.0, 0.0, 0.0};
  read_ori_from_state_interfaces(orientation_components);
  ori_ = Eigen::Quaterniond(
      orientation_components[0],
      orientation_components[1],
      orientation_components[2],
      orientation_components[3]);
  ori_.normalize();
  if (ori_.w() < 0.0)
  {
    ori_.coeffs() *= -1.0;
  }

  std::array<double, 3> lin_vel_global{0.0, 0.0, 0.0};
  std::array<double, 3> ang_vel_global{0.0, 0.0, 0.0};
  read_global_lin_vel_from_state_interfaces(lin_vel_global);
  read_global_ang_vel_from_state_interfaces(ang_vel_global);
  base_lin_vel_ = ori_.inverse() * Eigen::Vector3d(lin_vel_global[0], lin_vel_global[1], lin_vel_global[2]);
  base_ang_vel_ = ori_.inverse() * Eigen::Vector3d(ang_vel_global[0], ang_vel_global[1], ang_vel_global[2]);

  read_joint_states_from_state_interfaces(joint_pos_, joint_vel_);

  std::array<double, 3> base_pos{0.0, 0.0, 0.0};
  read_global_pos_from_state_interfaces(base_pos);
  base_pos_w_ = Eigen::Vector3d(base_pos[0], base_pos[1], base_pos[2]);

  q_ = pinocchio::neutral(model_);
  q_.segment<3>(0) = base_pos_w_;
  q_.segment<4>(3) << ori_.x(), ori_.y(), ori_.z(), ori_.w();
  for (size_t i = 0; i < joint_pos_.size(); ++i)
  {
    q_(7 + perm_[i]) = joint_pos_[i];
  }

  v_.setZero();
  v_.segment<3>(0) = base_lin_vel_;
  v_.segment<3>(3) = base_ang_vel_;
  for (size_t i = 0; i < joint_vel_.size(); ++i)
  {
    v_(6 + perm_[i]) = joint_vel_[i];
  }
}

void Tron1LocoManipulationNoGripperStandingPolicy::initialize_observation_manager()
{
  observation_manager_ = observation::ObservationManager();

  auto add_observation = [this](const std::string& name, int dim, std::function<std::vector<double>()> func)
  {
    observation::ObservationTermCfg cfg;
    cfg.name = name;
    cfg.observation_dim = dim;
    cfg.history_length = 1;
    observation_manager_.add_term(std::make_unique<observation::ObservationTerm>(cfg, std::move(func)));
  };

  add_observation("base_lin_vel", 3,
      [this]() -> std::vector<double> { return observation::passthrough(base_lin_vel_); });
  add_observation("base_ang_vel", 3,
      [this]() -> std::vector<double> { return observation::passthrough(base_ang_vel_); });
  add_observation("base_ori", 4,
      [this]() -> std::vector<double> { return {ori_.w(), ori_.x(), ori_.y(), ori_.z()}; });
  add_observation("joint_pos", static_cast<int>(joint_dim_),
      [this]() -> std::vector<double> { return observation::joint_values(joint_pos_, joint_pos_init_); });
  add_observation("joint_vel", static_cast<int>(joint_dim_),
      [this]() -> std::vector<double> { return observation::joint_values(joint_vel_, joint_vel_init_); });
  add_observation("actions", static_cast<int>(action_dim_),
      [this]() -> std::vector<double> { return observation::passthrough(actions_); });
}

void Tron1LocoManipulationNoGripperStandingPolicy::initialize_action_manager()
{
  action_manager_ = action::ActionManager();

  action::ActionTermCfg leg_cfg;
  leg_cfg.name = "joint_pos";
  leg_cfg.action_dim = static_cast<int>(leg_joint_dim_);
  leg_cfg.action_scale = std::vector<double>(
      action_scales_.begin(), action_scales_.begin() + static_cast<std::ptrdiff_t>(leg_joint_dim_));
  leg_cfg.action_offset = std::vector<double>(
      joint_pos_init_.begin(), joint_pos_init_.begin() + static_cast<std::ptrdiff_t>(leg_joint_dim_));
  action_manager_.add_term(std::make_unique<action::JointPositionActionTerm>(leg_cfg));

  action::ActionTermCfg arm_cfg;
  arm_cfg.name = "joint_impedance";
  arm_cfg.action_dim = static_cast<int>(arm_joint_dim_);
  arm_cfg.action_scale = std::vector<double>(
      action_scales_.begin() + static_cast<std::ptrdiff_t>(leg_joint_dim_),
      action_scales_.begin() + static_cast<std::ptrdiff_t>(joint_dim_));
  arm_cfg.action_offset = std::vector<double>(
      joint_pos_init_.begin() + static_cast<std::ptrdiff_t>(leg_joint_dim_),
      joint_pos_init_.begin() + static_cast<std::ptrdiff_t>(joint_dim_));
  action_manager_.add_term(std::make_unique<action::JointImpedanceActionTerm>(arm_cfg));
}

void Tron1LocoManipulationNoGripperStandingPolicy::apply_policy_actions()
{
  joint_pos_des_ = joint_pos_;
  joint_vel_des_.assign(joint_dim_, 0.0);
  joint_tau_des_.assign(joint_dim_, 0.0);

  const auto processed_actions = action_manager_.processed_actions();
  if (processed_actions.size() != action_dim_)
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1LocoManipulationNoGripperStandingPolicy expected processed action size %zu, got %zu.",
                 action_dim_, processed_actions.size());
    return;
  }

  for (size_t i = 0; i < joint_dim_; ++i)
  {
    const bool is_arm_joint = i >= leg_joint_dim_;
    joint_pos_des_[i] = is_arm_joint && !use_policy_arm_actions_ ? 0.0 : processed_actions[i];
  }

  const Eigen::VectorXd bias_torque = pinocchio::nonLinearEffects(model_, data_, q_, v_);
  const Eigen::VectorXd joint_bias_torque = pinocchio_utils::reorder_pinocchio_to_joint(
      bias_torque.tail(static_cast<Eigen::Index>(perm_.size())), perm_);
  for (size_t i = leg_joint_dim_; i < joint_dim_ && i < static_cast<size_t>(joint_bias_torque.size()); ++i)
  {
    joint_tau_des_[i] = joint_bias_torque(static_cast<Eigen::Index>(i));
  }
}

} // namespace tron1_deploy

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(tron1_deploy::Tron1LocoManipulationNoGripperStandingPolicy, controller_interface::ChainableControllerInterface)
