#include "tron1_deploy/tron1_loco_manipulation.hpp"

namespace tron1_deploy
{

controller_interface::CallbackReturn Tron1LocoManipulation::on_init()
{
  controller_interface::CallbackReturn state = BaseHumanoidController::on_init();
  if (planner_name_ != "" && ref_controller_name_ != "")
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1VelocityPolicy::on_init() failed because both planner_name and ref_controller_name are set.");
    return controller_interface::CallbackReturn::ERROR;
  }

  policy_type_ = auto_declare<std::string>("policy_type", policy_type_);
  steer_type_ = auto_declare<std::string>("steer_type", steer_type_);
  // sanity check
  if (policy_type_ != "base" && policy_type_ != "rnn" && policy_type_ != "residual")
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1VelocityPolicy::on_init() failed because policy_type '%s' is invalid. Supported types are 'base', 'rnn', and 'residual'.",
                 policy_type_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  if (steer_type_ != "none" && steer_type_ != "closed" && steer_type_ != "open_full" && steer_type_ != "open_ref")
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1VelocityPolicy::on_init() failed because steer_type '%s' is invalid. Supported types are 'none', 'closed', 'open_full', and 'open_ref'.",
                 steer_type_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  if(policy_type_ != "base" && steer_type_ != "none")
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1VelocityPolicy::on_init() failed because steer_type '%s' is not supported when policy_type is not 'base'. steer_type has to be 'none' when policy_type is 'rnn' or 'residual'.",
                 steer_type_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  // resize vectors
  actions_.resize(joint_names_.size() - 1, 0.0); // gripper_joint1 is follows the gripper joint
  vel_refs_.resize(3, 0.0);
  ee_pose_ref_.resize(7, 0.0);
  joint_pos_.resize(joint_names_.size(), 0.0);
  joint_vel_.resize(joint_names_.size(), 0.0);
  joint_pos_des_.resize(joint_names_.size(), 0.0);
  joint_pos_init_.resize(joint_names_.size(), 0.0);
  joint_vel_init_.resize(joint_names_.size(), 0.0);
  kp_gains_.resize(joint_names_.size(), 0.0);
  kd_gains_.resize(joint_names_.size(), 0.0);
  action_scales_.resize(joint_names_.size() - 1, 0.0);
  residual_action_scales_.resize(joint_names_.size() - 1, 0.0);

  // init pd gains and targets from parameters if exist
  joint_pos_init_ = auto_declare<std::vector<double>>("joint_pos_ref", joint_pos_init_);
  joint_vel_init_ = auto_declare<std::vector<double>>("joint_vel_ref", joint_vel_init_);
  kp_gains_ = auto_declare<std::vector<double>>("kp_gains", kp_gains_);
  kd_gains_ = auto_declare<std::vector<double>>("kd_gains", kd_gains_);
  action_scales_ = auto_declare<std::vector<double>>("action_scales", action_scales_);
  residual_action_scales_ = auto_declare<std::vector<double>>("residual_action_scales", residual_action_scales_);
  for (size_t i = 0; i < residual_action_scales_.size() && i < action_scales_.size(); ++i)
  {
    residual_action_scales_[i] *= action_scales_[i];
  }

  // init observation manager and action manager
  history_length_ = auto_declare<int>("history_length", history_length_);
  if (policy_type_ == "residual")
  {
    residual_history_length_ = auto_declare<int>("residual_history_length", residual_history_length_);
  }
  initialize_observation_managers();
  initialize_action_managers();
  actions_.resize(action_manager_.total_action_dim(), 0.0);

  // init policies
  observations_.resize(observation_manager_.total_observation_dim(), 0.0);
  std::string package_name = auto_declare<std::string>("package_name", "tron1_deploy");
  if (policy_type_ == "base")
  {
    std::string policy_path = auto_declare<std::string>("policy_path", "");
    const std::string resolved_policy_path = file_utils::resolve_file_path(policy_path, package_name);

    base_controllers::OnnxMLPCfg policy_cfg;
    policy_cfg.model_path = resolved_policy_path;
    policy_cfg.input_size = static_cast<int>(observations_.size());
    policy_cfg.output_size = action_manager_.total_action_dim();
    policy_cfg.validate_model_io = false;
    base_policy_ptr_ = std::make_shared<base_controllers::OnnxMLP>(policy_cfg);
  }
  else if (policy_type_ == "rnn")
  {
    std::string policy_path = auto_declare<std::string>("policy_path", "");
    const std::string resolved_policy_path = file_utils::resolve_file_path(policy_path, package_name);
    int hidden_dim = auto_declare<int>("hidden_dim", 256);
    int hidden_layers = auto_declare<int>("hidden_layers", 2);

    base_controllers::OnnxRNNPolicyCfg policy_cfg;
    policy_cfg.model_path = resolved_policy_path;
    policy_cfg.input_size = static_cast<int>(observations_.size());
    policy_cfg.output_size = action_manager_.total_action_dim();
    policy_cfg.hidden_dim = hidden_dim;
    policy_cfg.hidden_layers = hidden_layers;
    policy_cfg.validate_model_io = false;
    rnn_policy_ptr_ = std::make_shared<base_controllers::OnnxRNN>(policy_cfg);
  }
  else if (policy_type_ == "residual")
  {
    residual_actions_.resize(residual_action_manager_.total_action_dim(), 0.0);
    residual_observations_.resize(residual_observation_manager_.total_observation_dim(), 0.0);

    std::string base_policy_path = auto_declare<std::string>("base_policy_path", "");
    std::string residual_policy_path = auto_declare<std::string>("residual_policy_path", "");
    const std::string resolved_base_policy_path = file_utils::resolve_file_path(base_policy_path, package_name);
    const std::string resolved_residual_policy_path = file_utils::resolve_file_path(residual_policy_path, package_name);
    int hidden_dim = auto_declare<int>("hidden_dim", 256);
    int hidden_layers = auto_declare<int>("hidden_layers", 2);

    base_controllers::OnnxMLPCfg base_policy_cfg;
    base_policy_cfg.model_path = resolved_base_policy_path;
    base_policy_cfg.input_size = static_cast<int>(observations_.size());
    base_policy_cfg.output_size = action_manager_.total_action_dim();
    base_policy_cfg.validate_model_io = false;
    base_policy_ptr_ = std::make_shared<base_controllers::OnnxMLP>(base_policy_cfg);

    base_controllers::OnnxRNNPolicyCfg residual_policy_cfg;
    residual_policy_cfg.model_path = resolved_residual_policy_path;
    residual_policy_cfg.input_size = static_cast<int>(residual_observations_.size());
    residual_policy_cfg.output_size = residual_action_manager_.total_action_dim();
    residual_policy_cfg.hidden_dim = hidden_dim;
    residual_policy_cfg.hidden_layers = hidden_layers;
    residual_policy_cfg.validate_model_io = false;
    residual_policy_ptr_ = std::make_shared<base_controllers::OnnxRNN>(residual_policy_cfg);
  }

  // init steering related
  if (steer_type_ == "closed" || steer_type_ == "none"){} // do nothing
  else if (steer_type_ == "open_full" || steer_type_ == "open_ref")
  {
    // TODO: when needed fit in the logic of steering
  }
  else
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1Velocity::on_init() failed because steer_type '%s' is invalid. Supported types are 'none', 'closed', 'open_full', and 'open_ref'.",
                 steer_type_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  // init pinocchio related
  std::string description_package_name = auto_declare<std::string>("description_package_name", "tron1_description");
  std::string urdf_path = auto_declare<std::string>("urdf_path", "");
  std::string resolved_urdf_path = file_utils::resolve_file_path(urdf_path, description_package_name);
  pinocchio::urdf::buildModel(resolved_urdf_path, pinocchio::JointModelFreeFlyer(), model_);
  data_ = pinocchio::Data(model_);
  perm_ = pinocchio_utils::build_joint_reorder_map(joint_names_, model_);
  interface_name_ = auto_declare<std::string>("waypoint_name", interface_name_);
  base_frame_name_ = auto_declare<std::string>("base_frame_name", base_frame_name_);
  ee_frame_name_ = auto_declare<std::string>("ee_frame_name", ee_frame_name_);
  if (!model_.existFrame(base_frame_name_))
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1LocoManipulation::on_init() failed because base frame '%s' does not exist in the Pinocchio model.",
                 base_frame_name_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  if (!model_.existFrame(ee_frame_name_))
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1LocoManipulation::on_init() failed because end-effector frame '%s' does not exist in the Pinocchio model.",
                 ee_frame_name_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  base_frame_id_ = model_.getFrameId(base_frame_name_);
  ee_frame_id_ = model_.getFrameId(ee_frame_name_);

  // debug
  if (debug_)
  {
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pos_x", &ee_pos_debug_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pos_y", &ee_pos_debug_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pos_z", &ee_pos_debug_[2]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_ori_w", &ee_ori_debug_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_ori_x", &ee_ori_debug_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_ori_y", &ee_ori_debug_[2]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_ori_z", &ee_ori_debug_[3]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pos_ref_x", &ee_pos_ref_debug_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pos_ref_y", &ee_pos_ref_debug_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pos_ref_z", &ee_pos_ref_debug_[2]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_ori_ref_w", &ee_ori_ref_debug_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_ori_ref_x", &ee_ori_ref_debug_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_ori_ref_y", &ee_ori_ref_debug_[2]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_ori_ref_z", &ee_ori_ref_debug_[3]);
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration Tron1LocoManipulation::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config =
    humanoid_controllers::BaseHumanoidController::get_joint_command_interface_configuration();

  return config;
}

controller_interface::InterfaceConfiguration Tron1LocoManipulation::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration state_interface_config;
  state_interface_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // state estimation
  if (estimator_name_ == "")
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1VelocityPolicyRNN::state_interface_configuration() failed because estimator_name is not set.");
  }
  else
  {
    // joint interfaces
    auto joint_state_interface_config =
      humanoid_controllers::BaseHumanoidController::get_joint_state_interface_configuration();
    state_interface_config.names.insert(state_interface_config.names.end(),
                                      joint_state_interface_config.names.begin(),
                                      joint_state_interface_config.names.end());

    // position
    auto global_pos_interface_config =
      humanoid_controllers::BaseHumanoidController::get_global_pos_interface_configuration();
    state_interface_config.names.insert(state_interface_config.names.end(),
                                      global_pos_interface_config.names.begin(),
                                      global_pos_interface_config.names.end());

    // orientation
    auto ori_interface_config =
      humanoid_controllers::BaseHumanoidController::get_ori_interface_configuration();
    state_interface_config.names.insert(state_interface_config.names.end(),
                                      ori_interface_config.names.begin(),
                                      ori_interface_config.names.end());

    // linear velocity
    auto global_lin_vel_interface_config =
      humanoid_controllers::BaseHumanoidController::get_global_lin_vel_interface_configuration();
    state_interface_config.names.insert(state_interface_config.names.end(),
                                      global_lin_vel_interface_config.names.begin(),
                                      global_lin_vel_interface_config.names.end());

    // angular velocity
    auto global_ang_vel_interface_config =
      humanoid_controllers::BaseHumanoidController::get_global_ang_vel_interface_configuration();
    state_interface_config.names.insert(state_interface_config.names.end(),
                                      global_ang_vel_interface_config.names.begin(),
                                      global_ang_vel_interface_config.names.end());
  }

  // planner
  if (planner_name_ == "")
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1VelocityPolicy::state_interface_configuration() failed because planner_name is not set.");
  }
  else
  {
    // velocity target
    state_interface_config.names.push_back(planner_name_ + "/lin_x_vel_ref");
    state_interface_config.names.push_back(planner_name_ + "/lin_y_vel_ref");
    state_interface_config.names.push_back(planner_name_ + "/yaw_rate_ref");
    // waypoint target
    auto waypoint_interface_config =
      base_planners::Waypoint::get_state_interface_configuration(planner_name_, {interface_name_});
    state_interface_config.names.insert(state_interface_config.names.end(),
                                      waypoint_interface_config.names.begin(),
                                      waypoint_interface_config.names.end());
  }

  return state_interface_config;
}

controller_interface::CallbackReturn Tron1LocoManipulation::on_configure(const rclcpp_lifecycle::State& previous_state)
{
  controller_interface::CallbackReturn state = BaseHumanoidController::on_configure(previous_state);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1LocoManipulation::on_activate(const rclcpp_lifecycle::State& previous_state)
{
  auto ret = humanoid_controllers::BaseHumanoidController::on_activate(previous_state);

  if (ret != controller_interface::CallbackReturn::SUCCESS)
  {
    return ret;
  }

  // initialize actions
  (void)previous_state;
  read_joint_states_from_state_interfaces(joint_pos_);

  if (policy_type_ == "rnn" && rnn_policy_ptr_ != nullptr)
  {
    rnn_policy_ptr_->reset();

  }
  else if (policy_type_ == "residual" && residual_policy_ptr_ != nullptr)
  {
    residual_policy_ptr_->reset();
    residual_actions_.resize(residual_action_manager_.total_action_dim(), 0.0);
    residual_action_manager_.process_action(residual_actions_);
  }
  actions_.resize(action_manager_.total_action_dim(), 0.0);
  for (size_t i = 0; i < actions_.size(); ++i)
  {
    actions_[i] = (joint_pos_[i] - joint_pos_init_[i]) / action_scales_[i];
  }
  action_manager_.process_action(actions_);
  joint_pos_des_.assign(joint_names_.size(), 0.0);
  const auto processed_actions = action_manager_.processed_actions();
  for (size_t i = 0; i < processed_actions.size() && i < joint_pos_des_.size(); ++i)
  {
    joint_pos_des_[i] = processed_actions[i];
  }

  return controller_interface::CallbackReturn::SUCCESS;


  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1LocoManipulation::on_deactivate(const rclcpp_lifecycle::State& previous_state)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type Tron1LocoManipulation::update_and_write_commands(const rclcpp::Time& time,
                                                                                      const rclcpp::Duration& period)
{
  (void)time;
  (void)period;

  const size_t joint_dim = joint_names_.size();

  // 1. read estimator state
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
    ori_.coeffs() *= -1.0; // ensure the scalar part of the quaternion is non-negative
  }

  std::array<double, 3> lin_vel_global{0.0, 0.0, 0.0};
  std::array<double, 3> ang_vel_global{0.0, 0.0, 0.0};
  read_global_lin_vel_from_state_interfaces(lin_vel_global);
  read_global_ang_vel_from_state_interfaces(ang_vel_global);

  base_lin_vel_ = ori_.inverse() * Eigen::Vector3d(
      lin_vel_global[0], lin_vel_global[1], lin_vel_global[2]);
  base_ang_vel_ = ori_.inverse() * Eigen::Vector3d(
      ang_vel_global[0], ang_vel_global[1], ang_vel_global[2]);

  read_joint_states_from_state_interfaces(joint_pos_, joint_vel_);

  // 2. update ee pose using pinocchio and waypoint desired from interfaces
  std::array<double, 3> base_pos{};
  BaseHumanoidController::read_global_pos_from_state_interfaces(base_pos);
  Eigen::VectorXd q = pinocchio::neutral(model_);
  q.segment<3>(0) << base_pos[0], base_pos[1], base_pos[2];
  q.segment<4>(3) << ori_.x(), ori_.y(), ori_.z(), ori_.w();
  for (size_t i = 0; i < joint_pos_.size(); ++i)
  {
    q(7 + perm_[i]) = joint_pos_[i];
  }
  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);
  const pinocchio::SE3& base_frame = data_.oMf[base_frame_id_];
  const pinocchio::SE3& ee_frame = data_.oMf[ee_frame_id_];
  const pinocchio::SE3 base_to_ee = base_frame.actInv(ee_frame);
  ee_pos_ = base_to_ee.translation();
  ee_ori_ = Eigen::Quaterniond(base_to_ee.rotation());
  ee_ori_.normalize();

  base_planners::Waypoint waypoint;
  waypoint.name = interface_name_;
  base_planners::Waypoint::read_state_interfaces(planner_name_, waypoint, state_interfaces_);
  const Eigen::Vector3d waypoint_pos_b = waypoint.position;
  const Eigen::Quaterniond waypoint_ori_b = waypoint.orientation.normalized();

  ee_pose_ref_ = {
      waypoint_pos_b.x(),
      waypoint_pos_b.y(),
      waypoint_pos_b.z(),
      waypoint_ori_b.w(),
      waypoint_ori_b.x(),
      waypoint_ori_b.y(),
      waypoint_ori_b.z(),
  };
  ee_pos_debug_ = {ee_pos_.x(), ee_pos_.y(), ee_pos_.z()};
  ee_ori_debug_ = {ee_ori_.w(), ee_ori_.x(), ee_ori_.y(), ee_ori_.z()};
  ee_pos_ref_debug_ = {waypoint_pos_b.x(), waypoint_pos_b.y(), waypoint_pos_b.z()};
  ee_ori_ref_debug_ = {waypoint_ori_b.w(), waypoint_ori_b.x(), waypoint_ori_b.y(), waypoint_ori_b.z()};

  base_utils::get_state_interface_values(
    state_interfaces_,
    {
        planner_name_ + "/lin_x_vel_ref",
        planner_name_ + "/lin_y_vel_ref",
        planner_name_ + "/yaw_rate_ref",
    },
    vel_refs_);

  // 3. resolve policy mode
  if (policy_type_ == "base")
  {
    observations_ = observation_manager_.compute();
    actions_ = base_policy_ptr_->infer(observations_);

    action_manager_.process_action(actions_);
    joint_pos_des_.assign(joint_names_.size(), 0.0);
    const auto processed_actions = action_manager_.processed_actions();
    for (size_t i = 0; i < processed_actions.size() && i < joint_pos_des_.size(); ++i)
    {
      joint_pos_des_[i] = processed_actions[i];
      if(i == joint_dim-2)
      {
        joint_pos_des_[i] = 0.0;
      }
    }
  }
  else if (policy_type_ == "rnn")
  {
    observations_ = observation_manager_.compute();
    actions_ = rnn_policy_ptr_->infer(observations_);

    action_manager_.process_action(actions_);
    joint_pos_des_.assign(joint_names_.size(), 0.0);
    const auto processed_actions = action_manager_.processed_actions();
    for (size_t i = 0; i < processed_actions.size() && i < joint_pos_des_.size(); ++i)
    {
      joint_pos_des_[i] = processed_actions[i];
      if(i == joint_dim-2)
      {
        joint_pos_des_[i] = 0.0;
      }
    }
  }
  else if (policy_type_ == "residual")
  {
    observations_ = observation_manager_.compute();
    residual_observations_ = residual_observation_manager_.compute();

    actions_ = base_policy_ptr_->infer(observations_);
    residual_actions_ = residual_policy_ptr_->infer(residual_observations_);

    action_manager_.process_action(actions_);
    residual_action_manager_.process_action(residual_actions_);

    const auto base_cmd = action_manager_.processed_actions();
    const auto residual_cmd = residual_action_manager_.processed_actions();

    joint_pos_des_.assign(joint_names_.size(), 0.0);
    for (size_t i = 0; i < joint_dim && i < base_cmd.size() && i < residual_cmd.size(); ++i)
    {
      joint_pos_des_[i] = base_cmd[i] + residual_cmd[i];
      if(i == joint_dim-2)
      {
        joint_pos_des_[i] = 0.0;
      }
    }
  }

  // 4. steer
  if (steer_type_ == "none" || steer_type_ == "closed"){}
  else if (steer_type_ == "open_full")
  {}

  // 5. write commands
  std::vector<double> joint_vel_des(joint_dim, 0.0);
  std::vector<double> joint_tau_des(joint_dim, 0.0);
  Eigen::VectorXd v = Eigen::VectorXd::Zero(model_.nv);
  v.segment<3>(0) << base_lin_vel_[0], base_lin_vel_[1], base_lin_vel_[2];
  v.segment<3>(3) << base_ang_vel_[0], base_ang_vel_[1], base_ang_vel_[2];
  for (size_t i = 0; i < joint_vel_.size(); ++i)
  {
    v(6 + perm_[i]) = joint_vel_[i];
  }
  const Eigen::VectorXd bias_torque = pinocchio::nonLinearEffects(model_, data_, q, v);
  const Eigen::VectorXd joint_bias_torque =
      pinocchio_utils::reorder_pinocchio_to_joint(bias_torque.tail(static_cast<Eigen::Index>(perm_.size())), perm_);
  for (size_t i = 8; i < joint_dim - 1 && i < static_cast<size_t>(joint_bias_torque.size()); ++i)
  {
    joint_tau_des[i] = joint_bias_torque[static_cast<Eigen::Index>(i)];
    if(i == joint_dim-2)
    {
      joint_tau_des[i] = 0.0;
    }
  }

  write_joint_commands_to_command_interfaces(
      joint_pos_des_,
      joint_vel_des,
      joint_tau_des,
      kp_gains_,
      kd_gains_);

  return controller_interface::return_type::OK;
}

std::vector<hardware_interface::CommandInterface> Tron1LocoManipulation::on_export_reference_interfaces()
{
  std::vector<hardware_interface::CommandInterface> reference_interfaces;
  std::string controller_name = this->get_node()->get_name();
  reference_interfaces_.resize(1, 0.0);
  // lin_x_vel, lin_y_vel, ang_z_vel
  reference_interfaces.emplace_back(controller_name, "dummy", &vel_refs_[0]);

  return reference_interfaces;
}

void Tron1LocoManipulation::initialize_observation_managers()
{
  // TODO: whether add open full and open ref observation managers?
  observation_manager_ = observation::ObservationManager();
  residual_observation_manager_ = observation::ObservationManager();
  // open_full_observation_manager_ = observation::ObservationManager();
  // open_ref_observation_manager_ = observation::ObservationManager();

  const int joint_dim = static_cast<int>(joint_names_.size());
  const int action_dim = joint_dim - 1;

  auto add_loco_manipulation_policy_terms =
    [this, joint_dim, action_dim](observation::ObservationManager& manager,
                        int history_length,
                        std::vector<double>& last_actions)
  {
    std::vector<double>* last_actions_ptr = &last_actions;

    manager.add_term(
        std::make_unique<observation::ObservationTerm>(
            observation::ObservationTermCfg{
                .name = "base_lin_vel",
                .observation_dim = 3,
                .history_length = history_length,
            },
            [this]() -> std::vector<double>
            {
              return observation::passthrough(base_lin_vel_);
            }));

    manager.add_term(
        std::make_unique<observation::ObservationTerm>(
            observation::ObservationTermCfg{
                .name = "base_ang_vel",
                .observation_dim = 3,
                .history_length = history_length,
            },
            [this]() -> std::vector<double>
            {
              return observation::passthrough(base_ang_vel_);
            }));

    manager.add_term(
        std::make_unique<observation::ObservationTerm>(
            observation::ObservationTermCfg{
                .name = "base_orientation",
                .observation_dim = 4,
                .history_length = history_length,
            },
            [this]() -> std::vector<double>
            {
              return {ori_.w(), ori_.x(), ori_.y(), ori_.z()};
            }));

    manager.add_term(
        std::make_unique<observation::ObservationTerm>(
            observation::ObservationTermCfg{
                .name = "ee_position",
                .observation_dim = 3,
                .history_length = history_length,
            },
            [this]() -> std::vector<double>
            {
              return observation::passthrough(ee_pos_);
            }));

    manager.add_term(
        std::make_unique<observation::ObservationTerm>(
            observation::ObservationTermCfg{
                .name = "ee_orientation",
                .observation_dim = 4,
                .history_length = history_length,
            },
            [this]() -> std::vector<double>
            {
              return {ee_ori_.w(), ee_ori_.x(), ee_ori_.y(), ee_ori_.z()};
            }));
    manager.add_term(
        std::make_unique<observation::ObservationTerm>(
            observation::ObservationTermCfg{
                .name = "joint_pos",
                .observation_dim = joint_dim,
                .history_length = history_length,
            },
            [this]() -> std::vector<double>
            {
              return observation::joint_values(joint_pos_, joint_pos_init_);
            }));

    manager.add_term(
        std::make_unique<observation::ObservationTerm>(
            observation::ObservationTermCfg{
                .name = "joint_vel",
                .observation_dim = joint_dim,
                .history_length = history_length,
            },
            [this]() -> std::vector<double>
            {
              return observation::joint_values(joint_vel_, joint_vel_init_);
            }));

    manager.add_term(
        std::make_unique<observation::ObservationTerm>(
            observation::ObservationTermCfg{
                .name = "last_action",
                .observation_dim = action_dim,
                .history_length = history_length,
            },
            [last_actions_ptr]() -> std::vector<double>
            {
              return observation::passthrough(*last_actions_ptr);
            }));

    manager.add_term(
        std::make_unique<observation::ObservationTerm>(
            observation::ObservationTermCfg{
                .name = "ee_command",
                .observation_dim = 7,
                .history_length = history_length,
            },
            [this]() -> std::vector<double>
            {
              return observation::passthrough(ee_pose_ref_);
            }));

    manager.add_term(
        std::make_unique<observation::ObservationTerm>(
            observation::ObservationTermCfg{
                .name = "vel_command",
                .observation_dim = 3,
                .history_length = history_length,
            },
            [this]() -> std::vector<double>
            {
              return observation::passthrough(vel_refs_);
            }));
  };

  add_loco_manipulation_policy_terms(observation_manager_, history_length_, actions_);

  if(policy_type_ == "residual")
  {
    add_loco_manipulation_policy_terms(residual_observation_manager_, residual_history_length_, residual_actions_);
  }
  if (steer_type_ == "open_full")
  {
    throw std::runtime_error("open_full steer type is not implemented yet.");
    // add_loco_manipulation_policy_terms(open_full_observation_manager_, 1, actions_);
  }
  else if (steer_type_ == "open_ref")
  {
    throw std::runtime_error("open_ref steer type is not implemented yet.");
    // open_ref_observation_manager_.add_term(
    //   std::make_unique<observation::ObservationTerm>(
    //       observation::ObservationTermCfg{
    //           .name = "ee_position",
    //           .observation_dim = 3,
    //           .history_length = 1,
    //       },
    //       [this]() -> std::vector<double>
    //       {
    //         return observation::passthrough(ee_pos_);
    //       })
    // );
  }
}

void Tron1LocoManipulation::initialize_action_managers()
{
  action_manager_ = action::ActionManager();
  residual_action_manager_ = action::ActionManager();

  const int leg_joint_dim = 8;
  const int arm_joint_dim = 7;

  action_manager_.add_term(
      std::make_unique<action::JointPositionActionTerm>(
          action::ActionTermCfg{
              .name = "leg_position_action",
              .action_dim = leg_joint_dim,
              .action_scale = std::vector<double>(
                  action_scales_.begin(), action_scales_.begin() + leg_joint_dim),
              .action_offset = std::vector<double>(
                  joint_pos_init_.begin(), joint_pos_init_.begin() + leg_joint_dim),
          }));
  action_manager_.add_term(
      std::make_unique<action::JointImpedanceActionTerm>(
          action::ActionTermCfg{
              .name = "arm_impedance_action",
              .action_dim = arm_joint_dim,
              .action_scale = std::vector<double>(
                  action_scales_.begin() + leg_joint_dim, action_scales_.begin() + leg_joint_dim + arm_joint_dim),
              .action_offset = std::vector<double>(
                  joint_pos_init_.begin() + leg_joint_dim, joint_pos_init_.begin() + leg_joint_dim + arm_joint_dim),
          }));
  if(policy_type_ == "residual")
  {
    residual_action_manager_.add_term(
      std::make_unique<action::JointPositionActionTerm>(
          action::ActionTermCfg{
              .name = "residual_leg_position_action",
              .action_dim = leg_joint_dim,
              .action_scale = std::vector<double>(
                  residual_action_scales_.begin(), residual_action_scales_.begin() + leg_joint_dim),
              .action_offset = std::vector<double>(leg_joint_dim, 0.0),
          }));
    action_manager_.add_term(
      std::make_unique<action::JointImpedanceActionTerm>(
          action::ActionTermCfg{
              .name = "residual_arm_impedance_action",
              .action_dim = arm_joint_dim,
              .action_scale = std::vector<double>(
                  residual_action_scales_.begin() + leg_joint_dim, action_scales_.begin() + leg_joint_dim + arm_joint_dim),
              .action_offset = std::vector<double>(arm_joint_dim, 0.0),
    }));
  }
}

} // namespace tron1_deploy

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(tron1_deploy::Tron1LocoManipulation, controller_interface::ChainableControllerInterface)
