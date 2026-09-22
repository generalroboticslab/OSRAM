#include "tron1_deploy/tron1_velocity.hpp"

namespace tron1_deploy
{

controller_interface::CallbackReturn Tron1Velocity::on_init()
{
  controller_interface::CallbackReturn state = BaseHumanoidController::on_init();
  // check whether reference interfaces conflicted
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
  if((policy_type_ != "base" && policy_type_ != "rnn") && steer_type_ != "none")
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1VelocityPolicy::on_init() failed because steer_type '%s' is not supported when policy_type is not 'base' or 'rnn. steer_type has to be 'none' when policy_type is 'residual'.",
                 steer_type_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  // resize vectors
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
  residual_action_scales_.resize(joint_names_.size(), 0.0);

  // init pd gains and targets from parameters if exist
  joint_pos_init_ = auto_declare<std::vector<double>>("joint_pos_ref", joint_pos_init_);
  joint_vel_init_ = auto_declare<std::vector<double>>("joint_vel_ref", joint_vel_init_);
  kp_gains_ = auto_declare<std::vector<double>>("kp_gains", kp_gains_);
  kd_gains_ = auto_declare<std::vector<double>>("kd_gains", kd_gains_);
  action_scales_ = auto_declare<std::vector<double>>("action_scales", action_scales_);
  residual_action_scales_ = auto_declare<std::vector<double>>("residual_action_scales", residual_action_scales_);
  for (size_t i = 0; i < residual_action_scales_.size(); ++i)
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
    // planning related
    horizon_ = auto_declare<int>("horizon", horizon_);
    dt_ = auto_declare<double>("dt", dt_);
    node_name_ = auto_declare<std::string>("node_name", "");
    subscribe_topic_name_ = auto_declare<std::string>("subscribe_topic_name", "");
    publish_topic_name_ = auto_declare<std::string>("publish_topic_name", "");

    if (node_name_.empty() || subscribe_topic_name_.empty() || publish_topic_name_.empty())
    {
      RCLCPP_ERROR(this->get_node()->get_logger(),
                  "Tron1Velocity: 'node_name', 'publish_topic_name' or 'subscribe_topic_name' parameters are empty.");
      return controller_interface::CallbackReturn::FAILURE;
    }
    node_ptr_ = rclcpp::Node::make_shared(node_name_);
    auto qos = rclcpp::QoS(rclcpp::KeepLast(2), rmw_qos_profile_sensor_data);
    control_subscriber_ = node_ptr_->create_subscription<std_msgs::msg::Float64MultiArray>(
        subscribe_topic_name_, qos,
        [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg)
        {
          for (size_t i = 0; i < actions_.size() && i < msg->data.size(); ++i)
          {
            joint_pos_des_[i] = msg->data[i];
          }
        });
    state_publisher_ = node_ptr_->create_publisher<std_msgs::msg::Float64MultiArray>(publish_topic_name_, qos);
    realtime_state_publisher_ =
        std::make_unique<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(state_publisher_);
    executor_.add_node(node_ptr_);
    // configure the filter
    sampling_frequency_ = auto_declare<std::vector<double>>("sampling_frequency", {50.0});
    damping_frequency_ = auto_declare<std::vector<double>>("damping_frequency", {50.0});
    damping_intensity_ = auto_declare<std::vector<double>>("damping_intensity", {0.0});
    if (sampling_frequency_.size() == 1)
    {
      for (size_t i = 0; i < 3; ++i)
      {
        lp_filters_[i] = std::make_shared<control_toolbox::LowPassFilter<double>>(sampling_frequency_[0], damping_frequency_[0],
                                                                                  damping_intensity_[0]);
        lp_filters_[i]->configure();
      }
    }
    else if (sampling_frequency_.size() == 3)
    {
      for (size_t i = 0; i < 3; ++i)
      {
        lp_filters_[i] = std::make_shared<control_toolbox::LowPassFilter<double>>(sampling_frequency_[i], damping_frequency_[i],
                                                                                  damping_intensity_[i]);
        lp_filters_[i]->configure();
      }
    }
    else
    {
      RCLCPP_ERROR(this->get_node()->get_logger(),
                  "Tron1MPPIPlanner: 'sampling_frequency' parameter should be either a single value or a list of three values.");
      return controller_interface::CallbackReturn::FAILURE;
    }
  }
  else
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1Velocity::on_init() failed because steer_type '%s' is invalid. Supported types are 'none', 'closed', 'open_full', and 'open_ref'.",
                 steer_type_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  // debug
  if (debug_)
  {
    REGISTER_ROS2_CONTROL_INTROSPECTION("lin_vel_x_local", &base_lin_vel_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("lin_vel_y_local", &base_lin_vel_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ang_vel_yaw_local", &base_ang_vel_[2]);
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration Tron1Velocity::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config =
    humanoid_controllers::BaseHumanoidController::get_joint_command_interface_configuration();

  return config;
}

controller_interface::InterfaceConfiguration Tron1Velocity::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // state estimation
  if (estimator_name_ == "")
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1VelocityPolicyRNN::state_interface_configuration() failed because estimator_name is not set.");
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

  // planner
  if (planner_name_ == "")
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1VelocityPolicy::state_interface_configuration() failed because planner_name is not set.");
  }
  else
  {
    config.names.push_back(planner_name_ + "/lin_x_vel_ref");
    config.names.push_back(planner_name_ + "/lin_y_vel_ref");
    config.names.push_back(planner_name_ + "/yaw_rate_ref");
  }

  return config;
}

controller_interface::CallbackReturn Tron1Velocity::on_configure(const rclcpp_lifecycle::State& previous_state)
{
  controller_interface::CallbackReturn state = BaseHumanoidController::on_configure(previous_state);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1Velocity::on_activate(const rclcpp_lifecycle::State& previous_state)
{
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
  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    actions_[i] = (joint_pos_[i] - joint_pos_init_[i]) / action_scales_[i];
  }
  action_manager_.process_action(actions_);
  joint_pos_des_ = action_manager_.processed_actions();
  prev_actions_ = action_manager_.raw_actions();

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1Velocity::on_deactivate(const rclcpp_lifecycle::State& previous_state)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type Tron1Velocity::update_and_write_commands(const rclcpp::Time& time,
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

  std::array<double, 3> lin_vel_global{0.0, 0.0, 0.0};
  std::array<double, 3> ang_vel_global{0.0, 0.0, 0.0};
  read_global_lin_vel_from_state_interfaces(lin_vel_global);
  read_global_ang_vel_from_state_interfaces(ang_vel_global);

  base_lin_vel_ = ori_.inverse() * Eigen::Vector3d(
      lin_vel_global[0], lin_vel_global[1], lin_vel_global[2]);
  base_ang_vel_ = ori_.inverse() * Eigen::Vector3d(
      ang_vel_global[0], ang_vel_global[1], ang_vel_global[2]);

  read_joint_states_from_state_interfaces(joint_pos_, joint_vel_);

  // 2. read velocity references
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
    joint_pos_des_ = action_manager_.processed_actions();
    prev_actions_ = action_manager_.raw_actions();
  }
  else if (policy_type_ == "rnn")
  {
    observations_ = observation_manager_.compute();
    actions_ = rnn_policy_ptr_->infer(observations_);

    action_manager_.process_action(actions_);
    joint_pos_des_ = action_manager_.processed_actions();
    prev_actions_ = action_manager_.raw_actions();
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

    for (size_t i = 0; i < joint_dim; ++i)
    {
      joint_pos_des_[i] = base_cmd[i] + residual_cmd[i];
    }

    prev_actions_ = action_manager_.raw_actions();
  }

  // 4. steer
  if (steer_type_ == "none" || steer_type_ == "closed"){}
  else if (steer_type_ == "open_full")
  {
    // filter the observations
    double filtered_lin_x = base_lin_vel_[0];
    double filtered_lin_y = base_lin_vel_[1];
    double filtered_yaw_rate = base_ang_vel_[2];
    if (lp_filters_[0] != nullptr && lp_filters_[0]->update(base_lin_vel_[0], filtered_lin_x))
    {
      base_lin_vel_[0] = filtered_lin_x;
    }
    if (lp_filters_[1] != nullptr && lp_filters_[1]->update(base_lin_vel_[1], filtered_lin_y))
    {
      base_lin_vel_[1] = filtered_lin_y;
    }
    if (lp_filters_[2] != nullptr && lp_filters_[2]->update(base_ang_vel_[2], filtered_yaw_rate))
    {
      base_ang_vel_[2] = filtered_yaw_rate;
    }
    // construct obs
    std::vector<double> obs_pub = open_full_observation_manager_.compute();

    // get reference from planner
    std::vector<double> lin_vel_x_des_vec_(horizon_, 0.0);
    std::vector<double> lin_vel_y_des_vec_(horizon_, 0.0);
    std::vector<double> ang_vel_z_des_vec_(horizon_, 0.0);
    for (int i = 0; i < horizon_; i++)
    {
      lin_vel_x_des_vec_[i] = vel_refs_[0];
      lin_vel_y_des_vec_[i] = vel_refs_[1];
      ang_vel_z_des_vec_[i] = vel_refs_[2];
    }

    // publish obs
    if (realtime_state_publisher_ != nullptr)
    {
      auto msg = std_msgs::msg::Float64MultiArray();
      msg.data = obs_pub;
      msg.data.reserve(obs_pub.size() + 3 * horizon_ + actions_.size());
      for (int i = 0; i < horizon_; ++i)
      {
        msg.data.push_back(lin_vel_x_des_vec_[i]);
        msg.data.push_back(lin_vel_y_des_vec_[i]);
        msg.data.push_back(ang_vel_z_des_vec_[i]);
      }
      for (const auto& action : joint_pos_des_)
      {
        msg.data.push_back(action);
      }

      realtime_state_publisher_->lock();
      realtime_state_publisher_->msg_ = msg;
      realtime_state_publisher_->unlockAndPublish();
    }

    // overwrite joint position commands
    executor_.spin_some();
  }
  else if (steer_type_ == "open_ref")
  {
    // filter the observations
    double filtered_lin_x = base_lin_vel_[0];
    double filtered_lin_y = base_lin_vel_[1];
    double filtered_yaw_rate = base_ang_vel_[2];
    if (lp_filters_[0] != nullptr && lp_filters_[0]->update(base_lin_vel_[0], filtered_lin_x))
    {
      base_lin_vel_[0] = filtered_lin_x;
    }
    if (lp_filters_[1] != nullptr && lp_filters_[1]->update(base_lin_vel_[1], filtered_lin_y))
    {
      base_lin_vel_[1] = filtered_lin_y;
    }
    if (lp_filters_[2] != nullptr && lp_filters_[2]->update(base_ang_vel_[2], filtered_yaw_rate))
    {
      base_ang_vel_[2] = filtered_yaw_rate;
    }
    // construct obs
    std::vector<double> obs_pub = open_ref_observation_manager_.compute();

    // get reference from planner
    std::vector<double> lin_vel_x_des_vec_(horizon_, 0.0);
    std::vector<double> lin_vel_y_des_vec_(horizon_, 0.0);
    std::vector<double> ang_vel_z_des_vec_(horizon_, 0.0);
    for (int i = 0; i < horizon_; i++)
    {
      lin_vel_x_des_vec_[i] = vel_refs_[0];
      lin_vel_y_des_vec_[i] = vel_refs_[1];
      ang_vel_z_des_vec_[i] = vel_refs_[2];
    }

    // publish obs
    if (realtime_state_publisher_ != nullptr)
    {
      auto msg = std_msgs::msg::Float64MultiArray();
      msg.data = obs_pub;
      msg.data.reserve(obs_pub.size() + 3 * horizon_ + actions_.size());
      for (int i = 0; i < horizon_; ++i)
      {
        msg.data.push_back(lin_vel_x_des_vec_[i]);
        msg.data.push_back(lin_vel_y_des_vec_[i]);
        msg.data.push_back(ang_vel_z_des_vec_[i]);
      }
      for (const auto& action : joint_pos_des_)
      {
        msg.data.push_back(action);
      }

      realtime_state_publisher_->lock();
      realtime_state_publisher_->msg_ = msg;
      realtime_state_publisher_->unlockAndPublish();
    }

    // overwrite joint position commands
    executor_.spin_some();
  }

  // 5. write commands
  std::vector<double> joint_vel_des(joint_dim, 0.0);
  std::vector<double> joint_tau_des(joint_dim, 0.0);

  write_joint_commands_to_command_interfaces(
      joint_pos_des_,
      joint_vel_des,
      joint_tau_des,
      kp_gains_,
      kd_gains_);

  return controller_interface::return_type::OK;
}

std::vector<hardware_interface::CommandInterface> Tron1Velocity::on_export_reference_interfaces()
{
  std::vector<hardware_interface::CommandInterface> reference_interfaces;
  std::string controller_name = this->get_node()->get_name();
  reference_interfaces_.resize(1, 0.0);
  // lin_x_vel, lin_y_vel, ang_z_vel
  reference_interfaces.emplace_back(controller_name, "dummy", &vel_refs_[0]);

  return reference_interfaces;
}

void Tron1Velocity::initialize_observation_managers()
{
  observation_manager_ = observation::ObservationManager();
  residual_observation_manager_ = observation::ObservationManager();
  open_full_observation_manager_ = observation::ObservationManager();
  open_ref_observation_manager_ = observation::ObservationManager();

  const int joint_dim = static_cast<int>(joint_names_.size());

  auto add_velocity_policy_terms =
      [this, joint_dim](observation::ObservationManager& manager,
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
                .name = "projected_gravity",
                .observation_dim = 3,
                .history_length = history_length,
            },
            [this]() -> std::vector<double>
            {
              return observation::projected_gravity(ori_);
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
                .observation_dim = joint_dim,
                .history_length = history_length,
            },
            [last_actions_ptr]() -> std::vector<double>
            {
              return observation::passthrough(*last_actions_ptr);
            }));

    manager.add_term(
        std::make_unique<observation::ObservationTerm>(
            observation::ObservationTermCfg{
                .name = "command",
                .observation_dim = 3,
                .history_length = history_length,
            },
            [this]() -> std::vector<double>
            {
              return observation::passthrough(vel_refs_);
            }));
  };

  add_velocity_policy_terms(observation_manager_, history_length_, actions_);

  if (policy_type_ == "residual")
  {
    add_velocity_policy_terms(residual_observation_manager_, residual_history_length_, residual_actions_);
  }
  if (steer_type_ == "open_full")
  {
    add_velocity_policy_terms(open_full_observation_manager_, 1, actions_);
  }
  else if (steer_type_ == "open_ref")
  {
     open_ref_observation_manager_.add_term(
        std::make_unique<observation::ObservationTerm>(
            observation::ObservationTermCfg{
                .name = "lin_vel_real",
                .observation_dim = 3,
                .history_length = 1,
            },
            [this]() -> std::vector<double>
            {
              return observation::passthrough(std::vector<double>{
                  base_lin_vel_[0],
                  base_lin_vel_[1],
                  base_ang_vel_[2],
              });
            }));
  }
}

void Tron1Velocity::initialize_action_managers()
{
  action_manager_ = action::ActionManager();
  residual_action_manager_ = action::ActionManager();

  const int joint_dim = static_cast<int>(joint_names_.size());

  action_manager_.add_term(
      std::make_unique<action::JointPositionActionTerm>(
          action::ActionTermCfg{
              .name = "joint_position_action",
              .action_dim = joint_dim,
              .action_scale = action_scales_,
              .action_offset = joint_pos_init_,
          }));

  if (policy_type_ == "residual")
  {
    residual_action_manager_.add_term(
        std::make_unique<action::JointPositionActionTerm>(
            action::ActionTermCfg{
                .name = "residual_action",
                .action_dim = joint_dim,
                .action_scale = residual_action_scales_,
                .action_offset = std::vector<double>(joint_dim, 0.0),
            }));
  }
}

} // namespace tron1_deploy

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(tron1_deploy::Tron1Velocity, controller_interface::ChainableControllerInterface)
