#include "tron1_deploy/tron1_with_arm_planner.hpp"

#include <pinocchio/spatial/explog.hpp>
#include <stdexcept>

namespace tron1_deploy
{

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kTrajectoryTolerance = 1.0e-9;

double yaw_from_rotation(const Eigen::Matrix3d& rotation)
{
  return std::atan2(rotation(1, 0), rotation(0, 0));
}

double wrap_to_pi(double angle)
{
  return std::remainder(angle, 2.0 * kPi);
}

class GlobalSineLinePoseTrajectory final : public base_planners::PoseTrajectory
{
  public:
  struct Params
  {
    Eigen::Vector3d start{Eigen::Vector3d::Zero()};
    Eigen::Vector3d target{Eigen::Vector3d::UnitX()};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
    double duration{1.0};
    double amplitude{0.0};
    double wavelength{1.0};
    double phase{0.0};
    bool repeat{false};
  };

  explicit GlobalSineLinePoseTrajectory(const Params& params)
  {
    if (params.duration <= 0.0)
    {
      throw std::invalid_argument("Sine-line duration must be positive.");
    }
    if (params.wavelength <= kTrajectoryTolerance)
    {
      throw std::invalid_argument("Sine-line wavelength must be positive.");
    }
    if (params.orientation.norm() <= kTrajectoryTolerance)
    {
      throw std::invalid_argument("Sine-line orientation quaternion is invalid.");
    }
    if (params.target.x() <= params.start.x() + kTrajectoryTolerance)
    {
      throw std::invalid_argument("Sine-line target x must be greater than start x.");
    }
    if (std::abs(params.target.y() - params.start.y()) > kTrajectoryTolerance ||
        std::abs(params.target.z() - params.start.z()) > kTrajectoryTolerance)
    {
      throw std::invalid_argument("Sine-line start and target must have equal world y and z values.");
    }

    params_ = params;
    params_.orientation.normalize();
    if (params_.orientation.w() < 0.0)
    {
      params_.orientation.coeffs() *= -1.0;
    }
    forward_speed_ = (params_.target.x() - params_.start.x()) / params_.duration;
  }

  void reset(double time = 0.0) override
  {
    start_time_ = time;
  }

  base_planners::Waypoint sample(double time) const override
  {
    const double raw_elapsed_time = time - start_time_;
    double forward_distance = 0.0;
    double forward_velocity = 0.0;
    double forward_acceleration = 0.0;
    if (params_.repeat)
    {
      const double elapsed_time = std::max(0.0, raw_elapsed_time);
      const double cycle_time = std::fmod(elapsed_time, 2.0 * params_.duration);
      const double temporal_frequency = kPi / params_.duration;
      const double cycle_phase = temporal_frequency * cycle_time;
      const double line_length = params_.target.x() - params_.start.x();
      forward_distance = 0.5 * line_length * (1.0 - std::cos(cycle_phase));
      forward_velocity = 0.5 * line_length * temporal_frequency * std::sin(cycle_phase);
      forward_acceleration =
          0.5 * line_length * temporal_frequency * temporal_frequency * std::cos(cycle_phase);
    }
    else
    {
      const double elapsed_time = std::clamp(raw_elapsed_time, 0.0, params_.duration);
      const bool moving = raw_elapsed_time >= 0.0 && raw_elapsed_time < params_.duration;
      forward_distance = forward_speed_ * elapsed_time;
      forward_velocity = moving ? forward_speed_ : 0.0;
    }

    const double spatial_frequency = 2.0 * kPi / params_.wavelength;
    const double theta = params_.phase + spatial_frequency * forward_distance;
    const double theta_velocity = spatial_frequency * forward_velocity;
    const double theta_acceleration = spatial_frequency * forward_acceleration;

    base_planners::Waypoint waypoint;
    waypoint.position = Eigen::Vector3d(params_.start.x() + forward_distance,
                                        params_.start.y() + params_.amplitude * std::sin(theta), params_.start.z());
    waypoint.orientation = params_.orientation;
    waypoint.linear_velocity =
        Eigen::Vector3d(forward_velocity, params_.amplitude * std::cos(theta) * theta_velocity, 0.0);
    waypoint.linear_acceleration = Eigen::Vector3d(
        forward_acceleration,
        params_.amplitude *
            (std::cos(theta) * theta_acceleration - std::sin(theta) * theta_velocity * theta_velocity),
        0.0);
    waypoint.angular_velocity.setZero();
    waypoint.angular_acceleration.setZero();
    return waypoint;
  }

  private:
  Params params_{};
  double start_time_{0.0};
  double forward_speed_{0.0};
};

}  // namespace

controller_interface::CallbackReturn Tron1WithArmPlanner::on_init()
{
  auto ret = humanoid_planners::BaseHumanoidPlanner::on_init();
  if (ret != controller_interface::CallbackReturn::SUCCESS)
  {
    return ret;
  }

  std::string package_name = auto_declare<std::string>("package_name", "tron1_description");
  std::string urdf_path = auto_declare<std::string>("urdf_path", "");
  std::string resolved_urdf_path = file_utils::resolve_file_path(urdf_path, package_name);
  pinocchio::urdf::buildModel(resolved_urdf_path, pinocchio::JointModelFreeFlyer(), model_);
  data_ = pinocchio::Data(model_);
  q_ = pinocchio::neutral(model_);
  v_ = Eigen::VectorXd::Zero(model_.nv);

  waypoint_names_ = auto_declare<std::vector<std::string>>("waypoint_names", {});

  if (waypoint_names_.size() != 1)
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1WaypointPlanner requires exactly 1 waypoint, but got %zu waypoints", waypoint_names_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  perm_ = pinocchio_utils::build_joint_reorder_map(joint_names_, model_);
  base_frame_name_ = auto_declare<std::string>("base_frame_name", base_frame_name_);
  if (!model_.existFrame(base_frame_name_))
  {
    RCLCPP_ERROR(this->get_node()->get_logger(), "Frame %s does not exist in the Pinocchio model.",
                 base_frame_name_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  if (!model_.existFrame(waypoint_names_[0]))
  {
    RCLCPP_ERROR(this->get_node()->get_logger(), "Frame %s does not exist in the Pinocchio model.",
                 waypoint_names_[0].c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  base_frame_id_ = model_.getFrameId(base_frame_name_);
  ee_frame_id_ = model_.getFrameId(waypoint_names_[0]);

  std::vector<double> point_pos_b = auto_declare<std::vector<double>>("point_pos_b", {0.0, 0.0, 0.0});
  std::vector<double> point_rpy_b = auto_declare<std::vector<double>>("point_rpy_b", {0.0, 0.0, 0.0});
  if (point_pos_b.size() != 3 || point_rpy_b.size() != 3)
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Point pose planner expects 3 entries for point_pos_b and point_rpy_b.");
    return controller_interface::CallbackReturn::ERROR;
  }
  const Eigen::AngleAxisd roll_angle(point_rpy_b[0], Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch_angle(point_rpy_b[1], Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw_angle(point_rpy_b[2], Eigen::Vector3d::UnitZ());
  point_pos_b_ = Eigen::Vector3d(point_pos_b[0], point_pos_b[1], point_pos_b[2]);
  point_ori_b_ = Eigen::Quaterniond(yaw_angle * pitch_angle * roll_angle).normalized();
  const auto sine_start = auto_declare<std::vector<double>>("sine_line.start", {0.25, 0.0, 0.9});
  const auto sine_target = auto_declare<std::vector<double>>("sine_line.target", {2.05, 0.0, 0.9});
  const auto sine_orientation = auto_declare<std::vector<double>>("sine_line.orientation_wxyz", {1.0, 0.0, 0.0, 0.0});
  const double sine_duration = auto_declare<double>("sine_line.duration", 10.0);
  const double sine_amplitude = auto_declare<double>("sine_line.amplitude", 0.12);
  const double sine_wavelength = auto_declare<double>("sine_line.wavelength", 0.8);
  const double sine_phase = auto_declare<double>("sine_line.phase", 0.0);
  const bool sine_repeat = auto_declare<bool>("sine_line.repeat", false);
  if (sine_start.size() != 3 || sine_target.size() != 3 || sine_orientation.size() != 4)
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Sine-line trajectory requires start and target size 3 and orientation size 4.");
    return controller_interface::CallbackReturn::ERROR;
  }

  GlobalSineLinePoseTrajectory::Params sine_params;
  sine_params.start = Eigen::Vector3d(sine_start[0], sine_start[1], sine_start[2]);
  sine_params.target = Eigen::Vector3d(sine_target[0], sine_target[1], sine_target[2]);
  sine_params.orientation =
      Eigen::Quaterniond(sine_orientation[0], sine_orientation[1], sine_orientation[2], sine_orientation[3]);
  sine_params.duration = sine_duration;
  sine_params.amplitude = sine_amplitude;
  sine_params.wavelength = sine_wavelength;
  sine_params.phase = sine_phase;
  sine_params.repeat = sine_repeat;
  try
  {
    sine_line_trajectory_ = std::make_unique<GlobalSineLinePoseTrajectory>(sine_params);
  }
  catch (const std::exception& error)
  {
    RCLCPP_ERROR(this->get_node()->get_logger(), "Failed to create global sine-line trajectory: %s", error.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  const auto command_k_xy = auto_declare<std::vector<double>>("command_generator.k_xy", {1.0, 1.0});
  command_vxy_max_ = auto_declare<double>("command_generator.vxy_max", command_vxy_max_);
  command_k_yaw_ = auto_declare<double>("command_generator.k_yaw", command_k_yaw_);
  command_wz_max_ = auto_declare<double>("command_generator.wz_max", command_wz_max_);
  const auto command_k_arm =
      auto_declare<std::vector<double>>("command_generator.k_arm", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  const auto command_xi_max =
      auto_declare<std::vector<double>>("command_generator.xi_max", {0.15, 0.15, 0.15, 0.35, 0.35, 0.35});
  if (command_k_xy.size() != 2 || command_k_arm.size() != 6 || command_xi_max.size() != 6 || command_vxy_max_ < 0.0 ||
      command_wz_max_ < 0.0 ||
      std::any_of(command_xi_max.begin(), command_xi_max.end(), [](double limit) { return limit <= 0.0; }))
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Command generator requires k_xy size 2, k_arm and xi_max size 6, nonnegative velocity "
                 "limits, and positive xi_max entries.");
    return controller_interface::CallbackReturn::ERROR;
  }
  command_k_xy_ = Eigen::Vector2d(command_k_xy[0], command_k_xy[1]);
  for (Eigen::Index i = 0; i < 6; ++i)
  {
    command_k_arm_[i] = command_k_arm[static_cast<size_t>(i)];
    command_xi_max_[i] = command_xi_max[static_cast<size_t>(i)];
  }

  mode_button_ = auto_declare<int>("mode_button", mode_button_);
  const auto joy_qos = rclcpp::QoS(rclcpp::KeepLast(2), rmw_qos_profile_sensor_data);
  joy_subscriber_ = this->get_node()->create_subscription<sensor_msgs::msg::Joy>(
      "/joy", joy_qos,
      [this](const sensor_msgs::msg::Joy::SharedPtr msg)
      {
        const bool mode_button_pressed = mode_button_ >= 0 && static_cast<size_t>(mode_button_) < msg->buttons.size() &&
                                         msg->buttons[mode_button_] == 1;
        if (mode_button_pressed && !mode_button_pressed_prev_ && !sine_line_trajectory_requested_)
        {
          sine_line_trajectory_requested_ = true;
          sine_line_trajectory_started_ = false;
          has_meta_command_ = false;
        }
        mode_button_pressed_prev_ = mode_button_pressed;
      });

  // meta-dynamics related
  use_meta_dynamics_ = auto_declare<bool>("use_meta_dynamics", false);
  if (use_meta_dynamics_)
  {
    node_name_ = auto_declare<std::string>("node_name", "tron1_with_arm_planner");
    publish_topic_name_ = auto_declare<std::string>("publish_topic_name", "/tron1_with_arm_planner/debug/state");
    subscribe_topic_name_ = auto_declare<std::string>("subscribe_topic_name", "/tron1_with_arm_planner/debug/control");
    horizon_ = auto_declare<int>("horizon", 20);
    dt_ = auto_declare<double>("dt", 0.1);

    node_ptr_ = std::make_shared<rclcpp::Node>(node_name_);
    executor_.add_node(node_ptr_);

    auto qos = rclcpp::QoS(rclcpp::KeepLast(2), rmw_qos_profile_sensor_data);
    control_subscriber_ = node_ptr_->create_subscription<std_msgs::msg::Float64MultiArray>(
        subscribe_topic_name_, qos,
        [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg)
        {
          if (msg->data.size() < 7)
          {
            RCLCPP_WARN_THROTTLE(node_ptr_->get_logger(), *node_ptr_->get_clock(), 1000,
                                 "MPPI arm command has size %zu, expected at least 7.", msg->data.size());
            return;
          }

          const Eigen::Vector3d command_position_b(msg->data[0], msg->data[1], msg->data[2]);
          const Eigen::Quaterniond command_orientation_b(msg->data[3], msg->data[4], msg->data[5], msg->data[6]);
          if (!command_position_b.allFinite() || !command_orientation_b.coeffs().allFinite() ||
              command_orientation_b.norm() <= kTrajectoryTolerance)
          {
            RCLCPP_WARN_THROTTLE(node_ptr_->get_logger(), *node_ptr_->get_clock(), 1000,
                                 "MPPI arm command contains an invalid base-frame pose.");
            return;
          }

          // Store the MPPI desired EE pose in the base frame. The control loop
          // converts it back to world before generating matching arm/base commands.
          meta_command_pos_b_ = command_position_b;
          meta_command_ori_b_ = command_orientation_b.normalized();
          if (meta_command_ori_b_.w() < 0.0)
          {
            meta_command_ori_b_.coeffs() *= -1.0;
          }
          has_meta_command_ = true;
        });

    state_publisher_ = node_ptr_->create_publisher<std_msgs::msg::Float64MultiArray>(publish_topic_name_, qos);
    realtime_state_publisher_ =
        std::make_unique<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(state_publisher_);
  }

  debug_ = auto_declare<bool>("debug", false);
  if (debug_)
  {
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_ref_x_w", &ee_global_ref_position_debug_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_ref_y_w", &ee_global_ref_position_debug_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_ref_z_w", &ee_global_ref_position_debug_[2]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_ref_qw_w", &ee_global_ref_orientation_debug_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_ref_qx_w", &ee_global_ref_orientation_debug_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_ref_qy_w", &ee_global_ref_orientation_debug_[2]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_ref_qz_w", &ee_global_ref_orientation_debug_[3]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_real_x_w", &ee_global_real_position_debug_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_real_y_w", &ee_global_real_position_debug_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_real_z_w", &ee_global_real_position_debug_[2]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_real_qw_w", &ee_global_real_orientation_debug_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_real_qx_w", &ee_global_real_orientation_debug_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_real_qy_w", &ee_global_real_orientation_debug_[2]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ee_pose_real_qz_w", &ee_global_real_orientation_debug_[3]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("sine_trajectory_active", &sine_trajectory_active_debug_);
    REGISTER_ROS2_CONTROL_INTROSPECTION("sine_trajectory_elapsed_time", &sine_trajectory_elapsed_time_debug_);
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration Tron1WithArmPlanner::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  const auto joint_state_config = get_joint_state_interface_configuration();
  const auto pos_config = get_global_pos_interface_configuration();
  const auto ori_config = get_ori_interface_configuration();
  const auto lin_vel_config = get_global_lin_vel_interface_configuration();
  const auto ang_vel_config = get_global_ang_vel_interface_configuration();

  const auto append_names = [&config](const controller_interface::InterfaceConfiguration& source)
  { config.names.insert(config.names.end(), source.names.begin(), source.names.end()); };
  append_names(joint_state_config);
  append_names(pos_config);
  append_names(ori_config);
  append_names(lin_vel_config);
  append_names(ang_vel_config);

  if (ref_planner_name_ == "")
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1WithArmPlanner::state_interface_configuration() failed because ref_planner_name is not set.");
  }
  else
  {
    config.names.push_back(ref_planner_name_ + "/lin_x_vel_ref");
    config.names.push_back(ref_planner_name_ + "/lin_y_vel_ref");
    config.names.push_back(ref_planner_name_ + "/yaw_rate_ref");
  }

  return config;
}

controller_interface::InterfaceConfiguration Tron1WithArmPlanner::command_interface_configuration() const
{
  // use no command interface
  controller_interface::InterfaceConfiguration command_interface_config;
  command_interface_config.type = controller_interface::interface_configuration_type::NONE;

  return command_interface_config;  // command_interfaces_
}

controller_interface::CallbackReturn Tron1WithArmPlanner::on_configure(const rclcpp_lifecycle::State& previous_state)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1WithArmPlanner::on_activate(const rclcpp_lifecycle::State& previous_state)
{
  (void)previous_state;
  mode_button_pressed_prev_ = false;
  sine_line_trajectory_requested_ = false;
  sine_line_trajectory_started_ = false;
  sine_line_trajectory_start_time_seconds_ = 0.0;
  sine_trajectory_active_debug_ = 0.0;
  sine_trajectory_elapsed_time_debug_ = 0.0;
  has_meta_command_ = false;
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1WithArmPlanner::on_deactivate(const rclcpp_lifecycle::State& previous_state)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type Tron1WithArmPlanner::update_and_write_commands(const rclcpp::Time& time,
                                                                                 const rclcpp::Duration& period)
{
  (void)period;
  if (use_meta_dynamics_)
  {
    executor_.spin_some(std::chrono::milliseconds(1));
  }

  // 1. Get estimator data and update Pinocchio state.

  read_global_pos_from_state_interfaces(pos_);
  read_ori_from_state_interfaces(ori_);
  read_global_lin_vel_from_state_interfaces(lin_vel_);
  read_global_ang_vel_from_state_interfaces(ang_vel_);
  read_joint_states_from_state_interfaces(joint_pos_, joint_vel_, joint_tau_);

  // External velocity commands are used only before the sine trajectory starts.
  if (!sine_line_trajectory_requested_)
  {
    base_utils::get_state_interface_value(state_interfaces_, ref_planner_name_ + "/lin_x_vel_ref", velocity_cmd_[0]);
    base_utils::get_state_interface_value(state_interfaces_, ref_planner_name_ + "/lin_y_vel_ref", velocity_cmd_[1]);
    base_utils::get_state_interface_value(state_interfaces_, ref_planner_name_ + "/yaw_rate_ref", velocity_cmd_[2]);
  }

  // transform velocity to local frame
  Eigen::Quaterniond ori_quat(ori_[0], ori_[1], ori_[2], ori_[3]);  // w,x,y,z
  Eigen::Vector3d lin_vel_world_eig(lin_vel_[0], lin_vel_[1], lin_vel_[2]);
  Eigen::Vector3d ang_vel_world_eig(ang_vel_[0], ang_vel_[1], ang_vel_[2]);
  Eigen::Vector3d lin_vel_local_eig = ori_quat.conjugate() * lin_vel_world_eig;
  Eigen::Vector3d ang_vel_local_eig = ori_quat.conjugate() * ang_vel_world_eig;

  q_.segment<3>(0) << pos_[0], pos_[1], pos_[2];
  q_.segment<4>(3) << ori_[1], ori_[2], ori_[3], ori_[0];
  for (size_t i = 0; i < joint_pos_.size(); ++i)
  {
    q_(7 + perm_[i]) = joint_pos_[i];
  }

  v_.segment<3>(0) << lin_vel_local_eig[0], lin_vel_local_eig[1], lin_vel_local_eig[2];
  v_.segment<3>(3) << ang_vel_local_eig[0], ang_vel_local_eig[1], ang_vel_local_eig[2];
  for (size_t i = 0; i < joint_vel_.size(); ++i)
  {
    v_(6 + perm_[i]) = joint_vel_[i];
  }

  pinocchio::forwardKinematics(model_, data_, q_, v_);
  pinocchio::updateFramePlacements(model_, data_);

  const pinocchio::SE3& base_frame = data_.oMf[base_frame_id_];
  const pinocchio::SE3& ee_frame = data_.oMf[ee_frame_id_];
  const pinocchio::SE3 base_to_ee = base_frame.actInv(ee_frame);
  base_planners::Waypoint ee_pose_real_b;
  ee_pose_real_b.name = waypoint_names_[0];
  ee_pose_real_b.position = base_to_ee.translation();
  ee_pose_real_b.orientation = Eigen::Quaterniond(base_to_ee.rotation()).normalized();
  ee_pose_real_b.linear_velocity.setZero();
  ee_pose_real_b.angular_velocity.setZero();
  ee_pose_real_b.linear_acceleration.setZero();
  ee_pose_real_b.angular_acceleration.setZero();

  base_planners::Waypoint nominal_pose_b;
  nominal_pose_b.name = waypoint_names_[0];
  nominal_pose_b.position = point_pos_b_;
  nominal_pose_b.orientation = point_ori_b_.normalized();
  nominal_pose_b.linear_velocity.setZero();
  nominal_pose_b.angular_velocity.setZero();
  nominal_pose_b.linear_acceleration.setZero();
  nominal_pose_b.angular_acceleration.setZero();

  const pinocchio::SE3 nominal_ee_base(nominal_pose_b.orientation.toRotationMatrix(), nominal_pose_b.position);
  const pinocchio::SE3 nominal_ee_world = base_frame * nominal_ee_base;
  base_planners::Waypoint global_ee_reference;
  global_ee_reference.name = waypoint_names_[0];
  global_ee_reference.position = nominal_ee_world.translation();
  global_ee_reference.orientation = Eigen::Quaterniond(nominal_ee_world.rotation()).normalized();

  // Before the start button, retain the nominal local EE target.
  ee_pose_des_ = nominal_pose_b;
  if (sine_line_trajectory_requested_ && sine_line_trajectory_ != nullptr)
  {
    if (!sine_line_trajectory_started_)
    {
      sine_line_trajectory_start_time_seconds_ = time.seconds();
      sine_line_trajectory_->reset(sine_line_trajectory_start_time_seconds_);
      sine_line_trajectory_started_ = true;
      has_meta_command_ = false;
    }

    const base_planners::Waypoint world_reference = sine_line_trajectory_->sample(time.seconds());
    global_ee_reference = world_reference;
    base_planners::Waypoint command_world_reference = world_reference;
    if (use_meta_dynamics_ && has_meta_command_)
    {
      command_world_reference = express_base_reference_in_world(meta_command_pos_b_, meta_command_ori_b_);
    }

    auto generated_commands = generate_sine_commands(command_world_reference);
    ee_pose_des_ = std::move(generated_commands.first);
    velocity_cmd_ = generated_commands.second;
  }

  if (debug_)
  {
    sine_trajectory_active_debug_ = sine_line_trajectory_started_ ? 1.0 : 0.0;
    sine_trajectory_elapsed_time_debug_ =
        sine_line_trajectory_started_ ? std::max(0.0, time.seconds() - sine_line_trajectory_start_time_seconds_) : 0.0;

    Eigen::Quaterniond reference_orientation = global_ee_reference.orientation.normalized();
    if (reference_orientation.w() < 0.0)
    {
      reference_orientation.coeffs() *= -1.0;
    }
    Eigen::Quaterniond real_orientation(ee_frame.rotation());
    real_orientation.normalize();
    if (real_orientation.dot(reference_orientation) < 0.0)
    {
      real_orientation.coeffs() *= -1.0;
    }

    ee_global_ref_position_debug_ = {global_ee_reference.position.x(), global_ee_reference.position.y(),
                                     global_ee_reference.position.z()};
    ee_global_ref_orientation_debug_ = {reference_orientation.w(), reference_orientation.x(), reference_orientation.y(),
                                        reference_orientation.z()};
    ee_global_real_position_debug_ = {ee_frame.translation().x(), ee_frame.translation().y(),
                                      ee_frame.translation().z()};
    ee_global_real_orientation_debug_ = {real_orientation.w(), real_orientation.x(), real_orientation.y(),
                                         real_orientation.z()};
  }

  // Publish current local EE pose followed by the desired local EE horizon.
  if (use_meta_dynamics_ && realtime_state_publisher_ != nullptr && sine_line_trajectory_ != nullptr)
  {
    auto msg = std_msgs::msg::Float64MultiArray();
    msg.data.resize(7 * (horizon_ + 1), 0.0);

    auto write_waypoint = [&msg](int index, const base_planners::Waypoint& waypoint)
    {
      const Eigen::Quaterniond orientation = waypoint.orientation.normalized();
      msg.data[7 * index + 0] = waypoint.position.x();
      msg.data[7 * index + 1] = waypoint.position.y();
      msg.data[7 * index + 2] = waypoint.position.z();
      msg.data[7 * index + 3] = orientation.w();
      msg.data[7 * index + 4] = orientation.x();
      msg.data[7 * index + 5] = orientation.y();
      msg.data[7 * index + 6] = orientation.z();
    };

    write_waypoint(0, ee_pose_real_b);
    for (int i = 1; i <= horizon_; ++i)
    {
      if (sine_line_trajectory_requested_)
      {
        const auto world_reference = sine_line_trajectory_->sample(time.seconds() + static_cast<double>(i) * dt_);
        write_waypoint(i, express_world_reference_in_base(world_reference));
      }
      else
      {
        write_waypoint(i, nominal_pose_b);
      }
    }

    if (realtime_state_publisher_->trylock())
    {
      realtime_state_publisher_->msg_ = msg;
      realtime_state_publisher_->unlockAndPublish();
    }
  }

  return controller_interface::return_type::OK;
}

base_planners::Waypoint Tron1WithArmPlanner::express_world_reference_in_base(
    const base_planners::Waypoint& world_reference) const
{
  const pinocchio::SE3& current_base_world = data_.oMf[base_frame_id_];
  const pinocchio::SE3 desired_ee_world(world_reference.orientation.normalized().toRotationMatrix(),
                                        world_reference.position);
  const pinocchio::SE3 desired_ee_base = current_base_world.inverse() * desired_ee_world;

  base_planners::Waypoint desired_reference_b;
  desired_reference_b.name = waypoint_names_[0];
  desired_reference_b.position = desired_ee_base.translation();
  desired_reference_b.orientation = Eigen::Quaterniond(desired_ee_base.rotation()).normalized();
  if (desired_reference_b.orientation.w() < 0.0)
  {
    desired_reference_b.orientation.coeffs() *= -1.0;
  }
  desired_reference_b.linear_velocity.setZero();
  desired_reference_b.angular_velocity.setZero();
  desired_reference_b.linear_acceleration.setZero();
  desired_reference_b.angular_acceleration.setZero();
  return desired_reference_b;
}

base_planners::Waypoint Tron1WithArmPlanner::express_base_reference_in_world(
    const Eigen::Vector3d& position_b, const Eigen::Quaterniond& orientation_b) const
{
  const pinocchio::SE3& current_base_world = data_.oMf[base_frame_id_];
  const pinocchio::SE3 desired_ee_base(orientation_b.normalized().toRotationMatrix(), position_b);
  const pinocchio::SE3 desired_ee_world = current_base_world * desired_ee_base;

  base_planners::Waypoint desired_reference_w;
  desired_reference_w.name = waypoint_names_[0];
  desired_reference_w.position = desired_ee_world.translation();
  desired_reference_w.orientation = Eigen::Quaterniond(desired_ee_world.rotation()).normalized();
  if (desired_reference_w.orientation.w() < 0.0)
  {
    desired_reference_w.orientation.coeffs() *= -1.0;
  }
  desired_reference_w.linear_velocity.setZero();
  desired_reference_w.angular_velocity.setZero();
  desired_reference_w.linear_acceleration.setZero();
  desired_reference_w.angular_acceleration.setZero();
  return desired_reference_w;
}

std::pair<base_planners::Waypoint, std::array<double, 3>> Tron1WithArmPlanner::generate_sine_commands(
    const base_planners::Waypoint& world_reference) const
{
  const pinocchio::SE3& current_base_world = data_.oMf[base_frame_id_];
  const pinocchio::SE3 nominal_ee_base(point_ori_b_.toRotationMatrix(), point_pos_b_);
  const pinocchio::SE3 desired_ee_world(world_reference.orientation.normalized().toRotationMatrix(),
                                        world_reference.position);

  const pinocchio::SE3 desired_base_world = desired_ee_world * nominal_ee_base.inverse();
  const Eigen::Vector2d xy_error_world =
      desired_base_world.translation().head<2>() - current_base_world.translation().head<2>();
  Eigen::Vector2d velocity_world = command_k_xy_.cwiseProduct(xy_error_world);
  const double velocity_norm = velocity_world.norm();
  if (velocity_norm > command_vxy_max_ && velocity_norm > kTrajectoryTolerance)
  {
    velocity_world *= command_vxy_max_ / velocity_norm;
  }

  const double current_yaw = yaw_from_rotation(current_base_world.rotation());
  const double desired_yaw = yaw_from_rotation(desired_base_world.rotation());
  const double cosine_yaw = std::cos(current_yaw);
  const double sine_yaw = std::sin(current_yaw);
  const Eigen::Vector2d velocity_base(cosine_yaw * velocity_world.x() + sine_yaw * velocity_world.y(),
                                      -sine_yaw * velocity_world.x() + cosine_yaw * velocity_world.y());
  const double yaw_rate =
      std::clamp(command_k_yaw_ * wrap_to_pi(desired_yaw - current_yaw), -command_wz_max_, command_wz_max_);

  const pinocchio::SE3 desired_ee_base = current_base_world.inverse() * desired_ee_world;
  const Eigen::Matrix<double, 6, 1> residual = pinocchio::log6(nominal_ee_base.inverse() * desired_ee_base).toVector();
  Eigen::Matrix<double, 6, 1> bounded_residual = command_k_arm_.cwiseProduct(residual);
  bounded_residual = bounded_residual.cwiseMin(command_xi_max_).cwiseMax(-command_xi_max_);
  const pinocchio::SE3 commanded_ee_base = nominal_ee_base * pinocchio::exp6(bounded_residual);

  base_planners::Waypoint ee_command;
  ee_command.name = waypoint_names_[0];
  ee_command.position = commanded_ee_base.translation();
  ee_command.orientation = Eigen::Quaterniond(commanded_ee_base.rotation()).normalized();
  if (ee_command.orientation.w() < 0.0)
  {
    ee_command.orientation.coeffs() *= -1.0;
  }
  ee_command.linear_velocity.setZero();
  ee_command.angular_velocity.setZero();
  ee_command.linear_acceleration.setZero();
  ee_command.angular_acceleration.setZero();

  const std::array<double, 3> base_command{velocity_base.x(), velocity_base.y(), yaw_rate};
  return std::make_pair(ee_command, base_command);
}

controller_interface::return_type Tron1WithArmPlanner::update_reference_from_subscribers(const rclcpp::Time& time,
                                                                                         const rclcpp::Duration& period)
{
  return controller_interface::return_type::OK;
}

std::vector<hardware_interface::StateInterface> Tron1WithArmPlanner::on_export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  std::string planner_name = this->get_name();
  // waypoints
  ee_pose_des_.name = waypoint_names_.empty() ? std::string("") : waypoint_names_[0];
  base_planners::Waypoint::append_state_interfaces(planner_name, ee_pose_des_, state_interfaces);
  // velocities
  state_interfaces.emplace_back(hardware_interface::StateInterface(planner_name, "lin_x_vel_ref", &velocity_cmd_[0]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(planner_name, "lin_y_vel_ref", &velocity_cmd_[1]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(planner_name, "yaw_rate_ref", &velocity_cmd_[2]));

  return state_interfaces;
}

};  // namespace tron1_deploy

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(tron1_deploy::Tron1WithArmPlanner, controller_interface::ChainableControllerInterface);
