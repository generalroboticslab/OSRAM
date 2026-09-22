#include "tron1_deploy/tron1_mppi_planner.hpp"

namespace tron1_deploy
{

controller_interface::CallbackReturn Tron1MPPIPlanner::on_init()
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration Tron1MPPIPlanner::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration command_interface_config;
  command_interface_config.type = controller_interface::interface_configuration_type::NONE;
  return command_interface_config;  // command_interfaces_
}

controller_interface::InterfaceConfiguration Tron1MPPIPlanner::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration state_interface_config;
  state_interface_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // state estimation
  // Position (x, y, z)
  state_interface_config.names.push_back(estimator_name_ + "/" + pos_name_ + "_x_est");
  state_interface_config.names.push_back(estimator_name_ + "/" + pos_name_ + "_y_est");
  state_interface_config.names.push_back(estimator_name_ + "/" + pos_name_ + "_z_est");

  // Orientation (w, x, y, z)
  state_interface_config.names.push_back(estimator_name_ + "/" + ori_name_ + "_w_est");
  state_interface_config.names.push_back(estimator_name_ + "/" + ori_name_ + "_x_est");
  state_interface_config.names.push_back(estimator_name_ + "/" + ori_name_ + "_y_est");
  state_interface_config.names.push_back(estimator_name_ + "/" + ori_name_ + "_z_est");

  // Linear velocity (x, y, z)
  state_interface_config.names.push_back(estimator_name_ + "/" + lin_vel_name_ + "_x_est");
  state_interface_config.names.push_back(estimator_name_ + "/" + lin_vel_name_ + "_y_est");
  state_interface_config.names.push_back(estimator_name_ + "/" + lin_vel_name_ + "_z_est");

  // Angular velocity (x, y, z)
  state_interface_config.names.push_back(estimator_name_ + "/" + ang_vel_name_ + "_x_est");
  state_interface_config.names.push_back(estimator_name_ + "/" + ang_vel_name_ + "_y_est");
  state_interface_config.names.push_back(estimator_name_ + "/" + ang_vel_name_ + "_z_est");

  // Reference velocities from velocity planner
  state_interface_config.names.push_back(ref_planner_name_ + "/lin_x_vel_ref");
  state_interface_config.names.push_back(ref_planner_name_ + "/lin_y_vel_ref");
  state_interface_config.names.push_back(ref_planner_name_ + "/yaw_rate_ref");

  return state_interface_config;  // state_interfaces_
}

std::vector<hardware_interface::StateInterface> Tron1MPPIPlanner::on_export_state_interfaces()
{
  std::string planner_name = this->get_name();
  std::vector<hardware_interface::StateInterface> state_interfaces;

  // reference trajectory
  state_interfaces.emplace_back(planner_name, "lin_x_vel_ref", &velocity_cmd_[0]);  // vx, local
  state_interfaces.emplace_back(planner_name, "lin_y_vel_ref", &velocity_cmd_[1]);  // vy, local
  state_interfaces.emplace_back(planner_name, "yaw_rate_ref", &velocity_cmd_[2]);   // yaw rate, local
  return state_interfaces;
}

controller_interface::CallbackReturn Tron1MPPIPlanner::on_configure(const rclcpp_lifecycle::State&)
{
  // planning related
  horizon_ = auto_declare<int>("horizon", horizon_);
  dt_ = auto_declare<double>("dt", dt_);

  // sub and pub
  node_name_ = auto_declare<std::string>("node_name", "");
  subscribe_topic_name_ = auto_declare<std::string>("subscribe_topic_name", "");
  publish_topic_name_ = auto_declare<std::string>("publish_topic_name", "");
  estimator_name_ = auto_declare<std::string>("estimator_name", "");
  ref_planner_name_ = auto_declare<std::string>("ref_planner_name", "");

  if (node_name_.empty() || subscribe_topic_name_.empty() || publish_topic_name_.empty())
  {
    RCLCPP_ERROR(this->get_node()->get_logger(),
                 "Tron1MPPIPlanner: 'node_name', 'publish_topic_name' or 'subscribe_topic_name' parameters are empty.");
    return controller_interface::CallbackReturn::FAILURE;
  }
  node_ptr_ = rclcpp::Node::make_shared(node_name_);
  auto qos = rclcpp::QoS(rclcpp::KeepLast(2), rmw_qos_profile_sensor_data);
  control_subscriber_ = node_ptr_->create_subscription<std_msgs::msg::Float64MultiArray>(
      subscribe_topic_name_, qos,
      [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg)
      {
        velocity_cmd_[0] = msg->data[0];
        velocity_cmd_[1] = msg->data[1];
        velocity_cmd_[2] = msg->data[2];
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

  debug_ = auto_declare<bool>("debug", false);
  if(debug_)
  {
    REGISTER_ROS2_CONTROL_INTROSPECTION("lin_vel_x_cmd", &velocity_cmd_[0]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("lin_vel_y_cmd", &velocity_cmd_[1]);
    REGISTER_ROS2_CONTROL_INTROSPECTION("ang_vel_yaw_cmd", &velocity_cmd_[2]);
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1MPPIPlanner::on_cleanup(const rclcpp_lifecycle::State&)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1MPPIPlanner::on_activate(const rclcpp_lifecycle::State&)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1MPPIPlanner::on_deactivate(const rclcpp_lifecycle::State&)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

bool Tron1MPPIPlanner::on_set_chained_mode(bool chained_mode)
{
  return true;  // enable chaining since this is a controller and leverage command interface
}

controller_interface::return_type Tron1MPPIPlanner::update_and_write_commands(const rclcpp::Time& time,
                                                                              const rclcpp::Duration& period)
{
  // update states for the successor
  lin_vel_[0] = state_interfaces_[7].get_value();        // vx_global
  lin_vel_[1] = state_interfaces_[8].get_value();        // vy_global
  lin_vel_[2] = state_interfaces_[9].get_value();        // vz_global
  ang_vel_[0] = state_interfaces_[10].get_value();       // roll_rate_global
  ang_vel_[1] = state_interfaces_[11].get_value();       // pitch_rate_global
  ang_vel_[2] = state_interfaces_[12].get_value();       // yaw_rate_global
  velocity_des_[0] = state_interfaces_[13].get_value();  // vx_des_local
  velocity_des_[1] = state_interfaces_[14].get_value();  // vy_des_local
  velocity_des_[2] = state_interfaces_[15].get_value();  // yaw_rate_local
  // filter the velocity
  std::array<double, 3> lin_vel_raw = {lin_vel_[0], lin_vel_[1], lin_vel_[2]};
  std::array<double, 3> ang_vel_raw = {ang_vel_[0], ang_vel_[1], ang_vel_[2]};
  for (size_t i = 0; i < 2; ++i)
  {
    lp_filters_[i]->update(lin_vel_raw[i], lin_vel_[i]);
  }
  lp_filters_[2]->update(ang_vel_raw[2], ang_vel_[2]);

  // transform global velocity to local velocity
  Eigen::Quaterniond ori_quat(state_interfaces_[3].get_value(),  // w
                              state_interfaces_[4].get_value(),  // x
                              state_interfaces_[5].get_value(),  // y
                              state_interfaces_[6].get_value()   // z
  );
  ori_quat.normalize();
  Eigen::Vector3d lin_vel_global(lin_vel_[0], lin_vel_[1], lin_vel_[2]);
  Eigen::Vector3d ang_vel_global(ang_vel_[0], ang_vel_[1], ang_vel_[2]);
  Eigen::Vector3d lin_vel_local = ori_quat.inverse() * lin_vel_global;
  Eigen::Vector3d ang_vel_local = ori_quat.inverse() * ang_vel_global;
  velocity_est_[0] = lin_vel_local[0];  // vx_local
  velocity_est_[1] = lin_vel_local[1];  // vy_local
  velocity_est_[2] = ang_vel_local[2];  // yaw_rate_local

  std::vector<double> lin_vel_x_des_vec_(horizon_, 0.0);
  std::vector<double> lin_vel_y_des_vec_(horizon_, 0.0);
  std::vector<double> ang_vel_z_des_vec_(horizon_, 0.0);
  for (int i = 0; i < horizon_; i++)
  {
    lin_vel_x_des_vec_[i] = velocity_des_[0];
    lin_vel_y_des_vec_[i] = velocity_des_[1];
    ang_vel_z_des_vec_[i] = velocity_des_[2];
  }

  // publish states
  executor_.spin_some(std::chrono::milliseconds(1));
  auto msg = std_msgs::msg::Float64MultiArray();
  msg.data.resize(3 + 3 * horizon_);
  msg.data[0] = lin_vel_local[0];  // vx_local
  msg.data[1] = lin_vel_local[1];  // vy_local
  msg.data[2] = ang_vel_local[2];  // yaw_rate_local
  for (int i = 0; i < horizon_; i++)
  {
    msg.data[3 + 3 * i] = lin_vel_x_des_vec_[i];
    msg.data[3 + 3 * i + 1] = lin_vel_y_des_vec_[i];
    msg.data[3 + 3 * i + 2] = ang_vel_z_des_vec_[i];
  }
  realtime_state_publisher_->lock();
  realtime_state_publisher_->msg_ = msg;
  realtime_state_publisher_->unlockAndPublish();

  return controller_interface::return_type::OK;
}

controller_interface::return_type Tron1MPPIPlanner::update_reference_from_subscribers(const rclcpp::Time& time,
                                                                                      const rclcpp::Duration& period)
{
  return controller_interface::return_type::OK;
}

};  // namespace tron1_deploy

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(tron1_deploy::Tron1MPPIPlanner, controller_interface::ChainableControllerInterface);
