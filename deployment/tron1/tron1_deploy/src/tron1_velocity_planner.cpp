#include "tron1_deploy/tron1_velocity_planner.hpp"

namespace tron1_deploy
{
controller_interface::CallbackReturn Tron1VelocityPlanner::on_init()
{
  // configure the joystick related stuffs
  max_linear_velocity_x_ = auto_declare<double>("max_linear_velocity_x", 0.0);
  max_linear_velocity_y_ = auto_declare<double>("max_linear_velocity_y", 0.0);
  max_angular_velocity_yaw_ = auto_declare<double>("max_angular_velocity_yaw", 0.0);
  constant_velocity_x_ = auto_declare<double>("constant_linear_velocity_x", 0.0);
  constant_velocity_y_ = auto_declare<double>("constant_linear_velocity_y", 0.0);
  constant_yaw_rate_ = auto_declare<double>("constant_angular_velocity_yaw", 0.0);
  int velocity_x_button_ = auto_declare<int>("velocity_x_button", 0);
  int velocity_y_button_ = auto_declare<int>("velocity_y_button", 1);
  int yaw_rate_button_ = auto_declare<int>("yaw_rate_button", 2);
  int mode_button_ = auto_declare<int>("mode_button", 3);
  // configure the ros2 related stuffs
  node_ptr_ = rclcpp::Node::make_shared(std::string(this->get_node()->get_name()) + "_joystick_listener");
  auto qos = rclcpp::QoS(rclcpp::KeepLast(2), rmw_qos_profile_sensor_data);
  joy_subscriber_ = this->get_node()->create_subscription<sensor_msgs::msg::Joy>(
      "/joy", qos,
      [this, velocity_x_button_, velocity_y_button_, yaw_rate_button_, mode_button_](const sensor_msgs::msg::Joy::SharedPtr msg)
      {
        static bool last_button_state = false;
        bool current_button_state = msg->buttons[mode_button_];
        if (current_button_state && !last_button_state) {
          mode_ = !mode_;  // Toggle mode
        }
        last_button_state = current_button_state;

        // Update velocity commands based on mode
        if (mode_) {
          // Constant velocity mode
          velocity_cmd_raw_[0] = constant_velocity_x_;
          velocity_cmd_raw_[1] = constant_velocity_y_;
          velocity_cmd_raw_[2] = constant_yaw_rate_;
        } else {
          // Joystick control mode
          velocity_cmd_raw_[0] = msg->axes[velocity_x_button_];
          velocity_cmd_raw_[1] = msg->axes[velocity_y_button_];
          velocity_cmd_raw_[2] = msg->axes[yaw_rate_button_];
        }
      });
  executor_.add_node(node_ptr_);
  // configure the filter
  sampling_frequency_ = auto_declare<double>("sampling_frequency", 50.0);
  damping_frequency_ = auto_declare<double>("damping_frequency", 1.0);
  damping_intensity_ = auto_declare<double>("damping_intensity", 0.0);
  for (size_t i = 0; i < 3; ++i)
  {
    lp_filters_[i] = std::make_shared<control_toolbox::LowPassFilter<double>>(sampling_frequency_, damping_frequency_,
                                                                              damping_intensity_);
    lp_filters_[i]->configure();
  }
  // configure max velocity
  velocity_planner_ = base_planners::VelocityPlanner({max_linear_velocity_x_, max_linear_velocity_y_, max_angular_velocity_yaw_});
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration Tron1VelocityPlanner::state_interface_configuration() const
{
  // no state interfaces used
  controller_interface::InterfaceConfiguration state_interface_config;
  state_interface_config.type = controller_interface::interface_configuration_type::NONE;
  return state_interface_config;  // state_interfaces_
}

controller_interface::InterfaceConfiguration Tron1VelocityPlanner::command_interface_configuration() const
{
  // use no command interface
  controller_interface::InterfaceConfiguration command_interface_config;
  command_interface_config.type = controller_interface::interface_configuration_type::NONE;

  return command_interface_config;  // command_interfaces_
}

controller_interface::CallbackReturn Tron1VelocityPlanner::on_configure(const rclcpp_lifecycle::State&)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1VelocityPlanner::on_activate(const rclcpp_lifecycle::State&)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn Tron1VelocityPlanner::on_deactivate(const rclcpp_lifecycle::State&)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type Tron1VelocityPlanner::update_and_write_commands(const rclcpp::Time& time,
                                                                                     const rclcpp::Duration& period)
{
  // update the message
  executor_.spin_some(std::chrono::milliseconds(1));

  // filter the velocity commands
  for (size_t i = 0; i < 3; ++i)
  {
    lp_filters_[i]->update(velocity_cmd_raw_[i], velocity_cmd_[i]);
  }
  // clip the velocity commands
  velocity_cmd_ = velocity_planner_.compute(velocity_cmd_);
  return controller_interface::return_type::OK;
}

controller_interface::return_type Tron1VelocityPlanner::update_reference_from_subscribers(
    const rclcpp::Time& time, const rclcpp::Duration& period)
{
  return controller_interface::return_type::OK;
}

std::vector<hardware_interface::StateInterface> Tron1VelocityPlanner::on_export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  std::string planner_name = this->get_name();
  state_interfaces.emplace_back(hardware_interface::StateInterface(planner_name, "lin_x_vel_ref", &velocity_cmd_[0]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(planner_name, "lin_y_vel_ref", &velocity_cmd_[1]));
  state_interfaces.emplace_back(hardware_interface::StateInterface(planner_name, "yaw_rate_ref", &velocity_cmd_[2]));
  return state_interfaces;
}

}; // namespace tron1_deploy

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(tron1_deploy::Tron1VelocityPlanner, controller_interface::ChainableControllerInterface);
