#!/usr/bin/env python3
from ament_index_python.packages import get_package_share_directory
import os
from datetime import datetime
import rclpy
from rclpy.lifecycle import State, TransitionCallbackReturn, LifecycleNode
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Joy

import jax
import jax.numpy as jnp
from training.dynamics.training.tron1.tron1_model import Tron1Model, Tron1ModelConfig
from training.dynamics.training.model_trainer import JAXTrainer, JAXDataset, ReptileTrainerConfig
from training.controllers.MPPI import MPPI, MPPIConfig

class MPPIControlNode(LifecycleNode):
    def __init__(self, model_config: Tron1ModelConfig, MPPI_config: MPPIConfig):
        super().__init__('mppi_planner_node')

        # mppi params
        self.declare_parameter("mppi.num_samples", MPPI_config.num_samples)
        self.declare_parameter("mppi.noise", MPPI_config.noise)
        self.declare_parameter("mppi.lam", MPPI_config.lam)
        self.declare_parameter("mppi.horizon", MPPI_config.horizon)
        self.declare_parameter("mppi.gains", MPPI_config.gains)
        self.declare_parameter("mppi.cost_decay", MPPI_config.cost_decay)
        MPPI_config = MPPIConfig(
            horizon=int(self.get_parameter("mppi.horizon").value),
            dt=MPPI_config.dt,
            num_samples=int(self.get_parameter("mppi.num_samples").value),
            act_dim=MPPI_config.act_dim,
            obs_dim=MPPI_config.obs_dim,
            act_bounds=MPPI_config.act_bounds,
            noise=float(self.get_parameter("mppi.noise").value),
            lam=float(self.get_parameter("mppi.lam").value),
            spline_order=MPPI_config.spline_order,
            n_knots=MPPI_config.n_knots,
            gains=self.get_parameter("mppi.gains").value,
            cost_decay=self.get_parameter("mppi.cost_decay").value,
        )
        # model params
        self.declare_parameter("model.path", model_config.load_path)
        pkg_share = get_package_share_directory("tron1_deploy")
        model_path = os.path.join(pkg_share, self.get_parameter("model.path").value)
        model_config = Tron1ModelConfig(
            input_dim=model_config.input_dim,
            output_dim=model_config.output_dim,
            hidden_sizes=model_config.hidden_sizes,
            activation=model_config.activation,
            key=model_config.key,
            type=model_config.type,
            inference_mode=model_config.inference_mode,
            lambda_c=model_config.lambda_c,
            lambda_d=model_config.lambda_d,
            input_range=model_config.input_range,
            output_range=model_config.output_range,
            load_path=model_path
        )
        self.get_logger().info(f"Load model from path: {model_config.load_path}")

        self.model = Tron1Model(model_config)
        self.mppi = MPPI(self.model ,MPPI_config)
        self.key = jax.random.PRNGKey(MPPI_config.seed)
        self.get_logger().info("Action Dimension of MPPI Controller: {}".format(MPPI_config.act_dim))

        # data
        self.reference = jnp.zeros((1,MPPI_config.horizon, MPPI_config.act_dim)) # (1, horizon, command_dim)
        self.obs = jnp.zeros(MPPI_config.obs_dim)
        self.act = jnp.zeros(MPPI_config.act_dim)

        self.last_update_time = self.get_clock().now().nanoseconds * 1e-9  # seconds
        self.update_period = 0.02  # seconds, adjust as needed

        self.training_cfg = ReptileTrainerConfig(
            batch = 64,
            learning_rate = 1e-4,
            weight_decay= 0.0,
            eps=1e-8,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            # training
            epochs=1,
            evaluate=False,
            silent=True,
            seed=42,
            training_name="online_finetuning_tron1_mppi",
            # reptile
            task_batch=5,
            inner_steps=5
        )

        self.trainer = JAXTrainer(self.model, self.training_cfg)

        self.input_buff = jnp.zeros((self.training_cfg.batch + (self.training_cfg.inner_steps-1)*self.training_cfg.batch, model_config.input_dim))
        self.output_buff = jnp.zeros((self.training_cfg.batch + (self.training_cfg.inner_steps-1)*self.training_cfg.batch, model_config.output_dim))
        self.finetune_nums = 30 # 30, 80

        self.optimal_solution = jnp.zeros((MPPI_config.horizon, MPPI_config.act_dim))

        self.debug = False

    def set_params(self):
        self.declare_parameter("subscribe_topic_name", "")
        self.declare_parameter("publish_topic_name", "")
        self.subscribe_topic_name_ = self.get_parameter("subscribe_topic_name").get_parameter_value().string_value
        self.publish_topic_name_ = self.get_parameter("publish_topic_name").get_parameter_value().string_value

        # mode
        self.declare_parameter("mode_button", 0)
        self.mode_button_ = self.get_parameter("mode_button").get_parameter_value().integer_value

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.set_params()

        # setup the subscriber and the publisher
        qpos = qos_profile_sensor_data
        self.publisher_ = self.create_publisher(Float64MultiArray, self.publish_topic_name_, qpos)
        self.subscriber_ = self.create_subscription(Float64MultiArray, self.subscribe_topic_name_, self.get_observation_callback, qpos)

        self.joy_subscriber_ = self.create_subscription(Joy, "/joy", self.joy_callback, qpos)
        self.prev_mode_ = 0

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state) -> TransitionCallbackReturn:
        # setup the timer
        self.timer = self.create_timer(0.02, self.compute) # 50Hz

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        pkg_share = get_package_share_directory("tron1_deploy")
        self.model_save_dir = os.path.join(pkg_share, "bag", "models", "velocity")
        try:
            os.makedirs(self.model_save_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.model_save_dir, f"reptile_finetuned_{ts}.npz")
            self.model.save(save_path)
            self.get_logger().info(f"Saved finetuned model: {save_path}")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Failed to save finetuned model on cleanup: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    def get_observation_callback(self, msg: Float64MultiArray):
        array = jnp.array(msg.data, dtype=jnp.float32)

        self.input_buff = jnp.roll(self.input_buff, shift=-1, axis=0)
        self.output_buff = jnp.roll(self.output_buff, shift=-1, axis=0)
        self.input_buff = self.input_buff.at[-1].set(jnp.concatenate([self.obs, self.act]))
        self.output_buff = self.output_buff.at[-1].set(array[0:3])

        self.obs = jnp.concatenate([self.obs[3:], array[0:3]])
        # self.act = array[0:3]

        self.reference = self.reference.at[0,:].set(array[3:].reshape(self.mppi.horizon, self.mppi.act_dim))

    def joy_callback(self, msg: Joy):
        if (msg.buttons[self.mode_button_] == 1) and (self.prev_mode_ == 0):
            self.timer2 = self.create_timer(0.5, self.finetuning)
            self.prev_mode_ = 1

    def compute(self):
        # publish
        msg = Float64MultiArray()

        if self.finetune_nums > 0:
            msg.data = self.reference[0,0,:].flatten().tolist()
        else:
            new_command, self.key, total_cost =  self.mppi.act(self.obs, self.reference, self.key, self.reference[0])
            msg.data = new_command.tolist()
            # use original yaw value
            msg.data[2] = self.reference[0,0,2]
        if(self.debug):
            # TODO: use the model to do a prediction
            pass
        self.act = jnp.array(msg.data)
        self.publisher_.publish(msg)

    def finetuning(self):
        if self.finetune_nums <=0:
            if self.model.inference_mode == False:
                self.model.set_inference_mode(True)
            return
        dataset = JAXDataset([(self.input_buff[i], self.output_buff[i]) for i in range(len(self.input_buff))])
        self.trainer.finetune(dataset, shuffle_length=8)
        self.finetune_nums -=1

    def predict(self):
        pass

def main(args=None):
    mppi_cfg = MPPIConfig(
        horizon=8,
        dt=0.02,
        num_samples=5000,
        act_dim=3,
        obs_dim=30,
        act_bounds=[[-3.0], [3.0]],
        noise=0.3, # 0.3
        lam=0.04, # 0.05
        spline_order=3,
        n_knots=6,
        gains=[10.0, 0.5, 0.5], # 10.0, 2.0, 2.0 for 0.5m/s
        cost_decay=0.8,
    )
    pkg_share = get_package_share_directory("tron1_deploy")
    model_path = os.path.join(pkg_share, "model", "velocity_policy", "reptile_mppi.npz")
    model_cfg = Tron1ModelConfig(
        input_dim=33,
        output_dim=3,
        hidden_sizes=[32, 64, 256, 64, 32],
        activation="mish",
        key=42,
        type="mlp",
        inference_mode=False,
        lambda_c=0.0,
        lambda_d=0.0,
        input_range=tuple((-5.0, 5.0) for _ in range(33)),
        output_range=tuple((-5.0, 5.0) for _ in range(3)),
        load_path=model_path
    )

    rclpy.init(args = args)
    node_name = MPPIControlNode(model_cfg,mppi_cfg)
    configure_result = node_name.trigger_configure()
    if configure_result != TransitionCallbackReturn.SUCCESS:
        node_name.get_logger().error(f"Failed to configure node: {configure_result}")
    activate_result = node_name.trigger_activate()
    if activate_result != TransitionCallbackReturn.SUCCESS:
        node_name.get_logger().error(f"Failed to activate node: {activate_result}")

    try:
        rclpy.spin(node_name)
    except KeyboardInterrupt:
        node_name.get_logger().info("KeyboardInterrupt received, running lifecycle cleanup...")
    finally:
        try:
            node_name.trigger_deactivate()
        except Exception:
            pass
        try:
            node_name.trigger_cleanup()   # calls on_cleanup()
        except Exception:
            pass

        node_name.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()
