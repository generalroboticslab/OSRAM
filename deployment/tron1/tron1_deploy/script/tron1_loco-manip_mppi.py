#!/usr/bin/env python3
from ament_index_python.packages import get_package_share_directory
from datetime import datetime, timezone
import os
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.lifecycle import State, TransitionCallbackReturn, LifecycleNode
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Joy

import jax
import jax.numpy as jnp
from training.dynamics.training.tron1.tron1_model import Tron1Model, Tron1ModelConfig
from training.dynamics.training.model_trainer import JAXTrainer, ReptileTrainerConfig
from training.utils.model_utils import update as jitted_update
from training.controllers.MPPI_action_history import MPPIActionHistory, MPPIActionHistoryConfig


def normalize_mppi_noise(noise_value, act_dim):
    """Return scalar noise or an act_dim-sized JAX noise vector."""
    if isinstance(noise_value, (list, tuple)):
        if len(noise_value) != act_dim:
            raise ValueError(
                "mppi.noise array length must match mppi.act_dim: "
                f"got {len(noise_value)}, expected {act_dim}."
            )
        return jnp.asarray(noise_value, dtype=jnp.float32)

    return float(noise_value)


class MPPIControlNode(LifecycleNode):
    def __init__(self, model_config: Tron1ModelConfig, MPPI_config: MPPIActionHistoryConfig):
        super().__init__('mppi_planner_node')

        # mppi params
        self.declare_parameter("mppi.num_samples", MPPI_config.num_samples)
        self.declare_parameter(
            "mppi.noise",
            MPPI_config.noise,
            descriptor=ParameterDescriptor(
                description=(
                    "Sampling noise as one shared scalar or one value per "
                    "action dimension."
                ),
                dynamic_typing=True,
            ),
        )
        self.declare_parameter(
            "mppi.noise_annealing_coefficient",
            MPPI_config.noise_annealing_coefficient,
        )
        self.declare_parameter("mppi.lam", MPPI_config.lam)
        self.declare_parameter("mppi.horizon", MPPI_config.horizon)
        self.declare_parameter("mppi.gains", MPPI_config.gains)
        self.declare_parameter("mppi.terminal_gains", MPPI_config.terminal_gains)
        self.declare_parameter("mppi.cost_decay", MPPI_config.cost_decay)
        self.declare_parameter("mppi.act_dim", MPPI_config.act_dim)
        self.declare_parameter("mppi.obs_dim", MPPI_config.obs_dim)
        self.declare_parameter("mppi.action_history_len", MPPI_config.action_history_len)
        self.declare_parameter("mppi.n_iters", MPPI_config.n_iters)
        act_dim = int(self.get_parameter("mppi.act_dim").value)
        noise = normalize_mppi_noise(
            self.get_parameter("mppi.noise").value,
            act_dim,
        )
        MPPI_config = MPPIActionHistoryConfig(
            horizon=int(self.get_parameter("mppi.horizon").value),
            dt=MPPI_config.dt,
            num_samples=int(self.get_parameter("mppi.num_samples").value),
            act_dim=act_dim,
            obs_dim=int(self.get_parameter("mppi.obs_dim").value),
            action_history_len=int(self.get_parameter("mppi.action_history_len").value),
            act_bounds=MPPI_config.act_bounds,
            noise=noise,
            noise_annealing_coefficient=float(
                self.get_parameter("mppi.noise_annealing_coefficient").value
            ),
            lam=float(self.get_parameter("mppi.lam").value),
            spline_order=MPPI_config.spline_order,
            n_knots=MPPI_config.n_knots,
            gains=self.get_parameter("mppi.gains").value,
            terminal_gains=self.get_parameter("mppi.terminal_gains").value,
            cost_decay=float(self.get_parameter("mppi.cost_decay").value),
            n_iters=int(self.get_parameter("mppi.n_iters").value),
        )

        # model params
        # model.path remains the base checkpoint for online fine-tuning.
        # An optional finetuned_model.path bypasses fine-tuning and is used
        # directly for inference.
        self.declare_parameter("model.path", model_config.load_path)
        self.declare_parameter("finetuned_model.path", "")
        self.declare_parameter(
            "model.save_directory",
            "bag/models/loco-manipulation",
        )
        self.declare_parameter("model.input_dim", model_config.input_dim)
        self.declare_parameter("model.output_dim", model_config.output_dim)
        pkg_share = get_package_share_directory("tron1_deploy")
        model_path_param = str(self.get_parameter("model.path").value).strip()
        finetuned_model_path_param = str(
            self.get_parameter("finetuned_model.path").value
        ).strip()
        self.finetuning_enabled = not bool(finetuned_model_path_param)
        model_path_value = finetuned_model_path_param or model_path_param
        if not model_path_value:
            raise ValueError(
                "No model is available. Set model.path or finetuned_model.path."
            )
        expanded_model_path = os.path.expanduser(model_path_value)
        model_path = (
            expanded_model_path
            if os.path.isabs(expanded_model_path)
            else os.path.join(pkg_share, expanded_model_path)
        )
        model_save_directory = os.path.expanduser(
            str(self.get_parameter("model.save_directory").value)
        )
        self.model_save_directory = (
            model_save_directory
            if os.path.isabs(model_save_directory)
            else os.path.join(pkg_share, model_save_directory)
        )
        model_input_dim = int(self.get_parameter("model.input_dim").value)
        model_output_dim = int(self.get_parameter("model.output_dim").value)
        model_config = Tron1ModelConfig(
            input_dim=model_input_dim,
            output_dim=model_output_dim,
            hidden_sizes=model_config.hidden_sizes,
            activation=model_config.activation,
            key=model_config.key,
            type=model_config.type,
            obs_history_dim=model_config.obs_history_dim,
            inference_mode=(
                True
                if finetuned_model_path_param
                else model_config.inference_mode
            ),
            lambda_c=model_config.lambda_c,
            lambda_d=model_config.lambda_d,
            input_range=tuple((-1.0, 1.0) for _ in range(model_input_dim)),
            output_range=tuple((-1.0, 1.0) for _ in range(model_output_dim)),
            load_path=model_path,
            shuffle_length=model_config.shuffle_length,
            lambda_r=model_config.lambda_r,
            regressive_loss_discount=model_config.regressive_loss_discount,
        )
        self.get_logger().info(f"Load model from path: {model_config.load_path}")
        if self.finetuning_enabled:
            self.get_logger().info(
                "Using model.path as the base checkpoint; online fine-tuning is enabled."
            )
        else:
            self.get_logger().info(
                "A finetuned_model.path parameter was provided; using it "
                "directly with online fine-tuning disabled."
            )

        self.model = Tron1Model(model_config)
        self.mppi = MPPIActionHistory(self.model, MPPI_config)
        self.key = jax.random.PRNGKey(MPPI_config.seed)
        self.get_logger().info("Action Dimension of MPPI Controller: {}".format(MPPI_config.act_dim))

        # data
        self.reference = jnp.zeros((1, MPPI_config.horizon, MPPI_config.act_dim))  # deltarized reference: (1, horizon, command_dim) # absolute reference: (1, horizon+1, command_dim)
        self.act = jnp.zeros(MPPI_config.act_dim)
        self.act_hist = jnp.zeros(MPPI_config.action_history_len * MPPI_config.act_dim)
        self.base_obs_dim = MPPI_config.obs_dim
        self.base_obs = jnp.zeros(self.base_obs_dim)
        self.obs = jnp.concatenate([self.base_obs, self.act_hist])

        expected_input_dim = MPPI_config.obs_dim + (MPPI_config.action_history_len + 1) * MPPI_config.act_dim
        if model_config.input_dim != expected_input_dim:
            raise ValueError(
                f"Model input dimension {model_config.input_dim} should be "
                f"obs_dim + (action_history_len + 1) * act_dim = {expected_input_dim}."
            )

        self.last_update_time = self.get_clock().now().nanoseconds * 1e-9  # seconds
        self.update_period = 0.01  # seconds, adjust as needed

        self.training_cfg = None
        self.trainer = None
        self.finetune_update = None
        self.input_buff = None
        self.output_buff = None
        self.finetune_nums = 0
        self.finetune_updates = 0
        self.model_cleanup_complete = False

        if self.finetuning_enabled:
            self.training_cfg = ReptileTrainerConfig(
                batch=64,
                learning_rate=1e-4,
                weight_decay=0.0,
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
                inner_steps=5,
            )

            self.trainer = JAXTrainer(self.model, self.training_cfg)
            # Precompile the JIT-compatible numerical core of trainer.finetune().
            # finetune() itself constructs a PyTorch DataLoader and cannot be jitted.
            self.finetune_shuffle_length = 8
            if self.training_cfg.batch % self.finetune_shuffle_length != 0:
                raise ValueError(
                    "Finetune batch size must be divisible by shuffle length."
                )
            dummy_inputs = jnp.zeros(
                (self.training_cfg.batch, model_config.input_dim), dtype=jnp.float32
            )
            dummy_targets = jnp.zeros(
                (self.training_cfg.batch, model_config.output_dim), dtype=jnp.float32
            )
            self.get_logger().info("Precompiling fine-tuning update...")
            self.finetune_update = jitted_update.lower(
                self.model, dummy_inputs, dummy_targets, self.trainer.optimizer
            ).compile()
            self.get_logger().info("Fine-tuning update precompiled.")

            buffer_size = self.training_cfg.batch * self.training_cfg.inner_steps
            self.input_buff = jnp.zeros(
                (buffer_size, model_config.input_dim)
            )
            self.output_buff = jnp.zeros(
                (buffer_size, model_config.output_dim)
            )
            self.finetune_nums = 100 # 70

        self.optimal_solution = jnp.zeros((MPPI_config.horizon, MPPI_config.act_dim))
        self.compiled_mppi_act = None
        self.jitted_command_prediction = jax.jit(self.command_and_prediction)

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
        self.timer = self.create_timer(0.02, self.compute)  # 50Hz

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    def _report_model_cleanup(self, level, message, use_ros_logger):
        if use_ros_logger:
            getattr(self.get_logger(), level)(message)
        else:
            print(f"[tron1_mppi] [{level.upper()}] {message}", flush=True)

    def finalize_finetuned_model(
        self, use_ros_logger=True
    ) -> TransitionCallbackReturn:
        if self.model_cleanup_complete:
            return TransitionCallbackReturn.SUCCESS

        if not self.finetuning_enabled:
            self._report_model_cleanup(
                "info",
                "Skipping model save: finetuned_model.path was supplied and "
                "fine-tuning was disabled.",
                use_ros_logger,
            )
            self.model_cleanup_complete = True
            return TransitionCallbackReturn.SUCCESS

        if self.finetune_updates == 0:
            self._report_model_cleanup(
                "info",
                "Skipping model save: no fine-tuning updates were applied.",
                use_ros_logger,
            )
            self.model_cleanup_complete = True
            return TransitionCallbackReturn.SUCCESS

        try:
            os.makedirs(self.model_save_directory, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            save_path = os.path.join(
                self.model_save_directory,
                f"reptile_finetuned_{timestamp}.npz",
            )
            self.model.save(save_path)
            self._report_model_cleanup(
                "info",
                f"Saved finetuned model: {save_path}",
                use_ros_logger,
            )
            self.model_cleanup_complete = True
            return TransitionCallbackReturn.SUCCESS
        except Exception as error:
            self._report_model_cleanup(
                "error",
                f"Failed to save finetuned model on cleanup: {error}",
                use_ros_logger,
            )
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        return self.finalize_finetuned_model()

    def on_shutdown(self, state) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    def get_observation_callback(self, msg: Float64MultiArray):
        array = jnp.array(msg.data, dtype=jnp.float32)
        # structure of array: reference trajectory in current base frame
        assert array.shape[0] == (self.mppi.horizon+1) * self.mppi.act_dim, f"Received reference has incorrect shape: {array.shape}, expected {((self.mppi.horizon + 1) * self.mppi.act_dim,)}"

        if self.finetuning_enabled and self.finetune_nums > 0:
            self.input_buff = jnp.roll(self.input_buff, shift=-1, axis=0)
            self.output_buff = jnp.roll(self.output_buff, shift=-1, axis=0)
            self.input_buff = self.input_buff.at[-1].set(
                jnp.concatenate([self.obs, self.act])
            )
            self.output_buff = self.output_buff.at[-1].set(
                array[0 : self.mppi.act_dim]
            )

        self.base_obs = jnp.concatenate([
            self.base_obs[self.mppi.act_dim:],
            array[0:self.mppi.act_dim],
        ])
        self.act_hist = jnp.concatenate([
            self.act_hist[self.mppi.act_dim:],
            self.act,
        ])
        self.obs = jnp.concatenate([self.base_obs, self.act_hist])

        self.reference = self.reference.at[0,:].set(array[self.mppi.act_dim:].reshape(self.mppi.horizon, self.mppi.act_dim))

    def joy_callback(self, msg: Joy):
        if not self.finetuning_enabled:
            return
        if (msg.buttons[self.mode_button_] == 1) and (self.prev_mode_ == 0):
            self.timer2 = self.create_timer(0.5, self.finetuning)
            self.prev_mode_ = 1

    def command_and_prediction(self, obs, new_command):
        model_input = jnp.concatenate([obs, new_command])
        model_input_norm = self.model.input_normalize(model_input)
        pred_obs_norm = self.model.forward(model_input_norm)
        pred_obs = self.model.output_denormalize(pred_obs_norm)
        return jnp.concatenate([new_command.reshape(-1), pred_obs.reshape(-1)])

    def compute(self):
        # publish
        msg = Float64MultiArray()

        if self.finetune_nums > 0:
            # Publish the original non-delta reference command during warmup.
            msg.data = self.reference[0, 0, :].flatten().tolist()
            self.act = self.reference[0, 0, :]
            self.optimal_solution = self.reference[0]
        else:
            new_command, self.key, total_cost = self.mppi.act(self.obs, self.reference, self.key, self.reference[0])
            # new_command, self.key, total_cost = self.mppi.act(
            #     self.obs,
            #     self.reference,
            #     self.key,
            #     self.optimal_solution,
            # )
            # self.optimal_solution = jnp.roll(self.optimal_solution, shift=-1, axis=0)
            # self.optimal_solution = self.optimal_solution.at[-1].set(self.reference[0, -1])
            # self.optimal_solution = self.optimal_solution.at[0].set(new_command)

            command_and_prediction = self.jitted_command_prediction(
                self.obs, new_command
            )
            msg.data = command_and_prediction.tolist()
            self.act = new_command

        self.publisher_.publish(msg)

    def finetuning(self):
        if self.finetune_nums <= 0:
            return

        # Match trainer.finetune(..., shuffle_length=8): shuffle complete
        # temporal chunks, then use one batch while retaining order per chunk.
        num_chunks = len(self.input_buff) // self.finetune_shuffle_length
        chunks_per_batch = (
            self.training_cfg.batch // self.finetune_shuffle_length
        )
        chunk_indices = np.random.permutation(num_chunks)[:chunks_per_batch]
        offsets = np.arange(self.finetune_shuffle_length)
        batch_indices = (
            chunk_indices[:, None] * self.finetune_shuffle_length + offsets
        ).reshape(-1)
        batch_inputs = self.input_buff[batch_indices]
        batch_targets = self.output_buff[batch_indices]

        loss_val, _ = self.finetune_update(
            self.model, batch_inputs, batch_targets, self.trainer.optimizer
        )
        self.get_logger().info(
            f"Finetune Batch Loss: {loss_val.item():.6f}"
        )
        self.finetune_nums -= 1
        self.finetune_updates += 1

        if self.finetune_nums == 0:
            if self.model.inference_mode is False:
                self.model.set_inference_mode(True)

            self.mppi.update_model(self.model)

    def predict(self):
        pass


def main(args=None):
    mppi_cfg = MPPIActionHistoryConfig(
        horizon=20,
        dt=0.02,
        num_samples=5000,
        act_dim=7,
        obs_dim=70,
        action_history_len=10,
        act_bounds=[[-2.0], [2.0]],
        noise=0.1, # 0.3
        lam=0.01,# 0.1, 0.05
        spline_order=3,
        n_knots=3, # 6
        # gains=[20.0, 0.1, 0.2],
        gains=[0.0, 0.5, 0.5],
        terminal_gains=[0.0, 0.0, 0.0],
        cost_decay=0.5, # 0.95
        n_iters=1,
    )
    pkg_share = get_package_share_directory("tron1_deploy")
    model_path = os.path.join(
        pkg_share, "model", "pose_tracking", "reptile.npz"
    )
    model_cfg = Tron1ModelConfig(
        input_dim=147,
        output_dim=7,
        hidden_sizes=[32, 64, 256, 64, 32],
        activation="mish",
        key=42,
        type="mlp",
        obs_history_dim=70,
        inference_mode=False,
        lambda_c=0.0,
        lambda_d=0.0,
        input_range=tuple((-1.0, 1.0) for _ in range(147)),
        output_range=tuple((-1.0, 1.0) for _ in range(7)),
        load_path=model_path,
        shuffle_length=8,
        lambda_r=1.0,
        regressive_loss_discount=0.9,
    )

    rclpy.init(args=args)
    node_name = MPPIControlNode(model_cfg, mppi_cfg)
    configure_result = node_name.trigger_configure()
    if configure_result != TransitionCallbackReturn.SUCCESS:
        node_name.get_logger().error(f"Failed to configure node: {configure_result}")
        node_name.destroy_node()
        rclpy.shutdown()
        return

    activate_result = node_name.trigger_activate()
    if activate_result != TransitionCallbackReturn.SUCCESS:
        node_name.get_logger().error(f"Failed to activate node: {activate_result}")
        node_name.destroy_node()
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node_name)
    except KeyboardInterrupt:
        shutdown_message = "KeyboardInterrupt received, finalizing model..."
        if rclpy.ok():
            node_name.get_logger().info(shutdown_message)
        else:
            print(f"[tron1_mppi] [INFO] {shutdown_message}", flush=True)
    finally:
        if rclpy.ok():
            try:
                node_name.trigger_deactivate()
                node_name.trigger_cleanup()
            except Exception as error:
                node_name.get_logger().error(
                    f"Lifecycle cleanup failed; finalizing model directly: {error}"
                )
                node_name.finalize_finetuned_model()
        else:
            # The ROS signal handler invalidates the context before spin exits.
            # Lifecycle transitions cannot run then, but model persistence does
            # not depend on the lifecycle state machine or a valid ROS context.
            node_name.finalize_finetuned_model(use_ros_logger=False)

        node_name.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
