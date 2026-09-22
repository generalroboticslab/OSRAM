from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mjlab.entity import Entity
from mjlab.envs.mdp.actions.differential_ik import (
  DifferentialIKAction,
  DifferentialIKActionCfg,
)
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  compute_pose_error,
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_box_plus,
  quat_from_euler_xyz,
  quat_from_matrix,
  quat_inv,
  quat_mul,
  sample_uniform,
  wrap_to_pi,
  yaw_quat,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  import viser
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformPose2dCommand(CommandTerm):
  """Command generator that generates pose commands containing a 3-D position and heading.

  The command generator samples uniform 2D positions around the environment origin. It sets
  the height of the position command to the default root height of the robot. The heading
  command is either set to point towards the target or is sampled uniformly.
  This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
  the configuration.
  """

  cfg: UniformPose2dCommandCfg
  """Configuration for the command generator."""

  def __init__(self, cfg: UniformPose2dCommandCfg, env: ManagerBasedRlEnv):
    """Initialize the command generator class.

    Args:
        cfg: The configuration parameters for the command generator.
        env: The environment object.
    """
    # initialize the base class
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    # if self.cfg.ranges.heading and not self.cfg.heading_command:
    #   raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    # create buffers to store the command
    # -- commands: (x, y, z, heading)
    self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
    self.pos_command_b = torch.zeros_like(self.pos_command_w)
    self.heading_command_b = torch.zeros_like(self.heading_command_w)
    # -- metrics
    self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

  def __str__(self) -> str:
    msg = "PositionCommand:\n"
    msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
    msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
    return msg

  """
    Properties
    """

  @property
  def command(self) -> torch.Tensor:
    """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
    return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

  """
    Implementation specific functions.
    """

  def _update_metrics(self):
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_pos_2d"] += (
      torch.norm(
        self.pos_command_w[:, :2] - self.robot.data.root_link_pos_w[:, :2], dim=1
      )
      / max_command_step
    )
    self.metrics["error_heading"] += (
      torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor):
    # obtain env origins for the environments
    self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
    # offset the position command by the current root position
    r = torch.empty(len(env_ids), device=self.device)
    self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
    self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
    self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

    if self.cfg.heading_command:
      # set heading command to point towards target
      target_vec = (
        self.pos_command_w[env_ids] - self.robot.data.root_link_pos_w[env_ids]
      )
      target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
      flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

      # compute errors to find the closest direction to the current heading
      # this is done to avoid the discontinuity at the -pi/pi boundary
      curr_to_target = wrap_to_pi(
        target_direction - self.robot.data.heading_w[env_ids]
      ).abs()
      curr_to_flipped_target = wrap_to_pi(
        flipped_target_direction - self.robot.data.heading_w[env_ids]
      ).abs()

      # set the heading command to the closest direction
      self.heading_command_w[env_ids] = torch.where(
        curr_to_target < curr_to_flipped_target,
        target_direction,
        flipped_target_direction,
      )
    else:
      # random heading command
      self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

  def _update_command(self):
    """Re-target the position command to the current root state."""
    target_vec = self.pos_command_w - self.robot.data.root_link_pos_w[:, :3]
    self.pos_command_b[:] = quat_apply_inverse(
      yaw_quat(self.robot.data.root_link_quat_w), target_vec
    )
    self.heading_command_b[:] = wrap_to_pi(
      self.heading_command_w - self.robot.data.heading_w
    )

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """
    Draw the target position and heading arrows.

    Note: Only visualizes the selected environment (visualizer.env_idx).
    """
    batch = visualizer.env_idx

    if batch >= self.num_envs:
      return

    # visualize target position
    target_pos = self.pos_command_w[batch].cpu().numpy()
    visualizer.add_sphere(
      center=target_pos,
      radius=0.03,
      color=(0.2, 0.6, 0.2, 0.6),
      label="target_position",
    )

    # visualize heading direction
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    base_pos_w = base_pos_ws[batch]
    base_mat_w = base_mat_ws[batch]

    target_heading = self.heading_command_b[batch].cpu().item()
    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    def local_to_world(
      vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
    ) -> np.ndarray:
      return pos + mat @ vec

    # Arrow starts at the base position with z offset
    cmd_ang_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    # Arrow points in the heading direction in the xy plane
    heading_vec = (
      np.array([np.cos(target_heading), np.sin(target_heading), 0.0]) * scale
    )
    cmd_ang_to = local_to_world(np.array([0, 0, z_offset]) * scale + heading_vec)

    visualizer.add_arrow(
      cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
    )


@dataclass(kw_only=True)
class UniformPose2dCommandCfg(CommandTermCfg):
  """Configuration for the uniform 2D-pose command generator."""

  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0

  @dataclass
  class Ranges:
    """Uniform distribution ranges for the position commands."""

    pos_x: tuple[float, float]
    pos_y: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformPose2dCommand:
    return UniformPose2dCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )


class CircularPose2dCommand(CommandTerm):
  """Command generator that tracks a circular pose trajectory in world frame."""

  cfg: CircularPose2dCommandCfg

  def __init__(self, cfg: CircularPose2dCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
    self.pos_command_b = torch.zeros_like(self.pos_command_w)
    self.heading_command_b = torch.zeros_like(self.heading_command_w)

    self.circle_center_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.theta0 = torch.zeros(self.num_envs, device=self.device)
    self.elapsed_time = torch.zeros(self.num_envs, device=self.device)

    self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

  def _update_metrics(self):
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_pos_2d"] += (
      torch.norm(
        self.pos_command_w[:, :2] - self.robot.data.root_link_pos_w[:, :2], dim=1
      )
      / max_command_step
    )
    self.metrics["error_heading"] += (
      torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor):
    root_pos_w = self.robot.data.root_link_pos_w[env_ids]
    heading_w = self.robot.data.heading_w[env_ids]

    theta_init = heading_w - (torch.pi / 2.0)
    self.circle_center_w[env_ids] = root_pos_w[:, :2] - self.cfg.radius * torch.stack(
      [torch.cos(theta_init), torch.sin(theta_init)], dim=1
    )

    rel_pos = root_pos_w[:, :2] - self.circle_center_w[env_ids]
    self.theta0[env_ids] = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
    self.elapsed_time[env_ids] = 0.0

  def _update_command(self):
    self.elapsed_time += self._env.step_dt
    theta = self.theta0 + self.cfg.angular_speed * self.elapsed_time

    self.pos_command_w[:, 0] = self.circle_center_w[:, 0] + self.cfg.radius * torch.cos(
      theta
    )
    self.pos_command_w[:, 1] = self.circle_center_w[:, 1] + self.cfg.radius * torch.sin(
      theta
    )
    self.pos_command_w[:, 2] = self.robot.data.root_link_pos_w[:, 2]
    self.heading_command_w[:] = wrap_to_pi(theta + (torch.pi / 2.0))

    target_vec = self.pos_command_w - self.robot.data.root_link_pos_w[:, :3]
    self.pos_command_b[:] = quat_apply_inverse(
      yaw_quat(self.robot.data.root_link_quat_w), target_vec
    )
    self.heading_command_b[:] = wrap_to_pi(
      self.heading_command_w - self.robot.data.heading_w
    )

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    target_pos = self.pos_command_w[batch].cpu().numpy()
    visualizer.add_sphere(
      center=target_pos,
      radius=0.03,
      color=(0.2, 0.6, 0.2, 0.6),
      label="target_position",
    )

    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    base_pos_w = base_pos_ws[batch]
    base_mat_w = base_mat_ws[batch]

    target_heading = self.heading_command_b[batch].cpu().item()
    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    def local_to_world(
      vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
    ) -> np.ndarray:
      return pos + mat @ vec

    cmd_ang_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    heading_vec = (
      np.array([np.cos(target_heading), np.sin(target_heading), 0.0]) * scale
    )
    cmd_ang_to = local_to_world(np.array([0, 0, z_offset]) * scale + heading_vec)
    visualizer.add_arrow(
      cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
    )


@dataclass(kw_only=True)
class CircularPose2dCommandCfg(CommandTermCfg):
  """Configuration for circular 2D pose command generation."""

  entity_name: str
  radius: float = 0.6
  angular_speed: float = 0.3

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> CircularPose2dCommand:
    return CircularPose2dCommand(self, env)


def _normalize_unique_quat(quat: torch.Tensor) -> torch.Tensor:
  quat = torch.nn.functional.normalize(quat, p=2, dim=-1, eps=1e-8)
  return torch.where(quat[..., :1] < 0.0, -quat, quat)


def _skew_matrix(vector: torch.Tensor) -> torch.Tensor:
  x, y, z = vector.unbind(dim=-1)
  zeros = torch.zeros_like(x)
  return torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1).reshape(
    *vector.shape[:-1], 3, 3
  )


def _se3_exp(twist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Return ``Exp([linear, angular])`` as position and wxyz quaternion."""
  linear = twist[..., :3]
  angular = twist[..., 3:]
  theta_sq = torch.sum(torch.square(angular), dim=-1, keepdim=True)
  theta = torch.sqrt(theta_sq)
  small = theta_sq < 1e-8
  safe_theta = torch.clamp(theta, min=1e-8)
  safe_theta_sq = torch.clamp(theta_sq, min=1e-8)

  a_exact = (1.0 - torch.cos(theta)) / safe_theta_sq
  b_exact = (theta - torch.sin(theta)) / (safe_theta_sq * safe_theta)
  a_series = 0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0
  b_series = 1.0 / 6.0 - theta_sq / 120.0 + theta_sq * theta_sq / 5040.0
  a = torch.where(small, a_series, a_exact)[..., None]
  b = torch.where(small, b_series, b_exact)[..., None]

  skew = _skew_matrix(angular)
  identity = torch.eye(3, dtype=twist.dtype, device=twist.device).expand(
    *twist.shape[:-1], 3, 3
  )
  left_jacobian = identity + a * skew + b * torch.matmul(skew, skew)
  position = torch.matmul(left_jacobian, linear.unsqueeze(-1)).squeeze(-1)

  half_theta = 0.5 * theta
  scale_exact = torch.sin(half_theta) / safe_theta
  scale_series = 0.5 - theta_sq / 48.0 + theta_sq * theta_sq / 3840.0
  scale = torch.where(small, scale_series, scale_exact)
  quat = torch.cat((torch.cos(half_theta), scale * angular), dim=-1)
  return position, _normalize_unique_quat(quat)


def _se3_log(position: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
  """Return the Pinocchio-compatible ``Log`` vector ordered linear then angular."""
  quat = _normalize_unique_quat(quat)
  angular = axis_angle_from_quat(quat)
  theta_sq = torch.sum(torch.square(angular), dim=-1, keepdim=True)
  theta = torch.sqrt(theta_sq)
  small = theta_sq < 1e-8
  safe_theta_sq = torch.clamp(theta_sq, min=1e-8)
  safe_half_tan = torch.clamp(torch.abs(torch.tan(0.5 * theta)), min=1e-8)
  c_exact = (1.0 - 0.5 * theta / safe_half_tan) / safe_theta_sq
  c_series = 1.0 / 12.0 + theta_sq / 720.0 + theta_sq * theta_sq / 30240.0
  c = torch.where(small, c_series, c_exact)[..., None]

  skew = _skew_matrix(angular)
  identity = torch.eye(3, dtype=position.dtype, device=position.device).expand(
    *position.shape[:-1], 3, 3
  )
  left_jacobian_inv = identity - 0.5 * skew + c * torch.matmul(skew, skew)
  linear = torch.matmul(left_jacobian_inv, position.unsqueeze(-1)).squeeze(-1)
  return torch.cat((linear, angular), dim=-1)


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
  quat = _normalize_unique_quat(quat)
  w, x, y, z = quat.unbind(dim=-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _generate_tron1_with_arm_commands(
  current_base_pos_w: torch.Tensor,
  current_base_quat_w: torch.Tensor,
  desired_ee_pos_w: torch.Tensor,
  desired_ee_quat_w: torch.Tensor,
  nominal_ee_pos_b: torch.Tensor,
  nominal_ee_quat_b: torch.Tensor,
  k_xy: torch.Tensor,
  vxy_max: float,
  k_yaw: float,
  wz_max: float,
  k_arm: torch.Tensor,
  xi_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Generate coupled EE/base commands using the Tron1 waypoint algorithm."""
  current_base_quat_w = _normalize_unique_quat(current_base_quat_w)
  desired_ee_quat_w = _normalize_unique_quat(desired_ee_quat_w)
  nominal_ee_quat_b = _normalize_unique_quat(nominal_ee_quat_b)

  nominal_quat_inv = quat_inv(nominal_ee_quat_b)
  nominal_inverse_pos = quat_apply(nominal_quat_inv, -nominal_ee_pos_b)
  desired_base_pos_w = desired_ee_pos_w + quat_apply(
    desired_ee_quat_w, nominal_inverse_pos
  )
  desired_base_quat_w = _normalize_unique_quat(
    quat_mul(desired_ee_quat_w, nominal_quat_inv)
  )

  xy_error_w = desired_base_pos_w[:, :2] - current_base_pos_w[:, :2]
  velocity_w = k_xy * xy_error_w
  velocity_norm = torch.linalg.vector_norm(velocity_w, dim=-1, keepdim=True)
  velocity_scale = torch.clamp(vxy_max / torch.clamp(velocity_norm, min=1e-8), max=1.0)
  velocity_w = velocity_w * velocity_scale

  current_yaw = _yaw_from_quat(current_base_quat_w)
  desired_yaw = _yaw_from_quat(desired_base_quat_w)
  cosine_yaw = torch.cos(current_yaw)
  sine_yaw = torch.sin(current_yaw)
  velocity_b = torch.stack(
    (
      cosine_yaw * velocity_w[:, 0] + sine_yaw * velocity_w[:, 1],
      -sine_yaw * velocity_w[:, 0] + cosine_yaw * velocity_w[:, 1],
    ),
    dim=-1,
  )
  yaw_error = wrap_to_pi(desired_yaw - current_yaw)
  yaw_rate = torch.clamp(k_yaw * yaw_error, -wz_max, wz_max)

  current_base_quat_inv = quat_inv(current_base_quat_w)
  desired_ee_pos_b = quat_apply_inverse(
    current_base_quat_w, desired_ee_pos_w - current_base_pos_w
  )
  desired_ee_quat_b = _normalize_unique_quat(
    quat_mul(current_base_quat_inv, desired_ee_quat_w)
  )
  residual_pos = quat_apply(nominal_quat_inv, desired_ee_pos_b - nominal_ee_pos_b)
  residual_quat = _normalize_unique_quat(quat_mul(nominal_quat_inv, desired_ee_quat_b))
  residual = _se3_log(residual_pos, residual_quat)
  bounded_residual = torch.clamp(k_arm * residual, min=-xi_max, max=xi_max)
  command_offset_pos, command_offset_quat = _se3_exp(bounded_residual)
  commanded_ee_pos_b = nominal_ee_pos_b + quat_apply(
    nominal_ee_quat_b, command_offset_pos
  )
  commanded_ee_quat_b = _normalize_unique_quat(
    quat_mul(nominal_ee_quat_b, command_offset_quat)
  )

  command = torch.cat(
    (commanded_ee_pos_b, commanded_ee_quat_b, velocity_b, yaw_rate[:, None]),
    dim=-1,
  )
  return command, residual, xy_error_w, yaw_error


class UniformPoseCommand(CommandTerm):
  cfg: UniformPoseCommandCfg

  def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    body_ids, _ = self.robot.find_bodies(cfg.base_body_name)
    self._base_body_id = body_ids[0]
    if cfg.ee_frame_type == "site":
      site_ids, _ = self.robot.find_sites(cfg.ee_frame_name)
      self._ee_frame_id = site_ids[0]
      self._ee_is_site = True
    else:
      body_ids, _ = self.robot.find_bodies(cfg.ee_frame_name)
      self._ee_frame_id = body_ids[0]
      self._ee_is_site = False

    # Target sampled in base-link frame or world frame depending on cfg.sample_frame.
    self.target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_quat_b = torch.zeros(self.num_envs, 4, device=self.device)
    self.target_quat_b[:, 0] = 1.0
    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
    self.target_quat_w[:, 0] = 1.0
    # Command exposed to policy/action:
    # - delta_mode=True: [delta_pos, delta_rot_axis_angle] (6D)
    # - delta_mode=False: [target_pos_b, target_quat_b] (7D)
    self.delta_pose_command = torch.zeros(self.num_envs, 6, device=self.device)
    self.target_pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
    self.target_pose_command_b[:, 3] = 1.0
    self.target_pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
    self.target_pose_command_w[:, 3] = 1.0

    self.metrics["ee_pos_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["ee_rot_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    if self.cfg.delta_mode:
      return self.delta_pose_command
    if self.cfg.output_frame == "world":
      return self.target_pose_command_w
    return self.target_pose_command_b

  def _update_metrics(self) -> None:
    if self._ee_is_site:
      ee_pos = self.robot.data.site_pos_w[:, self._ee_frame_id]
      ee_quat = self.robot.data.site_quat_w[:, self._ee_frame_id]
    else:
      ee_pos = self.robot.data.body_link_pos_w[:, self._ee_frame_id]
      ee_quat = self.robot.data.body_link_quat_w[:, self._ee_frame_id]
    self.metrics["ee_pos_error"] = torch.norm(self.target_pos_w - ee_pos, dim=-1)
    _, rot_err = compute_pose_error(
      ee_pos,
      ee_quat,
      self.target_pos_w,
      self.target_quat_w,
      rot_error_type="axis_angle",
    )
    self.metrics["ee_rot_error"] = torch.linalg.vector_norm(rot_err, dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return

    r = self.cfg.ranges
    lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    roll = sample_uniform(r.roll[0], r.roll[1], (n,), device=self.device)
    pitch = sample_uniform(r.pitch[0], r.pitch[1], (n,), device=self.device)
    yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
    target_quat = quat_from_euler_xyz(roll, pitch, yaw)
    target_quat = _normalize_unique_quat(target_quat)

    sampled_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
    if self.cfg.sample_frame == "base":
      self.target_pos_b[env_ids] = sampled_pos
      self.target_quat_b[env_ids] = target_quat
    elif self.cfg.sample_frame == "world":
      base_pos_w = self.robot.data.body_link_pos_w[env_ids, self._base_body_id]
      base_quat_w = self.robot.data.body_link_quat_w[env_ids, self._base_body_id]
      yaw_base_quat_w = yaw_quat(base_quat_w)
      self.target_pos_w[env_ids] = base_pos_w + quat_apply(yaw_base_quat_w, sampled_pos)
      self.target_quat_w[env_ids] = _normalize_unique_quat(
        quat_mul(yaw_base_quat_w, target_quat)
      )
      if self.cfg.world_target_position_noise_std > 0.0:
        self.target_pos_w[env_ids] += (
          self.cfg.world_target_position_noise_std
          * torch.randn(n, 3, device=self.device, dtype=self.target_pos_w.dtype)
        )
      if self.cfg.world_target_orientation_noise_std > 0.0:
        rotvec_w = self.cfg.world_target_orientation_noise_std * torch.randn(
          n, 3, device=self.device, dtype=self.target_quat_w.dtype
        )
        _, noise_quat_w = _se3_exp(
          torch.cat((torch.zeros_like(rotvec_w), rotvec_w), dim=-1)
        )
        self.target_quat_w[env_ids] = _normalize_unique_quat(
          quat_mul(noise_quat_w, self.target_quat_w[env_ids])
        )
    else:
      raise ValueError(f"Unknown sample_frame: {self.cfg.sample_frame}")

  def _update_command(self) -> None:
    base_pos_w = self.robot.data.body_link_pos_w[:, self._base_body_id]
    base_quat_w = self.robot.data.body_link_quat_w[:, self._base_body_id]
    if self._ee_is_site:
      ee_pos_w = self.robot.data.site_pos_w[:, self._ee_frame_id]
      ee_quat_w = self.robot.data.site_quat_w[:, self._ee_frame_id]
    else:
      ee_pos_w = self.robot.data.body_link_pos_w[:, self._ee_frame_id]
      ee_quat_w = self.robot.data.body_link_quat_w[:, self._ee_frame_id]

    if self.cfg.sample_frame == "base":
      self.target_pos_w[:] = base_pos_w + quat_apply(base_quat_w, self.target_pos_b)
      self.target_quat_w[:] = _normalize_unique_quat(
        quat_mul(base_quat_w, self.target_quat_b)
      )
    elif self.cfg.sample_frame == "world":
      self.target_pos_b[:] = quat_apply_inverse(
        base_quat_w, self.target_pos_w - base_pos_w
      )
      self.target_quat_b[:] = quat_mul(quat_inv(base_quat_w), self.target_quat_w)
      self.target_quat_b[:] = torch.nn.functional.normalize(
        self.target_quat_b, p=2, dim=-1, eps=1e-8
      )
    else:
      raise ValueError(f"Unknown sample_frame: {self.cfg.sample_frame}")
    command_target_pos_b, command_target_quat_b = self._apply_command_noise_b(
      self.target_pos_b, self.target_quat_b
    )
    self.target_pose_command_b[:, :3] = command_target_pos_b
    self.target_pose_command_b[:, 3:] = command_target_quat_b
    self.target_pose_command_w[:, :3] = self.target_pos_w
    self.target_pose_command_w[:, 3:] = self.target_quat_w

    command_target_pos_w = base_pos_w + quat_apply(base_quat_w, command_target_pos_b)
    command_target_quat_w = quat_mul(base_quat_w, command_target_quat_b)
    pos_err, rot_err = compute_pose_error(
      ee_pos_w,
      ee_quat_w,
      command_target_pos_w,
      command_target_quat_w,
      rot_error_type="axis_angle",
    )
    self.delta_pose_command[:, :3] = pos_err
    self.delta_pose_command[:, 3:] = rot_err

  def _apply_command_noise_b(
    self, target_pos_b: torch.Tensor, target_quat_b: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    pos_std = float(self.cfg.command_position_noise_std)
    ori_std = float(self.cfg.command_orientation_noise_std)
    if pos_std <= 0.0 and ori_std <= 0.0:
      return target_pos_b, target_quat_b

    command_target_pos_b = target_pos_b
    command_target_quat_b = target_quat_b
    if pos_std > 0.0:
      command_target_pos_b = target_pos_b + pos_std * torch.randn_like(target_pos_b)
    if ori_std > 0.0:
      rotvec = ori_std * torch.randn(
        (target_quat_b.shape[0], 3),
        dtype=target_quat_b.dtype,
        device=target_quat_b.device,
      )
      command_target_quat_b = quat_box_plus(target_quat_b, rotvec)
      command_target_quat_b = torch.nn.functional.normalize(
        command_target_quat_b, p=2, dim=-1, eps=1e-8
      )
    return command_target_pos_b, command_target_quat_b

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    for batch in env_indices:
      if self._ee_is_site:
        ee_pos = self.robot.data.site_pos_w[batch, self._ee_frame_id].cpu().numpy()
        ee_quat = self.robot.data.site_quat_w[batch, self._ee_frame_id]
      else:
        ee_pos = self.robot.data.body_link_pos_w[batch, self._ee_frame_id].cpu().numpy()
        ee_quat = self.robot.data.body_link_quat_w[batch, self._ee_frame_id]
      ee_rotm = matrix_from_quat(ee_quat).cpu().numpy()

      target_pos = self.target_pos_w[batch].cpu().numpy()
      target_rotm = matrix_from_quat(self.target_quat_w[batch]).cpu().numpy()

      visualizer.add_sphere(
        center=target_pos,
        radius=self.cfg.viz.target_radius,
        color=self.cfg.viz.target_color,
        label=f"ee_target_pos_{batch}",
      )
      visualizer.add_frame(
        position=target_pos,
        rotation_matrix=target_rotm,
        scale=self.cfg.viz.frame_scale,
        axis_radius=self.cfg.viz.axis_radius,
        alpha=self.cfg.viz.frame_alpha,
        label=f"ee_target_frame_{batch}",
        axis_colors=self.cfg.viz.axis_colors,
      )
      visualizer.add_sphere(
        center=ee_pos,
        radius=self.cfg.viz.current_radius,
        color=self.cfg.viz.current_color,
        label=f"ee_current_pos_{batch}",
      )
      visualizer.add_frame(
        position=ee_pos,
        rotation_matrix=ee_rotm,
        scale=self.cfg.viz.current_frame_scale,
        axis_radius=self.cfg.viz.current_axis_radius,
        alpha=self.cfg.viz.current_frame_alpha,
        label=f"ee_current_frame_{batch}",
      )


class CurrentPoseOffsetCommand(UniformPoseCommand):
  cfg: CurrentPoseOffsetCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return

    base_pos_w = self.robot.data.body_link_pos_w[env_ids, self._base_body_id]
    base_quat_w = self.robot.data.body_link_quat_w[env_ids, self._base_body_id]
    if self._ee_is_site:
      ee_pos_w = self.robot.data.site_pos_w[env_ids, self._ee_frame_id]
      ee_quat_w = self.robot.data.site_quat_w[env_ids, self._ee_frame_id]
    else:
      ee_pos_w = self.robot.data.body_link_pos_w[env_ids, self._ee_frame_id]
      ee_quat_w = self.robot.data.body_link_quat_w[env_ids, self._ee_frame_id]

    current_pos_b = quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w)
    current_quat_b = quat_mul(quat_inv(base_quat_w), ee_quat_w)

    r = self.cfg.ranges
    lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    pos_offset_b = sample_uniform(lower, upper, (n, 3), device=self.device)

    roll = sample_uniform(r.roll[0], r.roll[1], (n,), device=self.device)
    pitch = sample_uniform(r.pitch[0], r.pitch[1], (n,), device=self.device)
    yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
    quat_offset_b = quat_from_euler_xyz(roll, pitch, yaw)

    self.target_pos_b[env_ids] = current_pos_b + pos_offset_b
    self.target_quat_b[env_ids] = quat_mul(current_quat_b, quat_offset_b)


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
  entity_name: str
  base_body_name: str
  ee_frame_type: Literal["site", "body"] = "site"
  ee_frame_name: str = ""
  delta_mode: bool = True
  sample_frame: Literal["base", "world"] = "base"
  output_frame: Literal["base", "world"] = "base"
  command_position_noise_std: float = 0.0
  command_orientation_noise_std: float = 0.0
  world_target_position_noise_std: float = 0.0
  world_target_orientation_noise_std: float = 0.0

  @dataclass
  class Ranges:
    x: tuple[float, float] = (0.20, 0.50)
    y: tuple[float, float] = (-0.20, 0.20)
    z: tuple[float, float] = (0.10, 0.40)
    roll: tuple[float, float] = (-0.2, 0.2)
    pitch: tuple[float, float] = (-0.2, 0.2)
    yaw: tuple[float, float] = (-3.14, 3.14)

  ranges: Ranges = field(default_factory=Ranges)

  @dataclass
  class VizCfg:
    target_radius: float = 0.02
    frame_scale: float = 0.08
    axis_radius: float = 0.007
    frame_alpha: float = 0.85
    target_color: tuple[float, float, float, float] = (0.9, 0.5, 0.1, 0.75)
    current_radius: float = 0.018
    current_frame_scale: float = 0.1
    current_axis_radius: float = 0.009
    current_frame_alpha: float = 0.95
    current_color: tuple[float, float, float, float] = (0.2, 0.9, 0.3, 0.8)
    axis_colors: tuple[tuple[float, float, float], ...] = (
      (1.0, 0.4, 0.0),
      (0.0, 0.85, 0.35),
      (0.2, 0.6, 1.0),
    )

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformPoseCommand:
    return UniformPoseCommand(self, env)

  def __post_init__(self):
    if not self.ee_frame_name and self.ee_site_name:
      self.ee_frame_name = self.ee_site_name
      self.ee_frame_type = "site"
    if not self.ee_frame_name:
      raise ValueError("`ee_frame_name` must be provided for UniformPoseCommandCfg.")
    if self.delta_mode and self.output_frame != "base":
      raise ValueError("`output_frame='world'` requires `delta_mode=False`.")
    if (
      self.world_target_position_noise_std < 0.0
      or self.world_target_orientation_noise_std < 0.0
    ):
      raise ValueError("World-target noise standard deviations must be non-negative.")
    if self.sample_frame != "world" and (
      self.world_target_position_noise_std > 0.0
      or self.world_target_orientation_noise_std > 0.0
    ):
      raise ValueError("World-target noise requires `sample_frame='world'`.")


@dataclass(kw_only=True)
class CurrentPoseOffsetCommandCfg(UniformPoseCommandCfg):
  """Sample an EE target pose around the current EE pose in the base frame."""

  def build(self, env: ManagerBasedRlEnv) -> CurrentPoseOffsetCommand:
    return CurrentPoseOffsetCommand(self, env)


class Tron1WithArmPlannerCommand(CommandTerm):
  """Convert a world-frame EE target into coupled arm-pose and base commands."""

  cfg: Tron1WithArmPlannerCommandCfg

  def __init__(self, cfg: Tron1WithArmPlannerCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self._base_body_id = self.robot.find_bodies(cfg.base_body_name)[0][0]

    nominal_pos = torch.tensor(
      cfg.nominal_ee_pos_b, dtype=torch.float, device=self.device
    )
    nominal_rpy = torch.tensor(
      cfg.nominal_ee_rpy_b, dtype=torch.float, device=self.device
    )
    nominal_quat = quat_from_euler_xyz(
      nominal_rpy[0:1], nominal_rpy[1:2], nominal_rpy[2:3]
    )[0]
    self._nominal_pos_b = nominal_pos.expand(self.num_envs, -1)
    self._nominal_quat_b = nominal_quat.expand(self.num_envs, -1)
    self._k_xy = torch.tensor(cfg.k_xy, dtype=torch.float, device=self.device)
    self._k_arm = torch.tensor(cfg.k_arm, dtype=torch.float, device=self.device)
    self._xi_max = torch.tensor(cfg.xi_max, dtype=torch.float, device=self.device)

    self._command = torch.zeros(self.num_envs, 10, device=self.device)
    self._command[:, 3] = 1.0
    self.metrics["base_xy_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["base_yaw_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["arm_residual_norm"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def _update_command(self) -> None:
    world_target = self._env.command_manager.get_command(self.cfg.source_command_name)
    if world_target.shape[-1] != 7:
      raise ValueError(
        f"Command '{self.cfg.source_command_name}' must be a 7D world pose, "
        f"got shape {tuple(world_target.shape)}."
      )

    desired_ee_pos_w = world_target[:, :3]
    desired_ee_quat_w = _normalize_unique_quat(world_target[:, 3:7])
    current_base_pos_w = self.robot.data.body_link_pos_w[:, self._base_body_id]
    current_base_quat_w = _normalize_unique_quat(
      self.robot.data.body_link_quat_w[:, self._base_body_id]
    )

    command, residual, xy_error_w, yaw_error = _generate_tron1_with_arm_commands(
      current_base_pos_w=current_base_pos_w,
      current_base_quat_w=current_base_quat_w,
      desired_ee_pos_w=desired_ee_pos_w,
      desired_ee_quat_w=desired_ee_quat_w,
      nominal_ee_pos_b=self._nominal_pos_b,
      nominal_ee_quat_b=self._nominal_quat_b,
      k_xy=self._k_xy,
      vxy_max=self.cfg.vxy_max,
      k_yaw=self.cfg.k_yaw,
      wz_max=self.cfg.wz_max,
      k_arm=self._k_arm,
      xi_max=self._xi_max,
    )
    self._command[:] = command
    self.metrics["base_xy_error"][:] = torch.linalg.vector_norm(xy_error_w, dim=-1)
    self.metrics["base_yaw_error"][:] = torch.abs(yaw_error)
    self.metrics["arm_residual_norm"][:] = torch.linalg.vector_norm(residual, dim=-1)


@dataclass(kw_only=True)
class Tron1WithArmPlannerCommandCfg(CommandTermCfg):
  entity_name: str
  source_command_name: str
  base_body_name: str
  nominal_ee_pos_b: tuple[float, float, float]
  nominal_ee_rpy_b: tuple[float, float, float]
  k_xy: tuple[float, float]
  vxy_max: float
  k_yaw: float
  wz_max: float
  k_arm: tuple[float, float, float, float, float, float]
  xi_max: tuple[float, float, float, float, float, float]

  def build(self, env: ManagerBasedRlEnv) -> Tron1WithArmPlannerCommand:
    return Tron1WithArmPlannerCommand(self, env)

  def __post_init__(self) -> None:
    if self.vxy_max <= 0.0:
      raise ValueError("`vxy_max` must be positive.")
    if self.wz_max <= 0.0:
      raise ValueError("`wz_max` must be positive.")
    if any(limit <= 0.0 for limit in self.xi_max):
      raise ValueError("Every `xi_max` component must be positive.")


class CommandSlice(CommandTerm):
  """Expose a contiguous slice of another command without recomputing it."""

  cfg: CommandSliceCfg

  @property
  def command(self) -> torch.Tensor:
    source = self._env.command_manager.get_command(self.cfg.source_command_name)
    return source[:, self.cfg.start : self.cfg.end]

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def _update_command(self) -> None:
    source = self._env.command_manager.get_command(self.cfg.source_command_name)
    if source.shape[-1] < self.cfg.end:
      raise ValueError(
        f"Command '{self.cfg.source_command_name}' has dim={source.shape[-1]}, "
        f"cannot expose slice [{self.cfg.start}:{self.cfg.end}]."
      )


@dataclass(kw_only=True)
class CommandSliceCfg(CommandTermCfg):
  source_command_name: str
  start: int
  end: int

  def build(self, env: ManagerBasedRlEnv) -> CommandSlice:
    return CommandSlice(self, env)

  def __post_init__(self) -> None:
    if self.start < 0 or self.end <= self.start:
      raise ValueError("Command slice must satisfy 0 <= start < end.")


class CircularPoseCommand(CommandTerm):
  cfg: CircularPoseCommandCfg

  def __init__(self, cfg: CircularPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    body_ids, _ = self.robot.find_bodies(cfg.base_body_name)
    self._base_body_id = body_ids[0]
    if cfg.ee_frame_type == "site":
      site_ids, _ = self.robot.find_sites(cfg.ee_frame_name)
      self._ee_frame_id = site_ids[0]
      self._ee_is_site = True
    else:
      body_ids, _ = self.robot.find_bodies(cfg.ee_frame_name)
      self._ee_frame_id = body_ids[0]
      self._ee_is_site = False

    self.center_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.base_quat_b = torch.zeros(self.num_envs, 4, device=self.device)
    self.base_quat_b[:, 0] = 1.0
    self.target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_quat_b = torch.zeros(self.num_envs, 4, device=self.device)
    self.target_quat_b[:, 0] = 1.0
    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
    self.target_quat_w[:, 0] = 1.0
    self.delta_pose_command = torch.zeros(self.num_envs, 6, device=self.device)

    self.circle_radius = torch.zeros(self.num_envs, device=self.device)
    self.angular_speed = torch.zeros(self.num_envs, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.elapsed_time = torch.zeros(self.num_envs, device=self.device)

    self.metrics["ee_pos_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["ee_rot_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.delta_pose_command

  def _update_metrics(self) -> None:
    if self._ee_is_site:
      ee_pos = self.robot.data.site_pos_w[:, self._ee_frame_id]
      ee_quat = self.robot.data.site_quat_w[:, self._ee_frame_id]
    else:
      ee_pos = self.robot.data.body_link_pos_w[:, self._ee_frame_id]
      ee_quat = self.robot.data.body_link_quat_w[:, self._ee_frame_id]
    self.metrics["ee_pos_error"] = torch.norm(self.target_pos_w - ee_pos, dim=-1)
    _, rot_err = compute_pose_error(
      ee_pos,
      ee_quat,
      self.target_pos_w,
      self.target_quat_w,
      rot_error_type="axis_angle",
    )
    self.metrics["ee_rot_error"] = torch.linalg.vector_norm(rot_err, dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return

    r = self.cfg.ranges
    lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    self.center_pos_b[env_ids] = sample_uniform(
      lower, upper, (n, 3), device=self.device
    )

    roll = sample_uniform(r.roll[0], r.roll[1], (n,), device=self.device)
    pitch = sample_uniform(r.pitch[0], r.pitch[1], (n,), device=self.device)
    yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
    self.base_quat_b[env_ids] = quat_from_euler_xyz(roll, pitch, yaw)

    self.circle_radius[env_ids] = sample_uniform(
      r.radius[0], r.radius[1], (n,), device=self.device
    )
    self.angular_speed[env_ids] = sample_uniform(
      r.angular_speed[0], r.angular_speed[1], (n,), device=self.device
    )
    self.phase[env_ids] = sample_uniform(
      r.phase[0], r.phase[1], (n,), device=self.device
    )
    self.elapsed_time[env_ids] = 0.0

  def _update_command(self) -> None:
    self.elapsed_time += self._env.step_dt
    theta = self.phase + self.angular_speed * self.elapsed_time

    self.target_pos_b[:, 0] = self.center_pos_b[:, 0] + self.circle_radius * torch.cos(
      theta
    )
    self.target_pos_b[:, 1] = self.center_pos_b[:, 1] + self.circle_radius * torch.sin(
      theta
    )
    self.target_pos_b[:, 2] = self.center_pos_b[:, 2]
    self.target_quat_b[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    base_pos_w = self.robot.data.body_link_pos_w[:, self._base_body_id]
    base_quat_w = self.robot.data.body_link_quat_w[:, self._base_body_id]
    if self._ee_is_site:
      ee_pos_w = self.robot.data.site_pos_w[:, self._ee_frame_id]
      ee_quat_w = self.robot.data.site_quat_w[:, self._ee_frame_id]
    else:
      ee_pos_w = self.robot.data.body_link_pos_w[:, self._ee_frame_id]
      ee_quat_w = self.robot.data.body_link_quat_w[:, self._ee_frame_id]

    self.target_pos_w[:] = base_pos_w + quat_apply(base_quat_w, self.target_pos_b)
    self.target_quat_w[:] = quat_mul(base_quat_w, self.target_quat_b)

    pos_err, rot_err = compute_pose_error(
      ee_pos_w,
      ee_quat_w,
      self.target_pos_w,
      self.target_quat_w,
      rot_error_type="axis_angle",
    )
    self.delta_pose_command[:, :3] = pos_err
    self.delta_pose_command[:, 3:] = rot_err

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    for batch in env_indices:
      if self._ee_is_site:
        ee_pos = self.robot.data.site_pos_w[batch, self._ee_frame_id].cpu().numpy()
        ee_quat = self.robot.data.site_quat_w[batch, self._ee_frame_id]
      else:
        ee_pos = self.robot.data.body_link_pos_w[batch, self._ee_frame_id].cpu().numpy()
        ee_quat = self.robot.data.body_link_quat_w[batch, self._ee_frame_id]
      ee_rotm = matrix_from_quat(ee_quat).cpu().numpy()

      target_pos = self.target_pos_w[batch].cpu().numpy()
      target_rotm = matrix_from_quat(self.target_quat_w[batch]).cpu().numpy()

      visualizer.add_sphere(
        center=target_pos,
        radius=self.cfg.viz.target_radius,
        color=self.cfg.viz.target_color,
        label=f"ee_target_pos_{batch}",
      )
      visualizer.add_frame(
        position=target_pos,
        rotation_matrix=target_rotm,
        scale=self.cfg.viz.frame_scale,
        axis_radius=self.cfg.viz.axis_radius,
        alpha=self.cfg.viz.frame_alpha,
        label=f"ee_target_frame_{batch}",
        axis_colors=self.cfg.viz.axis_colors,
      )
      visualizer.add_sphere(
        center=ee_pos,
        radius=self.cfg.viz.current_radius,
        color=self.cfg.viz.current_color,
        label=f"ee_current_pos_{batch}",
      )
      visualizer.add_frame(
        position=ee_pos,
        rotation_matrix=ee_rotm,
        scale=self.cfg.viz.current_frame_scale,
        axis_radius=self.cfg.viz.current_axis_radius,
        alpha=self.cfg.viz.current_frame_alpha,
        label=f"ee_current_frame_{batch}",
      )


@dataclass(kw_only=True)
class CircularPoseCommandCfg(CommandTermCfg):
  entity_name: str
  base_body_name: str
  ee_frame_type: Literal["site", "body"] = "site"
  ee_frame_name: str = ""

  @dataclass
  class Ranges:
    x: tuple[float, float] = (0.20, 0.50)
    y: tuple[float, float] = (-0.20, 0.20)
    z: tuple[float, float] = (0.10, 0.40)
    roll: tuple[float, float] = (0.0, 0.0)
    pitch: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (-3.14, 3.14)
    radius: tuple[float, float] = (0.03, 0.10)
    angular_speed: tuple[float, float] = (0.2, 0.8)
    phase: tuple[float, float] = (-3.14, 3.14)

  ranges: Ranges = field(default_factory=Ranges)

  @dataclass
  class VizCfg:
    target_radius: float = 0.02
    frame_scale: float = 0.08
    axis_radius: float = 0.007
    frame_alpha: float = 0.85
    target_color: tuple[float, float, float, float] = (0.9, 0.5, 0.1, 0.75)
    current_radius: float = 0.018
    current_frame_scale: float = 0.1
    current_axis_radius: float = 0.009
    current_frame_alpha: float = 0.95
    current_color: tuple[float, float, float, float] = (0.2, 0.9, 0.3, 0.8)
    axis_colors: tuple[tuple[float, float, float], ...] = (
      (1.0, 0.4, 0.0),
      (0.0, 0.85, 0.35),
      (0.2, 0.6, 1.0),
    )

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> CircularPoseCommand:
    return CircularPoseCommand(self, env)

  def __post_init__(self):
    if not self.ee_frame_name and self.ee_site_name:
      self.ee_frame_name = self.ee_site_name
      self.ee_frame_type = "site"
    if not self.ee_frame_name:
      raise ValueError("`ee_frame_name` must be provided for CircularPoseCfg.")


class LinePoseCommand(CommandTerm):
  cfg: LinePoseCommandCfg

  def __init__(self, cfg: LinePoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    body_ids, _ = self.robot.find_bodies(cfg.base_body_name)
    self._base_body_id = body_ids[0]
    if cfg.ee_frame_type == "site":
      site_ids, _ = self.robot.find_sites(cfg.ee_frame_name)
      self._ee_frame_id = site_ids[0]
      self._ee_is_site = True
    else:
      body_ids, _ = self.robot.find_bodies(cfg.ee_frame_name)
      self._ee_frame_id = body_ids[0]
      self._ee_is_site = False

    self.center_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.base_quat_b = torch.zeros(self.num_envs, 4, device=self.device)
    self.base_quat_b[:, 0] = 1.0
    self.target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_quat_b = torch.zeros(self.num_envs, 4, device=self.device)
    self.target_quat_b[:, 0] = 1.0
    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
    self.target_quat_w[:, 0] = 1.0
    self.delta_pose_command = torch.zeros(self.num_envs, 6, device=self.device)

    self.line_half_length = torch.zeros(self.num_envs, device=self.device)
    self.angular_speed = torch.zeros(self.num_envs, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.line_heading = torch.zeros(self.num_envs, device=self.device)
    self.elapsed_time = torch.zeros(self.num_envs, device=self.device)

    self.metrics["ee_pos_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["ee_rot_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.delta_pose_command

  def _update_metrics(self) -> None:
    if self._ee_is_site:
      ee_pos = self.robot.data.site_pos_w[:, self._ee_frame_id]
      ee_quat = self.robot.data.site_quat_w[:, self._ee_frame_id]
    else:
      ee_pos = self.robot.data.body_link_pos_w[:, self._ee_frame_id]
      ee_quat = self.robot.data.body_link_quat_w[:, self._ee_frame_id]
    self.metrics["ee_pos_error"] = torch.norm(self.target_pos_w - ee_pos, dim=-1)
    _, rot_err = compute_pose_error(
      ee_pos,
      ee_quat,
      self.target_pos_w,
      self.target_quat_w,
      rot_error_type="axis_angle",
    )
    self.metrics["ee_rot_error"] = torch.linalg.vector_norm(rot_err, dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return

    r = self.cfg.ranges
    lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    self.center_pos_b[env_ids] = sample_uniform(
      lower, upper, (n, 3), device=self.device
    )

    roll = sample_uniform(r.roll[0], r.roll[1], (n,), device=self.device)
    pitch = sample_uniform(r.pitch[0], r.pitch[1], (n,), device=self.device)
    yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
    self.base_quat_b[env_ids] = quat_from_euler_xyz(roll, pitch, yaw)

    self.line_half_length[env_ids] = sample_uniform(
      r.length[0], r.length[1], (n,), device=self.device
    )
    self.angular_speed[env_ids] = sample_uniform(
      r.angular_speed[0], r.angular_speed[1], (n,), device=self.device
    )
    self.phase[env_ids] = sample_uniform(
      r.phase[0], r.phase[1], (n,), device=self.device
    )
    self.line_heading[env_ids] = sample_uniform(
      r.heading[0], r.heading[1], (n,), device=self.device
    )
    self.elapsed_time[env_ids] = 0.0

  def _update_command(self) -> None:
    self.elapsed_time += self._env.step_dt
    theta = self.phase + self.angular_speed * self.elapsed_time
    displacement = self.line_half_length * torch.sin(theta)
    dir_x = torch.cos(self.line_heading)
    dir_y = torch.sin(self.line_heading)

    self.target_pos_b[:, 0] = self.center_pos_b[:, 0] + displacement * dir_x
    self.target_pos_b[:, 1] = self.center_pos_b[:, 1] + displacement * dir_y
    self.target_pos_b[:, 2] = self.center_pos_b[:, 2]
    self.target_quat_b[:] = self.base_quat_b

    base_pos_w = self.robot.data.body_link_pos_w[:, self._base_body_id]
    base_quat_w = self.robot.data.body_link_quat_w[:, self._base_body_id]
    if self._ee_is_site:
      ee_pos_w = self.robot.data.site_pos_w[:, self._ee_frame_id]
      ee_quat_w = self.robot.data.site_quat_w[:, self._ee_frame_id]
    else:
      ee_pos_w = self.robot.data.body_link_pos_w[:, self._ee_frame_id]
      ee_quat_w = self.robot.data.body_link_quat_w[:, self._ee_frame_id]

    self.target_pos_w[:] = base_pos_w + quat_apply(base_quat_w, self.target_pos_b)
    self.target_quat_w[:] = quat_mul(base_quat_w, self.target_quat_b)

    pos_err, rot_err = compute_pose_error(
      ee_pos_w,
      ee_quat_w,
      self.target_pos_w,
      self.target_quat_w,
      rot_error_type="axis_angle",
    )
    self.delta_pose_command[:, :3] = pos_err
    self.delta_pose_command[:, 3:] = rot_err

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    for batch in env_indices:
      if self._ee_is_site:
        ee_pos = self.robot.data.site_pos_w[batch, self._ee_frame_id].cpu().numpy()
        ee_quat = self.robot.data.site_quat_w[batch, self._ee_frame_id]
      else:
        ee_pos = self.robot.data.body_link_pos_w[batch, self._ee_frame_id].cpu().numpy()
        ee_quat = self.robot.data.body_link_quat_w[batch, self._ee_frame_id]
      ee_rotm = matrix_from_quat(ee_quat).cpu().numpy()

      target_pos = self.target_pos_w[batch].cpu().numpy()
      target_rotm = matrix_from_quat(self.target_quat_w[batch]).cpu().numpy()

      visualizer.add_sphere(
        center=target_pos,
        radius=self.cfg.viz.target_radius,
        color=self.cfg.viz.target_color,
        label=f"ee_target_pos_{batch}",
      )
      visualizer.add_frame(
        position=target_pos,
        rotation_matrix=target_rotm,
        scale=self.cfg.viz.frame_scale,
        axis_radius=self.cfg.viz.axis_radius,
        alpha=self.cfg.viz.frame_alpha,
        label=f"ee_target_frame_{batch}",
        axis_colors=self.cfg.viz.axis_colors,
      )
      visualizer.add_sphere(
        center=ee_pos,
        radius=self.cfg.viz.current_radius,
        color=self.cfg.viz.current_color,
        label=f"ee_current_pos_{batch}",
      )
      visualizer.add_frame(
        position=ee_pos,
        rotation_matrix=ee_rotm,
        scale=self.cfg.viz.current_frame_scale,
        axis_radius=self.cfg.viz.current_axis_radius,
        alpha=self.cfg.viz.current_frame_alpha,
        label=f"ee_current_frame_{batch}",
      )


@dataclass(kw_only=True)
class LinePoseCommandCfg(CommandTermCfg):
  entity_name: str
  base_body_name: str
  ee_frame_type: Literal["site", "body"] = "site"
  ee_frame_name: str = ""

  @dataclass
  class Ranges:
    x: tuple[float, float] = (0.20, 0.50)
    y: tuple[float, float] = (-0.20, 0.20)
    z: tuple[float, float] = (0.10, 0.40)
    roll: tuple[float, float] = (0.0, 0.0)
    pitch: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (-3.14, 3.14)
    length: tuple[float, float] = (0.03, 0.10)
    angular_speed: tuple[float, float] = (0.2, 0.8)
    phase: tuple[float, float] = (-3.14, 3.14)
    heading: tuple[float, float] = (-3.14, 3.14)

  ranges: Ranges = field(default_factory=Ranges)

  @dataclass
  class VizCfg:
    target_radius: float = 0.02
    frame_scale: float = 0.08
    axis_radius: float = 0.007
    frame_alpha: float = 0.85
    target_color: tuple[float, float, float, float] = (0.9, 0.5, 0.1, 0.75)
    current_radius: float = 0.018
    current_frame_scale: float = 0.1
    current_axis_radius: float = 0.009
    current_frame_alpha: float = 0.95
    current_color: tuple[float, float, float, float] = (0.2, 0.9, 0.3, 0.8)
    axis_colors: tuple[tuple[float, float, float], ...] = (
      (1.0, 0.4, 0.0),
      (0.0, 0.85, 0.35),
      (0.2, 0.6, 1.0),
    )

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> LinePoseCommand:
    return LinePoseCommand(self, env)

  def __post_init__(self):
    if not self.ee_frame_name and self.ee_site_name:
      self.ee_frame_name = self.ee_site_name
      self.ee_frame_type = "site"
    if not self.ee_frame_name:
      raise ValueError("`ee_frame_name` must be provided for LinePoseCommandCfg.")


class CommandDrivenDifferentialIK(CommandTerm):
  """Run Differential IK directly from a command tensor (no policy action term)."""

  cfg: CommandDrivenDifferentialIKCfg

  def __init__(self, cfg: CommandDrivenDifferentialIKCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    ik_cfg = DifferentialIKActionCfg(
      entity_name=cfg.entity_name,
      actuator_names=cfg.actuator_names,
      frame_type=cfg.frame_type,
      frame_name=cfg.frame_name,
      use_relative_mode=cfg.use_relative_mode,
      delta_pos_scale=cfg.delta_pos_scale,
      delta_ori_scale=cfg.delta_ori_scale,
      damping=cfg.damping,
      max_dq=cfg.max_dq,
      position_weight=cfg.position_weight,
      orientation_weight=cfg.orientation_weight,
      joint_limit_weight=cfg.joint_limit_weight,
      posture_weight=cfg.posture_weight,
      posture_target=cfg.posture_target,
    )
    self._ik = DifferentialIKAction(cfg=ik_cfg, env=env)
    joint_names = tuple(
      self._ik._entity.joint_names[int(i)] for i in self._ik._joint_ids
    )
    self._kp = self._resolve_gain(cfg.kp_ff, joint_names)
    self._kd = self._resolve_gain(cfg.kd_ff, joint_names)
    self._command = torch.zeros(self.num_envs, self._ik.action_dim, device=self.device)
    self.metrics["ik_action_l2"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    self.metrics["ik_action_l2"] = torch.linalg.vector_norm(self._command, dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self._command[env_ids] = 0.0
    self._ik.reset(env_ids)

  def _update_command(self) -> None:
    src = self._env.command_manager.get_command(self.cfg.source_command_name)
    if src.shape[1] < self._ik.action_dim:
      raise ValueError(
        f"Command '{self.cfg.source_command_name}' has dim={src.shape[1]}, "
        f"but IK expects dim={self._ik.action_dim}."
      )
    self._command[:] = src[:, : self._ik.action_dim]
    n_iter = max(1, int(self.cfg.ik_iterations))

    self._ik.process_actions(self._command)
    q_current = self._ik._entity.data.joint_pos[:, self._ik._joint_ids].clone()
    q_target = q_current.clone()
    try:
      for _ in range(n_iter):
        q_target = q_target + self._ik.compute_dq()
        self._ik._entity.write_joint_position_to_sim(
          q_target, joint_ids=self._ik._joint_ids
        )
        self._env.sim.forward()
    finally:
      self._ik._entity.write_joint_position_to_sim(
        q_current, joint_ids=self._ik._joint_ids
      )
      self._env.sim.forward()

    self._apply_joint_target(q_target)

  def _apply_joint_target(self, q_target: torch.Tensor) -> None:
    if self.cfg.control_mode == "position":
      self._ik._entity.set_joint_position_target(
        q_target, joint_ids=self._ik._joint_ids
      )
      return

    if self.cfg.control_mode != "impedance":
      raise ValueError(f"Unknown control_mode: {self.cfg.control_mode}")

    q = self._ik._entity.data.joint_pos[:, self._ik._joint_ids]
    qd = self._ik._entity.data.joint_vel[:, self._ik._joint_ids]
    tau_cmd = self._kp * (q_target - q) - self._kd * qd
    dof_ids = self._ik._entity.indexing.joint_v_adr[self._ik._joint_ids]

    if self.cfg.inertial_compensation:
      m_arm = self._env.sim.data.qM[:, dof_ids][:, :, dof_ids]
      tau_cmd = torch.einsum("bij,bj->bi", m_arm, tau_cmd)

    if self.cfg.gravity_compensation:
      tau_cmd = tau_cmd + self._env.sim.data.qfrc_bias[:, dof_ids]

    self._ik._entity.set_joint_effort_target(tau_cmd, joint_ids=self._ik._joint_ids)

  def _resolve_gain(
    self, gain: float | dict[str, float], joint_names: tuple[str, ...]
  ) -> torch.Tensor:
    gains = torch.ones(self.num_envs, len(joint_names), device=self.device)
    if isinstance(gain, (float, int)):
      gains *= float(gain)
      return gains
    if isinstance(gain, dict):
      index_list, _, value_list = resolve_matching_names_values(gain, joint_names)
      gains[:, index_list] = torch.tensor(value_list, device=self.device)
      return gains
    raise ValueError(f"Unsupported gain type: {type(gain)}. Use float or dict.")

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    del visualizer


@dataclass(kw_only=True)
class CommandDrivenDifferentialIKCfg(CommandTermCfg):
  """Configuration for command-driven Differential IK."""

  entity_name: str
  source_command_name: str
  actuator_names: tuple[str, ...] | list[str]
  frame_type: Literal["body", "site", "geom"] = "body"
  frame_name: str
  use_relative_mode: bool = True
  delta_pos_scale: float = 1.0
  delta_ori_scale: float = 1.0
  damping: float = 0.05
  max_dq: float = 0.5
  position_weight: float = 1.0
  orientation_weight: float = 1.0
  joint_limit_weight: float = 0.0
  posture_weight: float = 0.0
  posture_target: dict[str, float] | None = None
  ik_iterations: int = 1
  # Deprecated compatibility knob. The command term now always performs
  # iterated IK refinement, restores sim state, then applies one actuator target.
  iteration_mode: Literal["target", "state"] = "target"
  control_mode: Literal["position", "impedance"] = "position"
  kp_ff: float | dict[str, float] = 0.0
  kd_ff: float | dict[str, float] = 0.0
  gravity_compensation: bool = True
  inertial_compensation: bool = False

  # This term doesn't sample commands itself; keep a long timer to avoid resampling churn.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> CommandDrivenDifferentialIK:
    return CommandDrivenDifferentialIK(self, env)


class PseudoInverseVelocityCommand(CommandTerm):
  """Map an end-effector velocity command to arm joint velocity targets."""

  cfg: PseudoInverseVelocityCommandCfg

  def __init__(self, cfg: PseudoInverseVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    joint_ids, joint_names = self.robot.find_joints_by_actuator_names(
      cfg.actuator_names
    )
    self._joint_names = tuple(joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._joint_dof_ids = self.robot.indexing.joint_v_adr[self._joint_ids]
    self._num_joints = len(joint_ids)

    self._frame_type = cfg.frame_type
    if cfg.frame_type == "body":
      ids, _ = self.robot.find_bodies(cfg.frame_name)
      local_id = ids[0]
      self._frame_id = int(self.robot.indexing.body_ids[local_id].item())
      self._body_id = self._frame_id
    elif cfg.frame_type == "site":
      ids, _ = self.robot.find_sites(cfg.frame_name)
      local_id = ids[0]
      self._frame_id = int(self.robot.indexing.site_ids[local_id].item())
      self._body_id = int(self._env.sim.mj_model.site_bodyid[self._frame_id])
    elif cfg.frame_type == "geom":
      ids, _ = self.robot.find_geoms(cfg.frame_name)
      local_id = ids[0]
      self._frame_id = int(self.robot.indexing.geom_ids[local_id].item())
      self._body_id = int(self._env.sim.mj_model.geom_bodyid[self._frame_id])
    else:
      raise ValueError(f"Unknown frame_type: {cfg.frame_type}")

    nworld = self.num_envs
    nv = self._env.sim.mj_model.nv
    with wp.ScopedDevice(self._env.sim.wp_device):
      self._jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._jacr_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._point_wp = wp.zeros(nworld, dtype=wp.vec3)
      self._body_wp = wp.zeros(nworld, dtype=wp.int32)
      self._body_wp.fill_(self._body_id)

    self._jacp_torch = wp.to_torch(self._jacp_wp)
    self._jacr_torch = wp.to_torch(self._jacr_wp)
    self._point_torch = wp.to_torch(self._point_wp).view(nworld, 3)

    self.joint_velocity_command = torch.zeros(
      self.num_envs, self._num_joints, device=self.device
    )
    self.joint_effort_command = torch.zeros_like(self.joint_velocity_command)
    self.ee_velocity_command = torch.zeros(self.num_envs, 6, device=self.device)
    self.non_controlled_ee_velocity = torch.zeros(self.num_envs, 6, device=self.device)
    self._non_controlled_compensation_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._kp = self._resolve_gain(cfg.kp_ff)
    self._kd = self._resolve_gain(cfg.kd_ff)
    self.metrics["joint_velocity_l2"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["ee_velocity_l2"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["ee_linear_velocity_l2"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["ee_angular_velocity_l2"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["non_controlled_linear_velocity_l2"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["non_controlled_angular_velocity_l2"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["non_controlled_compensation_active"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["joint_effort_l2"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.joint_velocity_command

  def _update_metrics(self) -> None:
    self.metrics["joint_velocity_l2"] = torch.linalg.vector_norm(
      self.joint_velocity_command, dim=-1
    )
    self.metrics["ee_velocity_l2"] = torch.linalg.vector_norm(
      self.ee_velocity_command, dim=-1
    )
    self.metrics["ee_linear_velocity_l2"] = torch.linalg.vector_norm(
      self.ee_velocity_command[:, :3], dim=-1
    )
    self.metrics["ee_angular_velocity_l2"] = torch.linalg.vector_norm(
      self.ee_velocity_command[:, 3:], dim=-1
    )
    self.metrics["non_controlled_linear_velocity_l2"] = torch.linalg.vector_norm(
      self.non_controlled_ee_velocity[:, :3], dim=-1
    )
    self.metrics["non_controlled_angular_velocity_l2"] = torch.linalg.vector_norm(
      self.non_controlled_ee_velocity[:, 3:], dim=-1
    )
    self.metrics["non_controlled_compensation_active"] = (
      self._non_controlled_compensation_active.float()
    )
    self.metrics["joint_effort_l2"] = torch.linalg.vector_norm(
      self.joint_effort_command,
      dim=-1,
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self.joint_velocity_command[env_ids] = 0.0
    self.joint_effort_command[env_ids] = 0.0
    self.robot.data.joint_vel_target[env_ids[:, None], self._joint_ids[None, :]] = (
      self.joint_velocity_command[env_ids]
    )
    if self.cfg.control_mode == "impedance":
      self.robot.data.joint_effort_target[
        env_ids[:, None], self._joint_ids[None, :]
      ] = self.joint_effort_command[env_ids]

  def _update_command(self) -> None:
    src = self._env.command_manager.get_command(self.cfg.source_command_name)
    twist = self._extract_twist(src)
    if self.cfg.command_frame == "ee":
      _, frame_quat = self._get_frame_pose()
      twist = torch.cat(
        [
          quat_apply(frame_quat, twist[:, :3]),
          quat_apply(frame_quat, twist[:, 3:]),
        ],
        dim=-1,
      )

    self._compute_jacobian()
    jacp = self._jacp_torch[:, :, self._joint_dof_ids]
    jacr = self._jacr_torch[:, :, self._joint_dof_ids]
    if self.cfg.compensate_non_controlled_ee_velocity:
      active = self._compute_compensation_active(twist)
      non_controlled_twist = self._compute_non_controlled_ee_velocity(
        jacp, jacr
      ).clone()
      self.non_controlled_ee_velocity[:] = non_controlled_twist * active.unsqueeze(-1)
      twist = torch.where(
        active.unsqueeze(-1),
        twist - non_controlled_twist,
        twist,
      )
    else:
      self.non_controlled_ee_velocity.zero_()
      self._non_controlled_compensation_active.zero_()
    twist = self._limit_ee_twist(twist)
    self.ee_velocity_command[:] = twist

    w_pos = self.cfg.linear_weight
    w_ori = self.cfg.angular_weight
    lam = max(self.cfg.damping, 1e-6)

    JTJ = (w_pos * w_pos) * torch.einsum("bti,btj->bij", jacp, jacp)
    JTJ += (w_ori * w_ori) * torch.einsum("bti,btj->bij", jacr, jacr)
    JTdx = (w_pos * w_pos) * torch.einsum("bti,bt->bi", jacp, twist[:, :3])
    JTdx += (w_ori * w_ori) * torch.einsum("bti,bt->bi", jacr, twist[:, 3:])

    if self.cfg.posture_weight > 0.0:
      q = self.robot.data.joint_pos[:, self._joint_ids]
      qd = self.robot.data.joint_vel[:, self._joint_ids]
      posture_vel = self.cfg.posture_gain * (
        self.robot.data.default_joint_pos[:, self._joint_ids] - q
      )
      w_post2 = self.cfg.posture_weight * self.cfg.posture_weight
      JTJ.diagonal(dim1=-2, dim2=-1).add_(w_post2)
      JTdx.add_(w_post2 * (posture_vel - qd))

    JTJ.diagonal(dim1=-2, dim2=-1).add_(lam * lam)
    qd_cmd = torch.linalg.solve(JTJ, JTdx)
    qd_cmd = qd_cmd.clamp(-self.cfg.max_joint_velocity, self.cfg.max_joint_velocity)

    self.joint_velocity_command[:] = qd_cmd
    if self.cfg.control_mode == "velocity":
      self.robot.set_joint_velocity_target(qd_cmd, joint_ids=self._joint_ids)
    elif self.cfg.control_mode == "impedance":
      self._apply_impedance(qd_cmd)
    else:
      raise ValueError(f"Unknown control_mode: {self.cfg.control_mode}")

  def _compute_non_controlled_ee_velocity(
    self,
    jacp_arm: torch.Tensor,
    jacr_arm: torch.Tensor,
  ) -> torch.Tensor:
    qvel = self._env.sim.data.qvel
    full_lin = torch.einsum("bti,bi->bt", self._jacp_torch, qvel)
    full_ang = torch.einsum("bti,bi->bt", self._jacr_torch, qvel)

    qd_arm = qvel[:, self._joint_dof_ids]
    arm_lin = torch.einsum("bti,bi->bt", jacp_arm, qd_arm)
    arm_ang = torch.einsum("bti,bi->bt", jacr_arm, qd_arm)

    self.non_controlled_ee_velocity[:, :3] = full_lin - arm_lin
    self.non_controlled_ee_velocity[:, 3:] = full_ang - arm_ang
    return self.non_controlled_ee_velocity

  def _compute_compensation_active(self, twist: torch.Tensor) -> torch.Tensor:
    command_norm = torch.linalg.vector_norm(
      twist[:, :3], dim=-1
    ) + torch.linalg.vector_norm(twist[:, 3:], dim=-1)
    threshold = self.cfg.non_controlled_ee_compensation_command_threshold
    active = command_norm >= threshold
    self._non_controlled_compensation_active.copy_(active)
    return active

  def _limit_ee_twist(self, twist: torch.Tensor) -> torch.Tensor:
    lin_vel = self._limit_vector_norm(twist[:, :3], self.cfg.max_ee_linear_velocity)
    ang_vel = self._limit_vector_norm(twist[:, 3:], self.cfg.max_ee_angular_velocity)
    return torch.cat([lin_vel, ang_vel], dim=-1)

  def _limit_vector_norm(
    self, value: torch.Tensor, max_norm: float | None
  ) -> torch.Tensor:
    if max_norm is None:
      return value
    max_norm = float(max_norm)
    if max_norm < 0.0:
      raise ValueError("EE velocity limits must be non-negative or None.")
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / norm.clamp_min(1.0e-6), max=1.0)
    return value * scale

  def _apply_impedance(self, qd_cmd: torch.Tensor) -> None:
    q = self.robot.data.joint_pos[:, self._joint_ids]
    qd = self.robot.data.joint_vel[:, self._joint_ids]
    q_des = q + qd_cmd * self._env.step_dt * self.cfg.velocity_target_dt_scale

    tau_cmd = self._kp * (q_des - q) + self._kd * (qd_cmd - qd)
    dof_ids = self.robot.indexing.joint_v_adr[self._joint_ids]

    if self.cfg.inertial_compensation:
      m_arm = self._env.sim.data.qM[:, dof_ids][:, :, dof_ids]
      tau_cmd = torch.einsum("bij,bj->bi", m_arm, tau_cmd)

    if self.cfg.gravity_compensation:
      tau_cmd = tau_cmd + self._env.sim.data.qfrc_bias[:, dof_ids]

    self.joint_effort_command[:] = tau_cmd
    self.robot.set_joint_effort_target(tau_cmd, joint_ids=self._joint_ids)

  def _resolve_gain(self, gain: float | dict[str, float]) -> torch.Tensor:
    gains = torch.ones(self.num_envs, self._num_joints, device=self.device)
    if isinstance(gain, (float, int)):
      gains *= float(gain)
      return gains
    if isinstance(gain, dict):
      index_list, _, value_list = resolve_matching_names_values(gain, self._joint_names)
      gains[:, index_list] = torch.tensor(value_list, device=self.device)
      return gains
    raise ValueError(f"Unsupported gain type: {type(gain)}. Use float or dict.")

  def _extract_twist(self, command: torch.Tensor) -> torch.Tensor:
    if command.shape[1] < 6:
      raise ValueError(
        f"Command '{self.cfg.source_command_name}' has dim={command.shape[1]}, "
        "but PseudoInverseVelocityCommand expects at least 6."
      )
    if self.cfg.source_command_layout == "linear_angular":
      return command[:, :6]
    if self.cfg.source_command_layout == "twist_prefix":
      return command[:, [0, 1, 3, 4, 5, 2]]
    raise ValueError(f"Unknown source_command_layout: {self.cfg.source_command_layout}")

  def _get_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    data = self._env.sim.data
    if self._frame_type == "body":
      return data.xpos[:, self._frame_id], data.xquat[:, self._frame_id]
    if self._frame_type == "site":
      return (
        data.site_xpos[:, self._frame_id],
        quat_from_matrix(data.site_xmat[:, self._frame_id]),
      )
    assert self._frame_type == "geom"
    return (
      data.geom_xpos[:, self._frame_id],
      quat_from_matrix(data.geom_xmat[:, self._frame_id]),
    )

  def _compute_jacobian(self) -> None:
    frame_pos, _ = self._get_frame_pose()
    self._point_torch[:] = frame_pos
    with wp.ScopedDevice(self._env.sim.wp_device):
      mjwarp.jac(
        self._env.sim.wp_model,
        self._env.sim.wp_data,
        self._jacp_wp,
        self._jacr_wp,
        self._point_wp,
        self._body_wp,
      )


@dataclass(kw_only=True)
class PseudoInverseVelocityCommandCfg(CommandTermCfg):
  """Configuration for velocity-level Jacobian pseudoinverse control."""

  entity_name: str
  source_command_name: str
  actuator_names: tuple[str, ...] | list[str]
  frame_type: Literal["body", "site", "geom"] = "site"
  frame_name: str
  command_frame: Literal["world", "ee"] = "ee"
  source_command_layout: Literal["linear_angular", "twist_prefix"] = "linear_angular"
  control_mode: Literal["velocity", "impedance"] = "velocity"
  damping: float = 0.05
  max_joint_velocity: float = 1.0
  max_ee_linear_velocity: float | None = None
  max_ee_angular_velocity: float | None = None
  linear_weight: float = 1.0
  angular_weight: float = 1.0
  compensate_non_controlled_ee_velocity: bool = False
  non_controlled_ee_compensation_command_threshold: float = 0.0
  posture_weight: float = 0.0
  posture_gain: float = 1.0
  kp_ff: float | dict[str, float] = 0.0
  kd_ff: float | dict[str, float] = 1.0
  velocity_target_dt_scale: float = 0.0
  gravity_compensation: bool = True
  inertial_compensation: bool = False

  # This command follows another command term; keep a long timer.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> PseudoInverseVelocityCommand:
    return PseudoInverseVelocityCommand(self, env)


class Uniform3DVelocityCommand(CommandTerm):
  """A command term that generates uniform end-effector-frame 3D velocities."""

  cfg: Uniform3DVelocityCommandCfg

  def __init__(self, cfg: Uniform3DVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    if cfg.ee_frame_type == "site":
      site_ids, _ = self.robot.find_sites(cfg.ee_frame_name)
      self._ee_frame_id = site_ids[0]
      self._ee_is_site = True
    else:
      body_ids, _ = self.robot.find_bodies(cfg.ee_frame_name)
      self._ee_frame_id = body_ids[0]
      self._ee_is_site = False

    # End-effector-frame command layout keeps the 2D twist prefix compatible:
    # [vx, vy, wz, vz, wx, wy].
    self.target_velocity_ee = torch.zeros(self.num_envs, 6, device=self.device)
    self.metrics["linear_velocity_l2"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["angular_velocity_l2"] = torch.zeros(self.num_envs, device=self.device)

    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    """The desired end-effector-frame twist. Shape is (num_envs, 6).

    Layout: [vx, vy, wz, vz, wx, wy].
    """
    return self.target_velocity_ee

  def _update_metrics(self) -> None:
    self.metrics["linear_velocity_l2"] = torch.linalg.vector_norm(
      self.target_velocity_ee[:, [0, 1, 3]], dim=-1
    )
    self.metrics["angular_velocity_l2"] = torch.linalg.vector_norm(
      self.target_velocity_ee[:, [4, 5, 2]], dim=-1
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return

    r = self.cfg.ranges
    lower = torch.tensor(
      [
        r.lin_vel_x[0],
        r.lin_vel_y[0],
        r.ang_vel_z[0],
        r.lin_vel_z[0],
        r.ang_vel_x[0],
        r.ang_vel_y[0],
      ],
      device=self.device,
    )
    upper = torch.tensor(
      [
        r.lin_vel_x[1],
        r.lin_vel_y[1],
        r.ang_vel_z[1],
        r.lin_vel_z[1],
        r.ang_vel_x[1],
        r.ang_vel_y[1],
      ],
      device=self.device,
    )
    self.target_velocity_ee[env_ids] = sample_uniform(
      lower, upper, (n, 6), device=self.device
    )

  def _update_command(self) -> None:
    pass

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create end-effector-frame 3D velocity command sliders in the Viser viewer."""
    del request_action

    from viser import Icon

    r = self.cfg.ranges
    axes = [
      ("lin_vel_x", r.lin_vel_x),
      ("lin_vel_y", r.lin_vel_y),
      ("ang_vel_z", r.ang_vel_z),
      ("lin_vel_z", r.lin_vel_z),
      ("ang_vel_x", r.ang_vel_x),
      ("ang_vel_y", r.ang_vel_y),
    ]
    sliders: list[viser.GuiSliderHandle] = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, value_range in axes:
        initial_value = min(max(0.0, value_range[0]), value_range[1])
        slider = server.gui.add_slider(
          label,
          min=value_range[0],
          max=value_range[1],
          step=0.05,
          initial_value=initial_value,
        )

        @slider.on_update
        def _(_) -> None:
          if on_change is not None:
            on_change()

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for slider in sliders:
          slider.value = 0.0
        if on_change is not None:
          on_change()

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for slider, command_idx in zip(
        self._joystick_sliders, (0, 1, 2, 3, 4, 5), strict=True
      ):
        self.target_velocity_ee[idx, command_idx] = slider.value

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw commanded and actual end-effector velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self._ee_is_site:
      ee_pos_ws = self.robot.data.site_pos_w[:, self._ee_frame_id].cpu().numpy()
      ee_quat_ws = self.robot.data.site_quat_w[:, self._ee_frame_id]
      ee_vel_ws = self.robot.data.site_vel_w[:, self._ee_frame_id].cpu().numpy()
    else:
      ee_pos_ws = self.robot.data.body_link_pos_w[:, self._ee_frame_id].cpu().numpy()
      ee_quat_ws = self.robot.data.body_link_quat_w[:, self._ee_frame_id]
      lin_vel_w = self.robot.data.body_link_lin_vel_w[:, self._ee_frame_id]
      ang_vel_w = self.robot.data.body_link_ang_vel_w[:, self._ee_frame_id]
      ee_vel_ws = torch.cat([lin_vel_w, ang_vel_w], dim=-1).cpu().numpy()

    cmds = self.command
    cmd_lin_ws = quat_apply(ee_quat_ws, cmds[:, [0, 1, 3]]).cpu().numpy()
    cmd_ang_ws = quat_apply(ee_quat_ws, cmds[:, [4, 5, 2]]).cpu().numpy()
    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      ee_pos_w = ee_pos_ws[batch]
      if np.linalg.norm(ee_pos_w) < 1e-6:
        continue

      cmd_lin_w = cmd_lin_ws[batch]
      cmd_ang_w = cmd_ang_ws[batch]
      act_lin_w = ee_vel_ws[batch, :3]
      act_ang_w = ee_vel_ws[batch, 3:]

      origin = ee_pos_w + np.array([0.0, 0.0, z_offset])

      visualizer.add_arrow(
        origin,
        origin + cmd_lin_w * scale,
        color=(0.2, 0.2, 0.8, 0.75),
        width=0.015,
      )
      visualizer.add_arrow(
        origin,
        origin + cmd_ang_w * scale,
        color=(0.2, 0.8, 0.2, 0.75),
        width=0.015,
      )
      visualizer.add_arrow(
        origin,
        origin + act_lin_w * scale,
        color=(0.0, 0.7, 1.0, 0.75),
        width=0.01,
      )
      visualizer.add_arrow(
        origin,
        origin + act_ang_w * scale,
        color=(0.0, 1.0, 0.4, 0.75),
        width=0.01,
      )


@dataclass(kw_only=True)
class Uniform3DVelocityCommandCfg(CommandTermCfg):
  """Configuration for a uniform 3D velocity command term."""

  entity_name: str
  ee_frame_type: Literal["site", "body"] = "site"
  ee_frame_name: str = ""

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    lin_vel_z: tuple[float, float]
    ang_vel_x: tuple[float, float]
    ang_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> Uniform3DVelocityCommand:
    return Uniform3DVelocityCommand(self, env)
