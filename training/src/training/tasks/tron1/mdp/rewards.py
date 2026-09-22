from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_box_minus,
  quat_inv,
  quat_mul,
)
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_GRIPPER_ASSET_CFG = SceneEntityCfg("robot", joint_names=("gripper_joint",))


def position_command_error_tanh(
  env: ManagerBasedRlEnv, std: float, command_name: str
) -> torch.Tensor:
  """Reward position tracking with tanh kernel."""
  command = env.command_manager.get_command(command_name)
  des_pos_b = command[:, :3]
  distance = torch.norm(des_pos_b, dim=1)
  return 1 - torch.tanh(distance / std)


def heading_command_error_abs(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Penalize tracking orientation error."""
  command = env.command_manager.get_command(command_name)
  heading_b = command[:, 3]
  return heading_b.abs()


def foot_distance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold_min: float = 0.1,
  threshold_max: float = 1.0,
) -> torch.Tensor:
  """Reward distances between two foot."""
  asset: Entity = env.scene[asset_cfg.name]
  assert len(asset_cfg.site_ids) == 2, (
    "foot_distance reward requires exactly 2 sites to be specified."
  )
  foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, 0:2]
  feet_distance = torch.norm(foot_pos[:, 0, :2] - foot_pos[:, 1, :2], dim=-1)
  reward = torch.clip(threshold_min - feet_distance, 0, 1)
  reward += torch.clip(feet_distance - threshold_max, 0, 1)

  env.extras["log"]["Metrics/foot_distance"] = feet_distance
  return reward


def base_height_tracking(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking a target base height using an exponential kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  base_height = asset.data.root_link_pos_w[:, 2]
  height_error = base_height - target_height
  reward = torch.exp(-torch.square(height_error) / (std**2))

  env.extras["log"]["Metrics/base_height"] = base_height
  env.extras["log"]["Metrics/base_height_error"] = height_error
  return reward


def track_base_xy_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking commanded base-frame XY linear velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  return torch.exp(-error / std**2)


def track_base_yaw_rate(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking commanded base-frame yaw rate."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / std**2)


def joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize absolute joint deviation from the robot default pose."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)


def joint_zero_pos_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize squared joint deviation from zero position."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(joint_pos), dim=1)


def action_term_rate_l2(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """Penalize rate of a single action term's raw policy command."""
  start = 0
  for name, dim in zip(
    env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True
  ):
    end = start + dim
    if name == action_name:
      return torch.sum(
        torch.square(
          env.action_manager.action[:, start:end]
          - env.action_manager.prev_action[:, start:end]
        ),
        dim=1,
      )
    start = end
  raise ValueError(f"Action term '{action_name}' not found.")


def action_term_l2(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """Penalize magnitude of a single action term's raw policy command."""
  start = 0
  for name, dim in zip(
    env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True
  ):
    end = start + dim
    if name == action_name:
      return torch.sum(torch.square(env.action_manager.action[:, start:end]), dim=1)
    start = end
  raise ValueError(f"Action term '{action_name}' not found.")


def feet_air_when_ee_command_standing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  """Penalize feet being airborne when 3D EE velocity command is small."""
  contact_sensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  lin_norm = torch.linalg.vector_norm(command[:, [0, 1, 3]], dim=-1)
  ang_norm = torch.linalg.vector_norm(command[:, [4, 5, 2]], dim=-1)
  command_norm = lin_norm + ang_norm
  standing = (command_norm < command_threshold).float()

  in_air = (contact_sensor.data.found == 0).float()
  return torch.sum(in_air, dim=1) * standing


def _ee_pose_errors_in_base_frame(
  env: ManagerBasedRlEnv,
  command_name: str,
  base_body_name: str,
  ee_frame_name: str,
  ee_is_site: bool = True,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  base_body_id: int | None = None,
  ee_frame_id: int | None = None,
  use_step_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return EE position and orientation errors in base-frame command space.

  Supports two command formats from ``command_name``:
    - 7D target pose in base frame: ``[x, y, z, qw, qx, qy, qz]``
    - 6D delta pose command: ``[dx, dy, dz, d_rx, d_ry, d_rz]``
  """
  robot: Entity = env.scene[asset_cfg.name]
  frame_key = (base_body_name, ee_frame_name, ee_is_site)
  frame_cache = getattr(robot, "_ee_pose_error_frame_cache", None)
  if frame_cache is None:
    frame_cache = {}
    robot._ee_pose_error_frame_cache = frame_cache
  if base_body_id is None or ee_frame_id is None:
    cached_ids = frame_cache.get(frame_key)
    if cached_ids is None:
      cached_base_id = robot.find_bodies(base_body_name)[0][0]
      if ee_is_site:
        cached_ee_id = robot.find_sites(ee_frame_name)[0][0]
      else:
        cached_ee_id = robot.find_bodies(ee_frame_name)[0][0]
      cached_ids = (cached_base_id, cached_ee_id)
      frame_cache[frame_key] = cached_ids
    if base_body_id is None:
      base_body_id = cached_ids[0]
    if ee_frame_id is None:
      ee_frame_id = cached_ids[1]

  step_cache_key = (
    env.common_step_counter,
    command_name,
    asset_cfg.name,
    base_body_id,
    ee_frame_id,
    ee_is_site,
  )
  if use_step_cache:
    cached_errors = getattr(env, "_ee_pose_error_step_cache", None)
    if cached_errors is not None and cached_errors[0] == step_cache_key:
      return cached_errors[1]

  base_pos_w = robot.data.body_link_pos_w[:, base_body_id]
  base_quat_w = robot.data.body_link_quat_w[:, base_body_id]

  if ee_is_site:
    ee_pos_w = robot.data.site_pos_w[:, ee_frame_id]
    ee_quat_w = robot.data.site_quat_w[:, ee_frame_id]
  else:
    ee_pos_w = robot.data.body_link_pos_w[:, ee_frame_id]
    ee_quat_w = robot.data.body_link_quat_w[:, ee_frame_id]

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Delta-pose mode: command already stores tracking errors.
  if command.shape[1] == 6:
    pos_err = torch.linalg.vector_norm(command[:, :3], dim=-1)
    ori_err = torch.linalg.vector_norm(command[:, 3:], dim=-1)
  else:
    # Target-pose mode: command is desired EE pose in base frame.
    if command.shape[1] < 7:
      raise ValueError(
        f"Unsupported ee pose command shape {tuple(command.shape)} for '{command_name}'."
      )
    target_pos_b = command[:, :3]
    target_quat_b = torch.nn.functional.normalize(command[:, 3:7], p=2, dim=-1)

    ee_pos_b = quat_apply(quat_inv(base_quat_w), ee_pos_w - base_pos_w)
    ee_quat_b = quat_mul(quat_inv(base_quat_w), ee_quat_w)
    ee_quat_b = torch.nn.functional.normalize(ee_quat_b, p=2, dim=-1)

    pos_err = torch.linalg.vector_norm(target_pos_b - ee_pos_b, dim=-1)
    ori_err = torch.norm(quat_box_minus(target_quat_b, ee_quat_b), dim=-1)

  errors = (pos_err, ori_err)
  if use_step_cache:
    env._ee_pose_error_step_cache = (step_cache_key, errors)
  return errors


def ee_position_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  base_body_name: str,
  ee_frame_name: str,
  ee_is_site: bool = True,
  position_std: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward end-effector position tracking."""
  pos_err, _ = _ee_pose_errors_in_base_frame(
    env=env,
    command_name=command_name,
    base_body_name=base_body_name,
    ee_frame_name=ee_frame_name,
    ee_is_site=ee_is_site,
    asset_cfg=asset_cfg,
    use_step_cache=True,
  )
  pos_rew = torch.exp(-torch.square(pos_err / position_std))
  env.extras["log"]["Metrics/ee_pos_error"] = pos_err
  return pos_rew


def ee_orientation_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  base_body_name: str,
  ee_frame_name: str,
  ee_is_site: bool = True,
  orientation_std: float = 0.35,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward end-effector orientation tracking."""
  _, ori_err = _ee_pose_errors_in_base_frame(
    env=env,
    command_name=command_name,
    base_body_name=base_body_name,
    ee_frame_name=ee_frame_name,
    ee_is_site=ee_is_site,
    asset_cfg=asset_cfg,
    use_step_cache=True,
  )
  ori_rew = torch.exp(-torch.square(ori_err / orientation_std))
  env.extras["log"]["Metrics/ee_ori_error_rad"] = ori_err
  return ori_rew


def ee_pose_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  base_body_name: str,
  ee_frame_name: str,
  ee_is_site: bool = True,
  position_std: float = 0.05,
  orientation_std: float = 0.35,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Compatibility wrapper: product of position and orientation rewards."""
  pos_err, ori_err = _ee_pose_errors_in_base_frame(
    env=env,
    command_name=command_name,
    base_body_name=base_body_name,
    ee_frame_name=ee_frame_name,
    ee_is_site=ee_is_site,
    asset_cfg=asset_cfg,
    use_step_cache=True,
  )

  pos_rew = torch.exp(-torch.square(pos_err / position_std))
  ori_rew = torch.exp(-torch.square(ori_err / orientation_std))
  reward = pos_rew * ori_rew

  return reward


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    if "target_joint_pos" in cfg.params:
      _, _, target = resolve_matching_names_values(
        data=cfg.params["target_joint_pos"],
        list_of_strings=joint_names,
      )
      self.default_joint_pos = torch.tensor(
        target, device=env.device, dtype=torch.float32
      ).unsqueeze(0)  # match batch
    else:
      self.default_joint_pos = default_joint_pos

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
    target_joint_pos=None,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def gripper_tracking(
  env: ManagerBasedRlEnv,
  target: float = 0.0,
  std: float = 0.002,
  asset_cfg: SceneEntityCfg = _DEFAULT_GRIPPER_ASSET_CFG,
) -> torch.Tensor:
  """Reward high when gripper joint stays near target_open."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)  # [B]
  err2 = torch.square(q - target)
  return torch.exp(-err2 / (std**2))


def ee_linear_velocity_tracking(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  site_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  robot: Entity = env.scene[asset_cfg.name]
  ee_ids, _ = robot.find_sites(site_name)
  ee_id = ee_ids[0]
  ee_quat_w = robot.data.site_quat_w[:, ee_id]
  actual = quat_apply_inverse(ee_quat_w, robot.data.site_vel_w[:, ee_id, :3])

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  command_lin_vel = command[:, [0, 1, 3]]
  lin_vel_error = torch.sum(torch.square(command_lin_vel - actual), dim=1)
  return torch.exp(-lin_vel_error / std**2)


def ee_angular_velocity_tracking(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  site_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  robot: Entity = env.scene[asset_cfg.name]
  ee_ids, _ = robot.find_sites(site_name)
  ee_id = ee_ids[0]
  ee_quat_w = robot.data.site_quat_w[:, ee_id]
  actual = quat_apply_inverse(ee_quat_w, robot.data.site_vel_w[:, ee_id, 3:])

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  command_ang_vel = command[:, [4, 5, 2]]
  ang_vel_error = torch.sum(torch.square(command_ang_vel - actual), dim=1)
  return torch.exp(-ang_vel_error / std**2)


def base_lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize vertical base linear velocity in world frame."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_w[:, 2])


def base_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base roll and pitch angular velocity in the base frame."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
