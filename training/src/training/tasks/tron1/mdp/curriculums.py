from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from training.tasks.tron1.mdp.actions import ResidualPolicyAction
from training.tasks.tron1.mdp.commands import (
  Uniform3DVelocityCommandCfg,
  UniformPoseCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class PDGainStage(TypedDict):
  step: int
  kp_scale: tuple[float, float] | None
  kd_scale: tuple[float, float] | None


def pd_gain(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  randomization_name: str,
  pd_stages: PDGainStage,
):
  """Update the PD gains based on the curriculum stages."""
  del env_ids

  randomization_cfg = env.event_manager.get_term_cfg(randomization_name)

  current_kp_scale = randomization_cfg.params.get("kp_range", (1.0, 1.0))
  current_kd_scale = randomization_cfg.params.get("kd_range", (1.0, 1.0))

  for stage in pd_stages:
    if env.common_step_counter >= stage["step"]:
      if stage["kp_scale"] is not None:
        randomization_cfg.params["kp_range"] = stage["kp_scale"]
        current_kp_scale = stage["kp_scale"]
      if stage["kd_scale"] is not None:
        randomization_cfg.params["kd_range"] = stage["kd_scale"]
        current_kd_scale = stage["kd_scale"]

  # Return metrics for logging
  return {
    "kp_scale_min": torch.tensor(current_kp_scale[0]),
    "kp_scale_max": torch.tensor(current_kp_scale[1]),
    "kd_scale_min": torch.tensor(current_kd_scale[0]),
    "kd_scale_max": torch.tensor(current_kd_scale[1]),
  }


class ResidualClipStage(TypedDict):
  step: int
  clip: tuple[float, float] | None  # e.g. (-0.1, 0.1) -> ... -> (-1.0, 1.0)


def residual_clip_curriculum(
  env,
  env_ids: torch.Tensor,
  action_name: str,
  stages: list[ResidualClipStage],
):
  del env_ids  # global term config update

  action_term = env.action_manager.get_term(action_name)
  if not isinstance(action_term, ResidualPolicyAction):
    raise TypeError(
      f"Action '{action_name}' must be ResidualPolicyAction, got {type(action_term)}"
    )

  current = action_term.cfg.residual_clip
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      action_term.cfg.residual_clip = stage["clip"]
      current = stage["clip"]

  if current is None:
    return {"enabled": torch.tensor(0.0)}
  return {
    "enabled": torch.tensor(1.0),
    "clip_min": torch.tensor(current[0]),
    "clip_max": torch.tensor(current[1]),
  }


class EEPoseStage(TypedDict):
  step: int
  x: tuple[float, float] | None
  y: tuple[float, float] | None
  z: tuple[float, float] | None
  roll: tuple[float, float] | None
  pitch: tuple[float, float] | None
  yaw: tuple[float, float] | None


def commands_ee_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  ee_pose_stages: list[EEPoseStage],
) -> dict[str, torch.Tensor]:
  """Curriculum for progressively broadening EE pose command ranges."""
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformPoseCommandCfg, command_term.cfg)

  for stage in ee_pose_stages:
    if env.common_step_counter >= stage["step"]:
      if "x" in stage and stage["x"] is not None:
        cfg.ranges.x = stage["x"]
      if "y" in stage and stage["y"] is not None:
        cfg.ranges.y = stage["y"]
      if "z" in stage and stage["z"] is not None:
        cfg.ranges.z = stage["z"]
      if "roll" in stage and stage["roll"] is not None:
        cfg.ranges.roll = stage["roll"]
      if "pitch" in stage and stage["pitch"] is not None:
        cfg.ranges.pitch = stage["pitch"]
      if "yaw" in stage and stage["yaw"] is not None:
        cfg.ranges.yaw = stage["yaw"]

  return {
    "ee_x_min": torch.tensor(cfg.ranges.x[0]),
    "ee_x_max": torch.tensor(cfg.ranges.x[1]),
    "ee_y_min": torch.tensor(cfg.ranges.y[0]),
    "ee_y_max": torch.tensor(cfg.ranges.y[1]),
    "ee_z_min": torch.tensor(cfg.ranges.z[0]),
    "ee_z_max": torch.tensor(cfg.ranges.z[1]),
    "ee_roll_min": torch.tensor(cfg.ranges.roll[0]),
    "ee_roll_max": torch.tensor(cfg.ranges.roll[1]),
    "ee_pitch_min": torch.tensor(cfg.ranges.pitch[0]),
    "ee_pitch_max": torch.tensor(cfg.ranges.pitch[1]),
    "ee_yaw_min": torch.tensor(cfg.ranges.yaw[0]),
    "ee_yaw_max": torch.tensor(cfg.ranges.yaw[1]),
  }


class EEVelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  lin_vel_z: tuple[float, float] | None
  ang_vel_x: tuple[float, float] | None
  ang_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def commands_ee_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[EEVelocityStage],
) -> dict[str, torch.Tensor]:
  """Curriculum for progressively broadening 3D EE velocity command ranges."""
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(Uniform3DVelocityCommandCfg, command_term.cfg)

  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "lin_vel_z" in stage and stage["lin_vel_z"] is not None:
        cfg.ranges.lin_vel_z = stage["lin_vel_z"]
      if "ang_vel_x" in stage and stage["ang_vel_x"] is not None:
        cfg.ranges.ang_vel_x = stage["ang_vel_x"]
      if "ang_vel_y" in stage and stage["ang_vel_y"] is not None:
        cfg.ranges.ang_vel_y = stage["ang_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]

  return {
    "ee_lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "ee_lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "ee_lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "ee_lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ee_lin_vel_z_min": torch.tensor(cfg.ranges.lin_vel_z[0]),
    "ee_lin_vel_z_max": torch.tensor(cfg.ranges.lin_vel_z[1]),
    "ee_ang_vel_x_min": torch.tensor(cfg.ranges.ang_vel_x[0]),
    "ee_ang_vel_x_max": torch.tensor(cfg.ranges.ang_vel_x[1]),
    "ee_ang_vel_y_min": torch.tensor(cfg.ranges.ang_vel_y[0]),
    "ee_ang_vel_y_max": torch.tensor(cfg.ranges.ang_vel_y[1]),
    "ee_ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ee_ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


class ActionPDStage(TypedDict):
  step: int
  kp_scale: tuple[float, float] | None
  kd_scale: tuple[float, float] | None


def joint_impedance_pd_curriculum(
  env,
  env_ids: torch.Tensor,
  randomization_name: str,
  pd_stages: list[ActionPDStage],
):
  del env_ids
  randomization_cfg = env.event_manager.get_term_cfg(randomization_name)

  current_kp = randomization_cfg.params.get("kp_range", (1.0, 1.0))
  current_kd = randomization_cfg.params.get("kd_range", (1.0, 1.0))

  for stage in pd_stages:
    if env.common_step_counter >= stage["step"]:
      if stage["kp_scale"] is not None:
        randomization_cfg.params["kp_range"] = stage["kp_scale"]
        current_kp = stage["kp_scale"]
      if stage["kd_scale"] is not None:
        randomization_cfg.params["kd_range"] = stage["kd_scale"]
        current_kd = stage["kd_scale"]

  return {
    "imp_kp_min": torch.tensor(current_kp[0]),
    "imp_kp_max": torch.tensor(current_kp[1]),
    "imp_kd_min": torch.tensor(current_kd[0]),
    "imp_kd_max": torch.tensor(current_kd[1]),
  }


class CommandKDStage(TypedDict):
  step: int
  kd_scale: tuple[float, float]


def pseudo_inverse_command_kd_curriculum(
  env,
  env_ids: torch.Tensor,
  randomization_name: str,
  kd_stages: list[CommandKDStage],
):
  del env_ids
  randomization_cfg = env.event_manager.get_term_cfg(randomization_name)

  current_kd = randomization_cfg.params.get("kd_range", (1.0, 1.0))
  for stage in kd_stages:
    if env.common_step_counter >= stage["step"]:
      randomization_cfg.params["kd_range"] = stage["kd_scale"]
      current_kd = stage["kd_scale"]

  return {
    "pseudo_inv_kd_min": torch.tensor(current_kd[0]),
    "pseudo_inv_kd_max": torch.tensor(current_kd[1]),
  }
