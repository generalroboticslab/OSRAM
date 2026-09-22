from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot", joint_names=("gripper_joint",))


def gripper_commanded_position(
  env: ManagerBasedRlEnv,
  action_name: str = "gripper_pos",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-step applied gripper position target (after encoder-bias compensation)."""
  term = env.action_manager.get_term(action_name)
  asset: Entity = env.scene[asset_cfg.name]
  target_ids = term.target_ids
  processed = term._processed_actions[:, 0]
  encoder_bias = asset.data.encoder_bias[:, target_ids].squeeze(-1)
  return processed - encoder_bias


def gripper_real_position(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-step measured gripper joint position."""
  asset: Entity = env.scene[asset_cfg.name]
  # Robustly pick gripper_joint by exact name, so we always return shape [B].
  joint_ids, _ = asset.find_joints("gripper_joint")
  return asset.data.joint_pos[:, joint_ids[0]]


def action_term_value(
  env: ManagerBasedRlEnv,
  action_name: str,
  action_index: int,
) -> torch.Tensor:
  """Return one scalar component from an action term's raw action."""
  term = env.action_manager.get_term(action_name)
  return term.raw_action[:, action_index]


def processed_action_term_value(
  env: ManagerBasedRlEnv,
  action_name: str,
  action_index: int,
) -> torch.Tensor:
  """Return an action component after term-specific command processing."""
  term = env.action_manager.get_term(action_name)
  if not hasattr(term, "_processed_actions"):
    raise TypeError(f"Action term '{action_name}' has no processed actions.")
  return term._processed_actions[:, action_index]
