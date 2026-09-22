from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.differential_ik import (
  DifferentialIKAction,
  DifferentialIKActionCfg,
)

# from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationManager
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from mjlab.utils.lab_api.string import resolve_matching_names_values
from rsl_rl.runners import OnPolicyRunner

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class CommandDrivenDifferentialIKAction(ActionTerm):
  """Zero-dim arm controller driven by a command-manager EE pose command.

  The term contributes no policy action dimensions. It runs inside the action
  manager, so impedance effort is recomputed every decimation substep.
  """

  cfg: "CommandDrivenDifferentialIKActionCfg"

  def __init__(
    self, cfg: "CommandDrivenDifferentialIKActionCfg", env: ManagerBasedRlEnv
  ) -> None:
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
    self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
    self._command = torch.zeros(self.num_envs, self._ik.action_dim, device=self.device)

    joint_names = tuple(
      self._ik._entity.joint_names[int(i)] for i in self._ik._joint_ids
    )
    self._kp = self._resolve_gain(cfg.kp_ff, joint_names)
    self._kd = self._resolve_gain(cfg.kd_ff, joint_names)

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.shape[-1] != 0:
      raise ValueError(
        f"{self.__class__.__name__} expects zero action dims, got {actions.shape[-1]}."
      )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
    self._command[env_ids] = 0.0
    self._ik.reset(env_ids)

  def apply_actions(self) -> None:
    src = self._env.command_manager.get_command(self.cfg.source_command_name)
    if src.shape[1] < self._ik.action_dim:
      raise ValueError(
        f"Command '{self.cfg.source_command_name}' has dim={src.shape[1]}, "
        f"but IK expects dim={self._ik.action_dim}."
      )

    self._command[:] = src[:, : self._ik.action_dim]
    self._ik.process_actions(self._command)
    q_current = self._ik._entity.data.joint_pos[:, self._ik._joint_ids].clone()
    q_target = self._compute_joint_target(q_current)
    self._apply_joint_target(q_current, q_target)

  def _compute_joint_target(self, q_current: torch.Tensor) -> torch.Tensor:
    q_target = q_current.clone()
    n_iter = max(1, int(self.cfg.ik_iterations))
    if n_iter == 1:
      return q_target + self._ik.compute_dq()

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

    if self.cfg.target_delta_limit is not None:
      delta = (q_target - q_current).clamp(
        -self.cfg.target_delta_limit, self.cfg.target_delta_limit
      )
      q_target = q_current + delta
    return q_target

  def _apply_joint_target(
    self, q_current: torch.Tensor, q_target: torch.Tensor
  ) -> None:
    if self.cfg.control_mode == "position":
      self._ik._entity.set_joint_position_target(
        q_target, joint_ids=self._ik._joint_ids
      )
      return

    if self.cfg.control_mode != "impedance":
      raise ValueError(f"Unknown control_mode: {self.cfg.control_mode}")

    qd = self._ik._entity.data.joint_vel[:, self._ik._joint_ids]
    tau_cmd = self._kp * (q_target - q_current) - self._kd * qd
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


@dataclass(kw_only=True)
class CommandDrivenDifferentialIKActionCfg(ActionTermCfg):
  """Configuration for zero-dim command-driven Differential IK action."""

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
  target_delta_limit: float | None = None
  control_mode: Literal["position", "impedance"] = "impedance"
  kp_ff: float | dict[str, float] = 0.0
  kd_ff: float | dict[str, float] = 0.0
  gravity_compensation: bool = True
  inertial_compensation: bool = False

  def build(self, env: ManagerBasedRlEnv) -> CommandDrivenDifferentialIKAction:
    return CommandDrivenDifferentialIKAction(self, env)


class PreTrainedPolicyAction(ActionTerm):
  """
  Pre-trained policy action term.

  This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
  The raw actions correspond to the commands for the pre-trained policy.
  """

  cfg: PreTrainedPolicyActionCfg

  def __init__(self, cfg: PreTrainedPolicyActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    # load policy
    if cfg.policy_path is None:
      raise ValueError("policy_path must be specified for PreTrainedPolicyAction.")
    if not Path(cfg.policy_path).exists():
      raise FileNotFoundError(f"Pre-trained policy file not found: {cfg.policy_path}")
    print(f"[INFO]: Loading pre-trained policy from: {cfg.policy_path}")

    self.policy = self.load_policy_from_checkpoint(
      task_id=cfg.task_id, policy_path=cfg.policy_path, device=env.device
    )

    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)
    self._action_clip = None
    if cfg.action_clip is not None:
      if len(cfg.action_clip) != self.action_dim:
        raise ValueError(
          f"action_clip must have {self.action_dim} entries, got {len(cfg.action_clip)}."
        )
      self._action_clip = torch.tensor(
        cfg.action_clip, dtype=self._raw_actions.dtype, device=self.device
      ).unsqueeze(0)
    self._action_deadzone = None
    if cfg.action_deadzone is not None:
      if len(cfg.action_deadzone) != self.action_dim:
        raise ValueError(
          "action_deadzone must have "
          f"{self.action_dim} entries, got {len(cfg.action_deadzone)}."
        )
      if any(threshold < 0.0 for threshold in cfg.action_deadzone):
        raise ValueError("action_deadzone entries must be non-negative.")
      self._action_deadzone = torch.tensor(
        cfg.action_deadzone, dtype=self._raw_actions.dtype, device=self.device
      ).unsqueeze(0)

    # build low-level action terms
    self._low_level_terms = {}
    for name, action_cfg in cfg.low_level_actions.items():
      self._low_level_terms[name] = action_cfg.build(env)

    self.low_level_action_dim = sum(
      term.action_dim for term in self._low_level_terms.values()
    )
    self.low_level_actions = torch.zeros(
      self.num_envs, self.low_level_action_dim, device=self.device
    )

    def last_action():
      # reset the low level actions if the episode was reset
      if hasattr(env, "episode_length_buf"):
        self.low_level_actions[env.episode_length_buf == 0, :] = 0
      return self.low_level_actions

    # build low-level observation manager
    # change the command in the policy terms and critic terms
    cfg.low_level_observations.pop("critic")
    cfg.low_level_observations["actor"].terms["actions"].func = (
      lambda dummy_env: last_action()
    )
    cfg.low_level_observations["actor"].terms["actions"].params = {}
    cfg.low_level_observations["actor"].terms["command"].func = (
      lambda dummy_env: self._processed_actions
    )
    cfg.low_level_observations["actor"].terms["command"].params = {}

    self._low_level_obs_manager = ObservationManager(cfg.low_level_observations, env)

  def load_policy_from_checkpoint(
    self, task_id: str, policy_path: str, device: str = "cpu"
  ):
    # replicate the strategy of play.py
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(policy_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    return policy

  """
    Properties.
    """

  @property
  def action_dim(self) -> int:
    return self.cfg.action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  """
    Operations    
  """

  def apply_actions(self) -> None:
    # infer low-level policy by getting observations
    low_level_obs = self._low_level_obs_manager.compute(update_history=True)
    self.low_level_actions[:] = self.policy(low_level_obs)
    # apply low-level actions to robot
    start_idx = 0
    for term in self._low_level_terms.values():
      end_idx = start_idx + term.action_dim
      term.process_actions(self.low_level_actions[:, start_idx:end_idx])
      start_idx = end_idx
    for term in self._low_level_terms.values():
      term.apply_actions()

  def process_actions(self, actions: torch.Tensor):
    """Store raw actions and process commands sent to the lower-level policy."""
    self._raw_actions[:] = actions
    if self._action_clip is None:
      self._processed_actions[:] = actions
    else:
      self._processed_actions[:] = torch.clamp(
        actions, min=-self._action_clip, max=self._action_clip
      )
    if self._action_deadzone is not None:
      self._processed_actions[:] = torch.where(
        torch.abs(self._processed_actions) <= self._action_deadzone,
        torch.zeros_like(self._processed_actions),
        self._processed_actions,
      )
    # for debugging, only standing
    # self._raw_actions[:] = torch.zeros(self.num_envs, self.action_dim, device=self.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset upper-level actions to zero for specified environments."""
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0


class PoseSettledPreTrainedPolicyAction(PreTrainedPolicyAction):
  """Suppress pre-trained policy commands while EE pose tracking is settled."""

  cfg: "PoseSettledPreTrainedPolicyActionCfg"

  def __init__(
    self, cfg: "PoseSettledPreTrainedPolicyActionCfg", env: ManagerBasedRlEnv
  ) -> None:
    if cfg.position_enter < 0.0 or cfg.orientation_enter < 0.0:
      raise ValueError("Pose settling enter thresholds must be non-negative.")
    if cfg.position_exit < cfg.position_enter:
      raise ValueError("position_exit must be greater than or equal to position_enter.")
    if cfg.orientation_exit < cfg.orientation_enter:
      raise ValueError(
        "orientation_exit must be greater than or equal to orientation_enter."
      )

    super().__init__(cfg, env)
    self._base_body_id = self.robot.find_bodies(cfg.base_body_name)[0][0]
    if cfg.ee_is_site:
      self._ee_frame_id = self.robot.find_sites(cfg.ee_frame_name)[0][0]
    else:
      self._ee_frame_id = self.robot.find_bodies(cfg.ee_frame_name)[0][0]
    self._position_enter_sq = cfg.position_enter**2
    self._position_exit_sq = cfg.position_exit**2
    self._orientation_enter_cos_sq = math.cos(cfg.orientation_enter / 2.0) ** 2
    self._orientation_exit_cos_sq = math.cos(cfg.orientation_exit / 2.0) ** 2
    self._pose_settled = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    """Process commands and suppress them inside the settled hysteresis region."""
    super().process_actions(actions)

    position_error_sq, orientation_dot_sq, orientation_norm_product = (
      self._pose_error_comparison_terms()
    )
    enter_settled = (position_error_sq <= self._position_enter_sq) & (
      orientation_dot_sq >= self._orientation_enter_cos_sq * orientation_norm_product
    )
    exit_settled = (position_error_sq >= self._position_exit_sq) | (
      orientation_dot_sq <= self._orientation_exit_cos_sq * orientation_norm_product
    )

    self._pose_settled = torch.where(self._pose_settled, ~exit_settled, enter_settled)
    self._processed_actions.masked_fill_(self._pose_settled.unsqueeze(-1), 0.0)

  def _pose_error_comparison_terms(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return comparison terms without materializing pose-error magnitudes."""
    command = self._env.command_manager.get_command(self.cfg.pose_command_name)
    if command.shape[1] == 6:
      position_error_sq = torch.sum(torch.square(command[:, :3]), dim=-1)
      orientation_error_sq = torch.sum(torch.square(command[:, 3:]), dim=-1)
      # Encode a rotation-vector error so it uses the same threshold comparisons
      # as the quaternion path in ``process_actions``.
      orientation_dot_sq = torch.cos(0.5 * torch.sqrt(orientation_error_sq)) ** 2
      orientation_norm_product = torch.ones_like(orientation_dot_sq)
      return position_error_sq, orientation_dot_sq, orientation_norm_product

    if command.shape[1] < 7:
      raise ValueError(
        f"Unsupported ee pose command shape {tuple(command.shape)} for "
        f"'{self.cfg.pose_command_name}'."
      )

    base_pos_w = self.robot.data.body_link_pos_w[:, self._base_body_id]
    base_quat_w = self.robot.data.body_link_quat_w[:, self._base_body_id]
    if self.cfg.ee_is_site:
      ee_pos_w = self.robot.data.site_pos_w[:, self._ee_frame_id]
      ee_quat_w = self.robot.data.site_quat_w[:, self._ee_frame_id]
    else:
      ee_pos_w = self.robot.data.body_link_pos_w[:, self._ee_frame_id]
      ee_quat_w = self.robot.data.body_link_quat_w[:, self._ee_frame_id]

    target_pos_w = base_pos_w + quat_apply(base_quat_w, command[:, :3])
    target_quat_w = quat_mul(base_quat_w, command[:, 3:7])
    position_error_sq = torch.sum(torch.square(target_pos_w - ee_pos_w), dim=-1)
    orientation_dot = torch.sum(target_quat_w * ee_quat_w, dim=-1)
    orientation_dot_sq = torch.square(orientation_dot)
    orientation_norm_product = torch.sum(
      torch.square(target_quat_w), dim=-1
    ) * torch.sum(torch.square(ee_quat_w), dim=-1)
    return position_error_sq, orientation_dot_sq, orientation_norm_product

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset commands and per-environment settling state."""
    super().reset(env_ids)
    self._pose_settled[env_ids] = False


class ResidualPolicyAction(ActionTerm):
  """Residual high-level controller over a fixed pre-trained base policy.

  The frozen base policy receives its normal low-level observations and outputs
  low-level actions.
  The trainable policy outputs a residual action in that same low-level action
  space, and the final applied action is:

    final_low_level_action = base_policy_action + residual_action
  """

  cfg: "ResidualPolicyActionCfg"

  def __init__(self, cfg: ResidualPolicyActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    if cfg.policy_path is None:
      raise ValueError("policy_path must be specified for ResidualPolicyAction.")
    if not Path(cfg.policy_path).exists():
      raise FileNotFoundError(f"Pre-trained policy file not found: {cfg.policy_path}")
    print(f"[INFO]: Loading pre-trained policy from: {cfg.policy_path}")
    self.policy = self.load_policy_from_checkpoint(
      task_id=cfg.task_id, policy_path=cfg.policy_path, device=env.device
    )

    # build low-level action terms
    self._low_level_terms = {}
    for name, action_cfg in cfg.low_level_actions.items():
      self._low_level_terms[name] = action_cfg.build(env)

    self._low_level_action_dim = sum(
      term.action_dim for term in self._low_level_terms.values()
    )
    self.low_level_actions = torch.zeros(
      self.num_envs, self._low_level_action_dim, device=self.device
    )

    # Base policy observation manager with original task observations.
    low_level_obs_cfg = deepcopy(cfg.low_level_observations)
    low_level_obs_cfg.pop("critic")

    def last_action():
      if hasattr(env, "episode_length_buf"):
        self.low_level_actions[env.episode_length_buf == 0, :] = 0
      return self.low_level_actions

    low_level_obs_cfg["actor"].terms["actions"].func = lambda dummy_env: last_action()
    low_level_obs_cfg["actor"].terms["actions"].params = {}
    self._low_level_obs_manager = ObservationManager(low_level_obs_cfg, env)

    # Residual action lives in the same space as base policy output.
    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)
    self._residual_scale = self._to_vector(
      cfg.residual_scale, "residual_scale", expected_dim=self.action_dim
    )

  def load_policy_from_checkpoint(
    self, task_id: str, policy_path: str, device: str = "cpu"
  ):
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(policy_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    return policy

  @property
  def action_dim(self) -> int:
    return self._low_level_action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def _to_vector(
    self, value: float | tuple[float, ...], name: str, expected_dim: int
  ) -> torch.Tensor:
    """Convert scalar or sized tuple to shape (1, expected_dim)."""
    if isinstance(value, (float, int)):
      return torch.full((1, expected_dim), float(value), device=self.device)
    if len(value) != expected_dim:
      raise ValueError(f"{name} must have length {expected_dim}, got {len(value)}.")
    return torch.tensor(value, device=self.device, dtype=torch.float32).view(1, -1)

  def process_actions(self, actions: torch.Tensor):
    """Store residual action from the trainable policy."""
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._residual_scale

    if self.cfg.residual_clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions,
        min=self.cfg.residual_clip[0],
        max=self.cfg.residual_clip[1],
      )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0

  def apply_actions(self) -> None:
    # 1) Frozen base policy action from its regular low-level observations.
    low_level_obs = self._low_level_obs_manager.compute(update_history=True)
    base_actions = self.policy(low_level_obs)

    # 2) Add trainable residual action on top of base policy action.
    combined = base_actions + self._processed_actions
    if self.cfg.combined_clip is not None:
      combined = torch.clamp(
        combined,
        min=self.cfg.combined_clip[0],
        max=self.cfg.combined_clip[1],
      )

    self.low_level_actions[:] = combined
    start_idx = 0
    for term in self._low_level_terms.values():
      end_idx = start_idx + term.action_dim
      term.process_actions(self.low_level_actions[:, start_idx:end_idx])
      start_idx = end_idx
    for term in self._low_level_terms.values():
      term.apply_actions()


@dataclass(kw_only=True)
class PreTrainedPolicyActionCfg(ActionTermCfg):
  """
  Configuration for pre-trained policy action term.

  See :class:`PreTrainedPolicyAction` for more details.
  """

  task_id: str
  entity_name: str
  policy_path: str
  action_dim: int = 3
  low_level_decimation: int = 4
  low_level_actions: dict[str, ActionTermCfg]
  low_level_observations: dict[str, ObservationGroupCfg]
  action_clip: tuple[float, ...] | None = None
  action_deadzone: tuple[float, ...] | None = None
  debug_vis: bool = False

  def build(self, env: ManagerBasedRlEnv) -> PreTrainedPolicyAction:
    return PreTrainedPolicyAction(self, env)


@dataclass(kw_only=True)
class PoseSettledPreTrainedPolicyActionCfg(PreTrainedPolicyActionCfg):
  """Pre-trained policy action with hysteretic EE pose settling."""

  pose_command_name: str
  base_body_name: str
  ee_frame_name: str
  ee_is_site: bool = True
  position_enter: float = 0.03
  position_exit: float = 0.05
  orientation_enter: float = math.radians(5.0)
  orientation_exit: float = math.radians(8.0)

  def build(self, env: ManagerBasedRlEnv) -> PoseSettledPreTrainedPolicyAction:
    return PoseSettledPreTrainedPolicyAction(self, env)


@dataclass(kw_only=True)
class ResidualPolicyActionCfg(ActionTermCfg):
  """Configuration for residual policy over a fixed pre-trained base policy."""

  task_id: str
  entity_name: str
  policy_path: str
  low_level_decimation: int = 4
  low_level_actions: dict[str, ActionTermCfg]
  low_level_observations: dict[str, ObservationGroupCfg]
  debug_vis: bool = False
  residual_scale: float | tuple[float, ...] = 1.0
  residual_clip: tuple[float, float] | None = None
  combined_clip: tuple[float, float] | None = None

  def build(self, env: ManagerBasedRlEnv) -> ResidualPolicyAction:
    return ResidualPolicyAction(self, env)
