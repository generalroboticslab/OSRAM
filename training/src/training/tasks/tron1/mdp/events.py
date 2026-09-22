from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from training.tasks.tron1.mdp.actions import (
  CommandDrivenDifferentialIKAction,
)
from training.tasks.tron1.mdp.commands import (
  PseudoInverseVelocityCommand,
)
from training.tasks.tron1.mdp.impedance_actions import (
  JointImpedanceAction,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def randomize_joint_impedance_gains(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  action_name: str,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize custom JointImpedance action gains.

  This modifies action-term internal gains (`_kp`, `_kd`), not sim actuator gainprm.
  """
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, JointImpedanceAction):
    raise TypeError(
      f"Action '{action_name}' must be JointImpedanceAction, got {type(term)}"
    )

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)

  # Cache defaults once.
  if not hasattr(term, "_default_kp"):
    term._default_kp = term._kp.clone()
  if not hasattr(term, "_default_kd"):
    term._default_kd = term._kd.clone()

  n_env = len(env_ids)
  action_dim = term._kp.shape[1]

  if distribution == "uniform":
    kp_samples = torch.empty((n_env, action_dim), device=env.device).uniform_(*kp_range)
    kd_samples = torch.empty((n_env, action_dim), device=env.device).uniform_(*kd_range)
  elif distribution == "log_uniform":
    kp_log = torch.empty((n_env, action_dim), device=env.device).uniform_(
      torch.log(torch.tensor(kp_range[0], device=env.device)),
      torch.log(torch.tensor(kp_range[1], device=env.device)),
    )
    kd_log = torch.empty((n_env, action_dim), device=env.device).uniform_(
      torch.log(torch.tensor(kd_range[0], device=env.device)),
      torch.log(torch.tensor(kd_range[1], device=env.device)),
    )
    kp_samples = torch.exp(kp_log)
    kd_samples = torch.exp(kd_log)
  else:
    raise ValueError(f"Unsupported distribution: {distribution}")

  if operation == "scale":
    term._kp[env_ids] = term._default_kp[env_ids] * kp_samples
    term._kd[env_ids] = term._default_kd[env_ids] * kd_samples
  elif operation == "abs":
    term._kp[env_ids] = kp_samples
    term._kd[env_ids] = kd_samples
  else:
    raise ValueError(f"Unsupported operation: {operation}")


def randomize_pseudo_inverse_command_kd(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  command_name: str,
  kd_range: tuple[float, float],
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize PseudoInverseVelocityCommand damping gains.

  This modifies the command term's internal `_kd`, not MuJoCo actuator gainprm.
  """
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, PseudoInverseVelocityCommand):
    raise TypeError(
      f"Command '{command_name}' must be PseudoInverseVelocityCommand, got {type(term)}"
    )

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)

  if not hasattr(term, "_default_kd"):
    term._default_kd = term._kd.clone()

  n_env = len(env_ids)
  command_dim = term._kd.shape[1]

  if distribution == "uniform":
    kd_samples = torch.empty((n_env, command_dim), device=env.device).uniform_(
      *kd_range
    )
  elif distribution == "log_uniform":
    kd_log = torch.empty((n_env, command_dim), device=env.device).uniform_(
      torch.log(torch.tensor(kd_range[0], device=env.device)),
      torch.log(torch.tensor(kd_range[1], device=env.device)),
    )
    kd_samples = torch.exp(kd_log)
  else:
    raise ValueError(f"Unsupported distribution: {distribution}")

  if operation == "scale":
    term._kd[env_ids] = term._default_kd[env_ids] * kd_samples
  elif operation == "abs":
    term._kd[env_ids] = kd_samples
  else:
    raise ValueError(f"Unsupported operation: {operation}")


def randomize_command_driven_ik_gains(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  action_name: str,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize CommandDrivenDifferentialIKAction impedance gains per env and joint.

  This modifies the action term's internal `_kp` and `_kd`, not sim actuator
  gain parameters.
  """
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, CommandDrivenDifferentialIKAction):
    raise TypeError(
      f"Action '{action_name}' must be CommandDrivenDifferentialIKAction, got {type(term)}"
    )

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)

  if not hasattr(term, "_default_kp"):
    term._default_kp = term._kp.clone()
  if not hasattr(term, "_default_kd"):
    term._default_kd = term._kd.clone()

  n_env = len(env_ids)
  action_dim = term._kp.shape[1]

  if distribution == "uniform":
    kp_samples = torch.empty((n_env, action_dim), device=env.device).uniform_(*kp_range)
    kd_samples = torch.empty((n_env, action_dim), device=env.device).uniform_(*kd_range)
  elif distribution == "log_uniform":
    kp_log = torch.empty((n_env, action_dim), device=env.device).uniform_(
      torch.log(torch.tensor(kp_range[0], device=env.device)),
      torch.log(torch.tensor(kp_range[1], device=env.device)),
    )
    kd_log = torch.empty((n_env, action_dim), device=env.device).uniform_(
      torch.log(torch.tensor(kd_range[0], device=env.device)),
      torch.log(torch.tensor(kd_range[1], device=env.device)),
    )
    kp_samples = torch.exp(kp_log)
    kd_samples = torch.exp(kd_log)
  else:
    raise ValueError(f"Unsupported distribution: {distribution}")

  if operation == "scale":
    term._kp[env_ids] = term._default_kp[env_ids] * kp_samples
    term._kd[env_ids] = term._default_kd[env_ids] * kd_samples
  elif operation == "abs":
    term._kp[env_ids] = kp_samples
    term._kd[env_ids] = kd_samples
  else:
    raise ValueError(f"Unsupported operation: {operation}")


class ee_force_disturbance:
  """Apply a slowly changing noisy force to an end-effector site.

  MuJoCo applies external Cartesian wrenches to bodies, so this term resolves the
  body that owns the configured site and writes the force to that body.
  """

  def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRlEnv"):
    self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self.force_range: tuple[tuple[float, float], ...] = cfg.params["force_range"]
    self.profile_time_range_s: tuple[float, float] = cfg.params["profile_time_range_s"]
    self.noise_std: float = cfg.params["noise_std"]
    self.viz_force_scale: float = cfg.params.get("viz_force_scale", 0.03)
    self.viz_arrow_width: float = cfg.params.get("viz_arrow_width", 0.015)
    self.viz_color: tuple[float, float, float, float] = cfg.params.get(
      "viz_color", (1.0, 0.1, 0.1, 0.85)
    )

    self.asset = env.scene[self.asset_cfg.name]
    if isinstance(self.asset_cfg.site_ids, slice):
      raise ValueError("ee_force_disturbance requires exactly one resolved site.")
    if len(self.asset_cfg.site_ids) != 1:
      raise ValueError(
        f"ee_force_disturbance requires one site, got {len(self.asset_cfg.site_ids)}"
      )

    self.site_local_id = self.asset_cfg.site_ids[0]
    site_global_id = self.asset.indexing.site_ids[self.site_local_id]
    site_body_global_id = self.asset.data.model.site_bodyid[site_global_id]
    body_matches = (self.asset.indexing.body_ids == site_body_global_id).nonzero()
    if len(body_matches) != 1:
      raise ValueError(
        "Could not resolve the unique body that owns the disturbed site "
        f"{self.asset_cfg.site_names}."
      )
    self.body_local_id = int(body_matches[0].item())

    self.force_low = torch.tensor(
      [axis_range[0] for axis_range in self.force_range],
      dtype=torch.float32,
      device=env.device,
    )
    self.force_high = torch.tensor(
      [axis_range[1] for axis_range in self.force_range],
      dtype=torch.float32,
      device=env.device,
    )
    self.force = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    self.applied_force = torch.zeros_like(self.force)
    self.time_left_s = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    self.reset(env_ids=None)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
      count = self.force.shape[0]
    elif isinstance(env_ids, slice):
      count = self.force[env_ids].shape[0]
    else:
      count = len(env_ids)

    self.force[env_ids] = self._sample_force(count)
    self.time_left_s[env_ids] = self._sample_profile_time(count)

  def __call__(self, env: "ManagerBasedRlEnv", env_ids, **kwargs) -> None:
    del kwargs  # Parameters are cached in __init__; EventManager still passes them.
    del env_ids  # Step events run for all environments.
    self.time_left_s -= env.step_dt
    expired_env_ids = (self.time_left_s <= 0.0).nonzero().flatten()
    if len(expired_env_ids) > 0:
      self.reset(expired_env_ids)

    noisy_force = self.force + torch.randn_like(self.force) * self.noise_std
    self.applied_force[:] = noisy_force
    site_pos_w = self.asset.data.site_pos_w[:, self.site_local_id]
    body_com_pos_w = self.asset.data.body_com_pos_w[:, self.body_local_id]
    torque = torch.cross(site_pos_w - body_com_pos_w, noisy_force, dim=-1)
    self.asset.write_external_wrench_to_sim(
      forces=noisy_force.unsqueeze(1),
      torques=torque.unsqueeze(1),
      body_ids=[self.body_local_id],
    )

  def _sample_force(self, count: int) -> torch.Tensor:
    return (
      torch.rand((count, 3), device=self.force.device)
      * (self.force_high - self.force_low)
      + self.force_low
    )

  def _sample_profile_time(self, count: int) -> torch.Tensor:
    low, high = self.profile_time_range_s
    return torch.rand(count, device=self.force.device) * (high - low) + low

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.force.shape[0])
    if not env_indices:
      return

    for env_id in env_indices:
      start = self.asset.data.site_pos_w[env_id, self.site_local_id].cpu().numpy()
      force = self.applied_force[env_id].cpu().numpy()
      end = start + force * self.viz_force_scale
      visualizer.add_arrow(
        start=start,
        end=end,
        color=self.viz_color,
        width=self.viz_arrow_width,
        label=f"ee_force_disturbance_{env_id}",
      )
