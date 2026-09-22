from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from mjlab.rl import RslRlVecEnvWrapper

TensorLike = torch.Tensor | np.ndarray
SUPPORTED_EXPLORATION_TASKS = frozenset({"velocity", "loco-manipulation"})


@dataclass
class TrajectoryBatch:
  """Batched environment trajectories stored as PyTorch tensors."""

  obs: torch.Tensor
  actions: torch.Tensor
  dones: torch.Tensor
  states: torch.Tensor | None = None
  masks: torch.Tensor | None = None
  lengths: torch.Tensor | None = None
  infos: Any | None = None

  def __post_init__(self) -> None:
    n_envs = int(self.obs.shape[0])
    time_steps = int(self.obs.shape[1] - 1)
    if tuple(self.actions.shape[:2]) != (n_envs, time_steps):
      raise ValueError("actions must have shape (n_envs, T, ...)")
    if tuple(self.dones.shape[:2]) != (n_envs, time_steps):
      raise ValueError("dones must have shape (n_envs, T)")
    if self.states is not None and tuple(self.states.shape[:2]) != (
      n_envs,
      time_steps + 1,
    ):
      raise ValueError("states must have shape (n_envs, T+1, ...)")
    if self.masks is not None and tuple(self.masks.shape[:2]) != (
      n_envs,
      time_steps,
    ):
      raise ValueError("masks must have shape (n_envs, T)")
    if self.lengths is not None and int(self.lengths.shape[0]) != n_envs:
      raise ValueError("lengths must have shape (n_envs,)")

  @property
  def n_envs(self) -> int:
    return int(self.obs.shape[0])

  @property
  def T(self) -> int:
    return int(self.obs.shape[1] - 1)

  @classmethod
  def zeros(
    cls,
    n_envs: int,
    T: int,
    obs_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    state_shape: tuple[int, ...] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
  ) -> TrajectoryBatch:
    target_device = torch.device(device)
    states = None
    if state_shape is not None:
      states = torch.zeros(
        (n_envs, T + 1, *state_shape), dtype=dtype, device=target_device
      )
    return cls(
      obs=torch.zeros((n_envs, T + 1, *obs_shape), dtype=dtype, device=target_device),
      actions=torch.zeros(
        (n_envs, T, *action_shape), dtype=dtype, device=target_device
      ),
      dones=torch.zeros((n_envs, T), dtype=torch.bool, device=target_device),
      states=states,
      masks=torch.ones((n_envs, T), dtype=dtype, device=target_device),
      lengths=torch.zeros((n_envs,), dtype=torch.long, device=target_device),
    )

  def slice(self, start: int = 0, end: int | None = None) -> TrajectoryBatch:
    if end is None:
      end = self.T
    return TrajectoryBatch(
      obs=self.obs[:, start : end + 1].clone(),
      actions=self.actions[:, start:end].clone(),
      dones=self.dones[:, start:end].clone(),
      states=None if self.states is None else self.states[:, start : end + 1].clone(),
      masks=None if self.masks is None else self.masks[:, start:end].clone(),
      lengths=None
      if self.lengths is None
      else torch.clamp(self.lengths - start, min=0, max=end - start),
      infos=self.infos,
    )

  def to_torch(self, device: str | torch.device = "cpu") -> TrajectoryBatch:
    target_device = torch.device(device)

    def convert(value: TensorLike | None) -> torch.Tensor | None:
      if value is None:
        return None
      return torch.as_tensor(value).to(target_device)

    return TrajectoryBatch(
      obs=convert(self.obs),  # type: ignore[arg-type]
      actions=convert(self.actions),  # type: ignore[arg-type]
      dones=convert(self.dones),  # type: ignore[arg-type]
      states=convert(self.states),
      masks=convert(self.masks),
      lengths=convert(self.lengths),
      infos=self.infos,
    )

  def save(self, filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    def cpu(value: torch.Tensor | None) -> torch.Tensor | None:
      return None if value is None else value.detach().cpu()

    torch.save(
      {
        "obs": cpu(self.obs),
        "actions": cpu(self.actions),
        "dones": cpu(self.dones),
        "states": cpu(self.states),
        "masks": cpu(self.masks),
        "lengths": cpu(self.lengths),
        "infos": self.infos,
      },
      filepath,
    )

  @classmethod
  def load(cls, filepath: str, device: str | torch.device = "cpu") -> TrajectoryBatch:
    payload = torch.load(filepath, map_location=device, weights_only=False)
    return cls(
      obs=payload["obs"],
      actions=payload["actions"],
      dones=payload["dones"],
      states=payload.get("states"),
      masks=payload.get("masks"),
      lengths=payload.get("lengths"),
      infos=payload.get("infos"),
    )


def _align_loco_manip_quat_sign(
  obs: torch.Tensor, previous_ee_quat: torch.Tensor | None = None
) -> torch.Tensor:
  aligned = obs.clone()
  ee_quat = torch.nn.functional.normalize(aligned[:, 13:17], dim=-1, eps=1e-8)
  if previous_ee_quat is None:
    flip_ee = ee_quat[:, :1] < 0.0
  else:
    previous_ee_quat = torch.nn.functional.normalize(previous_ee_quat, dim=-1, eps=1e-8)
    flip_ee = torch.sum(ee_quat * previous_ee_quat, dim=-1, keepdim=True) < 0.0
  ee_quat = torch.where(flip_ee, -ee_quat, ee_quat)

  command_quat = torch.nn.functional.normalize(aligned[:, -7:-3], dim=-1, eps=1e-8)
  flip_command = torch.sum(ee_quat * command_quat, dim=-1, keepdim=True) < 0.0
  aligned[:, 13:17] = ee_quat
  aligned[:, -7:-3] = torch.where(flip_command, -command_quat, command_quat)
  return aligned


def explore(
  policy: Any, env: RslRlVecEnvWrapper, task: str = "velocity"
) -> TrajectoryBatch:
  """Run a policy for one maximum episode and collect actor observations."""
  if task not in SUPPORTED_EXPLORATION_TASKS:
    supported = ", ".join(sorted(SUPPORTED_EXPLORATION_TASKS))
    raise ValueError(f"Unsupported exploration task '{task}'. Choose from: {supported}")

  observations, _ = env.reset()
  actor_obs = observations["actor"].to(env.unwrapped.device)
  if task == "loco-manipulation":
    actor_obs = _align_loco_manip_quat_sign(actor_obs)

  obs_steps = [actor_obs]
  action_steps: list[torch.Tensor] = []
  done_steps: list[torch.Tensor] = []

  for step in range(env.max_episode_length):
    with torch.no_grad():
      action = policy(env.get_observations())
      next_observations, _, done, _ = env.step(action)
      next_actor_obs = next_observations["actor"].to(env.unwrapped.device)
      if task == "loco-manipulation":
        next_actor_obs = _align_loco_manip_quat_sign(
          next_actor_obs, previous_ee_quat=obs_steps[-1][:, 13:17]
        )

    action_steps.append(action)
    done_steps.append(done)
    obs_steps.append(next_actor_obs)
    print(f"Exploration step: {step + 1}/{env.max_episode_length}", end="\r")

  return TrajectoryBatch(
    obs=torch.stack(obs_steps, dim=1),
    actions=torch.stack(action_steps, dim=1),
    dones=torch.stack(done_steps, dim=1),
  )
