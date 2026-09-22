# Create: src/training/tasks/tron1/mdp/prbs_noise.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul
from mjlab.utils.noise.noise_cfg import NoiseCfg
from typing_extensions import override

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@dataclass
class PRBSNoiseCfg(NoiseCfg):
  """Pseudo-Random Binary Sequence noise.

  Note: This is a simplified stateless version. Each call randomly
  selects +amplitude or -amplitude, approximating PRBS behavior.
  For true PRBS with temporal correlation, use a custom observation term.
  """

  amplitude: float = 0.2
  switch_probability: float = 0.1  # Probability of switching per step

  def __post_init__(self):
    # Store current state per environment (will be managed externally)
    self._state = None

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    """Apply PRBS-like noise to data."""
    if self._state is None or self._state.shape[0] != data.shape[0]:
      # Initialize state
      self._state = torch.where(
        torch.rand(data.shape[0], 1, device=data.device) > 0.5,
        torch.ones(data.shape[0], 1, device=data.device),
        -torch.ones(data.shape[0], 1, device=data.device),
      )

    # Randomly switch state
    switch_mask = (
      torch.rand(data.shape[0], 1, device=data.device) < self.switch_probability
    )
    self._state = torch.where(switch_mask, -self._state, self._state)

    # Broadcast to match data dimensions
    prbs_signal = self.amplitude * self._state.expand_as(data)

    if self.operation == "add":
      return data + prbs_signal
    elif self.operation == "scale":
      return data * (1.0 + prbs_signal)
    elif self.operation == "abs":
      return prbs_signal
    else:
      raise ValueError(f"Unsupported operation: {self.operation}")


@dataclass
class PRBSPoseNoiseCfg(NoiseCfg):
  """PRBS noise for 7D poses with a normalized quaternion tail.

  The pose is expected to be ``[x, y, z, qw, qx, qy, qz]``. Position is
  perturbed additively, while orientation is perturbed in the tangent space of
  SO(3) and multiplied back onto the quaternion.
  """

  position_amplitude: float = 0.05
  orientation_amplitude: float = 0.05  # radians
  switch_probability: float = 0.1

  def __post_init__(self):
    self._position_state = None
    self._orientation_state = None

  def _update_state(
    self, state: torch.Tensor | None, shape: torch.Size, data: torch.Tensor
  ) -> torch.Tensor:
    if (
      state is None
      or state.shape != shape
      or state.device != data.device
      or state.dtype != data.dtype
    ):
      state = torch.where(
        torch.rand(shape, device=data.device) > 0.5,
        torch.ones(shape, device=data.device, dtype=data.dtype),
        -torch.ones(shape, device=data.device, dtype=data.dtype),
      )

    switch_mask = torch.rand(shape, device=data.device) < self.switch_probability
    return torch.where(switch_mask, -state, state)

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    """Apply PRBS noise while keeping the quaternion valid."""
    if data.shape[-1] != 7:
      raise ValueError(
        f"PRBSPoseNoiseCfg expects a 7D pose [..., 7], got shape {data.shape}."
      )
    if self.operation != "add":
      raise ValueError("PRBSPoseNoiseCfg only supports operation='add'.")

    position_shape = data.shape[:-1] + (3,)
    self._position_state = self._update_state(
      self._position_state, position_shape, data
    )
    self._orientation_state = self._update_state(
      self._orientation_state, position_shape, data
    )

    position = data[..., :3] + self.position_amplitude * self._position_state

    quat = torch.nn.functional.normalize(data[..., 3:7], p=2, dim=-1, eps=1e-8)
    rotvec = self.orientation_amplitude * self._orientation_state
    quat_noise = _quat_exp(rotvec.reshape(-1, 3)).reshape(quat.shape)
    noisy_quat = quat_mul(
      quat_noise.reshape(-1, 4),
      quat.reshape(-1, 4),
    ).reshape(quat.shape)
    noisy_quat = torch.nn.functional.normalize(noisy_quat, p=2, dim=-1, eps=1e-8)

    return torch.cat((position, noisy_quat), dim=-1)


@dataclass
class GaussianPoseNoiseCfg(NoiseCfg):
  """Gaussian noise for 7D poses with a normalized quaternion tail.

  The pose is expected to be ``[x, y, z, qw, qx, qy, qz]``. Position is
  perturbed with additive Gaussian noise. Orientation is perturbed by sampling
  a Gaussian rotation vector in radians, mapping it to a quaternion, and
  multiplying it onto the input quaternion.
  """

  position_mean: float = 0.0
  position_std: float = 0.05
  orientation_mean: float = 0.0
  orientation_std: float = 0.05  # radians

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian pose noise while keeping the quaternion valid."""
    if data.shape[-1] != 7:
      raise ValueError(
        f"GaussianPoseNoiseCfg expects a 7D pose [..., 7], got shape {data.shape}."
      )
    if self.operation != "add":
      raise ValueError("GaussianPoseNoiseCfg only supports operation='add'.")

    position_noise = self.position_mean + self.position_std * torch.randn_like(
      data[..., :3]
    )
    position = data[..., :3] + position_noise

    quat = torch.nn.functional.normalize(data[..., 3:7], p=2, dim=-1, eps=1e-8)
    rotvec = self.orientation_mean + self.orientation_std * torch.randn_like(
      data[..., :3]
    )
    quat_noise = _quat_exp(rotvec.reshape(-1, 3)).reshape(quat.shape)
    noisy_quat = quat_mul(
      quat_noise.reshape(-1, 4),
      quat.reshape(-1, 4),
    ).reshape(quat.shape)
    noisy_quat = torch.nn.functional.normalize(noisy_quat, p=2, dim=-1, eps=1e-8)

    return torch.cat((position, noisy_quat), dim=-1)


def ee_position(
  env: ManagerBasedRlEnv,
  base_name: str = "",
  ee_site_name: str = "",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  assert base_name, "base_name must be provided"
  assert ee_site_name, "ee_site_name must be provided"

  base_ids, _ = robot.find_bodies(base_name)
  ee_site_ids, _ = robot.find_sites(ee_site_name)
  base_id = base_ids[0]
  ee_site_id = ee_site_ids[0]

  base_pos_w = robot.data.body_link_pos_w[:, base_id]
  base_quat_w = robot.data.body_link_quat_w[:, base_id]
  ee_pos_w = robot.data.site_pos_w[:, ee_site_id]

  ee_pos_rel_w = ee_pos_w - base_pos_w
  ee_pos_b = quat_apply(quat_inv(base_quat_w), ee_pos_rel_w)
  return ee_pos_b


def ee_orientation(
  env: ManagerBasedRlEnv,
  base_name: str = "",
  ee_site_name: str = "",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  assert base_name, "base_name must be provided"
  assert ee_site_name, "ee_site_name must be provided"

  base_ids, _ = robot.find_bodies(base_name)
  ee_site_ids, _ = robot.find_sites(ee_site_name)
  base_id = base_ids[0]
  ee_site_id = ee_site_ids[0]

  base_quat_w = robot.data.body_link_quat_w[:, base_id]
  ee_quat_w = robot.data.site_quat_w[:, ee_site_id]
  ee_quat_b = quat_mul(quat_inv(base_quat_w), ee_quat_w)
  return torch.nn.functional.normalize(ee_quat_b, p=2, dim=-1, eps=1e-8)


def _quat_exp(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  # rotvec: [B, 3], axis-angle vector (Lie algebra so(3))
  theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)  # [B,1]
  half = 0.5 * theta
  w = torch.cos(half)
  scale = torch.sin(half) / torch.clamp(theta, min=eps)
  xyz = scale * rotvec
  q = torch.cat([w, xyz], dim=-1)  # [w, x, y, z]
  return torch.nn.functional.normalize(q, p=2, dim=-1)


def normalized_builtin_quat_lie_noise(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  noise_std: float = 0.03,  # radians
) -> torch.Tensor:
  """Quaternion observation with Lie algebra noise on SO(3)."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  q = torch.nn.functional.normalize(sensor.data, p=2, dim=-1)

  xi = noise_std * torch.randn(q.shape[0], 3, device=q.device, dtype=q.dtype)
  q_noise = _quat_exp(xi)
  q_noisy = quat_mul(q_noise, q)  # left-multiply: world-frame perturbation
  return torch.nn.functional.normalize(q_noisy, p=2, dim=-1)
