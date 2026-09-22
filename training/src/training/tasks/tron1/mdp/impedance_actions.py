from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class JointImpedanceAction(BaseAction):
  """Joint-position action implemented with explicit impedance control."""

  def __init__(self, cfg: JointImpedanceActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    if cfg.use_default_offset:
      self._offset = self._entity.data.default_joint_pos[:, self._target_ids].clone()
    self._kp = self._resolve_gain(cfg.kp_ff)
    self._kd = self._resolve_gain(cfg.kd_ff)

  def _resolve_gain(self, gain: float | dict[str, float]) -> torch.Tensor:
    gains = torch.ones(self.num_envs, self.action_dim, device=self.device)
    if isinstance(gain, (float, int)):
      gains *= float(gain)
      return gains
    if isinstance(gain, dict):
      index_list, _, value_list = resolve_matching_names_values(
        gain, self._target_names
      )
      gains[:, index_list] = torch.tensor(value_list, device=self.device)
      return gains
    raise ValueError(f"Unsupported gain type: {type(gain)}. Use float or dict.")

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    q_des = self._processed_actions - encoder_bias

    dof_ids = self._entity.data.indexing.joint_v_adr[self._target_ids]
    q = self._entity.data.joint_pos[:, self._target_ids]
    qd = self._entity.data.joint_vel[:, self._target_ids]
    pd_term = self._kp * (q_des - q) - self._kd * qd

    if self.cfg.inertial_compensation:
      mass_matrix = self._env.sim.data.qM[:, dof_ids][:, :, dof_ids]
      tau_cmd = torch.einsum("nij,nj->ni", mass_matrix, pd_term)
    else:
      tau_cmd = pd_term

    if self.cfg.gravity_compensation:
      tau_cmd = tau_cmd + self._env.sim.data.qfrc_bias[:, dof_ids]

    self._entity.set_joint_effort_target(tau_cmd, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class JointImpedanceActionCfg(BaseActionCfg):
  """Configuration for explicit joint impedance control."""

  kp_ff: float | dict[str, float] = 0.0
  kd_ff: float | dict[str, float] = 0.0
  inertial_compensation: bool = False
  gravity_compensation: bool = True
  use_default_offset: bool = True

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> JointImpedanceAction:
    return JointImpedanceAction(self, env)
