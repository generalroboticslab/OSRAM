import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from interpax import Interpolator1D
from jax import random

from training.dynamics.training.base import Model


@dataclass
class MPPIActionHistoryConfig:
  """Configuration for MPPI controller with explicit action-history state."""

  horizon: int = 20
  dt: float = 0.02
  num_samples: int = 500
  act_dim: int = 1

  # Flattened observation-history dimensions in the rollout state.
  obs_dim: int = 1

  # Number of historical actions stored in the rollout state.
  # Model input convention:
  #   model_input_dim = obs_dim + act_dim * (action_history_len + 1)
  # where the +1 is the current action concatenated at each rollout step.
  action_history_len: int = 0

  act_bounds: tuple[list[float], list[float]] = None  # Shape (2, act_dim)

  device: str = "gpu"
  seed: int = 42
  n_knots: int = 6
  spline_order: int = 3
  n_iters: int = 1  # number of MPPI iterations
  noise: float | list[float] | tuple[float, ...] = 0.3
  noise_annealing_coefficient: float = 1.0
  lam: float = 1.0

  cost_decay: float = 0.9
  gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
  terminal_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)


class MPPIActionHistory:
  """MPPI controller that rolls dynamics on [obs_history, action_history] state."""

  def __init__(self, model: Model, cfg: MPPIActionHistoryConfig):
    self.horizon = cfg.horizon
    self.dt = cfg.dt
    self.num_samples = cfg.num_samples
    self.model = model
    self.device = self._set_device(cfg.device)
    self.seed = cfg.seed
    self.key = random.PRNGKey(cfg.seed)

    self.act_dim = cfg.act_dim
    self.obs_dim = cfg.obs_dim

    self.action_history_len = cfg.action_history_len
    self.action_history_dim = self.act_dim * self.action_history_len
    self.state_dim = self.obs_dim + self.action_history_dim

    if self.obs_dim <= 0:
      raise ValueError(f"obs_dim must be positive, got {self.obs_dim}.")
    if self.action_history_len < 0:
      raise ValueError(
        f"action_history_len must be >= 0, got {self.action_history_len}."
      )
    expected_input_dim = self.obs_dim + self.act_dim * (self.action_history_len + 1)
    if self.model.input_dim != expected_input_dim:
      raise ValueError(
        "Model input mismatch: expected "
        f"obs_dim + act_dim * (action_history_len + 1) = {expected_input_dim}, "
        f"but model.input_dim is {self.model.input_dim}."
      )

    # Per-step observation prediction size from dynamics model output.
    self.obs_step_dim = self.model.output_dim
    if self.obs_dim % self.obs_step_dim != 0:
      raise ValueError(
        f"obs_dim ({self.obs_dim}) must be divisible by model.output_dim ({self.obs_step_dim})."
      )
    if self.action_history_dim > 0 and self.action_history_dim % self.act_dim != 0:
      raise ValueError(
        f"action_history_dim ({self.action_history_dim}) must be divisible by act_dim ({self.act_dim})."
      )

    if cfg.act_bounds is None:
      raise ValueError("act_bounds must be provided.")
    self.min_action_bounds = jnp.array(cfg.act_bounds[0])
    self.max_action_bounds = jnp.array(cfg.act_bounds[1])
    self.min_action_bounds_norm = self.model.output_normalize(self.min_action_bounds)
    self.max_action_bounds_norm = self.model.output_normalize(self.max_action_bounds)

    self.n_knots = cfg.n_knots
    self.spline_order = cfg.spline_order
    self.n_iters = cfg.n_iters
    self.noise = jnp.asarray(cfg.noise)
    if self.noise.ndim > 1 or (
      self.noise.ndim == 1 and self.noise.shape[0] != self.act_dim
    ):
      raise ValueError(
        f"noise must be a scalar or have shape ({self.act_dim},), got "
        f"{self.noise.shape}."
      )
    self.noise_annealing_coefficient = cfg.noise_annealing_coefficient
    self.lam = cfg.lam

    self.cost_decay = cfg.cost_decay
    self.gains = cfg.gains
    self.terminal_gains = cfg.terminal_gains

    assert self.spline_order in [0, 1, 3], "Spline order must be 0, 1, or 3."

    self.initial_solution = jnp.zeros(self.act_dim)
    self.previous_solution = jnp.tile(self.initial_solution, (self.horizon, 1))

    print("  ✓ JIT-compiling MPPIActionHistory controller functions...")
    a = time.time()
    self.act = jax.jit(self.compute_best_action)
    self.vectorized_rollout = jax.vmap(
      self.compute_rollout, in_axes=(None, 0), out_axes=(0, 0)
    )
    self.jit_vectorized_rollout = jax.jit(self.vectorized_rollout)
    b = time.time()
    print(f"    - JIT compilation done in {b - a:.5f} seconds.")

  def compute_best_action(self, obs, reference, key, previous_solution):
    """Compute best action with MPPI on an augmented rollout state."""

    def _mppi_step(carry, _):
      mu, key, i = carry
      key, subkey = random.split(key)

      knots = self.actions_to_knots(self.spline_order, mu, self.n_knots)
      knots, key = self._sample_actions(knots, subkey)

      actions = jax.vmap(self.knots_to_actions, in_axes=(None, 0, None), out_axes=0)(
        self.spline_order, knots, self.horizon
      )
      actions = jnp.clip(
        actions, self.min_action_bounds_norm, self.max_action_bounds_norm
      )

      next_states, final_states = self.jit_vectorized_rollout(obs_norm, actions)
      total_cost = self.compute_cost(next_states, final_states, reference_norm, actions)

      min_cost = jnp.min(total_cost)
      max_cost = jnp.max(total_cost)
      denom = jnp.maximum(max_cost - min_cost, 1e-8)
      exp_weights = jnp.exp(-1.0 / self.lam * ((total_cost - min_cost) / denom))

      mu_knots = jnp.sum(exp_weights[:, None, None] * knots, axis=0) / jnp.sum(
        exp_weights
      )
      mu_knots = jnp.clip(
        mu_knots, self.min_action_bounds_norm, self.max_action_bounds_norm
      )
      mu = self.knots_to_actions(self.spline_order, mu_knots, self.horizon)

      return [mu, key, i + 1], total_cost

    obs_norm = self.model.input_normalize(obs)
    reference_norm = jax.vmap(self.model.output_normalize, in_axes=0)(reference)
    init_mu = jax.vmap(self.model.output_normalize, in_axes=0, out_axes=0)(
      previous_solution
    )

    init_scan_state = [init_mu, key, 0]
    final_scan_state, total_cost = jax.lax.scan(
      _mppi_step, init_scan_state, None, length=self.n_iters
    )

    mu_norm = final_scan_state[0]
    action_to_execute = self.model.output_denormalize(mu_norm[0])
    action_to_execute = jnp.clip(
      action_to_execute, self.min_action_bounds, self.max_action_bounds
    )

    updated_key = final_scan_state[1]
    return action_to_execute, updated_key, total_cost

  def compute_rollout(self, initial_state, actions):
    """Roll forward state=[obs_history, action_history] over the horizon."""

    def step(state, action):
      next_obs = self.model.forward(jnp.concatenate([state, action]))

      obs_hist = state[: self.obs_dim]
      new_obs_hist = jnp.concatenate([obs_hist[self.obs_step_dim :], next_obs])

      if self.action_history_dim > 0:
        act_hist = state[self.obs_dim :]
        new_act_hist = jnp.concatenate([act_hist[self.act_dim :], action])
        updated_state = jnp.concatenate([new_obs_hist, new_act_hist])
      else:
        updated_state = new_obs_hist

      return updated_state, updated_state

    final_state, state_list = jax.lax.scan(step, initial_state, actions)
    return state_list, final_state

  def compute_cost(self, states, final_states, reference, actions):
    """Compute MPPI trajectory costs using observation-history channels only."""
    obs_states = states[:, :, : self.obs_dim]
    final_obs_states = final_states[:, : self.obs_dim]

    costs = self.cost_fn(obs_states, reference, actions)
    costs = jnp.where(jnp.isnan(costs) | jnp.isinf(costs), 1_000_000, costs)
    terminal_cost = self.terminal_cost_fn(final_obs_states, reference)
    costs *= self.dt
    return costs + terminal_cost

  def _sample_actions(self, mu, key):
    """Sample actions with exponentially annealed noise over trajectory knots."""
    key, subkey = random.split(key)
    noise = random.normal(subkey, shape=(self.num_samples, *mu.shape))
    annealing_kernel = self.noise_annealing_coefficient ** jnp.arange(mu.shape[0])
    annealing_kernel = annealing_kernel[:, None]
    samples = mu[None, ...] + (self.noise / self.n_iters * annealing_kernel * noise)
    return samples, key

  def _update_solution(self, best_action):
    """Update solution by receding-horizon shift."""
    updated_solution = jnp.roll(best_action, shift=-1, axis=0)
    return updated_solution.at[-1, :].set(self.initial_solution)

  def reset(self):
    """Reset previous solution and random key."""
    self.previous_solution = jnp.tile(self.initial_solution, (self.horizon, 1))
    self.key = random.PRNGKey(self.seed)

  def update_model(self, new_model):
    """Update model and recompile JIT functions."""
    self.model = new_model
    self.act = jax.jit(self.compute_best_action)
    self.vectorized_rollout = jax.vmap(
      self.compute_rollout, in_axes=(None, 0), out_axes=(0, 0)
    )
    self.jit_vectorized_rollout = jax.jit(self.vectorized_rollout)

  @staticmethod
  def actions_to_knots(spline_order, actions, n_knots):
    """Convert full action sequence into spline knots."""
    t = jnp.linspace(0, 1, actions.shape[0])
    t_sample = jnp.linspace(0, 1, n_knots)

    if spline_order == 0:
      indices = jnp.searchsorted(t_sample, t, side="right") - 1
      return actions[indices]
    if spline_order == 1:
      spline = Interpolator1D(t, actions, method="linear")
      return spline(t_sample)
    if spline_order == 3:
      spline = Interpolator1D(t, actions, method="cubic")
      return spline(t_sample)
    raise ValueError(f"Unknown spline order: {spline_order}")

  @staticmethod
  def knots_to_actions(spline_order, knots, horizon):
    """Convert spline knots back to a full action sequence."""
    t = jnp.linspace(0, 1, knots.shape[0])
    t_sample = jnp.linspace(0, 1, horizon)

    if spline_order == 0:
      indices = jnp.searchsorted(t_sample, t, side="right") - 1
      return knots[indices]
    if spline_order == 1:
      spline = Interpolator1D(t, knots, method="linear")
      return spline(t_sample)
    if spline_order == 3:
      spline = Interpolator1D(t, knots, method="cubic")
      return spline(t_sample)
    raise ValueError(f"Unknown spline order: {spline_order}")

  def _set_device(self, device):
    """Set computation device."""
    if device == "gpu":
      try:
        return jax.devices("gpu")[0]
      except Exception:
        print("GPU not available, using CPU")
    return jax.devices("cpu")[0]

  #########################################
  ### cost functions
  #########################################
  @staticmethod
  def _normalize_quat(quat):
    """Normalize quaternions with numerical safety."""
    return quat / jnp.clip(jnp.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)

  @staticmethod
  def _quat_conj(quat):
    """Quaternion conjugate for scalar-first convention [w, x, y, z]."""
    return jnp.concatenate([quat[..., :1], -quat[..., 1:]], axis=-1)

  @staticmethod
  def _quat_mul(q1, q2):
    """Quaternion multiply for scalar-first convention [w, x, y, z]."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jnp.stack(
      [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      ],
      axis=-1,
    )

  def _quat_box_minus(self, q_target, q_current):
    """SO(3) box-minus as a rotation vector from q_current to q_target."""
    q_target = self._normalize_quat(q_target)
    q_current = self._normalize_quat(q_current)
    q_err = self._quat_mul(q_target, self._quat_conj(q_current))
    # Enforce shortest-path representation.
    q_err = jnp.where(q_err[..., :1] < 0.0, -q_err, q_err)
    q_err = self._normalize_quat(q_err)

    w = jnp.clip(q_err[..., 0], -1.0, 1.0)
    v = q_err[..., 1:]
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * jnp.arctan2(v_norm[..., 0], w)
    axis = v / jnp.clip(v_norm, 1e-8, None)
    return axis * angle[..., None]

  def cost_fn(self, obs, reference, actions):
    """Reference tracking cost using obs-history channels and action regularization."""
    action_rate = jnp.diff(actions[..., :3], axis=1)
    # ref_action_rate = jnp.diff(reference, axis=1)

    num_samples, horizon, _ = obs.shape
    # ref_expanded = jnp.zeros((horizon, obs_dim))
    # for t in range(horizon):
    #   ref_start = max(0, t - history_length + 1)
    #   ref_slice = reference[0, ref_start : t + 1, :]
    #   if ref_slice.shape[0] < history_length:
    #     padding = jnp.zeros((history_length - ref_slice.shape[0], self.act_dim))
    #     ref_slice = jnp.concatenate([padding, ref_slice], axis=0)
    #   ref_expanded = ref_expanded.at[t].set(ref_slice.flatten())

    # time_weights = self.cost_decay ** jnp.arange(horizon)
    # if self.act_dim == 7:
    #   obs_pose = obs.reshape(num_samples, horizon, history_length, self.act_dim)
    #   ref_pose = ref_expanded.reshape(horizon, history_length, self.act_dim)
    #   ref_pose = ref_pose[None, :, :, :]

    #   pos_error_sq = jnp.sum((obs_pose[..., :3] - ref_pose[..., :3]) ** 2, axis=-1)
    #   quat_error = self._quat_box_minus(ref_pose[..., 3:], obs_pose[..., 3:])
    #   ori_error_sq = jnp.sum(quat_error**2, axis=-1)
    #   tracking_cost = jnp.sum(
    #     (pos_error_sq + ori_error_sq) * time_weights[None, :, None], axis=(1, 2)
    #   )
    # else:
    #   tracking_error = (obs - ref_expanded[None, :, :]) ** 2
    #   weighted_error = tracking_error * time_weights[None, :, None]
    #   tracking_cost = jnp.sum(weighted_error, axis=(1, 2))

    newest_obs = obs[:, :, -self.act_dim :]
    ref_current = reference[0, :, :][None, :, :]
    time_weights = self.cost_decay ** jnp.arange(horizon)
    if self.act_dim == 7:
      # pos_error_sq = jnp.sum((newest_obs[..., :3] - ref_current[..., :3]) ** 2, axis=-1)
      # quat_error = self._quat_box_minus(ref_current[..., 3:], newest_obs[..., 3:])
      # ori_error_sq = jnp.sum(quat_error**2, axis=-1)
      pos_error_sq = jnp.sum((newest_obs[..., :3] - ref_current[..., :3]) ** 2, axis=-1)
      tracking_cost = jnp.sum((pos_error_sq) * time_weights[None, :], axis=1)
      # tracking_cost = jnp.sum(
      #   (pos_error_sq + 0.1 * ori_error_sq) * time_weights[None, :], axis=1
      # )
    else:
      tracking_error_sq = jnp.sum((newest_obs - ref_current) ** 2, axis=-1)
      tracking_cost = jnp.sum(tracking_error_sq * time_weights[None, :], axis=1)

    action_to_ref_decay = self.cost_decay ** jnp.arange(horizon)
    action_to_ref_error = (actions - reference[0, :, :]) ** 2 * action_to_ref_decay[
      None, :, None
    ]

    return (
      self.gains[0] * tracking_cost
      + self.gains[1] * jnp.sum((action_rate) ** 2, axis=(1, 2))  #  - ref_action_rate
      + self.gains[2] * jnp.sum(action_to_ref_error, axis=(1, 2))
    )

  def terminal_cost_fn(self, final_states, reference):
    """Terminal cost on the most recent observation channels."""
    # final_states are observation-history only here.
    final_obs = final_states[:, -reference.shape[2] :]
    final_ref = reference[0, -1, :]
    if self.act_dim == 7:
      pos_error_sq = jnp.sum((final_obs[:, :3] - final_ref[None, :3]) ** 2, axis=-1)
      quat_error = self._quat_box_minus(final_ref[None, 3:], final_obs[:, 3:])
      ori_error_sq = jnp.sum(quat_error**2, axis=-1)
      error_cost = pos_error_sq + ori_error_sq
    else:
      error = final_obs - final_ref[None, :]
      error_cost = jnp.sum(error**2, axis=1)
    return self.terminal_gains[0] * error_cost
