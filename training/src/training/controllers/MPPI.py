import time

# from abc import ABC
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from interpax import Interpolator1D
from jax import random

from training.dynamics.training.base import Model


@dataclass
class MPPIConfig:
  """Configuration for MPPI controller."""

  horizon: int = 20
  dt: float = 0.02
  num_samples: int = 500
  act_dim: int = 1
  obs_dim: int = 1  # obs_dim * his_length
  act_bounds: tuple[list[float], list[float]] = None  # Shape (2, act_dim)

  device: str = "gpu"
  seed: int = 42
  n_knots: int = 6
  spline_order: int = 3
  n_iters: int = 1  # number of MPPI iterations
  noise: float = 0.3
  lam: float = 1.0

  cost_decay: float = 0.9
  gains: tuple[float, float, float] = (1.0, 1.0, 1.0)


class MPPI:
  def __init__(self, model: Model, cfg: MPPIConfig):
    """
    MPPI controller with spline knots for action parameterization.

    Args:
        horizon (int): Planning horizon.
        dt (float): Time step.
        num_samples (int): Number of samples.
        model (object): Dynamics model.
        device (str): "cpu" or "gpu".
    """
    self.horizon = cfg.horizon
    self.dt = cfg.dt
    self.num_samples = cfg.num_samples
    self.model = model
    self.device = self._set_device(cfg.device)
    self.key = random.PRNGKey(cfg.seed)

    self.obs_dim = cfg.obs_dim
    self.act_dim = cfg.act_dim
    self.min_action_bounds = jnp.array(cfg.act_bounds[0])
    self.max_action_bounds = jnp.array(cfg.act_bounds[1])
    self.min_action_bounds_norm = self.model.output_normalize(self.min_action_bounds)
    self.max_action_bounds_norm = self.model.output_normalize(self.max_action_bounds)

    self.n_knots = cfg.n_knots
    self.spline_order = cfg.spline_order
    self.n_iters = cfg.n_iters
    self.noise = cfg.noise
    self.lam = cfg.lam

    self.cost_decay = cfg.cost_decay
    self.gains = cfg.gains

    assert self.spline_order in [0, 1, 3], "Spline order must be 0, 1, or 3."

    self.initial_solution = jnp.zeros(self.act_dim)
    self.previous_solution = jnp.tile(self.initial_solution, (self.horizon, 1))
    # JIT-compiled function
    print("  ✓ JIT-compiling MPPI controller functions...")
    a = time.time()
    self.act = jax.jit(self.compute_best_action)
    self.vectorized_rollout = jax.vmap(
      self.compute_rollout, in_axes=(None, 0), out_axes=(0, 0)
    )
    self.jit_vectorized_rollout = jax.jit(self.vectorized_rollout)
    b = time.time()
    print(f"    - JIT compilation done in {b - a:.5f} seconds.")

  def compute_best_action(self, obs, reference, key, previous_solution):
    """
    Compute the best action using the Cross-Entropy Method (CEM).

    Args:
        obs (jnp.array): Current obs of the system. (obs_dim,).
        reference (jnp.array): Target trajectory/reference. (horizon, obs_dim).
        key (jax.random.PRNGKey): Random key for sampling.
        previous_solution (jnp.array): Previous solution (horizon, action_dim).

    Returns:
        tuple: (action, updated_key, updated_solution)
    """

    def _mppi_step(carry, _):
      # Generate samples
      mu, key, i = carry
      key, subkey = random.split(key)
      # knots = jax.vmap(self.actions_to_knots, in_axes=(None, -1, None), out_axes=-1)(self.spline_order, mu, self.n_knots)  # (n_knots, action_dim )
      knots = self.actions_to_knots(
        self.spline_order, mu, self.n_knots
      )  # (n_knots, action_dim )
      # debug.print("knots={vals}", vals=knots)
      knots, key = self._sample_actions(knots, subkey)
      # debug.print("shape of knots ={shape}, knots={vals}", shape=knots.shape, vals=knots)
      # knots = jnp.clip(knots, self.min_action_bounds, self.max_action_bounds) # (num_samples, n_knots, action_dim)
      actions = jax.vmap(self.knots_to_actions, in_axes=(None, 0, None), out_axes=0)(
        self.spline_order, knots, self.horizon
      )
      actions = jnp.clip(
        actions, self.min_action_bounds_norm, self.max_action_bounds_norm
      )
      # debug.print("shape of actions ={shape}, actions={vals}", shape=actions.shape, vals=actions)
      # breakpoint()
      # Rollout trajectories and compute costs
      next_obs, final_obs = self.jit_vectorized_rollout(obs_norm, actions)
      total_cost = self.compute_cost(next_obs, final_obs, reference_norm, actions)

      # Update mu using MPPI formula
      min_cost = jnp.min(total_cost)
      max_cost = jnp.max(total_cost)
      denom = jnp.maximum(max_cost - min_cost, 1e-8)
      exp_weights = jnp.exp(-1 / self.lam * ((total_cost - min_cost) / denom))
      mu_knots = jnp.sum(exp_weights[:, None, None] * knots, axis=0) / jnp.sum(
        exp_weights
      )
      mu_knots = jnp.clip(
        mu_knots, self.min_action_bounds_norm, self.max_action_bounds_norm
      )
      mu = self.knots_to_actions(self.spline_order, mu_knots, self.horizon)
      # mu = jnp.clip(mu, self.min_action_bounds_norm, self.max_action_bounds_norm)

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

    # updated_solution = jax.vmap(self.model.output_denormalize, in_axes=0, out_axes=0)(
    #   mu_norm
    # )
    # updated_solution = self._update_solution(updated_solution)
    updated_key = final_scan_state[1]

    # return action_to_execute, updated_key, updated_solution
    return action_to_execute, updated_key, total_cost

  def compute_rollout(self, initial_obs, actions):
    """
    Propagate states over the horizon.

    Args:
        initial_obs (jnp.array): Shape (obs_dim,).
        actions (jnp.array): Shape (horizon, action_dim).

    Returns:
        obs_list (jnp.array): history of obs (horizon, obs_dim).
        final_obs (jnp.array): final observation after applying all actions (obs_dim,).
    """

    def step(state, action):
      next_obs = self.model.forward(jnp.concatenate([state, action]))
      obs_per_timestep = next_obs.shape[
        0
      ]  # e.g., if next_obs is (2,), then obs_per_timestep = 2
      # Remove first obs_per_timestep elements and append next_obs
      updated_obs = jnp.concatenate(
        [
          state[obs_per_timestep:],  # Remove first n elements
          next_obs,  # Append new observation
        ]
      )
      return updated_obs, updated_obs

    final_obs, obs_list = jax.lax.scan(step, initial_obs, actions)
    return obs_list, final_obs

  def compute_cost(self, obs, final_obs, reference, actions):
    """
    Compute cost for states.

    Args:
        states (jnp.array): Shape (sample_size, horizon, obs_dim).
        reference (jnp.array): Shape (obs_dim, ).
        actions (jnp.array): Shape (num_samples, n_horizon, action_dim).

    Returns:
        jnp.array: Cost (num_samples,).
    """
    costs = self.cost_fn(obs, reference, actions)
    costs = jnp.where(jnp.isnan(costs) | jnp.isinf(costs), 1_000_000, costs)
    terminal_cost = self.terminal_cost_fn(final_obs, reference)
    costs *= self.dt
    total_cost = costs + terminal_cost
    return total_cost

  def _sample_actions(self, mu, key):
    """Sample actions using normal distribution."""
    key, subkey = random.split(key)
    noise = random.normal(subkey, shape=(self.num_samples, *mu.shape))
    samples = mu[None, ...] + (self.noise * noise)
    return samples, key

  def _update_solution(self, best_action):
    """Update the solution by rolling and replacing the last action."""
    updated_solution = jnp.roll(best_action, shift=-1, axis=0)
    # return updated_solution
    return updated_solution.at[-1, :].set(self.initial_solution)

  def reset(self):
    """
    Reset the initial solution and previous solution.
    """
    self.previous_solution = jnp.tile(self.initial_solution, (self.horizon, 1))
    self.key = random.PRNGKey(self.seed)  # Reset the random key

  def update_model(self, new_model):
    """
    Update the internal dynamics model and recompile JIT functions that depend on it.
    """
    self.model = new_model

    # Recompile all JAX functions that depend on self.model
    self.act = jax.jit(self.compute_best_action)
    self.vectorized_rollout = jax.vmap(
      self.compute_rollout, in_axes=(None, 0), out_axes=0
    )
    self.jit_vectorized_rollout = jax.jit(self.vectorized_rollout)

  @staticmethod
  def actions_to_knots(spline_order, actions, n_knots):
    """
    Convert a full action sequence into spline knots.

    Args:
        spline_order: Spline interpolation order (0, 1, or 3).
        actions: Action sequence (T x n_action).
        n_knots: Number of spline knots to extract.

    Returns:
        Reduced set of action knots.
    """
    t = jnp.linspace(0, 1, actions.shape[0])
    t_sample = jnp.linspace(0, 1, n_knots)
    if spline_order == 0:
      indices = jnp.searchsorted(t_sample, t, side="right") - 1
      return actions[indices]
    elif spline_order == 1:
      spline = Interpolator1D(t, actions, method="linear")
      return spline(t_sample)
    elif spline_order == 3:
      spline = Interpolator1D(t, actions, method="cubic")
      return spline(t_sample)
    else:
      raise ValueError(f"Unknown spline order: {spline_order}")

  @staticmethod
  def knots_to_actions(spline_order, knots, horizon):
    """
    Convert spline knots back to a full action sequence.

    Args:
        spline_order: Spline interpolation order (0, 1, or 3).
        knots: Knot points (n_knots x n_action).
        horizon: Desired output sequence length.

    Returns:
        Full interpolated action sequence (horizon x n_action).
    """

    t = jnp.linspace(0, 1, knots.shape[0])
    t_sample = jnp.linspace(0, 1, horizon)
    if spline_order == 0:
      indices = jnp.searchsorted(t_sample, t, side="right") - 1
      return knots[indices]
    elif spline_order == 1:
      spline = Interpolator1D(t, knots, method="linear")
      return spline(t_sample)
    elif spline_order == 3:
      spline = Interpolator1D(t, knots, method="cubic")
      return spline(t_sample)
    else:
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
  def cost_fn(self, obs, reference, actions):
    """
    reference tracking cost function

    Args:
        obs (jnp.array): Shape (sample_size, horizon, obs_dim).
        reference (jnp.array): Shape (1, horizon, act_dim).
        actions (jnp.array): Shape (num_samples, horizon, action_dim).

    Returns:
        jnp.array: Cost (num_samples,).
    """
    # ref = jnp.broadcast_to(reference, obs.shape)  # (sample_size, horizon, obs_dim)
    # tracking_error = obs - ref
    # tracking_error = 0
    # for i in range(obs.shape[1]):
    #     start = max(0, i - obs.shape[2])
    #     tracking_error += (obs[:,start: i, -actions.shape[2]:] - reference[:, start:i, :])**2

    # tracking_error = obs[:,:-1, -actions.shape[2]:] - reference[:, :-1, :]
    action_rate = jnp.diff(actions, axis=1)  # (sample_size, horizon, act_dim)
    # ref_action_rate = jnp.diff(reference, axis=1)  # (1, horizon, act_dim)
    # ref_velocity = reference[0, :, :]
    # ref_accel = jnp.diff(ref_velocity, axis=0, prepend=ref_velocity[:1])
    # curvature = jnp.sqrt(jnp.sum(ref_accel ** 2, axis=1))  # (horizon,)
    # curvature_norm = curvature / (jnp.max(curvature) + 1e-6)
    # curvature_boost = 1.0 + 10.0 * curvature_norm
    # return 10*jnp.sum(tracking_error ** 2, axis=(1, 2)) + 0.0 * jnp.sum(action_rate ** 2, axis=(1, 2))
    # 10, 5

    # try decaying weights
    # decay = 0.0 # 0.5, 0.01
    # weights = decay ** jnp.arange(tracking_error.shape[1]) # (horizon,)
    # weighted_error = tracking_error ** 2 * weights[None, :, None]

    # try a more regularized cost
    num_samples, horizon, obs_dim = obs.shape
    # history_length = obs_dim // self.act_dim
    # ref_expanded = jnp.zeros((horizon, obs_dim))
    # for t in range(horizon):
    #   ref_start = max(0, t - history_length + 1)
    #   ref_slice = reference[0, ref_start : t + 1, :]
    #   if ref_slice.shape[0] < history_length:
    #     padding = jnp.zeros((history_length - ref_slice.shape[0], self.act_dim))
    #     ref_slice = jnp.concatenate([padding, ref_slice], axis=0)
    #   ref_expanded = ref_expanded.at[t].set(ref_slice.flatten())
    # tracking_error = (
    #   obs - ref_expanded[None, :, :]
    # ) ** 2  # (num_samples, horizon, obs_dim)
    newest_obs = obs[:, :, -self.act_dim :]
    ref_current = reference[0, :, :][None, :, :]
    tracking_error = (newest_obs[..., :2] - ref_current[..., :2]) ** 2
    # tracking_error = (newest_obs - ref_current) ** 2
    time_weights = self.cost_decay ** jnp.arange(horizon)
    weighted_error = tracking_error * time_weights[None, :, None]

    action_to_ref_decay = self.cost_decay ** jnp.arange(horizon)  # 0.5
    action_to_ref_error = (actions - reference[0, :, :]) ** 2 * action_to_ref_decay[
      None, :, None
    ]
    return (
      self.gains[0] * jnp.sum(weighted_error, axis=(1, 2))
      + self.gains[1] * jnp.sum((action_rate) ** 2, axis=(1, 2))  #  - ref_action_rate
      + self.gains[2] * jnp.sum(action_to_ref_error, axis=(1, 2))  # 3.0
    )  # 10.0 5.0

  def terminal_cost_fn(self, final_states, reference):
    """
    Terminal cost function

    Args:
        final_states (jnp.array): Shape (sample_size, obs_dim).
        reference (jnp.array): Shape (1, horizon, act_dim).

    Returns:
        jnp.array: Cost (num_samples,).
    """
    error = (
      final_states[:, -reference.shape[2] :] - reference[0, -1, :]
    )  # only consider the last timestep reference
    return 0.0 * jnp.sum(error**2, axis=1)  # only consider the last timestep reference
    # 0.0
