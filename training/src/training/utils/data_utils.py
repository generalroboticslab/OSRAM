import os
from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@partial(jax.jit, static_argnums=(2, 3))
def _construct_dataset_arrays(
  u: jnp.ndarray, x: jnp.ndarray, input_horizon: int, output_horizon: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """JIT-compiled version that returns arrays instead of list of tuples.

  Returns:
      all_inputs: (total_samples, input_feature_dim)
      all_targets: (total_samples, output_feature_dim)
  """
  num_envs, T, in_dim = u.shape
  _, _, out_dim = x.shape

  def process_single_sample(u_traj, x_traj, i):
    """Process a single time index for one trajectory using dynamic slicing."""
    # History obs
    x_hist = lax.dynamic_slice(
      x_traj, (i - input_horizon, 0), (input_horizon, out_dim)
    ).ravel()

    # Future actions
    u_fut = lax.dynamic_slice(u_traj, (i - 1, 0), (output_horizon, in_dim)).ravel()

    # Concatenate into network input
    network_in = jnp.concatenate([x_hist, u_fut], axis=-1)

    # Target: future observations
    target = lax.dynamic_slice(x_traj, (i, 0), (output_horizon, out_dim)).ravel()

    return network_in, target

  def process_trajectory(u_traj, x_traj):
    """Process all valid samples in a single trajectory using vmap."""
    indices = jnp.arange(input_horizon, T - output_horizon + 1)

    inputs, targets = jax.vmap(lambda i: process_single_sample(u_traj, x_traj, i))(
      indices
    )

    return inputs, targets

  # Vmap over all environments
  all_inputs, all_targets = jax.vmap(process_trajectory)(u, x)

  # Reshape to (num_envs * num_samples, feature_dim)
  all_inputs = all_inputs.reshape(-1, all_inputs.shape[-1])
  all_targets = all_targets.reshape(-1, all_targets.shape[-1])

  return all_inputs, all_targets


@partial(jax.jit, static_argnums=(2, 3, 4))
def _construct_dataset_arrays_with_action_context(
  u: jnp.ndarray,
  x: jnp.ndarray,
  input_horizon: int,
  output_horizon: int,
  action_history_horizon: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Construct arrays with fixed past+future action context.

  For each sample index ``n`` this uses:
    - state history: ``x[n-input_horizon : n]``
    - action context: ``u[n-1-action_history_horizon : n-1+output_horizon]``
    - target: ``x[n : n+output_horizon]``

  Note:
    This is the fixed-window version of using richer action context.
    It avoids variable-length input vectors and is compatible with MLP training.
  """
  num_envs, T, in_dim = u.shape
  _, _, out_dim = x.shape
  action_context_len = action_history_horizon + output_horizon
  first_valid_n = max(input_horizon, action_history_horizon + 1)

  def process_single_sample(u_traj, x_traj, n):
    x_hist = lax.dynamic_slice(
      x_traj, (n - input_horizon, 0), (input_horizon, out_dim)
    ).ravel()

    u_ctx = lax.dynamic_slice(
      u_traj,
      (n - 1 - action_history_horizon, 0),
      (action_context_len, in_dim),
    ).ravel()

    network_in = jnp.concatenate([x_hist, u_ctx], axis=-1)
    target = lax.dynamic_slice(x_traj, (n, 0), (output_horizon, out_dim)).ravel()
    return network_in, target

  def process_trajectory(u_traj, x_traj):
    indices = jnp.arange(first_valid_n, T - output_horizon + 1)
    inputs, targets = jax.vmap(lambda n: process_single_sample(u_traj, x_traj, n))(
      indices
    )
    return inputs, targets

  all_inputs, all_targets = jax.vmap(process_trajectory)(u, x)
  all_inputs = all_inputs.reshape(-1, all_inputs.shape[-1])
  all_targets = all_targets.reshape(-1, all_targets.shape[-1])
  return all_inputs, all_targets


def construct_dataset(
  u: jnp.ndarray, x: jnp.ndarray, input_horizon: int, output_horizon: int
):
  """
  Construct dataset from two given input output trajectories. The dataset is used to train the dynamics model:
      x_{[n:n+k]} = f(x_{[0,n-1]}, u_{[n;n+k]})
  @param:
      u: target
      x: real
      obs_horzion (n)
      output_horizon (k)
  """
  # validate dataset size
  if u.ndim != 3 or x.ndim != 3:
    raise ValueError(
      f"This function only takes in dataset of size (num_envs, T, dim), but got dimension for input: {u.ndim} and out: {x.ndim}."
    )

  if u.shape[0] != x.shape[0] or u.shape[1] != x.shape[1]:
    raise ValueError(
      f"u and x must have same num_envs and T, got u.shape={u.shape}, x.shape={x.shape}"
    )

  num_samples = u.shape[1] - input_horizon - output_horizon + 1
  if num_samples <= 0:
    raise ValueError(
      f"Not enough timesteps: T={u.shape[1]}, input_horizon={input_horizon}, output_horizon={output_horizon}"
    )

  # Call JIT-compiled version
  all_inputs, all_targets = _construct_dataset_arrays(
    u, x, input_horizon, output_horizon
  )

  # Convert to list of tuples for compatibility
  dataset = [(all_inputs[i], all_targets[i]) for i in range(all_inputs.shape[0])]

  return dataset


def construct_dataset_with_action_context(
  u: jnp.ndarray,
  x: jnp.ndarray,
  input_horizon: int,
  output_horizon: int,
  action_history_horizon: int,
):
  """
  Construct dataset with past+future action context:
      x_{[n:n+k]} = f(x_{[n-h:n-1]}, u_{[n-1-h_u : n-1+k]})

  Args:
      u: Action trajectory, shape (num_envs, T, action_dim)
      x: State/observation trajectory, shape (num_envs, T, obs_dim)
      input_horizon: Number of historical x steps (h)
      output_horizon: Number of predicted future x steps (k)
      action_history_horizon: Number of additional past actions before n-1 (h_u)

  Returns:
      dataset: List[(network_input, target)]
  """
  if u.ndim != 3 or x.ndim != 3:
    raise ValueError(
      f"This function only takes in dataset of size (num_envs, T, dim), but got dimension for input: {u.ndim} and out: {x.ndim}."
    )
  if u.shape[0] != x.shape[0] or u.shape[1] != x.shape[1]:
    raise ValueError(
      f"u and x must have same num_envs and T, got u.shape={u.shape}, x.shape={x.shape}"
    )
  if action_history_horizon < 0:
    raise ValueError(
      f"action_history_horizon must be non-negative, got {action_history_horizon}"
    )

  first_valid_n = max(input_horizon, action_history_horizon + 1)
  num_samples = u.shape[1] - output_horizon - first_valid_n + 1
  if num_samples <= 0:
    raise ValueError(
      "Not enough timesteps for requested horizons: "
      f"T={u.shape[1]}, input_horizon={input_horizon}, "
      f"output_horizon={output_horizon}, action_history_horizon={action_history_horizon}"
    )

  all_inputs, all_targets = _construct_dataset_arrays_with_action_context(
    u, x, input_horizon, output_horizon, action_history_horizon
  )
  dataset = [(all_inputs[i], all_targets[i]) for i in range(all_inputs.shape[0])]
  return dataset


def save_dataset(
  dataset: List[Tuple[jnp.ndarray, jnp.ndarray]], save_path: str, compress: bool = True
):
  """
  Save the processed dataset to a file using NumPy's compressed or uncompressed format.

  @param dataset: List of (input, target) tuples
  @param save_path: Path to save the dataset (.npz or .npy)
  @param compress: Whether to use compressed format (.npz)
  """
  # Create directory if it doesn't exist
  os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

  # Convert list of tuples to two arrays
  if len(dataset) == 0:
    raise ValueError("Cannot save empty dataset")

  inputs = jnp.stack([sample[0] for sample in dataset])
  targets = jnp.stack([sample[1] for sample in dataset])

  # Convert to NumPy for saving
  inputs_np = np.asarray(inputs)
  targets_np = np.asarray(targets)

  # Save with metadata
  if compress or save_path.endswith(".npz"):
    np.savez_compressed(
      save_path,
      inputs=inputs_np,
      targets=targets_np,
      num_samples=len(dataset),
      input_dim=inputs_np.shape[-1],
      target_dim=targets_np.shape[-1],
    )
    print(f"Saved compressed dataset to {save_path}")
  else:
    # Save as uncompressed .npz
    if not save_path.endswith(".npz"):
      save_path = save_path + ".npz"
    np.savez(
      save_path,
      inputs=inputs_np,
      targets=targets_np,
      num_samples=len(dataset),
      input_dim=inputs_np.shape[-1],
      target_dim=targets_np.shape[-1],
    )
    print(f"Saved uncompressed dataset to {save_path}")

  print(
    f"  Dataset info: {len(dataset)} samples, input_dim={inputs_np.shape[-1]}, target_dim={targets_np.shape[-1]}"
  )


def load_dataset(load_path: str) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
  """
  Load the processed dataset from a file.

  @param load_path: Path to the saved dataset (.npz)
  @return: List of (input, target) tuples as JAX arrays
  """
  if not os.path.exists(load_path):
    raise FileNotFoundError(f"Dataset file not found: {load_path}")

  # Load the .npz file
  data = np.load(load_path)

  # Extract arrays
  if "inputs" not in data or "targets" not in data:
    raise ValueError("Invalid dataset file: missing 'inputs' or 'targets' keys")

  inputs_np = data["inputs"]
  targets_np = data["targets"]

  # Convert to JAX arrays
  inputs_jax = jnp.asarray(inputs_np)
  targets_jax = jnp.asarray(targets_np)

  # Convert to list of tuples
  dataset = [(inputs_jax[i], targets_jax[i]) for i in range(inputs_jax.shape[0])]

  # Print metadata if available
  if "num_samples" in data:
    print(f"Loaded dataset from {load_path}")
    print(
      f"  Dataset info: {data['num_samples']} samples, input_dim={data['input_dim']}, target_dim={data['target_dim']}"
    )
  else:
    print(f"Loaded dataset from {load_path}: {len(dataset)} samples")

  return dataset


def normalize_dataset(
  dataset: List[Tuple[jnp.ndarray, jnp.ndarray]],
  input_range: Tuple[Tuple[float, float], ...],
  output_range: Tuple[Tuple[float, float], ...],
  method: str = "min_max",
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
  """
  Normalize dataset using input and output statistics.

  @param dataset: List of (input, target) tuples
  @param input_range: Tuple of (min, max) or (mean, std) for each input dimension
  @param output_range: Tuple of (min, max) or (mean, std) for each output dimension
  @param method: "min_max" or "std_mean"/"mean_std" (z-score)
  @return: Normalized dataset
  """
  method = method.lower()
  std_mean_aliases = {"std_mean", "mean_std", "zscore", "standard"}

  def input_normalize(x: jnp.ndarray) -> jnp.ndarray:
    """Normalizes the input data based on the provided method."""
    x_norm = jnp.zeros_like(x)
    for i in range(x.shape[-1]):
      first, second = input_range[i]
      if method == "min_max":
        min_val, max_val = first, second
        x_norm = x_norm.at[..., i].set(
          2 * (x[..., i] - min_val) / (max_val - min_val) - 1
        )
      elif method in std_mean_aliases:
        mean, std = first, second
        safe_std = std if abs(std) > 1e-12 else 1.0
        x_norm = x_norm.at[..., i].set((x[..., i] - mean) / safe_std)
      else:
        raise ValueError(
          f"Unknown normalization method '{method}'. Use 'min_max' or 'std_mean'."
        )
    return x_norm

  def output_normalize(y: jnp.ndarray) -> jnp.ndarray:
    """Normalizes the output data based on the provided method."""
    y_norm = jnp.zeros_like(y)
    for i in range(y.shape[-1]):
      first, second = output_range[i]
      if method == "min_max":
        min_val, max_val = first, second
        y_norm = y_norm.at[..., i].set(
          2 * (y[..., i] - min_val) / (max_val - min_val) - 1
        )
      elif method in std_mean_aliases:
        mean, std = first, second
        safe_std = std if abs(std) > 1e-12 else 1.0
        y_norm = y_norm.at[..., i].set((y[..., i] - mean) / safe_std)
      else:
        raise ValueError(
          f"Unknown normalization method '{method}'. Use 'min_max' or 'std_mean'."
        )
    return y_norm

  normalized_dataset = []
  for input_data, output_data in dataset:
    normalized_input = input_normalize(jnp.array(input_data))
    normalized_output = output_normalize(jnp.array(output_data))
    normalized_dataset.append((normalized_input, normalized_output))

  return normalized_dataset


def denormalize_dataset(
  dataset: List[Tuple[jnp.ndarray, jnp.ndarray]],
  input_range: Tuple[Tuple[float, float], ...],
  output_range: Tuple[Tuple[float, float], ...],
  method: str = "min_max",
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
  """
  Denormalize dataset using input and output statistics.

  @param dataset: List of (input, target) tuples (normalized)
  @param input_range: Tuple of (min, max) or (mean, std) for each input dimension
  @param output_range: Tuple of (min, max) or (mean, std) for each output dimension
  @param method: "min_max" or "std_mean"/"mean_std" (z-score inverse)
  @return: Denormalized dataset
  """
  method = method.lower()
  std_mean_aliases = {"std_mean", "mean_std", "zscore", "standard"}

  def input_denormalize(x: jnp.ndarray) -> jnp.ndarray:
    """Denormalizes the input data based on the provided method."""
    x_denorm = jnp.zeros_like(x)
    for i in range(x.shape[-1]):
      first, second = input_range[i]
      if method == "min_max":
        min_val, max_val = first, second
        x_denorm = x_denorm.at[..., i].set(
          (x[..., i] + 1) * (max_val - min_val) / 2 + min_val
        )
      elif method in std_mean_aliases:
        mean, std = first, second
        safe_std = std if abs(std) > 1e-12 else 1.0
        x_denorm = x_denorm.at[..., i].set(x[..., i] * safe_std + mean)
      else:
        raise ValueError(
          f"Unknown denormalization method '{method}'. Use 'min_max' or 'std_mean'."
        )
    return x_denorm

  def output_denormalize(y: jnp.ndarray) -> jnp.ndarray:
    """Denormalizes the output data based on the provided method."""
    y_denorm = jnp.zeros_like(y)
    for i in range(y.shape[-1]):
      first, second = output_range[i]
      if method == "min_max":
        min_val, max_val = first, second
        y_denorm = y_denorm.at[..., i].set(
          (y[..., i] + 1) * (max_val - min_val) / 2 + min_val
        )
      elif method in std_mean_aliases:
        mean, std = first, second
        safe_std = std if abs(std) > 1e-12 else 1.0
        y_denorm = y_denorm.at[..., i].set(y[..., i] * safe_std + mean)
      else:
        raise ValueError(
          f"Unknown denormalization method '{method}'. Use 'min_max' or 'std_mean'."
        )
    return y_denorm

  denormalized_dataset = []
  for input_data, output_data in dataset:
    denormalized_input = input_denormalize(jnp.array(input_data))
    denormalized_output = output_denormalize(jnp.array(output_data))
    denormalized_dataset.append((denormalized_input, denormalized_output))

  return denormalized_dataset


def filter_dataset(
  dataset: List[Tuple[jnp.ndarray, jnp.ndarray]], cutoff_freq=5.0, sample_rate=50.0
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
  """Apply Butterworth low-pass filter to dataset.

  Args:
      dataset: List of (input, target) tuples
      cutoff_freq: Cutoff frequency in Hz
      sample_rate: Sampling rate in Hz (1/dt, e.g., 50 Hz for dt=0.02s)

  Returns:
      Filtered dataset with same structure
  """
  from scipy.signal import butter, filtfilt

  # Stack all samples for filtering
  all_inputs = jnp.array([sample[0] for sample in dataset])
  all_targets = jnp.array([sample[1] for sample in dataset])

  # Design Butterworth low-pass filter
  nyquist = sample_rate / 2
  normal_cutoff = cutoff_freq / nyquist
  b, a = butter(4, normal_cutoff, btype="low", analog=False)

  # Apply filter (filtfilt for zero phase delay)
  all_inputs_filtered = filtfilt(b, a, np.array(all_inputs), axis=0)
  all_targets_filtered = filtfilt(b, a, np.array(all_targets), axis=0)

  # Convert back to JAX and reconstruct dataset
  all_inputs_filtered = jnp.array(all_inputs_filtered)
  all_targets_filtered = jnp.array(all_targets_filtered)

  filtered_dataset = [
    (all_inputs_filtered[i], all_targets_filtered[i]) for i in range(len(dataset))
  ]

  print(f"Applied low-pass filter: cutoff={cutoff_freq} Hz, order=4")

  return filtered_dataset


def filter_trajectories(
  u: jnp.ndarray,
  x: jnp.ndarray,
  cutoff_freq: float = 5.0,
  sample_rate: float = 50.0,
  order: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Apply Butterworth low-pass filter on raw trajectories along time axis.

  Args:
      u: action/command trajectories, shape (num_envs, T, u_dim)
      x: state/observation trajectories, shape (num_envs, T, x_dim)
      cutoff_freq: cutoff frequency in Hz
      sample_rate: sampling rate in Hz
      order: Butterworth filter order

  Returns:
      (u_filtered, x_filtered), both with the same shapes as inputs
  """
  from scipy.signal import butter, filtfilt

  if u.ndim != 3 or x.ndim != 3:
    raise ValueError(
      f"Expected u and x to have shape (num_envs, T, dim), got u.ndim={u.ndim}, x.ndim={x.ndim}"
    )
  if u.shape[0] != x.shape[0] or u.shape[1] != x.shape[1]:
    raise ValueError(f"u and x must share num_envs and T, got {u.shape} vs {x.shape}")
  if sample_rate <= 0:
    raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
  if cutoff_freq <= 0:
    raise ValueError(f"cutoff_freq must be > 0, got {cutoff_freq}")

  nyquist = sample_rate / 2.0
  normal_cutoff = cutoff_freq / nyquist
  if not (0.0 < normal_cutoff < 1.0):
    raise ValueError(
      f"cutoff_freq must be in (0, Nyquist={nyquist}), got cutoff_freq={cutoff_freq}"
    )

  b, a = butter(order, normal_cutoff, btype="low", analog=False)

  # SciPy filtfilt requires enough timesteps along filtered axis.
  padlen = 3 * max(len(a), len(b))
  T = u.shape[1]
  if T <= padlen:
    raise ValueError(
      f"Trajectory too short for filtfilt: T={T}, required > {padlen} (order={order})"
    )

  u_filtered = filtfilt(b, a, np.asarray(u), axis=1)
  x_filtered = filtfilt(b, a, np.asarray(x), axis=1)

  print(
    f"Applied trajectory low-pass filter: cutoff={cutoff_freq} Hz, order={order}, sample_rate={sample_rate} Hz"
  )

  return jnp.asarray(u_filtered), jnp.asarray(x_filtered)
