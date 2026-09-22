"""This file prepare data for dynamics model training, finetuning and plotting"""

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Literal, Optional

import tyro
import yaml

from training.dynamics.exploration import TrajectoryBatch
from training.utils.data_utils import (
  construct_dataset,
  construct_dataset_with_action_context,
  filter_trajectories,
  normalize_dataset,
  save_dataset,
)
from training.utils.tensor_utils import torch_to_jax


@dataclass
class DataConfig:
  type: Literal["one", "task", "check"] = (
    "task"  # "one": single trajectory file; "task": multiple trajectory files from a task
  )
  data_path: Path | None = None  # trajectory batch format
  config_file: Optional[str] = None
  save_path: Path | None = None
  device: str = "cuda:0"
  # model input and output
  obs_list: tuple[int, ...] | None = None
  command_list: tuple[int, ...] | None = None
  input_horizon: int = 10  # history horizon length
  output_horizon: int = 10  # future horizon length
  action_history: bool = False  # whether to include action history in the input

  #### preprocessing
  # normalization
  normalize: bool = True
  normalize_method: str = "min_max"
  input_range: float | tuple[tuple[float, float], ...] = 1.0
  output_range: float | tuple[tuple[float, float], ...] = 1.0
  # low pass filtering
  filter: bool = False
  cutoff_freq: float = 0.1  # cutoff frequency for low pass filter
  sample_rate: float = 1.0  # sample rate of the trajectory data

  def __post_init__(self):
    # check whether loading from config file
    if self.config_file is not None:
      config_path = Path(self.config_file)
      if config_path.exists():
        with open(config_path, "r") as f:
          yaml_config = yaml.safe_load(f)
        # Override dataclass fields with YAML values
        for key, value in yaml_config.items():
          if hasattr(self, key):
            setattr(self, key, value)
      else:
        print(
          f"Warning: config file {self.config_file} does not exist, using command line args only."
        )
    if self.data_path is None:
      raise ValueError("data_path must be specified to a trajectory file!")
    if not Path(self.data_path).exists():
      raise FileNotFoundError(f"Trajectory path does not exist: {self.data_path}")
    elif (
      (self.obs_list is None) or (self.command_list is None)
    ) and self.type != "check":
      raise ValueError(
        "command_list and obs_list must be specified to identify command entries!"
      )


def check_dataset(data_path: Path):
  """Load and check dataset dimensions and statistics."""
  import jax.numpy as jnp

  from training.utils.data_utils import load_dataset

  print(f"\n{'=' * 60}")
  print(f"Checking dataset: {data_path}")
  print(f"{'=' * 60}\n")

  # Load dataset
  dataset = load_dataset(data_path)

  if not dataset:
    print("ERROR: Dataset is empty!")
    return

  # Get first sample to inspect structure
  sample_input, sample_output = dataset[0]

  # Convert to JAX arrays if needed
  if not isinstance(sample_input, jnp.ndarray):
    sample_input = jnp.array(sample_input)
  if not isinstance(sample_output, jnp.ndarray):
    sample_output = jnp.array(sample_output)

  # Collect all inputs and outputs for statistics
  all_inputs = jnp.array([sample[0] for sample in dataset])
  all_outputs = jnp.array([sample[1] for sample in dataset])

  # Print dimensions
  print("Dataset Information:")
  print(f"  Number of samples: {len(dataset)}")
  print(f"  Input dimension:   {sample_input.shape} -> {sample_input.size} features")
  print(f"  Output dimension:  {sample_output.shape} -> {sample_output.size} features")

  # Print statistics
  print("\nInput Statistics:")
  print(f"  Min:  {jnp.min(all_inputs, axis=0)}")
  print(f"  Max:  {jnp.max(all_inputs, axis=0)}")
  print(f"  Mean: {jnp.mean(all_inputs, axis=0)}")
  print(f"  Std:  {jnp.std(all_inputs, axis=0)}")

  print("\nOutput Statistics:")
  print(f"  Min:  {jnp.min(all_outputs, axis=0)}")
  print(f"  Max:  {jnp.max(all_outputs, axis=0)}")
  print(f"  Mean: {jnp.mean(all_outputs, axis=0)}")
  print(f"  Std:  {jnp.std(all_outputs, axis=0)}")

  # Check for NaN or Inf
  has_nan_input = jnp.any(jnp.isnan(all_inputs))
  has_inf_input = jnp.any(jnp.isinf(all_inputs))
  has_nan_output = jnp.any(jnp.isnan(all_outputs))
  has_inf_output = jnp.any(jnp.isinf(all_outputs))

  print("\nData Quality:")
  print(f"  Input NaN:  {has_nan_input}")
  print(f"  Input Inf:  {has_inf_input}")
  print(f"  Output NaN: {has_nan_output}")
  print(f"  Output Inf: {has_inf_output}")

  print(f"\n{'=' * 60}\n")


def main():
  def _build_dim_ranges(
    range_cfg: float | tuple[tuple[float, float], ...] | list,
    dim: int,
    field_name: str,
    method: str,
  ) -> tuple[tuple[float, float], ...]:
    """Build per-dimension (a, b) pairs from scalar/single-pair/per-dim config."""
    method = method.lower()

    def _validate_pair(pair_obj):
      if not isinstance(pair_obj, (list, tuple)) or len(pair_obj) != 2:
        raise ValueError(f"{field_name} pair must have length 2, got: {pair_obj}")
      a, b = pair_obj
      if not isinstance(a, Real) or not isinstance(b, Real):
        raise ValueError(f"{field_name} pair must be numeric, got: {pair_obj}")
      return float(a), float(b)

    # Scalar: keep current min_max behavior; for std_mean treat scalar as shared std with zero mean.
    if isinstance(range_cfg, Real):
      val = float(range_cfg)
      if method in {"std_mean", "mean_std", "zscore", "standard"}:
        if val <= 0:
          raise ValueError(
            f"{field_name} scalar std must be > 0 for '{method}', got {val}"
          )
        return tuple((0.0, val) for _ in range(dim))
      return tuple((-val, val) for _ in range(dim))

    if isinstance(range_cfg, (list, tuple)):
      if len(range_cfg) == 0:
        raise ValueError(f"{field_name} cannot be empty.")

      # Single pair -> broadcast to all dimensions.
      if (
        len(range_cfg) == 2
        and isinstance(range_cfg[0], Real)
        and isinstance(range_cfg[1], Real)
      ):
        pair = (float(range_cfg[0]), float(range_cfg[1]))
        if method in {"std_mean", "mean_std", "zscore", "standard"} and pair[1] <= 0:
          raise ValueError(
            f"{field_name} std must be > 0 for '{method}', got {pair[1]}"
          )
        return tuple(pair for _ in range(dim))

      # Per-dimension pairs.
      pairs = tuple(_validate_pair(p) for p in range_cfg)
      if len(pairs) != dim:
        raise ValueError(
          f"{field_name} length mismatch: expected {dim} pairs, got {len(pairs)}."
        )
      if method in {"std_mean", "mean_std", "zscore", "standard"}:
        bad_idx = [i for i, (_, std) in enumerate(pairs) if std <= 0]
        if bad_idx:
          raise ValueError(
            f"{field_name} std must be > 0 for '{method}'. Invalid indices: {bad_idx}"
          )
      return pairs

    raise ValueError(
      f"Unsupported {field_name} type: {type(range_cfg)}. "
      "Use float, (a,b), or sequence of (a,b) pairs."
    )

  args = tyro.cli(DataConfig)

  if args.type == "check":
    check_dataset(args.data_path)
    return

  traj = TrajectoryBatch.load(args.data_path, device=args.device)

  obs_jax = torch_to_jax(traj.obs)
  input_horizon = args.input_horizon
  output_horizon = args.output_horizon
  if args.type == "task":
    num_tasks = obs_jax.shape[0]
    datasets = []
    for i in range(num_tasks):
      print(f"Constructing dataset for task {i} ...")
      u_i = obs_jax[i : i + 1, :, :][:, :, args.command_list]
      x_i = obs_jax[i : i + 1, :, :][:, :, args.obs_list]
      if args.filter:
        u_i, x_i = filter_trajectories(
          u_i, x_i, cutoff_freq=args.cutoff_freq, sample_rate=args.sample_rate
        )
      if not args.action_history:
        dataset = construct_dataset(
          u_i,
          x_i,
          input_horizon,
          output_horizon,
        )
      else:
        dataset = construct_dataset_with_action_context(
          u_i, x_i, input_horizon, output_horizon, input_horizon
        )
      if args.normalize:
        input_ranges = _build_dim_ranges(
          args.input_range,
          dataset[0][0].shape[0],
          "input_range",
          args.normalize_method,
        )
        output_ranges = _build_dim_ranges(
          args.output_range,
          dataset[0][1].shape[0],
          "output_range",
          args.normalize_method,
        )
        dataset = normalize_dataset(
          dataset,
          input_range=input_ranges,
          output_range=output_ranges,
          method=args.normalize_method,
        )
      datasets.append(dataset)
      # Save to disk
      if args.save_path is not None:
        save_dir = Path(args.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path_i = save_dir / f"dataset_{i}.npz"
        save_dataset(dataset, save_path_i, compress=True)
  elif args.type == "one":
    print("Constructing dataset from single trajectory file ...")
    u = obs_jax[:, :, args.command_list]
    x = obs_jax[:, :, args.obs_list]
    if args.filter:
      u, x = filter_trajectories(
        u, x, cutoff_freq=args.cutoff_freq, sample_rate=args.sample_rate
      )
    if not args.action_history:
      dataset = construct_dataset(
        u,
        x,
        input_horizon,
        output_horizon,
      )
    else:
      dataset = construct_dataset_with_action_context(
        u, x, input_horizon, output_horizon, input_horizon
      )
    if args.normalize:
      input_ranges = _build_dim_ranges(
        args.input_range,
        dataset[0][0].shape[0],
        "input_range",
        args.normalize_method,
      )
      output_ranges = _build_dim_ranges(
        args.output_range,
        dataset[0][1].shape[0],
        "output_range",
        args.normalize_method,
      )
      dataset = normalize_dataset(
        dataset,
        input_range=input_ranges,
        output_range=output_ranges,
        method=args.normalize_method,
      )
    # Save to disk
    if args.save_path is not None:
      save_dataset(dataset, args.save_path, compress=True)


if __name__ == "__main__":
  main()
