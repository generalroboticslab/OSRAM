import os
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import List, Literal, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tyro
import yaml
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


@dataclass
class PlotConfig:
  plot_save_path: str | None = None
  command_list: tuple[int, ...] | None = None
  obs_list: tuple[int, ...] | None = None
  config_file: Optional[str] = None
  # for prediction and gradient
  model: Literal["tron1"] = "tron1"
  model_path: str | List[str] | None = None
  hidden_dims: tuple[int, ...] = (64, 64)
  activation: str = "relu"
  seed: int = 42
  type: str = "mlp"
  Lipschitz_ub: jnp.ndarray = jnp.inf
  J_Lipschitz_ub: jnp.ndarray = jnp.inf
  lambda_c: float = 0.0
  lambda_d: float = 0.0
  inference_mode: bool = False
  input_range: float | tuple[float, float] | tuple[tuple[float, float], ...] = 1.0
  output_range: float | tuple[float, float] | tuple[tuple[float, float], ...] = 1.0
  denormalize: bool = False
  denormalize_method: str = "min_max"
  data_path: str | List[str] | None = None
  rollout_steps: int = 1
  prediction_dims: tuple[int, ...] | None = None
  plot_absolute_errors: bool = False
  time_step: float = 1.0
  # for tracking
  traj_path: str | List[str] | None = None
  traj_start: int = 0  # starting index for trajectory plotting
  traj_length: int = 1000  # length of trajectories to plot
  env_idx: int | None = None  # specific env index to plot if None skip

  task: Literal["tracking", "prediction", "gradient"] = "tracking"
  device: str = "cuda:0"

  # optional matplotlib style overrides
  plot_figure_size: tuple[float, float] | None = None
  plot_font_size: float | None = None
  plot_font_family: str | None = None
  plot_font_serif: tuple[str, ...] | None = None
  plot_axes_labelsize: float | None = None
  plot_xtick_labelsize: float | None = None
  plot_ytick_labelsize: float | None = None
  plot_legend_fontsize: float | None = None
  plot_legend_loc: str | None = None
  plot_legend_bbox_to_anchor: tuple[float, float] | None = None
  plot_separate_legend: bool = False
  plot_legend_save_path: str | None = None
  plot_legend_figure_size: tuple[float, float] | None = None
  plot_legend_ncols: int = 1
  plot_title_fontsize: float | None = None
  plot_xlabel: str | None = None
  plot_ylabel: str | None = None
  plot_ylim: tuple[float, float] | None = None
  plot_ytick_format: str | None = None
  plot_inset_enabled: bool = False
  plot_inset_bounds: tuple[float, float, float, float] | None = None
  plot_inset_time_range: tuple[float, float] | None = None
  plot_inset_ylim: tuple[float, float] | None = None
  plot_inset_xticks: tuple[float, ...] | None = None
  plot_inset_yticks: tuple[float, ...] | None = None
  plot_inset_mark_zoom: bool = True
  plot_detail_save_path: str | None = None
  plot_detail_figure_size: tuple[float, float] | None = None
  plot_detail_time_range: tuple[float, float] | None = None
  plot_detail_ylim: tuple[float, float] | None = None
  plot_detail_xticks: tuple[float, ...] | None = None
  plot_detail_yticks: tuple[float, ...] | None = None
  plot_detail_hide_axes: bool = False
  plot_error_ylabel: str | None = None
  plot_dim_title_template: str | None = None
  plot_error_title_template: str | None = None
  plot_figure_headline: str | None = None
  plot_match_math_font: bool = True
  plot_target_legend_label: str | None = None
  plot_target_color: str | None = None
  plot_baseline_legend_label: str | None = None
  plot_baseline_color: str = "#6b7280"
  plot_baseline_linestyle: str = "-"
  plot_baseline_linewidth: float = 1.5
  plot_model_colors: tuple[str, ...] | None = None
  plot_model_linestyles: tuple[str, ...] | None = None
  plot_model_legend_names: tuple[str, ...] | None = None
  plot_error_legend_names: tuple[str, ...] | None = None
  prediction_config_paths: tuple[str, ...] | None = None
  prediction_config_labels: tuple[str, ...] | None = None
  prediction_config_dims: tuple[int, ...] | None = None

  def __post_init__(self):
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
    if self.model != "tron1":
      raise ValueError("model must be 'tron1'")
    if self.task == "tracking":
      if self.traj_path is None:
        raise ValueError("traj_path must be specified for tracking task!")
      if (self.obs_list is None) or (self.command_list is None):
        raise ValueError("obs_list and command_list must be specified!")
      elif len(self.obs_list) != len(self.command_list):
        raise ValueError("obs_list and command_list must have the same length!")
    elif self.task == "prediction" or self.task == "gradient":
      using_prediction_configs = self.prediction_config_paths is not None
      if not using_prediction_configs:
        if self.model_path is None:
          raise ValueError("model_path must be specified for prediction task!")
        if self.data_path is None:
          raise ValueError("data_path must be specified for prediction task!")
      if (
        self.prediction_config_labels is not None
        and self.prediction_config_paths is not None
        and len(self.prediction_config_labels) != len(self.prediction_config_paths)
      ):
        raise ValueError(
          "prediction_config_labels length must match prediction_config_paths length!"
        )
      if (
        self.prediction_config_dims is not None
        and self.prediction_config_paths is not None
        and len(self.prediction_config_dims) != len(self.prediction_config_paths)
      ):
        raise ValueError(
          "prediction_config_dims length must match prediction_config_paths length!"
        )
      if self.time_step <= 0:
        raise ValueError(f"time_step must be > 0, got {self.time_step}")


def resolve_jax_device(device: str):
  device = str(device).strip().lower()
  if device in {"cpu", "cpu:0"}:
    platform = "cpu"
    device_idx = 0
  elif device in {"gpu", "cuda"}:
    platform = "gpu"
    device_idx = 0
  elif device.startswith("gpu:") or device.startswith("cuda:"):
    platform = "gpu"
    device_idx = int(device.split(":", 1)[1])
  elif device.startswith("cpu:"):
    platform = "cpu"
    device_idx = int(device.split(":", 1)[1])
  else:
    raise ValueError(
      f"Unsupported JAX device '{device}'. Use 'cpu', 'gpu', 'cuda', or 'cuda:N'."
    )

  devices = jax.devices(platform)
  if device_idx >= len(devices):
    available = ", ".join(str(dev) for dev in devices) or "none"
    raise ValueError(
      f"Requested JAX device '{device}' but only {len(devices)} {platform} "
      f"device(s) are visible: {available}"
    )
  return devices[device_idx]


def apply_plot_style(args: PlotConfig):
  rc_updates = {}
  if args.plot_figure_size is not None:
    rc_updates["figure.figsize"] = tuple(args.plot_figure_size)
  if args.plot_font_size is not None:
    rc_updates["font.size"] = args.plot_font_size
  if args.plot_font_family is not None:
    rc_updates["font.family"] = args.plot_font_family
  if args.plot_font_serif is not None:
    rc_updates["font.serif"] = list(args.plot_font_serif)
  if args.plot_axes_labelsize is not None:
    rc_updates["axes.labelsize"] = args.plot_axes_labelsize
  if args.plot_xtick_labelsize is not None:
    rc_updates["xtick.labelsize"] = args.plot_xtick_labelsize
  if args.plot_ytick_labelsize is not None:
    rc_updates["ytick.labelsize"] = args.plot_ytick_labelsize
  if args.plot_legend_fontsize is not None:
    rc_updates["legend.fontsize"] = args.plot_legend_fontsize
  if args.plot_title_fontsize is not None:
    rc_updates["axes.titlesize"] = args.plot_title_fontsize
  if rc_updates:
    plt.rcParams.update(rc_updates)

  if args.plot_match_math_font:
    preferred_font = None
    if args.plot_font_family == "serif" and args.plot_font_serif is not None:
      if len(args.plot_font_serif) > 0:
        preferred_font = args.plot_font_serif[0]
    elif args.plot_font_family is not None:
      preferred_font = args.plot_font_family

    if preferred_font is not None:
      plt.rcParams.update(
        {
          "mathtext.fontset": "custom",
          "mathtext.rm": preferred_font,
          "mathtext.it": f"{preferred_font}:italic",
          "mathtext.bf": f"{preferred_font}:bold",
          "mathtext.cal": f"{preferred_font}:italic",
          "mathtext.sf": preferred_font,
          "mathtext.tt": preferred_font,
        }
      )


def plot_tracking(args: PlotConfig):
  from training.dynamics.exploration import TrajectoryBatch
  from training.utils.tensor_utils import torch_to_numpy

  # Handle single path or list of paths
  if isinstance(args.traj_path, str):
    traj_paths = [args.traj_path]
  else:
    traj_paths = args.traj_path

  # Load all trajectories
  all_trajectories = []
  for path in traj_paths:
    traj = TrajectoryBatch.load(path, device=args.device)
    all_trajectories.append(traj)

  # Process each trajectory
  all_real_trajs = []

  # Extract reference trajectory from first trajectory only (should be same for all)
  obs_numpy_first = torch_to_numpy(all_trajectories[0].obs)
  traj_end = args.traj_start + args.traj_length
  reference_trajs = obs_numpy_first[:, args.traj_start : traj_end, args.command_list]

  for traj in all_trajectories:
    obs_numpy = torch_to_numpy(traj.obs)  # shape (n_envs, T, obs_dim)
    real_trajs = obs_numpy[:, args.traj_start : traj_end, args.obs_list]

    if real_trajs.ndim == 2:
      real_trajs = real_trajs[..., None]

    all_real_trajs.append(real_trajs)

  if reference_trajs.ndim == 2:
    reference_trajs = reference_trajs[..., None]

  # Calculate worst env (using first trajectory for reference)
  tracking_errors = np.abs(reference_trajs - all_real_trajs[0])
  # mae_per_env = np.mean(tracking_errors, axis=1)
  rmse_per_env = np.sqrt(np.mean(tracking_errors**2, axis=1))
  worst_env_idx = np.argmax(rmse_per_env, axis=0)

  n_channels = reference_trajs.shape[2]
  n_rows = n_channels

  # Color palette for different trajectories
  colors = ["tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]

  ### Plot worst environments
  fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), squeeze=False)
  for c in range(n_channels):
    ax = axes[c, 0]
    env_idx = int(worst_env_idx[c])

    # Plot reference (only once, same for all)
    ref = reference_trajs[env_idx, :, c]
    ax.plot(ref, label="Reference", color="tab:blue", linewidth=2, zorder=10)

    # Plot all trajectories with errors in labels
    for traj_idx, (real_trajs, traj_path) in enumerate(
      zip(all_real_trajs, traj_paths, strict=True)
    ):
      real = real_trajs[env_idx, :, c]
      err = np.abs(ref - real)
      mae_val = np.mean(err)
      rmse_val = np.sqrt(np.mean(err**2))
      color = colors[traj_idx % len(colors)]
      label = f"Traj {traj_idx + 1} (MAE={mae_val:.4f}, RMSE={rmse_val:.4f}): {Path(traj_path).stem}"
      ax.plot(real, label=label, color=color, linewidth=1.5, alpha=0.8)

    ax.set_title(
      f"Channel {c}: Worst Env {env_idx} (Steps {args.traj_start}-{traj_end})"
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

  plt.tight_layout()
  if args.plot_save_path is not None:
    os.makedirs(os.path.dirname(args.plot_save_path) or ".", exist_ok=True)
    plt.savefig(args.plot_save_path + "_worse", dpi=300, bbox_inches="tight")
  plt.show()

  ### Plot specific environment if requested
  if args.env_idx is not None:
    env_idx = args.env_idx
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), squeeze=False)
    for c in range(n_channels):
      ax = axes[c, 0]

      # Plot reference (only once)
      ref = reference_trajs[env_idx, :, c]
      ax.plot(ref, label="Reference", color="tab:blue", linewidth=2, zorder=10)

      # Plot all trajectories with errors in labels
      for traj_idx, (real_trajs, traj_path) in enumerate(
        zip(all_real_trajs, traj_paths, strict=True)
      ):
        real = real_trajs[env_idx, :, c]
        err = np.abs(ref - real)
        mae_val = np.mean(err)
        rmse_val = np.sqrt(np.mean(err**2))
        color = colors[traj_idx % len(colors)]
        label = f"Traj {traj_idx + 1} (MAE={mae_val:.4f}, RMSE={rmse_val:.4f}): {Path(traj_path).stem}"
        ax.plot(real, label=label, color=color, linewidth=1.5, alpha=0.8)

      ax.set_title(f"Channel {c}: Env {env_idx} (Steps {args.traj_start}-{traj_end})")
      ax.set_xlabel("Time step")
      ax.set_ylabel("Value")
      ax.legend(loc="upper right", fontsize=8)
      ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.plot_save_path is not None:
      plt.savefig(
        args.plot_save_path + "_" + str(args.env_idx), dpi=300, bbox_inches="tight"
      )
    plt.show()


def plot_prediction(args: PlotConfig):
  from training.utils.data_utils import load_dataset

  def _build_dim_ranges(
    range_cfg: float | tuple[float, float] | tuple[tuple[float, float], ...] | list,
    dim: int,
    field_name: str,
    method: str,
  ) -> tuple[tuple[float, float], ...]:
    """Build per-dimension (a, b) pairs from scalar/single-pair/per-dim config."""
    method = method.lower()
    std_mean_aliases = {"std_mean", "mean_std", "zscore", "standard"}

    def _validate_pair(pair_obj):
      if not isinstance(pair_obj, (list, tuple)) or len(pair_obj) != 2:
        raise ValueError(f"{field_name} pair must have length 2, got: {pair_obj}")
      a, b = pair_obj
      if not isinstance(a, Real) or not isinstance(b, Real):
        raise ValueError(f"{field_name} pair must be numeric, got: {pair_obj}")
      return float(a), float(b)

    # Scalar: min_max -> (-v, v); std_mean -> (0, v)
    if isinstance(range_cfg, Real):
      val = float(range_cfg)
      if method in std_mean_aliases:
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
        if method in std_mean_aliases and pair[1] <= 0:
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
      if method in std_mean_aliases:
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

  def _create_prediction_model(
    model_name: str,
    actual_input_dim: int,
    actual_output_dim: int,
    cfg_args: PlotConfig,
    input_ranges,
    output_ranges,
  ):
    if model_name != "tron1":
      raise ValueError(f"Unsupported model type: {model_name}")

    from training.dynamics.training.tron1.tron1_model import (
      Tron1Model,
      Tron1ModelConfig,
    )

    cfg = Tron1ModelConfig(
      input_dim=actual_input_dim,
      output_dim=actual_output_dim,
      hidden_sizes=cfg_args.hidden_dims,
      activation=cfg_args.activation,
      key=cfg_args.seed,
      type=cfg_args.type,
      Lipschitz_ub=cfg_args.Lipschitz_ub,
      J_Lipschitz_ub=cfg_args.J_Lipschitz_ub,
      inference_mode=cfg_args.inference_mode,
      lambda_c=cfg_args.lambda_c,
      lambda_d=cfg_args.lambda_d,
      input_range=input_ranges,
      output_range=output_ranges,
    )
    return Tron1Model(cfg)

  def _run_prediction(cfg_args: PlotConfig):
    if isinstance(cfg_args.model_path, str):
      model_paths = [cfg_args.model_path]
    else:
      model_paths = cfg_args.model_path

    dataset = load_dataset(cfg_args.data_path)
    inputs = jnp.array([sample[0] for sample in dataset])
    targets = jnp.array([sample[1] for sample in dataset])

    if inputs.ndim != 2:
      raise ValueError("Currently only support 2D inputs/targets (num_samples, dim)!")

    traj_end = cfg_args.traj_start + cfg_args.traj_length
    inputs = inputs[cfg_args.traj_start : traj_end, :]
    targets = targets[cfg_args.traj_start : traj_end, :]
    actual_input_dim = inputs.shape[1]
    actual_output_dim = targets.shape[1]
    denorm_method = cfg_args.denormalize_method.lower()

    input_ranges = _build_dim_ranges(
      cfg_args.input_range, actual_input_dim, "input_range", denorm_method
    )
    output_ranges = _build_dim_ranges(
      cfg_args.output_range, actual_output_dim, "output_range", denorm_method
    )

    print(
      f"Using trajectory slice: [{cfg_args.traj_start}:{traj_end}] ({len(inputs)} samples)"
    )

    if cfg_args.command_list is None or cfg_args.obs_list is None:
      raise ValueError(
        "command_list and obs_list must be specified for rollout predictions!"
      )

    num_command_dims = len(cfg_args.command_list)
    num_obs_dims = len(cfg_args.obs_list)
    output_horizon = actual_output_dim // num_obs_dims
    command_part_size = num_command_dims * output_horizon
    obs_history_size = actual_input_dim - command_part_size
    input_horizon = obs_history_size // num_obs_dims

    print(
      f"Inferred structure: input_horizon={input_horizon}, output_horizon={output_horizon}"
    )
    print(f"  obs_dims={num_obs_dims}, command_dims={num_command_dims}")

    all_predictions = []
    all_errors = []

    for model_idx, model_path in enumerate(model_paths):
      print(f"\nLoading model {model_idx + 1}/{len(model_paths)}: {model_path}")
      model = _create_prediction_model(
        cfg_args.model,
        actual_input_dim,
        actual_output_dim,
        cfg_args,
        input_ranges,
        output_ranges,
      )

      model.load(model_path)
      model.set_inference_mode(True)

      num_samples = inputs.shape[0]
      max_rollout_samples = min(
        num_samples - cfg_args.rollout_steps, cfg_args.traj_length
      )

      predictions = []

      for i in range(max_rollout_samples):
        current_input = inputs[i]
        rollout_preds = []

        for step in range(cfg_args.rollout_steps):
          pred = model(current_input)
          rollout_preds.append(pred)

          if i + step + 1 >= num_samples:
            break

          obs_history = current_input[:obs_history_size]
          obs_history_reshaped = obs_history.reshape(input_horizon, num_obs_dims)
          pred_reshaped = pred.reshape(output_horizon, num_obs_dims)

          if input_horizon > output_horizon:
            new_obs_history = jnp.concatenate(
              [obs_history_reshaped[output_horizon:], pred_reshaped], axis=0
            )
          else:
            new_obs_history = pred_reshaped[-input_horizon:]

          new_obs_history_flat = new_obs_history.ravel()
          next_commands = inputs[i + step + 1, -command_part_size:]
          current_input = jnp.concatenate([new_obs_history_flat, next_commands])

        if len(rollout_preds) == cfg_args.rollout_steps:
          predictions.append(rollout_preds[-1])

      if len(predictions) == 0:
        raise ValueError(f"No valid predictions generated for model {model_path}")

      predictions = jnp.stack(predictions, axis=0)
      targets_aligned = targets[
        cfg_args.rollout_steps - 1 : cfg_args.rollout_steps - 1 + len(predictions)
      ]

      if cfg_args.denormalize:
        if denorm_method == "min_max":
          predictions = model.output_denormalize(predictions)
          targets_aligned_denorm = model.output_denormalize(targets_aligned)
        elif denorm_method in {"std_mean", "mean_std", "zscore", "standard"}:
          predictions = model.output_denormalize_std(predictions)
          targets_aligned_denorm = model.output_denormalize_std(targets_aligned)
        else:
          raise ValueError("unknown denormalize method")
      else:
        targets_aligned_denorm = targets_aligned

      errors = predictions - targets_aligned_denorm
      mse = jnp.mean(errors**2)
      mae = jnp.mean(jnp.abs(errors))
      rmse = jnp.sqrt(mse)

      print(
        f"  Model {model_idx + 1} - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}"
      )

      all_predictions.append(predictions)
      all_errors.append(errors)

    targets_aligned = targets[
      cfg_args.rollout_steps - 1 : cfg_args.rollout_steps - 1 + len(all_predictions[0])
    ]
    if cfg_args.denormalize:
      temp_model = _create_prediction_model(
        cfg_args.model,
        actual_input_dim,
        actual_output_dim,
        cfg_args,
        input_ranges,
        output_ranges,
      )
      if denorm_method == "min_max":
        targets_aligned = temp_model.output_denormalize(targets_aligned)
      elif denorm_method in {"std_mean", "mean_std", "zscore", "standard"}:
        targets_aligned = temp_model.output_denormalize_std(targets_aligned)
      else:
        raise ValueError("unknown denormalization method")

    targets_np = np.asarray(targets_aligned)
    predictions_list = [np.asarray(pred) for pred in all_predictions]

    predictions_reshaped_list = [
      pred.reshape(-1, output_horizon, num_obs_dims) for pred in predictions_list
    ]
    targets_reshaped = targets_np.reshape(-1, output_horizon, num_obs_dims)

    return {
      "model_paths": model_paths,
      "target_final": targets_reshaped[:, -1, :],
      "pred_finals": [pred[:, -1, :] for pred in predictions_reshaped_list],
    }

  def _load_plot_config(config_path: str) -> PlotConfig:
    return PlotConfig(config_file=config_path, task="prediction")

  def _plot_prediction_overlay(config_paths: tuple[str, ...]):
    run_cfgs = [_load_plot_config(path) for path in config_paths]
    for cfg in run_cfgs:
      cfg.rollout_steps = args.rollout_steps
      cfg.traj_start = args.traj_start
      cfg.traj_length = args.traj_length

    run_results = []
    for cfg in run_cfgs:
      jax_device = resolve_jax_device(cfg.device)
      print(f"Using JAX device for prediction: {jax_device}")
      with jax.default_device(jax_device):
        run_results.append(_run_prediction(cfg))

    if args.prediction_config_labels is not None:
      labels = list(args.prediction_config_labels)
    else:
      labels = [Path(path).stem for path in config_paths]

    if args.prediction_config_dims is not None:
      dims = list(args.prediction_config_dims)
    else:
      dims = [0] * len(run_cfgs)

    time_seconds = None
    fig, ax = plt.subplots(1, 1)
    colors = (
      list(args.plot_model_colors)
      if args.plot_model_colors is not None
      else [
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
      ]
    )
    linestyles = (
      list(args.plot_model_linestyles)
      if args.plot_model_linestyles is not None
      else ["-", "--", "-.", ":"]
    )

    plotted_target = False
    overlay_series = []
    for idx, (cfg_args, result, dim, label) in enumerate(
      zip(run_cfgs, run_results, dims, labels, strict=True)
    ):
      target_final = result["target_final"]
      pred_finals = result["pred_finals"]

      if dim < 0 or dim >= target_final.shape[1]:
        raise ValueError(
          f"prediction_config_dims[{idx}]={dim} is out of range for "
          f"{config_paths[idx]} with output dim {target_final.shape[1]}"
        )

      if len(pred_finals) != 1:
        raise ValueError(
          "Overlay mode expects each referenced config to contain exactly one model_path"
        )

      run_time = np.arange(target_final.shape[0]) * args.time_step
      if time_seconds is None:
        time_seconds = run_time

      target_dim = target_final[:, dim]
      pred_dim = pred_finals[0][:, dim]
      mae = float(np.mean(np.abs(pred_dim - target_dim)))
      print(f"  Overlay {label} dim {dim} MAE: {mae:.6f}")
      color = colors[idx % len(colors)]
      linestyle = linestyles[idx % len(linestyles)]

      if not plotted_target:
        target_label = (
          args.plot_target_legend_label
          if args.plot_target_legend_label is not None
          else (
            "Target"
            if cfg_args.plot_target_legend_label is None
            else cfg_args.plot_target_legend_label
          )
        )
        ax.plot(
          run_time,
          target_dim,
          label=target_label,
          color="black" if args.plot_target_color is None else args.plot_target_color,
          linewidth=2,
          linestyle="--",
          zorder=1,
        )
        plotted_target = True
        overlay_series.append(
          {
            "time": run_time,
            "values": target_dim,
            "color": "black"
            if args.plot_target_color is None
            else args.plot_target_color,
            "linestyle": "--",
            "linewidth": 2,
            "zorder": 1,
          }
        )

      ax.plot(
        run_time,
        pred_dim,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=1.5,
        alpha=1.0,
        zorder=2 + idx,
      )
      overlay_series.append(
        {
          "time": run_time,
          "values": pred_dim,
          "color": color,
          "linestyle": linestyle,
          "linewidth": 1.5,
          "zorder": 2 + idx,
        }
      )

    x_label = "Time (s)" if args.plot_xlabel is None else args.plot_xlabel
    y_label = "Value" if args.plot_ylabel is None else args.plot_ylabel
    title = (
      "Prediction Comparison"
      if args.plot_figure_headline is None
      else args.plot_figure_headline
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if args.plot_ylim is not None:
      ax.set_ylim(args.plot_ylim)
    if args.plot_ytick_format is not None:
      ax.yaxis.set_major_formatter(FormatStrFormatter(args.plot_ytick_format))
    ax.grid(True, alpha=0.3)

    if args.plot_inset_enabled:
      inset_bounds = (
        tuple(args.plot_inset_bounds)
        if args.plot_inset_bounds is not None
        else (0.52, 0.12, 0.42, 0.36)
      )
      inset_ax = ax.inset_axes(inset_bounds)
      for series in overlay_series:
        inset_ax.plot(
          series["time"],
          series["values"],
          color=series["color"],
          linestyle=series["linestyle"],
          linewidth=series["linewidth"],
          zorder=series["zorder"],
        )
      if args.plot_inset_time_range is not None:
        inset_ax.set_xlim(tuple(args.plot_inset_time_range))
      if args.plot_inset_ylim is not None:
        inset_ax.set_ylim(tuple(args.plot_inset_ylim))
      if args.plot_inset_xticks is not None:
        inset_ax.set_xticks(tuple(args.plot_inset_xticks))
      if args.plot_inset_yticks is not None:
        inset_ax.set_yticks(tuple(args.plot_inset_yticks))
      if args.plot_ytick_format is not None:
        inset_ax.yaxis.set_major_formatter(FormatStrFormatter(args.plot_ytick_format))
      inset_ax.grid(True, alpha=0.25)
      inset_ax.tick_params(labelsize=max(6, (args.plot_xtick_labelsize or 8) - 2))
      if args.plot_inset_mark_zoom:
        ax.indicate_inset_zoom(inset_ax, edgecolor="black", alpha=0.6)

    if args.plot_detail_save_path is not None:
      detail_fig, detail_ax = plt.subplots(
        1,
        1,
        figsize=(
          tuple(args.plot_detail_figure_size)
          if args.plot_detail_figure_size is not None
          else tuple(args.plot_figure_size or (2.5, 2.2))
        ),
      )
      for series in overlay_series:
        detail_ax.plot(
          series["time"],
          series["values"],
          color=series["color"],
          linestyle=series["linestyle"],
          linewidth=series["linewidth"],
          zorder=series["zorder"],
        )
      if args.plot_detail_time_range is not None:
        detail_ax.set_xlim(tuple(args.plot_detail_time_range))
      if args.plot_detail_ylim is not None:
        detail_ax.set_ylim(tuple(args.plot_detail_ylim))
      if args.plot_detail_xticks is not None:
        detail_ax.set_xticks(tuple(args.plot_detail_xticks))
      if args.plot_detail_yticks is not None:
        detail_ax.set_yticks(tuple(args.plot_detail_yticks))
      if args.plot_ytick_format is not None:
        detail_ax.yaxis.set_major_formatter(FormatStrFormatter(args.plot_ytick_format))
      if args.plot_detail_hide_axes:
        detail_ax.set_xlabel("")
        detail_ax.set_ylabel("")
        detail_ax.set_xticks([])
        detail_ax.set_yticks([])
        detail_ax.tick_params(
          left=False,
          bottom=False,
          labelleft=False,
          labelbottom=False,
        )
        detail_ax.grid(False)
      else:
        detail_ax.set_xlabel(x_label)
        detail_ax.set_ylabel(y_label)
        detail_ax.grid(True, alpha=0.3)
      detail_fig.tight_layout()
      os.makedirs(os.path.dirname(args.plot_detail_save_path) or ".", exist_ok=True)
      detail_fig.savefig(args.plot_detail_save_path, dpi=300, bbox_inches="tight")
      plt.close(detail_fig)

    handles, legend_labels = ax.get_legend_handles_labels()
    if args.plot_baseline_legend_label is not None:
      handles.append(
        Line2D(
          [0],
          [0],
          color=args.plot_baseline_color,
          linestyle=args.plot_baseline_linestyle,
          linewidth=args.plot_baseline_linewidth,
        )
      )
      legend_labels.append(args.plot_baseline_legend_label)
    if args.plot_separate_legend:
      legend_save_path = args.plot_legend_save_path
      if legend_save_path is None and args.plot_save_path is not None:
        root, ext = os.path.splitext(args.plot_save_path)
        legend_save_path = f"{root}_legend{ext or '.png'}"
      legend_fig = plt.figure(
        figsize=(
          tuple(args.plot_legend_figure_size)
          if args.plot_legend_figure_size is not None
          else (3.2, max(0.5, 0.25 * len(handles)))
        )
      )
      legend_fig.legend(
        handles,
        legend_labels,
        loc="center",
        ncols=args.plot_legend_ncols,
        frameon=False,
      )
      legend_fig.tight_layout()
      if legend_save_path is not None:
        os.makedirs(os.path.dirname(legend_save_path) or ".", exist_ok=True)
        legend_fig.savefig(legend_save_path, dpi=300, bbox_inches="tight")
      plt.close(legend_fig)
      plt.tight_layout()
    else:
      legend_kwargs = {
        "loc": "best" if args.plot_legend_loc is None else args.plot_legend_loc
      }
      if args.plot_legend_bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = tuple(args.plot_legend_bbox_to_anchor)
        legend_kwargs["borderaxespad"] = 0.0
      ax.legend(handles, legend_labels, **legend_kwargs)
      if args.plot_legend_bbox_to_anchor is not None:
        plt.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
      else:
        plt.tight_layout()

    if args.plot_save_path is not None:
      os.makedirs(os.path.dirname(args.plot_save_path) or ".", exist_ok=True)
      plt.savefig(args.plot_save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return

  if args.prediction_config_paths is not None:
    _plot_prediction_overlay(args.prediction_config_paths)
    return

  jax_device = resolve_jax_device(args.device)
  print(f"Using JAX device for prediction: {jax_device}")
  with jax.default_device(jax_device):
    result = _run_prediction(args)
  model_paths = result["model_paths"]
  target_final = result["target_final"]
  pred_finals = result["pred_finals"]

  output_dim = target_final.shape[1]
  if args.prediction_dims is None:
    selected_dims = list(range(output_dim))
  else:
    selected_dims = list(args.prediction_dims)
    bad_dims = [d for d in selected_dims if d < 0 or d >= output_dim]
    if bad_dims:
      raise ValueError(
        f"prediction_dims contains invalid indices {bad_dims}. Valid range: [0, {output_dim - 1}]"
      )
    if len(selected_dims) == 0:
      raise ValueError("prediction_dims cannot be empty when provided")

  n_rows = 2 if args.plot_absolute_errors else 1
  fig_height = 8 if args.plot_absolute_errors else 4
  fig, axes = plt.subplots(
    n_rows,
    len(selected_dims),
    figsize=(5 * len(selected_dims), fig_height),
    squeeze=False,
  )

  time_seconds = np.arange(len(target_final)) * args.time_step
  colors = (
    list(args.plot_model_colors)
    if args.plot_model_colors is not None
    else [
      "tab:orange",
      "tab:green",
      "tab:purple",
      "tab:brown",
      "tab:pink",
      "tab:cyan",
    ]
  )

  x_label = "Time (s)" if args.plot_xlabel is None else args.plot_xlabel
  y_label = "Value" if args.plot_ylabel is None else args.plot_ylabel
  error_ylabel = (
    "Absolute Error" if args.plot_error_ylabel is None else args.plot_error_ylabel
  )
  dim_title_template = (
    "Dim {dim}: Predictions vs Targets"
    if args.plot_dim_title_template is None
    else args.plot_dim_title_template
  )
  error_title_template = (
    "Dim {dim}: Absolute Errors"
    if args.plot_error_title_template is None
    else args.plot_error_title_template
  )
  target_legend_label = (
    "Targets"
    if args.plot_target_legend_label is None
    else args.plot_target_legend_label
  )

  if args.plot_model_legend_names is not None:
    if len(args.plot_model_legend_names) != len(model_paths):
      raise ValueError(
        "plot_model_legend_names length must match number of models: "
        f"{len(args.plot_model_legend_names)} != {len(model_paths)}"
      )
    model_legend_names = list(args.plot_model_legend_names)
  else:
    model_legend_names = [
      f"Model {idx + 1}: {Path(model_path).stem}"
      for idx, model_path in enumerate(model_paths)
    ]

  if args.plot_error_legend_names is not None:
    if len(args.plot_error_legend_names) != len(model_paths):
      raise ValueError(
        "plot_error_legend_names length must match number of models: "
        f"{len(args.plot_error_legend_names)} != {len(model_paths)}"
      )
    error_legend_names = list(args.plot_error_legend_names)
  else:
    error_legend_names = model_legend_names

  def _safe_format(template: str, **kwargs) -> str:
    try:
      return template.format(**kwargs)
    except Exception:
      return template

  for col, dim in enumerate(selected_dims):
    target_dim = target_final[:, dim]

    # Top row: Predictions vs Targets
    axes[0, col].plot(
      time_seconds,
      target_dim,
      label=target_legend_label,
      color="tab:blue" if args.plot_target_color is None else args.plot_target_color,
      alpha=0.9,
      linewidth=2,
      zorder=1,
    )

    for model_idx, (pred_final, _model_path) in enumerate(
      zip(pred_finals, model_paths, strict=True)
    ):
      pred_dim = pred_final[:, dim]
      color = colors[model_idx % len(colors)]
      label = model_legend_names[model_idx]
      axes[0, col].plot(
        time_seconds,
        pred_dim,
        label=label,
        color=color,
        alpha=0.7,
        linewidth=1.5,
        zorder=2 + model_idx,
      )

    axes[0, col].set_title(_safe_format(dim_title_template, dim=dim))
    axes[0, col].set_xlabel(x_label)
    axes[0, col].set_ylabel(y_label)
    if args.plot_ylim is not None:
      axes[0, col].set_ylim(args.plot_ylim)
    if args.plot_ytick_format is not None:
      axes[0, col].yaxis.set_major_formatter(FormatStrFormatter(args.plot_ytick_format))
    axes[0, col].legend(fontsize=8, loc="best")
    axes[0, col].grid(True, alpha=0.3)

    if args.plot_absolute_errors:
      # Bottom row: Absolute Errors
      for model_idx, (pred_final, _model_path) in enumerate(
        zip(pred_finals, model_paths, strict=True)
      ):
        error_dim = np.abs(pred_final[:, dim] - target_dim)
        color = colors[model_idx % len(colors)]
        label = error_legend_names[model_idx]
        axes[1, col].plot(
          time_seconds,
          error_dim,
          color=color,
          alpha=0.7,
          linewidth=1.5,
          label=label,
        )

      axes[1, col].set_title(_safe_format(error_title_template, dim=dim))
      axes[1, col].set_xlabel(x_label)
      axes[1, col].set_ylabel(error_ylabel)
      axes[1, col].legend(fontsize=8, loc="best")
      axes[1, col].grid(True, alpha=0.3)

  title_suffix = " (Denormalized)" if args.denormalize else " (Normalized)"
  figure_headline_template = (
    "Multi-Model Predictions {rollout_steps}-step Rollout{title_suffix}"
    if args.plot_figure_headline is None
    else args.plot_figure_headline
  )
  figure_headline = _safe_format(
    figure_headline_template,
    rollout_steps=args.rollout_steps,
    title_suffix=title_suffix,
    num_dims=len(selected_dims),
  )
  fig.suptitle(figure_headline, fontsize=14)
  plt.tight_layout()

  if args.plot_save_path is not None:
    os.makedirs(os.path.dirname(args.plot_save_path) or ".", exist_ok=True)
    plt.savefig(args.plot_save_path, dpi=300, bbox_inches="tight")

  plt.show()


def plot_gradient(args: PlotConfig):
  # plot the gradient of the model w.r.t. inputs
  raise NotImplementedError("Gradient plotting not implemented yet.")


def main():
  args = tyro.cli(PlotConfig)
  apply_plot_style(args)
  if args.task == "tracking":
    plot_tracking(args)
  elif args.task == "prediction":
    plot_prediction(args)


if __name__ == "__main__":
  main()
