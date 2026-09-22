import abc

# Force single GPU usage before any JAX imports
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import jax.numpy as jnp
from flax import nnx

if "XLA_PYTHON_CLIENT_DEVICE_COUNT" not in os.environ:
  os.environ["XLA_PYTHON_CLIENT_DEVICE_COUNT"] = "1"


@dataclass
class ModelConfig:
  input_dim: int  # [obs_{t-h}, ..., obs_t, act_{t-h}, ..., act_t]
  output_dim: int
  input_range: Tuple[Tuple[float, float]] | Tuple[float, float] = (
    None  # for denormalization
  )
  output_range: Tuple[Tuple[float, float]] | Tuple[float, float] = None
  load_path: str | None = None
  save_path: str | None = None
  key: int | None = 0
  hidden_sizes: Sequence[int] = (
    None  # MLP structure: list of hidden layer sizes (e.g. [64, 64]). If None, raise an error.
  )
  activation: str = "relu"  # activation: "relu" | "tanh" | "identity" | "leaky_relu" | "softplus" | "mish"
  type: str = "mlp"  # type of model, "SNS, mlp, Lipschitz", if this param is not mlp then the activation entry will be overridden.
  obs_history_dim: int = 30  # obs size * obs history length.

  # Lipschitz or SNS specific parameters
  Lipschitz_ub: Optional[float] = jnp.inf  # upper bound on Lipschitz constant
  J_Lipschitz_ub: Optional[float] = (
    jnp.inf
  )  # upper bound on Jacobian Lipschitz constant
  inference_mode: Optional[bool] = False
  lambda_c: Optional[float] = (
    0.0  # weight for Lipschitz constant penalty in the loss function
  )
  lambda_d: Optional[float] = 0.0  # weight for Jacobian Lips

  # auto regressive training parameters
  shuffle_length: Optional[int] = (
    None  # length of sequences for auto regressive training
  )
  lambda_r: Optional[float] = 0.0  # weight for auto regressive loss
  regressive_loss_discount: Optional[float] = 1.0

  def __post_init__(self):
    if self.input_dim <= 0:
      raise ValueError(f"input_dim must be positive, got {self.input_dim}")
    if self.output_dim <= 0:
      raise ValueError(f"output_dim must be positive, got {self.output_dim}")

    if self.obs_history_dim <= 0:
      raise ValueError(f"obs_history_dim must be positive, got {self.obs_history_dim}")
    if self.obs_history_dim > self.input_dim:
      raise ValueError(
        "obs_history_dim cannot exceed input_dim: "
        f"{self.obs_history_dim} > {self.input_dim}"
      )
    # not valid for open loop case
    # if self.obs_history_dim % self.output_dim != 0:
    #   raise ValueError(
    #     "obs_history_dim must be a multiple of output_dim for history shifting: "
    #     f"obs_history_dim={self.obs_history_dim}, output_dim={self.output_dim}"
    #   )

    if self.shuffle_length is not None and self.shuffle_length <= 1:
      raise ValueError(
        f"shuffle_length must be > 1 for autoregressive training, got {self.shuffle_length}"
      )
    if self.regressive_loss_discount is not None and self.regressive_loss_discount < 0:
      raise ValueError(
        "regressive_loss_discount must be non-negative, got "
        f"{self.regressive_loss_discount}"
      )


class Model(nnx.Module, abc.ABC):
  """Base abstract class for all dynamics models as neural networks in JAX."""

  def __init__(self, config: ModelConfig):
    super().__init__()

    self.config = config

    self.input_dim = config.input_dim
    self.output_dim = config.output_dim
    self.input_range = config.input_range
    self.output_range = config.output_range
    if self.input_range is None:
      self.input_range = tuple((-1.0, 1.0) for _ in range(self.input_dim))
      print(
        "Warning: input_range is not specified, defaulting to (-1, 1) for all input dimensions."
      )
    if self.output_range is None:
      self.output_range = tuple((-1.0, 1.0) for _ in range(self.output_dim))
      print(
        "Warning: output_range is not specified, defaulting to (-1, 1) for all output dimensions."
      )

    self.hidden_sizes = config.hidden_sizes if config.hidden_sizes is not None else []
    self.n_layers = (
      len(config.hidden_sizes) + 1 if config.hidden_sizes is not None else 1
    )
    self.obs_history_dim = config.obs_history_dim

    # for loss
    self.lambda_c = config.lambda_c
    self.lambda_d = config.lambda_d
    self.Lipschitz_ub = config.Lipschitz_ub
    self.J_Lipschitz_ub = config.J_Lipschitz_ub

    self.lambda_r = config.lambda_r
    self.shuffle_length = config.shuffle_length
    self.regressive_loss_discount = config.regressive_loss_discount

  @abc.abstractmethod
  def forward(self, x: jnp.ndarray, *args, **kwargs) -> Tuple[jnp.ndarray, ...]:
    """Computes the output of the network. build up the network architecture"""
    pass

  @abc.abstractmethod
  def save(self, path: str):
    """Saves the network parameters."""
    pass

  @abc.abstractmethod
  def load(self, path: str) -> Dict[str, Any]:
    """Loads the network parameters."""
    pass

  def __len__(self):
    return 1

  def input_normalize(self, x: jnp.ndarray) -> jnp.ndarray:
    """Normalizes the input data based on the provided input range."""
    x_norm = jnp.zeros_like(x)
    for i in range(x.shape[-1]):
      min_val, max_val = self.config.input_range[i]
      x_norm = x_norm.at[..., i].set(
        2 * (x[..., i] - min_val) / (max_val - min_val) - 1
      )
    return x_norm

  def output_normalize(self, y: jnp.ndarray) -> jnp.ndarray:
    """Normalizes the output data based on the provided output range."""
    y_norm = jnp.zeros_like(y)
    for i in range(y.shape[-1]):
      min_val, max_val = self.config.output_range[i]
      y_norm = y_norm.at[..., i].set(
        2 * (y[..., i] - min_val) / (max_val - min_val) - 1
      )
    return y_norm

  def input_denormalize(self, x: jnp.ndarray) -> jnp.ndarray:
    """Denormalizes the input data based on the provided input range."""
    x_denorm = jnp.zeros_like(x)
    for i in range(x.shape[-1]):
      min_val, max_val = self.config.input_range[i]
      x_denorm = x_denorm.at[..., i].set(
        (x[..., i] + 1) * (max_val - min_val) / 2 + min_val
      )
    return x_denorm

  def output_denormalize(self, y: jnp.ndarray) -> jnp.ndarray:
    """Denormalizes the output data based on the provided output range."""
    y_denorm = jnp.zeros_like(y)
    for i in range(y.shape[-1]):
      min_val, max_val = self.config.output_range[i]
      y_denorm = y_denorm.at[..., i].set(
        (y[..., i] + 1) * (max_val - min_val) / 2 + min_val
      )
    return y_denorm

  def input_normalize_std(self, x: jnp.ndarray) -> jnp.ndarray:
    """Z-score normalize input data using (mean, std) pairs in input_range."""
    x_norm = jnp.zeros_like(x)
    for i in range(x.shape[-1]):
      mean_val, std_val = self.config.input_range[i]
      safe_std = std_val if abs(std_val) > 1e-12 else 1.0
      x_norm = x_norm.at[..., i].set((x[..., i] - mean_val) / safe_std)
    return x_norm

  def output_normalize_std(self, y: jnp.ndarray) -> jnp.ndarray:
    """Z-score normalize output data using (mean, std) pairs in output_range."""
    y_norm = jnp.zeros_like(y)
    for i in range(y.shape[-1]):
      mean_val, std_val = self.config.output_range[i]
      safe_std = std_val if abs(std_val) > 1e-12 else 1.0
      y_norm = y_norm.at[..., i].set((y[..., i] - mean_val) / safe_std)
    return y_norm

  def input_denormalize_std(self, x: jnp.ndarray) -> jnp.ndarray:
    """Inverse z-score for input data using (mean, std) in input_range."""
    x_denorm = jnp.zeros_like(x)
    for i in range(x.shape[-1]):
      mean_val, std_val = self.config.input_range[i]
      safe_std = std_val if abs(std_val) > 1e-12 else 1.0
      x_denorm = x_denorm.at[..., i].set(x[..., i] * safe_std + mean_val)
    return x_denorm

  def output_denormalize_std(self, y: jnp.ndarray) -> jnp.ndarray:
    """Inverse z-score for output data using (mean, std) in output_range."""
    y_denorm = jnp.zeros_like(y)
    for i in range(y.shape[-1]):
      mean_val, std_val = self.config.output_range[i]
      safe_std = std_val if abs(std_val) > 1e-12 else 1.0
      y_denorm = y_denorm.at[..., i].set(y[..., i] * safe_std + mean_val)
    return y_denorm
