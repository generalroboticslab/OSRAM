import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from training.dynamics.training.base import Model, ModelConfig
from training.utils.model_utils import (
  Lipschitz_normalization,
  calc_Lipschitz_constants,
  init_fully_connected,
)


@dataclass
class Tron1ModelConfig(ModelConfig):
  """configuration class for Tron1 model and the associated neural networks"""

  pass


class Tron1Model(Model):
  """Tron1 dynamics model as SNS, mlp or Lipschitz neural network in JAX."""

  def __init__(self, config: Tron1ModelConfig):
    super().__init__(config)

    self.create_network(config)

    if config.load_path is not None:
      self.load(config.load_path)
      print(f"Model parameters loaded from {config.load_path}")
      if config.inference_mode is True:
        self.set_inference_mode(config.inference_mode)

  def create_network(self, config: Tron1ModelConfig):
    """Create the neural network based on the configuration."""
    keys = jax.random.split(
      jax.random.PRNGKey(config.key if config.key is not None else 0),
      len(config.hidden_sizes or []) + 1,
    )
    weights, biases = init_fully_connected(
      config.hidden_sizes, self.input_dim, self.output_dim, keys, dtype=jnp.float32
    )

    self.weights = nnx.List([nnx.Param(w) for w in weights])
    self.biases = nnx.List([nnx.Param(b) for b in biases])

    # initialize Lipschitz constants and the parameterization
    init_c = calc_Lipschitz_constants(weights)
    if config.type == "SNS":
      self.model_type = "SNS"
      self.theta_c = nnx.List([nnx.Param(jnp.log(c)) for c in init_c])
    elif config.type == "Lipschitz":
      self.model_type = "Lipschitz"
      self.theta_c = nnx.List([nnx.Param(jnp.log(jnp.exp(c) - 1)) for c in init_c])
    elif config.type == "mlp":
      self.model_type = "mlp"
      self.theta_c = nnx.List([nnx.Param(c) for c in init_c])
    else:
      raise ValueError(f"Unknown model type: {config.type}")

    self.inference_mode = config.inference_mode
    if config.activation == "relu":
      self.activation = jax.nn.relu
    elif config.activation == "tanh":
      self.activation = jnp.tanh
    elif config.activation == "softplus":
      self.activation = jax.nn.softplus
    elif config.activation == "mish":
      self.activation = jax.nn.mish

  def forward(self, x: jnp.ndarray):
    def _fwd(self, x, W, b, theta_c):
      if self.inference_mode or self.model_type == "mlp":
        return x @ W + b
      elif self.model_type == "Lipschitz":
        c = jax.nn.softplus(theta_c)
      elif self.model_type == "SNS":
        c = jnp.exp(theta_c)
      W = Lipschitz_normalization(W, c)
      return x @ W + b

    weights, biases = self.weights, self.biases
    theta_c = self.theta_c
    for i in range(self.n_layers):
      x = _fwd(self, x, weights[i].value, biases[i].value, theta_c[i].value)
      if i < self.n_layers - 1:
        x = self.activation(x)

    return x

  def __call__(self, x: jnp.ndarray):
    """Computes the output of the network."""
    return self.forward(x)

  def save(self, path: str):
    """Saves the model parameters."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Extract parameters
    params_dict = {}
    weights, biases = self.weights, self.biases

    # Save layer parameters
    for i in range(self.n_layers):
      params_dict[f"l{i}_w"] = np.asarray(weights[i])
      params_dict[f"l{i}_b"] = np.asarray(biases[i])

    # Save Lipschitz constants
    for i in range(self.n_layers):
      params_dict[f"l{i}_theta_c"] = np.asarray(self.theta_c[i])

    # Save model metadata
    params_dict["input_dim"] = self.input_dim
    params_dict["output_dim"] = self.output_dim
    params_dict["hidden_sizes"] = np.array(self.hidden_sizes)
    params_dict["num_layers"] = self.n_layers
    params_dict["model_type"] = self.model_type

    # Save as compressed numpy archive
    if not path.endswith(".npz"):
      path = path + ".npz"

    np.savez_compressed(path, **params_dict)

    total_params = sum(w.size + b.size for w, b in zip(weights, biases, strict=True))

    print(f"Model saved to {path}")
    print(f"Model info: {self.n_layers} layers")
    # print(f"  Architecture: {self.n_layers}")
    print(f"Total parameters: {total_params}")

  def load(self, path: str, *args, **kwargs) -> Dict[str, Any]:
    """Load parameters from checkpoint file.

    Args:
        path: Path to checkpoint file
        layer_sizes: Expected layer sizes for validation
    """
    try:
      archive = np.load(path)
    except Exception as e:
      raise RuntimeError(f"Failed to load model from {path}: {e}") from e

    # Verify model compatibility
    saved_input_dim = int(archive["input_dim"])
    saved_output_dim = int(archive["output_dim"])
    saved_n_layers = int(archive["num_layers"])
    saved_model_type = str(archive["model_type"])

    if saved_input_dim != self.input_dim:
      raise ValueError(
        f"Loaded model input_dim ({saved_input_dim}) doesn't match current model ({self.input_dim})"
      )
    if saved_output_dim != self.output_dim:
      raise ValueError(
        f"Loaded model output_dim ({saved_output_dim}) doesn't match current model ({self.output_dim})"
      )
    if saved_n_layers != self.n_layers:
      raise ValueError(
        f"Loaded model n_layers ({saved_n_layers}) doesn't match current model ({self.n_layers})"
      )

    # Load weights, biases, and theta_c
    weights = []
    biases = []
    theta_c = []

    for idx in range(self.n_layers):
      w_key = f"l{idx}_w"
      b_key = f"l{idx}_b"
      c_key = f"l{idx}_theta_c"

      if w_key not in archive or b_key not in archive or c_key not in archive:
        raise RuntimeError(f"Missing keys in checkpoint: {w_key}, {b_key}, {c_key}")

      w = jnp.asarray(archive[w_key])
      b = jnp.asarray(archive[b_key])
      c = jnp.asarray(archive[c_key])

      weights.append(w)
      biases.append(b)
      theta_c.append(c)

    # Update model parameters
    self.weights = nnx.List([nnx.Param(w) for w in weights])
    self.biases = nnx.List([nnx.Param(b) for b in biases])
    self.theta_c = nnx.List([nnx.Param(c) for c in theta_c])
    self.model_type = saved_model_type

    # Calculate total parameters
    total_params = sum(w.size + b.size for w, b in zip(weights, biases, strict=True))

    print(f"Model loaded from {path}")
    print(f"  Input dim: {self.input_dim}, Output dim: {self.output_dim}")
    print(f"  Number of layers: {self.n_layers}")
    print(f"  Hidden sizes: {self.hidden_sizes}")
    print(f"  Total parameters: {total_params}")
    print(f"  Model type: {self.model_type}")
    archive.close()

  def set_inference_mode(self, mode: bool, path: Optional[str] = None):
    """ "
    This function should be used with load model to load in a model
    """
    self.inference_mode = mode
    # breakpoint()
    if path is not None:
      self.load(path)  # set weights and biases

    # initialize Lipschitz constants and the parameterization
    if self.inference_mode and (self.model_type != "mlp"):
      if self.model_type == "SNS":
        self.c = nnx.data(jax.tree.map(lambda x: jnp.exp(x), self.theta_c))
      elif self.model_type == "Lipschitz":
        self.c = nnx.data(jax.tree.map(lambda x: jax.nn.softplus(x), self.theta_c))
      normalized_weights = [
        Lipschitz_normalization(w, ci)
        for w, ci in zip(self.weights, self.c, strict=True)
      ]
      self.weights = nnx.List([nnx.Param(w) for w in normalized_weights])
    else:
      self.c = nnx.data(calc_Lipschitz_constants(self.weights))
