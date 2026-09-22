from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from training.dynamics.training.base import Model


###
# Initialization
###
def init_params_xavier(input_dim, output_dim, key, dtype=jnp.float32):
  limit = jnp.sqrt(6.0 / (input_dim + output_dim))
  w = jax.random.uniform(
    key, (input_dim, output_dim), minval=-limit, maxval=limit, dtype=dtype
  )
  return w


def init_params_he(input_dim, output_dim, key, dtype=jnp.float32):
  std = jnp.sqrt(2.0 / input_dim)
  w = jax.random.normal(key, (input_dim, output_dim), dtype=dtype) * std
  return w


def init_bias(output_size, dtype=jnp.float32):
  """Initialize a single bias vector of length output_size."""
  return jnp.zeros(output_size, dtype=dtype)


def init_fully_connected(hidden_dims, in_dim, out_dim, key_seq, dtype=jnp.float32):
  weights, biases = [], []
  prev_dim = in_dim
  for h_dim, k in zip(hidden_dims, key_seq[:-1], strict=True):
    w = init_params_he(prev_dim, h_dim, k, dtype=dtype)
    b = init_bias(h_dim, dtype=dtype)
    weights.append(w)
    biases.append(b)
    prev_dim = h_dim
  w_final = init_params_xavier(prev_dim, out_dim, key_seq[-2], dtype=dtype)
  b_final = init_bias(out_dim, dtype=dtype)
  weights.append(w_final)
  biases.append(b_final)
  return weights, biases


###
# Lipschitz constant related
###


def calc_Lipschitz_constants(weights):
  """Calculate the Lipschitz constant of each layer of the network given its weights."""
  return [infinity_norm(w) for w in weights]


def calc_Lipschitz_constant_network(params, type: str):
  """Different network uses different parameterization for the Lipschitz constant."""
  if type == "mlp":
    c = params
  elif type == "Lipschitz":
    c = jax.tree.map(lambda x: jax.nn.softplus(x), params)
  elif type == "SNS":
    c = jax.tree.map(lambda x: jnp.exp(x), params)

  ## calculate total Lipschitz constant and its jacobian
  cs, _ = jax.tree_util.tree_flatten(c)
  cs = jnp.array(cs)
  C = jnp.prod(cs)
  prefix = jnp.cumprod(cs)
  S = jnp.sum(prefix)
  return C, C * S


def infinity_norm(W):
  """Compute the infinity norm of a kernel matrix."""
  return jnp.max(jnp.sum(jnp.abs(W), axis=1))


def Lipschitz_normalization(W, c):
  """Normalize the weight matrix W to have Lipschitz constant at most c."""
  row_sums = jnp.sum(jnp.abs(W), axis=1)
  scale = jnp.minimum(1.0, c / row_sums)
  return W * scale[:, None]


###
# Training related
###
def loss(
  model: Model,
  model_in: jnp.ndarray,
  target: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
  def calc_lipschitz_constant_residue(model):
    # Flatten and prepare Lipschitz constants
    if model.model_type == "standard":
      c = calc_Lipschitz_constants(model.weights)
    elif model.model_type == "Lipschitz":
      c = jax.tree.map(lambda x: jax.nn.softplus(x), model.theta_c)
    else:
      c = jax.tree.map(lambda x: jnp.exp(x), model.theta_c)
    c, _ = jax.tree_util.tree_flatten(c)
    c = jax.tree.map(lambda x: x.reshape(1, -1), c)
    c = jnp.concatenate(c, axis=-1).flatten()

    C_target = model.Lipschitz_ub
    D_target = model.J_Lipschitz_ub
    C = jnp.prod(c)  # lipschitz_ub_model

    prefix = jnp.cumprod(c)
    S = jnp.sum(prefix)

    if C_target < jnp.inf:
      residual_c = C / C_target
      residual_c = jnp.maximum(1.0, residual_c)
    else:
      residual_c = 0.0

    if D_target < jnp.inf:
      residual_d = (C * S) / D_target
      residual_d = jnp.maximum(1.0, residual_d)
    else:
      residual_d = 0.0

    return residual_c, residual_d

  if target is None:
    raise ValueError("target must be provided for loss computation")
  x = jnp.asarray(model_in)
  y = jnp.asarray(target)
  if x.ndim != 1:
    raise ValueError(
      f"forward expects a single example of shape (input_dim,), got shape {x.shape}"
    )
  elif y.ndim != 1:
    raise ValueError(
      f"forward expects a single example of shape (output_dim,), got shape {y.shape}"
    )
  pred = model(x)
  mse_per_sample = jnp.mean((pred - y) ** 2)
  res_c, res_d = calc_lipschitz_constant_residue(model)

  reg_loss = model.lambda_c * res_c + model.lambda_d * res_d

  C, D = calc_Lipschitz_constant_network(model.theta_c, model.model_type)

  loss = mse_per_sample + reg_loss

  metrics = {
    "mse_per_sample": mse_per_sample,
    "lipschitz_residue_c": res_c,
    "lipschitz_residue_d": res_d,
    "lipschitz_constant_C": C,
    "lipschitz_constant_D": D,
  }
  return loss, metrics


@nnx.jit
def update(
  model: Model,
  model_in: jnp.ndarray,
  target: jnp.ndarray,
  optimizer: optax.GradientTransformation,
) -> Tuple[Any, Optional[optax.OptState], float, Dict[str, Any]]:
  """Update parameters using optax optimizer on a batch (with JIT).

  - model_in: (B, input_dim)
  - target: (B, output_dim)
  """
  if target is None:
    raise ValueError("target must be provided for update")
  if optimizer is None:
    raise ValueError("optimizer must be provided for update")

  x = jnp.asarray(model_in)
  y = jnp.asarray(target)

  if x.ndim != 2 or y.ndim != 2:
    raise ValueError(
      "update expects batched input: model_in (B, input_dim), target (B, output_dim)"
    )

  def loss_fn(model, x_batch, y_batch):
    losses, metrics = jax.vmap(lambda x, y: loss(model, x, y))(x_batch, y_batch)
    mean_loss = jnp.mean(losses)
    mean_metrics = {k: jnp.mean(v) for k, v in metrics.items()}
    return mean_loss, mean_metrics

  def loss_fn_autoregressive(model, x_batch, y_batch):
    # First compute standard loss
    losses, metrics = jax.vmap(lambda x, y: loss(model, x, y))(x_batch, y_batch)
    mean_loss = jnp.mean(losses)
    mean_metrics = {k: jnp.mean(v) for k, v in metrics.items()}

    # Add autoregressive component
    batch_size = x_batch.shape[0]
    shuffle_length = model.shuffle_length
    num_sequences = batch_size // shuffle_length
    obs_dim = y_batch.shape[1]
    # input_dim = x_batch.shape[1]  # [obs_{t-h}, ..., obs_t, act_{t-h}, ..., act_t]
    obs_history_dim = model.obs_history_dim  # dimension of observation inputs

    x_sequences = x_batch.reshape(num_sequences, shuffle_length, -1)
    y_sequences = y_batch.reshape(num_sequences, shuffle_length, -1)

    def sequence_loss(x_seq, y_seq):
      """Compute autoregressive loss for one sequence.

      Args:
          x_seq: (shuffle_length, input_dim)
          y_seq: (shuffle_length, obs_dim)
      """

      def step_fn(x_current, t):
        """One autoregressive prediction step."""
        y_target = y_seq[t]
        y_pred = model(x_current)
        step_mse = jnp.mean((y_pred - y_target) ** 2)

        # Apply discount: later predictions get lower weight
        discount = model.regressive_loss_discount**t
        discounted_mse = step_mse * discount

        # Prepare next input: shift history and append prediction
        # x_current: [obs_{t-h}, ..., obs_t, act_{t-h}, ..., act_t]
        obs_history = x_current[:obs_history_dim]
        next_action_context = x_seq[t + 1, obs_history_dim:]

        new_obs_history = jnp.concatenate([obs_history[obs_dim:], y_pred])
        x_next = jnp.concatenate([new_obs_history, next_action_context])

        return x_next, discounted_mse

      # Run autoregressive rollout
      _, step_losses = jax.lax.scan(step_fn, x_seq[0], jnp.arange(shuffle_length - 1))
      return jnp.mean(step_losses)

    # vmap over sequences
    seq_losses = jax.vmap(sequence_loss)(x_sequences, y_sequences)
    autoregressive_loss = jnp.mean(seq_losses)

    # Combine losses
    total_loss = mean_loss + model.lambda_r * autoregressive_loss
    mean_metrics["autoregressive_loss"] = autoregressive_loss

    return total_loss, mean_metrics

  if model.shuffle_length is not None:
    chosen_loss_fn = loss_fn_autoregressive
  else:
    chosen_loss_fn = loss_fn

  grad_fn = nnx.value_and_grad(chosen_loss_fn, has_aux=True)
  (loss_val, metrics), grads = grad_fn(model, x, y)
  optimizer.update(model, grads)

  return loss_val, metrics


def eval(model: Model, dataset) -> jnp.ndarray:
  """Evaluate model on a dataset and return mean loss.

  Args:
      dataset: JAXDataset containing (input, target) pairs

  Returns:
      mean_mse: Mean squared error over the entire dataset
  """
  # Evaluate on entire dataset
  inputs = jnp.array([dataset.inputs[i] for i in range(len(dataset))])
  targets = jnp.array([dataset.targets[i] for i in range(len(dataset))])

  # Vectorized prediction over all samples
  preds = jax.vmap(model)(inputs)

  # Compute MSE
  mean_mse = jnp.mean((preds - targets) ** 2)
  return mean_mse
