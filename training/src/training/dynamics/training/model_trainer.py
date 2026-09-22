from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from flax import nnx
from torch.utils.data import DataLoader, Dataset

from training.dynamics.training.base import Model
from training.utils.model_utils import eval, update


@dataclass
class TrainerConfig:
  # optimizer
  learning_rate: float = 1e-4
  weight_decay: float = 0.0
  eps: float = 1e-8
  # data loader
  batch: int = 64
  shuffle: bool = True
  num_workers: int = 0
  drop_last: bool = False
  # trainer
  epochs: int = 1
  evaluate: bool = True
  silent: bool = False
  seed: int = 42
  # logger related
  training_name: str = "dynamics_model_training"


@dataclass
class MAMLTrainerConfig(TrainerConfig):
  task_batch: int = 5  # number of tasks sampled each time training
  inner_steps: int = 1  # for reptile, this will be used as number of SGD steps
  inner_lr: float = 1e-3


@dataclass
class ReptileTrainerConfig(TrainerConfig):
  task_batch: int = 5  # number of tasks sampled each time training
  inner_steps: int = 1  # number of SGD steps


class JAXDataset(Dataset):
  """PyTorch Dataset wrapper for JAX data."""

  def __init__(self, data: List[Tuple[jnp.ndarray, jnp.ndarray]]):
    """
    Args:
        data: List of (input, target) tuples
    """
    self.inputs = []
    self.targets = []

    for input_sample, target_sample in data:
      # Convert JAX arrays to numpy for PyTorch compatibility
      self.inputs.append(np.asarray(input_sample))
      self.targets.append(np.asarray(target_sample))

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    # Return numpy arrays (PyTorch will handle tensor conversion)
    return self.inputs[idx], self.targets[idx]

  @staticmethod
  def collate_fn(batch):
    """Custom collate function to convert PyTorch tensors back to JAX arrays."""
    inputs, targets = zip(*batch, strict=True)

    # Stack and convert to JAX arrays
    batch_inputs = jnp.array(np.stack(inputs, axis=0))
    batch_targets = jnp.array(np.stack(targets, axis=0))

    return batch_inputs, batch_targets


class JAXTrainer:
  """Trainer for JAX dynamics models.
  Support models:
      1. normal mlp
      2. Lipschitz mlp
      3. SNS mlp
  Support training methods:
      1. Standard training
      2. MAML
      3. Reptile
      4. finetuning from existing model
  """

  def __init__(self, model: Model, cfg: TrainerConfig):
    self.model = model

    # training setups
    self.cfg = cfg
    self.rng = jax.random.PRNGKey(cfg.seed)

    # optimizer
    self.optimizer = self.create_optimizer()

    # logger
    if not cfg.silent:
      self.run = wandb.init(project=cfg.training_name, config=cfg)

  def create_optimizer(self) -> optax.GradientTransformation:
    self.tx = optax.chain(
      optax.add_decayed_weights(self.cfg.weight_decay),  # Add weight decay
      optax.adam(
        learning_rate=self.cfg.learning_rate, eps=self.cfg.eps
      ),  # Adam optimizer
    )
    opt_state = nnx.Optimizer(self.model, self.tx, wrt=nnx.Param)
    return opt_state

  def create_dataloader(
    self,
    dataset: List[Tuple[jnp.ndarray, jnp.ndarray]],
    shuffle_length: Optional[int] = None,
  ) -> DataLoader:
    pytorch_dataset = dataset

    if shuffle_length is not None:
      # Chunk-based shuffling: shuffle chunks but maintain order within each chunk
      num_samples = len(pytorch_dataset)
      num_complete_chunks = num_samples // shuffle_length  # Only complete chunks

      # Truncate to complete chunks only
      # truncated_samples = num_complete_chunks * shuffle_length

      # if truncated_samples < num_samples:
      #     if not self.cfg.silent:
      #         print(f"Warning: Truncating dataset from {num_samples} to {truncated_samples} samples to fit shuffle_length={shuffle_length}")

      # Create chunk indices for complete chunks only
      chunk_indices = list(range(num_complete_chunks))
      np.random.shuffle(chunk_indices)

      # Build new sample order by concatenating shuffled chunks
      new_indices = []
      for chunk_idx in chunk_indices:
        start_idx = chunk_idx * shuffle_length
        end_idx = start_idx + shuffle_length  # Safe because we only use complete chunks
        # Add indices in original order within this chunk
        new_indices.extend(range(start_idx, end_idx))

      # Reorder the dataset with ONLY valid indices
      reordered_data = [
        (pytorch_dataset.inputs[i], pytorch_dataset.targets[i]) for i in new_indices
      ]
      pytorch_dataset = JAXDataset(reordered_data)
      should_shuffle = False  # Already shuffled at chunk level
      drop_last = True  # Ensure batches don't cross chunk boundaries
    else:
      should_shuffle = self.cfg.shuffle
      drop_last = self.cfg.drop_last

    return DataLoader(
      pytorch_dataset,
      batch_size=self.cfg.batch,
      shuffle=should_shuffle,
      num_workers=self.cfg.num_workers,
      collate_fn=pytorch_dataset.collate_fn,
      drop_last=drop_last,
      pin_memory=True,  # Set to True if using GPU
    )

  def train_standard(
    self,
    train_dataset: JAXDataset,
    val_dataset: Optional[JAXDataset] = None,
    shuffle_length: Optional[int] = None,
  ):
    # dataset
    eval_dataset = train_dataset if val_dataset is None else val_dataset

    if self.cfg.evaluate:
      init_val_score = eval(self.model, eval_dataset)
      self.run.log({"val_loss": init_val_score})

    train_loader = self.create_dataloader(train_dataset, shuffle_length)

    # check batch size and shuffle_length
    if shuffle_length is not None and self.cfg.batch % shuffle_length != 0:
      raise ValueError(
        f"batch size ({self.cfg.batch}) should be divisible by shuffle_length ({shuffle_length}) for auto-regressive training."
      )

    # training info
    epochs = range(self.cfg.epochs)
    total_samples = len(train_dataset)
    if not self.cfg.silent:
      print("Training setup:")
      print(f"  Total samples: {total_samples}")
      print(f"  Batch size: {self.cfg.batch}")
      print(f"  Batches per epoch: {len(train_loader)}")

    # training
    for epoch in epochs:
      epoch_loss = 0.0
      batch_losses = []
      samples_processed = 0

      progress_bar = tqdm.tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{self.cfg.epochs}",
        disable=self.cfg.silent,
        total=len(train_loader),
        leave=True,
      )

      for batch_inputs, batch_targets in progress_bar:
        current_batch_size = batch_inputs.shape[0]
        samples_processed += current_batch_size

        # Update model
        loss_val, metrics = update(
          self.model, batch_inputs, batch_targets, self.optimizer
        )
        batch_losses.append(loss_val)

        # Update progress bar
        progress_bar.set_postfix(
          {
            "Batch Loss": f"{loss_val.item():.6f}",
            "Samples": f"{samples_processed}/{total_samples}",
          }
        )
      # Force a newline
      if not self.cfg.silent:
        print()

      epoch_loss = jnp.mean(jnp.array(batch_losses))

      # evaluation
      if self.cfg.evaluate:
        val_score = eval(self.model, eval_dataset)

      # wandb logging
      lipschitz_residue_c = metrics.get("lipschitz_residue_c", None)
      lipschitz_residue_d = metrics.get("lipschitz_residue_d", None)
      lipschitz_constant_C = metrics.get("lipschitz_constant_C", None)
      lipschitz_constant_D = metrics.get("lipschitz_constant_D", None)
      autoregressive_loss = metrics.get("autoregressive_loss", None)
      log_dict = {
        "train_loss": epoch_loss.item(),
        "mse_loss": metrics.get("mse_per_sample", None).item(),
        "val_loss": val_score.item() if self.cfg.evaluate else None,
        "lipschitz_residue_c": lipschitz_residue_c.item()
        if lipschitz_residue_c is not None
        else None,
        "lipschitz_residue_d": lipschitz_residue_d.item()
        if lipschitz_residue_d is not None
        else None,
        "lipschitz_constant_C": lipschitz_constant_C.item()
        if lipschitz_constant_C is not None
        else None,
        "lipschitz_constant_D": lipschitz_constant_D.item()
        if lipschitz_constant_D is not None
        else None,
        "autoregressive_loss": autoregressive_loss.item()
        if autoregressive_loss is not None
        else None,
      }
      self.run.log(log_dict)

  ###
  # Meta-learning algorithms
  ###
  def sample_task_batch(
    self, datasets: List[JAXDataset], task_batch_size: int
  ) -> List[JAXDataset]:
    """Sample a batch of tasks for meta-learning."""
    indices = np.random.choice(len(datasets), size=task_batch_size, replace=False)
    return [datasets[i] for i in indices]

  def train_maml(
    self,
    train_datasets: List[JAXDataset],
    val_datasets: Optional[List[JAXDataset]] = None,
  ):
    raise NotImplementedError("MAML training not implemented yet.")

  def train_reptile(
    self,
    train_datasets: List[JAXDataset],
    val_datasets: Optional[List[JAXDataset]] = None,
    shuffle_length: Optional[int] = None,
  ):
    """
    Trains the model using Reptile approach for fast online adaptation.
    Steps:
        1. Sample task_samples number of tasks from dataset_train
        2. For each task:
            a. Copy the current model parameters
            b. Perform 'steps' number of gradient descent updates on the task data
            c. Compute the difference between the updated parameters and the original parameters
        3. Average the parameter differences across tasks
        4. Update the original model parameters in the direction of the averaged differences
        5. Repeat for the specified number of epochs
    """
    # Validate that the training can be performed
    if not hasattr(self.cfg, "task_batch"):
      raise ValueError("Reptile training requires 'task_batch' in TrainerConfig.")
    elif not hasattr(self.cfg, "inner_steps"):
      raise ValueError("Reptile training requires 'inner_steps' in TrainerConfig.")

    eval_datasets = train_datasets if val_datasets is None else val_datasets

    # check batch size and shuffle_length
    if shuffle_length is not None and self.cfg.batch % shuffle_length != 0:
      raise ValueError(
        f"batch size ({self.cfg.batch}) should be divisible by shuffle_length ({shuffle_length}) for auto-regressive training."
      )

    epoch_pbar = tqdm.tqdm(
      range(self.cfg.epochs), desc="Reptile Training", disable=self.cfg.silent
    )

    def individual_task_update(
      model, task_dataset: JAXDataset, shuffle_length: Optional[int] = None
    ) -> Model:
      """Perform inner loop updates for a single task."""
      # Extract trainable parameters
      graph, initial_state = nnx.split(model, nnx.Param)
      adapted_model = nnx.merge(graph, initial_state)

      # Create dataloader for the task
      task_loader = self.create_dataloader(task_dataset, shuffle_length)

      # Create inner optimizer
      inner_optimizer = nnx.Optimizer(adapted_model, self.tx, wrt=nnx.Param)

      inner_pbar = tqdm.tqdm(
        range(self.cfg.inner_steps),
        desc="Task Specific Steps",
        leave=False,
        disable=self.cfg.silent,
      )

      support_iter = iter(task_loader)
      for _step in inner_pbar:
        try:
          batch_inputs, batch_targets = next(support_iter)
        except StopIteration:
          support_iter = iter(task_loader)
          batch_inputs, batch_targets = next(support_iter)
        loss_val, metrics = update(
          adapted_model, batch_inputs, batch_targets, inner_optimizer
        )
        inner_pbar.set_postfix({"loss": f"{float(loss_val):.4f}"})

      _, updated_state = nnx.split(adapted_model, nnx.Param)
      return updated_state, metrics

    for epoch in epoch_pbar:
      # sample task batch datasets
      task_batch = self.sample_task_batch(train_datasets, self.cfg.task_batch)
      # store updated parameters for each task
      updated_params_list = []
      metrics_avg = {}
      for task_dataset in task_batch:
        updated_state, metrics = individual_task_update(
          self.model, task_dataset, shuffle_length
        )
        updated_params_list.append(updated_state)
        for k, v in metrics.items():
          metrics_avg[k] = metrics_avg.get(k, 0.0) + v
      metrics_avg = {k: v / len(task_batch) for k, v in metrics_avg.items()}

      graph, base_state = nnx.split(self.model, nnx.Param)
      n_tasks = len(updated_params_list)
      param_diff = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), base_state)
      for updated_params in updated_params_list:
        delta = jax.tree_util.tree_map(lambda a, b: a - b, updated_params, base_state)
        param_diff = jax.tree_util.tree_map(lambda acc, d: acc + d, param_diff, delta)
      avg_diff = jax.tree_util.tree_map(
        lambda x, n_tasks=n_tasks: x / float(n_tasks), param_diff
      )
      scaled_update = jax.tree_util.tree_map(
        lambda d: self.cfg.learning_rate * d, avg_diff
      )
      # update model param
      new_state = jax.tree_util.tree_map(lambda p, u: p + u, base_state, scaled_update)
      self.model = nnx.merge(graph, new_state)

      if self.cfg.evaluate:
        # evaluation on validation tasks
        val_task_batch = self.sample_task_batch(
          eval_datasets, max(1, self.cfg.task_batch // 5)
        )  # use same amount of task for validation
        val_losses = []
        for val_task_dataset in val_task_batch:
          updated_params, _ = individual_task_update(
            self.model, val_task_dataset, shuffle_length
          )
          updated_model = nnx.merge(graph, updated_params)
          val_loss = eval(updated_model, val_task_dataset)
          val_losses.append(val_loss)

      # wandb logging
      lipschitz_residue_c = metrics_avg.get("lipschitz_residue_c", None)
      lipschitz_residue_d = metrics_avg.get("lipschitz_residue_d", None)
      lipschitz_constant_C = metrics_avg.get("lipschitz_constant_C", None)
      lipschitz_constant_D = metrics_avg.get("lipschitz_constant_D", None)
      autoregressive_loss = metrics_avg.get("autoregressive_loss", None)
      log_dict = {
        "mse_loss": metrics_avg.get("mse_per_sample", None).item(),
        "lipschitz_residue_c": lipschitz_residue_c.item()
        if lipschitz_residue_c is not None
        else None,
        "lipschitz_residue_d": lipschitz_residue_d.item()
        if lipschitz_residue_d is not None
        else None,
        "lipschitz_constant_C": lipschitz_constant_C.item()
        if lipschitz_constant_C is not None
        else None,
        "lipschitz_constant_D": lipschitz_constant_D.item()
        if lipschitz_constant_D is not None
        else None,
        "autoregressive_loss": autoregressive_loss.item()
        if autoregressive_loss is not None
        else None,
      }
      self.run.log(log_dict)
      # log model
      if (epoch + 1) % 5 == 0:
        model_path = f"{self.run.dir}/model_epoch_{epoch + 1}.npz"
        self.model.save(model_path)
        self.run.save(model_path)  # Save to wandb
        if not self.cfg.silent:
          print(f"Model saved to wandb: {model_path}")

  def finetune(
    self,
    finetune_dataset: JAXDataset,
    shuffle_length: Optional[int] = None,
  ):
    """
    Function for online finetuning of the trained reptile model.
    Time sensitive so no logging and evaluation inside.
    """
    # Create dataloader with proper batching
    finetune_loader = self.create_dataloader(finetune_dataset, shuffle_length)

    batch_inputs, batch_targets = next(iter(finetune_loader))
    loss_val, _ = update(self.model, batch_inputs, batch_targets, self.optimizer)

    # if self.cfg.silent is False:
    print(f"Finetune Batch Loss: {loss_val.item():.6f}", flush=True)

    # inputs = jnp.array([finetune_dataset.inputs[i] for i in range(len(finetune_dataset))])
    # targets = jnp.array([finetune_dataset.targets[i] for i in range(len(finetune_dataset))])
    # loss_val, _ = update(self.model, inputs, targets, self.optimizer)

    # if self.cfg.silent is False:
    #     print(f"Finetune Loss: {loss_val.item():.6f}")
