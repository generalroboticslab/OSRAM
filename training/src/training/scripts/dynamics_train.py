"""Script to train dynamics models"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import jax.numpy as jnp
import tyro
import yaml


@dataclass
class TrainerConfig:
  data_path: str | None = None
  config_file: Optional[str] = None
  # forward: for direct supervised learning
  # reptile: for meta-learning with Reptile
  # finetuning: for finetuining pre-trained MAML model
  task: Literal["forward", "reptile", "finetune"] = "reptile"
  model_type: Literal["tron1"] = "tron1"
  seed: int = 42
  # optimizer
  lr: float = 1e-3
  weight_decay: Optional[float] = 0.0
  eps: float = 1e-8
  batch_size: int = 64
  # training
  num_epochs: int = 1000
  evaluate: bool = True
  silent: bool = False
  # dataset
  data_size: int = 10000
  shuffle: bool = True
  drop_last: bool = False
  train_size: float = 0.8  # number of training samples
  val_size: float = 0.2  # number of validation samples
  # logging
  model_log_path: Optional[str] = None
  training_name: Optional[str] = "training"

  # model related
  input_dim: int = 1
  output_dim: int = 1
  hidden_dims: tuple[int, ...] = (128, 128)
  input_range: float = 1.0
  output_range: float = 1.0
  activation: str = "relu"
  type: str = "mlp"
  obs_history_dim: int = 30
  Lipschitz_ub: jnp.ndarray = jnp.inf
  J_Lipschitz_ub: jnp.ndarray = jnp.inf
  inference_mode: bool = False
  lambda_c: float = 0.0
  lambda_d: float = 0.0
  shuffle_length: Optional[int] = None  # for auto-regressive training
  lambda_r: float = 0.0  # weight for auto-regressive loss
  regressive_loss_discount: float = 1.0  # default not discount

  # reptile
  num_tasks: int = 10
  inner_steps: int = 5

  # finetuning
  model_load_path: Optional[str] = None
  finetune_steps: int = 10

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
    # check required fields
    if self.data_path is None:
      raise ValueError("data_path must be specified!")
    if not Path(self.data_path).exists():
      raise FileNotFoundError(f"Dataset path does not exist: {self.data_path}")
    if self.task not in ["forward", "reptile", "finetune"]:
      raise ValueError("task must be one of ['forward', 'reptile', 'finetune']")
    if self.model_type != "tron1":
      raise ValueError("model_type must be 'tron1'")
    if self.task == "finetune" and self.model_load_path is None:
      raise ValueError("model_load_path must be specified for finetuning task!")
    if self.model_load_path is not None and not Path(self.model_load_path).exists():
      raise FileNotFoundError(
        f"Pretrained model path does not exist: {self.model_load_path}"
      )
    if self.train_size + self.val_size != 1.0:
      raise ValueError("train_size and val_size must sum to 1.0")
    if self.inference_mode is not False:
      print("Warning: inference_mode is set to True during training!")


def train_forward(model, args: TrainerConfig):
  from training.dynamics.training.model_trainer import (
    JAXDataset,
    JAXTrainer,
    TrainerConfig,
  )
  from training.utils.data_utils import load_dataset

  def _truncate_dataset(dataset, limit: int):
    if limit is None or limit <= 0:
      return dataset
    if len(dataset) <= limit:
      return dataset
    print(
      f"Truncating dataset from {len(dataset)} to {limit} samples (data_size={limit})."
    )
    return dataset[:limit]

  # load dataset, assumed all the pre-processes are down
  dataset = load_dataset(args.data_path)
  dataset = _truncate_dataset(dataset, args.data_size)
  train_length = int(len(dataset) * args.train_size)
  train_data = dataset[:train_length]
  val_data = dataset[train_length:]

  train_dataset = JAXDataset(train_data)
  val_dataset = JAXDataset(val_data)

  training_cfg = TrainerConfig(
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    eps=args.eps,
    # data
    batch=args.batch_size,
    shuffle=args.shuffle,
    num_workers=0,
    drop_last=args.drop_last,
    # training
    epochs=args.num_epochs,
    evaluate=args.evaluate,
    silent=args.silent,
    seed=args.seed,
    training_name=args.training_name,
  )

  trainer = JAXTrainer(model, training_cfg)

  trainer.train_standard(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    shuffle_length=args.shuffle_length,
  )

  return trainer.model


def train_reptile(model, args: TrainerConfig):
  import glob

  from training.dynamics.training.model_trainer import (
    JAXDataset,
    JAXTrainer,
    ReptileTrainerConfig,
  )
  from training.utils.data_utils import load_dataset

  def _truncate_dataset(dataset, limit: int):
    if limit is None or limit <= 0:
      return dataset
    if len(dataset) <= limit:
      return dataset
    return dataset[:limit]

  dataset_files = glob.glob(
    args.data_path + "/*.npz"
  )  # or whatever extension your datasets have
  if not dataset_files:
    raise ValueError(f"No dataset files found in {args.data_path}")

  print(f"Found {len(dataset_files)} dataset files for Reptile training")
  datasets = []
  for file_path in dataset_files:
    dataset = load_dataset(file_path)
    original_len = len(dataset)
    dataset = _truncate_dataset(dataset, args.data_size)
    dataset = JAXDataset(dataset)
    datasets.append(dataset)
    if original_len != len(dataset):
      print(
        f"Loaded dataset: {file_path} with {len(dataset)} samples "
        f"(truncated from {original_len}, data_size={args.data_size})"
      )
    else:
      print(f"Loaded dataset: {file_path} with {len(dataset)} samples")

  num_train_tasks = int(args.train_size * len(datasets))
  train_datasets = datasets[:num_train_tasks]
  val_datasets = datasets[num_train_tasks:]

  training_cfg = ReptileTrainerConfig(
    # optimizer
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    eps=1e-8,
    # dataset
    shuffle=args.shuffle,
    num_workers=0,
    drop_last=args.drop_last,
    # training
    batch=args.batch_size,
    epochs=args.num_epochs,
    evaluate=args.evaluate,
    silent=args.silent,
    seed=args.seed,
    training_name=args.training_name,
    # reptile
    task_batch=args.num_tasks,
    inner_steps=args.inner_steps,
  )

  trainer = JAXTrainer(model, training_cfg)

  trainer.train_reptile(
    train_datasets=train_datasets,
    val_datasets=val_datasets,
    shuffle_length=args.shuffle_length,
  )

  return trainer.model


def finetune(model, args: TrainerConfig):
  from training.dynamics.training.model_trainer import (
    JAXDataset,
    JAXTrainer,
    ReptileTrainerConfig,
  )
  from training.utils.data_utils import load_dataset

  def _truncate_dataset(dataset, limit: int):
    if limit is None or limit <= 0:
      return dataset
    if len(dataset) <= limit:
      return dataset
    print(
      f"Truncating finetune dataset from {len(dataset)} to {limit} samples (data_size={limit})."
    )
    return dataset[:limit]

  # load dataset
  dataset = load_dataset(args.data_path)
  dataset = _truncate_dataset(dataset, args.data_size)
  dataset = JAXDataset(dataset)

  training_cfg = ReptileTrainerConfig(
    # optimizer
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    eps=1e-8,
    # dataset
    shuffle=args.shuffle,
    num_workers=0,
    drop_last=args.drop_last,
    # training
    batch=args.batch_size,
    epochs=args.num_epochs,
    evaluate=args.evaluate,
    silent=args.silent,
    seed=args.seed,
    training_name=args.training_name,
    # reptile
    task_batch=args.num_tasks,
    inner_steps=args.inner_steps,
  )
  trainer = JAXTrainer(model, training_cfg)

  # Finetune on random batches from dataset
  for _step in range(args.finetune_steps):
    trainer.finetune(dataset, shuffle_length=args.shuffle_length)

  print("Finetuning completed.")

  return trainer.model


def _build_tron1_model(args: TrainerConfig):
  from training.dynamics.training.tron1.tron1_model import (
    Tron1Model,
    Tron1ModelConfig,
  )

  cfg = Tron1ModelConfig(
    input_dim=args.input_dim,
    output_dim=args.output_dim,
    hidden_sizes=args.hidden_dims,
    activation=args.activation,
    key=args.seed,
    type=args.type,
    obs_history_dim=args.obs_history_dim,
    Lipschitz_ub=args.Lipschitz_ub,
    J_Lipschitz_ub=args.J_Lipschitz_ub,
    inference_mode=args.inference_mode,
    lambda_c=args.lambda_c,
    lambda_d=args.lambda_d,
    shuffle_length=args.shuffle_length,
    lambda_r=args.lambda_r,
    regressive_loss_discount=args.regressive_loss_discount,
    input_range=tuple(
      (-args.input_range, args.input_range)
      for _ in range(args.input_dim + args.output_dim)
    ),
    output_range=tuple(
      (-args.output_range, args.output_range) for _ in range(args.output_dim)
    ),
  )
  return Tron1Model(cfg)


def main():
  args = tyro.cli(TrainerConfig)
  model = _build_tron1_model(args)

  if args.task == "forward":
    model = train_forward(model, args)
  elif args.task == "reptile":
    model = train_reptile(model, args)
  elif args.task == "finetune":
    model.load(args.model_load_path)
    model.set_inference_mode(False)
    model = finetune(model, args)

  if args.model_log_path is not None:
    model.save(args.model_log_path)


if __name__ == "__main__":
  main()
