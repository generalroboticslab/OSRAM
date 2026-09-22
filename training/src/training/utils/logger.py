import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import wandb


@dataclass
class BaseLoggerConfig:
  """Base configuration for all logging."""

  # General settings
  log_frequency: int = 1  # Log every N steps/epochs
  save_artifacts: bool = False


@dataclass
class WandbConfig(BaseLoggerConfig):
  """Configuration for Weights & Biases logging."""

  enabled: bool = False
  project: str = "dynamics-training"
  entity: Optional[str] = None
  run_name: Optional[str] = None
  tags: Optional[List[str]] = None
  notes: Optional[str] = None
  watch_model: bool = False  # Whether to watch model gradients/parameters
  log_frequency: int = 100  # How often to log gradients (if watch_model=True)


@dataclass
class ConsoleLoggingConfig(BaseLoggerConfig):
  """Configuration for console logging."""

  enabled: bool = True
  log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
  show_progress_bars: bool = True
  colorize_output: bool = True
  max_line_length: int = 120  # Maximum length of logged lines


@dataclass
class FileLoggingConfig(BaseLoggerConfig):
  """Configuration for JSON file logging."""

  enabled: bool = True
  log_dir: str = "logs"
  log_filename: str = "training.json"
  create_timestamp_dir: bool = True  # Create subdirectory with timestamp


@dataclass
class TensorBoardConfig(BaseLoggerConfig):
  """Configuration for TensorBoard logging."""

  enabled: bool = False
  log_dir: str = "tensorboard_logs"
  flush_secs: int = 30  # How often to flush data to disk
  max_queue: int = 10  # Maximum queue size for pending logs


@dataclass
class LoggerConfig:
  """Main configuration class that combines all logging backends."""

  wandb: Optional[WandbConfig] = None
  console: Optional[ConsoleLoggingConfig] = None
  file: Optional[FileLoggingConfig] = None
  tensorboard: Optional[TensorBoardConfig] = None


class LoggingBackend(ABC):
  """Abstract base class for logging backends."""

  @abstractmethod
  def initialize(self, config: BaseLoggerConfig, run_config: Dict[str, Any]) -> None:
    """Initialize the logging backend."""
    pass

  @abstractmethod
  def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics."""
    pass

  @abstractmethod
  def log_artifact(
    self, filepath: str, name: str, artifact_type: str = "model"
  ) -> None:
    """Log an artifact (model, plot, etc.)."""
    pass

  @abstractmethod
  def finish(self) -> None:
    """Finish logging."""
    pass


class WandbBackend(LoggingBackend):
  """Weights & Biases logging backend."""

  def __init__(self):
    self.initialized = False
    self.config = None

  def initialize(self, config: WandbConfig, run_config: Dict[str, Any]) -> None:
    """Initialize wandb."""
    if not config.enabled:
      return

    self.config = config
    wandb.init(
      project=config.project,
      entity=config.entity,
      name=config.run_name,
      tags=config.tags,
      notes=config.notes,
      config=run_config,
    )
    self.initialized = True

  def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to wandb."""
    if not self.initialized:
      return

    log_dict = {}
    for key, value in metrics.items():
      if isinstance(value, (np.ndarray, list)):
        if len(value) == 1:
          log_dict[key] = (
            float(value[0]) if hasattr(value[0], "__float__") else value[0]
          )
        else:
          log_dict[key] = value  # wandb can handle arrays
      else:
        log_dict[key] = float(value) if hasattr(value, "__float__") else value

    if step is not None:
      log_dict["step"] = step

    wandb.log(log_dict, step=step)

  def log_artifact(
    self, filepath: str, name: str, artifact_type: str = "model"
  ) -> None:
    """Log artifact to wandb."""
    if not self.initialized:
      return

    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(filepath)
    wandb.log_artifact(artifact)

  def finish(self) -> None:
    """Finish wandb run."""
    if self.initialized:
      wandb.finish()
      self.initialized = False


class FileBackend(LoggingBackend):
  """JSON file logging backend."""

  def __init__(self):
    self.log_data = []
    self.config = None
    self.filepath = None

  def initialize(self, config: FileLoggingConfig, run_config: Dict[str, Any]) -> None:
    """Initialize file logging."""
    if not config.enabled:
      return

    self.config = config

    # Create log directory
    if config.create_timestamp_dir:
      timestamp = time.strftime("%Y%m%d_%H%M%S")
      log_dir = os.path.join(config.log_dir, timestamp)
    else:
      log_dir = config.log_dir

    os.makedirs(log_dir, exist_ok=True)

    # Create unique filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = config.log_filename.replace(".json", "")
    self.filepath = os.path.join(log_dir, f"{base_name}_{timestamp}.json")

    # Save initial config
    initial_log = {"timestamp": timestamp, "config": run_config, "metrics": []}

    with open(self.filepath, "w") as f:
      json.dump(initial_log, f, indent=2)

  def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to file."""
    if not self.config or not self.config.enabled:
      return

    # Convert metrics to JSON-serializable format
    serializable_metrics = {}
    for key, value in metrics.items():
      if isinstance(value, np.ndarray):
        serializable_metrics[key] = value.tolist()
      elif hasattr(value, "__float__"):
        serializable_metrics[key] = float(value)
      else:
        serializable_metrics[key] = value

    log_entry = {
      "timestamp": time.time(),
      "step": step,
      "metrics": serializable_metrics,
    }

    # Append to existing file
    if os.path.exists(self.filepath):
      with open(self.filepath, "r") as f:
        data = json.load(f)
      data["metrics"].append(log_entry)
      with open(self.filepath, "w") as f:
        json.dump(data, f, indent=2)

  def log_artifact(
    self, filepath: str, name: str, artifact_type: str = "model"
  ) -> None:
    """Copy artifact to log directory."""
    if not self.config or not self.config.enabled:
      return

    import shutil

    artifact_dir = os.path.dirname(self.filepath)
    artifact_subdir = os.path.join(artifact_dir, "artifacts")
    os.makedirs(artifact_subdir, exist_ok=True)

    dest_path = os.path.join(artifact_subdir, f"{name}_{artifact_type}")
    shutil.copy2(filepath, dest_path)

  def finish(self) -> None:
    """Finish file logging."""
    pass


class TensorBoardBackend(LoggingBackend):
  """TensorBoard logging backend."""

  def __init__(self):
    self.writer = None
    self.config = None

  def initialize(self, config: TensorBoardConfig, run_config: Dict[str, Any]) -> None:
    """Initialize TensorBoard logging."""
    if not config.enabled:
      return

    try:
      from torch.utils.tensorboard import SummaryWriter

      self.config = config

      # Create timestamped log directory
      timestamp = time.strftime("%Y%m%d_%H%M%S")
      log_dir = os.path.join(config.log_dir, timestamp)

      self.writer = SummaryWriter(
        log_dir=log_dir, flush_secs=config.flush_secs, max_queue=config.max_queue
      )

      # Log run configuration as text
      config_text = "\n".join([f"{k}: {v}" for k, v in run_config.items()])
      self.writer.add_text("config", config_text, 0)

    except ImportError:
      print(
        "Warning: torch.utils.tensorboard not available, TensorBoard logging disabled"
      )
      self.config = None

  def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to TensorBoard."""
    if not self.writer or not self.config:
      return

    if step is None:
      step = 0

    for key, value in metrics.items():
      if isinstance(value, (int, float, np.number)):
        self.writer.add_scalar(key, float(value), step)
      elif isinstance(value, np.ndarray):
        if value.ndim == 0:  # Scalar array
          self.writer.add_scalar(key, float(value), step)
        elif value.ndim == 1:  # 1D array - can be logged as histogram
          self.writer.add_histogram(key, value, step)
        elif value.ndim == 2:  # 2D array - can be logged as image
          self.writer.add_image(key, value, step)
      elif isinstance(value, list) and len(value) > 0:
        if isinstance(value[0], (int, float)):
          # Log as histogram
          self.writer.add_histogram(key, np.array(value), step)

  def log_artifact(
    self, filepath: str, name: str, artifact_type: str = "model"
  ) -> None:
    """Log artifact info to TensorBoard."""
    if not self.writer or not self.config:
      return

    # TensorBoard doesn't directly support arbitrary file artifacts
    # But we can log the filepath as text
    self.writer.add_text(f"artifacts/{artifact_type}", f"{name}: {filepath}")

  def finish(self) -> None:
    """Finish TensorBoard logging."""
    if self.writer:
      self.writer.close()
      self.writer = None


class ConsoleBackend(LoggingBackend):
  """Console logging backend."""

  def __init__(self):
    self.config = None

  def initialize(
    self, config: ConsoleLoggingConfig, run_config: Dict[str, Any]
  ) -> None:
    """Initialize console logging."""
    if not config.enabled:
      return

    self.config = config

    # Add colors if enabled
    if config.colorize_output:
      self.colors = {
        "INFO": "\033[92m",  # Green
        "DEBUG": "\033[94m",  # Blue
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "ENDC": "\033[0m",  # End color
      }
    else:
      self.colors = {k: "" for k in ["INFO", "DEBUG", "WARNING", "ERROR", "ENDC"]}

    print(
      f"{self.colors['INFO']}🚀 Starting training with config:{self.colors['ENDC']}"
    )
    for key, value in run_config.items():
      print(f"  {key}: {value}")

  def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to console."""
    if not self.config or not self.config.enabled:
      return

    # Format metrics for display
    formatted_metrics = []
    for key, value in metrics.items():
      if isinstance(value, float):
        formatted_metrics.append(f"{key}={value:.6f}")
      elif isinstance(value, int):
        formatted_metrics.append(f"{key}={value}")
      else:
        formatted_metrics.append(f"{key}={value}")

    # Join metrics and respect max line length
    metrics_str = ", ".join(formatted_metrics)
    if len(metrics_str) > self.config.max_line_length:
      # Split into multiple lines
      lines = []
      current_line = ""
      for metric in formatted_metrics:
        if len(current_line + metric) > self.config.max_line_length:
          lines.append(current_line.rstrip(", "))
          current_line = f"  {metric}, "
        else:
          current_line += f"{metric}, "
      if current_line:
        lines.append(current_line.rstrip(", "))
      metrics_str = "\n".join(lines)

    step_str = f"Step {step}: " if step is not None else ""
    print(f"{self.colors['INFO']}📊 {step_str}{metrics_str}{self.colors['ENDC']}")

  def log_artifact(
    self, filepath: str, name: str, artifact_type: str = "model"
  ) -> None:
    """Log artifact info to console."""
    if not self.config or not self.config.enabled:
      return
    print(
      f"{self.colors['INFO']}💾 Saved {artifact_type}: {name} -> {filepath}{self.colors['ENDC']}"
    )

  def finish(self) -> None:
    """Finish console logging."""
    if self.config and self.config.enabled:
      print(f"{self.colors['INFO']}✅ Training completed!{self.colors['ENDC']}")


class UniversalLogger:
  """Universal logger that can handle multiple backends."""

  def __init__(self, config: LoggerConfig):
    self.config = config
    self.backends: List[LoggingBackend] = []
    self.backend_configs: List[BaseLoggerConfig] = []
    self.step_counter = 0

    # Only initialize backends that are provided and enabled
    if config.wandb is not None and config.wandb.enabled:
      self.backends.append(WandbBackend())
      self.backend_configs.append(config.wandb)

    if config.console is not None and config.console.enabled:
      self.backends.append(ConsoleBackend())
      self.backend_configs.append(config.console)

    if config.file is not None and config.file.enabled:
      self.backends.append(FileBackend())
      self.backend_configs.append(config.file)

    if config.tensorboard is not None and config.tensorboard.enabled:
      self.backends.append(TensorBoardBackend())
      self.backend_configs.append(config.tensorboard)

  def initialize(self, run_config: Dict[str, Any]) -> None:
    """Initialize all backends."""
    for backend, backend_config in zip(
      self.backends, self.backend_configs, strict=True
    ):
      try:
        backend.initialize(backend_config, run_config)
      except Exception as e:
        print(f"Warning: Failed to initialize {type(backend).__name__}: {e}")

  def log(
    self, metrics: Dict[str, Any], step: Optional[int] = None, force_log: bool = False
  ) -> None:
    """Log metrics to all backends."""
    # Use internal step counter if no step provided
    if step is None:
      step = self.step_counter

    # Check log frequency (use minimum frequency from all configs)
    if not force_log and self.backend_configs:
      min_frequency = min(config.log_frequency for config in self.backend_configs)
      if step % min_frequency != 0:
        return

    # Log to all backends
    for backend in self.backends:
      try:
        backend.log_metrics(metrics, step)
      except Exception as e:
        print(f"Warning: Logging backend failed: {e}")

    self.step_counter += 1

  def log_epoch(self, epoch: int, **metrics) -> None:
    """Convenience method for epoch logging."""
    epoch_metrics = {"epoch": epoch, **metrics}
    self.log(epoch_metrics, step=epoch, force_log=True)

  def log_batch(self, batch_idx: int, **metrics) -> None:
    """Convenience method for batch logging."""
    batch_metrics = {"batch": batch_idx, **metrics}
    self.log(batch_metrics, step=self.step_counter)

  def log_artifact(
    self, filepath: str, name: str, artifact_type: str = "model"
  ) -> None:
    """Log artifact to all backends."""
    # Check if any backend has save_artifacts enabled
    save_artifacts = any(
      getattr(config, "save_artifacts", False) for config in self.backend_configs
    )

    if not save_artifacts:
      return

    for backend in self.backends:
      try:
        backend.log_artifact(filepath, name, artifact_type)
      except Exception as e:
        print(f"Warning: Artifact logging failed: {e}")

  def finish(self) -> None:
    """Finish all backends."""
    for backend in self.backends:
      try:
        backend.finish()
      except Exception as e:
        print(f"Warning: Backend finish failed: {e}")

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.finish()
