"""Registers the task before running mjlab's training pipeline."""

from mjlab.scripts.train import main

import training.tasks  # noqa: F401

if __name__ == "__main__":
  main()
