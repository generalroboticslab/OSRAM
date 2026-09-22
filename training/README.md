# Training

The OSRAM training module trains learned Tron1 dynamics models from policy
rollouts and adapts them online. It integrates with
[Mjlab](https://github.com/mujocolab/mjlab) for policy training, playback, and
MuJoCo simulation.

> This repository does not yet include a software license. Its source is publicly visible, but no permission to use, modify, or redistribute it is granted until a license is added.

## Supported workflows

The public task surface covers:

- Tron1 flat-ground velocity policies (MLP and GRU)
- Residual velocity policies, including GRU finetuning
- Tron1 standing
- Tron1-with-arm velocity and standing
- Velocity and Tron1-with-arm exploration
- Tron1 loco-manipulation and exploration
- Forward, Reptile, and finetuning dynamics-model training
- Reusable MPPI, open-loop MPPI, and action-history MPPI controllers

Run `uv run list-envs` after installation for the authoritative task IDs.

## Requirements

- Linux x86-64
- Python 3.12 or 3.13
- [`uv`](https://docs.astral.sh/uv/)
- NVIDIA CUDA 12.8 for simulator training and GPU-backed JAX workflows
- A W&B account only when loading checkpoints from W&B

Automated unit tests and metadata checks run on CPU. Environment stepping, policy training, exploration, and the full dynamics pipeline require a CUDA-capable system.

## Installation

Install the locked CPU development environment:

```bash
uv sync --frozen --dev
```

Install the CUDA JAX extra on a compatible workstation:

```bash
uv sync --frozen --dev --extra cuda
```

Activate the environment if you want to call commands without `uv run`:

```bash
source .venv/bin/activate
```

## Policy workflows

List registered tasks:

```bash
uv run list-envs
```

Train a policy:

```bash
uv run policy-train Mjlab-Velocity-Flat-Tron1
```

Inspect playback options, then play a checkpoint:

```bash
uv run policy-play Mjlab-Velocity-Flat-Tron1 --help
uv run policy-play Mjlab-Velocity-Flat-Tron1 --checkpoint-file checkpoints/model.pt
```

Residual tasks expect the base velocity checkpoint at `logs/rsl_rl/tron1_velocity/base_policy.pt`. Place the policy there before constructing or running a residual environment.

## Exploration and datasets

Collect velocity trajectories with a local policy checkpoint:

```bash
uv run policy-explore Mjlab-Velocity-Flat-Tron1-Explore \
  --task velocity \
  --agent trained \
  --checkpoint-file checkpoints/model.pt \
  --save-trajectory-file artifacts/trajectories/velocity.pt
```

Collect loco-manipulation trajectories:

```bash
uv run policy-explore Mjlab-Loco-Manip-Flat-Tron1-Explore \
  --task loco-manipulation \
  --agent trained \
  --checkpoint-file checkpoints/loco-manip.pt \
  --save-trajectory-file artifacts/trajectories/loco-manipulation.pt
```

Zero and random agents do not need a checkpoint:

```bash
uv run policy-explore Mjlab-Velocity-Flat-Tron1-Explore \
  --task velocity \
  --agent random \
  --device cpu \
  --num-envs 2 \
  --save-trajectory-file artifacts/trajectories/random.pt
```

Dataset and dynamics commands accept YAML configuration files. The supported
dynamics-model target is `tron1`. Runnable templates live under
`src/training/tasks/tron1/config/dynamics_training_cfg/` and use
repository-relative paths under `artifacts/`.

```bash
uv run create-dataset --config-file src/training/tasks/tron1/config/dynamics_training_cfg/one_dataset_cfg.yaml
uv run dynamics-train --config-file src/training/tasks/tron1/config/dynamics_training_cfg/standard_mlp_cfg.yaml
uv run dynamics-train --config-file src/training/tasks/tron1/config/dynamics_training_cfg/reptile_mlp_cfg.yaml
uv run dynamics-train --config-file src/training/tasks/tron1/config/dynamics_training_cfg/finetune_mlp_cfg.yaml
uv run plot --config-file src/training/tasks/tron1/config/dynamics_training_cfg/plot_prediction_cfg.yaml
```

Use the `velocity/` and `loco-manipulation/` subdirectories for their specialized templates. Input trajectories, datasets, checkpoints, models, and plots are intentionally excluded from Git.

## Development checks

```bash
uv run ruff format --check .
uv run ruff check .
JAX_PLATFORMS=cpu MPLBACKEND=Agg uv run pytest -q
uv run cffconvert --validate
uv build
```

## Troubleshooting

- If CUDA is unavailable, use the base installation and restrict work to CPU-safe tests and utilities.
- If a local checkpoint is supplied, W&B is not contacted. Without `--checkpoint-file`, exploration requires `--wandb-run-path`.
- Missing dataset, model, and checkpoint paths are user-provided artifacts; generate them with the preceding pipeline or update the YAML template.
- Generated files belong under `artifacts/`, `checkpoints/`, `trajectories/`, or `logs/` and are ignored by Git.
