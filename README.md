# Online Sim-to-Real Adaptation via Closed-Loop System Modeling
![OSRAM](assets/pipeline.svg)

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)]()
[<img src="https://img.shields.io/badge/Framework-JAX-red.svg"/>](https://github.com/google/jax)
[![CUDA Version](https://img.shields.io/badge/CUDA-%3E13.0-green.svg)]()
[![Ubuntu](https://img.shields.io/badge/Ubuntu-24.04-purple.svg)]()

[Yuhao Huang](https://hyh2001.github.io/),
[Samuel A. Moore](https://samavmoore.github.io/), and
[Boyuan Chen](http://boyuanchen.com/)
<br>
*Duke University*
<br>

OSRAM (Online Sim-to-Real Adaptation via Closed-Loop System Modeling) is a
research framework for training robot policies and learned dynamics models,
adapting those models from online observations, and deploying the resulting
controllers on TRON1 in simulation or on hardware.

This repository combines the training and deployment code in one checkout. The
two modules remain independently installable and keep their own dependencies,
configuration, and documentation.

<!-- > [!IMPORTANT]
> This is a pre-release research codebase. -->
<!-- > [!IMPORTANT]
> This is a pre-release research codebase. The imported modules do not currently
> include software licenses. Public visibility alone does not grant permission
> to use, modify, or redistribute the source, model weights, or robot meshes. -->

## Modules

| Module | Purpose | Source snapshot |
| --- | --- | --- |
| [`training`](training/) | Python package for Mjlab policy training, rollout collection, learned-dynamics training, online finetuning, and MPPI controllers. See its [README](training/README.md). 
| [`deployment`](deployment/) | ROS 2 packages, robot descriptions, launch files, policies, and planners for TRON1 simulation and hardware deployment. See its [README](deployment/README.md).

## How the modules fit together

```text
Mjlab policy training and simulation
              |
              v
      trajectory collection
              |
              v
learned dynamics training / finetuning
              |
              v
 policy and dynamics checkpoints
              |
              v
 ROS 2 simulation or TRON1 deployment
```

The training module owns the Python package and the training pipeline. The
deployment module owns the ROS 2 runtime and robot assets. Its MPPI nodes import
`training`, so install and activate the Python environment
before running those nodes. Moving a newly trained checkpoint into a deployment
configuration is currently an explicit step; the modules do not automatically
synchronize generated artifacts.

## Getting started

### Training and dynamics finetuning

Requirements include Linux, Python 3.12 or 3.13, and
[`uv`](https://docs.astral.sh/uv/). CUDA 12.8 is required for simulator training
and GPU-backed JAX workflows.

```bash
cd training
uv sync --frozen --dev
uv run list-envs
```

On a compatible CUDA workstation, include the CUDA extra:

```bash
uv sync --frozen --dev --extra cuda
```

See the [training module guide](training/README.md)
for policy training, exploration, dataset creation, model training, plotting,
and development checks.

### ROS 2 deployment

The deployment module targets Ubuntu 24.04 and ROS 2 Jazzy. It also requires
the packages supplied by
[`deployment_code_base`](https://github.com/Hyh2001/deployment_code_base).
Place this repository and that dependency under a ROS 2 workspace's `src/`
directory, then build the two TRON1 packages:

```bash
cd ~/tron1_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select tron1_description tron1_deploy
source install/setup.bash
```

For MPPI launches, first activate the environment in which the training
module is installed and confirm the import succeeds:

```bash
python3 -c "import training"
```

See the [deployment module guide](deployment/README.md)
for simulation commands, hardware launch arguments, runtime-model locations,
and troubleshooting.

> [!WARNING]
> Real-hardware launch files command a physical robot. Validate the selected
> policy and controller configuration in simulation, provide a working
> emergency stop, suspend the robot, and clear the work area before enabling
> actuators.

## Repository layout

```text
OSRAM/
├── training/                                  # Python training package
├── deployment/                                # ROS 2 deployment packages
└── README.md
```

Run build, test, and formatting commands from the relevant module directory;
there is no root-level environment that merges their Python and ROS 2
dependencies.

## Citation
If you find our paper or codebase helpful, please consider citing:
```

```