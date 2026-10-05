# Deployment

ROS 2 deployment packages for TRON1 locomotion and loco-manipulation with
learned policies and optional online dynamics finetuning. The supported release
surface is limited to the `tron1_description` and `tron1_deploy` packages.

> [!WARNING]
> Real-hardware launch files command a physical robot. Validate the controller
> configuration in simulation, provide an emergency stop, suspend the robot,
> and keep the work area clear before enabling actuators.

## Supported workflows

The repository provides simulation and real-hardware launch files for base
velocity control, MPPI and recurrent-policy MPPI, arm control, and no-gripper
loco-manipulation. Runtime policies under `tron1/tron1_deploy/model` remain
provisional release assets until their redistribution rights are confirmed.

## Requirements

- Ubuntu 24.04 and ROS 2 Jazzy, the target platform of the upstream stack.
- A ROS 2 workspace with `colcon`, `rosdep`, and the required hardware drivers.
- [deployment_code_base](https://github.com/Hyh2001/deployment_code_base/tree/1ed0718de910d44e4f84ff96920be51f902c060e)
  at commit `1ed0718de910d44e4f84ff96920be51f902c060e`, including its required
  submodules.
- The sibling [`training`](../training/) Python package and its JAX
  dependencies for MPPI.

## Installation

```bash
mkdir -p ~/tron1_ws/src
cd ~/tron1_ws/src
git clone https://github.com/Hyh2001/deployment_code_base.git
git -C deployment_code_base checkout 1ed0718de910d44e4f84ff96920be51f902c060e
git clone git@github.com:generalroboticslab/OSRAM.git
```

Follow the `deployment_code_base` README to initialize its required submodules
and dependencies. Then build this repository:

```bash
cd ~/tron1_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select tron1_description tron1_deploy
source install/setup.bash
```

For MPPI, verify that its external Python dependency is available:

```bash
python3 -c "import training"
```

## Simulation

```bash
ros2 launch tron1_deploy tron1_sim_deploy_launch.py
ros2 launch tron1_deploy tron1_sim_deploy_mppi_launch.py
ros2 launch tron1_deploy tron1_sim_deploy_rnn_mppi_launch.py
ros2 launch tron1_deploy tron1_with_arm_sim_launch.py
ros2 launch tron1_deploy tron1_with_arm_no_gripper_sim_launch.py
```

The no-gripper launch supports `use_keyboard_joy:=true`; otherwise the launch
files use a joystick.

## Real hardware

Real-hardware launches require the TRON1 address explicitly. Workflows using
Vicon also require the VRPN server address:

```bash
ros2 launch tron1_deploy tron1_real_deploy_launch.py robot_ip:=ROBOT_IP
ros2 launch tron1_deploy tron1_real_deploy_mppi_launch.py \
  robot_ip:=ROBOT_IP vicon_server:=VICON_HOST
ros2 launch tron1_deploy tron1_real_deploy_rnn_mppi_launch.py \
  robot_ip:=ROBOT_IP vicon_server:=VICON_HOST
ros2 launch tron1_deploy tron1_with_arm_real_launch.py robot_ip:=ROBOT_IP
ros2 launch tron1_deploy tron1_with_arm_no_gripper_real_launch.py \
  robot_ip:=ROBOT_IP vicon_server:=VICON_HOST
```

Replace `ROBOT_IP` and `VICON_HOST` with local values. Arm launches also require
the corresponding AIRBOT hardware interface and CAN setup.

## Runtime models

Supported configurations retain only these assets:

- Base control: `standing_policy3.onnx`, `velocity_policy.onnx`, `test.onnx`.
- RNN/residual modes: `velocity_policy_gru.onnx` and
  `velocity_policy_residual.onnx`.
- Arm control: `loco-manipulation_policy.onnx` and
  `loco-manipulation_standing_policy.onnx`.
- No-gripper control: `loco-manipulation_policy_no_gripper.onnx` and
  `loco-manipulation_standing_policy_no_gripper_new3.onnx`.
- MPPI dynamics: `velocity_policy/reptile_mppi.npz`,
  `velocity_policy/reptile_gru.npz`, and `pose_tracking/reptile.npz`.

Generated bags and finetuned checkpoints belong under ignored `bag/`
directories and must not be committed.

## Troubleshooting

- Missing packages: source ROS 2, `deployment_code_base`, and this workspace in
  that order, then check the package with `ros2 pkg prefix <package>`.
- Missing models: confirm the model exists under `share/tron1_deploy/model` and
  the selected YAML uses a package-relative path.
- Hardware errors: verify launch arguments, routing, firewall rules, and the
  hardware-interface package before enabling controllers.
- MPPI import errors: activate the Python environment containing JAX and
  `training` before sourcing the workspace.

## Repository layout

- `tron1/tron1_description`: URDF, Xacro, meshes, MuJoCo descriptions, and RViz.
- `tron1/tron1_deploy`: controllers, planners, launch files, configs, and models.
