# Unitree G1 experiments

MuJoCo and the physical G1 share one ROS 2 Foxy RViz view and one Linux x86-64 Docker image, `g1-experiment:foxy`. The image supplies MuJoCo, ROS and the Unitree SDK. The checkout includes HPIPM and BLASFEO.

## 1. Install Docker and configure GPU acceleration

| Host | One-time setup |
| --- | --- |
| Ubuntu 22.04, 24.04 or 26.04 | Install Docker Engine for your Ubuntu release and run the launcher from a local X11 or XWayland desktop. [Docker installation](https://docs.docker.com/engine/install/ubuntu/) |
| Ubuntu 18.04 or 20.04 | Use a Docker Engine and graphics driver compatible with that release. These are legacy hosts outside Docker’s current Ubuntu installation list. The reference workstation runs 20.04. [Docker host requirements](https://docs.docker.com/engine/install/ubuntu/#os-requirements) |
| Windows with WSL2 | Update WSL with `wsl --update` in PowerShell, install the Windows GPU driver, and enable Docker Desktop’s WSL2 backend and Ubuntu integration. Run the launcher in the Ubuntu terminal with WSLg available. The Docker backend needs access to the WSLg X11 socket, `/dev/dxg` and `/usr/lib/wsl`. [WSL GUI setup](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) · [Docker WSL integration](https://docs.docker.com/desktop/features/wsl/) · [WSLg containers](https://github.com/microsoft/wslg/blob/main/samples/container/Containers.md) |

**Native NVIDIA:** install the host driver. After stopping experiments, run `./scripts/setup-docker-nvidia` once to configure the NVIDIA Container Toolkit. The script restarts Docker. Intel and AMD integrated graphics use the host DRM render devices and the image’s Mesa driver. WSLg uses the Windows GPU through Mesa D3D12. [NVIDIA toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) · [WSLg adapter selection](https://github.com/microsoft/wslg/wiki/GPU-selection-in-WSLg)

On creation of a graphical session, the launcher probes OpenGL, prefers NVIDIA and falls back to the available Mesa adapter. Read **OpenGL renderer** in the first terminal. A GPU model or `D3D12 (GPU model)` identifies hardware rendering. `llvmpipe` or `softpipe` means CPU rendering.

## 2. Enter Docker, build and run the simulation

Run **`./scripts/g1-docker run sim` in both host terminals** from the same checkout. The first starts the container and the second joins it. On first use, the launcher builds the image if needed. At either container prompt, type `build` once and wait for it to finish.

| Inside terminal 1 | Inside terminal 2 |
| --- | --- |
| `rviz` opens the observer. | `sim` starts MuJoCo. |

Edit source on the host. To test a planner or WBC change, stop MuJoCo with Ctrl+C and type **`build && sim`**. RViz can stay open. The build cache persists in `build/docker-dev/`. Stop both programs before exiting the first shell.

For offline deployment, verify the supplied files with `sha256sum -c SHA256SUMS` and import the image with `docker load -i g1-experiment-foxy-linux-amd64.tar.gz`. To develop, run `git clone -b dai-Experiment g1-simulation-source.bundle g1-simulation` and use its launcher. A standalone launcher with only the image runs prebuilt programs and skips the source build.

## 3. Connect and test a physical G1

1. Connect the robot Ethernet cable to a native Linux host and configure the interface on the robot subnet. Run **`./scripts/g1-docker run robot` in both host terminals**. If several wired interfaces are active, append the robot-facing interface name. Use the source build from section 2.
2. Type `rviz` in the first shell. Confirm joint motion matches the G1 and the status reads `Dex3 left live, right live`. RViz displays the Model 13 mesh. The default hardware feedback mapping is Mode 5. The default hardware frame, `pelvis`, shows relative pose. A world-aligned foot height requires a validated base transform and physical contact confirmation.
3. **Mode 5 suspended arm test:** keep the robot fully supported by the hoist, start in its firmware’s zero-torque condition, clear both arms, stop other controllers and station an operator at the physical stop. Type `robot` in the second shell. Press `d`, then separately `y`, to request Develop mode. Releasing the motion service can remove whole-body support. Wait for `Develop verified. Arms idle.` before pressing `1` for the four-second arm movement. The current Mode 5 test ends at step 1.
4. Space, `x`, Esc or Ctrl+C requests a software stop. Use the physical stop for an emergency or lost communication. To hand control back while ready or holding, press `r`, then confirm separately with `y`. Keep the hoist loaded and verify the restored service and posture locally. If the robot started in Develop mode, recover with its remote. [Unitree deployment procedure](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/README.md#startup-process)

**Mode 13 hardware only:** use `rviz hardware_mode:=13`. With the robot supported and the firmware-specific Develop handoff confirmed locally, `robot --mode13` starts the separate WBC experiment. The controller checks machine mode and motion-service release. Its gamepad uses A for initial IMU calibration, X for CoM feedback, B for a walking-state change and Y to exit.

## 4. Six staged actions in MuJoCo

Focus the **MuJoCo window**, press the key shown in its title and wait for the next prompt. The full sequence below runs in simulation.

| Key | Action | What to check |
| --- | --- | --- |
| 1 | Open both arms slightly. | The palms clear the hips before the fingers move. |
| 2 | Close both Dex3 hands. | Both sets of fingers curl into fists. |
| 3 | Raise both forearms slowly. | The forearms finish pointing forward and level. |
| 4 | Enter PD standing with both feet planted. | Inspect sole placement relative to the ground before switching control. |
| 5 | Switch to WBC standing. | Check balance, support forces and desired versus measured positions. |
| 6 | Shift weight and walk forward. | Follow the support change, planned footprints and lateral ZMP tracking. |

## 5. Read the RViz simulation display

Simulation uses the `world` fixed frame. Toggle groups under **Displays → Walking diagnostics → Namespaces**. Force arrows start at the application point and point along the force. **100 N = 10 cm**.

| Marker | Namespace | Meaning |
| --- | --- | --- |
| Pink arrows | `contact` | MuJoCo contact forces. One sole can have several contact points. |
| Cyan arrows | `wbc` | Resultant support force commanded by WBC for each foot. |
| Purple arrows | `estimated` | Foot support force estimated by the residual estimator. |
| Red arrows | `external` | External force applied to a simulated body, such as a torso push. |
| Large translucent orange dots | `desired_com`, `desired_left_foot`, `desired_right_foot` | Desired center of mass (CoM) and sole positions. |
| Small cyan dots | `feedback_com`, `feedback_left_foot`, `feedback_right_foot` | Measured positions. The gap to each orange dot shows tracking error. |
| Green dot and line | `current_com`, `mpc_com` | Filtered CoM used by MPC and its predicted trajectory. |
| Yellow dot and line | `current_zmp`, `mpc_zmp` | Filtered zero moment point (ZMP) and its predicted trajectory. |
| Blue and orange rectangles | `planned_left`, `planned_right` | Planned sole placement and orientation. Blue is left, orange is right. |

The white label gives gait phase, support side and simulation time. **Compare TF frames** overlays the `wbc_*` frames. On hardware, the orange skeleton shows commanded joint positions and purple cubes mark available foot targets relative to the pelvis.

**Right-hand plots** show the last 10 seconds. Select a signal and, for joint data, a motor. Orange dashed is the reference or command. Cyan solid is feedback. For `ZMP Y`, compare the lateral reference with the ZMP from simulated ground contact (net normal force above 1 N). For `Joint torque`, compare WBC torque with applied actuator torque. Joint angle and velocity show measured state in simulation. A dash means the sample is unavailable.
