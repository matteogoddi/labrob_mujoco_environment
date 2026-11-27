# Unitree G1 Experiment Guide

The MuJoCo simulation and the physical G1 share a ROS 2 Foxy visualization interface in RViz2. Run the controller and RViz2 in two terminals attached to the same experiment container. You can open or restart RViz2 at any point during an experiment. If you observe simulation and hardware at the same time, use separate ROS domains for their observers so that topics with the same name stay separate.

## Docker deployment

The image runs on a Linux x86-64 workstation with Docker Engine. MuJoCo and RViz2 use the host's local X11 or XWayland display. The image includes ROS 2 Foxy, the Unitree message types and DDS middleware, the native SDK2 controller, the robot models and the RViz plot panel.

### Two interactive Docker terminals

Build the current image once with `./scripts/g1-docker build`, or load the team's current image archive. Then open both terminals in the project directory.

| Session | Terminal 1 on the host | At the first container prompt | Terminal 2 on the host | At the second container prompt |
| --- | --- | --- | --- | --- |
| MuJoCo | `./scripts/g1-docker run sim` | `rviz` | `./scripts/g1-docker run sim` | `sim` |
| Physical G1 | `./scripts/g1-docker run robot` | `rviz` | `./scripts/g1-docker run robot` | `ros2 topic echo /g1_observer/status` |

Use the same host command in both terminals. Simulation and hardware have distinct container names and can run concurrently. On first use, `run robot` selects the single active physical Ethernet interface, or the only physical Ethernet interface if its link is down. On multi-NIC hosts, specify the robot-facing interface with `./scripts/g1-docker run robot enp3s0`. Later terminals reuse the session's interface. The first command starts an interactive container with `docker run --rm -it`. The second terminal joins it with `docker exec -it`. Both prompts show `(g1-sim)` or `(g1-robot)`. The terminals share the controller's local telemetry socket. Ctrl+C stops the foreground program and returns to its prompt. Stop both programs before typing `exit` in the first shell, which removes the container.

On a physical G1 reporting Mode 5, `rviz` displays the Model 13 mesh and live body and Dex3 feedback. The separate `step1` command in the second terminal opens the suspended arm test described in section 2. On a matching Mode 13 G1, use `rviz hardware_mode:=13`. After completing the Develop-mode handoff in section 2, `robot` in the second terminal starts the physical **Mode 13** motor controller. Starting the shell or viewer only observes the robot.

For source development, use `./scripts/g1-docker dev run sim` (or `dev run robot INTERFACE`) in the first terminal. Type `build` once inside that shell, then `rviz`. Repeat the same `dev run` command in the second terminal. The short commands use the mounted checkout and its build cache.

#### Using Docker directly

The launcher configures graphics, mounts and network settings. For a local NVIDIA/X11 workstation, the equivalent interactive simulation session can also be started directly:

```bash
docker run --rm -it --init --name g1-lab --network none \
  --hostname "$(hostname)" --user "$(id -u):$(id -g)" -e HOME=/tmp/g1-home \
  --gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,display,utility,compute \
  -e DISPLAY -e QT_X11_NO_MITSHM=1 -e XAUTHORITY=/tmp/g1.xauth \
  -v "${XAUTHORITY:-$HOME/.Xauthority}:/tmp/g1.xauth:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro g1-experiment:foxy
```

Type `rviz` inside. From a second host terminal, run `docker exec -it g1-lab g1-env bash` and type `sim`. For a hardware session, replace `--network none` in the first command with `--network host -e G1_SOURCE=robot -e G1_INTERFACE=enp3s0` using the robot-facing interface. Mount a writable host directory at `/opt/g1/experiments` to keep saved hardware logs after exit.

### Develop from a checkout on Ubuntu 22.04, 24.04 or 26.04

Use an x86-64 Linux host with Docker Engine and a local X11 or XWayland session. The host supplies the kernel, display and graphics driver. The image supplies Ubuntu 20.04, ROS 2 Foxy, MuJoCo, the SDK and the compiler. CMake builds HPIPM and BLASFEO directly from the two source directories in your checkout. Different host installations of HPIPM, ROS or Python can coexist with this environment. Host installation instructions are available in the [Docker Engine Ubuntu guide](https://docs.docker.com/engine/install/ubuntu/).

From a clone or download of this branch, the shortest development workflow is:

```bash
./scripts/g1-docker dev up       # Builds the environment image if missing
./scripts/g1-docker dev build    # Builds this checkout inside the container
./scripts/g1-docker dev sim      # Terminal 1
./scripts/g1-docker dev rviz     # Terminal 2, in the same checkout
```

The first image build downloads dependencies. Loading the team's image archive first skips that build. The development container uses your user ID and a checkout-specific name. Keep using the launcher from the same checkout in both terminals.

For an offline handoff, copy the image archive, source Git bundle and checksum file to the same directory. Import the image and clone the editable snapshot:

```bash
sha256sum -c SHA256SUMS
docker load -i g1-experiment-foxy-linux-amd64.tar.gz
git clone -b dai-Experiment g1-simulation-source.bundle g1-simulation
cd g1-simulation
git switch -c my-experiment
./scripts/g1-docker dev up
./scripts/g1-docker dev build
```

From the checkout, run `./scripts/g1-docker dev sim` in one terminal and `./scripts/g1-docker dev rviz` in another. Check the GPU with `./scripts/g1-docker dev renderer`. NVIDIA hosts need the NVIDIA Container Toolkit for accelerated graphics. The development build selects the AVX2 solver kernels on CPUs with AVX2 and FMA and the generic kernels on other supported x86-64 machines. Frame rates depend on the workstation.

Edit the checkout using your normal editor. After changing C++ or the plot plugin, close the affected application, rerun `./scripts/g1-docker dev build`, and restart it. Use `./scripts/g1-docker dev exec bash` for a shell in the mounted checkout. Compiled files persist in `build/docker-dev/` across `dev down` and `dev up`. Each developer should build their own cache. BLASFEO configures `blasfeo/include/blasfeo_target.h` for the local CPU. If a generic build changes that generated file, leave it out of your commits. Commit changes on your local branch and add your team's Git remote to share them. The bundle contains the handoff history and can be cloned offline.

To test your working tree on hardware, close the simulation applications, then switch the development container to the robot network:

```bash
./scripts/g1-docker dev down
./scripts/g1-docker dev up robot enp3s0    # Use the host's robot Ethernet interface
./scripts/g1-docker dev rviz               # Mode 5 read-only viewer
# In a second terminal
./scripts/g1-docker dev exec ros2 topic hz /lowstate
```

The compiled targets persist when you switch sessions. On a matching Mode 13 G1, start the viewer with `./scripts/g1-docker dev rviz hardware_mode:=13`. Complete the manual Develop-mode handoff in section 2 before running `./scripts/g1-docker dev control` in a second terminal. That command runs the physical Mode 13 controller and can move the robot. Use `step1` for the suspended Mode 5 arm-only test in section 2. Stages 2–6 and the ground-frame handoff still require implementation and validation.

### Run a packaged image

Build once from the project root. To distribute this build, give users the compressed image and the executable `scripts/g1-docker` launcher. Users who receive the archive can start at the load command.

```bash
# On the build machine
./scripts/g1-docker build
docker save g1-experiment:foxy | gzip > g1-experiment-foxy-linux-amd64.tar.gz

# On another workstation
docker load -i g1-experiment-foxy-linux-amd64.tar.gz
```

#### Simulation

```bash
# Terminal 1
./scripts/g1-docker up sim
./scripts/g1-docker sim

# Terminal 2
./scripts/g1-docker rviz
```

Press keys 1 through 6 in the MuJoCo window as described below. Both programs run in the same container so the observer can reconnect to the controller's local telemetry socket. Restart RViz independently with Ctrl+C and the same command.

#### Physical robot

Connect the host Ethernet interface to the G1 and configure its IP on the robot's subnet. Stop the simulation session before reusing its container name. Start the hardware viewer with the interface name reported by `ip -br addr`.

```bash
./scripts/g1-docker down
ip -br addr
./scripts/g1-docker up robot enp3s0    # replace with your Ethernet interface
./scripts/g1-docker rviz
```

From a second terminal, inspect the live bus with `./scripts/g1-docker exec ros2 topic hz /lowstate`. The robot container uses host networking and selects the specified interface for Cyclone DDS. Starting the container or RViz only observes the robot. The Mode 13 controller procedure below applies only to matching hardware. The Mode 5 arm test has its own `step1` command.

For Mode 13, start the viewer with `./scripts/g1-docker rviz hardware_mode:=13`. To run the physical experiment, open a control terminal with `./scripts/g1-docker exec bash`, change to `/opt/g1/build`, and use the hardware command below with the same Ethernet interface.

Stop the controller and RViz before running `./scripts/g1-docker down`. Saved hardware experiments appear in `./experiments` on the host. Simulation text logs written to `/tmp` stay in the container until removal. Use `docker cp` to copy those logs before teardown. Set `G1_DATA` before startup to choose the hardware experiment directory. The launcher automatically selects NVIDIA graphics when the Docker NVIDIA runtime and driver are available. Use `./scripts/g1-docker renderer` to inspect the actual OpenGL renderer. Set `G1_NVIDIA=1` to require NVIDIA or `G1_NVIDIA=0` to disable it. For a five-second RViz frame-rate report, run `./scripts/g1-docker exec env G1_REPORT_FPS=1 g1-env rviz` in the viewer terminal. Set `G1_CONTAINER` in both terminals to choose another session name. For headless topic inspection, unset `DISPLAY` at startup and pass `gui:=false` to the RViz launcher.

## 1. MuJoCo simulation (Mode 13)

**Terminal 1: Start MuJoCo.** Run the executable from `build/` so it can locate the scene and robot model.

```bash
source /opt/ros/foxy/setup.bash
cd ~/G1_Experiment/labrob_mujoco_environment/build
./main --sim --rviz
```

For a new working directory or after a controller change, first run `cmake -S . -B build` and `cmake --build build --target main --parallel 2` from the project root.

**Terminal 2: Open RViz2.** The script loads Foxy. Before you press the first key in MuJoCo, RViz2 can already display the initial pose of all 29 body joints and 14 finger joints. To restart the observer during a run, press `Ctrl+C` in this terminal and run the script again.

```bash
cd ~/G1_Experiment/labrob_mujoco_environment
./scripts/run_rviz.sh sim
```

Click the **MuJoCo window** to give it keyboard focus. Press the number shown in its title and wait for the next prompt. The simulation terminal also prints each prompt.

| Key | Simulation action | What to watch in RViz2 |
| --- | --- | --- |
| `1` | Open both arms slightly to give the fingers clearance. | Both shoulders and arms move with the model. |
| `2` | Close both hands into fists. | The Dex3 fingers curl on both hands. |
| `3` | Raise the forearms over about six seconds until they point forward, level with the ground. | Select `Joint angle` and an elbow joint to follow the motion. |
| `4` | Stand with both feet planted under joint proportional-derivative (PD) control. | `Joint torque` starts showing the torque applied to each simulated joint. |
| `5` | Switch to standing under whole-body control (WBC). | Desired and feedback points, foot force arrows, center of mass (CoM) and zero moment point (ZMP) predictions, and WBC torque commands for the body motors become available. |
| `6` | Shift weight and walk forward. | Watch the planned left and right foot placements and the lateral ZMP as support shifts between feet. |

## 2. Physical G1 experiments

**Terminal 1: Connect to the G1 and observe its state.** Connect the host to the robot's Ethernet port. Use `ip -br addr` to find the host interface and check that `NetworkInterface` in `~/unitree_ros2/setup.sh` names that interface. This terminal opens a read-only view of the robot state and motor commands on the Unitree bus.

```bash
cd ~/G1_Experiment/labrob_mujoco_environment
source ~/unitree_ros2/setup.sh
./scripts/run_rviz.sh robot
```

RViz uses the Model 13 URDF with both Dex3-1 hands for simulation and hardware. The hardware viewer expects `mode_machine=5` and `mode_pr=0` by default. On a robot reporting 13, pass `hardware_mode:=13` to enable the corresponding body data checks and WBC telemetry. Body feedback markers use the physical configuration's kinematic model, so on a Mode 5 robot they can differ from points on the Model 13 display mesh. Neither the mesh nor a value of 5 or 13 confirms Develop mode or foot contact.

**Check both hands before control.** The viewer reads 29 body joints from `/lowstate` and seven measured finger joints per hand from `unitree_hg/msg/HandState`. It selects a fresh `/dex3/<side>/state` stream for each hand, falling back to `/lf/dex3/<side>/state`. Look for `Dex3 left live, right live` in the viewer status. A side marked `waiting` has not supplied a usable sample. A side marked `stale` has stopped supplying fresh measurements. The last visible pose may persist in TF. During live body feedback, 29, 36 and 43 published joint names correspond to zero, one and two fresh hands. [Unitree Dex3 motor order](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/g1/g1_joint_index_dds.md)

In a second terminal, run the following inside the robot container (`./scripts/g1-docker dev exec bash`) or a shell with the Unitree ROS environment. Use the hand topics listed for your robot. Inspect publishers and incoming messages on each side.

```bash
ros2 topic list -t | grep -E 'dex3/.*/state|lowstate'
ros2 topic info /lf/dex3/left/state
ros2 topic info /lf/dex3/right/state
ros2 topic echo /g1_observer/status
```

### Mode 5 suspended arm test in Docker

Keep the hoist supporting the full robot throughout this test. The release request can remove whole-body support. Start in the suspended zero-torque condition specified for the robot firmware, provide clearance for both arms, stop other low-level control programs, and keep an operator at the physical stop. State packets and RViz cannot verify hoist support. [Unitree physical deployment procedure](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/README.md#startup-process)

Keep the hardware RViz window open. In a second terminal, enter the updated hardware container with `./scripts/g1-docker run robot`, then run `step1`. Each key is immediate in this terminal; independent confirmations must be pressed separately. In an existing development checkout, run `build` after updating the source. A container created from an earlier packaged image retains its original commands. To keep an older viewer open and start a separate control shell using the updated image, use `G1_CONTAINER=g1-arm-test ./scripts/g1-docker run robot INTERFACE` on the host.

| Key | Effect |
| --- | --- |
| `d`, then `y` | Confirm that the robot is suspended, in zero torque and clear. Request release of the active motion service, then check its reported state. Wait for `Develop verified. Arms idle.` |
| `1` | Capture measured arm angles and open both arms once over four seconds. The terminal reports when measured angles settle at the targets and the position hold begins. |
| **Space**, `x`, Esc or Ctrl+C | Abort the test from any phase. End the arm stream; while ownership is verified, first send disable packets. Keep supporting the arms and body. Use the independent physical stop for an emergency, lost terminal or network failure. A switch RPC already sent can still complete. |
| `r`, then `y` | While ready or holding, release the arms first. After separate confirmation, request the motion service recorded before this session and verify its name. Restoring a service can change the robot's posture. If the robot was already in Develop mode at entry, use the physical remote for recovery. |

The first three shoulder axes and elbow on each side receive the simulation Stage 1 **change** from the current pose. On the left the offsets are approximately −0.008, +0.220, +0.070 and +0.218 radians. Right shoulder roll and yaw are mirrored. Only those eight motors are enabled. Other body slots in `/lowcmd` are disabled and no Dex3 commands are published. Arm PD uses the SDK example gains (40 and 1) with a stiffness ramp. These settings and tracking limits require validation on the physical unit. The container requests serial PR coordinates and machine mode 5, checks CRC and advancing state ticks, and monitors service status, other command writers, joint feedback and command timing. The keyboard abort is a best-effort software stop. Confirm the physical state with the remote before recovery. [Unitree motion switcher](https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/utils/motion_switcher.py) · [Unitree low-level example](https://github.com/unitreerobotics/unitree_sdk2/blob/f29ee9f234851e9e79f75102c0f9e83008d8fdd1/example/g1/low_level/g1_ankle_swing_example.cpp)

### Mode 13 WBC experiment

**Enter Develop mode before a low-level control test.** The read-only viewer can run as soon as state packets arrive. For a motor-command test, support the G1 on its hoist and follow the procedure for its firmware and remote. Unitree's G1 deployment guide starts with the suspended robot in zero torque, then uses `L2 + R2` to enter Debug (Develop) mode with damping. Confirm the mode locally before starting the controller. [Unitree physical deployment instructions](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/README.md#startup-process)

**Terminal 2. Run the hardware WBC experiment on a matching Mode 13 robot.** The controller checks the reported machine mode and queries the motion service before opening its low-level command publisher. An active service or a failed query stops startup. Enter the same network interface used by the observer.

```bash
cd ~/G1_Experiment/labrob_mujoco_environment/build
read -r -p "Enter the network interface connected to the G1: " g1_iface
./main --robot "$g1_iface" --rviz
```

At startup, enter the options required for the experiment, separated by spaces. Option `1` enables CoM feedback control, `2` enables whole-body feedback control, and `3` enables the extended Kalman filter (EKF). Press Enter to confirm. In RViz2, `WBC live` indicates that the observer has received controller data aligned in time with the encoder readings.

| Gamepad button | Action |
| --- | --- |
| `A` | The first press starts IMU calibration. Subsequent presses toggle whole-body feedback control when both CoM feedback and the EKF are active. |
| `X` | Toggle CoM feedback control when the EKF is active. |
| `B` | Request a change in walking state. From standing, this requests the first step. |
| `Y` | Exit the control loop. The control terminal prompts you to save or discard the logs. |

### Choose the RViz2 fixed frame

The default frame, `pelvis`, shows joint and foot poses relative to the pelvis. To see how far the robot has walked, supply a base pose from one of the localization sources below. Append the launch arguments to `./scripts/run_rviz.sh robot`.

| Source | RViz2 setup |
| --- | --- |
| Encoders and IMU | Use `pelvis` for relative joint poses. Use `fixed_frame:=g1_attitude` to inspect body tilt in a level frame that translates with the pelvis. |
| Controller EKF | The default is `base_source:=controller`. Once the controller provides a valid transform from `world` to `pelvis`, set `Fixed Frame` to `world` in RViz2. |
| External localization or motion capture | Use `base_source:=odometry fixed_frame:=world`. Publish odometry for `pelvis` in `world` on `/g1_observer/base_odometry`, aligned with the controller's world frame. |
| Unitree sport odometry | Use `base_source:=sport fixed_frame:=unitree_odom`. Comparing tracking errors in world coordinates also requires a calibrated transform from `world` to `unitree_odom`. |

## 3. RViz2 controls and legend

### 3D view: arrows, points, and trajectories

In the left pane, open **Displays > Walking diagnostics > Namespaces** to toggle individual marker groups. Each force arrow starts at the point of application and points in the direction of the force. The scale is **1 mm/N**, so a 100 N force appears as a 10 cm arrow.

| Marker | RViz namespace | Meaning |
| --- | --- | --- |
| Pink arrows | `contact` | MuJoCo contact force at each contact point. A planted foot can have several arrows. |
| Cyan arrows | `wbc` | Resultant support force computed by the WBC for each foot. |
| Purple arrows | `estimated` | Resultant support force from the residual estimator. Compare it with the cyan arrow on the same foot to see how the estimate differs from the WBC solution. |
| Red arrows | `external` | External force applied to a simulated body, such as a push on the torso. |
| Large translucent orange points | `desired_com`, `desired_left_foot`, `desired_right_foot` | Desired positions of the CoM and both soles for the current control cycle. |
| Small cyan points | `feedback_com`, `feedback_left_foot`, `feedback_right_foot` | Feedback positions for the same three points. In a common frame, the distance between a cyan point and its orange target gives the magnitude of the position tracking error. |
| Green point and line | `current_com`, `mpc_com` | The point is the current filtered CoM position used by the model predictive controller (MPC). The line traces the predicted CoM trajectory from that point. |
| Yellow point and line | `current_zmp`, `mpc_zmp` | The point is the current filtered ZMP used by the MPC. The line traces the predicted ZMP trajectory. |
| Blue and orange rectangles | `planned_left`, `planned_right` | Planned foot placement and orientation. Blue marks the left foot and orange marks the right foot. |

On hardware, an **orange skeleton** shows the commanded pose along joint chains with active position control. When every joint in a leg has position control enabled, a **small purple cube** marks the foot target calculated from `LowCmd` relative to the pelvis. Enable **Displays > Compare TF frames** to compare the robot model frames with the controller frames named `wbc_*`.

### Live plots: the last 10 seconds

The upper chart defaults to lateral ZMP and the lower chart to waist yaw torque. Each chart has its own signal selector. For a joint signal, choose the motor from the second selector. On the time axis, `0` at the right edge means now. The legend shows the latest values, and a dash indicates that no fresh sample is available. Selecting a different signal starts a new trace.

| Signal | Orange dashed line | Cyan solid line |
| --- | --- | --- |
| `ZMP X` / `ZMP Y` (m) | `Ref` is the controller's ZMP reference for the current cycle. | `Contact` is the simulated ZMP calculated from the net foot contact force and moment on the horizontal reference plane. It is published when the net normal force exceeds 1 N. On hardware, only the reference trace is available when WBC data is synchronized. |
| `Joint torque` (N·m) | WBC torque command in simulation or `LowCmd.tau` feedforward torque on hardware. | Applied joint drive torque (`qfrc_actuator`) in simulation or estimated joint torque (`LowState.tau_est`) on hardware. |
| `Joint angle` / `Joint velocity` (rad / rad/s) | Commanded angle or velocity on hardware when a synchronized `LowCmd` is available. | Joint state feedback, available from the initial simulation pose or from valid hardware encoder data. |

To inspect a **side-to-side weight shift**, select `ZMP Y` and compare the reference and contact traces. The yellow trajectory in the 3D view shows where the MPC predicts the ZMP will move next. To investigate **waist oscillation**, select the same joint, such as `waist_pitch_joint`, in both charts. Plot `Joint angle` in one and `Joint torque` in the other to compare when the angle and torque change.
