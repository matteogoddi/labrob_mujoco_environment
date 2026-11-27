# Unitree G1 Experiment Guide

The MuJoCo simulation and the physical G1 share a ROS 2 Foxy visualization interface in RViz2. Run the controller and RViz2 in two terminals attached to the same experiment container. You can open or restart RViz2 at any point during an experiment. If you observe simulation and hardware at the same time, use separate ROS domains for their observers so that topics with the same name stay separate.

## Docker deployment

The image runs on a Linux x86-64 workstation with Docker Engine. MuJoCo and RViz2 use the host's local X11 or XWayland display. The image includes ROS 2 Foxy, the Unitree message types and DDS middleware, the native SDK2 controller, the robot models and the RViz plot panel.

### Two interactive Docker terminals

Open both host terminals in the same project directory. Both commands use **`g1-experiment:foxy`**. The launcher prints the image and the source of the programs when you enter.

| Session | Host command in both terminals | First container shell | Second container shell |
| --- | --- | --- | --- |
| MuJoCo | `./scripts/g1-docker run sim` | `rviz` | `sim` |
| Physical G1 (Mode 5) | `./scripts/g1-docker run robot` | `rviz` | `robot` |

**First use from source:** type `build` once in either container shell and wait for it to finish before starting the applications. The launcher mounts the checkout beside the script at `/workspace/g1`. Both sessions use its compiled programs from `build/docker-dev/` on the host. A standalone launcher distributed with only the image archive prints `packaged binaries` and runs the prebuilt programs immediately.

Use the same host command in both terminals. Simulation and hardware have distinct container names and can run concurrently. On first use, `run robot` selects the single active physical Ethernet interface, or the only physical Ethernet interface if its link is down. On multi-NIC hosts, specify the robot-facing interface with `./scripts/g1-docker run robot enp3s0`. Later terminals reuse the session's interface. The first command starts an interactive container with `docker run --rm -it`. The second terminal joins it with `docker exec -it`. Both prompts show `(g1-sim)` or `(g1-robot)`. The terminals share the controller's local telemetry socket. Ctrl+C stops the foreground program and returns to its prompt. Stop both programs before typing `exit` in the first shell, which removes the container.

On a physical G1 reporting Mode 5, `rviz` displays the Model 13 mesh and live body and Dex3 feedback. The `robot` command in the second terminal opens the suspended arm test described in section 2. On a matching Mode 13 G1, use `rviz hardware_mode:=13`. After completing the Develop-mode handoff in section 2, `robot --mode13` in the second terminal starts the physical **Mode 13** motor controller. Starting the shell or viewer only observes the robot.

### Edit and rebuild a planner or controller

Edit the files on the host. Stop the simulation with Ctrl+C and run the following inside the same simulation shell. RViz can stay open during a controller rebuild.

```bash
build && sim
```

The build is incremental. It uses CMake for the native programs and colcon for the RViz observer. Restart RViz after changing its plugin. Build once per checkout and use the same host user for subsequent builds. The cache survives container exit. The previous `dev run` spelling remains an alias in a checkout.

### Deploy on another workstation

Use an x86-64 Linux host with Docker Engine and a local X11 or XWayland display. The image supplies Ubuntu 20.04, ROS 2 Foxy, MuJoCo, the Unitree SDK and the compiler. The build uses the HPIPM and BLASFEO sources in the checkout. The host supplies the kernel and graphics driver. NVIDIA graphics require the NVIDIA Container Toolkit. [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

From a clone of this branch, `./scripts/g1-docker run sim` builds the environment image on first use if it is missing. To work offline, copy the image archive, source bundle and checksum file into the same directory:

```bash
sha256sum -c SHA256SUMS
docker load -i g1-experiment-foxy-linux-amd64.tar.gz
git clone -b dai-Experiment g1-simulation-source.bundle g1-simulation
cd g1-simulation
./scripts/g1-docker run sim
```

Use the two-terminal workflow above. Each developer keeps a local build cache. Source builds select AVX2 solver kernels when the CPU supports AVX2 and FMA. A generic BLASFEO build can update the generated `blasfeo/include/blasfeo_target.h`.

For a binary-only handoff, distribute the launcher beside the image archive in the same layout (`scripts/g1-docker`). Load the archive once and use the same `run sim` or `run robot` command. To publish an updated prebuilt image from the source checkout:

```bash
./scripts/g1-docker build
docker save g1-experiment:foxy | gzip > g1-experiment-foxy-linux-amd64.tar.gz
```

When switching from an older session, secure the robot, stop its programs, and exit its shells before reopening them. Running containers keep the image they started with. Saved hardware experiments appear in `./experiments` on the host. Set `G1_DATA` before startup to use another directory. Copy logs from `/tmp` before container removal. The launcher selects available NVIDIA graphics automatically. Set `G1_NVIDIA=1` to require NVIDIA or `G1_NVIDIA=0` to disable it. For headless observation, unset `DISPLAY` before startup and run `rviz gui:=false`. Use `G1_REPORT_FPS=1 rviz` for a five-second frame-rate report.

## 1. MuJoCo simulation (Mode 13)

Open two shells with `./scripts/g1-docker run sim`. Start `sim` in one and `rviz` in the other. RViz can display the initial pose of all 29 body joints and 14 finger joints before the first MuJoCo keypress. To restart the observer, press Ctrl+C in its terminal and type `rviz` again.

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

Connect the host to the G1 Ethernet port and configure its IP on the robot subnet. Open the first terminal with `./scripts/g1-docker run robot`, then type `rviz` for a read-only view. Use `ip -br addr` on the host if you need to select an interface explicitly.

RViz uses the Model 13 URDF with both Dex3-1 hands for simulation and hardware. The hardware viewer expects `mode_machine=5` and `mode_pr=0` by default. On a robot reporting 13, pass `hardware_mode:=13` to enable the corresponding body data checks and WBC telemetry. Body feedback markers use the physical configuration's kinematic model, so on a Mode 5 robot they can differ from points on the Model 13 display mesh. Neither the mesh nor a value of 5 or 13 confirms Develop mode or foot contact.

**Check both hands before control.** The viewer reads 29 body joints from `/lowstate` and seven measured finger joints per hand from `unitree_hg/msg/HandState`. It selects a fresh `/dex3/<side>/state` stream for each hand, falling back to `/lf/dex3/<side>/state`. Look for `Dex3 left live, right live` in the viewer status. A side marked `waiting` has not supplied a usable sample. A side marked `stale` has stopped supplying fresh measurements. The last visible pose may persist in TF. During live body feedback, 29, 36 and 43 published joint names correspond to zero, one and two fresh hands. [Unitree Dex3 motor order](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/g1/g1_joint_index_dds.md)

In a second terminal, run the following inside the robot container (`./scripts/g1-docker run robot`) or a shell with the Unitree ROS environment. Use the hand topics listed for your robot. Inspect publishers and incoming messages on each side.

```bash
ros2 topic list -t | grep -E 'dex3/.*/state|lowstate'
ros2 topic info /lf/dex3/left/state
ros2 topic info /lf/dex3/right/state
ros2 topic echo /g1_observer/status
```

### Mode 5 suspended arm test in Docker

Keep the hoist supporting the full robot throughout this test. The release request can remove whole-body support. Start in the suspended zero-torque condition specified for the robot firmware, provide clearance for both arms, stop other low-level control programs, and keep an operator at the physical stop. State packets and RViz cannot verify hoist support. [Unitree physical deployment procedure](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/README.md#startup-process)

Keep the hardware RViz window open. In a second terminal, enter the updated hardware container with `./scripts/g1-docker run robot`, then type `robot`. Opening the controller starts observation. The operator requests each transition from that terminal. The former `step1` command is an alias. In a development checkout, run `build` after updating the source.

| Key | Effect |
| --- | --- |
| `d`, then `y` | Confirm that the robot is suspended, in zero torque and clear. Request release of the active motion service, then check its reported state. Wait for `Develop verified. Arms idle.` |
| `1` | Capture measured arm angles and open both arms once over four seconds. The terminal reports when measured angles settle at the targets and the position hold begins. |
| **Space**, `x`, Esc or Ctrl+C | Abort the test from any phase. End the arm stream; while ownership is verified, first send disable packets. Keep supporting the arms and body. Use the independent physical stop for an emergency, lost terminal or network failure. A switch RPC already sent can still complete. |
| `r`, then `y` | While ready or holding, release the arms first. After separate confirmation, request the motion service recorded before this session and verify its name. Restoring a service can change the robot's posture. If the robot was already in Develop mode at entry, use the physical remote for recovery. |

The first three shoulder axes and elbow on each side receive the simulation Stage 1 **change** from the current pose. On the left the offsets are approximately −0.008, +0.220, +0.070 and +0.218 radians. Right shoulder roll and yaw are mirrored. Only those eight motors are enabled. Other body slots in `/lowcmd` are disabled and no Dex3 commands are published. Arm PD uses the SDK example gains (40 and 1) with a stiffness ramp. These settings and tracking limits require validation on the physical unit. The container requests serial PR coordinates and machine mode 5, checks CRC and advancing state ticks, and monitors service status, other command writers, joint feedback and command timing. The keyboard abort is a best-effort software stop. Confirm the physical state with the remote before recovery. [Unitree motion switcher](https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/utils/motion_switcher.py) · [Unitree low-level example](https://github.com/unitreerobotics/unitree_sdk2/blob/f29ee9f234851e9e79f75102c0f9e83008d8fdd1/example/g1/low_level/g1_ankle_swing_example.cpp)

### Mode 13 WBC experiment

**Enter Develop mode before a low-level control test.** The read-only viewer can run as soon as state packets arrive. For a motor-command test, support the G1 on its hoist and follow the procedure for its firmware and remote. Unitree's G1 deployment guide starts with the suspended robot in zero torque, then uses `L2 + R2` to enter Debug (Develop) mode with damping. Confirm the mode locally before starting the controller. [Unitree physical deployment instructions](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/README.md#startup-process)

**Terminal 2. Run the hardware WBC experiment on a matching Mode 13 robot.** The controller checks the reported machine mode and queries the motion service before opening its low-level command publisher. An active service or a failed query stops startup. Open the second terminal with the same `./scripts/g1-docker run robot` command. Inside the shell, type `robot --mode13`.

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
