# Unitree G1 Mode 13 Experiment Guide

The MuJoCo simulation and the physical G1 share a ROS 2 Foxy visualization interface in RViz2. Run the controller and RViz2 in separate terminals under the same user account on the same Linux host. You can open or restart RViz2 at any point during an experiment. If you observe simulation and hardware at the same time, use separate ROS domains for their observers so that topics with the same name stay separate.

## 1. MuJoCo simulation

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

Once `/lowstate` arrives, look for `G1 mode 13 | encoders live` in the status text. The model expects `mode_machine=13` and serial joint coordinates (`mode_pr=0`). RViz2 displays the 29 body joints, including the arms up to the palms. This terminal is sufficient for inspecting joint poses, IMU data, or existing `/lowcmd` messages.

**Terminal 2: Run the hardware WBC experiment.** Use the native Unitree SDK2 control terminal and enter the same network interface used by the observer. **This program checks the machine mode, releases the active motion service, and sends low-level motor commands to the physical robot.**

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
