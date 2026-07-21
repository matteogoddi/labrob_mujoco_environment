import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform
from math import ceil, floor, sqrt
from collections import defaultdict
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
import os
import io
import imageio.v2 as imageio
# import cv2


if __name__ == '__main__':
    #request input from terminal
    expNumber = input("Enter 0 to plot data from the last simulation or the number of the experiment: ")
    if expNumber == '0':
        folder = '/tmp/robot_logs'
        expType = "Simulation"
    else:
        folder = 'experiments/experiment_' + expNumber
        expType = "Experiment"

    endPlot = input("Enter the time (in seconds) at which you want to end the plots (or press Enter to plot all data): ")
    if endPlot != '':
        endPlot = int(float(endPlot) * 500)  # Assuming a control frequency of 500 Hz
    else:
        endPlot = 10

    if os.path.isdir(folder + '/robot_logs'):
        folder = folder + '/robot_logs'

    joint_names = open(folder + '/joint_names.txt').readlines()

    startPlot = 0

    fb_com_position = np.loadtxt(folder + '/com_position.txt')
    num_samples = fb_com_position.shape[0] - endPlot
    sl = slice(startPlot, num_samples)

    def _load(fname, ncols):
        p = folder + '/' + fname
        if os.path.exists(p):
            data = np.loadtxt(p)
            return data[sl, :] if data.ndim > 1 else data[sl].reshape(-1, 1)
        print(f"[INFO] {fname} not found — using zeros.")
        return np.zeros((num_samples - startPlot, ncols))

    def _load1d(fname):
        p = folder + '/' + fname
        if os.path.exists(p):
            return np.loadtxt(p)[sl]
        print(f"[INFO] {fname} not found — using zeros.")
        return np.zeros(num_samples - startPlot)

    fb_com_position = fb_com_position[sl, :]
    fb_com_velocity = _load('com_velocity.txt', 3)
    fb_zmp_position = _load('zmp_position.txt', 3)
    kf_com_position = _load('kf_com_position.txt', 3)
    kf_com_velocity = _load('kf_com_velocity.txt', 3)
    kf_zmp_position = _load('kf_zmp_position.txt', 3)
    des_com_position = _load('des_com_position.txt', 3)
    des_com_velocity = _load('des_com_velocity.txt', 3)
    des_zmp_position = _load('des_zmp_position.txt', 3)
    des_com_acceleration = _load('des_com_acceleration.txt', 3)
    current_disturbance = _load('current_disturbance.txt', 3)
    angular_momentum = _load('angular_momentum.txt', 3)
    angular_momentum_rate = _load('angular_momentum_rate.txt', 3)

    input_torque = _load('input_torque.txt', 29)
    motor_torque_filt = _load('motor_torque_filt.txt', 29)  # real-robot only (EMA-filtered motor torques)

    ef_zmp_position = _load('ef_zmp_position.txt', 3)

    p_lsole_fb = _load('p_lsole.txt', 3)
    p_rsole_fb = _load('p_rsole.txt', 3)
    v_lsole_fb = _load('v_lsole.txt', 3)
    v_rsole_fb = _load('v_rsole.txt', 3)
    p_lsole_des = _load('p_lsole_des.txt', 3)
    p_rsole_des = _load('p_rsole_des.txt', 3)
    v_lsole_des = _load('v_lsole_des.txt', 3)
    v_rsole_des = _load('v_rsole_des.txt', 3)

    fb_lsole_orientation = _load('lsole_orientation.txt', 3)
    fb_rsole_orientation = _load('rsole_orientation.txt', 3)
    des_lsole_orientation = _load('des_lsole_orientation.txt', 3)
    des_rsole_orientation = _load('des_rsole_orientation.txt', 3)

    estimated_force_lsole = _load('estimated_force_lsole.txt', 3)
    estimated_force_rsole = _load('estimated_force_rsole.txt', 3)
    wbc_accelerations = _load('wbc_accelerations.txt', 35)
    wbc_force_lsole = _load('wbc_force_lsole.txt', 6)
    wbc_force_rsole = _load('wbc_force_rsole.txt', 6)
    wbc_corner_forces_left = _load('wbc_corner_forces_left.txt', 12)
    wbc_corner_forces_right = _load('wbc_corner_forces_right.txt', 12)
    wbc_friction_coefficient = _load1d('wbc_friction_coefficient.txt')
    # Friction cone ratios (|fx|/fz, |fy|/fz) computed and logged by the WBC itself,
    # one scalar channel per foot/axis/corner (fl, fr, bl, br).
    friction_cone_ratio_left_x = np.stack([
        _load1d(f'friction_cone_ratio_left_x_{c}.txt') for c in ('fl', 'fr', 'bl', 'br')
    ], axis=1)
    friction_cone_ratio_left_y = np.stack([
        _load1d(f'friction_cone_ratio_left_y_{c}.txt') for c in ('fl', 'fr', 'bl', 'br')
    ], axis=1)
    friction_cone_ratio_right_x = np.stack([
        _load1d(f'friction_cone_ratio_right_x_{c}.txt') for c in ('fl', 'fr', 'bl', 'br')
    ], axis=1)
    friction_cone_ratio_right_y = np.stack([
        _load1d(f'friction_cone_ratio_right_y_{c}.txt') for c in ('fl', 'fr', 'bl', 'br')
    ], axis=1)
    wbc_friction_coefficient = _load1d('wbc_friction_coefficient.txt')

    # C++ side (main.cpp / main_g1.cpp) only logs a single base-frame EKF
    # estimate ("filtered_base_*"), not a separate IMU-frame one — reuse it
    # for the ekf_imu_* channels too.
    ekf_base_position = _load('filtered_base_position.txt', 3)
    ekf_base_velocity = _load('filtered_base_velocity.txt', 3)
    ekf_base_orientation = _load('filtered_base_quat.txt', 4)
    ekf_base_orientation_rpy = _load('filtered_base_rpy.txt', 3)
    ekf_base_angular_velocity = _load('filtered_base_ang_vel.txt', 3)
    ekf_imu_orientation = _load('filtered_base_quat.txt', 4)
    ekf_imu_orientation_rpy = _load('filtered_base_rpy.txt', 3)
    ekf_imu_angular_velocity = _load('filtered_base_ang_vel.txt', 3)
    ekf_joint_position = _load('filtered_joint_position.txt', 29)
    ekf_joint_velocity = _load('filtered_joint_velocity.txt', 29)

    torso_orientation = _load('torso_orientation.txt', 3)
    torso_angular_velocity = _load('torso_angular_velocity.txt', 3)
    des_torso_orientation = _load('des_torso_orientation.txt', 3)
    des_torso_angular_velocity = _load('des_torso_angular_velocity.txt', 3)

    execution_time_ekf = _load1d('execution_time_ekf.txt')
    execution_time_kf = _load1d('execution_time_kf.txt')
    execution_time_mpc = _load1d('execution_time_mpc.txt')
    execution_time_wbc = _load1d('execution_time_wbc.txt')
    execution_time_res_obs = _load1d('execution_time_res_obs.txt')
    execution_time_hac = _load1d('execution_time_hac.txt')
    execution_time_coop_planner = _load1d('execution_time_coop_planner.txt')
    execution_time_update = _load1d('execution_time_update.txt')

    odometry_base_position = _load('odom_pos.txt', 3)
    odometry_base_velocity = _load('odom_vel.txt', 3)
    odometry_imu_orientation = _load('odom_quat.txt', 4)
    odometry_imu_orientation_rpy = _load('odom_rpy.txt', 3)
    measured_joint_position = _load('joint_pos.txt', 29)
    measured_joint_velocity = _load('joint_vel.txt', 29)
    # Not currently logged anywhere in the C++ side — stays zero until a
    # "measured_joint_torque" channel is added to sensor_logger.
    measured_joint_torque = _load('measured_joint_torque.txt', 29)
    # Only main_g1.cpp logs pelvis_quat/pelvis_rpy (main.cpp doesn't) — will
    # still read as zero when plotting logs produced by main.cpp.
    measured_imu_orientation = _load('pelvis_quat.txt', 4)
    measured_imu_orientation_rpy = _load('pelvis_rpy.txt', 3)
    measured_imu_angular_velocity = _load('pelvis_gyro.txt', 3)
    measured_imu_accelerometer = _load('pelvis_acc.txt', 3)
        
    # rotate relative positions depending on the actual yaw angle
    for i in range(num_samples - startPlot):
        yaw = odometry_imu_orientation_rpy[0, 2]
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])
        p_lsole_fb[i, :] = rotation_matrix.T @ p_lsole_fb[i, :]
        p_rsole_fb[i, :] = rotation_matrix.T @ p_rsole_fb[i, :]
        p_lsole_des[i, :] = rotation_matrix.T @ p_lsole_des[i, :]
        p_rsole_des[i, :] = rotation_matrix.T @ p_rsole_des[i, :]
        kf_com_position[i, :] = rotation_matrix.T @ kf_com_position[i, :]
        kf_zmp_position[i, :] = rotation_matrix.T @ kf_zmp_position[i, :]
        des_com_position[i, :] = rotation_matrix.T @ des_com_position[i, :]
        des_zmp_position[i, :] = rotation_matrix.T @ des_zmp_position[i, :]


    reference_positions = np.array([
        -0.44,  # l_hip_p
        0.04,  # l_hip_r
        0.0,  # l_hip_y
        0.95,  # l_knee
        -0.50,  # l_ankle_p
        0.00,  # l_ankle_r
        -0.44,  # r_hip_p
        -0.04,  # r_hip_r
        0.0,  # r_hip_y
        0.95,  # r_knee
        -0.50,  # r_ankle_p
        0.00,  # r_ankle_r
        0.0,  # waist_y
        0.07,  # l_shoulder_p
        0.25,  # l_shoulder_r
        0.0,  # l_shoulder_y
        3.14 / 2.0 - 0.44,   # l_elbow_p
        0.0, # wrist_roll
        0.0, # wrist_pitch
        0.0, # wrist_yaw
        0.07,  # r_shoulder_p
        -0.25,  # r_shoulder_r
        0.0,  # r_shoulder_y
        3.14 / 2.0 - 0.44,  # r_elbow_p
        0.0, # wrist_roll
        0.0, # wrist_pitch
        0.0 # wrist_yaw
    ])

    delta = 1 / 500  # Assuming a control frequency of 500 Hz
    t = np.linspace(0.0, delta * (num_samples - startPlot), num_samples - startPlot)
    # num_joints = 27
    num_joints = 29


    if not os.path.exists('images/feedback/joints/positions'):
        os.makedirs('images/feedback/joints/positions')
    if not os.path.exists('images/feedback/joints/velocities'):
        os.makedirs('images/feedback/joints/velocities')
    if not os.path.exists('images/feedback/base'):
        os.makedirs('images/feedback/base')
    if not os.path.exists('images/ekf/joints/positions'):
        os.makedirs('images/ekf/joints/positions')
    if not os.path.exists('images/ekf/joints/velocities'):
        os.makedirs('images/ekf/joints/velocities')
    if not os.path.exists('images/ekf/joints/error/positions'):
        os.makedirs('images/ekf/joints/error/positions')
    if not os.path.exists('images/ekf/joints/error/velocities'):
        os.makedirs('images/ekf/joints/error/velocities')
    if not os.path.exists('images/ekf/base/errors'):
        os.makedirs('images/ekf/base/errors')
    if not os.path.exists('images/ekf/performance'):
        os.makedirs('images/ekf/performance')
    if not os.path.exists('images/execution_times'):
        os.makedirs('images/execution_times')
    if not os.path.exists('images/com/references'):
        os.makedirs('images/com/references')
    if not os.path.exists('images/com/errors'):
        os.makedirs('images/com/errors')
    if not os.path.exists('images/wbc_solutions/wbc_joint_accelerations'):
        os.makedirs('images/wbc_solutions/wbc_joint_accelerations')
    if not os.path.exists('images/wbc_solutions/wbc_base_accelerations'):
        os.makedirs('images/wbc_solutions/wbc_base_accelerations')
    if not os.path.exists('images/wbc_solutions/wbc_joint_torques_ffw'):
        os.makedirs('images/wbc_solutions/wbc_joint_torques_ffw')
    if not os.path.exists('images/wbc_solutions/wbc_sole_forces'):
        os.makedirs('images/wbc_solutions/wbc_sole_forces')
    if not os.path.exists('images/wbc_solutions/friction_cone'):
        os.makedirs('images/wbc_solutions/friction_cone')
    if not os.path.exists('images/wrench_estimations/sole_wrenches'):
        os.makedirs('images/wrench_estimations/sole_wrenches')
    if not os.path.exists('images/soles/references'):
        os.makedirs('images/soles/references')
    if not os.path.exists('images/soles/errors'):
        os.makedirs('images/soles/errors')
    if not os.path.exists('images/mpc'):
        os.makedirs('images/mpc')
    if not os.path.exists('images/feedback/motor_torques'):
        os.makedirs('images/feedback/motor_torques')
    if not os.path.exists('images/residuals/right_arm'):
        os.makedirs('images/residuals/right_arm')
    if not os.path.exists('images/residuals/left_arm'):
        os.makedirs('images/residuals/left_arm')
    if not os.path.exists('images/tau_g/right_arm'):
        os.makedirs('images/tau_g/right_arm')
    if not os.path.exists('images/tau_g/left_arm'):
        os.makedirs('images/tau_g/left_arm')

    grouped_indices = defaultdict(list)

    for idx, name in enumerate(joint_names):
        base_name = '_'.join(name.split('_')[:2])  # E.g., "left_ankle" da "left_ankle_roll_joint"
        grouped_indices[base_name].append(idx)

    #################################
    # MPC PREDICTED TRAJECTORIES
    #################################
    #disabilita per ora
    if False:
        frames = []
        mpc_horizon = 20              # number of predicted MPC points
        mpc_dt = 0.1                  # MPC sample time (0.1 s)
        prediction_window = mpc_horizon * mpc_dt   # should be 2 seconds

        num_blocks = num_samples

        for block in range(0, num_blocks, 50):
            # Index range of the predictions in mpc_pred_com_pos
            start = block * mpc_horizon
            end = start + mpc_horizon

            # Compute the real time at which this block starts
            t0 = t[block]

            # Build the MPC prediction timeline (20 points over exactly 2 seconds)
            t_pred = np.linspace(t0, t0 + prediction_window, mpc_horizon)

            fig, ax = plt.subplots()

            # Plot MPC predicted COM (using uniform 2-second time axis)
            ax.plot(t_pred, mpc_pred_com_pos[start:end, 0],
                    label='MPC Pred COM X', linestyle='--')
            ax.plot(t_pred, mpc_pred_com_pos[start:end, 1],
                    label='MPC Pred COM Y', linestyle='--')
            ax.plot(t_pred, mpc_pred_com_pos[start:end, 2],
                    label='MPC Pred COM Z', linestyle='--')

            ax.set_xlabel('Time [s]')
            ax.set_ylabel('COM Position [m]')
            ax.set_title(f'MPC Predicted COM Position (Block {block})')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()

            # Convert figure to image frame
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)

        # Save GIF
        # imageio.mimsave('images/mpc/mpc_com_prediction.gif', frames, duration=0.2)

        with imageio.get_writer('images/mpc/mpc_pred_com_pos.mp4', fps=5) as writer:
            for frame in frames:
                writer.append_data(frame)

        frames = []
        mpc_horizon = 20              # number of predicted MPC points
        mpc_dt = 0.1                  # MPC sample time (0.1 s)
        prediction_window = mpc_horizon * mpc_dt   # should be 2 seconds

        num_blocks = num_samples

        for block in range(0, num_blocks, 50):
            # Index range of the predictions in mpc_pred_com_pos
            start = block * mpc_horizon
            end = start + mpc_horizon

            # Compute the real time at which this block starts
            t0 = t[block]

            # Build the MPC prediction timeline (20 points over exactly 2 seconds)
            t_pred = np.linspace(t0, t0 + prediction_window, mpc_horizon)

            fig, ax = plt.subplots()

            # Plot MPC predicted COM (using uniform 2-second time axis)
            ax.plot(t_pred, mpc_pred_com_vel[start:end, 0],
                    label='MPC Pred COM Vel X', linestyle='--')
            ax.plot(t_pred, mpc_pred_com_vel[start:end, 1],
                    label='MPC Pred COM Vel Y', linestyle='--')
            ax.plot(t_pred, mpc_pred_com_vel[start:end, 2],
                    label='MPC Pred COM Vel Z', linestyle='--')

            ax.set_xlabel('Time [s]')
            ax.set_ylabel('COM Velocity [m]')
            ax.set_title(f'MPC Predicted COM Velocity (Block {block})')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()

            # Convert figure to image frame
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)

        # Save GIF
        # imageio.mimsave('images/mpc/mpc_com_prediction.gif', frames, duration=0.2)

        with imageio.get_writer('images/mpc/mpc_pred_com_vel.mp4', fps=5) as writer:
            for frame in frames:
                writer.append_data(frame)

        frames = []
        mpc_horizon = 20              # number of predicted MPC points
        mpc_dt = 0.1                  # MPC sample time (0.1 s)
        prediction_window = mpc_horizon * mpc_dt   # should be 2 seconds

        num_blocks = num_samples

        for block in range(0, num_blocks, 50):
            # Index range of the predictions in mpc_pred_com_pos
            start = block * mpc_horizon
            end = start + mpc_horizon

            # Compute the real time at which this block starts
            t0 = t[block]

            # Build the MPC prediction timeline (20 points over exactly 2 seconds)
            t_pred = np.linspace(t0, t0 + prediction_window, mpc_horizon)

            fig, ax = plt.subplots()

            # Plot MPC predicted ZMP (using uniform 2-second time axis)
            ax.plot(t_pred, mpc_pred_zmp_pos[start:end, 0],
                    label='MPC Pred ZMP X', linestyle='--')
            ax.plot(t_pred, mpc_pred_zmp_pos[start:end, 1],
                    label='MPC Pred ZMP Y', linestyle='--')
            ax.plot(t_pred, mpc_pred_zmp_pos[start:end, 2],
                    label='MPC Pred ZMP Z', linestyle='--')

            ax.set_xlabel('Time [s]')
            ax.set_ylabel('ZMP Position [m]')
            ax.set_title(f'MPC Predicted ZMP Position (Block {block})')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()

            # Convert figure to image frame
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)

        # Save GIF
        # imageio.mimsave('images/mpc/mpc_com_prediction.gif', frames, duration=0.2)

        with imageio.get_writer('images/mpc/mpc_pred_zmp_pos.mp4', fps=5) as writer:
            for frame in frames:
                writer.append_data(frame)



    


    #################################
    # JOINT TORQUES & ESTIMATED FORCES ON SOLES
    #################################

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, input_torque[:, i], label=joint_names[i].strip())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Torque [Nm]')
        ax.set_title(f'Input Joint Torques - {group_name}')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"images/wbc_solutions/wbc_joint_torques_ffw/{group_name}_input_joint_torques.png")
        plt.close(fig)
        figs.append(fig)

    fig, ax = plt.subplots()
    ax.plot(t, wbc_accelerations[:, 0], label='Acceleration X', color='blue')
    ax.plot(t, wbc_accelerations[:, 1], label='Acceleration Y', color='orange')
    ax.plot(t, wbc_accelerations[:, 2], label='Acceleration Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Acceleration [N]')
    ax.set_title('Acceleration base')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wbc_solutions/wbc_base_accelerations/linear_acceleration.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, wbc_accelerations[:, 3], label='Acceleration X', color='blue')
    ax.plot(t, wbc_accelerations[:, 4], label='Acceleration Y', color='orange')
    ax.plot(t, wbc_accelerations[:, 5], label='Acceleration Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Acceleration [N]')
    ax.set_title('Acceleration base')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wbc_solutions/wbc_base_accelerations/angular_acceleration.png")
    plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, wbc_accelerations[:, i + 6], label=joint_names[i].strip())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Acceleration [rad/s^2]')
        ax.set_title(f'WBC Acceleration - {group_name}')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"images/wbc_solutions/wbc_joint_accelerations/{group_name}_acceleration.png")
        plt.close(fig)
        figs.append(fig)

    wbc_wrench_labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    fig, ax = plt.subplots()
    for i, label in enumerate(wbc_wrench_labels):
        ax.plot(t, wbc_force_lsole[:, i], label=label)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N] / Torque [Nm]')
    ax.set_title('WBC Optimal Left Foot Wrench')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wbc_solutions/wbc_sole_forces/wbc_force_left_sole.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for i, label in enumerate(wbc_wrench_labels):
        ax.plot(t, wbc_force_rsole[:, i], label=label)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N] / Torque [Nm]')
    ax.set_title('WBC Optimal Right Foot Wrench')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wbc_solutions/wbc_sole_forces/wbc_force_right_sole.png")
    plt.close(fig)

    # WBC corner (contact-point) forces, one figure per foot, one subplot per force component
    corner_labels = ['Front-Left', 'Front-Right', 'Back-Left', 'Back-Right']
    corner_components = ['Fx', 'Fy', 'Fz']
    n_corners = len(corner_labels)

    def _plot_corner_wrenches(data, foot_name, filename):
        fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
        for comp_idx, ax in enumerate(axes):
            for corner_idx, corner_label in enumerate(corner_labels):
                ax.plot(
                    t, data[:, 3 * corner_idx + comp_idx],
                    label=corner_label, linewidth=2.0
                )
            ax.set_ylabel(f'{corner_components[comp_idx]} [N]', fontsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(loc='best', frameon=True, fontsize=9)
            ax.tick_params(axis='both', labelsize=9)
        axes[-1].set_xlabel('Time [s]', fontsize=11)
        fig.suptitle(f'WBC Corner Forces - {foot_name} Sole', fontsize=12)
        fig.tight_layout()
        fig.savefig(f"images/wbc_solutions/wbc_sole_forces/{filename}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    _plot_corner_wrenches(wbc_corner_forces_left, 'Left', 'wbc_corner_forces_left')
    _plot_corner_wrenches(wbc_corner_forces_right, 'Right', 'wbc_corner_forces_right')

    # Friction cone check: |fx|/fz and |fy|/fz ratios vs +-mu for each of the 4 corners.
    # The linearized friction pyramid used in the WBC enforces |fx| <= mu*fz and
    # |fy| <= mu*fz, so the ratios must stay within the +-mu band. Ratios are computed
    # and logged directly by the WBC (WalkingManager::frictionConeRatios).
    def _plot_friction_cone(ratio_x, ratio_y, foot_name, filename):
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        for corner_idx, corner_label in enumerate(corner_labels):
            axes[0].plot(t, ratio_x[:, corner_idx], label=corner_label, linewidth=1.5)
            axes[1].plot(t, ratio_y[:, corner_idx], label=corner_label, linewidth=1.5)
        for ax, ratio_name in zip(axes, ['|fx| / fz', '|fy| / fz']):
            ax.plot(t, wbc_friction_coefficient, color='red', linestyle='--', linewidth=1.5, label=r'$\mu$')
            ax.set_ylabel(ratio_name, fontsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(loc='best', frameon=True, fontsize=9)
            ax.tick_params(axis='both', labelsize=9)
        axes[-1].set_xlabel('Time [s]', fontsize=11)
        fig.suptitle(f'Friction Cone Constraint - {foot_name} Sole', fontsize=12)
        fig.tight_layout()
        fig.savefig(f"images/wbc_solutions/friction_cone/{filename}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    _plot_friction_cone(friction_cone_ratio_left_x, friction_cone_ratio_left_y, 'Left', 'friction_cone_left')
    _plot_friction_cone(friction_cone_ratio_right_x, friction_cone_ratio_right_y, 'Right', 'friction_cone_right')

    fig, ax = plt.subplots()
    ax.plot(t, estimated_force_lsole[:, 0], label='Estimated Force Left Sole X', color='blue')
    ax.plot(t, estimated_force_lsole[:, 1], label='Estimated Force Left Sole Y', color='orange')
    ax.plot(t, estimated_force_lsole[:, 2], label='Estimated Force Left Sole Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Estimated Force [N]')
    ax.set_title('Estimated Forces on Left Sole')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wrench_estimations/sole_wrenches/estimated_force_left_sole.png")
    plt.close(fig)

    #plot estimated forces on right sole
    fig, ax = plt.subplots()
    ax.plot(t, estimated_force_rsole[:, 0], label='Estimated Force Right Sole X', color='blue')
    ax.plot(t, estimated_force_rsole[:, 1], label='Estimated Force Right Sole Y', color='orange')
    ax.plot(t, estimated_force_rsole[:, 2], label='Estimated Force Right Sole Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Estimated Force [N]')
    ax.set_title('Estimated Forces on Right Sole')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wrench_estimations/sole_wrenches/estimated_force_right_sole.png")
    plt.close(fig)
    


    #################################
    #  COM AND ZMP PLOTS
    #################################

    # REFERENCES PLOTS

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_acceleration[:, 0],
        label=r'Desired CoM Acceleration $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_acceleration[:, 1],
        label=r'Desired CoM Acceleration $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_acceleration[:, 2],
        label=r'Desired CoM Acceleration $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Acceleration [$\mathrm{m/s^2}$]', fontsize=11)
    ax.set_title('Desired Center of Mass Acceleration', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/references/des_com_acceleration_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, current_disturbance[:, 0],
        label=r'Disturbance $x$',
        linewidth=2.0
    )
    ax.plot(
        t, current_disturbance[:, 1],
        label=r'Disturbance $y$',
        linewidth=2.0
    )
    ax.plot(
        t, current_disturbance[:, 2],
        label=r'Disturbance $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Disturbance [$\mathrm{m/s^2}$]', fontsize=11)
    ax.set_title('PLIP Disturbance Term (before integration)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/references/current_disturbance_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_position[:, 0] - des_com_position[0, 0],
        label=r'Desired CoM Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_position[:, 1] - des_com_position[0, 1],
        label=r'Desired CoM Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_position[:, 2],
        label=r'Desired CoM Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Desired Center of Mass Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/references/des_com_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_velocity[:, 0],
        label=r'Desired CoM Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_velocity[:, 1],
        label=r'Desired CoM Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_velocity[:, 2],
        label=r'Desired CoM Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Desired Center of Mass Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/references/des_com_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_zmp_position[:, 0] - des_zmp_position[0, 0],
        label=r'Desired ZMP Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_zmp_position[:, 1] - des_zmp_position[0, 1],
        label=r'Desired ZMP Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_zmp_position[:, 2],
        label=r'Desired ZMP Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Desired Zero Moment Point Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/references/des_zmp_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)



    # ERROR PLOTS

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_zmp_position[:,0] - kf_zmp_position[:, 0],
        label=r'ZMP Position Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_zmp_position[:, 1] - kf_zmp_position[:, 1],
        label=r'ZMP Position Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_zmp_position[:, 2] - kf_zmp_position[:, 2],
        label=r'ZMP Position Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Zero Moment Point Position Error', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/errors/error_zmp_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_position[:,0] - kf_com_position[:, 0],
        label=r'CoM Position Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_position[:, 1] - kf_com_position[:, 1],
        label=r'CoM Position Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_position[:, 2] - kf_com_position[:, 2],
        label=r'CoM Position Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Center of Mass Position Error', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/errors/error_com_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_velocity[:,0] - kf_com_velocity[:, 0],
        label=r'CoM Velocity Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_velocity[:, 1] - kf_com_velocity[:, 1],
        label=r'CoM Velocity Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_velocity[:, 2] - kf_com_velocity[:, 2],
        label=r'CoM Velocity Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Center of Mass Velocity Error', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/errors/error_com_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_zmp_position[:, 0] - des_zmp_position[0, 0],
        label=r'Desired ZMP Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_zmp_position[:, 0] - kf_zmp_position[0, 0],
        label=r'Actual ZMP Position $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_zmp_position[:, 1] - des_zmp_position[0, 1],
        label=r'Desired ZMP Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_zmp_position[:, 1] - kf_zmp_position[0, 1],
        label=r'Actual ZMP Position $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_zmp_position[:, 2],
        label=r'Desired ZMP Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_zmp_position[:, 2],
        label=r'Actual ZMP Position $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Comparison between reference and actual Zero Moment Point Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/errors/comparison_zmp_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_position[:,0] - des_com_position[0, 0],
        label=r'Desired CoM Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_com_position[:,0] - kf_com_position[0, 0],
        label=r'Actual CoM Position $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_com_position[:, 1] - des_com_position[0, 1],
        label=r'Desired CoM Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_com_position[:, 1] - kf_com_position[0, 1],
        label=r'Actual CoM Position $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_com_position[:, 2],
        label=r'Desired CoM Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_com_position[:, 2],
        label=r'Actual CoM Position $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Comparison between reference and actual Center of Mass Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/errors/comparison_com_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_velocity[:,0],
        label=r'Desired CoM Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_com_velocity[:,0],
        label=r'Actual CoM Velocity $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_com_velocity[:, 1],
        label=r'Desired CoM Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_velocity[:, 1],
        label=r'Actual CoM Velocity $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_com_velocity[:, 2],
        label=r'Desired CoM Velocity $z$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_com_velocity[:, 2],
        label=r'Actual CoM Velocity $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Comparison between reference and actual Center of Mass Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/errors/comparison_com_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, kf_com_position[:, 0] - kf_com_position[0, 0],
        label=r'CoM Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_zmp_position[:, 0] - kf_zmp_position[0, 0],
        label=r'ZMP Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_fb[:, 0] - p_lsole_fb[0, 0],
        label=r'Left Foot Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_fb[:, 0] - p_rsole_fb[0, 0],
        label=r'Right Foot Position $x$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position $x$ [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Motion in the forward direction', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/motion_x.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, kf_com_position[:, 1],
        label=r'CoM Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_zmp_position[:, 1],
        label=r'ZMP Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_fb[:, 1],
        label=r'Left Foot Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_fb[:, 1],
        label=r'Right Foot Position $y$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position $y$ [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Motion in the lateral direction', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='upper left',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/motion_y.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)
    

    fig, ax = plt.subplots()
    ax.plot(t, ef_zmp_position[:, 0], label='residual based ZMP X', color='blue')
    ax.plot(t, ef_zmp_position[:, 1], label='residual based ZMP Y', color='orange')
    ax.plot(t, ef_zmp_position[:, 2], label='residual based ZMP Z', color='green')
    ax.plot(t, fb_zmp_position[:, 0], label='lip based ZMP X', color='blue', linestyle='--')
    ax.plot(t, fb_zmp_position[:, 1], label='lip based ZMP Y', color='orange', linestyle='--')
    ax.plot(t, fb_zmp_position[:, 2], label='lip based ZMP Z', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('ZMP Position Feedback: LIP-based vs Residual-based')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/zmp_lip_vs_residual_plot.png")

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    axis_labels = ['x', 'y', 'z']
    for i, ax in enumerate(axes):
        ax_rate = ax.twinx()
        l1, = ax.plot(
            t, angular_momentum[:, i],
            label=r'Angular Momentum $%s$' % axis_labels[i],
            color='blue', linewidth=2.0
        )
        l2, = ax_rate.plot(
            t, angular_momentum_rate[:, i],
            label=r'Angular Momentum Rate $%s$' % axis_labels[i],
            color='orange', linewidth=2.0, linestyle='--'
        )
        ax.set_ylabel(r'$L_%s$ [$\mathrm{kg\,m^2/s}$]' % axis_labels[i], fontsize=10)
        ax_rate.set_ylabel(r'$\dot{L}_%s$ [$\mathrm{kg\,m^2/s^2}$]' % axis_labels[i], fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(handles=[l1, l2], loc='best', frameon=True, fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        ax_rate.tick_params(axis='both', labelsize=9)
    axes[-1].set_xlabel('Time [s]', fontsize=11)
    fig.suptitle('Centroidal Angular Momentum and its Rate of Change', fontsize=12)
    fig.tight_layout()
    fig.savefig(
        "images/com/angular_momentum_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


    ##########################
    #  FEET PLOT
    ##########################

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, p_lsole_des[:, 0],
        label=r'Desired Left Sole Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 1],
        label=r'Desired Left Sole Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 2],
        label=r'Desired Left Sole Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Desired Left Sole Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/references/desired_left_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, p_lsole_des[:, 0],
        label=r'Desired Right Sole Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 1],
        label=r'Desired Right Sole Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 2],
        label=r'Desired Right Sole Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Desired Right Sole Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/references/desired_right_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, p_lsole_des[:, 0] - p_lsole_fb[:, 0],
        label=r'Left Sole Position Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 1] - p_lsole_fb[:, 1],
        label=r'Left Sole Position Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 2] - p_lsole_fb[:, 2],
        label=r'Left Sole Position Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Error between Desired and Actual Left Sole Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/error_left_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, p_rsole_des[:, 0] - p_rsole_fb[:, 0],
        label=r'Right Sole Position Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_des[:, 1] - p_rsole_fb[:, 1],
        label=r'Right Sole Position Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_des[:, 2] - p_rsole_fb[:, 2],
        label=r'Right Sole Position Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Error between Desired and Actual Right Sole Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/error_right_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, p_lsole_fb[:, 0] - p_lsole_fb[0, 0],
        label=r'Actual Left Sole Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 0] - p_lsole_des[0, 0],
        label=r'Desired Left Sole Position $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, p_lsole_fb[:, 1],
        label=r'Actual Left Sole Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 1],
        label=r'Desired Left Sole Position $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, p_lsole_fb[:, 2],
        label=r'Actual Left Sole Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 2],
        label=r'Desired Left Sole Position $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Comparison between Desired and Actual Left Sole Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/comparison_left_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, p_rsole_fb[:, 0] - p_rsole_fb[0, 0],
        label=r'Actual Right Sole Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_des[:, 0] - p_rsole_des[0, 0],
        label=r'Desired Right Sole Position $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, p_rsole_fb[:, 1],
        label=r'Actual Right Sole Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_des[:, 1],
        label=r'Desired Right Sole Position $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, p_rsole_fb[:, 2],
        label=r'Actual Right Sole Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_des[:, 2],
        label=r'Desired Right Sole Position $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Comparison between Desired and Actual Right Sole Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/comparison_right_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, v_lsole_des[:, 0] - v_lsole_fb[:, 0],
        label=r'Left Sole Velocity Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, v_lsole_des[:, 1] - v_lsole_fb[:, 1],
        label=r'Left Sole Velocity Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, v_lsole_des[:, 2] - v_lsole_fb[:, 2],
        label=r'Left Sole Velocity Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Error between Desired and Actual Left Sole Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/error_left_sole_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, v_rsole_des[:, 0] - v_rsole_fb[:, 0],
        label=r'Right Sole Velocity Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, v_rsole_des[:, 1] - v_rsole_fb[:, 1],
        label=r'Right Sole Velocity Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, v_rsole_des[:, 2] - v_rsole_fb[:, 2],
        label=r'Right Sole Velocity Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Error between Desired and Actual Right Sole Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/error_right_sole_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, v_lsole_fb[:, 0],
        label=r'Actual Left Sole Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, v_lsole_des[:, 0],
        label=r'Desired Left Sole Velocity $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, v_lsole_fb[:, 1],
        label=r'Actual Left Sole Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, v_lsole_des[:, 1],
        label=r'Desired Left Sole Velocity $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, v_lsole_fb[:, 2],
        label=r'Actual Left Sole Velocity $z$',
        linewidth=2.0
    )
    ax.plot(
        t, v_lsole_des[:, 2],
        label=r'Desired Left Sole Velocity $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Comparison between Desired and Actual Left Sole Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/comparison_left_sole_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, v_rsole_fb[:, 0],
        label=r'Actual Right Sole Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, v_rsole_des[:, 0],
        label=r'Desired Right Sole Velocity $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, v_rsole_fb[:, 1],
        label=r'Actual Right Sole Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, v_rsole_des[:, 1],
        label=r'Desired Right Sole Velocity $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, v_rsole_fb[:, 2],
        label=r'Actual Right Sole Velocity $z$',
        linewidth=2.0
    )
    ax.plot(
        t, v_rsole_des[:, 2],
        label=r'Desired Right Sole Velocity $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Comparison between Desired and Actual Right Sole Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/comparison_right_sole_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)



    ##########################
    #  EKF PLOTS
    ##########################

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_position[:, i],
                label=r'Measured Position'+ f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0
            )
            ax.plot(
                t, ekf_joint_position[:, i],
                label=r'Filtered Position' + f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0,
                linestyle='--'
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=12)
        ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=7
        )
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/ekf/joints/positions/{group_name}_position_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_velocity[:, i],
                label=r'Measured Velocity'+ f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0
            )
            ax.plot(
                t, ekf_joint_velocity[:, i],
                label=r'Filtered Velocity' + f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0,
                linestyle='--'
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Velocity [$\mathrm{rad/s}$]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=12)
        ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=7
        )
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/ekf/joints/velocities/{group_name}_velocity_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_position[:, i] - ekf_joint_position[:, i],
                label=r'Error Position'+ f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=12)
        ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=7
        )
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/ekf/joints/error/positions/error_{group_name}_position_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_velocity[:, i] - ekf_joint_velocity[:, i],
                label=r'Error Velocity'+ f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Velocity [$\mathrm{rad/s}$]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=12)
        ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=7
        )
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/ekf/joints/error/velocities/error_{group_name}_velocity_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    colormap = plt.colormaps['tab10'] 
    line_styles = ['-', '--', '-.', ':']
    for i in range(num_joints):
        color = colormap(i % 10)
        linestyle = line_styles[(i // 10) % len(line_styles)]  # cambia stile ogni 10 joint
        error = ekf_joint_position[:, i] - measured_joint_position[:, i]
        ax.plot(t, error,
                label=joint_names[i].strip(),
                color=color,
                linestyle=linestyle,
                linewidth=2)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{rad}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured Joints Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=4
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/joints/error/error_joint_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    colormap = plt.colormaps['tab10'] 
    line_styles = ['-', '--', '-.', ':']
    for i in range(num_joints):
        color = colormap(i % 10)
        linestyle = line_styles[(i // 10) % len(line_styles)]  # cambia stile ogni 10 joint
        error = ekf_joint_velocity[:, i] - measured_joint_velocity[:, i]
        ax.plot(t, error,
                label=joint_names[i].strip(),
                color=color,
                linestyle=linestyle,
                linewidth=2)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{rad/s}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured Joints Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=4
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/joints/error/error_joint_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_position[:, 0],
        label=r'EKF Base Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_position[:, 1],
        label=r'EKF Base Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_position[:, 2],
        label=r'EKF Base Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('EKF Base Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/base_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_velocity[:, 0],
        label=r'EKF Base Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_velocity[:, 1],
        label=r'EKF Base Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_velocity[:, 2],
        label=r'EKF Base Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('EKF Base Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/base_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_orientation[:, 0],
        label=r'EKF Base Orientation $W$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_orientation[:, 1],
        label=r'EKF Base Orientation $X$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_orientation[:, 2],
        label=r'EKF Base Orientation $Y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_orientation[:, 3],
        label=r'EKF Base Orientation $Z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{quat}$]', fontsize=11)
    ax.set_title('EKF Base Orientation Quat', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/base_orientation_quat_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_orientation_rpy[:, 0],
        label=r'EKF Base Orientation $R$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_orientation_rpy[:, 1],
        label=r'EKF Base Orientation $P$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_orientation_rpy[:, 2],
        label=r'EKF Base Orientation $Y$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{grad}$]', fontsize=11)
    ax.set_title('EKF Base Orientation RPY', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/base_orientation_rpy_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_angular_velocity[:, 0],
        label=r'EKF Base Angular Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_angular_velocity[:, 1],
        label=r'EKF Base Angular Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_angular_velocity[:, 2],
        label=r'EKF Base Angular Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('EKF Base Angular Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/base_angular_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


    # plot mean squared error between ekf joint position and simulated joint position
    mse_position = np.mean((ekf_joint_position - measured_joint_position) ** 2, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_joints), mse_position, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(r'Mean Squared Error', fontsize=14)
    ax.set_title('Mean Squared Error between EKF Joint Position and Feedback Joint Position', fontsize=16)
    ax.set_xticks(range(num_joints))
    ax.set_xticklabels([name.strip().replace("_"," ").replace("joint", "") for name in joint_names], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig("images/ekf/performance/mse_joint_position_plot.png")
    plt.close(fig)

    #plot mean squared error between ekf joint velocity and simulated joint velocity
    mse_velocity = np.mean((ekf_joint_velocity - measured_joint_velocity) ** 2, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_joints), mse_velocity, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(r'Mean Squared Error', fontsize=14)
    ax.set_title('Mean Squared Error between EKF Joint Velocity and Feedback Joint Velocity', fontsize=16)
    ax.set_xticks(range(num_joints))
    ax.set_xticklabels([name.strip().replace("_"," ").replace("joint", "") for name in joint_names], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig("images/ekf/performance/mse_joint_velocity_plot.png")
    plt.close(fig)

    #plot mean squared error between ekf base position and simulated base position, orientation, velocity, angular velocity
    mse_base_position = np.mean((ekf_base_position - odometry_base_position) ** 2, axis=0)
    mse_base_velocity = np.mean((ekf_base_velocity - odometry_base_velocity) ** 2, axis=0)
    mse_base_orientation = np.mean((ekf_base_orientation - odometry_imu_orientation) ** 2, axis=0)
    mse_base_angular_velocity = np.mean((ekf_base_angular_velocity - measured_imu_angular_velocity) ** 2, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(3), mse_base_position, label='Position MSE', color='skyblue', alpha=0.7)
    ax.bar(range(3, 6), mse_base_velocity, label='Velocity MSE', color='orange', alpha=0.7)
    ax.bar(range(6, 10), mse_base_orientation, label='Orientation MSE', color='green', alpha=0.7)
    ax.bar(range(10, 13), mse_base_angular_velocity, label='Angular Velocity MSE', color='red', alpha=0.7)
    ax.set_xlabel('Base State Index', fontsize=14)
    ax.set_ylabel('Mean Squared Error', fontsize=14)
    ax.set_title('Mean Squared Error between EKF Base States and Simulated Base States', fontsize=16)
    ax.set_xticks(range(13))
    ax.set_xticklabels(['Position X', 'Position Y', 'Position Z', 'Velocity X', 'Velocity Y', 'Velocity Z', 'Orientation W', 'Orientation X', 'Orientation Y', 'Orientation Z',
                        'Angular Velocity X', 'Angular Velocity Y', 'Angular Velocity Z'], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/performance/mse_base_states_plot.png")
    plt.close(fig)

    #plot variance between ekf joint position and simulated joint position
    variance_position = np.var(ekf_joint_position - measured_joint_position, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_joints), variance_position, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(r'Variance', fontsize=14)
    ax.set_title('Variance between EKF Joint Position and Feedback Joint Position', fontsize=16)
    ax.set_xticks(range(num_joints))
    ax.set_xticklabels([name.strip().replace("_"," ").replace("joint", "") for name in joint_names], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig("images/ekf/performance/var_joint_position_plot.png")
    plt.close(fig)

    #plot variance between ekf joint position and simulated joint position
    variance_velocity = np.var(ekf_joint_velocity - measured_joint_velocity, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_joints), variance_velocity, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(r'Variance', fontsize=14)
    ax.set_title('Variance between EKF Joint Velocity and Feedback Joint Velocity', fontsize=16)
    ax.set_xticks(range(num_joints))
    ax.set_xticklabels([name.strip().replace("_"," ").replace("joint", "") for name in joint_names], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig("images/ekf/performance/var_joint_velocity_plot.png")
    plt.close(fig)

    #plot variance between ekf base position and simulated base position, orientation, velocity, angular velocity
    variance_base_position = np.var(ekf_base_position - odometry_base_position, axis=0)
    variance_base_velocity = np.var(ekf_base_velocity - odometry_base_velocity, axis=0)
    variance_base_orientation = np.var(ekf_base_orientation - odometry_imu_orientation, axis=0)
    variance_base_angular_velocity = np.var(ekf_base_angular_velocity - measured_imu_angular_velocity, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(3), variance_base_position, label='Position Variance', color='skyblue', alpha=0.7)
    ax.bar(range(3, 6), variance_base_velocity, label='Velocity Variance', color='orange', alpha=0.7)
    ax.bar(range(6, 10), variance_base_orientation, label='Orientation Variance', color='green', alpha=0.7)
    ax.bar(range(10, 13), variance_base_angular_velocity, label='Angular Velocity Variance', color='red', alpha=0.7)
    ax.set_xlabel('Base State Index', fontsize=14)
    ax.set_ylabel('Variance', fontsize=14)
    ax.set_title('Variance between EKF Base States and Simulated Base States', fontsize=16)
    ax.set_xticks(range(13))
    ax.set_xticklabels(['Position X', 'Position Y', 'Position Z', 'Velocity X', 'Velocity Y', 'Velocity Z', 'Orientation W', 'Orientation X', 'Orientation Y', 'Orientation Z',
                        'Angular Velocity X', 'Angular Velocity Y', 'Angular Velocity Z'], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/performance/var_base_states_plot.png")
    plt.close(fig)

    #plot variance of measured joint velocity
    variance_measured_velocity = np.var(measured_joint_velocity, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_joints), variance_measured_velocity, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(r'Variance', fontsize=14)
    ax.set_title('Variance of Feedback Joint Velocity', fontsize=16)
    ax.set_xticks(range(num_joints))
    ax.set_xticklabels([name.strip().replace("_"," ").replace("joint", "") for name in joint_names], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig("images/ekf/performance/var_joint_velocity_measured_plot.png")
    plt.close(fig)

    #plot variance of ekf joint velocity
    variance_ekf_velocity = np.var(ekf_joint_velocity, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_joints), variance_ekf_velocity, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(r'Variance', fontsize=14)
    ax.set_title('Variance of Feedback Joint Velocity', fontsize=16)
    ax.set_xticks(range(num_joints))
    ax.set_xticklabels([name.strip().replace("_"," ").replace("joint", "") for name in joint_names], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig("images/ekf/performance/var_joint_velocity_filtered_plot.png")
    plt.close(fig)

    #plot torso orientation error
    fig, ax = plt.subplots()
    ax.plot(t, torso_orientation[:, 0] - des_torso_orientation[:, 0], label='Torso Orientation Roll Error', color='blue')
    ax.plot(t, torso_orientation[:, 1] - des_torso_orientation[:, 1], label='Torso Orientation Pitch Error', color='orange')
    ax.plot(t, torso_orientation[:, 2] - des_torso_orientation[:, 2], label='Torso Orientation Yaw Error', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Orientation [rad]')
    ax.set_title('Torso Orientation Error between feedback and desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/torso_orientation_error_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, torso_angular_velocity[:, 0], label='Torso Angular Velocity X', color='blue')
    ax.plot(t, torso_angular_velocity[:, 1], label='Torso Angular Velocity Y', color='orange')
    ax.plot(t, torso_angular_velocity[:, 2], label='Torso Angular Velocity Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular Velocity [rad/s]')
    ax.set_title('Torso Angular Velocity from feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/torso_angular_velocity_plot.png")
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_position[:, 0],
        label=r'EKF Base Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_position[:, 1],
        label=r'EKF Base Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_position[:, 2],
        label=r'EKF Base Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_base_position[:, 0],
        label=r'Measured Base Position $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_base_position[:, 1],
        label=r'Measured Base Position $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_base_position[:, 2],
        label=r'Measured Base Position $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Comparison between EKF and Measured Base Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_base_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_velocity[:, 0],
        label=r'EKF Base Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_velocity[:, 1],
        label=r'EKF Base Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_velocity[:, 2],
        label=r'EKF Base Velocity $z$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_base_velocity[:, 0],
        label=r'Measured Base Velocity $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_base_velocity[:, 1],
        label=r'Measured Base Velocity $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_base_velocity[:, 2],
        label=r'Measured Base Velocity $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Comparison between EKF and Measured Base Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_base_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_imu_orientation[:, 0],
        label=r'EKF IMU Orientation $W$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation[:, 1],
        label=r'EKF IMU Orientation $X$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation[:, 2],
        label=r'EKF IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation[:, 3],
        label=r'EKF IMU Orientation $Z$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation[:, 0],
        label=r'Measured IMU Orientation $W$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_imu_orientation[:, 1],
        label=r'Measured IMU Orientation $X$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_imu_orientation[:, 2],
        label=r'Measured IMU Orientation $Y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_imu_orientation[:, 3],
        label=r'Measured IMU Orientation $Z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{quat}$]', fontsize=11)
    ax.set_title('Comparison between EKF and Measured IMU Orientation Quat', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_imu_orientation_quat_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_imu_orientation_rpy[:, 0],
        label=r'EKF IMU Orientation $R$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation_rpy[:, 1],
        label=r'EKF IMU Orientation $P$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation_rpy[:, 2],
        label=r'EKF IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation_rpy[:, 0],
        label=r'Measured IMU Orientation $R$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_imu_orientation_rpy[:, 1],
        label=r'Measured IMU Orientation $P$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_imu_orientation_rpy[:, 2],
        label=r'Measured IMU Orientation $Y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{rad}$]', fontsize=11)
    ax.set_title('Comparison between EKF and Measured IMU Orientation RPY', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_imu_orientation_rpy_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_imu_angular_velocity[:, 0],
        label=r'EKF IMU Angular Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_angular_velocity[:, 1],
        label=r'EKF IMU Angular Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_angular_velocity[:, 2],
        label=r'EKF IMU Angular Velocity $z$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_angular_velocity[:, 0],
        label=r'Measured IMU Angular Velocity $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, measured_imu_angular_velocity[:, 1],
        label=r'Measured IMU Angular Velocity $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, measured_imu_angular_velocity[:, 2],
        label=r'Measured IMU Angular Velocity $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Angular Velocity [$\mathrm{rad/s}$]', fontsize=11)
    ax.set_title('Comparison between EKF and Measured IMU Angular Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_imu_angular_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_position[:, 0] - odometry_base_position[:, 0],
        label=r'Error Base Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_position[:, 1] - odometry_base_position[:, 1],
        label=r'Error Base Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_position[:, 2] - odometry_base_position[:, 2],
        label=r'Error Base Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured Base Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/error_base_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_velocity[:, 0] - odometry_base_velocity[:, 0],
        label=r'EKF Base Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_velocity[:, 1] - odometry_base_velocity[:, 1],
        label=r'EKF Base Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_velocity[:, 2] - odometry_base_velocity[:, 2],
        label=r'EKF Base Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured Base Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/error_base_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_imu_orientation[:, 0] - odometry_imu_orientation[:, 0],
        label=r'Error IMU Orientation $W$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation[:, 1] - odometry_imu_orientation[:, 1],
        label=r'Error IMU Orientation $X$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation[:, 2] - odometry_imu_orientation[:, 2],
        label=r'Error IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation[:, 3] - odometry_imu_orientation[:, 3],
        label=r'Error IMU Orientation $Z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{quat}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured IMU Orientation Quat', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/error_imu_orientation_quat_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_imu_orientation_rpy[:, 0] - odometry_imu_orientation_rpy[:, 0],
        label=r'Error IMU Orientation $R$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation_rpy[:, 1] - odometry_imu_orientation_rpy[:, 1],
        label=r'Error IMU Orientation $P$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation_rpy[:, 2] - odometry_imu_orientation_rpy[:, 2],
        label=r'Error IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{rad}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured IMU Orientation RPY', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/error_imu_orientation_rpy_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_imu_angular_velocity[:, 0] - measured_imu_angular_velocity[:, 0],
        label=r'Error IMU Angular Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_angular_velocity[:, 1] - measured_imu_angular_velocity[:, 1],
        label=r'Error IMU Angular Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_angular_velocity[:, 2] - measured_imu_angular_velocity[:, 2],
        label=r'Error IMU Angular Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Angular Velocity [$\mathrm{rad/s}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured IMU Angular Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/error_imu_angular_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


    ##########################
    #  FEEDBACK PLOTS
    ##########################

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, odometry_base_position[:, 0],
        label=r'Odometry Base Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_base_position[:, 1],
        label=r'Odometry Base Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_base_position[:, 2],
        label=r'Odometry Base Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Odometry Base Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/odometry_base_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, odometry_base_velocity[:, 0],
        label=r'Odometry Base Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_base_velocity[:, 1],
        label=r'Odometry Base Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_base_velocity[:, 2],
        label=r'Odometry Base Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Odometry Base Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/odometry_base_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, odometry_imu_orientation[:, 0],
        label=r'Odometry IMU Orientation $W$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation[:, 1],
        label=r'Odometry IMU Orientation $X$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation[:, 2],
        label=r'Odometry IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation[:, 3],
        label=r'Odometry IMU Orientation $Z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{quat}$]', fontsize=11)
    ax.set_title('Odometry IMU Orientation Quat', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/odometry_imu_orientation_quat_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, odometry_imu_orientation_rpy[:, 0],
        label=r'Odometry IMU Orientation $R$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation_rpy[:, 1],
        label=r'Odometry IMU Orientation $P$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation_rpy[:, 2],
        label=r'Odometry IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{rad}$]', fontsize=11)
    ax.set_title('Odometry IMU Orientation RPY', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/odometry_imu_orientation_rpy_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, measured_imu_orientation[:, 0],
        label=r'Measured IMU Orientation $W$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_orientation[:, 1],
        label=r'Measured IMU Orientation $X$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_orientation[:, 2],
        label=r'Measured IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_orientation[:, 3],
        label=r'Measured IMU Orientation $Z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{quat}$]', fontsize=11)
    ax.set_title('Measured IMU Orientation Quat', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/measured_imu_orientation_quat_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, measured_imu_orientation_rpy[:, 0],
        label=r'Measured IMU Orientation $R$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_orientation_rpy[:, 1],
        label=r'Measured IMU Orientation $P$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_orientation_rpy[:, 2],
        label=r'Measured IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{rad}$]', fontsize=11)
    ax.set_title('Measured IMU Orientation RPY', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/measured_imu_orientation_rpy_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, measured_imu_angular_velocity[:, 0],
        label=r'Measured IMU Angular Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_angular_velocity[:, 1],
        label=r'Measured IMU Angular Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_angular_velocity[:, 2],
        label=r'Measured IMU Angular Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Angular Velocity [$\mathrm{rad/s}$]', fontsize=11)
    ax.set_title('Measured Base Angular Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/measured_imu_angular_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, measured_imu_accelerometer[:, 0],
        label=r'Measured IMU Acceleration $x$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_accelerometer[:, 1],
        label=r'Measured IMU Acceleration $y$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_accelerometer[:, 2],
        label=r'Measured IMU Acceleration $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Acceleration [$\mathrm{m/s^2}$]', fontsize=11)
    ax.set_title('Measured IMU Acceleration', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/measured_imu_acceleration_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    # plot feedback joint velocity
    fig, ax = plt.subplots(figsize=(18, 12))
    for i in range(measured_joint_velocity.shape[1]):
        ax.plot(t, measured_joint_velocity[:, i], label=joint_names[i].strip())
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [rad/s]')
    ax.set_title('Feedback Joint Velocities')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/joints/velocities/overall_joint_velocity_plot.png")
    plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_position[:, i],
                label=r'Measured Position'+ f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Position [$\mathrm{rad}$]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=12)
        ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=7
        )
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/feedback/joints/positions/{group_name}_position_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_velocity[:, i],
                label=r'Measured Velocity'+ f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Velocity [$\mathrm{rad/s}$]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=12)
        ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=7
        )
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/feedback/joints/velocities/{group_name}_velocity_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)




    ##########################
    #  MEASURED MOTOR TORQUES (robot experiment)
    ##########################

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_torque[:, i],
                label=joint_names[i].replace(group_name, '').replace('_', '').replace('joint', '').strip(),
                linewidth=2.0
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Torque [$\mathrm{Nm}$]', fontsize=11)
        ax.set_title(f'Measured Motor Torques — {group_name.replace("_", " ").title()}', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', frameon=True, fontsize=7)
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/feedback/motor_torques/{group_name}_measured_torque.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)

    ##########################
    #  EXECUTION TIME PLOTS
    ##########################

    exec_times = {
        'EKF': execution_time_ekf,
        'KF': execution_time_kf,
        'MPC': execution_time_mpc,
        'WBC': execution_time_wbc,
        'RB-WO': execution_time_res_obs,
        'HAC': execution_time_hac,
        'COOP_PLANNER': execution_time_coop_planner,
        'Update': execution_time_update
    }

    for name, times in exec_times.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            times,
            linewidth=2.0,
            label=f'{name}'
        )
        if name == 'Update':
            ax.axhline(
                y=2000,
                linestyle='--',
                linewidth=1.5,
                label='Real-time threshold (2000 µs)'
            )
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(r'Execution Time [$\mu s$]', fontsize=11)
        ax.set_title(f'{name} Execution Time per Iteration', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(frameon=True, fontsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/execution_times/{name}_execution_time_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig)

    total_execution_time = (
        execution_time_ekf +
        execution_time_kf +
        execution_time_mpc +
        execution_time_wbc +
        execution_time_res_obs +
        execution_time_hac +
        execution_time_coop_planner
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        total_execution_time,
        linewidth=2.0,
        label='Total Execution Time'
    )
    ax.axhline(
        y=2000,
        linestyle='--',
        linewidth=1.5,
        label='Real-time threshold (2000 µs)',
        color='red'
    )
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel(r'Total Execution Time [$\mu s$]', fontsize=11)
    ax.set_title('Total Execution Time per Iteration', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/execution_times/total_execution_time_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)



    # -----------------------------------------------------------------------
    # Hand Admittance Controller (HAC) — e_h and e_h_dot
    # Replicates the plots produced by plot_hac.py.
    # Data: hac_eh.txt (N,2), hac_eh_dot.txt (N,2) from the selected folder.
    # Output: images/hac/
    # -----------------------------------------------------------------------
    if os.path.exists(folder + '/hac_eh.txt') and os.path.exists(folder + '/hac_eh_dot.txt'):
        hac_eh = np.loadtxt(folder + '/hac_eh.txt')[startPlot:num_samples, :]
        hac_eh_dot = np.loadtxt(folder + '/hac_eh_dot.txt')[startPlot:num_samples, :]
    else:
        print("[INFO] hac_eh.txt or hac_eh_dot.txt not found — skipped HAC plots.")
        hac_eh = np.zeros((num_samples - startPlot, 2))
        hac_eh_dot = np.zeros((num_samples - startPlot, 2))

    if not os.path.exists('images/hac'):
        os.makedirs('images/hac')

    N_hac = hac_eh.shape[0]
    t_hac = np.arange(N_hac) / 500.0  # control frequency 500 Hz

    # Plot 1 — e_h (average hand position error, F frame xy)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(t_hac, hac_eh[:, 0], linewidth=1.8, label=r'$e_{h,x}$')
    axes[0].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel(r'$e_{h,x}$ [m]', fontsize=11)
    axes[0].set_title(r'Average Hand Position Error $e_h$ (F frame)', fontsize=12)
    axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0].legend(fontsize=10)

    axes[1].plot(t_hac, hac_eh[:, 1], linewidth=1.8, color='tab:orange', label=r'$e_{h,y}$')
    axes[1].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel(r'$e_{h,y}$ [m]', fontsize=11)
    axes[1].set_xlabel('Time [s]', fontsize=11)
    axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1].legend(fontsize=10)

    fig.tight_layout()
    fig.savefig('images/hac/eh.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot 2 — e_h_dot (derivative of average hand error, F frame xy)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(t_hac, hac_eh_dot[:, 0], linewidth=1.8, color='tab:green', label=r'$\dot{e}_{h,x}$')
    axes[0].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel(r'$\dot{e}_{h,x}$ [m/s]', fontsize=11)
    axes[0].set_title(r'Average Hand Error Derivative $\dot{e}_h$ (F frame)', fontsize=12)
    axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0].legend(fontsize=10)

    axes[1].plot(t_hac, hac_eh_dot[:, 1], linewidth=1.8, color='tab:red', label=r'$\dot{e}_{h,y}$')
    axes[1].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel(r'$\dot{e}_{h,y}$ [m/s]', fontsize=11)
    axes[1].set_xlabel('Time [s]', fontsize=11)
    axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1].legend(fontsize=10)

    fig.tight_layout()
    fig.savefig('images/hac/ehdot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot 3 — combined overview (4 subplots in one figure)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    axes[0, 0].plot(t_hac, hac_eh[:, 0], linewidth=1.8, label=r'$e_{h,x}$')
    axes[0, 1].plot(t_hac, hac_eh[:, 1], linewidth=1.8, color='tab:orange', label=r'$e_{h,y}$')
    axes[1, 0].plot(t_hac, hac_eh_dot[:, 0], linewidth=1.8, color='tab:green', label=r'$\dot{e}_{h,x}$')
    axes[1, 1].plot(t_hac, hac_eh_dot[:, 1], linewidth=1.8, color='tab:red', label=r'$\dot{e}_{h,y}$')

    hac_labels = [r'$e_{h,x}$ [m]', r'$e_{h,y}$ [m]',
                  r'$\dot{e}_{h,x}$ [m/s]', r'$\dot{e}_{h,y}$ [m/s]']

    for ax, lbl in zip(axes.flat, hac_labels):
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax.set_ylabel(lbl, fontsize=11)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=9)

    fig.suptitle('Hand Admittance Controller — Errors', fontsize=13)
    fig.tight_layout()
    fig.savefig('images/hac/hac_overview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Wrist force estimation — replicates plot_wrist_force.py
    # (the initial-transient skipping is removed: the whole signal is plotted)
    # Data read from `folder`; output saved in images/wrench_estimations/wrist_force/
    # and images/residuals/, consistently with the other plots of this script.
    # -----------------------------------------------------------------------
    if not os.path.exists('images/wrench_estimations/wrist_force'):
        os.makedirs('images/wrench_estimations/wrist_force')
    if not os.path.exists('images/residuals'):
        os.makedirs('images/residuals')

    WF_FREQ = 500
    WF_LABELS = ['x', 'y', 'z']
    WF_EST_COLORS = ['tab:blue', 'tab:orange', 'tab:green']
    WF_GT_COLORS = ['tab:cyan', 'tab:red', 'tab:olive']

    def _wf_load(filename, required=True):
        path = folder + '/' + filename
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"File non trovato: {path}")
            print(f"[INFO] {filename} non trovato — ground truth non visualizzato.")
            return None
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    def _wf_trim(a, b):
        n = min(len(a), len(b))
        return a[:n], b[:n]

    def _wf_save(fig, name, outdir='images/wrench_estimations/wrist_force'):
        out = os.path.join(outdir, name)
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out}")

    f_right = _wf_load('estimated_force_rwrist.txt')
    f_left = _wf_load('estimated_force_lwrist.txt')
    # The ground truth MuJoCo makes sense only for the simulation (expType == "Simulation",
    # i.e., folder == '/tmp'). For a real experiment, only the estimates are plotted.
    if expType == "Simulation":
        _gt_base = os.path.dirname(folder)
        def _load_gt(name):
            p = _gt_base + '/' + name
            if os.path.exists(p):
                return np.loadtxt(p)
            print(f"[INFO] {name} non trovato in {_gt_base} — ground truth non visualizzato.")
            return None
        gt_right = _load_gt('gt_right_wrist.txt')
        gt_left = _load_gt('gt_left_wrist.txt')
    else:
        gt_right = None
        gt_left = None
        print("[INFO] Real experiment — ground truth not plotted, only estimates.")
    residual = _wf_load('residual_norm.txt', required=False)

    # Allinea stima destra e sinistra
    Nwf = min(len(f_right), len(f_left))
    f_right, f_left = f_right[:Nwf], f_left[:Nwf]

    # Allinea ground truth con la rispettiva stima
    if gt_right is not None:
        f_right, gt_right = _wf_trim(f_right, gt_right)
        Nwf = len(f_right)
    if gt_left is not None:
        f_left, gt_left = _wf_trim(f_left, gt_left)
        Nwf = min(Nwf, len(f_left))

    f_right = f_right[:Nwf]
    f_left = f_left[:Nwf]

    # Nessun transitorio escluso: si plotta tutto il segnale
    sl_wf = slice(0, Nwf)
    t_wf = np.arange(Nwf)[sl_wf] / WF_FREQ

    print(f"Total samples (wrist force): {Nwf}  ({Nwf/WF_FREQ:.2f} s)")

    def _plot_wrist(f_est, gt, side_label, side_tag):
        has_gt = gt is not None

        # Componenti Fx / Fy / Fz
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        fig.suptitle(f'Force estimate — Wrist {side_label}', fontsize=13, fontweight='bold')

        for i, (ax, lbl, ec, gc) in enumerate(zip(axes, WF_LABELS, WF_EST_COLORS, WF_GT_COLORS)):
            ax.plot(t_wf, f_est[sl_wf, i], linewidth=1.5, color=ec,
                    label=rf'Estimated $F_{{{lbl}}}$')
            if has_gt:
                ax.plot(t_wf, gt[sl_wf, i], linewidth=1.5, color=gc, linestyle='--',
                        label=rf'Ground truth $F_{{{lbl}}}$')
            ax.axhline(0, color='k', linewidth=0.7, linestyle=':', alpha=0.5)
            ax.set_ylabel(rf'$F_{{{lbl}}}$ [N]', fontsize=11)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

        axes[-1].set_xlabel('Time [s]', fontsize=11)
        fig.tight_layout()
        _wf_save(fig, f'{side_tag}_components.png')

        # Norma ||F||
        norm_est = np.linalg.norm(f_est[sl_wf], axis=1)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t_wf, norm_est, linewidth=1.5, color=WF_EST_COLORS[0],
                label=r'$\|\hat{F}\|$ estimated')
        if has_gt:
            norm_gt = np.linalg.norm(gt[sl_wf], axis=1)
            ax.plot(t_wf, norm_gt, linewidth=1.5, color=WF_GT_COLORS[0], linestyle='--',
                    label=r'$\|F\|$ ground truth')
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'$\|F\|$ [N]', fontsize=11)
        ax.set_title(f'Force norm — Wrist {side_label}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        _wf_save(fig, f'{side_tag}_norm.png')

    _plot_wrist(f_right, gt_right, 'RIGHT', 'right_wrist')
    _plot_wrist(f_left, gt_left, 'LEFT', 'left_wrist')

    # Confronto norma destro vs sinistro
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_wf, np.linalg.norm(f_right[sl_wf], axis=1), linewidth=1.5,
            color='tab:blue', label='Wrist right — estimated')
    ax.plot(t_wf, np.linalg.norm(f_left[sl_wf], axis=1), linewidth=1.5,
            color='tab:orange', linestyle='--', label='Wrist left — estimated')

    if gt_right is not None:
        ax.plot(t_wf, np.linalg.norm(gt_right[sl_wf], axis=1), linewidth=1.2,
                color='tab:cyan', linestyle=':', label='GT right')
    if gt_left is not None:
        ax.plot(t_wf, np.linalg.norm(gt_left[sl_wf], axis=1), linewidth=1.2,
                color='tab:red', linestyle=':', label='GT left')

    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'$\|F\|$', fontsize=11)
    ax.set_title('Comparison of norms: both wrists', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    _wf_save(fig, 'comparison_norm.png')

    # Norma del vettore residuo -> images/residuals
    if residual is not None:
        r = np.ravel(residual)
        Nr = len(r)
        sl_r = slice(0, Nr)
        t_r = np.arange(Nr)[sl_r] / WF_FREQ

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t_r, r[sl_r], linewidth=1.5, color='tab:purple',
                label=r'$\|r\|$ residual vector norm')
        ax.axhline(0, color='k', linewidth=0.7, linestyle=':', alpha=0.5)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'$\|r\|$', fontsize=11)
        ax.set_title('Residual vector norm', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        _wf_save(fig, 'residual_norm.png', outdir='images/residuals')

        print(f"\n[RESIDUAL] samples: {Nr}  "
              f"mean: {np.mean(r[sl_r]):.4f}  max: {np.max(r[sl_r]):.4f}")
    else:
        print("[INFO] residual_norm.txt not found — skipped residual plots.")

    # Error stats
    def _print_error_stats(f_est, gt, side_label):
        if gt is None:
            print(f"[{side_label}] Ground not available — no stats computed.")
            return
        err = f_est[sl_wf] - gt[sl_wf]
        norm_err = np.linalg.norm(err, axis=1)
        print(f"\n{'='*50}")
        print(f"  Error — Wrist {side_label}")
        print(f"{'='*50}")
        print(f"  {'Axis':<6} {'Mean Error [N]':>20} {'Variance [N²]':>18}")
        print(f"  {'-'*46}")
        for i, lbl in enumerate(WF_LABELS):
            mean_i = np.mean(err[:, i])
            var_i = np.var(err[:, i])
            print(f"  F_{lbl:<4}  {mean_i:>20.4f} {var_i:>18.4f}")
        print(f"  {'-'*46}")
        print(f"  {'||err||':<6} {'Mean Error [N]':>20} {'Variance [N²]':>18}")
        print(f"  {'':6}  {np.mean(norm_err):>20.4f} {np.var(norm_err):>18.4f}")
        print(f"{'='*50}")

    _print_error_stats(f_right, gt_right, 'RIGHT')
    _print_error_stats(f_left, gt_left, 'LEFT')

    ##########################
    #  ARM RESIDUALS PLOTS
    ##########################

    right_arm_joint_labels = [
        'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw',
        'r_elbow', 'r_wrist_roll', 'r_wrist_pitch', 'r_wrist_yaw'
    ]
    left_arm_joint_labels = [
        'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw',
        'l_elbow', 'l_wrist_roll', 'l_wrist_pitch', 'l_wrist_yaw'
    ]

    right_arm_residuals_path = folder + '/right_arm_residual.txt'
    left_arm_residuals_path  = folder + '/left_arm_residual.txt'

    if os.path.exists(right_arm_residuals_path):
        right_arm_res = np.loadtxt(right_arm_residuals_path)
        if right_arm_res.ndim == 1:
            right_arm_res = right_arm_res.reshape(-1, 1)
        Nr_arm = right_arm_res.shape[0]
        t_arm = np.linspace(0.0, delta * Nr_arm, Nr_arm)
        n_joints_right = min(right_arm_res.shape[1], len(right_arm_joint_labels))
        for i in range(n_joints_right):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t_arm, right_arm_res[:, i], linewidth=1.5)
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel('Residual [Nm]', fontsize=11)
            ax.set_title(f'Right Arm Residual — {right_arm_joint_labels[i]}', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            fig.tight_layout()
            fig.savefig(f'images/residuals/right_arm/{right_arm_joint_labels[i]}_residual.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(n_joints_right):
            ax.plot(t_arm, right_arm_res[:, i], label=right_arm_joint_labels[i], linewidth=1.5)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel('Residual [Nm]', fontsize=11)
        ax.set_title('Right Arm — All Joint Residuals', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        fig.savefig('images/residuals/right_arm/all_joints_residual.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Right arm residual plots saved ({Nr_arm} samples).")
    else:
        print("[INFO] right_arm_residuals.txt not found — skipped right arm residual plots.")

    if os.path.exists(left_arm_residuals_path):
        left_arm_res = np.loadtxt(left_arm_residuals_path)
        if left_arm_res.ndim == 1:
            left_arm_res = left_arm_res.reshape(-1, 1)
        Nl_arm = left_arm_res.shape[0]
        t_arm_l = np.linspace(0.0, delta * Nl_arm, Nl_arm)
        n_joints_left = min(left_arm_res.shape[1], len(left_arm_joint_labels))
        for i in range(n_joints_left):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t_arm_l, left_arm_res[:, i], linewidth=1.5)
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel('Residual [Nm]', fontsize=11)
            ax.set_title(f'Left Arm Residual — {left_arm_joint_labels[i]}', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            fig.tight_layout()
            fig.savefig(f'images/residuals/left_arm/{left_arm_joint_labels[i]}_residual.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(n_joints_left):
            ax.plot(t_arm_l, left_arm_res[:, i], label=left_arm_joint_labels[i], linewidth=1.5)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel('Residual [Nm]', fontsize=11)
        ax.set_title('Left Arm — All Joint Residuals', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        fig.savefig('images/residuals/left_arm/all_joints_residual.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Left arm residual plots saved ({Nl_arm} samples).")
    else:
        print("[INFO] left_arm_residuals.txt not found — skipped left arm residual plots.")

    ##########################
    #  GENERALIZED MOMENTUM
    ##########################

    gm_path  = folder + '/generalized_momentum.txt'
    gm0_path = folder + '/initialized_generalized_momentum.txt'

    if os.path.exists(gm_path) and os.path.exists(gm0_path):
        p_data  = np.loadtxt(gm_path)
        p0_data = np.loadtxt(gm0_path)

        if p_data.ndim == 1:
            p_data = p_data.reshape(1, -1)
        if p0_data.ndim == 1:
            p0_data = p0_data.reshape(1, -1)

        N_gm  = p_data.shape[0]
        n_dof = p_data.shape[1]
        t_gm  = np.linspace(0.0, delta * N_gm, N_gm)

        # DOF labels: first 6 are floating base, rest are joints
        base_labels = ['base_vx', 'base_vy', 'base_vz', 'base_wx', 'base_wy', 'base_wz']
        joint_labels_gm = [jn.strip() for jn in joint_names]  # from joint_names.txt loaded earlier
        dof_labels = base_labels + joint_labels_gm
        dof_labels = dof_labels[:n_dof]  # trim if needed

        if not os.path.exists('images/generalized_momentum'):
            os.makedirs('images/generalized_momentum')
        if not os.path.exists('images/generalized_momentum/dofs'):
            os.makedirs('images/generalized_momentum/dofs')

        cmap_gm = plt.colormaps['tab20']

        # -- Separate: p_ overview (all DOFs) --
        n_cols_gm = 5
        n_rows_gm = int(np.ceil(n_dof / n_cols_gm))
        fig_p, axes_p = plt.subplots(n_rows_gm, n_cols_gm,
                                      figsize=(4 * n_cols_gm, 3 * n_rows_gm))
        axes_p = np.array(axes_p).flatten()
        for i in range(n_dof):
            ax = axes_p[i]
            ax.plot(t_gm, p_data[:, i], color=cmap_gm(i % 20), linewidth=1.2)
            ax.set_title(dof_labels[i] if i < len(dof_labels) else f'DOF {i}', fontsize=7)
            ax.set_xlabel('t [s]', fontsize=6)
            ax.set_ylabel('p [kg·m²/s]', fontsize=6)
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        for j in range(n_dof, len(axes_p)):
            axes_p[j].set_visible(False)
        fig_p.suptitle('Generalized Momentum p(t) — all DOFs', fontsize=13)
        fig_p.tight_layout()
        fig_p.savefig('images/generalized_momentum/p_all_dofs.png', dpi=150, bbox_inches='tight')
        plt.close(fig_p)

        # -- Separate: p0_ overview (all DOFs) --
        fig_p0, axes_p0 = plt.subplots(n_rows_gm, n_cols_gm,
                                        figsize=(4 * n_cols_gm, 3 * n_rows_gm))
        axes_p0 = np.array(axes_p0).flatten()
        for i in range(n_dof):
            ax = axes_p0[i]
            ax.plot(t_gm, p0_data[:, i], color=cmap_gm(i % 20), linewidth=1.2, linestyle='--')
            ax.set_title(dof_labels[i] if i < len(dof_labels) else f'DOF {i}', fontsize=7)
            ax.set_xlabel('t [s]', fontsize=6)
            ax.set_ylabel('p₀ [kg·m²/s]', fontsize=6)
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        for j in range(n_dof, len(axes_p0)):
            axes_p0[j].set_visible(False)
        fig_p0.suptitle('Initial Generalized Momentum p₀ — all DOFs', fontsize=13)
        fig_p0.tight_layout()
        fig_p0.savefig('images/generalized_momentum/p0_all_dofs.png', dpi=150, bbox_inches='tight')
        plt.close(fig_p0)

        # -- Insieme: p_ and p0_ overlaid per DOF (subplots) --
        fig_cmp, axes_cmp = plt.subplots(n_rows_gm, n_cols_gm,
                                          figsize=(4 * n_cols_gm, 3 * n_rows_gm))
        axes_cmp = np.array(axes_cmp).flatten()
        for i in range(n_dof):
            ax = axes_cmp[i]
            color = cmap_gm(i % 20)
            ax.plot(t_gm, p_data[:, i],  color=color, linewidth=1.2, label='p')
            ax.plot(t_gm, p0_data[:, i], color=color, linewidth=1.2, linestyle='--', label='p₀')
            ax.set_title(dof_labels[i] if i < len(dof_labels) else f'DOF {i}', fontsize=7)
            ax.set_xlabel('t [s]', fontsize=6)
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
            if i == 0:
                ax.legend(fontsize=6)
        for j in range(n_dof, len(axes_cmp)):
            axes_cmp[j].set_visible(False)
        fig_cmp.suptitle('Generalized Momentum p(t) vs p₀ — all DOFs', fontsize=13)
        fig_cmp.tight_layout()
        fig_cmp.savefig('images/generalized_momentum/p_vs_p0_all_dofs.png', dpi=150, bbox_inches='tight')
        plt.close(fig_cmp)

        # -- Per-DOF separate PNGs (p_ and p0_ together per DOF) --
        for i in range(n_dof):
            label = dof_labels[i] if i < len(dof_labels) else f'dof_{i}'
            safe_label = label.replace('/', '_').replace(' ', '_')
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(t_gm, p_data[:, i],  linewidth=1.5, label='p(t)')
            ax.plot(t_gm, p0_data[:, i], linewidth=1.5, linestyle='--', label='p₀')
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel('[kg·m²/s]', fontsize=11)
            ax.set_title(f'Generalized Momentum — {label}', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
            fig.tight_layout()
            fig.savefig(f'images/generalized_momentum/dofs/{i:02d}_{safe_label}.png',
                        dpi=120, bbox_inches='tight')
            plt.close(fig)

        print(f"[INFO] Generalized momentum plots saved ({N_gm} samples, {n_dof} DOFs).")
    else:
        print("[INFO] generalized_momentum.txt or initial_generalized_momentum.txt not found — skipped.")

    ##########################
    #  TAU_M - G  ARM PLOTS
    ##########################

    right_arm_joint_labels = [
        'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw',
        'r_elbow', 'r_wrist_roll', 'r_wrist_pitch', 'r_wrist_yaw'
    ]
    left_arm_joint_labels = [
        'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw',
        'l_elbow', 'l_wrist_roll', 'l_wrist_pitch', 'l_wrist_yaw'
    ]

    right_arm_tau_g_path = folder + '/right_arm_tau_g.txt'
    left_arm_tau_g_path  = folder + '/left_arm_tau_g.txt'

    if os.path.exists(right_arm_tau_g_path):
        right_arm_tg = np.loadtxt(right_arm_tau_g_path)
        if right_arm_tg.ndim == 1:
            right_arm_tg = right_arm_tg.reshape(-1, 1)
        Nr_tg = right_arm_tg.shape[0]
        t_tg = np.linspace(0.0, delta * Nr_tg, Nr_tg)
        n_joints_right_tg = min(right_arm_tg.shape[1], len(right_arm_joint_labels))
        for i in range(n_joints_right_tg):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t_tg, right_arm_tg[:, i], linewidth=1.5)
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel(r'$\tau_m - g$ [Nm]', fontsize=11)
            ax.set_title(fr'Right Arm $\tau_m - g$ — {right_arm_joint_labels[i]}', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            fig.tight_layout()
            fig.savefig(f'images/tau_g/right_arm/{right_arm_joint_labels[i]}_tau_g.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(n_joints_right_tg):
            ax.plot(t_tg, right_arm_tg[:, i], label=right_arm_joint_labels[i], linewidth=1.5)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'$\tau_m - g$ [Nm]', fontsize=11)
        ax.set_title(r'Right Arm — $\tau_m - g$ All Joints', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        fig.savefig('images/tau_g/right_arm/all_joints_tau_g.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Right arm tau_m-g plots saved ({Nr_tg} samples).")
    else:
        print("[INFO] right_arm_tau_g.txt not found — skipped right arm tau_m-g plots.")

    if os.path.exists(left_arm_tau_g_path):
        left_arm_tg = np.loadtxt(left_arm_tau_g_path)
        if left_arm_tg.ndim == 1:
            left_arm_tg = left_arm_tg.reshape(-1, 1)
        Nl_tg = left_arm_tg.shape[0]
        t_tg_l = np.linspace(0.0, delta * Nl_tg, Nl_tg)
        n_joints_left_tg = min(left_arm_tg.shape[1], len(left_arm_joint_labels))
        for i in range(n_joints_left_tg):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t_tg_l, left_arm_tg[:, i], linewidth=1.5)
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel(r'$\tau_m - g$ [Nm]', fontsize=11)
            ax.set_title(fr'Left Arm $\tau_m - g$ — {left_arm_joint_labels[i]}', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            fig.tight_layout()
            fig.savefig(f'images/tau_g/left_arm/{left_arm_joint_labels[i]}_tau_g.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(n_joints_left_tg):
            ax.plot(t_tg_l, left_arm_tg[:, i], label=left_arm_joint_labels[i], linewidth=1.5)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'$\tau_m - g$ [Nm]', fontsize=11)
        ax.set_title(r'Left Arm — $\tau_m - g$ All Joints', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        fig.savefig('images/tau_g/left_arm/all_joints_tau_g.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Left arm tau_m-g plots saved ({Nl_tg} samples).")
    else:
        print("[INFO] left_arm_tau_g.txt not found — skipped left arm tau_m-g plots.")