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
        folder = '/tmp'
        expType = "Simulation"
    else:
        folder = 'experiments/experiment_' + expNumber
        expType = "Experiment"

    endPlot = input("Enter the time (in seconds) at which you want to end the plots (or press Enter to plot all data): ")
    if endPlot != '':
        endPlot = int(float(endPlot) * 500)  # Assuming a control frequency of 500 Hz
    else:
        endPlot = 10

    joint_names = open(folder + '/joint_names.txt').readlines()

    parameters_log = np.loadtxt(folder + '/parameters_log.txt')

    startTimeWBCCL = parameters_log
    startPlot = int(0.001 * startTimeWBCCL * 500 + 10)  # Assuming a control frequency of 500 Hz
    startPlot = 0

    fb_com_position = np.loadtxt(folder + '/fb_com_position.txt')
    num_samples = fb_com_position.shape[0] - endPlot
    fb_com_position = np.loadtxt(folder + '/fb_com_position.txt')[startPlot:num_samples, :]
    fb_com_velocity = np.loadtxt(folder + '/fb_com_velocity.txt')[startPlot:num_samples, :]
    fb_zmp_position = np.loadtxt(folder + '/fb_zmp_position.txt')[startPlot:num_samples, :]
    kf_com_position =  np.loadtxt(folder + '/kf_com_position.txt')[startPlot:num_samples, :]
    kf_com_velocity =  np.loadtxt(folder + '/kf_com_velocity.txt')[startPlot:num_samples, :]
    kf_zmp_position =  np.loadtxt(folder + '/kf_zmp_position.txt')[startPlot:num_samples, :]
    des_com_position = np.loadtxt(folder + '/des_com_position.txt')[startPlot:num_samples, :]
    des_com_velocity = np.loadtxt(folder + '/des_com_velocity.txt')[startPlot:num_samples, :]
    des_zmp_position = np.loadtxt(folder + '/des_zmp_position.txt')[startPlot:num_samples, :]
    des_com_acceleration = np.loadtxt(folder + '/des_com_acceleration.txt')[startPlot:num_samples, :]

    input_torque: np.ndarray = np.loadtxt(folder +'/input_torque.txt')[startPlot:num_samples, :]

    ef_zmp_position = np.loadtxt(folder + '/ef_zmp_position.txt')[startPlot:num_samples, :]

    p_lsole_fb = np.loadtxt(folder + '/p_lsole_fb.txt')[startPlot:num_samples, :]
    p_rsole_fb = np.loadtxt(folder + '/p_rsole_fb.txt')[startPlot:num_samples, :]
    v_lsole_fb = np.loadtxt(folder + '/v_lsole_fb.txt')[startPlot:num_samples, :]
    v_rsole_fb = np.loadtxt(folder + '/v_rsole_fb.txt')[startPlot:num_samples, :]
    p_lsole_des = np.loadtxt(folder + '/p_lsole_des.txt')[startPlot:num_samples, :]
    p_rsole_des = np.loadtxt(folder + '/p_rsole_des.txt')[startPlot:num_samples, :]
    v_lsole_des = np.loadtxt(folder + '/v_lsole_des.txt')[startPlot:num_samples, :]
    v_rsole_des = np.loadtxt(folder + '/v_rsole_des.txt')[startPlot:num_samples, :]

    fb_lsole_orientation = np.loadtxt(folder + '/fb_lsole_orientation.txt')[startPlot:num_samples, :]
    fb_rsole_orientation = np.loadtxt(folder + '/fb_rsole_orientation.txt')[startPlot:num_samples, :]
    des_lsole_orientation = np.loadtxt(folder + '/des_lsole_orientation.txt')[startPlot:num_samples, :]
    des_rsole_orientation = np.loadtxt(folder + '/des_rsole_orientation.txt')[startPlot:num_samples, :]

    estimated_force_lsole = np.loadtxt(folder + '/estimated_force_lsole.txt')[startPlot:num_samples, :]
    estimated_force_rsole = np.loadtxt(folder + '/estimated_force_rsole.txt')[startPlot:num_samples, :]
    wbc_accelerations = np.loadtxt(folder + '/wbc_accelerations.txt')[startPlot:num_samples, :]

    ekf_base_position = np.loadtxt(folder + '/ekf_base_position.txt')[startPlot:num_samples, :]
    ekf_base_velocity = np.loadtxt(folder + '/ekf_base_velocity.txt')[startPlot:num_samples, :]
    ekf_base_orientation = np.loadtxt(folder + '/ekf_base_orientation.txt')[startPlot:num_samples, :]
    ekf_base_orientation_rpy = np.loadtxt(folder + '/ekf_base_orientation_rpy.txt')[startPlot:num_samples, :]
    ekf_base_angular_velocity = np.loadtxt(folder + '/ekf_base_angular_velocity.txt')[startPlot:num_samples, :]
    ekf_imu_orientation = np.loadtxt(folder + '/ekf_imu_orientation.txt')[startPlot:num_samples, :]
    ekf_imu_orientation_rpy = np.loadtxt(folder + '/ekf_imu_orientation_rpy.txt')[startPlot:num_samples, :]
    ekf_imu_angular_velocity = np.loadtxt(folder + '/ekf_imu_angular_velocity.txt')[startPlot:num_samples, :]
    ekf_joint_position = np.loadtxt(folder + '/ekf_joint_position.txt')[startPlot:num_samples, :]
    ekf_joint_velocity = np.loadtxt(folder + '/ekf_joint_velocity.txt')[startPlot:num_samples, :]

    torso_orientation = np.loadtxt(folder + '/torso_orientation.txt')[startPlot:num_samples, :]
    torso_angular_velocity = np.loadtxt(folder + '/torso_angular_velocity.txt')[startPlot:num_samples, :]
    des_torso_orientation = np.loadtxt(folder + '/des_torso_orientation.txt')[startPlot:num_samples, :]
    des_torso_angular_velocity = np.loadtxt(folder + '/des_torso_angular_velocity.txt')[startPlot:num_samples, :]

    mpc_pred_com_pos = np.loadtxt(folder + '/mpc_pred_com_pos.txt')
    mpc_pred_com_vel = np.loadtxt(folder + '/mpc_pred_com_vel.txt')
    mpc_pred_zmp_pos = np.loadtxt(folder + '/mpc_pred_zmp_pos.txt')

    execution_time_ekf = np.loadtxt(folder + '/execution_time_ekf.txt')[startPlot:num_samples]
    execution_time_kf = np.loadtxt(folder + '/execution_time_kf.txt')[startPlot:num_samples]
    execution_time_mpc = np.loadtxt(folder + '/execution_time_mpc.txt')[startPlot:num_samples]
    execution_time_wbc = np.loadtxt(folder + '/execution_time_wbc.txt')[startPlot:num_samples]
    execution_time_update = np.loadtxt(folder + '/execution_time_update.txt')[startPlot:num_samples]

    odometry_base_position = np.loadtxt(folder + '/odometry_base_position.txt')[startPlot:num_samples, :]
    odometry_base_velocity = np.loadtxt(folder + '/odometry_base_velocity.txt')[startPlot:num_samples, :]
    odometry_imu_orientation = np.loadtxt(folder + '/odometry_imu_orientation.txt')[startPlot:num_samples, :]
    odometry_imu_orientation_rpy = np.loadtxt(folder + '/odometry_imu_orientation_rpy.txt')[startPlot:num_samples, :]
    measured_joint_position: np.ndarray = np.loadtxt(folder +'/measured_joint_position.txt')[startPlot:num_samples, :]
    measured_joint_velocity: np.ndarray = np.loadtxt(folder +'/measured_joint_velocity.txt')[startPlot:num_samples, :]
    measured_imu_orientation: np.ndarray = np.loadtxt(folder + '/measured_imu_orientation.txt')[startPlot:num_samples, :]
    measured_imu_orientation_rpy: np.ndarray = np.loadtxt(folder + '/measured_imu_orientation_rpy.txt')[startPlot:num_samples, :]
    measured_imu_angular_velocity: np.ndarray = np.loadtxt(folder + '/measured_imu_angular_velocity.txt')[startPlot:num_samples, :]
    measured_imu_accelerometer: np.ndarray = np.loadtxt(folder + '/measured_imu_accelerometer.txt')[startPlot:num_samples, :]
        
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


    labels_xyz = ['x', 'y', 'z']
    labels_quat = ['w', 'x', 'y', 'z']
    labels_rpy = ['r', 'p', 'y']

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
    if not os.path.exists('images/forces_torques/joints'):
        os.makedirs('images/forces_torques/joints')
    if not os.path.exists('images/soles/references'):
        os.makedirs('images/soles/references')
    if not os.path.exists('images/soles/errors'):
        os.makedirs('images/soles/errors')
    if not os.path.exists('images/mpc'):
        os.makedirs('images/mpc')

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
        fig.savefig(f"images/forces_torques/joints/{group_name}_input_joint_torques.png")
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
    fig.savefig("images/forces_torques/wbc_base_linear_acceleration.png")
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
    fig.savefig("images/forces_torques/wbc_base_angular_acceleration.png")
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
        fig.savefig(f"images/forces_torques/joints/{group_name}_acceleration.png")
        plt.close(fig)
        figs.append(fig)

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
    fig.savefig("images/forces_torques/estimated_force_left_sole.png")
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
    fig.savefig("images/forces_torques/estimated_force_right_sole.png")
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

    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    for i in range(3):
        axs[i].plot(
            t, kf_zmp_position[:, i] - kf_zmp_position[0, i],
            label=fr'Actual ZMP Position ${labels_xyz[i]}$',
            linewidth=2.0
        )
        axs[i].plot(
            t, des_zmp_position[:, i] - des_zmp_position[0, i],
            label=fr'Desired ZMP Position ${labels_xyz[i]}$',
            linewidth=2.0,
            linestyle='--'
        )
        
        axs[i].set_ylabel(r'[$\mathrm{m}$]', fontsize=10)
        axs[i].set_title(f'Comparison between reference and actual ZMP {labels_xyz[i]}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)

    axs[-1].set_xlabel('Time [s]', fontsize=11)

    fig.tight_layout()
    fig.savefig(
        "images/com/errors/comparison_zmp_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for i in range(3):
        axs[i].plot(
            t, kf_com_position[:, i] - kf_com_position[0, i],
            label=fr'Actual CoM Position ${labels_xyz[i]}$',
            linewidth=2.0
        )
        axs[i].plot(
            t, des_com_position[:, i] - des_com_position[0, i],
            label=fr'Desired CoM Position ${labels_xyz[i]}$',
            linewidth=2.0,
            linestyle='--'
        )
        
        axs[i].set_ylabel(r'[$\mathrm{m}$]', fontsize=10)
        axs[i].set_title(f'Comparison between reference and actual ZMP {labels_xyz[i]}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)

    axs[-1].set_xlabel('Time [s]', fontsize=11)

    fig.tight_layout()
    fig.savefig(
        "images/com/errors/comparison_com_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for i in range(3):
        axs[i].plot(
            t, kf_com_velocity[:, i] - kf_com_velocity[0, i],
            label=fr'Actual CoM Velocity ${labels_xyz[i]}$',
            linewidth=2.0
        )
        axs[i].plot(
            t, des_com_velocity[:, i] - des_com_velocity[0, i],
            label=fr'Desired CoM Velocity ${labels_xyz[i]}$',
            linewidth=2.0,
            linestyle='--'
        )
        
        axs[i].set_ylabel(r'[$\mathrm{m}$]', fontsize=10)
        axs[i].set_title(f'Comparison between reference and actual ZMP {labels_xyz[i]}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)

    axs[-1].set_xlabel('Time [s]', fontsize=11)

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

    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for i in range(3):
        axs[i].plot(
            t, p_lsole_fb[:, i] - p_lsole_fb[0, i],
            label=fr'Actual Left Sole Position ${labels_xyz[i]}$',
            linewidth=2.0
        )
        axs[i].plot(
            t, p_lsole_des[:, i] - p_lsole_des[0, i],
            label=fr'Desired Left Sole Position ${labels_xyz[i]}$',
            linewidth=2.0,
            linestyle='--'
        )
        
        axs[i].set_ylabel(r'[$\mathrm{m}$]', fontsize=10)
        axs[i].set_title(f'Comparison between Desired and Actual Left Sole Position {labels_xyz[i]}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)

    axs[-1].set_xlabel('Time [s]', fontsize=11)

    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/comparison_left_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for i in range(3):
        axs[i].plot(
            t, p_rsole_fb[:, i] - p_rsole_fb[0, i],
            label=fr'Actual Right Sole Position ${labels_xyz[i]}$',
            linewidth=2.0
        )
        axs[i].plot(
            t, p_rsole_des[:, i] - p_rsole_des[0, i],
            label=fr'Desired Right Sole Position ${labels_xyz[i]}$',
            linewidth=2.0,
            linestyle='--'
        )
        
        axs[i].set_ylabel(r'[$\mathrm{m}$]', fontsize=10)
        axs[i].set_title(f'Comparison between Desired and Actual Right Sole Position {labels_xyz[i]}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)

    axs[-1].set_xlabel('Time [s]', fontsize=11)

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

    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for i in range(3):
        axs[i].plot(
            t, v_lsole_fb[:, i],
            label=fr'Actual Left Sole Velocity ${labels_xyz[i]}$',
            linewidth=2.0
        )
        axs[i].plot(
            t, v_lsole_des[:, i],
            label=fr'Desired Left Sole Velocity ${labels_xyz[i]}$',
            linewidth=2.0,
            linestyle='--'
        )
        
        axs[i].set_ylabel(r'[$\mathrm{m}$]', fontsize=10)
        axs[i].set_title(f'Comparison between Desired and Actual Left Sole Velocity {labels_xyz[i]}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)

    axs[-1].set_xlabel('Time [s]', fontsize=11)

    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/comparison_left_sole_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for i in range(3):
        axs[i].plot(
            t, v_rsole_fb[:, i],
            label=fr'Actual Right Sole Velocity ${labels_xyz[i]}$',
            linewidth=2.0
        )
        axs[i].plot(
            t, v_rsole_des[:, i],
            label=fr'Desired Right Sole Velocity ${labels_xyz[i]}$',
            linewidth=2.0,
            linestyle='--'
        )
        
        axs[i].set_ylabel(r'[$\mathrm{m}$]', fontsize=10)
        axs[i].set_title(f'Comparison between Desired and Actual Right Sole Velocity {labels_xyz[i]}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)

    axs[-1].set_xlabel('Time [s]', fontsize=11)

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

    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for i in range(3):
        axs[i].plot(
            t, ekf_base_position[:, i],
            label=fr'EKF ${labels_xyz[i]}$',
            linewidth=2.0
        )
        axs[i].plot(
            t, odometry_base_position[:, i],
            label=fr'Measured ${labels_xyz[i]}$',
            linewidth=2.0,
            linestyle='--'
        )
        
        axs[i].set_ylabel(r'[$\mathrm{m}$]', fontsize=10)
        axs[i].set_title(f'Base Position {labels_xyz[i]}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)

    axs[-1].set_xlabel('Time [s]', fontsize=11)

    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_base_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for i in range(3):
        axs[i].plot(
            t, ekf_base_velocity[:, i],
            label=fr'EKF Velocity ${labels_xyz[i]}$',
            linewidth=2.0
        )
        axs[i].plot(
            t, odometry_base_velocity[:, i],
            label=fr'Odometry Velocity ${labels_xyz[i]}$',
            linewidth=2.0,
            linestyle='--'
        )
        
        axs[i].set_ylabel(r'[$\mathrm{m}$]', fontsize=10)
        axs[i].set_title(f'Base Velocity {labels_xyz[i]}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)

    axs[-1].set_xlabel('Time [s]', fontsize=11)

    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_base_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, axs = plt.subplots(4, 1, figsize=(7, 8), sharex=True)

    for i in range(4):
        axs[i].plot(
            t, ekf_imu_orientation[:, i],
            label=fr'EKF Orientation ${labels_quat[i]}$',
            linewidth=2.0
        )
        axs[i].plot(
            t, odometry_imu_orientation[:, i],
            label=fr'Odometry Orientation ${labels_quat[i]}$',
            linewidth=2.0,
            linestyle='--'
        )
        
        axs[i].set_ylabel(r'[$\mathrm{m}$]', fontsize=10)
        axs[i].set_title(f'Base Orientation Quaternion {labels_quat[i]}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)

    axs[-1].set_xlabel('Time [s]', fontsize=11)

    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_imu_orientation_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for i in range(3):
        axs[i].plot(
            t, ekf_imu_orientation_rpy[:, i],
            label=fr'EKF Orientation ${labels_rpy[i]}$',
            linewidth=2.0
        )
        axs[i].plot(
            t, odometry_imu_orientation_rpy[:, i],
            label=fr'Odometry Orientation ${labels_rpy[i]}$',
            linewidth=2.0,
            linestyle='--'
        )
        
        axs[i].set_ylabel(r'[$\mathrm{m}$]', fontsize=10)
        axs[i].set_title(f'Base Orientation RPY {labels_rpy[i]}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)

    axs[-1].set_xlabel('Time [s]', fontsize=11)

    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_imu_orientation_rpy_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for i in range(3):
        axs[i].plot(
            t, ekf_imu_angular_velocity[:, i],
            label=fr'EKF Angular Velocity ${labels_xyz[i]}$',
            linewidth=2.0
        )
        axs[i].plot(
            t, measured_imu_angular_velocity[:, i],
            label=fr'Odometry Angular Velocity ${labels_xyz[i]}$',
            linewidth=2.0,
            linestyle='--'
        )
        
        axs[i].set_ylabel(r'[$\mathrm{m}$]', fontsize=10)
        axs[i].set_title(f'Base Angular Velocity {labels_xy[i]}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)

    axs[-1].set_xlabel('Time [s]', fontsize=11)

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
    #  EXECUTION TIME PLOTS
    ##########################

    exec_times = {
        'EKF': execution_time_ekf,
        'KF': execution_time_kf,
        'MPC': execution_time_mpc,
        'WBC': execution_time_wbc,
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
        execution_time_wbc
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


