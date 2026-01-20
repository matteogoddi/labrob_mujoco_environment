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
import cv2


if __name__ == '__main__':
    #request input from terminal
    number = input("Enter 0 to plot data from the last simulation or the number of the experiment: ")
    if number == '0':
        folder = '/tmp'
    else:
        folder = 'experiments/experiment_' + number
    
    joint_names = open(folder + '/joint_names.txt').readlines()

    parameters_log = np.loadtxt(folder + '/parameters_log.txt')

    startTimeWBCCL = parameters_log
    startPlot = int(0.001 * startTimeWBCCL * 500)  # Assuming a control frequency of 500 Hz

    sim_com_position =  np.loadtxt(folder + '/sim_com_position.txt')
    num_samples = sim_com_position.shape[0] - 10
    sim_com_position =  np.loadtxt(folder + '/sim_com_position.txt')[startPlot:num_samples, :]
    sim_com_velocity =  np.loadtxt(folder + '/sim_com_velocity.txt')[startPlot:num_samples, :]
    sim_zmp_position =  np.loadtxt(folder + '/sim_zmp_position.txt')[startPlot:num_samples, :]
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

    p_lsole_sim = np.loadtxt(folder + '/p_lsole_sim.txt')[startPlot:num_samples, :]
    p_rsole_sim = np.loadtxt(folder + '/p_rsole_sim.txt')[startPlot:num_samples, :]
    v_lsole_sim = np.loadtxt(folder + '/v_lsole_sim.txt')[startPlot:num_samples, :]
    v_rsole_sim = np.loadtxt(folder + '/v_rsole_sim.txt')[startPlot:num_samples, :]
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
    ekf_joint_position = np.loadtxt(folder + '/ekf_joint_position.txt')[startPlot:num_samples, :]
    ekf_joint_velocity = np.loadtxt(folder + '/ekf_joint_velocity.txt')[startPlot:num_samples, :]
    sim_base_position = np.loadtxt(folder + '/sim_base_position.txt')[startPlot:num_samples, :]
    sim_base_velocity = np.loadtxt(folder + '/sim_base_velocity.txt')[startPlot:num_samples, :]
    sim_base_orientation = np.loadtxt(folder + '/sim_base_orientation.txt')[startPlot:num_samples, :]
    sim_base_orientation_rpy = np.loadtxt(folder + '/sim_base_orientation_rpy.txt')[startPlot:num_samples, :]
    sim_base_angular_velocity = np.loadtxt(folder + '/sim_base_angular_velocity.txt')[startPlot:num_samples, :]
    sim_joint_position: np.ndarray = np.loadtxt(folder + '/sim_joint_position.txt')[startPlot:num_samples, :]
    sim_joint_velocity: np.ndarray = np.loadtxt(folder + '/sim_joint_velocity.txt')[startPlot:num_samples, :]

    torso_orientation = np.loadtxt(folder + '/torso_orientation.txt')[startPlot:num_samples, :]
    torso_angular_velocity = np.loadtxt(folder + '/torso_angular_velocity.txt')[startPlot:num_samples, :]
    des_torso_orientation = np.loadtxt(folder + '/des_torso_orientation.txt')[startPlot:num_samples, :]
    des_torso_angular_velocity = np.loadtxt(folder + '/des_torso_angular_velocity.txt')[startPlot:num_samples, :]


    go_base_position = np.loadtxt(folder + '/go_base_position.txt')[startPlot:num_samples, :]
    go_base_velocity = np.loadtxt(folder + '/go_base_velocity.txt')[startPlot:num_samples, :]
    go_base_orientation = np.loadtxt(folder + '/go_base_orientation.txt')[startPlot:num_samples, :]
    go_base_orientation_rpy = np.loadtxt(folder + '/go_base_orientation_rpy.txt')[startPlot:num_samples, :]
    go_base_angular_velocity = np.loadtxt(folder + '/go_base_angular_velocity.txt')[startPlot:num_samples, :]
    go_base_accelerometer = np.loadtxt(folder + '/go_base_accelerometer.txt')[startPlot:num_samples, :]

    mpc_pred_com_pos = np.loadtxt(folder + '/mpc_pred_com_pos.txt')
    mpc_pred_com_vel = np.loadtxt(folder + '/mpc_pred_com_vel.txt')
    mpc_pred_zmp_pos = np.loadtxt(folder + '/mpc_pred_zmp_pos.txt')

    execution_time_ekf = np.loadtxt(folder + '/execution_time_ekf.txt')[startPlot:num_samples]
    execution_time_kf = np.loadtxt(folder + '/execution_time_kf.txt')[startPlot:num_samples]
    execution_time_mpc = np.loadtxt(folder + '/execution_time_mpc.txt')[startPlot:num_samples]
    execution_time_wbc = np.loadtxt(folder + '/execution_time_wbc.txt')[startPlot:num_samples]
    execution_time_update = np.loadtxt(folder + '/execution_time_update.txt')[startPlot:num_samples]

    measured_joint_position: np.ndarray = np.loadtxt(folder +'/measured_joint_position.txt')[startPlot:num_samples, :]
    measured_joint_velocity: np.ndarray = np.loadtxt(folder +'/measured_joint_velocity.txt')[startPlot:num_samples, :]
    measured_imu_orientation: np.ndarray = np.loadtxt(folder + '/measured_imu_orientation.txt')[startPlot:num_samples, :]
    measured_imu_angular_velocity: np.ndarray = np.loadtxt(folder + '/measured_imu_angular_velocity.txt')[startPlot:num_samples, :]
    measured_imu_accelerometer: np.ndarray = np.loadtxt(folder + '/measured_imu_accelerometer.txt')[startPlot:num_samples, :]
        
    delta = 1 / 500  # Assuming a control frequency of 500 Hz
    t = np.linspace(0.0, delta * (num_samples - startPlot), num_samples - startPlot)
    num_joints = 27

    if not os.path.exists('images/simulation/positions'):
        os.makedirs('images/simulation/positions')
    if not os.path.exists('images/simulation/velocities'):
        os.makedirs('images/simulation/velocities')
    if not os.path.exists('images/feedback/positions'):
        os.makedirs('images/feedback/positions')
    if not os.path.exists('images/feedback/velocities'):
        os.makedirs('images/feedback/velocities')
    if not os.path.exists('images/ekf'):
        os.makedirs('images/ekf')
    if not os.path.exists('images/execution_times'):
        os.makedirs('images/execution_times')
    if not os.path.exists('images/com'):
        os.makedirs('images/com')
    if not os.path.exists('images/com/references'):
        os.makedirs('images/com/references')
    if not os.path.exists('images/com/errors'):
        os.makedirs('images/com/errors')
    if not os.path.exists('images/forces_torques'):
        os.makedirs('images/forces_torques')
    if not os.path.exists('images/forces_torques/joints'):
        os.makedirs('images/forces_torques/joints')
    if not os.path.exists('images/soles'):
        os.makedirs('images/soles')
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
        t, des_com_position[:, 0],
        label=r'Desired CoM Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_position[:, 1],
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
        t, des_zmp_position[:, 0],
        label=r'Desired ZMP Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_zmp_position[:, 1],
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
        t, des_zmp_position[:,0],
        label=r'Desired ZMP Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_zmp_position[:,0],
        label=r'Actual ZMP Position $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_zmp_position[:, 1],
        label=r'Desired ZMP Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_zmp_position[:, 1],
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
        t, des_com_position[:,0],
        label=r'Desired CoM Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_com_position[:,0],
        label=r'Actual CoM Position $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_com_position[:, 1],
        label=r'Desired CoM Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_position[:, 1],
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
        t, kf_com_position[:, 0],
        label=r'CoM Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_zmp_position[:, 0],
        label=r'ZMP Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_fb[:, 0],
        label=r'Left Foot Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_fb[:, 0],
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
    #plot des, sim and fb lsole position
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 0], label='FB Left Sole X', color='blue')
    ax.plot(t, p_lsole_fb[:, 1], label='FB Left Sole Y', color='orange')
    ax.plot(t, p_lsole_fb[:, 2], label='FB Left Sole Z', color='green')
    ax.plot(t, p_lsole_des[:, 0], label='Des Left Sole X', color='blue', linestyle=':')
    ax.plot(t, p_lsole_des[:, 1], label='Des Left Sole Y', color='orange', linestyle=':')
    ax.plot(t, p_lsole_des[:, 2], label='Des Left Sole Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Left Sole Position [m]')
    ax.set_title('Left Sole Position Simulation vs Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/soles/sim_vs_fb_left_sole_position_plot.png")
    plt.close(fig)

    #plot des, sim and fb rsole position
    fig, ax = plt.subplots()
    ax.plot(t, p_rsole_fb[:, 0], label='FB Right Sole X', color='blue')
    ax.plot(t, p_rsole_fb[:, 1], label='FB Right Sole Y', color='orange')
    ax.plot(t, p_rsole_fb[:, 2], label='FB Right Sole Z', color='green')
    ax.plot(t, p_rsole_des[:, 0], label='Des Right Sole X', color='blue', linestyle=':')
    ax.plot(t, p_rsole_des[:, 1], label='Des Right Sole Y', color='orange', linestyle=':')
    ax.plot(t, p_rsole_des[:, 2], label='Des Right Sole Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Right Sole Position [m]')
    ax.set_title('Right Sole Position Simulation vs Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/soles/sim_vs_fb_right_sole_position_plot.png")
    plt.close(fig)

    #plot des, sim and fb rsole position
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 2], label='FB Right Sole Y', color='orange')
    ax.plot(t, p_rsole_fb[:, 2], label='FB Right Sole Z', color='green')
    ax.plot(t, p_lsole_des[:, 2], label='Des Right Sole Y', color='orange', linestyle=':')
    ax.plot(t, p_rsole_des[:, 2], label='Des Right Sole Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Right Sole Position [m]')
    ax.set_title('Left and Right Sole Position Simulation vs Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/soles/lsole_vs_rsole_z_position_plot.png")
    plt.close(fig)

    #plot des, sim and fb lsole velocity
    fig, ax = plt.subplots()
    ax.plot(t, v_lsole_fb[:, 0], label='FB Left Sole Vel X', color='blue')
    ax.plot(t, v_lsole_fb[:, 1], label='FB Left Sole Vel Y', color='orange')
    ax.plot(t, v_lsole_fb[:, 2], label='FB Left Sole Vel Z', color='green')
    ax.plot(t, v_lsole_des[:, 0], label='Des Left Sole Vel X', color='blue', linestyle=':')
    ax.plot(t, v_lsole_des[:, 1], label='Des Left Sole Vel Y', color='orange', linestyle=':')
    ax.plot(t, v_lsole_des[:, 2], label='Des Left Sole Vel Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Left Sole Velocity [m/s]')
    ax.set_title('Left Sole Velocity Simulation vs Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/soles/sim_vs_fb_left_sole_velocity_plot.png")
    plt.close(fig)

    #plot des, sim and fb rsole velocity
    fig, ax = plt.subplots()
    ax.plot(t, v_rsole_fb[:, 0], label='FB Right Sole Vel X', color='blue')
    ax.plot(t, v_rsole_fb[:, 1], label='FB Right Sole Vel Y', color='orange')
    ax.plot(t, v_rsole_fb[:, 2], label='FB Right Sole Vel Z', color='green')
    ax.plot(t, v_rsole_des[:, 0], label='Des Right Sole Vel X', color='blue', linestyle=':')
    ax.plot(t, v_rsole_des[:, 1], label='Des Right Sole Vel Y', color='orange', linestyle=':')
    ax.plot(t, v_rsole_des[:, 2], label='Des Right Sole Vel Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Right Sole Velocity [m/s]')
    ax.set_title('Right Sole Velocity Simulation vs Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/soles/sim_vs_fb_right_sole_velocity_plot.png")
    plt.close(fig)


    ##########################
    #  SIMULATION JOINTS PLOTS
    ##########################
    

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, sim_joint_position[:, i], label=joint_names[i])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [rad]')
        ax.set_title(group_name.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        fig.savefig(f"images/simulation/positions/{group_name}_position_plot.png")
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, sim_joint_velocity[:, i], label=joint_names[i])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angular velocities [rad/s]')
        ax.set_title(group_name.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        fig.savefig(f"images/simulation/velocities/{group_name}_velocities_plot.png")
        plt.close(fig)

    #plot simulation joint velocities
    fig, ax = plt.subplots(figsize=(18, 12))
    for i in range(sim_joint_velocity.shape[1]):
        ax.plot(t, sim_joint_velocity[:, i], label=joint_names[i].strip())
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [rad/s]')
    ax.set_title('Simulation Joint Velocities')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/simulation/velocities/simulation_joint_velocities_plot.png")
    plt.close(fig)




    ##########################
    #  EKF PLOTS
    ##########################
    
    # Plot EKF base position
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_position[:, 0], label='EKF Base Position X', color='blue')
    ax.plot(t, ekf_base_position[:, 1], label='EKF Base Position Y', color='orange')
    ax.plot(t, ekf_base_position[:, 2], label='EKF Base Position Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Position [m]')
    ax.set_title('Base Position of EKF estimation')
    ax.grid(True)
    ax.legend()
    # fig.tight_layout()
    fig.savefig("images/ekf/base_position_plot.png")
    plt.close(fig)

    # Plot EKF base velocity
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_velocity[:, 0], label='EKF Base Velocity X', color='blue')
    ax.plot(t, ekf_base_velocity[:, 1], label='EKF Base Velocity Y', color='orange')
    ax.plot(t, ekf_base_velocity[:, 2], label='EKF Base Velocity Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Velocity [m/s]')
    ax.set_title('Base Velocity of EKF estimation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_velocity_plot.png")
    plt.close(fig)

    # Plot EKF base orientation in quaternion format
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_orientation[:, 0], label='EKF Base Orientation W', color='blue')
    ax.plot(t, ekf_base_orientation[:, 1], label='EKF Base Orientation X', color='orange')
    ax.plot(t, ekf_base_orientation[:, 2], label='EKF Base Orientation Y', color='green')
    ax.plot(t, ekf_base_orientation[:, 3], label='EKF Base Orientation Z', color='red')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Orientation [Quaternion]')
    ax.set_title('Base Orientation of EKF estimation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_orientation_plot.png")
    plt.close(fig)

    # Plot EKF base angular velocity
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_angular_velocity[:, 0], label='EKF Base Angular Velocity X', color='blue')
    ax.plot(t, ekf_base_angular_velocity[:, 1], label='EKF Base Angular Velocity Y', color='orange')
    ax.plot(t, ekf_base_angular_velocity[:, 2], label='EKF Base Angular Velocity Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Angular Velocity [rad/s]')
    ax.set_title('Angular Velocity of EKF estimation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_angular_velocity_plot.png")
    plt.close(fig)


    # Plot velocity error between ekf joint velocity and simulated joint velocity
    fig, ax = plt.subplots(figsize=(18, 12))
    colormap = plt.colormaps['tab10'] 
    line_styles = ['-', '--', '-.', ':']
    for i in range(num_joints):
        color = colormap(i % 10)
        linestyle = line_styles[(i // 10) % len(line_styles)]  # cambia stile ogni 10 joint
        error = ekf_joint_velocity[:, i]
        ax.plot(t, error,
                label=joint_names[i].strip(),
                color=color,
                linestyle=linestyle,
                linewidth=2)
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Velocity [rad/s]', fontsize=14)
    ax.set_title('Joint Velocity of EKF estimation', fontsize=16)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=12, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.savefig("images/ekf/joint_velocities_plot.png")
    plt.close(fig)

    #plot position error between ekf joint position and fb joint position
    fig, ax = plt.subplots(figsize=(18, 12))
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
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Position [rad]', fontsize=14)
    ax.set_title('Joint Position Error between EKF estimation and Feedback', fontsize=16)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=12, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.savefig("images/ekf/joint_position_error_vs_feedback_plot.png")
    plt.close(fig)

    #plot velocity error between ekf joint velocity and fb joint velocity
    fig, ax = plt.subplots(figsize=(18, 12))
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
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Velocity [rad/s]', fontsize=14)
    ax.set_title('Joint Velocity Error between EKF estimation and Feedback', fontsize=16)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=12, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.savefig("images/ekf/joint_velocities_error_vs_feedback_plot.png")
    plt.close(fig)

    ##########################
    #  FEEDBACK PLOTS
    ##########################

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
    fig.savefig("images/feedback/torso_orientation_error_plot.png")
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
    fig.savefig("images/feedback/torso_angular_velocity_plot.png")
    plt.close(fig)

    # Plot EKF base position
    fig, ax = plt.subplots()
    ax.plot(t, go_base_position[:, 0] - ekf_base_position[:, 0], label='Base Position Error X', color='blue')
    ax.plot(t, go_base_position[:, 1] - ekf_base_position[:, 1], label='Base Position Error Y', color='orange')
    ax.plot(t, go_base_position[:, 2] - ekf_base_position[:, 2], label='Base Position Error Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('Base Position Error between odometry and ekf')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/base_position_error_plot.png")
    plt.close(fig)

    # Plot go base velocity
    fig, ax = plt.subplots()
    ax.plot(t, go_base_velocity[:, 0] - ekf_base_velocity[:, 0], label='Error Base Velocity X', color='blue')
    ax.plot(t, go_base_velocity[:, 1] - ekf_base_velocity[:, 1], label='Error Base Velocity Y', color='orange')
    ax.plot(t, go_base_velocity[:, 2] - ekf_base_velocity[:, 2], label='Error Base Velocity Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Base Velocity Error between fb velocity and ekf')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/base_velocity_error_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, go_base_position[:, 0], label='Odom Base position X', color='blue')
    ax.plot(t, ekf_base_position[:, 0], label='EKF Base position X', color='blue', linestyle = "--")
    ax.plot(t, go_base_position[:, 1], label='Odom Base position Y', color='orange')
    ax.plot(t, ekf_base_position[:, 1], label='EKF Base position Y', color='orange', linestyle = "--")
    ax.plot(t, go_base_position[:, 2], label='Odom Base position Z', color='green')
    ax.plot(t, ekf_base_position[:, 2], label='EKF Base position Z', color='green', linestyle = "--")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m/s]')
    ax.set_title('Base Position of go estimation and ekf')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/base_position_plot.png")
    plt.close(fig)

    #plot go base position and fb com position
    fig, ax = plt.subplots()
    ax.plot(t, go_base_position[:, 0], label='Odom Base position X', color='blue')
    ax.plot(t, fb_com_position[:, 0], label='FB COM position X', color='blue', linestyle = "--")
    ax.plot(t, go_base_position[:, 1], label='Odom Base position Y', color='orange')
    ax.plot(t, fb_com_position[:, 1], label='FB COM position Y', color='orange', linestyle = "--")
    ax.plot(t, go_base_position[:, 2], label='Odom Base position Z', color='green')
    ax.plot(t, fb_com_position[:, 2], label='FB COM position Z', color='green', linestyle = "--")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m/s]')
    ax.set_title('Base Position of odometry and FB COM position')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/base_position_fb_com_position_plot.png")
    plt.close(fig)


    # Plot go base orientation in quaternion format
    fig, ax = plt.subplots()
    ax.plot(t, go_base_orientation[:, 0] - ekf_base_orientation[:, 0], label='Odom Base Orientation W', color='blue')
    ax.plot(t, go_base_orientation[:, 1] - ekf_base_orientation[:, 1], label='Odom Base Orientation X', color='orange')
    ax.plot(t, go_base_orientation[:, 2] - ekf_base_orientation[:, 2], label='Odom Base Orientation Y', color='green')
    ax.plot(t, go_base_orientation[:, 3] - ekf_base_orientation[:, 3], label='Odom Base Orientation Z', color='red')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Orientation [Quaternion]')
    ax.set_title('Base Orientation Error between fb and ekf')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/base_orientation_error_plot.png")
    plt.close(fig)

    # Plot go base orientation in quaternion format
    fig, ax = plt.subplots()
    ax.plot(t, go_base_orientation_rpy[:, 0] - ekf_base_orientation_rpy[:, 0], label='Error Base Orientation R', color='blue')
    ax.plot(t, go_base_orientation_rpy[:, 1] - ekf_base_orientation_rpy[:, 1], label='Error Base Orientation P', color='orange')
    ax.plot(t, go_base_orientation_rpy[:, 2] - ekf_base_orientation_rpy[:, 2], label='Error Base Orientation Y', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Base Orientation [RPY]')
    ax.set_title('Base Orientation Error between fb and ekf')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/base_orientation_rpy_error_plot.png")
    plt.close(fig)

    # Plot go base orientation in quaternion format
    fig, ax = plt.subplots()
    ax.plot(t, go_base_orientation_rpy[:, 0], label='FB Base Orientation R', color='blue')
    ax.plot(t, go_base_orientation_rpy[:, 1], label='FB Base Orientation P', color='orange')
    ax.plot(t, go_base_orientation_rpy[:, 2], label='FB Base Orientation Y', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Base Orientation [RPY]')
    ax.set_title('Base Orientation Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/base_orientation_rpy_plot.png")
    plt.close(fig)

    # Plot go base orientation in quaternion format
    fig, ax = plt.subplots()
    ax.plot(t, go_base_orientation[:, 0] , label='FB Base Orientation W', color='blue')
    ax.plot(t, go_base_orientation[:, 1], label='FB Base Orientation X', color='orange')
    ax.plot(t, go_base_orientation[:, 2] , label='FB Base Orientation Y', color='green')
    ax.plot(t, go_base_orientation[:, 3], label='FB Base Orientation Z', color='red')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('go Base Orientation [Quaternion]')
    ax.set_title('Base Orientation Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/base_orientation_plot.png")
    plt.close(fig)

    # plot imu orientation
    fig, ax = plt.subplots()
    ax.plot(t, measured_imu_orientation[:, 0], label='IMU Orientation W', color='blue')
    ax.plot(t, measured_imu_orientation[:, 1], label='IMU Orientation X', color='orange')
    ax.plot(t, measured_imu_orientation[:, 2], label='IMU Orientation Y', color='green')
    ax.plot(t, measured_imu_orientation[:, 3], label='IMU Orientation Z', color='red')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('IMU Orientation [Quaternion]')
    ax.set_title('IMU Orientation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/measured_imu_orientation_quaternion_plot.png")
    plt.close(fig)

    # plot imu angular velocity
    fig, ax = plt.subplots()
    ax.plot(t, measured_imu_angular_velocity[:, 0], label='IMU Angular Velocity X', color='blue')
    ax.plot(t, measured_imu_angular_velocity[:, 1], label='IMU Angular Velocity Y', color='orange')
    ax.plot(t, measured_imu_angular_velocity[:, 2], label='IMU Angular Velocity Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('IMU Angular Velocity [rad/s]')
    ax.set_title('IMU Angular Velocity')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/measured_imu_angular_velocity_plot.png")
    plt.close(fig)
    
    # plot imu accelerometer
    fig, ax = plt.subplots()
    ax.plot(t, measured_imu_accelerometer[:, 0], label='IMU Accelerometer X', color='blue')
    ax.plot(t, measured_imu_accelerometer[:, 1], label='IMU Accelerometer Y', color='orange')
    ax.plot(t, measured_imu_accelerometer[:, 2], label='IMU Accelerometer Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('IMU Accelerometer [m/s^2]')
    ax.set_title('IMU Accelerometer')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/measured_imu_accelerometer_plot.png")
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
    fig.savefig("images/feedback/measured_joint_velocity_plot.png")
    plt.close(fig)


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

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, measured_joint_position[:, i], label=f"{joint_names[i].strip()} FB" )
            ax.plot(t, ekf_joint_position[:, i], label=f"{joint_names[i].strip()} EKF", linestyle='--')
            ax.hlines(reference_positions[i], t[0], t[-1], colors='gray', linestyles=':', label=f"{joint_names[i].strip()} Ref")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [rad]')
        ax.set_title(group_name.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)
        fig.savefig(f"images/feedback/positions/{group_name}_fb_position_plot.png")
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, measured_joint_velocity[:, i], label=f"{joint_names[i].strip()} FB" )
            ax.plot(t, ekf_joint_velocity[:, i], label=f"{joint_names[i].strip()} EKF", linestyle=':')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [rad/s]')
        ax.set_title(group_name.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)
        fig.savefig(f"images/feedback/velocities/{group_name}_fb_velocity_plot.png")
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


