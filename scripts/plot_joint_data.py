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
    input_torque: np.ndarray = np.loadtxt(folder +'/input_torque.txt')

    sim_com_position =  np.loadtxt(folder + '/sim_com_position.txt')
    sim_com_velocity =  np.loadtxt(folder + '/sim_com_velocity.txt')
    sim_zmp_position =  np.loadtxt(folder + '/sim_zmp_position.txt')
    fb_com_position = np.loadtxt(folder + '/fb_com_position.txt')
    fb_com_velocity = np.loadtxt(folder + '/fb_com_velocity.txt')
    fb_zmp_position = np.loadtxt(folder + '/fb_zmp_position.txt')
    kf_com_position =  np.loadtxt(folder + '/kf_com_position.txt')
    kf_com_velocity =  np.loadtxt(folder + '/kf_com_velocity.txt')
    kf_zmp_position =  np.loadtxt(folder + '/kf_zmp_position.txt')
    des_com_position = np.loadtxt(folder + '/des_com_position.txt')
    des_com_velocity = np.loadtxt(folder + '/des_com_velocity.txt')
    des_zmp_position = np.loadtxt(folder + '/des_zmp_position.txt')
    des_com_acceleration = np.loadtxt(folder + '/des_com_acceleration.txt')

    ef_zmp_position = np.loadtxt(folder + '/ef_zmp_position.txt')

    p_lsole_sim = np.loadtxt(folder + '/p_lsole_sim.txt')
    p_rsole_sim = np.loadtxt(folder + '/p_rsole_sim.txt')
    v_lsole_sim = np.loadtxt(folder + '/v_lsole_sim.txt')
    v_rsole_sim = np.loadtxt(folder + '/v_rsole_sim.txt')
    p_lsole_fb = np.loadtxt(folder + '/p_lsole_fb.txt')
    p_rsole_fb = np.loadtxt(folder + '/p_rsole_fb.txt')
    v_lsole_fb = np.loadtxt(folder + '/v_lsole_fb.txt')
    v_rsole_fb = np.loadtxt(folder + '/v_rsole_fb.txt')
    p_lsole_des = np.loadtxt(folder + '/p_lsole_des.txt')
    p_rsole_des = np.loadtxt(folder + '/p_rsole_des.txt')
    v_lsole_des = np.loadtxt(folder + '/v_lsole_des.txt')
    v_rsole_des = np.loadtxt(folder + '/v_rsole_des.txt')

    fb_lsole_orientation = np.loadtxt(folder + '/fb_lsole_orientation.txt')
    fb_rsole_orientation = np.loadtxt(folder + '/fb_rsole_orientation.txt')
    des_lsole_orientation = np.loadtxt(folder + '/des_lsole_orientation.txt')
    des_rsole_orientation = np.loadtxt(folder + '/des_rsole_orientation.txt')

    estimated_force_lsole = np.loadtxt(folder + '/estimated_force_lsole.txt')
    estimated_force_rsole = np.loadtxt(folder + '/estimated_force_rsole.txt')
    # wbc_accelerations = np.loadtxt(folder + '/wbc_accelerations.txt')

    ekf_base_position = np.loadtxt(folder + '/ekf_base_position.txt')
    ekf_base_velocity = np.loadtxt(folder + '/ekf_base_velocity.txt')
    ekf_base_orientation = np.loadtxt(folder + '/ekf_base_orientation.txt')
    ekf_base_orientation_rpy = np.loadtxt(folder + '/ekf_base_orientation_rpy.txt')
    ekf_base_angular_velocity = np.loadtxt(folder + '/ekf_base_angular_velocity.txt')
    ekf_joint_position = np.loadtxt(folder + '/ekf_joint_position.txt')
    ekf_joint_velocity = np.loadtxt(folder + '/ekf_joint_velocity.txt')
    sim_base_position = np.loadtxt(folder + '/sim_base_position.txt')
    sim_base_velocity = np.loadtxt(folder + '/sim_base_velocity.txt')
    sim_base_orientation = np.loadtxt(folder + '/sim_base_orientation.txt')
    sim_base_orientation_rpy = np.loadtxt(folder + '/sim_base_orientation_rpy.txt')
    sim_base_angular_velocity = np.loadtxt(folder + '/sim_base_angular_velocity.txt')
    sim_joint_position: np.ndarray = np.loadtxt(folder + '/sim_joint_position.txt')
    sim_joint_velocity: np.ndarray = np.loadtxt(folder + '/sim_joint_velocity.txt')


    go_base_position = np.loadtxt(folder + '/go_base_position.txt')
    go_base_velocity = np.loadtxt(folder + '/go_base_velocity.txt')
    go_base_orientation = np.loadtxt(folder + '/go_base_orientation.txt')
    go_base_orientation_rpy = np.loadtxt(folder + '/go_base_orientation_rpy.txt')
    go_base_angular_velocity = np.loadtxt(folder + '/go_base_angular_velocity.txt')
    go_base_accelerometer = np.loadtxt(folder + '/go_base_accelerometer.txt')

    mpc_pred_com_pos = np.loadtxt(folder + '/mpc_pred_com_pos.txt')
    mpc_pred_com_vel = np.loadtxt(folder + '/mpc_pred_com_vel.txt')
    mpc_pred_zmp_pos = np.loadtxt(folder + '/mpc_pred_zmp_pos.txt')

    execution_time_ekf = np.loadtxt(folder + '/execution_time_ekf.txt')
    execution_time_kf = np.loadtxt(folder + '/execution_time_kf.txt')
    execution_time_mpc = np.loadtxt(folder + '/execution_time_mpc.txt')
    execution_time_wbc = np.loadtxt(folder + '/execution_time_wbc.txt')
    execution_time_update = np.loadtxt(folder + '/execution_time_update.txt')

    measured_joint_position: np.ndarray = np.loadtxt(folder +'/measured_joint_position.txt')
    measured_joint_velocity: np.ndarray = np.loadtxt(folder +'/measured_joint_velocity.txt')
    measured_imu_orientation: np.ndarray = np.loadtxt(folder + '/measured_imu_orientation.txt')
    measured_imu_angular_velocity: np.ndarray = np.loadtxt(folder + '/measured_imu_angular_velocity.txt')
    measured_imu_accelerometer: np.ndarray = np.loadtxt(folder + '/measured_imu_accelerometer.txt')


    num_samples = sim_joint_position.shape[0] - 10
    input_torque = sim_joint_position[:num_samples, :]

    sim_com_position = sim_com_position[:num_samples, :]
    sim_com_velocity = sim_com_velocity[:num_samples, :]
    sim_zmp_position = sim_zmp_position[:num_samples, :]
    fb_com_position = fb_com_position[:num_samples, :]
    fb_com_velocity = fb_com_velocity[:num_samples, :]
    fb_zmp_position = fb_zmp_position[:num_samples, :]
    kf_com_position = kf_com_position[:num_samples, :]
    kf_com_velocity = kf_com_velocity[:num_samples, :]
    kf_zmp_position = kf_zmp_position[:num_samples, :]
    des_com_position = des_com_position[:num_samples, :]
    des_com_velocity = des_com_velocity[:num_samples, :]
    des_zmp_position = des_zmp_position[:num_samples, :]
    des_com_acceleration = des_com_acceleration[:num_samples, :]

    fb_lsole_orientation = fb_lsole_orientation[:num_samples, :]
    fb_rsole_orientation = fb_rsole_orientation[:num_samples, :]
    des_lsole_orientation = des_lsole_orientation[:num_samples, :]
    des_rsole_orientation = des_rsole_orientation[:num_samples, :]

    ef_zmp_position = ef_zmp_position[:num_samples, :]

    p_lsole_sim = p_lsole_sim[:num_samples, :]
    p_rsole_sim = p_rsole_sim[:num_samples, :]
    v_lsole_sim = v_lsole_sim[:num_samples, :]
    v_rsole_sim = v_rsole_sim[:num_samples, :]
    p_lsole_fb = p_lsole_fb[:num_samples, :]
    p_rsole_fb = p_rsole_fb[:num_samples, :]
    v_lsole_fb = v_lsole_fb[:num_samples, :]
    v_rsole_fb = v_rsole_fb[:num_samples, :]
    p_lsole_des = p_lsole_des[:num_samples, :]
    p_rsole_des = p_rsole_des[:num_samples, :]
    v_lsole_des = v_lsole_des[:num_samples, :]
    v_rsole_des = v_rsole_des[:num_samples, :]



    estimated_force_lsole = estimated_force_lsole[:num_samples, :]
    estimated_force_rsole = estimated_force_rsole[:num_samples, :]
    # wbc_accelerations = wbc_accelerations[:num_samples, :]
    
    ekf_base_position = ekf_base_position[:num_samples, :]
    ekf_base_velocity = ekf_base_velocity[:num_samples, :]
    ekf_base_orientation = ekf_base_orientation[:num_samples, :]
    ekf_base_orientation_rpy = ekf_base_orientation_rpy[:num_samples, :]
    ekf_base_angular_velocity = ekf_base_angular_velocity[:num_samples, :]
    ekf_joint_position = ekf_joint_position[:num_samples, :]
    ekf_joint_velocity = ekf_joint_velocity[:num_samples, :]
    sim_base_position = sim_base_position[:num_samples, :]
    sim_base_velocity = sim_base_velocity[:num_samples, :]
    sim_base_orientation = sim_base_orientation[:num_samples, :]
    sim_base_orientation_rpy = sim_base_orientation_rpy[:num_samples, :]
    sim_base_angular_velocity = sim_base_angular_velocity[:num_samples, :]
    sim_joint_position = sim_joint_position[:num_samples, :]
    sim_joint_velocity = sim_joint_velocity[:num_samples, :]
    measured_joint_position = measured_joint_position[:num_samples, :]
    measured_joint_velocity = measured_joint_velocity[:num_samples, :]
    measured_imu_orientation = measured_imu_orientation[:num_samples, :]
    measured_imu_angular_velocity = measured_imu_angular_velocity[:num_samples, :]
    measured_imu_accelerometer = measured_imu_accelerometer[:num_samples, :]


    go_base_position = go_base_position[:num_samples, :]
    go_base_velocity = go_base_velocity[:num_samples, :]
    go_base_orientation = go_base_orientation[:num_samples, :]
    go_base_orientation_rpy = go_base_orientation_rpy[:num_samples, :]
    go_base_angular_velocity = go_base_angular_velocity[:num_samples, :]
    go_base_accelerometer = go_base_accelerometer[:num_samples, :]

    # mpc_pred_com_pos = mpc_pred_com_pos[:num_samples, :]
    # mpc_pred_com_vel = mpc_pred_com_vel[:num_samples, :]
    # mpc_pred_zmp_pos = mpc_pred_zmp_pos[:num_samples, :]

    execution_time_ekf = execution_time_ekf[:num_samples]
    execution_time_kf = execution_time_kf[:num_samples]
    execution_time_mpc = execution_time_mpc[:num_samples]
    execution_time_wbc = execution_time_wbc[:num_samples]
    execution_time_update = execution_time_update[:num_samples]
    
        
    delta = 1 / 500  # Assuming a control frequency of 500 Hz
    t = np.linspace(0.0, delta * num_samples, num_samples)
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
    if not os.path.exists('images/forces_torques'):
        os.makedirs('images/forces_torques')
    if not os.path.exists('images/forces_torques/joints'):
        os.makedirs('images/forces_torques/joints')
    if not os.path.exists('images/soles'):
        os.makedirs('images/soles')
    if not os.path.exists('images/go'):
        os.makedirs('images/go')
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

    # fig, ax = plt.subplots()
    # ax.plot(t, wbc_accelerations[:, 0], label='Acceleration X', color='blue')
    # ax.plot(t, wbc_accelerations[:, 1], label='Acceleration Y', color='orange')
    # ax.plot(t, wbc_accelerations[:, 2], label='Acceleration Z', color='green')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Acceleration [N]')
    # ax.set_title('Acceleration base')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig("images/forces_torques/wbc_base_linear_acceleration.png")
    # plt.close(fig)

    # fig, ax = plt.subplots()
    # ax.plot(t, wbc_accelerations[:, 3], label='Acceleration X', color='blue')
    # ax.plot(t, wbc_accelerations[:, 4], label='Acceleration Y', color='orange')
    # ax.plot(t, wbc_accelerations[:, 5], label='Acceleration Z', color='green')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Acceleration [N]')
    # ax.set_title('Acceleration base')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig("images/forces_torques/wbc_base_angular_acceleration.png")
    # plt.close(fig)

    # figs = []
    # for group_name, indices in grouped_indices.items():
    #     fig, ax = plt.subplots()
    #     for i in indices:
    #         ax.plot(t, wbc_accelerations[:, i + 6], label=joint_names[i].strip())
    #     ax.set_xlabel('Time [s]')
    #     ax.set_ylabel('Acceleration [rad/s^2]')
    #     ax.set_title(f'WBC Acceleration - {group_name}')
    #     ax.grid(True)
    #     ax.legend()
    #     fig.tight_layout()
    #     fig.savefig(f"images/forces_torques/joints/{group_name}_acceleration.png")
    #     plt.close(fig)
    #     figs.append(fig)

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


    ##############################
    # GO PLOTS
    #############################

     # Plot EKF base position
    fig, ax = plt.subplots()
    ax.plot(t, go_base_position[:, 0] - ekf_base_position[:, 0], label='go Base Position X', color='blue')
    ax.plot(t, go_base_position[:, 1] - ekf_base_position[:, 1], label='go Base Position Y', color='orange')
    ax.plot(t, go_base_position[:, 2] - ekf_base_position[:, 2], label='go Base Position Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('go Base Position [m]')
    ax.set_title('Base Position Error between go estimation and ekf')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/go/base_position_error_plot.png")
    plt.close(fig)

    # Plot go base velocity
    fig, ax = plt.subplots()
    ax.plot(t, go_base_velocity[:, 0] - ekf_base_velocity[:, 0], label='go Base Velocity X', color='blue')
    ax.plot(t, go_base_velocity[:, 1] - ekf_base_velocity[:, 1], label='go Base Velocity Y', color='orange')
    ax.plot(t, go_base_velocity[:, 2] - ekf_base_velocity[:, 2], label='go Base Velocity Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('go Base Velocity [m/s]')
    ax.set_title('Base Velocity Error between go estimation and ekf')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/go/base_velocity_error_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, go_base_position[:, 0], label='go Base position X', color='blue')
    ax.plot(t, ekf_base_position[:, 0], label='Base position X', color='blue', linestyle = "--")
    ax.plot(t, go_base_position[:, 1], label='go Base position Y', color='orange')
    ax.plot(t, ekf_base_position[:, 1], label='Base position Y', color='orange', linestyle = "--")
    ax.plot(t, go_base_position[:, 2], label='go Base position Z', color='green')
    ax.plot(t, ekf_base_position[:, 2], label='Base position Z', color='green', linestyle = "--")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('go Base Position [m/s]')
    ax.set_title('Base Position of go estimation and ekf')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/go/base_position_plot.png")
    plt.close(fig)

    #plot go base position and fb com position
    fig, ax = plt.subplots()
    ax.plot(t, go_base_position[:, 0], label='go Base position X', color='blue')
    ax.plot(t, fb_com_position[:, 0], label='FB COM position X', color='blue', linestyle = "--")
    ax.plot(t, go_base_position[:, 1], label='go Base position Y', color='orange')
    ax.plot(t, fb_com_position[:, 1], label='FB COM position Y', color='orange', linestyle = "--")
    ax.plot(t, go_base_position[:, 2], label='go Base position Z', color='green')
    ax.plot(t, fb_com_position[:, 2], label='FB COM position Z', color='green', linestyle = "--")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('go Base Position [m/s]')
    ax.set_title('Base Position of go estimation and FB COM position')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/go/base_position_fb_com_position_plot.png")
    plt.close(fig)

    #plot sim base position and sim com position
    fig, ax = plt.subplots()
    ax.plot(t, sim_base_position[:, 0], label='Sim Base position X', color='blue')
    ax.plot(t, sim_com_position[:, 0], label='Sim COM position X', color='blue', linestyle = "--")
    ax.plot(t, sim_base_position[:, 1], label='Sim Base position Y', color='orange')
    ax.plot(t, sim_com_position[:, 1], label='Sim COM position Y', color='orange', linestyle = "--")
    ax.plot(t, sim_base_position[:, 2], label='Sim Base position Z', color='green')
    ax.plot(t, sim_com_position[:, 2], label='Sim COM position Z', color='green', linestyle = "--") 
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Sim Base Position [m/s]')
    ax.set_title('Base Position of sim and Sim COM position')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/go/sim_base_position_sim_com_position_plot.png")
    plt.close(fig)

    #compare error between go base position and fb com position and sim base position and sim com position
    fig, ax = plt.subplots()
    ax.plot(t, go_base_position[:, 0] - fb_com_position[:, 0], label='go Base position X - FB COM position X', color='blue')
    ax.plot(t, go_base_position[:, 1] - fb_com_position[:, 1], label='go Base position Y - FB COM position Y', color='orange')
    ax.plot(t, go_base_position[:, 2] - fb_com_position[:, 2], label='go Base position Z - FB COM position Z', color='green')
    ax.plot(t, sim_base_position[:, 0] - sim_com_position[:, 0], label='Sim Base position X - Sim COM position X', color='blue', linestyle = "--")
    ax.plot(t, sim_base_position[:, 1] - sim_com_position[:, 1], label='Sim Base position Y - Sim COM position Y', color='orange', linestyle = "--")
    ax.plot(t, sim_base_position[:, 2] - sim_com_position[:, 2], label='Sim Base position Z - Sim COM position Z', color='green', linestyle = "--")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Error [m]')
    ax.set_title('Position Error Comparison between go-FB COM and sim-Sim COM')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/go/position_error_comparison_plot.png")
    plt.close(fig)


    # Plot go base orientation in quaternion format
    fig, ax = plt.subplots()
    ax.plot(t, go_base_orientation[:, 0] - ekf_base_orientation[:, 0], label='go Base Orientation W', color='blue')
    ax.plot(t, go_base_orientation[:, 1] - ekf_base_orientation[:, 1], label='go Base Orientation X', color='orange')
    ax.plot(t, go_base_orientation[:, 2] - ekf_base_orientation[:, 2], label='go Base Orientation Y', color='green')
    ax.plot(t, go_base_orientation[:, 3] - ekf_base_orientation[:, 3], label='go Base Orientation Z', color='red')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('go Base Orientation [Quaternion]')
    ax.set_title('Base Orientation Error between go estimation and ekf')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/go/base_orientation_error_plot.png")
    plt.close(fig)

    # Plot go base orientation in quaternion format
    fig, ax = plt.subplots()
    ax.plot(t, go_base_orientation_rpy[:, 0] - ekf_base_orientation_rpy[:, 0], label='go Base Orientation R', color='blue')
    ax.plot(t, go_base_orientation_rpy[:, 1] - ekf_base_orientation_rpy[:, 1], label='go Base Orientation P', color='orange')
    ax.plot(t, go_base_orientation_rpy[:, 2] - ekf_base_orientation_rpy[:, 2], label='go Base Orientation Y', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Base Orientation [RPY]')
    ax.set_title('Base Orientation Error between go estimation and ekf')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/go/base_orientation_rpy_error_plot.png")
    plt.close(fig)

    # Plot go base orientation in quaternion format
    fig, ax = plt.subplots()
    ax.plot(t, go_base_orientation_rpy[:, 0], label='go Base Orientation R', color='blue')
    ax.plot(t, go_base_orientation_rpy[:, 1], label='go Base Orientation P', color='orange')
    ax.plot(t, go_base_orientation_rpy[:, 2], label='go Base Orientation Y', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Base Orientation [RPY]')
    ax.set_title('Base Orientation Error between go estimation and sim')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/go/base_orientation_rpy_plot.png")
    plt.close(fig)

    # Plot go base orientation in quaternion format
    fig, ax = plt.subplots()
    ax.plot(t, go_base_orientation[:, 0] , label='go Base Orientation W', color='blue')
    ax.plot(t, go_base_orientation[:, 1], label='go Base Orientation X', color='orange')
    ax.plot(t, go_base_orientation[:, 2] , label='go Base Orientation Y', color='green')
    ax.plot(t, go_base_orientation[:, 3], label='go Base Orientation Z', color='red')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('go Base Orientation [Quaternion]')
    ax.set_title('Base Orientation go estimation ')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/go/base_orientation_plot.png")
    plt.close(fig)
    


    #################################
    #  COM AND ZMP PLOTS
    #################################
    #plot desired acceleration of com
    fig, ax = plt.subplots()
    ax.plot(t, des_com_acceleration[:, 0], label='des COM Acc X', color='blue')
    ax.plot(t, des_com_acceleration[:, 1], label='des COM Acc Y', color='orange')
    ax.plot(t, des_com_acceleration[:, 2], label='des COM Acc Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Acceleration [m/s²]')
    ax.set_title('COM Desired Acceleration')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/des_com_acceleration_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, fb_zmp_position[:, 0], label='used ZMP X', color='blue', linestyle=':')
    ax.plot(t, fb_zmp_position[:, 1], label='used ZMP Y', color='orange', linestyle=':')
    ax.plot(t, fb_zmp_position[:, 1], label='used ZMP Y', color='green', linestyle=':')
    ax.plot(t, ef_zmp_position[:, 0], label='not used ZMP X', color='blue', linestyle='--')
    ax.plot(t, ef_zmp_position[:, 1], label='not used ZMP Y', color='orange', linestyle='--')
    ax.plot(t, ef_zmp_position[:, 2], label='not used ZMP Y', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('ZMP X & Y Position Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/fb_used_and_not_used_zmp_plot.png")

    fig, ax = plt.subplots()
    ax.plot(t, des_zmp_position[:, 0], label='des ZMP X', color='blue')
    ax.plot(t, des_zmp_position[:, 1], label='des ZMP Y', color='orange')
    ax.plot(t, des_zmp_position[:, 2], label='des ZMP Z', color='green')
    ax.plot(t, des_com_position[:, 0], label='des COM X', color='blue', linestyle='-.')
    ax.plot(t, des_com_position[:, 1], label='des COM Y', color='orange', linestyle='-.')
    ax.plot(t, des_com_position[:, 2], label='des COM Z', color='green', linestyle='-.')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('ZMP and COM Position Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/des_zmp_and_com_plot.png")

    fig, ax = plt.subplots()
    ax.plot(t, fb_com_position[:, 0], label='FB COM X', color='blue')
    ax.plot(t, fb_com_position[:, 1], label='FB COM Y', color='orange')
    ax.plot(t, fb_com_position[:, 2], label='FB COM Z', color='green')
    ax.plot(t, des_com_position[:, 0], label='des COM X', color='blue', linestyle='-.')
    ax.plot(t, des_com_position[:, 1], label='des COM Y', color='orange', linestyle='-.')
    ax.plot(t, des_com_position[:, 2], label='des COM Z', color='green', linestyle='-.')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('ZMP and COM Position Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/des_vs_fb_com_plot.png")

    #plot x position of com, left and right sole
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 0], label='fb left sole X', color='blue')
    ax.plot(t, p_rsole_fb[:, 0], label='fb right sole X', color='orange')
    ax.plot(t, kf_com_position[:, 0], label='kf COM X', color='green')
    ax.plot(t, kf_zmp_position[:, 0], label='kf ZMP X', color='red')
    ax.plot(t, p_lsole_des[:, 0], label='des left sole X', color='blue', linestyle='--')
    ax.plot(t, p_rsole_des[:, 0], label='des right sole X', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 0], label='des COM X', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position X [m]')
    ax.set_title('Left Sole, Right Sole and COM X Position Filtered vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_left_right_sole_and_com_x_plot.png")

    #plot y position of com, left and right sole
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 1], label='fb left sole Y', color='blue')
    ax.plot(t, p_rsole_fb[:, 1], label='fb right sole Y', color='orange')
    ax.plot(t, kf_com_position[:, 1], label='kf COM Y', color='green')
    ax.plot(t, kf_zmp_position[:, 1], label='kf ZMP Y', color='red')
    ax.plot(t, p_lsole_des[:, 1], label='des left sole Y', color='blue', linestyle='--')
    ax.plot(t, p_rsole_des[:, 1], label='des right sole Y', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 1], label='des COM Y', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Y [m]')
    ax.set_title('Left Sole, Right Sole and COM Y Position Filtered vs Desired')
    ax.grid(True)
    #lim between 24 and 26
    # ax.set_xlim([27.99, 28.02])
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_left_right_sole_and_com_y_plot.png")

    #plot x position of com, left and right sole
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 2], label='fb left sole z', color='blue')
    ax.plot(t, p_rsole_fb[:, 2], label='fb right sole z', color='orange')
    ax.plot(t, kf_com_position[:, 2], label='kf COM z', color='green')
    ax.plot(t, kf_zmp_position[:, 2], label='kf ZMP z', color='red')
    ax.plot(t, p_lsole_des[:, 2], label='des left sole z', color='blue', linestyle='--')
    ax.plot(t, p_rsole_des[:, 2], label='des right sole z', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 2], label='des COM z', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position X [m]')
    ax.set_title('Left Sole, Right Sole and COM X Position Filtered vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_left_right_sole_and_com_z_plot.png")

    #plot y position of com, left and right sole
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 2], label='fb left sole Z', color='blue')
    ax.plot(t, p_rsole_fb[:, 2], label='fb right sole Z', color='orange')
    ax.plot(t, fb_com_position[:, 2], label='fb COM Z', color='green')
    ax.plot(t, fb_zmp_position[:, 2], label='fb ZMP Z', color='red')
    ax.plot(t, p_lsole_des[:, 2], label='des left sole Z', color='blue', linestyle='--')
    ax.plot(t, p_rsole_des[:, 2], label='des right sole Z', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 2], label='des COM Z', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Y [m]')
    ax.set_title('Left Sole, Right Sole and COM Y Position Filtered vs Desired')
    ax.grid(True)
    #lim between 24 and 26
    # ax.set_xlim([27.99, 28.02])
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/fb_left_right_sole_and_com_z_plot.png")

    #plot x position of com, left and right sole
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 0], label='fb left sole X', color='blue')
    ax.plot(t, p_rsole_fb[:, 0], label='fb right sole X', color='orange')
    ax.plot(t, fb_com_position[:, 0], label='fb COM X', color='green')
    ax.plot(t, fb_zmp_position[:, 0], label='fb ZMP X', color='red')
    ax.plot(t, p_lsole_des[:, 0], label='des left sole X', color='blue', linestyle='--')
    ax.plot(t, p_rsole_des[:, 0], label='des right sole X', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 0], label='des COM X', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position X [m]')
    ax.set_title('Left Sole, Right Sole and COM X Position Feedback vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/fb_left_right_sole_and_com_x_plot.png")

    #plot y position of com, left and right sole
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 1], label='fb left sole Y', color='blue')
    ax.plot(t, p_rsole_fb[:, 1], label='fb right sole Y', color='orange')
    ax.plot(t, fb_com_position[:, 1], label='fb COM Y', color='green')
    ax.plot(t, fb_zmp_position[:, 1], label='fb ZMP Y', color='red')
    ax.plot(t, p_lsole_des[:, 1], label='des left sole Y', color='blue', linestyle='--')
    ax.plot(t, p_rsole_des[:, 1], label='des right sole Y', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 1], label='des COM Y', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Y [m]')
    ax.set_title('Left Sole, Right Sole and COM Y Position Feedback vs Desired')
    ax.grid(True)
    #lim between 24 and 26
    # ax.set_xlim([27.99, 28.02])
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/fb_left_right_sole_and_com_y_plot.png")

    fig, ax = plt.subplots()
    ax.plot(t, kf_zmp_position[:, 0], label='kf ZMP X', color='blue')
    ax.plot(t, kf_zmp_position[:, 1], label='kf ZMP Y', color='orange')
    ax.plot(t, kf_com_position[:, 0], label='kf COM X', color='blue', linestyle='-.')
    ax.plot(t, kf_com_position[:, 1], label='kf COM Y', color='orange', linestyle='-.')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('ZMP and COM X & Y Position Filtered')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_zmp_and_com_plot.png")


    fig, ax = plt.subplots()
    ax.plot(t, fb_zmp_position[:, 0], label='FB ZMP X', color='blue')
    ax.plot(t, fb_zmp_position[:, 1], label='FB ZMP Y', color='orange')
    ax.plot(t, fb_zmp_position[:, 2], label='FB ZMP Z', color='green')
    ax.plot(t, des_zmp_position[:, 0], label='Des ZMP X', color='blue', linestyle=':')
    ax.plot(t, des_zmp_position[:, 1], label='Des ZMP Y', color='orange', linestyle=':')
    ax.plot(t, des_zmp_position[:, 2], label='Des ZMP Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('ZMP Position [m]')
    ax.set_title('ZMP Position Simulation vs Feedback vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/fb_vs_des_zmp_plot.png")
    plt.close(fig)

    #plot filtered zmp sim and fb
    fig, ax = plt.subplots()
    ax.plot(t, kf_zmp_position[:, 0], label='kf ZMP X', color='blue')
    ax.plot(t, kf_zmp_position[:, 1], label='kf ZMP Y', color='orange')
    ax.plot(t, kf_zmp_position[:, 2], label='kf ZMP Z', color='green')
    ax.plot(t, des_zmp_position[:, 0], label='Des ZMP X', color='blue', linestyle=':')
    ax.plot(t, des_zmp_position[:, 1], label='Des ZMP Y', color='orange', linestyle=':')
    ax.plot(t, des_zmp_position[:, 2], label='Des ZMP Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Filtered ZMP Position [m]')
    ax.set_title('Filtered vs Desired ZMP Position')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_vs_des_zmp_plot.png")
    plt.close(fig)

    #plot filtered com sim and fb
    fig, ax = plt.subplots()
    ax.plot(t, kf_com_position[:, 0], label='Kf COM X', color='blue')
    ax.plot(t, kf_com_position[:, 1], label='Kf COM Y', color='orange')
    ax.plot(t, kf_com_position[:, 2], label='Kf COM Z', color='green')
    ax.plot(t, des_com_position[:, 0], label='Des COM X', color='blue', linestyle=':')
    ax.plot(t, des_com_position[:, 1], label='Des COM Y', color='orange', linestyle=':')
    ax.plot(t, des_com_position[:, 2], label='Des COM Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('Filtered vs Desired X & Y COM Position')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_vs_des_com_plot.png")
    plt.close(fig)

    # plot com velocities sim and fb
    fig, ax = plt.subplots()
    ax.plot(t, fb_com_velocity[:, 0], label='FB COM Vel X', color='blue')
    ax.plot(t, fb_com_velocity[:, 1], label='FB COM Vel Y', color='orange')
    ax.plot(t, fb_com_velocity[:, 2], label='FB COM Vel Z', color='green')
    ax.plot(t, des_com_velocity[:, 0], label='Des COM Vel X', color='blue', linestyle=':')
    ax.plot(t, des_com_velocity[:, 1], label='Des COM Vel Y', color='orange', linestyle=':')
    ax.plot(t, des_com_velocity[:, 2], label='Des COM Vel Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('COM Velocity Simulation vs Feedback vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/fb_vs_des_com_velocity_plot.png")
    plt.close(fig)

    # plot filtered com velocities sim and fb
    fig, ax = plt.subplots()
    ax.plot(t, kf_com_velocity[:, 0], label='Kf COM Vel X', color='blue')
    ax.plot(t, kf_com_velocity[:, 1], label='Kf COM Vel Y', color='orange')
    ax.plot(t, kf_com_velocity[:, 2], label='Kf COM Vel Z', color='green')
    ax.plot(t, des_com_velocity[:, 0], label='Des COM Vel X', color='blue', linestyle=':')
    ax.plot(t, des_com_velocity[:, 1], label='Des COM Vel Y', color='orange', linestyle=':')
    ax.plot(t, des_com_velocity[:, 2], label='Des COM Vel Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Filtered vs Desired COM Velocity')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_vs_des_com_velocity_plot.png")
    plt.close(fig)


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

    #plot des, sim and fb rsole position
    fig, ax = plt.subplots()
    ax.plot(t, fb_lsole_orientation[:, 0], label='FB Right Sole X', color='orange')
    ax.plot(t, des_lsole_orientation[:, 0], label='Des Right Sole X', color='orange', linestyle=':')
    ax.plot(t, fb_lsole_orientation[:, 1], label='FB Right Sole Y', color='blue')
    ax.plot(t, des_lsole_orientation[:, 1], label='Des Right Sole Y', color='blue', linestyle=':')
    ax.plot(t, fb_lsole_orientation[:, 2], label='FB Right Sole Z', color='green')
    ax.plot(t, des_lsole_orientation[:, 2], label='Des Right Sole Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Right Sole Position [m]')
    ax.set_title('Left and Right Sole Position Simulation vs Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/soles/lsole_orientation_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, fb_rsole_orientation[:, 0], label='FB Right Sole X', color='green')
    ax.plot(t, des_rsole_orientation[:, 0], label='Des Right Sole X', color='green', linestyle=':')
    ax.plot(t, fb_rsole_orientation[:, 1], label='FB Right Sole Y', color='blue')
    ax.plot(t, des_rsole_orientation[:, 1], label='Des Right Sole Y', color='blue', linestyle=':')
    ax.plot(t, fb_rsole_orientation[:, 2], label='FB Right Sole Z', color='red')
    ax.plot(t, des_rsole_orientation[:, 2], label='Des Right Sole Z', color='red', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Right Sole Position [m]')
    ax.set_title('Left and Right Sole Position Simulation vs Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/soles/rsole_orientation_plot.png")
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

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, measured_joint_position[:, i], label=joint_names[i])
            ax.plot(t, ekf_joint_position[:, i], label=f"{joint_names[i].strip()} EKF", linestyle=':')
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
            ax.plot(t, measured_joint_velocity[:, i], label=joint_names[i])
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

    #plot execution times over the itarations, first in different plots, then summed up in a single plot with a line at 2000
    figs = []
    exec_times = {
        'EKF': execution_time_ekf,
        'KF': execution_time_kf,
        'MPC': execution_time_mpc,
        'WBC': execution_time_wbc,
        'Update': execution_time_update
    }
    for name, times in exec_times.items():
        fig, ax = plt.subplots()
        ax.plot(times, label=f'{name} Execution Time', color='blue')
        if name == 'Update':
            ax.axhline(y=2000, color='r', linestyle='--', label='2000 microseconds')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Execution Time [microseconds]')
        ax.set_title(f'{name} Execution Time per Iteration')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        fig.savefig(f"images/execution_times/{name}_execution_time_plot.png")
        plt.close(fig)

    #plot the sum of each execution time
    total_execution_time = (execution_time_ekf + execution_time_kf + execution_time_mpc + execution_time_wbc)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(total_execution_time, label='Total Execution Time', color='blue')
    ax.axhline(y=2000, color='r', linestyle='--', label='2000 microseconds')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Execution Time [microseconds]')
    ax.set_title('Total Execution Time per Iteration')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/execution_times/total_execution_time_plot.png")
    plt.close(fig)
