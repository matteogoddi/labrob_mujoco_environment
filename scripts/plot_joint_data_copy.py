import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform
from math import ceil, floor, sqrt
from collections import defaultdict
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
import os

if __name__ == '__main__':
    def load_optional_matrix(path):
        if not os.path.exists(path):
            return None
        if os.path.getsize(path) == 0:
            return None
        try:
            data = np.loadtxt(path)
        except (ValueError, OSError):
            return None
        if np.size(data) == 0:
            return None
        data = np.atleast_2d(data)
        return data

    #request input from terminal
    number = input("Enter 0 to plot data from the last simulation or the number of the experiment: ")
    if number == '0':
        folder = '/tmp'
    else:
        folder = 'experiments/experiment_' + number
    
    joint_names = open(folder + '/joint_names.txt').readlines()
    input_torque: np.ndarray = np.loadtxt(folder +'/input_torque.txt')
    wrist_states = {
        'left_wrist_roll': load_optional_matrix(folder + '/left_wrist_roll_state.txt'),
        'left_wrist_pitch': load_optional_matrix(folder + '/left_wrist_pitch_state.txt'),
        'left_wrist_yaw': load_optional_matrix(folder + '/left_wrist_yaw_state.txt'),
        'right_wrist_roll': load_optional_matrix(folder + '/right_wrist_roll_state.txt'),
        'right_wrist_pitch': load_optional_matrix(folder + '/right_wrist_pitch_state.txt'),
        'right_wrist_yaw': load_optional_matrix(folder + '/right_wrist_yaw_state.txt')
    }
    if wrist_states['left_wrist_roll'] is None:
        wrist_states['left_wrist_roll'] = load_optional_matrix(folder + '/left_wrist_state.txt')
    if wrist_states['right_wrist_roll'] is None:
        wrist_states['right_wrist_roll'] = load_optional_matrix(folder + '/right_wrist_state.txt')

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

    ef_zmp_position = np.loadtxt(folder + '/ef_zmp_position.txt')

    # base_estimate = np.loadtxt(folder + '/base_estimate.txt')
    # orientation_estimate = np.loadtxt(folder + '/orientation_estimate.txt')
    # left_foot_position_base_estimation = np.loadtxt(folder + '/left_foot_position_base_estimation.txt')
    # right_foot_position_base_estimation = np.loadtxt(folder + '/right_foot_position_base_estimation.txt')
    # left_foot_position_with_zero_base = np.loadtxt(folder + '/left_foot_position_with_zero_base.txt')
    # right_foot_position_with_zero_base = np.loadtxt(folder + '/right_foot_position_with_zero_base.txt')

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

    estimated_force_lsole = np.loadtxt(folder + '/estimated_force_lsole.txt')
    estimated_force_rsole = np.loadtxt(folder + '/estimated_force_rsole.txt')

    ekf_base_position = np.loadtxt(folder + '/ekf_base_position.txt')
    ekf_base_velocity = np.loadtxt(folder + '/ekf_base_velocity.txt')
    ekf_base_orientation = np.loadtxt(folder + '/ekf_base_orientation.txt')
    ekf_base_angular_velocity = np.loadtxt(folder + '/ekf_base_angular_velocity.txt')
    ekf_joint_position = np.loadtxt(folder + '/ekf_joint_position.txt')
    ekf_joint_velocity = np.loadtxt(folder + '/ekf_joint_velocity.txt')
    sim_base_position = np.loadtxt(folder + '/sim_base_position.txt')
    sim_base_velocity = np.loadtxt(folder + '/sim_base_velocity.txt')
    sim_base_orientation = np.loadtxt(folder + '/sim_base_orientation.txt')
    sim_base_angular_velocity = np.loadtxt(folder + '/sim_base_angular_velocity.txt')
    sim_joint_position: np.ndarray = np.loadtxt(folder + '/sim_joint_position.txt')
    sim_joint_velocity: np.ndarray = np.loadtxt(folder + '/sim_joint_velocity.txt')

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
    # estimated_imu_accelerometer: np.ndarray = np.loadtxt(folder + '/estimated_imu_accelerometer.txt')
    # estimated_imu_angular_velocity: np.ndarray = np.loadtxt(folder + '/estimated_imu_angular_velocity.txt')
    # estimated_imu_orientation: np.ndarray = np.loadtxt(folder + '/estimated_imu_orientation.txt')


    num_samples = sim_joint_position.shape[0] - 10
    input_torque = input_torque[:num_samples, :]

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

    ef_zmp_position = ef_zmp_position[:num_samples, :]

    # base_estimate = base_estimate[:num_samples, :]
    # orientation_estimate = orientation_estimate[:num_samples, :]
    # left_foot_position_base_estimation = left_foot_position_base_estimation[:num_samples, :]
    # right_foot_position_base_estimation = right_foot_position_base_estimation[:num_samples, :]
    # left_foot_position_with_zero_base = left_foot_position_with_zero_base[:num_samples, :]
    # right_foot_position_with_zero_base = right_foot_position_with_zero_base[:num_samples, :]

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
    
    ekf_base_position = ekf_base_position[:num_samples, :]
    ekf_base_velocity = ekf_base_velocity[:num_samples, :]
    ekf_base_orientation = ekf_base_orientation[:num_samples, :]
    ekf_base_angular_velocity = ekf_base_angular_velocity[:num_samples, :]
    ekf_joint_position = ekf_joint_position[:num_samples, :]
    ekf_joint_velocity = ekf_joint_velocity[:num_samples, :]
    sim_base_position = sim_base_position[:num_samples, :]
    sim_base_velocity = sim_base_velocity[:num_samples, :]
    sim_base_orientation = sim_base_orientation[:num_samples, :]
    sim_base_angular_velocity = sim_base_angular_velocity[:num_samples, :]
    sim_joint_position = sim_joint_position[:num_samples, :]
    sim_joint_velocity = sim_joint_velocity[:num_samples, :]
    measured_joint_position = measured_joint_position[:num_samples, :]
    measured_joint_velocity = measured_joint_velocity[:num_samples, :]
    measured_imu_orientation = measured_imu_orientation[:num_samples, :]
    measured_imu_angular_velocity = measured_imu_angular_velocity[:num_samples, :]
    measured_imu_accelerometer = measured_imu_accelerometer[:num_samples, :]
    for key, value in wrist_states.items():
        if value is not None:
            wrist_states[key] = value[:num_samples, :]

    execution_time_ekf = execution_time_ekf[:num_samples]
    execution_time_kf = execution_time_kf[:num_samples]
    execution_time_mpc = execution_time_mpc[:num_samples]
    execution_time_wbc = execution_time_wbc[:num_samples]
    execution_time_update = execution_time_update[:num_samples]
    
    #add 1000 zeros to estimated imu
    # estimated_imu_accelerometer = np.vstack((np.zeros((2000, 3)), estimated_imu_accelerometer))
    # estimated_imu_angular_velocity = np.vstack((np.zeros((2000, 3)), estimated_imu_angular_velocity))
    # estimated_imu_orientation = np.vstack((np.zeros((2000, 4)), estimated_imu_orientation))
    # estimated_imu_accelerometer = estimated_imu_accelerometer[:num_samples, :]
    # estimated_imu_angular_velocity = estimated_imu_angular_velocity[:num_samples, :]
    # estimated_imu_orientation = estimated_imu_orientation[:num_samples, :]
    
        
    delta = 1 / 500  # Assuming a control frequency of 500 Hz
    t = np.linspace(0.0, delta * num_samples, num_samples)

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
    if not os.path.exists('images/forces_torques/wrists'):
        os.makedirs('images/forces_torques/wrists')
    if not os.path.exists('images/soles'):
        os.makedirs('images/soles')

    grouped_indices = defaultdict(list)

    for idx, name in enumerate(joint_names):
        base_name = '_'.join(name.split('_')[:2])  # E.g., "left_ankle" da "left_ankle_roll_joint"
        grouped_indices[base_name].append(idx)

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

    for wrist_name, wrist_state in wrist_states.items():
        if wrist_state is None or wrist_state.shape[1] < 4:
            continue

        wrist_samples = min(num_samples, wrist_state.shape[0])
        t_wrist = t[:wrist_samples]
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        labels = ['q [rad]', 'dq [rad/s]', 'ddq [rad/s^2]', 'tau [Nm]']
        wrist_title = wrist_name.replace('_', ' ').title()
        color = 'tab:blue' if wrist_name.startswith('left_') else 'tab:orange'
        for i in range(4):
            axs[i].plot(t_wrist, wrist_state[:wrist_samples, i], color=color)
            axs[i].set_ylabel(labels[i])
            axs[i].set_title(f'{wrist_title} {labels[i].split()[0]}')
            axs[i].grid(True)
        axs[-1].set_xlabel('Time [s]')
        fig.tight_layout()
        fig.savefig(f'images/forces_torques/wrists/{wrist_name}_state.png')
        plt.close(fig)



    #################################
    #  COM AND ZMP PLOTS
    #################################
    fig, ax = plt.subplots()
    ax.plot(t, fb_zmp_position[:, 0], label='used ZMP X', color='blue', linestyle=':')
    ax.plot(t, fb_zmp_position[:, 1], label='used ZMP Y', color='orange', linestyle=':')
    ax.plot(t, ef_zmp_position[:, 0], label='not used ZMP X', color='blue', linestyle='--')
    ax.plot(t, ef_zmp_position[:, 1], label='not used ZMP Y', color='orange', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('ZMP X & Y Position Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/fb_used_and_not_used_zmp_plot.png")

    fig, ax = plt.subplots()
    ax.plot(t, des_zmp_position[:, 0], label='des ZMP X', color='blue', linestyle=':')
    ax.plot(t, des_zmp_position[:, 1], label='des ZMP Y', color='orange', linestyle=':')
    ax.plot(t, des_com_position[:, 0], label='des COM X', color='blue', linestyle='-.')
    ax.plot(t, des_com_position[:, 1], label='des COM Y', color='orange', linestyle='-.')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('ZMP and COM Position Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/des_zmp_and_com_plot.png")

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
    # ax.plot(t, sim_com_position[:, 1], label='sim COM Y', color='green')
    # ax.plot(t, sim_zmp_position[:, 1], label='sim ZMP Y', color='red')
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
    # ax.plot(t, sim_com_position[:, 2], label='sim COM Z', color='green')
    # ax.plot(t, sim_zmp_position[:, 2], label='sim ZMP Z', color='red')
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
    ax.plot(t, kf_zmp_position[:, 0], label='kf ZMP X', color='blue', linestyle=':')
    ax.plot(t, kf_zmp_position[:, 1], label='kf ZMP Y', color='orange', linestyle=':')
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
    ax.plot(t, sim_zmp_position[:, 0], label='Sim ZMP X', color='blue')
    ax.plot(t, sim_zmp_position[:, 1], label='Sim ZMP Y', color='orange')
    ax.plot(t, sim_zmp_position[:, 2], label='Sim ZMP Z', color='green')
    ax.plot(t, fb_zmp_position[:, 0], label='FB ZMP X', color='blue', linestyle='--')
    ax.plot(t, fb_zmp_position[:, 1], label='FB ZMP Y', color='orange', linestyle='--')
    ax.plot(t, fb_zmp_position[:, 2], label='FB ZMP Z', color='green', linestyle='--')
    ax.plot(t, des_zmp_position[:, 0], label='Des ZMP X', color='blue', linestyle=':')
    ax.plot(t, des_zmp_position[:, 1], label='Des ZMP Y', color='orange', linestyle=':')
    ax.plot(t, des_zmp_position[:, 2], label='Des ZMP Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('ZMP Position [m]')
    ax.set_title('ZMP Position Simulation vs Feedback vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/sim_vs_fb_vs_des_zmp_plot.png")
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

    #plot com sim and fb
    fig, ax = plt.subplots()
    ax.plot(t, sim_com_position[:, 0], label='Sim COM X', color='blue')
    ax.plot(t, sim_com_position[:, 1], label='Sim COM Y', color='orange')
    ax.plot(t, fb_com_position[:, 0], label='FB COM X', color='blue', linestyle='--')
    ax.plot(t, fb_com_position[:, 1], label='FB COM Y', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 0], label='Des COM X', color='blue', linestyle=':')
    ax.plot(t, des_com_position[:, 1], label='Des COM Y', color='orange', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('COM Position [m]')
    ax.set_title('COM Position Simulation vs Feedback vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/sim_vs_fb_vs_des_com_plot.png")
    plt.close(fig)

    #plot filtered com sim and fb
    fig, ax = plt.subplots()
    ax.plot(t, kf_com_position[:, 0], label='Kf COM X', color='blue')
    ax.plot(t, kf_com_position[:, 1], label='Kf COM Y', color='orange')
    ax.plot(t, des_com_position[:, 0], label='Des COM X', color='blue', linestyle=':')
    ax.plot(t, des_com_position[:, 1], label='Des COM Y', color='orange', linestyle=':')
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
    ax.plot(t, sim_com_velocity[:, 0], label='Sim COM Vel X', color='blue')
    ax.plot(t, sim_com_velocity[:, 1], label='Sim COM Vel Y', color='orange')
    ax.plot(t, fb_com_velocity[:, 0], label='FB COM Vel X', color='blue', linestyle='--')
    ax.plot(t, fb_com_velocity[:, 1], label='FB COM Vel Y', color='orange', linestyle='--')
    ax.plot(t, des_com_velocity[:, 0], label='Des COM Vel X', color='blue', linestyle=':')
    ax.plot(t, des_com_velocity[:, 1], label='Des COM Vel Y', color='orange', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('COM Velocity Simulation vs Feedback vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/sim_vs_fb_vs_des_com_velocity_plot.png")
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
    ax.plot(t, p_lsole_sim[:, 0], label='Sim Left Sole X', color='blue')
    ax.plot(t, p_lsole_sim[:, 1], label='Sim Left Sole Y', color='orange')
    ax.plot(t, p_lsole_sim[:, 2], label='Sim Left Sole Z', color='green')
    ax.plot(t, p_lsole_fb[:, 0], label='FB Left Sole X', color='blue', linestyle='--')
    ax.plot(t, p_lsole_fb[:, 1], label='FB Left Sole Y', color='orange', linestyle='--')
    ax.plot(t, p_lsole_fb[:, 2], label='FB Left Sole Z', color='green', linestyle='--')
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
    ax.plot(t, p_rsole_sim[:, 0], label='Sim Right Sole X', color='blue')
    ax.plot(t, p_rsole_sim[:, 1], label='Sim Right Sole Y', color='orange')
    ax.plot(t, p_rsole_sim[:, 2], label='Sim Right Sole Z', color='green')
    ax.plot(t, p_rsole_fb[:, 0], label='FB Right Sole X', color='blue', linestyle='--')
    ax.plot(t, p_rsole_fb[:, 1], label='FB Right Sole Y', color='orange', linestyle='--')
    ax.plot(t, p_rsole_fb[:, 2], label='FB Right Sole Z', color='green', linestyle='--')
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

    #plot des, sim and fb lsole velocity
    fig, ax = plt.subplots()
    ax.plot(t, v_lsole_sim[:, 0], label='Sim Left Sole Vel X', color='blue')
    ax.plot(t, v_lsole_sim[:, 1], label='Sim Left Sole Vel Y', color='orange')
    ax.plot(t, v_lsole_sim[:, 2], label='Sim Left Sole Vel Z', color='green')
    ax.plot(t, v_lsole_fb[:, 0], label='FB Left Sole Vel X', color='blue', linestyle='--')
    ax.plot(t, v_lsole_fb[:, 1], label='FB Left Sole Vel Y', color='orange', linestyle='--')
    ax.plot(t, v_lsole_fb[:, 2], label='FB Left Sole Vel Z', color='green', linestyle='--')
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
    ax.plot(t, v_rsole_sim[:, 0], label='Sim Right Sole Vel X', color='blue')
    ax.plot(t, v_rsole_sim[:, 1], label='Sim Right Sole Vel Y', color='orange')
    ax.plot(t, v_rsole_sim[:, 2], label='Sim Right Sole Vel Z', color='green')
    ax.plot(t, v_rsole_fb[:, 0], label='FB Right Sole Vel X', color='blue', linestyle='--')
    ax.plot(t, v_rsole_fb[:, 1], label='FB Right Sole Vel Y', color='orange', linestyle='--')
    ax.plot(t, v_rsole_fb[:, 2], label='FB Right Sole Vel Z', color='green', linestyle='--')
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
    ax.plot(t, ekf_base_position[:, 0] - sim_base_position[:, 0], label='EKF Base Position X', color='blue')
    ax.plot(t, ekf_base_position[:, 1] - sim_base_position[:, 1], label='EKF Base Position Y', color='orange')
    ax.plot(t, ekf_base_position[:, 2] - sim_base_position[:, 2], label='EKF Base Position Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Position [m]')
    ax.set_title('Base Position Error between EKF estimation and Simulation')
    ax.grid(True)
    ax.legend()
    # fig.tight_layout()
    fig.savefig("images/ekf/base_position_error_plot.png")
    plt.close(fig)

    # Plot EKF base velocity
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_velocity[:, 0] - sim_base_velocity[:, 0], label='EKF Base Velocity X', color='blue')
    ax.plot(t, ekf_base_velocity[:, 1] - sim_base_velocity[:, 1], label='EKF Base Velocity Y', color='orange')
    ax.plot(t, ekf_base_velocity[:, 2] - sim_base_velocity[:, 2], label='EKF Base Velocity Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Velocity [m/s]')
    ax.set_title('Base Velocity Error between EKF estimation and Simulation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_velocity_error_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_position[:, 0], label='EKF Base position X', color='blue')
    ax.plot(t, sim_base_position[:, 0], label='Base position X', color='blue', linestyle = "--")
    ax.plot(t, ekf_base_position[:, 1], label='EKF Base position Y', color='orange')
    ax.plot(t, sim_base_position[:, 1], label='Base position Y', color='orange', linestyle = "--")
    ax.plot(t, ekf_base_position[:, 2] - sim_base_position[:, 2], label='EKF Base position Z', color='green')
    ax.plot(t, sim_base_position[:, 2] - sim_base_position[:, 2], label='Base position Z', color='green', linestyle = "--")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Position [m/s]')
    ax.set_title('Base Position of EKF estimation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_position_plot.png")
    plt.close(fig)

    # Plot EKF base orientation in quaternion format
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_orientation[:, 0] - sim_base_orientation[:, 0], label='EKF Base Orientation W', color='blue')
    ax.plot(t, ekf_base_orientation[:, 1] - sim_base_orientation[:, 1], label='EKF Base Orientation X', color='orange')
    ax.plot(t, ekf_base_orientation[:, 2] - sim_base_orientation[:, 2], label='EKF Base Orientation Y', color='green')
    ax.plot(t, ekf_base_orientation[:, 3] - sim_base_orientation[:, 3], label='EKF Base Orientation Z', color='red')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Orientation [Quaternion]')
    ax.set_title('Base Orientation Error between EKF estimation and Simulation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_orientation_error_plot.png")
    plt.close(fig)

    #plot ekf orientation in euler angles
    ekf_base_orientation_euler = R.from_quat(ekf_base_orientation).as_euler('xyz', degrees=False)
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_orientation_euler[:, 0], label='EKF Base Orientation Roll', color='blue')
    ax.plot(t, ekf_base_orientation_euler[:, 1], label='EKF Base Orientation Pitch', color='orange')
    ax.plot(t, ekf_base_orientation_euler[:, 2], label='EKF Base Orientation Yaw', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Orientation [Euler angles]')
    ax.set_title('Base Orientation EKF estimation in Euler angles')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_orientation_euler_plot.png")
    plt.close(fig)

    #plot error between orientation and simulation in euler angles 
    fig, ax = plt.subplots()
    sim_base_orientation_euler = R.from_quat(sim_base_orientation).as_euler('xyz', degrees=False)
    #check every value of the error with a for and if a value is greater than 5 radiants, remove 6.28
    for i in range(len(ekf_base_orientation_euler)):
        for j in range(3):
            error = ekf_base_orientation_euler[i, j] - sim_base_orientation_euler[i, j]
            if abs(error) > 3.14:
                if error > 0:
                    ekf_base_orientation_euler[i, j] -= 2 * 3.14
                else:
                    ekf_base_orientation_euler[i, j] += 2 * 3.14
    ax.plot(t, ekf_base_orientation_euler[:, 0] - sim_base_orientation_euler[:, 0], label='EKF Base Orientation Roll', color='blue')
    ax.plot(t, ekf_base_orientation_euler[:, 1] - sim_base_orientation_euler[:, 1], label='EKF Base Orientation Pitch', color='orange')
    ax.plot(t, ekf_base_orientation_euler[:, 2] - sim_base_orientation_euler[:, 2], label='EKF Base Orientation Yaw', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Orientation Error [Euler angles]')
    ax.set_title('Base Orientation Error between EKF estimation and Simulation in Euler angles')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_orientation_euler_error_plot.png")
    plt.close(fig)

    # Plot EKF base angular velocity
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_angular_velocity[:, 0] - sim_base_angular_velocity[:, 0], label='EKF Base Angular Velocity X', color='blue')
    ax.plot(t, ekf_base_angular_velocity[:, 1] - sim_base_angular_velocity[:, 1], label='EKF Base Angular Velocity Y', color='orange')
    ax.plot(t, ekf_base_angular_velocity[:, 2] - sim_base_angular_velocity[:, 2], label='EKF Base Angular Velocity Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Angular Velocity [rad/s]')
    ax.set_title('Angular Velocity Error between EKF estimation and Simulation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_angular_velocity_error_plot.png")
    plt.close(fig)

    # Plot position error between ekf joint position and simulated joint position
    fig, ax = plt.subplots(figsize=(18, 12))
    num_joints = sim_joint_position.shape[1]
    colormap = plt.colormaps['tab10'] 
    line_styles = ['-', '--', '-.', ':']
    for i in range(num_joints):
        color = colormap(i % 10)
        linestyle = line_styles[(i // 10) % len(line_styles)]  # cambia stile ogni 10 joint
        error = ekf_joint_position[:, i] - sim_joint_position[:, i]
        ax.plot(t, error,
                label=joint_names[i].strip(),
                color=color,
                linestyle=linestyle,
                linewidth=2)
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Position [rad]', fontsize=14)
    ax.set_title('Joint Position Error between EKF estimation and Simulation', fontsize=16)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=12, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.savefig("images/ekf/joint_position_error_plot.png")
    plt.close(fig)


    # Plot velocity error between ekf joint velocity and simulated joint velocity
    fig, ax = plt.subplots(figsize=(18, 12))
    colormap = plt.colormaps['tab10'] 
    line_styles = ['-', '--', '-.', ':']
    for i in range(num_joints):
        color = colormap(i % 10)
        linestyle = line_styles[(i // 10) % len(line_styles)]  # cambia stile ogni 10 joint
        error = ekf_joint_velocity[:, i] - sim_joint_velocity[:, i]
        ax.plot(t, error,
                label=joint_names[i].strip(),
                color=color,
                linestyle=linestyle,
                linewidth=2)
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Velocity [rad/s]', fontsize=14)
    ax.set_title('Joint Velocity Error between EKF estimation and Simulation', fontsize=16)
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

    # # plot imu orientation estimated
    # fig, ax = plt.subplots()
    # ax.plot(t, estimated_imu_orientation[:, 0], label='estimated IMU Orientation W', color='blue')
    # ax.plot(t, estimated_imu_orientation[:, 1], label='estimated IMU Orientation X', color='orange')
    # ax.plot(t, estimated_imu_orientation[:, 2], label='estimated IMU Orientation Y', color='green')
    # ax.plot(t, estimated_imu_orientation[:, 3], label='estimated IMU Orientation Z', color='red')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('estimated IMU Orientation [Quaternion]')
    # ax.set_title('estimated IMU Orientation')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig("images/ekf/estimated_imu_orientation_plot.png")
    # plt.close(fig)

    # # plot imu accelerometer estimated
    # # fig, ax = plt.subplots()
    # # ax.plot(t, estimated_imu_accelerometer[:, 0], label='estimated IMU Accelerometer X', color='blue')
    # # ax.plot(t, estimated_imu_accelerometer[:, 1], label='estimated IMU Accelerometer Y', color='orange')
    # # ax.plot(t, estimated_imu_accelerometer[:, 2], label='estimated IMU Accelerometer Z', color='green')
    # # ax.set_xlabel('Time [s]')
    # # ax.set_ylabel('estimated IMU Accelerometer [m/s^2]')
    # # ax.set_title('estimated IMU Accelerometer')
    # # ax.grid(True)
    # # ax.legend()
    # # fig.tight_layout()
    # # fig.savefig("images/ekf/estimated_imu_accelerometer_plot.png")
    # # plt.close(fig)

    # # plot imu angular velocity estimated
    # fig, ax = plt.subplots()
    # ax.plot(t, estimated_imu_angular_velocity[:, 0], label='estimated IMU Angular Velocity X', color='blue')
    # ax.plot(t, estimated_imu_angular_velocity[:, 1], label='estimated IMU Angular Velocity Y', color='orange')
    # ax.plot(t, estimated_imu_angular_velocity[:, 2], label='estimated IMU Angular Velocity Z', color='green')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('estimated IMU Angular Velocity [rad/s]')
    # ax.set_title('estimated IMU Angular Velocity')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig("images/ekf/estimated_imu_angular_velocity_plot.png")
    # plt.close(fig)

    # plot error between real and estimated imu accelerometer
    # fig, ax = plt.subplots()
    # ax.plot(t, measured_imu_accelerometer[:, 0] - estimated_imu_accelerometer[:, 0], label='IMU Accelerometer X Error', color='blue')
    # ax.plot(t, measured_imu_accelerometer[:, 1] - estimated_imu_accelerometer[:, 1], label='IMU Accelerometer Y Error', color='orange')
    # ax.plot(t, measured_imu_accelerometer[:, 2] - estimated_imu_accelerometer[:, 2], label='IMU Accelerometer Z Error', color='green')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('IMU Accelerometer Error [m/s^2]')
    # ax.set_title('IMU Accelerometer Error: Real - estimated')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig("images/ekf/imu_accelerometer_error_plot.png")
    # plt.close(fig)

    # plot error between real and estimated imu angular velocity
    # fig, ax = plt.subplots()
    # ax.plot(t, measured_imu_angular_velocity[:, 0] - estimated_imu_angular_velocity[:, 0], label='IMU Angular Velocity X Error', color='blue')
    # ax.plot(t, measured_imu_angular_velocity[:, 1] - estimated_imu_angular_velocity[:, 1], label='IMU Angular Velocity Y Error', color='orange')
    # ax.plot(t, measured_imu_angular_velocity[:, 2] - estimated_imu_angular_velocity[:, 2], label='IMU Angular Velocity Z Error', color='green')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('IMU Angular Velocity Error [rad/s]')
    # ax.set_title('IMU Angular Velocity Error: Real - estimated')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig("images/ekf/imu_angular_velocity_error_plot.png")
    # plt.close(fig)

    # # plot error between real and estimated imu orientation
    # fig, ax = plt.subplots()
    # ax.plot(t, measured_imu_orientation[:, 0] - estimated_imu_orientation[:, 0], label='IMU Orientation W Error', color='blue')
    # ax.plot(t, measured_imu_orientation[:, 1] - estimated_imu_orientation[:, 1], label='IMU Orientation X Error', color='orange')
    # ax.plot(t, measured_imu_orientation[:, 2] - estimated_imu_orientation[:, 2], label='IMU Orientation Y Error', color='green')
    # ax.plot(t, measured_imu_orientation[:, 3] - estimated_imu_orientation[:, 3], label='IMU Orientation Z Error', color='red')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('IMU Orientation Error [Quaternion]')
    # ax.set_title('IMU Orientation Error: Real - estimated')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig("images/ekf/imu_orientation_error_plot.png")
    # plt.close(fig)

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

    #convert quaternion w x y z to euler angles and plot them not using R
    measured_imu_orientation_euler = R.from_quat(measured_imu_orientation[:, 0:4].copy()).as_euler('xyz', degrees=True)
    fig, ax = plt.subplots()
    ax.plot(t, measured_imu_orientation_euler[:, 0], label='IMU Orientation Roll', color='blue')
    ax.plot(t, measured_imu_orientation_euler[:, 1], label='IMU Orientation Pitch', color='orange')
    ax.plot(t, measured_imu_orientation_euler[:, 2], label='IMU Orientation Yaw', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('IMU Orientation [Degrees]')
    ax.set_title('IMU Orientation in Euler Angles')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/measured_imu_orientation_euler_plot.png")
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
            ax.plot(t, sim_joint_position[:, i], label=f"{joint_names[i].strip()} Simulation", linestyle='--')
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


    # plot velocity error between input command and feedback joint velocity, everything in one single plot
    fig, ax = plt.subplots(figsize=(18, 12))
    for i in range(measured_joint_velocity.shape[1]):
        error = measured_joint_velocity[:, i] - sim_joint_velocity[:, i]
        ax.plot(t, error, label=joint_names[i].strip())
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity Error [rad/s]')
    ax.set_title('Velocity Error between Simulation and Feedback Joint Velocity')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"images/feedback/velocity_error_plot.png")
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

    def plot_force_torque(t, F, Tau, title, save_path):
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

        ax_f = axes[0]
        ax_f.plot(t, F[:, 0], label="Fx")
        ax_f.plot(t, F[:, 1], label="Fy")
        ax_f.plot(t, F[:, 2], label="Fz")
        ax_f.set_ylabel("Force [N]")
        ax_f.legend()
        ax_f.grid(True)
        ax_f.set_title(title)

        ax_tau = axes[1]
        if Tau is not None:
            ax_tau.plot(t, Tau[:, 0], label="Tx")
            ax_tau.plot(t, Tau[:, 1], label="Ty")
            ax_tau.plot(t, Tau[:, 2], label="Tz")
            ax_tau.legend()
        else:
            ax_tau.text(0.5, 0.5, "Torque channels not available in this log", ha="center", va="center", transform=ax_tau.transAxes)
        ax_tau.set_xlabel("Time [s]")
        ax_tau.set_ylabel("Torque [Nm]")
        ax_tau.grid(True)

        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    connect_force_path = folder + "/box_connect_forces.txt"
    if os.path.exists(connect_force_path) and os.path.getsize(connect_force_path) > 0:
        with open(connect_force_path, "r") as f:
            header = f.readline().strip().split()

        data = np.loadtxt(connect_force_path, skiprows=1)
        data = np.atleast_2d(data)

        if data.shape[0] > 0 and data.shape[1] >= 4:
            t = data[:, 0]
            header_index = {name: idx for idx, name in enumerate(header)}

            connect_channels = [
                ("lf", "left_front", "r1 left finger"),
                ("lb", "left_back", "r1 right finger"),
                ("rf", "right_front", "r2 left finger"),
                ("rb", "right_back", "r2 right finger")
            ]

            os.makedirs("images/forces_torques", exist_ok=True)

            for key, file_suffix, pretty_name in connect_channels:
                col_x = f"{key}_x"
                col_y = f"{key}_y"
                col_z = f"{key}_z"
                col_tx = f"{key}_tx"
                col_ty = f"{key}_ty"
                col_tz = f"{key}_tz"

                if col_x in header_index and col_y in header_index and col_z in header_index:
                    F = np.column_stack([
                        data[:, header_index[col_x]],
                        data[:, header_index[col_y]],
                        data[:, header_index[col_z]]
                    ])
                else:
                    continue

                Tau = None
                if col_tx in header_index and col_ty in header_index and col_tz in header_index:
                    Tau = np.column_stack([
                        data[:, header_index[col_tx]],
                        data[:, header_index[col_ty]],
                        data[:, header_index[col_tz]]
                    ])

                valid_rows = np.isfinite(F).all(axis=1)
                if Tau is not None:
                    valid_rows = valid_rows & np.isfinite(Tau).all(axis=1)
                if not np.any(valid_rows):
                    continue

                plot_force_torque(
                    t[valid_rows],
                    F[valid_rows, :],
                    Tau[valid_rows, :] if Tau is not None else None,
                    f"box_connect_wrench_{key} ({pretty_name})",
                    f"images/forces_torques/box_connect_wrench_{key}_{file_suffix}.png"
                )