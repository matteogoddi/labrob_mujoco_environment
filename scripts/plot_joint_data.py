import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor, sqrt
from collections import defaultdict
import matplotlib.cm as cm
import os

if __name__ == '__main__':
    #request input from terminal
    number = input("Enter 0 to plot data from the last simulation or the number of the experiment: ")
    if number == '0':
        folder = '/tmp'
    else:
        folder = 'experiments/experiment_' + number
    joint_pos: np.ndarray = np.loadtxt(folder + '/joint_pos.txt')
    joint_vel: np.ndarray = np.loadtxt(folder + '/joint_vel.txt')
    joint_eff: np.ndarray = np.loadtxt(folder +'/joint_eff.txt')
    com_simulation: np.ndarray = np.loadtxt(folder + '/com.txt')
    com_desired: np.ndarray = np.loadtxt(folder + '/mpc_com.txt')
    joint_names = open(folder + '/joint_names.txt').readlines()

    ekf_base_position = np.loadtxt(folder + '/ekf_base_position.txt')
    ekf_base_velocity = np.loadtxt(folder + '/ekf_base_velocity.txt')
    ekf_base_orientation = np.loadtxt(folder + '/ekf_base_orientation.txt')
    ekf_base_angular_velocity = np.loadtxt(folder + '/ekf_base_angular_velocity.txt')
    ekf_joint_position = np.loadtxt(folder + '/ekf_joint_position.txt')
    ekf_joint_velocity = np.loadtxt(folder + '/ekf_joint_velocity.txt')
    base_position = np.loadtxt(folder + '/base_position.txt')
    base_velocity = np.loadtxt(folder + '/base_velocity.txt')
    base_orientation = np.loadtxt(folder + '/base_orientation.txt')
    base_angular_velocity = np.loadtxt(folder + '/base_angular_velocity.txt')

    num_samples = joint_pos.shape[0] - 2000
    joint_pos = joint_pos[:num_samples, :]
    joint_vel = joint_vel[:num_samples, :]
    joint_eff = joint_eff[:num_samples, :]
    com_simulation = com_simulation[:num_samples, :]
    com_desired = com_desired[:num_samples, :]
    ekf_base_position = ekf_base_position[:num_samples, :]
    ekf_base_velocity = ekf_base_velocity[:num_samples, :]
    ekf_base_orientation = ekf_base_orientation[:num_samples, :]
    ekf_base_angular_velocity = ekf_base_angular_velocity[:num_samples, :]
    ekf_joint_position = ekf_joint_position[:num_samples, :]
    ekf_joint_velocity = ekf_joint_velocity[:num_samples, :]
    base_position = base_position[:num_samples, :]
    base_velocity = base_velocity[:num_samples, :]
    base_orientation = base_orientation[:num_samples, :]
    base_angular_velocity = base_angular_velocity[:num_samples, :]

    if number != '0':
        fb_joint_pos: np.ndarray = np.loadtxt(folder +'/fb_joint_pos.txt')
        fb_joint_vel: np.ndarray = np.loadtxt(folder +'/fb_joint_vel.txt')
        fb_com: np.ndarray = np.loadtxt(folder + '/real_com.txt')
        input_command: np.ndarray = np.loadtxt(folder + '/input_command.txt')
        imu_orientation: np.ndarray = np.loadtxt(folder + '/imu_orientation.txt')
        imu_angular_velocity: np.ndarray = np.loadtxt(folder + '/imu_angular_velocity.txt')
        imu_accelerometer: np.ndarray = np.loadtxt(folder + '/imu_accelerometer.txt')
        predicted_imu_accelerometer: np.ndarray = np.loadtxt(folder + '/predicted_imu_accelerometer.txt')
        predicted_imu_angular_velocity: np.ndarray = np.loadtxt(folder + '/predicted_imu_angular_velocity.txt')
        predicted_imu_orientation: np.ndarray = np.loadtxt(folder + '/predicted_imu_orientation.txt')

        fb_joint_pos = fb_joint_pos[:num_samples, :]
        fb_joint_vel = fb_joint_vel[:num_samples, :]
        fb_com = fb_com[:num_samples, :]
        input_command = input_command[:num_samples, :]
        imu_orientation = imu_orientation[:num_samples, :]
        imu_angular_velocity = imu_angular_velocity[:num_samples, :]
        imu_accelerometer = imu_accelerometer[:num_samples, :]
        predicted_imu_accelerometer = predicted_imu_accelerometer[:num_samples, :]
        predicted_imu_angular_velocity = predicted_imu_angular_velocity[:num_samples, :]
        predicted_imu_orientation = predicted_imu_orientation[:num_samples, :]
        
    delta = 1e-3
    t = np.linspace(0.0, delta * num_samples, num_samples)

    if not os.path.exists('images/joints/positions'):
        os.makedirs('images/joints/positions')
    if not os.path.exists('images/joints/velocities'):
        os.makedirs('images/joints/velocities')
    if not os.path.exists('images/joints/torques'):
        os.makedirs('images/joints/torques')
    if not os.path.exists('images/feedback/positions'):
        os.makedirs('images/feedback/positions')
    if not os.path.exists('images/ekf'):
        os.makedirs('images/ekf')

    grouped_indices = defaultdict(list)

    for idx, name in enumerate(joint_names):
        base_name = '_'.join(name.split('_')[:2])  # E.g., "left_ankle" da "left_ankle_roll_joint"
        grouped_indices[base_name].append(idx)
    
    # Crea un plot per ogni gruppo
    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, joint_eff[:, i], label=joint_names[i])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Torque [Nm]')
        ax.set_title(group_name.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        fig.savefig(f"images/joints/torques/{group_name}_torque_plot.png")
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, joint_vel[:, i], label=joint_names[i])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angular velocities [rad/s]')
        ax.set_title(group_name.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        fig.savefig(f"images/joints/velocities/{group_name}_velocities_plot.png")
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, joint_pos[:, i], label=joint_names[i])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [rad]')
        ax.set_title(group_name.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        fig.savefig(f"images/joints/positions/{group_name}_position_plot.png")
        plt.close(fig)


    # Plot EKF base position
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_position[:, 0] - base_position[:, 0], label='EKF Base Position X', color='blue')
    ax.plot(t, ekf_base_position[:, 1] - base_position[:, 1], label='EKF Base Position Y', color='orange')
    ax.plot(t, ekf_base_position[:, 2] - base_position[:, 2], label='EKF Base Position Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Position [m]')
    ax.set_title('Base Position Error between EKF estimation and Simulation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_position_error_plot.png")
    plt.close(fig)

    # Plot EKF base velocity
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_velocity[:, 0] - base_velocity[:, 0], label='EKF Base Velocity X', color='blue')
    ax.plot(t, ekf_base_velocity[:, 1] - base_velocity[:, 1], label='EKF Base Velocity Y', color='orange')
    ax.plot(t, ekf_base_velocity[:, 2] - base_velocity[:, 2], label='EKF Base Velocity Z', color='green')
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
    ax.plot(t, base_position[:, 0], label='Base position X', color='blue', linestyle = "--")
    ax.plot(t, ekf_base_position[:, 1], label='EKF Base position Y', color='orange')
    ax.plot(t, base_position[:, 1], label='Base position Y', color='orange', linestyle = "--")
    ax.plot(t, ekf_base_position[:, 2] - base_position[:, 2], label='EKF Base position Z', color='green')
    ax.plot(t, base_position[:, 2] - base_position[:, 2], label='Base position Z', color='green', linestyle = "--")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Position [m/s]')
    ax.set_title('Base Position of EKF estimation and Simulation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_position_plot.png")
    plt.close(fig)

    # Plot EKF base orientation in quaternion format
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_orientation[:, 0] - base_orientation[:, 0], label='EKF Base Orientation W', color='blue')
    ax.plot(t, ekf_base_orientation[:, 1] - base_orientation[:, 1], label='EKF Base Orientation X', color='orange')
    ax.plot(t, ekf_base_orientation[:, 2] - base_orientation[:, 2], label='EKF Base Orientation Y', color='green')
    ax.plot(t, ekf_base_orientation[:, 3] - base_orientation[:, 3], label='EKF Base Orientation Z', color='red')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Orientation [Quaternion]')
    ax.set_title('Base Orientation Error between EKF estimation and Simulation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_orientation_error_plot.png")
    plt.close(fig)

    # Plot EKF base angular velocity
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_angular_velocity[:, 0] - base_angular_velocity[:, 0], label='EKF Base Angular Velocity X', color='blue')
    ax.plot(t, ekf_base_angular_velocity[:, 1] - base_angular_velocity[:, 1], label='EKF Base Angular Velocity Y', color='orange')
    ax.plot(t, ekf_base_angular_velocity[:, 2] - base_angular_velocity[:, 2], label='EKF Base Angular Velocity Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Angular Velocity [rad/s]')
    ax.set_title('Angular Velocity Error between EKF estimation and Simulation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_angular_velocity_error_plot.png")
    plt.close(fig)

    # Plot position error between input command and feedback joint position
    fig, ax = plt.subplots(figsize=(18, 12))
    num_joints = joint_pos.shape[1]
    colormap = plt.colormaps['tab10'] 
    line_styles = ['-', '--', '-.', ':']
    for i in range(num_joints):
        color = colormap(i % 10)
        linestyle = line_styles[(i // 10) % len(line_styles)]  # cambia stile ogni 10 joint
        error = ekf_joint_position[:, i] - joint_pos[:, i]
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


    # Plot EKF joint velocity
    fig, ax = plt.subplots(figsize=(18, 12))
    colormap = plt.colormaps['tab10'] 
    line_styles = ['-', '--', '-.', ':']
    for i in range(num_joints):
        color = colormap(i % 10)
        linestyle = line_styles[(i // 10) % len(line_styles)]  # cambia stile ogni 10 joint
        error = ekf_joint_velocity[:, i] - joint_vel[:, i]
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

    # plot ekf joint velocity
    fig, ax = plt.subplots(figsize=(18, 12))
    for i in range(ekf_joint_velocity.shape[1]):
        ax.plot(t, ekf_joint_velocity[:, i], label=joint_names[i].strip())
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [rad/s]')
    ax.set_title('EKF Joint Velocities')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/ekf_joint_velocities_plot.png")
    plt.close(fig)

    #plot simulation joint velocities
    fig, ax = plt.subplots(figsize=(18, 12))
    for i in range(joint_vel.shape[1]):
        ax.plot(t, joint_vel[:, i], label=joint_names[i].strip())
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [rad/s]')
    ax.set_title('Simulation Joint Velocities')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/joints/velocities/simulation_joint_velocities_plot.png")
    plt.close(fig)


    if number != '0':

        # plot feedback joint velocity
        fig, ax = plt.subplots(figsize=(18, 12))
        for i in range(fb_joint_vel.shape[1]):
            ax.plot(t, fb_joint_vel[:, i], label=joint_names[i].strip())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [rad/s]')
        ax.set_title('Feedback Joint Velocities')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/feedback/fb_joint_velocities_plot.png")
        plt.close(fig)

        #plot error between ekf joint pos and fb joint pos
        fig, ax = plt.subplots(figsize=(18, 12))
        for i in range(ekf_joint_position.shape[1]):
            error = ekf_joint_position[:, i] - fb_joint_pos[:, i]
            ax.plot(t, error, label=joint_names[i].strip())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position Error [rad]')
        ax.set_title('Position Error between EKF Joint Position and Feedback Joint Position')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"images/ekf/ekf_fb_joint_position_error_plot.png")
        plt.close(fig)

        #plot error between ekf joint vel and fb joint vel
        fig, ax = plt.subplots(figsize=(18, 12))
        for i in range(ekf_joint_velocity.shape[1]):
            error = ekf_joint_velocity[:, i] - fb_joint_vel[:, i]
            ax.plot(t, error, label=joint_names[i].strip())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity Error [rad/s]')
        ax.set_title('Velocity Error between EKF Joint Velocity and Feedback Joint Velocity')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"images/ekf/ekf_fb_joint_velocity_error_plot.png")
        plt.close(fig)
        
        # plot imu orientation predicted
        fig, ax = plt.subplots()
        ax.plot(t, predicted_imu_orientation[:, 0], label='Predicted IMU Orientation W', color='blue')
        ax.plot(t, predicted_imu_orientation[:, 1], label='Predicted IMU Orientation X', color='orange')
        ax.plot(t, predicted_imu_orientation[:, 2], label='Predicted IMU Orientation Y', color='green')
        ax.plot(t, predicted_imu_orientation[:, 3], label='Predicted IMU Orientation Z', color='red')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Predicted IMU Orientation [Quaternion]')
        ax.set_title('Predicted IMU Orientation')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/feedback/predicted_imu_orientation_plot.png")
        plt.close(fig)

        # plot imu accelerometer predicted
        fig, ax = plt.subplots()
        ax.plot(t, predicted_imu_accelerometer[:, 0], label='Predicted IMU Accelerometer X', color='blue')
        ax.plot(t, predicted_imu_accelerometer[:, 1], label='Predicted IMU Accelerometer Y', color='orange')
        ax.plot(t, predicted_imu_accelerometer[:, 2], label='Predicted IMU Accelerometer Z', color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Predicted IMU Accelerometer [m/s^2]')
        ax.set_title('Predicted IMU Accelerometer')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/feedback/predicted_imu_accelerometer_plot.png")
        plt.close(fig)

        # plot imu angular velocity predicted
        fig, ax = plt.subplots()
        ax.plot(t, predicted_imu_angular_velocity[:, 0], label='Predicted IMU Angular Velocity X', color='blue')
        ax.plot(t, predicted_imu_angular_velocity[:, 1], label='Predicted IMU Angular Velocity Y', color='orange')
        ax.plot(t, predicted_imu_angular_velocity[:, 2], label='Predicted IMU Angular Velocity Z', color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Predicted IMU Angular Velocity [rad/s]')
        ax.set_title('Predicted IMU Angular Velocity')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/feedback/predicted_imu_angular_velocity_plot.png")
        plt.close(fig)

        # plot imu orientation
        fig, ax = plt.subplots()
        ax.plot(t, imu_orientation[:, 0], label='IMU Orientation W', color='blue')
        ax.plot(t, imu_orientation[:, 1], label='IMU Orientation X', color='orange')
        ax.plot(t, imu_orientation[:, 2], label='IMU Orientation Y', color='green')
        ax.plot(t, imu_orientation[:, 3], label='IMU Orientation Z', color='red')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('IMU Orientation [Quaternion]')
        ax.set_title('IMU Orientation')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/feedback/imu_orientation_plot.png")
        plt.close(fig)

        # plot imu angular velocity
        fig, ax = plt.subplots()
        ax.plot(t, imu_angular_velocity[:, 0], label='IMU Angular Velocity X', color='blue')
        ax.plot(t, imu_angular_velocity[:, 1], label='IMU Angular Velocity Y', color='orange')
        ax.plot(t, imu_angular_velocity[:, 2], label='IMU Angular Velocity Z', color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('IMU Angular Velocity [rad/s]')
        ax.set_title('IMU Angular Velocity')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/feedback/imu_angular_velocity_plot.png")
        plt.close(fig)
        
        # plot imu accelerometer
        fig, ax = plt.subplots()
        ax.plot(t, imu_accelerometer[:, 0], label='IMU Accelerometer X', color='blue')
        ax.plot(t, imu_accelerometer[:, 1], label='IMU Accelerometer Y', color='orange')
        ax.plot(t, imu_accelerometer[:, 2], label='IMU Accelerometer Z', color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('IMU Accelerometer [m/s^2]')
        ax.set_title('IMU Accelerometer')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/feedback/imu_accelerometer_plot.png")
        plt.close(fig)

        # plot error between real and predicted imu accelerometer
        fig, ax = plt.subplots()
        ax.plot(t, imu_accelerometer[:, 0] - predicted_imu_accelerometer[:, 0], label='IMU Accelerometer X Error', color='blue')
        ax.plot(t, imu_accelerometer[:, 1] - predicted_imu_accelerometer[:, 1], label='IMU Accelerometer Y Error', color='orange')
        ax.plot(t, imu_accelerometer[:, 2] - predicted_imu_accelerometer[:, 2], label='IMU Accelerometer Z Error', color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('IMU Accelerometer Error [m/s^2]')
        ax.set_title('IMU Accelerometer Error: Real - Predicted')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/feedback/imu_accelerometer_error_plot.png")
        plt.close(fig)

        # plot error between real and predicted imu angular velocity
        fig, ax = plt.subplots()
        ax.plot(t, imu_angular_velocity[:, 0] - predicted_imu_angular_velocity[:, 0], label='IMU Angular Velocity X Error', color='blue')
        ax.plot(t, imu_angular_velocity[:, 1] - predicted_imu_angular_velocity[:, 1], label='IMU Angular Velocity Y Error', color='orange')
        ax.plot(t, imu_angular_velocity[:, 2] - predicted_imu_angular_velocity[:, 2], label='IMU Angular Velocity Z Error', color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('IMU Angular Velocity Error [rad/s]')
        ax.set_title('IMU Angular Velocity Error: Real - Predicted')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/feedback/imu_angular_velocity_error_plot.png")
        plt.close(fig)

        # plot error between real and predicted imu orientation
        fig, ax = plt.subplots()
        ax.plot(t, imu_orientation[:, 0] - predicted_imu_orientation[:, 0], label='IMU Orientation W Error', color='blue')
        ax.plot(t, imu_orientation[:, 1] - predicted_imu_orientation[:, 1], label='IMU Orientation X Error', color='orange')
        ax.plot(t, imu_orientation[:, 2] - predicted_imu_orientation[:, 2], label='IMU Orientation Y Error', color='green')
        ax.plot(t, imu_orientation[:, 3] - predicted_imu_orientation[:, 3], label='IMU Orientation Z Error', color='red')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('IMU Orientation Error [Quaternion]')
        ax.set_title('IMU Orientation Error: Real - Predicted')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/feedback/imu_orientation_error_plot.png")
        plt.close(fig)

        figs = []
        for group_name, indices in grouped_indices.items():
            fig, ax = plt.subplots()
            for i in indices:
                ax.plot(t, fb_joint_pos[:, i], label=joint_names[i])
                # plot input command as a dotted line 
                ax.plot(t, input_command[:, i], label=f"{joint_names[i].strip()} Input Command", linestyle='--')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Position [rad]')
            ax.set_title(group_name.replace('_', ' ').title())
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            figs.append(fig)

            fig.savefig(f"images/feedback/positions/{group_name}_fb_position_plot.png")
            plt.close(fig)

        # Plot position error between input command and feedback joint position
        fig, ax = plt.subplots(figsize=(18, 12))
        num_joints = fb_joint_pos.shape[1]
        colormap = plt.colormaps['tab10'] 
        line_styles = ['-', '--', '-.', ':']
        for i in range(num_joints):
            error = fb_joint_pos[:, i] - input_command[:, i]
            color = colormap(i % 10)
            linestyle = line_styles[(i // 10) % len(line_styles)]  # cambia stile ogni 10 joint
            ax.plot(t, error,
                    label=joint_names[i].strip(),
                    color=color,
                    linestyle=linestyle,
                    linewidth=2)
        ax.set_xlabel('Time [s]', fontsize=14)
        ax.set_ylabel('Position Error [rad]', fontsize=14)
        ax.set_title('Position Error between Input Command and Feedback Joint Position', fontsize=16)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize=12, loc='upper right', ncol=2)
        fig.tight_layout()
        fig.savefig("images/feedback/position_error_plot.png")
        plt.close(fig)


        #plot velocity error between input command and feedback joint velocity, everything in one single plot
        fig, ax = plt.subplots(figsize=(18, 12))
        for i in range(fb_joint_vel.shape[1]):
            error = fb_joint_vel[:, i] - joint_vel[:, i]
            ax.plot(t, error, label=joint_names[i].strip())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity Error [rad/s]')
        ax.set_title('Velocity Error between Input Command and Feedback Joint Velocity')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"images/feedback/velocity_error_plot.png")
        plt.close(fig)

        #plot error between ekf joint pos and fb joint pos
        fig, ax = plt.subplots(figsize=(18, 12))
        for i in range(ekf_joint_position.shape[1]):
            error = ekf_joint_position[:, i] - fb_joint_pos[:, i]
            ax.plot(t, error, label=joint_names[i].strip())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position Error [rad]')
        ax.set_title('Position Error between EKF Joint Position and Feedback Joint Position')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"images/feedback/ekf_joint_position_error_plot.png")
        plt.close(fig)

        #plot error between ekf joint vel and fb joint vel
        fig, ax = plt.subplots(figsize=(18, 12))
        for i in range(ekf_joint_velocity.shape[1]):
            error = ekf_joint_velocity[:, i] - fb_joint_vel[:, i]
            ax.plot(t, error, label=joint_names[i].strip())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity Error [rad/s]')
        ax.set_title('Velocity Error between EKF Joint Velocity and Feedback Joint Velocity')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"images/feedback/ekf_joint_velocity_error_plot.png")
        plt.close(fig)


        # Plot CoM simulation and real data
        fig, ax = plt.subplots()
        ax.plot(t, com_simulation[:, 0], label='CoM Simulation X', color='blue')
        ax.plot(t, com_simulation[:, 1], label='CoM Simulation Y', color='orange')
        ax.plot(t, com_simulation[:, 2], label='CoM Simulation Z', color='green')
        ax.plot(t, fb_com[:, 0], label='CoM Real X', color='red', linestyle='--')
        ax.plot(t, fb_com[:, 1], label='CoM Real Y', color='purple', linestyle='--')
        ax.plot(t, fb_com[:, 2], label='CoM Real Z', color='brown', linestyle='--')
        ax.plot(t, com_desired[:, 0], label='CoM Desired X', color='cyan', linestyle=':')
        ax.plot(t, com_desired[:, 1], label='CoM Desired Y', color='magenta', linestyle=':')
        ax.plot(t, com_desired[:, 2], label='CoM Desired Z', color='yellow', linestyle=':')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('CoM Position [m]')
        ax.set_title('CoM Position: Simulation vs Real')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/feedback/com_plot.png")
        plt.close(fig)  

        # Plot CoM error
        com_error = com_simulation - fb_com
        fig, ax = plt.subplots()
        ax.plot(t, com_error[:, 0], label='CoM Error X', color='blue')
        ax.plot(t, com_error[:, 1], label='CoM Error Y', color='orange')
        ax.plot(t, com_error[:, 2], label='CoM Error Z', color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('CoM Error [m]')
        ax.set_title('CoM Error: Simulation - Real')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/feedback/com_error_plot.png")
        plt.close(fig)

        #plot error between feedback orientation and ekf orientation
        fig, ax = plt.subplots()
        ax.plot(t, ekf_base_orientation[:, 0] - base_orientation[:, 0], label='EKF Base Orientation W Error', color='blue')
        ax.plot(t, ekf_base_orientation[:, 1] - base_orientation[:, 1], label='EKF Base Orientation X Error', color='orange')
        ax.plot(t, ekf_base_orientation[:, 2] - base_orientation[:, 2], label='EKF Base Orientation Y Error', color='green')
        ax.plot(t, ekf_base_orientation[:, 3] - base_orientation[:, 3], label='EKF Base Orientation Z Error', color='red')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Orientation Error [Quaternion]')
        ax.set_title('Orientation Error between EKF estimation and Simulation')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

