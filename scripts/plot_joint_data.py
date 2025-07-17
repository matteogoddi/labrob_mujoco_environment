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

    if number != '0':
        fb_joint_pos: np.ndarray = np.loadtxt(folder +'/fb_joint_pos.txt')
        fb_joint_vel: np.ndarray = np.loadtxt(folder +'/fb_joint_vel.txt')
        fb_com: np.ndarray = np.loadtxt(folder + '/real_com.txt')
        input_command: np.ndarray = np.loadtxt(folder + '/input_command.txt')

        # sometimes the number of samples in com_simulation, fb_com and com_desired is different
        # this is a fix to ensure they have the same number of samples
        if com_simulation.shape[0] != fb_com.shape[0] or com_desired.shape[0] != fb_com.shape[0]:
            com_simulation = com_simulation[:min(com_simulation.shape[0], fb_com.shape[0], com_desired.shape[0]), :]
            com_desired = com_desired[:min(com_simulation.shape[0], fb_com.shape[0], com_desired.shape[0]), :]

    if ekf_base_position.shape[0] != joint_pos.shape[0]:
        min_samples = min(ekf_base_position.shape[0], joint_pos.shape[0])
        ekf_base_position = ekf_base_position[:min_samples, :]
        ekf_base_velocity = ekf_base_velocity[:min_samples, :]
        ekf_base_orientation = ekf_base_orientation[:min_samples, :]
        ekf_base_angular_velocity = ekf_base_angular_velocity[:min_samples, :]
        ekf_joint_position = ekf_joint_position[:min_samples, :]
        ekf_joint_velocity = ekf_joint_velocity[:min_samples, :]
        base_position = base_position[:min_samples, :]
        base_velocity = base_velocity[:min_samples, :]
        base_orientation = base_orientation[:min_samples, :]
        base_angular_velocity = base_angular_velocity[:min_samples, :]
    delta = 1e-3
    num_samples = joint_pos.shape[0]
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
    fig.savefig("images/ekf/base_position_plot.png")
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
    fig.savefig("images/ekf/base_velocity_plot.png")
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

    if number != '0':

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

