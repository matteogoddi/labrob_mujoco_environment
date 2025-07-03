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
        folder = 'tmp'
    else:
        folder = 'experiments/experiment_' + number
    joint_pos: np.ndarray = np.loadtxt(folder + '/joint_pos.txt')
    joint_vel: np.ndarray = np.loadtxt(folder + '/joint_vel.txt')
    joint_eff: np.ndarray = np.loadtxt(folder +'/joint_eff.txt')
    fb_joint_pos: np.ndarray = np.loadtxt(folder +'/fb_joint_pos.txt')
    fb_joint_vel: np.ndarray = np.loadtxt(folder +'/fb_joint_vel.txt')
    input_command: np.ndarray = np.loadtxt(folder + '/input_command.txt')
    com_simulation: np.ndarray = np.loadtxt(folder + '/com.txt')
    com_real: np.ndarray = np.loadtxt(folder + '/real_com.txt')
    com_desired: np.ndarray = np.loadtxt(folder + '/mpc_com.txt')
    joint_names = open(folder + '/joint_names.txt').readlines()

    delta = 1e-3
    num_samples = fb_joint_pos.shape[0]
    t = np.linspace(0.0, delta * num_samples, num_samples)

    if not os.path.exists('images/joints/positions'):
        os.makedirs('images/joints/positions')
    if not os.path.exists('images/joints/velocities'):
        os.makedirs('images/joints/velocities')
    if not os.path.exists('images/joints/torques'):
        os.makedirs('images/joints/torques')
    if not os.path.exists('images/feedback/positions'):
        os.makedirs('images/feedback/positions')

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
            ax.plot(t, joint_eff[:, i], label=joint_names[i])
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

    # #plot position error between input command and feedback joint position, everything in one single plot
    # fig, ax = plt.subplots(figsize=(18, 12))
    # for i in range(fb_joint_pos.shape[1]):
    #     error = fb_joint_pos[:, i] - input_command[:, i]
    #     ax.plot(t, error, label=joint_names[i].strip())
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Position Error [rad]')
    # ax.set_title('Position Error between Input Command and Feedback Joint Position')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig(f"images/feedback/position_error_plot.png")
    # plt.close(fig)

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

    # Create a histogram with the last error value of each joint
    last_errors = [fb_joint_pos[-1, i] - input_command[-1, i] for i in range(fb_joint_pos.shape[1])]
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.bar([joint_names[i].strip() for i in range(fb_joint_pos.shape[1])], last_errors)
    ax.set_xlabel('Joint')
    ax.set_ylabel('Last Position Error [rad]')
    ax.set_title('Last Position Error for Each Joint')
    ax.grid(True)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    fig.tight_layout()
    fig.savefig("images/feedback/last_position_error_plot.png")
    plt.close(fig)

    # Plot CoM simulation and real data
    fig, ax = plt.subplots()
    ax.plot(t, com_simulation[:, 0], label='CoM Simulation X', color='blue')
    ax.plot(t, com_simulation[:, 1], label='CoM Simulation Y', color='orange')
    ax.plot(t, com_simulation[:, 2], label='CoM Simulation Z', color='green')
    ax.plot(t, com_real[:, 0], label='CoM Real X', color='red', linestyle='--')
    ax.plot(t, com_real[:, 1], label='CoM Real Y', color='purple', linestyle='--')
    ax.plot(t, com_real[:, 2], label='CoM Real Z', color='brown', linestyle='--')
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
    com_error = com_simulation - com_real
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

