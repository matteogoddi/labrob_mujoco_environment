import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_joint_data(path, ylabel, title):
    # Load the data from file
    # Each row = one timestep, each column = one joint
    data = np.loadtxt(path)

    n_steps, n_joints = data.shape
    print(f"Loaded {n_steps} timesteps, {n_joints} joints")

    # Time axis (just indices; scale if you know dt)
    time = np.arange(n_steps)

    # Plot each joint torque
    plt.figure(figsize=(10, 6))
    for j in range(n_joints):
        plt.plot(time, data[:, j], label=f"Joint {j+1}")

    plt.xlabel("Control cycle")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description='plot joint efforts / joint velocities')
    parser.add_argument('-eff', action='store_true', help="Plot efforts from /tmp/joint_eff.txt")
    parser.add_argument('-vel', action='store_true', help="Plot velocities from /tmp/joint_vel.txt")
    args = parser.parse_args()

    if args.eff:
        path = "/tmp/joint_eff.txt"
        ylabel = "Torque [Nm]"
        title = "Joint torques over time"
        plot_joint_data(path, ylabel, title)

    if args.vel:
        path = "/tmp/joint_vel.txt"
        ylabel = "Velocity [rad/s]"
        title = "Joint velocities over time"
        plot_joint_data(path, ylabel, title)

    plt.show()
    
if __name__ == '__main__':
    main()