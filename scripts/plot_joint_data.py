import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor, sqrt

if __name__ == '__main__':
    joint_vel: np.ndarray = np.loadtxt('/tmp/joint_vel.txt')
    joint_vel_des: np.ndarray = np.loadtxt('/tmp/joint_vel_des.txt')
    joint_eff: np.ndarray = np.loadtxt('/tmp/joint_eff.txt')
    joint_names = open('/tmp/joint_names.txt').readlines()

    delta = 1e-3
    num_samples = joint_vel.shape[0]
    t = np.linspace(0.0, delta * num_samples, num_samples)

    num_joints = joint_vel.shape[1]

    plots_per_fig = 4

    num_figs = ceil(num_joints / plots_per_fig)

    n = ceil(sqrt(plots_per_fig))

    figs = []
    for i in range(num_joints):
        if i % plots_per_fig == 0:
            figs.append(plt.figure())
        figs[-1].add_subplot(n, n, i % plots_per_fig + 1)
        plt.xlabel('Time [s]')
        plt.ylabel('Angular Velocity [rad/s]')
        plt.plot(t, joint_vel_des[:, i], label='Desired')
        plt.plot(t, joint_vel[:, i], label='Actual')
        plt.title(joint_names[i])
        plt.legend()
        plt.grid()
        plt.tight_layout()

    # figs = []
    # for i in range(num_joints):
    #     if i % plots_per_fig == 0:
    #         figs.append(plt.figure())
    #     figs[-1].add_subplot(n, n, i % plots_per_fig + 1)
    #     plt.xlabel('Time [s]')
    #     plt.ylabel('Torque [Nm]')
    #     plt.plot(t, joint_eff[:, i])
    #     plt.title(joint_names[i])
    #     plt.grid()
    #     plt.tight_layout()

    plt.show()