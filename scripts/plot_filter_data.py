import numpy as np
import matplotlib.pyplot as plt
import os

CTRL_HZ = 500  # g1_controller.cpp Control() tick rate (control_dt_ = 0.002s)
DT = 1.0 / CTRL_HZ


def try_load(folder, name):
    path = f"{folder}/{name}.txt"
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def plot_compare(t, filtered, odom, labels, title, ylabel, path):
    if filtered is None:
        print(f"skip: {title} (no filtered_* data)")
        return
    n = min(len(t), len(filtered))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, lbl in enumerate(labels):
        ax.plot(t[:n], filtered[:n, i], label=f'filtered {lbl}', linewidth=1.8)
    if odom is not None:
        n = min(n, len(odom))
        for i, lbl in enumerate(labels):
            ax.plot(t[:n], odom[:n, i], label=f'odom {lbl}', linestyle='--', linewidth=1.2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    expNumber = input("Enter 0 for the last run (/tmp/robot_logs) or the experiment number: ").strip()
    if expNumber == '0':
        folder = '/tmp/robot_logs'
    else:
        folder = f'experiments/experiment_{expNumber}/robot_logs'

    filtered_base_position = try_load(folder, 'filtered_base_position')
    filtered_base_velocity = try_load(folder, 'filtered_base_velocity')
    filtered_base_quat     = try_load(folder, 'filtered_base_quat')
    filtered_base_rpy      = try_load(folder, 'filtered_base_rpy')
    filtered_base_ang_vel  = try_load(folder, 'filtered_base_ang_vel')

    odom_pos  = try_load(folder, 'odom_pos')
    odom_vel  = try_load(folder, 'odom_vel')
    odom_quat = try_load(folder, 'odom_quat')

    if filtered_base_position is None:
        print(f"No filtered_base_*.txt found in {folder}")
        exit(1)

    n = len(filtered_base_position)
    t = np.linspace(0.0, DT * n, n)

    plot_compare(t, filtered_base_position, odom_pos, ['x', 'y', 'z'],
        'Base Position — filtered vs odometry', 'Position [m]',
        'images/filter/base_position_plot.png')
    plot_compare(t, filtered_base_velocity, odom_vel, ['x', 'y', 'z'],
        'Base Velocity — filtered vs odometry', 'Velocity [m/s]',
        'images/filter/base_velocity_plot.png')
    plot_compare(t, filtered_base_quat, odom_quat, ['w', 'x', 'y', 'z'],
        'Base Orientation (quat) — filtered vs odometry', 'Quaternion',
        'images/filter/base_orientation_quat_plot.png')
    plot_compare(t, filtered_base_rpy, None, ['roll', 'pitch', 'yaw'],
        'Base Orientation (RPY) — filtered', 'Angle [rad]',
        'images/filter/base_orientation_rpy_plot.png')
    plot_compare(t, filtered_base_ang_vel, None, ['x', 'y', 'z'],
        'Base Angular Velocity — filtered', 'Angular velocity [rad/s]',
        'images/filter/base_angular_velocity_plot.png')

    print(f"Plots saved in images/filter/ (source: {folder})")