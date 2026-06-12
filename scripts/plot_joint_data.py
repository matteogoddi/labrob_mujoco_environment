import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

CTRL_HZ = 500
DT = 1.0 / CTRL_HZ


# ── helpers ───────────────────────────────────────────────────────────────────

def load(folder, name, start, end):
    data = np.loadtxt(f"{folder}/{name}.txt")
    return data[start:end] if data.ndim == 1 else data[start:end, :]


def plot_components(t, data, labels, title, ylabel, path,
                    figsize=(7, 4), loc='best'):
    fig, ax = plt.subplots(figsize=figsize)
    for col, lbl in zip(data.T, labels):
        ax.plot(t, col, label=lbl, linewidth=2.0)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc=loc, frameon=True, fontsize=9)
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(t, actual, desired, coord_labels, title_prefix, ylabel, path):
    n = len(coord_labels)
    fig, axs = plt.subplots(n, 1, figsize=(7, 3 * n), sharex=True)
    if n == 1:
        axs = [axs]
    for i, lbl in enumerate(coord_labels):
        axs[i].plot(t, actual[:, i],  label=f'Actual {lbl}',  linewidth=2.0)
        axs[i].plot(t, desired[:, i], label=f'Desired {lbl}', linewidth=2.0, linestyle='--')
        axs[i].set_ylabel(ylabel, fontsize=10)
        axs[i].set_title(f'{title_prefix} {lbl}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)
    axs[-1].set_xlabel('Time [s]', fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_bar_joint(values, joint_names, title, ylabel, path, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(values)), values, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(
        [n.strip().replace('_', ' ').replace('joint', '') for n in joint_names],
        rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    expNumber = input("Enter 0 to plot data from the last simulation or the number of the experiment: ")
    if expNumber == '0':
        folder = '/tmp/robot_logs'
        expType = "Simulation"
    else:
        folder = 'experiments/experiment_' + expNumber + '/robot_logs'
        expType = "Experiment"

    end_input = input("Enter the time (in seconds) at which you want to end the plots "
                      "(or press Enter to plot all data): ")
    end_trim = int(float(end_input) * CTRL_HZ) if end_input.strip() else 0

    joint_names = open(folder + '/joint_names.txt').readlines()
    parameters_log  = np.loadtxt(folder + '/parameters_log.txt')
    startTimeWBCCL  = parameters_log

    _ref   = np.loadtxt(f"{folder}/fb_com_position.txt")
    total  = _ref.shape[0]
    start  = 0
    end    = total - end_trim if end_trim > 0 else total

    def L(name):
        return load(folder, name, start, end)

    # ── load all signals ──────────────────────────────────────────────────────

    fb_com_position          = L('fb_com_position')
    fb_com_velocity          = L('fb_com_velocity')
    fb_zmp_position          = L('fb_zmp_position')
    kf_com_position          = L('kf_com_position')
    kf_com_velocity          = L('kf_com_velocity')
    kf_zmp_position          = L('kf_zmp_position')
    des_com_position         = L('des_com_position')
    des_com_velocity         = L('des_com_velocity')
    des_zmp_position         = L('des_zmp_position')
    des_com_acceleration     = L('des_com_acceleration')
    ef_zmp_position          = L('ef_zmp_position')
    input_torque             = L('input_torque')
    wbc_accelerations        = L('wbc_accelerations')
    estimated_force_lsole    = L('estimated_force_lsole')
    estimated_force_rsole    = L('estimated_force_rsole')
    p_lsole_fb               = L('p_lsole_fb')
    p_rsole_fb               = L('p_rsole_fb')
    v_lsole_fb               = L('v_lsole_fb')
    v_rsole_fb               = L('v_rsole_fb')
    p_lsole_des              = L('p_lsole_des')
    p_rsole_des              = L('p_rsole_des')
    v_lsole_des              = L('v_lsole_des')
    v_rsole_des              = L('v_rsole_des')
    fb_lsole_orientation     = L('fb_lsole_orientation')
    fb_rsole_orientation     = L('fb_rsole_orientation')
    des_lsole_orientation    = L('des_lsole_orientation')
    des_rsole_orientation    = L('des_rsole_orientation')
    ekf_base_position        = L('ekf_base_position')
    ekf_base_velocity        = L('ekf_base_velocity')
    ekf_base_orientation     = L('ekf_base_orientation')
    ekf_base_orientation_rpy = L('ekf_base_orientation_rpy')
    ekf_base_angular_velocity= L('ekf_base_angular_velocity')
    ekf_imu_orientation      = L('ekf_imu_orientation')
    ekf_imu_orientation_rpy  = L('ekf_imu_orientation_rpy')
    ekf_imu_angular_velocity = L('ekf_imu_angular_velocity')
    ekf_joint_position       = L('ekf_joint_position')
    ekf_joint_velocity       = L('ekf_joint_velocity')
    torso_orientation        = L('torso_orientation')
    torso_angular_velocity   = L('torso_angular_velocity')
    des_torso_orientation    = L('des_torso_orientation')
    des_torso_angular_velocity = L('des_torso_angular_velocity')
    odometry_base_position       = L('odometry_base_position')
    odometry_base_velocity       = L('odometry_base_velocity')
    odometry_imu_orientation     = L('odometry_imu_orientation')
    odometry_imu_orientation_rpy = L('odometry_imu_orientation_rpy')
    measured_joint_position      = L('measured_joint_position')
    measured_joint_velocity      = L('measured_joint_velocity')
    measured_imu_orientation     = L('measured_imu_orientation')
    measured_imu_orientation_rpy = L('measured_imu_orientation_rpy')
    measured_imu_angular_velocity= L('measured_imu_angular_velocity')
    measured_imu_accelerometer   = L('measured_imu_accelerometer')
    execution_time_ekf    = L('execution_time_ekf')
    execution_time_kf     = L('execution_time_kf')
    execution_time_mpc    = L('execution_time_mpc')
    execution_time_wbc    = L('execution_time_wbc')
    execution_time_update = L('execution_time_update')

    # ── rotate foot/CoM positions into body frame ─────────────────────────────
    n = end - start
    for i in range(n):
        yaw = odometry_imu_orientation_rpy[0, 2]
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        for arr in (p_lsole_fb, p_rsole_fb, p_lsole_des, p_rsole_des,
                    kf_com_position, kf_zmp_position,
                    des_com_position, des_zmp_position):
            arr[i] = Rz.T @ arr[i]

    t          = np.linspace(0.0, DT * n, n)
    iter_t     = np.arange(n)
    num_joints = 29
    labels_xyz  = ['x', 'y', 'z']
    labels_quat = ['w', 'x', 'y', 'z']
    labels_rpy  = ['r', 'p', 'y']
    grouped_indices = defaultdict(list)   # populate to enable per-group joint plots

    # ══════════════════════════════════════════════════════════════════════════
    #  JOINT TORQUES & ESTIMATED FORCES
    # ══════════════════════════════════════════════════════════════════════════

    for gname, idx in grouped_indices.items():
        plot_components(t, input_torque[:, idx],
            [joint_names[i].strip() for i in idx],
            f'Input Joint Torques – {gname}', 'Torque [Nm]',
            f'images/forces_torques/joints/{gname}_input_joint_torques.png')

        plot_components(t, wbc_accelerations[:, [i + 6 for i in idx]],
            [joint_names[i].strip() for i in idx],
            f'WBC Acceleration – {gname}', r'Acceleration [rad/s²]',
            f'images/forces_torques/joints/{gname}_acceleration.png')

    plot_components(t, wbc_accelerations[:, :3],
        [f'Linear acc {l}' for l in labels_xyz],
        'WBC Base Linear Acceleration', r'Acceleration [m/s²]',
        'images/forces_torques/wbc_base_linear_acceleration.png')

    plot_components(t, wbc_accelerations[:, 3:6],
        [f'Angular acc {l}' for l in labels_xyz],
        'WBC Base Angular Acceleration', r'Acceleration [rad/s²]',
        'images/forces_torques/wbc_base_angular_acceleration.png')

    plot_components(t, estimated_force_lsole,
        [f'Left sole force {l}' for l in labels_xyz],
        'Estimated Forces on Left Sole', 'Force [N]',
        'images/forces_torques/estimated_force_left_sole.png')

    plot_components(t, estimated_force_rsole,
        [f'Right sole force {l}' for l in labels_xyz],
        'Estimated Forces on Right Sole', 'Force [N]',
        'images/forces_torques/estimated_force_right_sole.png')

    # ══════════════════════════════════════════════════════════════════════════
    #  CoM AND ZMP
    # ══════════════════════════════════════════════════════════════════════════

    plot_components(t, des_com_acceleration,
        [fr'Des CoM Acc ${l}$' for l in labels_xyz],
        'Desired Center of Mass Acceleration', r'Acceleration [$\mathrm{m/s^2}$]',
        'images/com/references/des_com_acceleration_plot.png')

    plot_components(t, des_com_position - des_com_position[0],
        [fr'Des CoM Pos ${l}$' for l in labels_xyz],
        'Desired Center of Mass Position', r'Position [$\mathrm{m}$]',
        'images/com/references/des_com_position_plot.png')

    plot_components(t, des_com_velocity,
        [fr'Des CoM Vel ${l}$' for l in labels_xyz],
        'Desired Center of Mass Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/com/references/des_com_velocity_plot.png')

    plot_components(t, des_zmp_position - des_zmp_position[0],
        [fr'Des ZMP Pos ${l}$' for l in labels_xyz],
        'Desired Zero Moment Point Position', r'Position [$\mathrm{m}$]',
        'images/com/references/des_zmp_position_plot.png')

    plot_components(t, des_zmp_position - kf_zmp_position,
        [fr'ZMP Error ${l}$' for l in labels_xyz],
        'Zero Moment Point Position Error', r'Position [$\mathrm{m}$]',
        'images/com/errors/error_zmp_position_plot.png')

    plot_components(t, des_com_position - kf_com_position,
        [fr'CoM Pos Error ${l}$' for l in labels_xyz],
        'Center of Mass Position Error', r'Position [$\mathrm{m}$]',
        'images/com/errors/error_com_position_plot.png')

    plot_components(t, des_com_velocity - kf_com_velocity,
        [fr'CoM Vel Error ${l}$' for l in labels_xyz],
        'Center of Mass Velocity Error', r'Velocity [$\mathrm{m/s}$]',
        'images/com/errors/error_com_velocity_plot.png')

    plot_comparison(t,
        kf_zmp_position  - kf_zmp_position[0],
        des_zmp_position - des_zmp_position[0],
        [fr'${l}$' for l in labels_xyz],
        'ZMP Position', r'[$\mathrm{m}$]',
        'images/com/errors/comparison_zmp_position_plot.png')

    plot_comparison(t, kf_com_position, des_com_position,
        [fr'${l}$' for l in labels_xyz],
        'CoM Position', r'[$\mathrm{m}$]',
        'images/com/errors/comparison_com_position_plot.png')

    plot_comparison(t, kf_com_velocity, des_com_velocity,
        [fr'${l}$' for l in labels_xyz],
        'CoM Velocity', r'[$\mathrm{m/s}$]',
        'images/com/errors/comparison_com_velocity_plot.png')

    for axis, direction, fname in [(0, 'forward', 'motion_x'), (1, 'lateral', 'motion_y')]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, kf_com_position[:, axis], label=fr'CoM ${labels_xyz[axis]}$',        linewidth=2.0)
        ax.plot(t, kf_zmp_position[:, axis], label=fr'ZMP ${labels_xyz[axis]}$',        linewidth=2.0)
        ax.plot(t, p_lsole_fb[:, axis],      label=fr'Left foot ${labels_xyz[axis]}$',  linewidth=2.0)
        ax.plot(t, p_rsole_fb[:, axis],      label=fr'Right foot ${labels_xyz[axis]}$', linewidth=2.0)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(fr'Position ${labels_xyz[axis]}$ [$\mathrm{{m}}$]', fontsize=11)
        ax.set_title(f'Motion in the {direction} direction', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best' if axis == 0 else 'upper left', frameon=True, fontsize=9)
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        os.makedirs('images/com', exist_ok=True)
        fig.savefig(f'images/com/{fname}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots()
    for i, (c, l) in enumerate(zip(['blue', 'orange', 'green'], labels_xyz)):
        ax.plot(t, ef_zmp_position[:, i], label=f'residual ZMP {l}', color=c)
        ax.plot(t, fb_zmp_position[:, i], label=f'lip ZMP {l}',      color=c, linestyle='--')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Position [m]')
    ax.set_title('ZMP: LIP-based vs Residual-based')
    ax.grid(True); ax.legend(); fig.tight_layout()
    os.makedirs('images/com', exist_ok=True)
    fig.savefig('images/com/zmp_lip_vs_residual_plot.png')
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    #  FEET
    # ══════════════════════════════════════════════════════════════════════════

    plot_components(t, p_lsole_des,
        [fr'Des L Sole ${l}$' for l in labels_xyz],
        'Desired Left Sole Position', r'Position [$\mathrm{m}$]',
        'images/soles/references/desired_left_sole_position_plot.png')

    plot_components(t, p_rsole_des,
        [fr'Des R Sole ${l}$' for l in labels_xyz],
        'Desired Right Sole Position', r'Position [$\mathrm{m}$]',
        'images/soles/references/desired_right_sole_position_plot.png')

    plot_components(t, p_lsole_des - p_lsole_fb,
        [fr'L Sole Pos Error ${l}$' for l in labels_xyz],
        'Error – Left Sole Position', r'Position [$\mathrm{m}$]',
        'images/soles/errors/error_left_sole_position_plot.png')

    plot_components(t, p_rsole_des - p_rsole_fb,
        [fr'R Sole Pos Error ${l}$' for l in labels_xyz],
        'Error – Right Sole Position', r'Position [$\mathrm{m}$]',
        'images/soles/errors/error_right_sole_position_plot.png')

    plot_components(t, v_lsole_des - v_lsole_fb,
        [fr'L Sole Vel Error ${l}$' for l in labels_xyz],
        'Error – Left Sole Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/soles/errors/error_left_sole_velocity_plot.png')

    plot_components(t, v_rsole_des - v_rsole_fb,
        [fr'R Sole Vel Error ${l}$' for l in labels_xyz],
        'Error – Right Sole Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/soles/errors/error_right_sole_velocity_plot.png')

    plot_comparison(t, p_lsole_fb - p_lsole_fb[0], p_lsole_des - p_lsole_des[0],
        [fr'${l}$' for l in labels_xyz],
        'Left Sole Position', r'[$\mathrm{m}$]',
        'images/soles/errors/comparison_left_sole_position_plot.png')

    plot_comparison(t, p_rsole_fb - p_rsole_fb[0], p_rsole_des - p_rsole_des[0],
        [fr'${l}$' for l in labels_xyz],
        'Right Sole Position', r'[$\mathrm{m}$]',
        'images/soles/errors/comparison_right_sole_position_plot.png')

    plot_comparison(t, v_lsole_fb, v_lsole_des,
        [fr'${l}$' for l in labels_xyz],
        'Left Sole Velocity', r'[$\mathrm{m/s}$]',
        'images/soles/errors/comparison_left_sole_velocity_plot.png')

    plot_comparison(t, v_rsole_fb, v_rsole_des,
        [fr'${l}$' for l in labels_xyz],
        'Right Sole Velocity', r'[$\mathrm{m/s}$]',
        'images/soles/errors/comparison_right_sole_velocity_plot.png')

    # ══════════════════════════════════════════════════════════════════════════
    #  EKF
    # ══════════════════════════════════════════════════════════════════════════

    for gname, idx in grouped_indices.items():
        plot_components(t,
            np.column_stack([measured_joint_position[:, i] for i in idx] +
                            [ekf_joint_position[:, i]       for i in idx]),
            [f'Meas {joint_names[i].strip()}' for i in idx] +
            [f'Filt {joint_names[i].strip()}' for i in idx],
            gname.replace('_', ' ').title(), r'Position [$\mathrm{rad}$]',
            f'images/ekf/joints/positions/{gname}_position_plot.png', loc='upper left')

        plot_components(t,
            np.column_stack([measured_joint_velocity[:, i] for i in idx] +
                            [ekf_joint_velocity[:, i]       for i in idx]),
            [f'Meas {joint_names[i].strip()}' for i in idx] +
            [f'Filt {joint_names[i].strip()}' for i in idx],
            gname.replace('_', ' ').title(), r'Velocity [$\mathrm{rad/s}$]',
            f'images/ekf/joints/velocities/{gname}_velocity_plot.png', loc='upper left')

        plot_components(t,
            np.column_stack([measured_joint_position[:, i] - ekf_joint_position[:, i] for i in idx]),
            [f'Err {joint_names[i].strip()}' for i in idx],
            gname.replace('_', ' ').title(), r'Position [$\mathrm{rad}$]',
            f'images/ekf/joints/error/positions/error_{gname}_position_plot.png', loc='upper left')

        plot_components(t,
            np.column_stack([measured_joint_velocity[:, i] - ekf_joint_velocity[:, i] for i in idx]),
            [f'Err {joint_names[i].strip()}' for i in idx],
            gname.replace('_', ' ').title(), r'Velocity [$\mathrm{rad/s}$]',
            f'images/ekf/joints/error/velocities/error_{gname}_velocity_plot.png', loc='upper left')

    colormap   = plt.colormaps['tab10']
    line_styles = ['-', '--', '-.', ':']
    for quantity, measured, filtered, ylabel, fname in [
        ('Position', measured_joint_position, ekf_joint_position,
         r'Position [$\mathrm{rad}$]',   'error_joint_position_plot'),
        ('Velocity', measured_joint_velocity, ekf_joint_velocity,
         r'Velocity [$\mathrm{rad/s}$]', 'error_joint_velocity_plot'),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in range(num_joints):
            ax.plot(t, filtered[:, i] - measured[:, i],
                    label=joint_names[i].strip(),
                    color=colormap(i % 10),
                    linestyle=line_styles[(i // 10) % 4],
                    linewidth=2)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'Error EKF vs Measured – Joint {quantity}', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', frameon=True, fontsize=4)
        ax.tick_params(axis='both', labelsize=10); fig.tight_layout()
        os.makedirs('images/ekf/joints/error', exist_ok=True)
        fig.savefig(f'images/ekf/joints/error/{fname}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    plot_components(t, ekf_base_position,
        [fr'EKF Base Pos ${l}$' for l in labels_xyz],
        'EKF Base Position', r'Position [$\mathrm{m}$]',
        'images/ekf/base/base_position_plot.png')

    plot_components(t, ekf_base_velocity,
        [fr'EKF Base Vel ${l}$' for l in labels_xyz],
        'EKF Base Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/ekf/base/base_velocity_plot.png')

    plot_components(t, ekf_base_orientation,
        [fr'EKF Base Orient ${l}$' for l in labels_quat],
        'EKF Base Orientation Quat', r'Orientation [quat]',
        'images/ekf/base/base_orientation_quat_plot.png')

    plot_components(t, ekf_base_orientation_rpy,
        [fr'EKF Base Orient ${l}$' for l in labels_rpy],
        'EKF Base Orientation RPY', r'Orientation [$\mathrm{rad}$]',
        'images/ekf/base/base_orientation_rpy_plot.png')

    plot_components(t, ekf_base_angular_velocity,
        [fr'EKF Base AngVel ${l}$' for l in labels_xyz],
        'EKF Base Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/ekf/base/base_angular_velocity_plot.png')

    for metric_name, metric_fn in [
        ('Mean Squared Error', lambda a, b: np.mean((a - b) ** 2, axis=0)),
        ('Variance',           lambda a, b: np.var(a - b,         axis=0)),
    ]:
        short = 'mse' if 'Squared' in metric_name else 'var'
        plot_bar_joint(metric_fn(ekf_joint_position, measured_joint_position),
            joint_names, f'{metric_name} – EKF vs Measured Joint Position', metric_name,
            f'images/ekf/performance/{short}_joint_position_plot.png')
        plot_bar_joint(metric_fn(ekf_joint_velocity, measured_joint_velocity),
            joint_names, f'{metric_name} – EKF vs Measured Joint Velocity', metric_name,
            f'images/ekf/performance/{short}_joint_velocity_plot.png')

        m_pos  = metric_fn(ekf_base_position,         odometry_base_position)
        m_vel  = metric_fn(ekf_base_velocity,         odometry_base_velocity)
        m_ori  = metric_fn(ekf_base_orientation,      odometry_imu_orientation)
        m_angv = metric_fn(ekf_base_angular_velocity, measured_imu_angular_velocity)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(3),     m_pos,  label='Position',         color='skyblue', alpha=0.7)
        ax.bar(range(3,  6), m_vel,  label='Velocity',         color='orange',  alpha=0.7)
        ax.bar(range(6, 10), m_ori,  label='Orientation',      color='green',   alpha=0.7)
        ax.bar(range(10,13), m_angv, label='Angular Velocity', color='red',     alpha=0.7)
        ax.set_xlabel('Base State Index', fontsize=14)
        ax.set_ylabel(metric_name, fontsize=14)
        ax.set_title(f'{metric_name} – EKF Base States vs Simulated', fontsize=16)
        ax.set_xticks(range(13))
        ax.set_xticklabels(['Pos X','Pos Y','Pos Z','Vel X','Vel Y','Vel Z',
                            'Ori W','Ori X','Ori Y','Ori Z',
                            'AngVel X','AngVel Y','AngVel Z'], rotation=45, fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7); ax.legend(); fig.tight_layout()
        os.makedirs('images/ekf/performance', exist_ok=True)
        fig.savefig(f'images/ekf/performance/{short}_base_states_plot.png')
        plt.close(fig)

    plot_bar_joint(np.var(measured_joint_velocity, axis=0),
        joint_names, 'Variance of Feedback Joint Velocity', 'Variance',
        'images/ekf/performance/var_joint_velocity_measured_plot.png')

    plot_bar_joint(np.var(ekf_joint_velocity, axis=0),
        joint_names, 'Variance of Filtered Joint Velocity', 'Variance',
        'images/ekf/performance/var_joint_velocity_filtered_plot.png')

    plot_components(t, torso_orientation - des_torso_orientation,
        ['Roll Error', 'Pitch Error', 'Yaw Error'],
        'Torso Orientation Error', r'Orientation [$\mathrm{rad}$]',
        'images/ekf/torso_orientation_error_plot.png')

    plot_components(t, torso_angular_velocity,
        [f'AngVel {l}' for l in labels_xyz],
        'Torso Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/ekf/torso_angular_velocity_plot.png')

    plot_comparison(t, ekf_base_position,        odometry_base_position,
        [fr'${l}$' for l in labels_xyz],  'Base Position',  r'[$\mathrm{m}$]',
        'images/ekf/base/errors/comparison_base_position_plot.png')

    plot_comparison(t, ekf_base_velocity,        odometry_base_velocity,
        [fr'${l}$' for l in labels_xyz],  'Base Velocity',  r'[$\mathrm{m/s}$]',
        'images/ekf/base/errors/comparison_base_velocity_plot.png')

    plot_comparison(t, ekf_imu_orientation,      odometry_imu_orientation,
        [fr'${l}$' for l in labels_quat], 'IMU Orientation Quat', r'[quat]',
        'images/ekf/base/errors/comparison_imu_orientation_plot.png')

    plot_comparison(t, ekf_imu_orientation_rpy,  odometry_imu_orientation_rpy,
        [fr'${l}$' for l in labels_rpy],  'IMU Orientation RPY', r'[$\mathrm{rad}$]',
        'images/ekf/base/errors/comparison_imu_orientation_rpy_plot.png')

    plot_comparison(t, ekf_imu_angular_velocity, measured_imu_angular_velocity,
        [fr'${l}$' for l in labels_xyz],  'IMU Angular Velocity', r'[$\mathrm{rad/s}$]',
        'images/ekf/base/errors/comparison_imu_angular_velocity_plot.png')

    plot_components(t, ekf_base_position - odometry_base_position,
        [fr'Error Pos ${l}$' for l in labels_xyz],
        'Error – EKF vs Odometry Base Position', r'Position [$\mathrm{m}$]',
        'images/ekf/base/errors/error_base_position_plot.png')

    plot_components(t, ekf_base_velocity - odometry_base_velocity,
        [fr'Error Vel ${l}$' for l in labels_xyz],
        'Error – EKF vs Odometry Base Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/ekf/base/errors/error_base_velocity_plot.png')

    plot_components(t, ekf_imu_orientation - odometry_imu_orientation,
        [fr'Error Orient ${l}$' for l in labels_quat],
        'Error – EKF vs Measured IMU Orientation Quat', r'Orientation [quat]',
        'images/ekf/base/errors/error_imu_orientation_quat_plot.png')

    plot_components(t, ekf_imu_orientation_rpy - odometry_imu_orientation_rpy,
        [fr'Error Orient ${l}$' for l in labels_rpy],
        'Error – EKF vs Measured IMU Orientation RPY', r'Orientation [$\mathrm{rad}$]',
        'images/ekf/base/errors/error_imu_orientation_rpy_plot.png')

    plot_components(t, ekf_imu_angular_velocity - measured_imu_angular_velocity,
        [fr'Error AngVel ${l}$' for l in labels_xyz],
        'Error – EKF vs Measured IMU Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/ekf/base/errors/error_imu_angular_velocity_plot.png')

    # ══════════════════════════════════════════════════════════════════════════
    #  FEEDBACK
    # ══════════════════════════════════════════════════════════════════════════

    plot_components(t, odometry_base_position,
        [fr'Odometry Pos ${l}$' for l in labels_xyz],
        'Odometry Base Position', r'Position [$\mathrm{m}$]',
        'images/feedback/base/odometry_base_position_plot.png')

    plot_components(t, odometry_base_velocity,
        [fr'Odometry Vel ${l}$' for l in labels_xyz],
        'Odometry Base Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/feedback/base/odometry_base_velocity_plot.png')

    plot_components(t, odometry_imu_orientation,
        [fr'Odometry IMU Orient ${l}$' for l in labels_quat],
        'Odometry IMU Orientation Quat', r'Orientation [quat]',
        'images/feedback/base/odometry_imu_orientation_quat_plot.png')

    plot_components(t, odometry_imu_orientation_rpy,
        [fr'Odometry IMU Orient ${l}$' for l in labels_rpy],
        'Odometry IMU Orientation RPY', r'Orientation [$\mathrm{rad}$]',
        'images/feedback/base/odometry_imu_orientation_rpy_plot.png')

    plot_components(t, measured_imu_orientation,
        [fr'Measured IMU Orient ${l}$' for l in labels_quat],
        'Measured IMU Orientation Quat', r'Orientation [quat]',
        'images/feedback/base/measured_imu_orientation_quat_plot.png')

    plot_components(t, measured_imu_orientation_rpy,
        [fr'Measured IMU Orient ${l}$' for l in labels_rpy],
        'Measured IMU Orientation RPY', r'Orientation [$\mathrm{rad}$]',
        'images/feedback/base/measured_imu_orientation_rpy_plot.png')

    plot_components(t, measured_imu_angular_velocity,
        [fr'Measured AngVel ${l}$' for l in labels_xyz],
        'Measured IMU Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/feedback/base/measured_imu_angular_velocity_plot.png')

    plot_components(t, measured_imu_accelerometer,
        [fr'Measured Acc ${l}$' for l in labels_xyz],
        'Measured IMU Acceleration', r'Acceleration [$\mathrm{m/s^2}$]',
        'images/feedback/base/measured_imu_acceleration_plot.png')

    fig, ax = plt.subplots(figsize=(18, 12))
    for i in range(measured_joint_velocity.shape[1]):
        ax.plot(t, measured_joint_velocity[:, i], label=joint_names[i].strip())
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Velocity [rad/s]')
    ax.set_title('Feedback Joint Velocities'); ax.grid(True); ax.legend()
    fig.tight_layout()
    os.makedirs('images/feedback/joints/velocities', exist_ok=True)
    fig.savefig('images/feedback/joints/velocities/overall_joint_velocity_plot.png')
    plt.close(fig)

    for gname, idx in grouped_indices.items():
        plot_components(t, measured_joint_position[:, idx],
            [joint_names[i].strip() for i in idx],
            gname.replace('_', ' ').title(), r'Position [$\mathrm{rad}$]',
            f'images/feedback/joints/positions/{gname}_position_plot.png', loc='upper left')

        plot_components(t, measured_joint_velocity[:, idx],
            [joint_names[i].strip() for i in idx],
            gname.replace('_', ' ').title(), r'Velocity [$\mathrm{rad/s}$]',
            f'images/feedback/joints/velocities/{gname}_velocity_plot.png', loc='upper left')

    # ══════════════════════════════════════════════════════════════════════════
    #  EXECUTION TIMES
    # ══════════════════════════════════════════════════════════════════════════

    exec_times = {
        'EKF':    execution_time_ekf,
        'KF':     execution_time_kf,
        'MPC':    execution_time_mpc,
        'WBC':    execution_time_wbc,
        'Update': execution_time_update,
    }

    for name, times in exec_times.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(iter_t, times, linewidth=2.0, label=name)
        if name == 'Update':
            ax.axhline(y=2000, linestyle='--', linewidth=1.5,
                       label='Real-time threshold (2000 µs)')
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(r'Execution Time [$\mu s$]', fontsize=11)
        ax.set_title(f'{name} Execution Time per Iteration', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(frameon=True, fontsize=10); fig.tight_layout()
        os.makedirs('images/execution_times', exist_ok=True)
        fig.savefig(f'images/execution_times/{name}_execution_time_plot.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    total_time = sum(exec_times[k] for k in ('EKF', 'KF', 'MPC', 'WBC'))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iter_t, total_time, linewidth=2.0, label='Total Execution Time')
    ax.axhline(y=2000, linestyle='--', linewidth=1.5,
               label='Real-time threshold (2000 µs)', color='red')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel(r'Total Execution Time [$\mu s$]', fontsize=11)
    ax.set_title('Total Execution Time per Iteration', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(frameon=True, fontsize=10); fig.tight_layout()
    fig.savefig('images/execution_times/total_execution_time_plot.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)