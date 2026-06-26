import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

CTRL_HZ = 500
DT = 1.0 / CTRL_HZ


# ── helpers ───────────────────────────────────────────────────────────────────

def _sub(a, b):
    """a - b clipped to min length, or None if either operand is None."""
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    return a[:n] - b[:n]

def _sub0(a):
    """a - a[0], or None if a is None."""
    return None if a is None else a - a[0]


def plot_components(t, data, labels, title, ylabel, path,
                    figsize=(7, 4), loc='best', mean_last_s=5.0):
    if data is None:
        return
    if data.ndim == 1:
        data = data[:, np.newaxis]
    t = t[:len(data)]
    fig, ax = plt.subplots(figsize=figsize)
    for col, lbl in zip(data.T, labels):
        ax.plot(t, col, label=lbl, linewidth=2.0)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc=loc, frameon=True, fontsize=9)
    ax.tick_params(axis='both', labelsize=10)
    if mean_last_s > 0 and len(data) > 0:
        n_win = min(int(mean_last_s * CTRL_HZ), len(data))
        lines = '\n'.join(
            f'{lbl}: {np.mean(col[-n_win:]):.3f}'
            for col, lbl in zip(data.T, labels)
        )
        ax.text(0.98, 0.02, f'Mean last {mean_last_s:.0f}s\n{lines}',
                transform=ax.transAxes, fontsize=7, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.6))
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(t, actual, desired, coord_labels, title_prefix, ylabel, path):
    if actual is None or desired is None:
        return
    _nr = min(len(actual), len(desired))
    actual, desired, t = actual[:_nr], desired[:_nr], t[:_nr]
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


def try_load(folder, name):
    """Load name.txt if it exists, else return None.

    Guarantees 2-D output for multi-column files even with a single data row.
    """
    path = f"{folder}/{name}.txt"
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path)
    if data.ndim == 1:
        with open(path) as f:
            first_line = f.readline().strip()
        if len(first_line.split()) > 1:
            data = data.reshape(1, -1)
    return data


def plot_bar_joint(values, joint_names, title, ylabel, path, figsize=(10, 6)):
    if values is None:
        return
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
    num_joints  = len(joint_names)

    # ── load all sensor / EKF logs ────────────────────────────────────────────
    _sens_raw = {}
    _T_sensor = None
    for _nm in ('pelvis_acc', 'pelvis_gyro', 'pelvis_rpy', 'pelvis_quat',
                'torso_acc',  'torso_gyro',  'torso_rpy',  'torso_quat',
                'odom_pos', 'odom_vel', 'odom_quat', 'odom_rpy',
                'joint_pos', 'joint_vel',
                'filtered_base_position', 'filtered_base_velocity',
                'filtered_base_quat', 'filtered_base_rpy', 'filtered_base_ang_vel',
                'filtered_joint_position', 'filtered_joint_velocity'):
        _d = try_load(folder, _nm)
        if _d is not None:
            _sens_raw[_nm] = _d
            if _T_sensor is None:
                _T_sensor = _d.shape[0]

    # ── determine WBC window length ───────────────────────────────────────────
    _ctrl_anchor = None
    for _try in ('com_position', 'input_torque', 'wbc_accelerations', 'execution_time_wbc'):
        _ctrl_anchor = try_load(folder, _try)
        if _ctrl_anchor is not None:
            break
    if _ctrl_anchor is not None:
        total = _ctrl_anchor.shape[0] if _ctrl_anchor.ndim > 1 else len(_ctrl_anchor)
    elif _T_sensor is not None:
        total = _T_sensor
    else:
        print("No data found in", folder)
        exit(1)

    n_wbc = total - end_trim if end_trim > 0 else total
    t     = np.linspace(0.0, DT * n_wbc, n_wbc)
    iter_t = np.arange(n_wbc)

    # L(): load WBC/controller signal aligned to the WBC window
    def L(name):
        d = try_load(folder, name)
        if d is None:
            return None
        if d.ndim == 0:
            d = d.reshape(1)
        rows = d.shape[0]
        s = max(0, rows - n_wbc - end_trim)
        e = rows - end_trim if end_trim > 0 else rows
        return d[s:e]

    # ── load WBC / controller signals ─────────────────────────────────────────
    com_position             = L('com_position')
    com_velocity             = L('com_velocity')
    zmp_position             = L('zmp_position')
    kf_com_position          = L('kf_com_position')
    kf_com_velocity          = L('kf_com_velocity')
    kf_zmp_position          = L('kf_zmp_position')
    des_com_position         = L('des_com_position')
    des_com_velocity         = L('des_com_velocity')
    des_zmp_position         = L('des_zmp_position')
    des_com_acceleration     = L('des_com_acceleration')
    input_torque             = L('input_torque')
    wbc_accelerations        = L('wbc_accelerations')
    estimated_force_lsole    = L('estimated_force_lsole')
    estimated_force_rsole    = L('estimated_force_rsole')
    wbc_force_lsole          = L('wbc_force_lsole')
    wbc_force_rsole          = L('wbc_force_rsole')
    p_lsole                  = L('p_lsole')
    p_rsole                  = L('p_rsole')
    v_lsole                  = L('v_lsole')
    v_rsole                  = L('v_rsole')
    p_lsole_des              = L('p_lsole_des')
    p_rsole_des              = L('p_rsole_des')
    v_lsole_des              = L('v_lsole_des')
    v_rsole_des              = L('v_rsole_des')
    lsole_orientation        = L('lsole_orientation')
    rsole_orientation        = L('rsole_orientation')
    des_lsole_orientation    = L('des_lsole_orientation')
    des_rsole_orientation    = L('des_rsole_orientation')
    torso_orientation        = L('torso_orientation')
    torso_angular_velocity   = L('torso_angular_velocity')
    des_torso_orientation    = L('des_torso_orientation')
    des_torso_angular_velocity = L('des_torso_angular_velocity')
    execution_time_ekf       = L('execution_time_ekf')
    execution_time_kf        = L('execution_time_kf')
    execution_time_mpc       = L('execution_time_mpc')
    execution_time_wbc       = L('execution_time_wbc')
    execution_time_update    = L('execution_time_update')

    # ── sensor / EKF time axis (full, from tick 0) ────────────────────────────
    if _T_sensor is not None:
        _ae    = _T_sensor - end_trim if end_trim > 0 else _T_sensor
        t_full = np.linspace(0.0, DT * _ae, _ae)

        def _full(nm):
            d = _sens_raw.get(nm)
            return d[:_ae] if d is not None else None
    else:
        t_full = t
        def _full(nm): return None

    # ── sensor / EKF variables (aligned to feedback, use t_full) ─────────────
    odometry_base_position         = _full('odom_pos')
    odometry_base_velocity         = _full('odom_vel')
    odometry_imu_orientation       = _full('odom_quat')
    odometry_imu_orientation_rpy   = _full('odom_rpy')
    measured_joint_position        = _full('joint_pos')
    measured_joint_velocity        = _full('joint_vel')
    measured_imu_pelvis_angular_velocity = _full('pelvis_gyro')
    measured_imu_pelvis_accelerometer    = _full('pelvis_acc')
    measured_imu_pelvis_rpy              = _full('pelvis_rpy')
    measured_imu_pelvis_quaternion       = _full('pelvis_quat')
    measured_imu_torso_rpy               = _full('torso_rpy')
    measured_imu_torso_quaternion        = _full('torso_quat')
    measured_imu_torso_accelerometer     = _full('torso_acc')
    measured_imu_torso_angular_velocity  = _full('torso_gyro')
    filtered_base_position         = _full('filtered_base_position')
    filtered_base_velocity         = _full('filtered_base_velocity')
    filtered_base_orientation      = _full('filtered_base_quat')
    filtered_base_orientation_rpy  = _full('filtered_base_rpy')
    filtered_base_angular_velocity = _full('filtered_base_ang_vel')
    filtered_joint_position        = _full('filtered_joint_position')
    filtered_joint_velocity        = _full('filtered_joint_velocity')

    # ── labels / joint groups ─────────────────────────────────────────────────
    labels_xyz  = ['x', 'y', 'z']
    labels_quat = ['w', 'x', 'y', 'z']
    labels_rpy  = ['r', 'p', 'y']

    grouped_indices = defaultdict(list)
    _kw_groups = {
        'hips':      'hip',
        'knees':     'knee',
        'ankles':    'ankle',
        'waist':     'waist',
        'shoulders': 'shoulder',
        'elbows':    'elbow',
        'wrists':    'wrist',
    }
    for i, _jname in enumerate(joint_names):
        _jname_lower = _jname.strip().lower()
        for gname, kw in _kw_groups.items():
            if kw in _jname_lower:
                grouped_indices[gname].append(i)
                break


    # ══════════════════════════════════════════════════════════════════════════
    #  WBC SOLUTIONS
    # ══════════════════════════════════════════════════════════════════════════

    if input_torque is not None:
        for gname, idx in grouped_indices.items():
            plot_components(t, input_torque[:, idx],
                [joint_names[i].strip() for i in idx],
                f'WBC Joint Torques – {gname}', 'Torque [Nm]',
                f'images/wbc_solutions/torques/{gname}_torques.png')

    if wbc_accelerations is not None:
        plot_components(t, wbc_accelerations[:, :3],
            [f'Linear acc {l}' for l in labels_xyz],
            'WBC Base Linear Acceleration', r'Acceleration [m/s²]',
            'images/wbc_solutions/accelerations/base_linear_acceleration.png')
        plot_components(t, wbc_accelerations[:, 3:6],
            [f'Angular acc {l}' for l in labels_xyz],
            'WBC Base Angular Acceleration', r'Acceleration [rad/s²]',
            'images/wbc_solutions/accelerations/base_angular_acceleration.png')
        for gname, idx in grouped_indices.items():
            plot_components(t, wbc_accelerations[:, [i + 6 for i in idx]],
                [joint_names[i].strip() for i in idx],
                f'WBC Joint Acceleration – {gname}', r'Acceleration [rad/s²]',
                f'images/wbc_solutions/accelerations/{gname}_acceleration.png')

    labels_wrench = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    plot_components(t, wbc_force_lsole,  labels_wrench,
        'WBC Optimal Left Foot Wrench',  r'Force [N] / Torque [Nm]',
        'images/wbc_solutions/forces/wbc_force_left_sole.png')
    plot_components(t, wbc_force_rsole,  labels_wrench,
        'WBC Optimal Right Foot Wrench', r'Force [N] / Torque [Nm]',
        'images/wbc_solutions/forces/wbc_force_right_sole.png')
    plot_components(t, estimated_force_lsole,
        [f'Left sole force {l}' for l in labels_xyz],
        'Estimated Forces on Left Sole', 'Force [N]',
        'images/soles/forces/estimated_force_left_sole.png')
    plot_components(t, estimated_force_rsole,
        [f'Right sole force {l}' for l in labels_xyz],
        'Estimated Forces on Right Sole', 'Force [N]',
        'images/soles/forces/estimated_force_right_sole.png')

    # ══════════════════════════════════════════════════════════════════════════
    #  CoM AND ZMP
    # ══════════════════════════════════════════════════════════════════════════

    plot_components(t, des_com_acceleration,
        [fr'Des CoM Acc ${l}$' for l in labels_xyz],
        'Desired CoM Acceleration', r'Acceleration [$\mathrm{m/s^2}$]',
        'images/com/references/des_com_acceleration_plot.png')
    plot_components(t, _sub0(des_com_position),
        [fr'Des CoM Pos ${l}$' for l in labels_xyz],
        'Desired CoM Position', r'Position [$\mathrm{m}$]',
        'images/com/references/des_com_position_plot.png')
    plot_components(t, des_com_velocity,
        [fr'Des CoM Vel ${l}$' for l in labels_xyz],
        'Desired CoM Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/com/references/des_com_velocity_plot.png')
    plot_components(t, _sub0(des_zmp_position),
        [fr'Des ZMP Pos ${l}$' for l in labels_xyz],
        'Desired ZMP Position', r'Position [$\mathrm{m}$]',
        'images/com/references/des_zmp_position_plot.png')

    plot_components(t, _sub(des_zmp_position, kf_zmp_position),
        [fr'ZMP Error ${l}$' for l in labels_xyz],
        'ZMP Position Error', r'Position [$\mathrm{m}$]',
        'images/com/errors/error_zmp_position_plot.png')
    plot_components(t, _sub(des_com_position, kf_com_position),
        [fr'CoM Pos Error ${l}$' for l in labels_xyz],
        'CoM Position Error', r'Position [$\mathrm{m}$]',
        'images/com/errors/error_com_position_plot.png')
    plot_components(t, _sub(des_com_velocity, kf_com_velocity),
        [fr'CoM Vel Error ${l}$' for l in labels_xyz],
        'CoM Velocity Error', r'Velocity [$\mathrm{m/s}$]',
        'images/com/errors/error_com_velocity_plot.png')

    plot_comparison(t, _sub0(kf_zmp_position), _sub0(des_zmp_position),
        [fr'${l}$' for l in labels_xyz], 'ZMP Position', r'[$\mathrm{m}$]',
        'images/com/errors/comparison_zmp_position_plot.png')
    plot_comparison(t, kf_com_position, des_com_position,
        [fr'${l}$' for l in labels_xyz], 'CoM Position', r'[$\mathrm{m}$]',
        'images/com/errors/comparison_com_position_plot.png')
    plot_comparison(t, kf_com_velocity, des_com_velocity,
        [fr'${l}$' for l in labels_xyz], 'CoM Velocity', r'[$\mathrm{m/s}$]',
        'images/com/errors/comparison_com_velocity_plot.png')

    if kf_com_position is not None and kf_zmp_position is not None and \
            p_lsole is not None and p_rsole is not None:
        for axis, direction, fname in [(0, 'forward', 'motion_x'), (1, 'lateral', 'motion_y')]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t, kf_com_position[:, axis], label=fr'CoM ${labels_xyz[axis]}$',        linewidth=2.0)
            ax.plot(t, kf_zmp_position[:, axis], label=fr'ZMP ${labels_xyz[axis]}$',        linewidth=2.0)
            ax.plot(t, p_lsole[:, axis],         label=fr'Left foot ${labels_xyz[axis]}$',  linewidth=2.0)
            ax.plot(t, p_rsole[:, axis],         label=fr'Right foot ${labels_xyz[axis]}$', linewidth=2.0)
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

    plot_components(t, _sub(p_lsole_des, p_lsole),
        [fr'L Sole Pos Error ${l}$' for l in labels_xyz],
        'Error – Left Sole Position', r'Position [$\mathrm{m}$]',
        'images/soles/errors/error_left_sole_position_plot.png')
    plot_components(t, _sub(p_rsole_des, p_rsole),
        [fr'R Sole Pos Error ${l}$' for l in labels_xyz],
        'Error – Right Sole Position', r'Position [$\mathrm{m}$]',
        'images/soles/errors/error_right_sole_position_plot.png')
    plot_components(t, _sub(v_lsole_des, v_lsole),
        [fr'L Sole Vel Error ${l}$' for l in labels_xyz],
        'Error – Left Sole Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/soles/errors/error_left_sole_velocity_plot.png')
    plot_components(t, _sub(v_rsole_des, v_rsole),
        [fr'R Sole Vel Error ${l}$' for l in labels_xyz],
        'Error – Right Sole Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/soles/errors/error_right_sole_velocity_plot.png')

    plot_comparison(t, _sub0(p_lsole), _sub0(p_lsole_des),
        [fr'${l}$' for l in labels_xyz], 'Left Sole Position', r'[$\mathrm{m}$]',
        'images/soles/errors/comparison_left_sole_position_plot.png')
    plot_comparison(t, _sub0(p_rsole), _sub0(p_rsole_des),
        [fr'${l}$' for l in labels_xyz], 'Right Sole Position', r'[$\mathrm{m}$]',
        'images/soles/errors/comparison_right_sole_position_plot.png')
    plot_comparison(t, v_lsole, v_lsole_des,
        [fr'${l}$' for l in labels_xyz], 'Left Sole Velocity', r'[$\mathrm{m/s}$]',
        'images/soles/errors/comparison_left_sole_velocity_plot.png')
    plot_comparison(t, v_rsole, v_rsole_des,
        [fr'${l}$' for l in labels_xyz], 'Right Sole Velocity', r'[$\mathrm{m/s}$]',
        'images/soles/errors/comparison_right_sole_velocity_plot.png')

    # ══════════════════════════════════════════════════════════════════════════
    #  EKF  (all plots use t_full — aligned with feedback, not WBC)
    # ══════════════════════════════════════════════════════════════════════════

    # ── base state ────────────────────────────────────────────────────────────
    plot_components(t_full, filtered_base_position,
        [fr'Base Pos ${l}$' for l in labels_xyz],
        'Filtered Base Position', r'Position [$\mathrm{m}$]',
        'images/ekf/base/base_position_plot.png')
    plot_components(t_full, filtered_base_velocity,
        [fr'Base Vel ${l}$' for l in labels_xyz],
        'Filtered Base Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/ekf/base/base_velocity_plot.png')
    plot_components(t_full, filtered_base_orientation,
        [fr'Base Orient ${l}$' for l in labels_quat],
        'Filtered Base Orientation Quat', r'Orientation [quat]',
        'images/ekf/base/base_orientation_quat_plot.png')
    plot_components(t_full, filtered_base_orientation_rpy,
        [fr'Base Orient ${l}$' for l in labels_rpy],
        'Filtered Base Orientation RPY', r'Orientation [$\mathrm{rad}$]',
        'images/ekf/base/base_orientation_rpy_plot.png')
    plot_components(t_full, filtered_base_angular_velocity,
        [fr'Base AngVel ${l}$' for l in labels_xyz],
        'Filtered Base Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/ekf/base/base_angular_velocity_plot.png')

    # ── base vs odometry comparison ───────────────────────────────────────────
    plot_comparison(t_full, filtered_base_position, odometry_base_position,
        [fr'${l}$' for l in labels_xyz], 'Base Position', r'[$\mathrm{m}$]',
        'images/ekf/base/errors/comparison_base_position_plot.png')
    plot_comparison(t_full, filtered_base_velocity, odometry_base_velocity,
        [fr'${l}$' for l in labels_xyz], 'Base Velocity', r'[$\mathrm{m/s}$]',
        'images/ekf/base/errors/comparison_base_velocity_plot.png')
    plot_comparison(t_full, filtered_base_orientation, odometry_imu_orientation,
        [fr'${l}$' for l in labels_quat], 'Base Orientation Quat', r'[quat]',
        'images/ekf/base/errors/comparison_base_orientation_plot.png')
    plot_comparison(t_full, filtered_base_orientation_rpy, odometry_imu_orientation_rpy,
        [fr'${l}$' for l in labels_rpy], 'Base Orientation RPY', r'[$\mathrm{rad}$]',
        'images/ekf/base/errors/comparison_base_orientation_rpy_plot.png')
    plot_comparison(t_full, filtered_base_angular_velocity, measured_imu_pelvis_angular_velocity,
        [fr'${l}$' for l in labels_xyz], 'Base Angular Velocity', r'[$\mathrm{rad/s}$]',
        'images/ekf/base/errors/comparison_base_angular_velocity_plot.png')

    plot_components(t_full, _sub(filtered_base_position, odometry_base_position),
        [fr'Error Pos ${l}$' for l in labels_xyz],
        'Error – Filtered vs Odometry Base Position', r'Position [$\mathrm{m}$]',
        'images/ekf/base/errors/error_base_position_plot.png')
    plot_components(t_full, _sub(filtered_base_velocity, odometry_base_velocity),
        [fr'Error Vel ${l}$' for l in labels_xyz],
        'Error – Filtered vs Odometry Base Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/ekf/base/errors/error_base_velocity_plot.png')
    plot_components(t_full, _sub(filtered_base_orientation, odometry_imu_orientation),
        [fr'Error Orient ${l}$' for l in labels_quat],
        'Error – Filtered vs Odometry Orientation Quat', r'Orientation [quat]',
        'images/ekf/base/errors/error_base_orientation_quat_plot.png')
    plot_components(t_full, _sub(filtered_base_orientation_rpy, odometry_imu_orientation_rpy),
        [fr'Error Orient ${l}$' for l in labels_rpy],
        'Error – Filtered vs Odometry Orientation RPY', r'Orientation [$\mathrm{rad}$]',
        'images/ekf/base/errors/error_base_orientation_rpy_plot.png')
    plot_components(t_full, _sub(filtered_base_angular_velocity, measured_imu_pelvis_angular_velocity),
        [fr'Error AngVel ${l}$' for l in labels_xyz],
        'Error – Filtered vs Measured Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/ekf/base/errors/error_base_angular_velocity_plot.png')

    # ── joint position / velocity (EKF vs measured) ───────────────────────────
    if all(x is not None for x in (measured_joint_position, measured_joint_velocity,
                                    filtered_joint_position, filtered_joint_velocity)):
        for gname, idx in grouped_indices.items():
            _nj = min(len(measured_joint_position), len(filtered_joint_position))
            plot_components(t_full,
                np.column_stack([measured_joint_position[:_nj, i] for i in idx] +
                                [filtered_joint_position[:_nj, i] for i in idx]),
                [f'Meas {joint_names[i].strip()}' for i in idx] +
                [f'Filt {joint_names[i].strip()}' for i in idx],
                gname.replace('_', ' ').title(), r'Position [$\mathrm{rad}$]',
                f'images/ekf/joints/positions/{gname}_position_plot.png', loc='upper left')

            _nj = min(len(measured_joint_velocity), len(filtered_joint_velocity))
            plot_components(t_full,
                np.column_stack([measured_joint_velocity[:_nj, i] for i in idx] +
                                [filtered_joint_velocity[:_nj, i] for i in idx]),
                [f'Meas {joint_names[i].strip()}' for i in idx] +
                [f'Filt {joint_names[i].strip()}' for i in idx],
                gname.replace('_', ' ').title(), r'Velocity [$\mathrm{rad/s}$]',
                f'images/ekf/joints/velocities/{gname}_velocity_plot.png', loc='upper left')

            plot_components(t_full,
                _sub(measured_joint_position[:, idx], filtered_joint_position[:, idx]),
                [f'Err {joint_names[i].strip()}' for i in idx],
                gname.replace('_', ' ').title(), r'Position [$\mathrm{rad}$]',
                f'images/ekf/joints/error/positions/error_{gname}_position_plot.png', loc='upper left')

            plot_components(t_full,
                _sub(measured_joint_velocity[:, idx], filtered_joint_velocity[:, idx]),
                [f'Err {joint_names[i].strip()}' for i in idx],
                gname.replace('_', ' ').title(), r'Velocity [$\mathrm{rad/s}$]',
                f'images/ekf/joints/error/velocities/error_{gname}_velocity_plot.png', loc='upper left')

        colormap    = plt.colormaps['tab10']
        line_styles = ['-', '--', '-.', ':']
        for quantity, measured, filtered, ylabel, fname in [
            ('Position', measured_joint_position, filtered_joint_position,
             r'Position [$\mathrm{rad}$]',   'error_joint_position_plot'),
            ('Velocity', measured_joint_velocity, filtered_joint_velocity,
             r'Velocity [$\mathrm{rad/s}$]', 'error_joint_velocity_plot'),
        ]:
            _nj = min(len(measured), len(filtered))
            fig, ax = plt.subplots(figsize=(7, 4))
            for i in range(num_joints):
                ax.plot(t_full[:_nj], filtered[:_nj, i] - measured[:_nj, i],
                        label=joint_names[i].strip(),
                        color=colormap(i % 10),
                        linestyle=line_styles[(i // 10) % 4],
                        linewidth=2)
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'Error EKF vs Measured – Joint {quantity}', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(loc='best', frameon=True, fontsize=4)
            ax.tick_params(axis='both', labelsize=10)
            fig.tight_layout()
            os.makedirs('images/ekf/joints/error', exist_ok=True)
            fig.savefig(f'images/ekf/joints/error/{fname}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

    # ── EKF performance metrics ───────────────────────────────────────────────
    if filtered_joint_position is not None and measured_joint_position is not None:
        for metric_name, metric_fn in [
            ('Mean Squared Error', lambda a, b: np.mean((_sub(a, b)) ** 2, axis=0)),
            ('Variance',           lambda a, b: np.var(_sub(a, b),         axis=0)),
        ]:
            short = 'mse' if 'Squared' in metric_name else 'var'
            plot_bar_joint(metric_fn(filtered_joint_position, measured_joint_position),
                joint_names, f'{metric_name} – Filtered vs Measured Joint Position', metric_name,
                f'images/ekf/performance/{short}_joint_position_plot.png')
            plot_bar_joint(metric_fn(filtered_joint_velocity, measured_joint_velocity),
                joint_names, f'{metric_name} – Filtered vs Measured Joint Velocity', metric_name,
                f'images/ekf/performance/{short}_joint_velocity_plot.png')

            if all(x is not None for x in (filtered_base_position, odometry_base_position,
                                            filtered_base_velocity, odometry_base_velocity,
                                            filtered_base_orientation, odometry_imu_orientation,
                                            filtered_base_angular_velocity,
                                            measured_imu_pelvis_angular_velocity)):
                m_pos  = metric_fn(filtered_base_position,         odometry_base_position)
                m_vel  = metric_fn(filtered_base_velocity,         odometry_base_velocity)
                m_ori  = metric_fn(filtered_base_orientation,      odometry_imu_orientation)
                m_angv = metric_fn(filtered_base_angular_velocity, measured_imu_pelvis_angular_velocity)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(3),     m_pos,  label='Position',         color='skyblue', alpha=0.7)
                ax.bar(range(3,  6), m_vel,  label='Velocity',         color='orange',  alpha=0.7)
                ax.bar(range(6, 10), m_ori,  label='Orientation',      color='green',   alpha=0.7)
                ax.bar(range(10,13), m_angv, label='Angular Velocity', color='red',     alpha=0.7)
                ax.set_xlabel('Base State Index', fontsize=14)
                ax.set_ylabel(metric_name, fontsize=14)
                ax.set_title(f'{metric_name} – Filtered Base States vs Odometry', fontsize=16)
                ax.set_xticks(range(13))
                ax.set_xticklabels(['Pos X','Pos Y','Pos Z','Vel X','Vel Y','Vel Z',
                                    'Ori W','Ori X','Ori Y','Ori Z',
                                    'AngVel X','AngVel Y','AngVel Z'], rotation=45, fontsize=8)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.legend()
                fig.tight_layout()
                os.makedirs('images/ekf/performance', exist_ok=True)
                fig.savefig(f'images/ekf/performance/{short}_base_states_plot.png')
                plt.close(fig)

    plot_bar_joint(
        np.var(measured_joint_velocity, axis=0) if measured_joint_velocity is not None else None,
        joint_names, 'Variance of Feedback Joint Velocity', 'Variance',
        'images/ekf/performance/var_joint_velocity_measured_plot.png')
    plot_bar_joint(
        np.var(filtered_joint_velocity, axis=0) if filtered_joint_velocity is not None else None,
        joint_names, 'Variance of Filtered Joint Velocity', 'Variance',
        'images/ekf/performance/var_joint_velocity_filtered_plot.png')

    # ── torso (WBC signals, use t) ────────────────────────────────────────────
    plot_components(t, _sub(torso_orientation, des_torso_orientation),
        ['Roll Error', 'Pitch Error', 'Yaw Error'],
        'Torso Orientation Error', r'Orientation [$\mathrm{rad}$]',
        'images/ekf/torso_orientation_error_plot.png')
    plot_components(t, torso_angular_velocity,
        [f'AngVel {l}' for l in labels_xyz],
        'Torso Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/ekf/torso_angular_velocity_plot.png')

    # ══════════════════════════════════════════════════════════════════════════
    #  FEEDBACK  (full sensor data – from tick 0, use t_full)
    # ══════════════════════════════════════════════════════════════════════════

    plot_components(t_full, odometry_base_position,
        [fr'Odometry Pos ${l}$' for l in labels_xyz],
        'Odometry Base Position', r'Position [$\mathrm{m}$]',
        'images/feedback/base/odometry_base_position_plot.png')
    plot_components(t_full, odometry_base_velocity,
        [fr'Odometry Vel ${l}$' for l in labels_xyz],
        'Odometry Base Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/feedback/base/odometry_base_velocity_plot.png')
    plot_components(t_full, odometry_imu_orientation,
        [fr'Odometry IMU Orient ${l}$' for l in labels_quat],
        'Odometry IMU Orientation Quat', r'Orientation [quat]',
        'images/feedback/base/odometry_imu_orientation_quat_plot.png')
    plot_components(t_full, odometry_imu_orientation_rpy,
        [fr'Odometry IMU Orient ${l}$' for l in labels_rpy],
        'Odometry IMU Orientation RPY', r'Orientation [$\mathrm{rad}$]',
        'images/feedback/base/odometry_imu_orientation_rpy_plot.png')

    plot_components(t_full, measured_imu_pelvis_angular_velocity,
        [fr'Pelvis AngVel ${l}$' for l in labels_xyz],
        'Pelvis IMU – Gyroscope', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/feedback/base/pelvis_imu_angular_velocity_plot.png')
    plot_components(t_full, measured_imu_pelvis_accelerometer,
        [fr'Pelvis Acc ${l}$' for l in labels_xyz],
        'Pelvis IMU – Accelerometer', r'Acceleration [$\mathrm{m/s^2}$]',
        'images/feedback/base/pelvis_imu_acceleration_plot.png')
    plot_components(t_full, measured_imu_pelvis_rpy,
        [fr'Pelvis RPY ${l}$' for l in labels_rpy],
        'Pelvis IMU – RPY', r'RPY [$\mathrm{rad}$]',
        'images/feedback/base/pelvis_imu_rpy_plot.png')
    plot_components(t_full, measured_imu_pelvis_quaternion,
        [fr'Pelvis Quat ${l}$' for l in labels_quat],
        'Pelvis IMU – Quaternion', r'Quaternion',
        'images/feedback/base/pelvis_imu_quaternion_plot.png')

    plot_components(t_full, measured_imu_torso_rpy,
        [fr'Torso RPY ${l}$' for l in labels_rpy],
        'Torso IMU – RPY', r'RPY [$\mathrm{rad}$]',
        'images/feedback/base/torso_imu_rpy_plot.png')
    plot_components(t_full, measured_imu_torso_quaternion,
        [fr'Torso Quat ${l}$' for l in labels_quat],
        'Torso IMU – Quaternion', r'Quaternion',
        'images/feedback/base/torso_imu_quaternion_plot.png')
    plot_components(t_full, measured_imu_torso_accelerometer,
        [fr'Torso Acc ${l}$' for l in labels_xyz],
        'Torso IMU – Accelerometer', r'Acceleration [$\mathrm{m/s^2}$]',
        'images/feedback/base/torso_imu_acceleration_plot.png')
    plot_components(t_full, measured_imu_torso_angular_velocity,
        [fr'Torso AngVel ${l}$' for l in labels_xyz],
        'Torso IMU – Gyroscope', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/feedback/base/torso_imu_angular_velocity_plot.png')

    if measured_joint_velocity is not None:
        fig, ax = plt.subplots(figsize=(18, 12))
        for i in range(measured_joint_velocity.shape[1]):
            ax.plot(t_full, measured_joint_velocity[:, i], label=joint_names[i].strip())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [rad/s]')
        ax.set_title('Feedback Joint Velocities')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        os.makedirs('images/feedback/joints/velocities', exist_ok=True)
        fig.savefig('images/feedback/joints/velocities/overall_joint_velocity_plot.png')
        plt.close(fig)

    for gname, idx in grouped_indices.items():
        if measured_joint_position is not None:
            plot_components(t_full, measured_joint_position[:, idx],
                [joint_names[i].strip() for i in idx],
                gname.replace('_', ' ').title(), r'Position [$\mathrm{rad}$]',
                f'images/feedback/joints/positions/{gname}_position_plot.png', loc='upper left')
        if measured_joint_velocity is not None:
            plot_components(t_full, measured_joint_velocity[:, idx],
                [joint_names[i].strip() for i in idx],
                gname.replace('_', ' ').title(), r'Velocity [$\mathrm{rad/s}$]',
                f'images/feedback/joints/velocities/{gname}_velocity_plot.png', loc='upper left')

    # ══════════════════════════════════════════════════════════════════════════
    #  EXECUTION TIMES  (WBC-aligned, use iter_t)
    # ══════════════════════════════════════════════════════════════════════════

    exec_times = {k: v for k, v in {
        'EKF':    execution_time_ekf,
        'KF':     execution_time_kf,
        'MPC':    execution_time_mpc,
        'WBC':    execution_time_wbc,
        'Update': execution_time_update,
    }.items() if v is not None}

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
        ax.legend(frameon=True, fontsize=10)
        fig.tight_layout()
        os.makedirs('images/execution_times', exist_ok=True)
        fig.savefig(f'images/execution_times/{name}_execution_time_plot.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)