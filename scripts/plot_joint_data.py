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


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    expNumber = input("Enter 0 to plot data from the last simulation or the number of the experiment: ")
    if expNumber == '0':
        folder = '/tmp/robot_logs'
        expType = "Simulation"
    else:
        folder = 'experiments/experiment_' + expNumber + '/robot_logs'
        expType = "Experiment"

    wbc_only = input("Enter 1 to plot only from WBC activation onward, "
                     "or press Enter to plot all data: ").strip() == '1'

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
                'filtered_joint_velocity'):
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

    n_wbc = total
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
        s = max(0, rows - n_wbc)
        return d[s:]

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
    torso_orientation          = L('torso_orientation')
    torso_angular_velocity     = L('torso_angular_velocity')
    des_torso_orientation      = L('des_torso_orientation')
    des_torso_angular_velocity = L('des_torso_angular_velocity')
    pelvis_orientation          = L('pelvis_orientation')
    pelvis_angular_velocity     = L('pelvis_angular_velocity')
    des_pelvis_orientation      = L('des_pelvis_orientation')
    des_pelvis_angular_velocity = L('des_pelvis_angular_velocity')
    execution_time_ekf       = L('execution_time_ekf')
    execution_time_kf        = L('execution_time_kf')
    execution_time_mpc       = L('execution_time_mpc')
    execution_time_wbc       = L('execution_time_wbc')
    execution_time_update    = L('execution_time_update')

    # ── sensor / EKF time axis ────────────────────────────────────────────────
    # wbc_only: keep only the tail of the sensor log matching the WBC window.
    # otherwise: full sensor log, from tick 0.
    if _T_sensor is not None:
        _ae    = min(n_wbc, _T_sensor) if wbc_only else _T_sensor
        t_full = np.linspace(0.0, DT * _ae, _ae)

        def _full(nm):
            d = _sens_raw.get(nm)
            if d is None:
                return None
            return d[len(d) - _ae:] if wbc_only else d[:_ae]
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

    # Build left-right joint pairs, skip waist
    _jnames_stripped = [jn.strip() for jn in joint_names]
    _lr_pairs = []
    for li, lname in enumerate(_jnames_stripped):
        if not lname.startswith('left_') or 'waist' in lname:
            continue
        suffix = lname[len('left_'):]
        rname  = 'right_' + suffix
        if rname in _jnames_stripped:
            _lr_pairs.append((li, _jnames_stripped.index(rname), suffix))


    # ══════════════════════════════════════════════════════════════════════════
    #  WBC SOLUTIONS
    # ══════════════════════════════════════════════════════════════════════════

    _waist_idx = grouped_indices.get('waist', [])
    if input_torque is not None:
        for li, ri, suffix in _lr_pairs:
            plot_components(t,
                np.column_stack([input_torque[:, li], input_torque[:, ri]]),
                [f'Left {suffix}', f'Right {suffix}'],
                f'WBC Joint Torque – {suffix}', 'Torque [Nm]',
                f'images/wbc_solutions/torques/{suffix}_torque.png')
        if _waist_idx:
            plot_components(t, input_torque[:, _waist_idx],
                [_jnames_stripped[i] for i in _waist_idx],
                'WBC Joint Torque – waist', 'Torque [Nm]',
                'images/wbc_solutions/torques/waist_torque.png')

    if wbc_accelerations is not None:
        plot_components(t, wbc_accelerations[:, :3],
            [f'Linear acc {l}' for l in labels_xyz],
            'WBC Base Linear Acceleration', r'Acceleration [m/s²]',
            'images/wbc_solutions/accelerations/base_linear_acceleration.png')
        plot_components(t, wbc_accelerations[:, 3:6],
            [f'Angular acc {l}' for l in labels_xyz],
            'WBC Base Angular Acceleration', r'Acceleration [rad/s²]',
            'images/wbc_solutions/accelerations/base_angular_acceleration.png')
        for li, ri, suffix in _lr_pairs:
            plot_components(t,
                np.column_stack([wbc_accelerations[:, li + 6], wbc_accelerations[:, ri + 6]]),
                [f'Left {suffix}', f'Right {suffix}'],
                f'WBC Joint Acceleration – {suffix}', r'Acceleration [rad/s²]',
                f'images/wbc_solutions/accelerations/{suffix}_acceleration.png')
        if _waist_idx:
            plot_components(t, wbc_accelerations[:, [i + 6 for i in _waist_idx]],
                [_jnames_stripped[i] for i in _waist_idx],
                'WBC Joint Acceleration – waist', r'Acceleration [rad/s²]',
                'images/wbc_solutions/accelerations/waist_acceleration.png')

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
        'images/task_soles/forces/estimated_force_left_sole.png')
    plot_components(t, estimated_force_rsole,
        [f'Right sole force {l}' for l in labels_xyz],
        'Estimated Forces on Right Sole', 'Force [N]',
        'images/task_soles/forces/estimated_force_right_sole.png')

    # ══════════════════════════════════════════════════════════════════════════
    #  CoM AND ZMP
    # ══════════════════════════════════════════════════════════════════════════

    plot_components(t, des_com_acceleration,
        [fr'Des CoM Acc ${l}$' for l in labels_xyz],
        'Desired CoM Acceleration', r'Acceleration [$\mathrm{m/s^2}$]',
        'images/task_com/references/des_com_acceleration_plot.png')
    plot_components(t, des_com_position,
        [fr'Des CoM Pos ${l}$' for l in labels_xyz],
        'Desired CoM Position', r'Position [$\mathrm{m}$]',
        'images/task_com/references/des_com_position_plot.png')
    plot_components(t, des_com_velocity,
        [fr'Des CoM Vel ${l}$' for l in labels_xyz],
        'Desired CoM Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/task_com/references/des_com_velocity_plot.png')
    plot_components(t, des_zmp_position,
        [fr'Des ZMP Pos ${l}$' for l in labels_xyz],
        'Desired ZMP Position', r'Position [$\mathrm{m}$]',
        'images/task_com/references/des_zmp_position_plot.png')

    plot_components(t, _sub(des_zmp_position, kf_zmp_position),
        [fr'ZMP Error ${l}$' for l in labels_xyz],
        'ZMP Position Error', r'Position [$\mathrm{m}$]',
        'images/task_com/errors/error_zmp_position_plot.png')
    plot_components(t, _sub(des_com_position, kf_com_position),
        [fr'CoM Pos Error ${l}$' for l in labels_xyz],
        'CoM Position Error', r'Position [$\mathrm{m}$]',
        'images/task_com/errors/error_com_position_plot.png')
    plot_components(t, _sub(des_com_velocity, kf_com_velocity),
        [fr'CoM Vel Error ${l}$' for l in labels_xyz],
        'CoM Velocity Error', r'Velocity [$\mathrm{m/s}$]',
        'images/task_com/errors/error_com_velocity_plot.png')

    plot_comparison(t, kf_zmp_position, des_zmp_position,
        [fr'${l}$' for l in labels_xyz], 'ZMP Position', r'[$\mathrm{m}$]',
        'images/task_com/errors/comparison_zmp_position_plot.png')
    plot_comparison(t, kf_com_position, des_com_position,
        [fr'${l}$' for l in labels_xyz], 'CoM Position', r'[$\mathrm{m}$]',
        'images/task_com/errors/comparison_com_position_plot.png')
    plot_comparison(t, kf_com_velocity, des_com_velocity,
        [fr'${l}$' for l in labels_xyz], 'CoM Velocity', r'[$\mathrm{m/s}$]',
        'images/task_com/errors/comparison_com_velocity_plot.png')

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
            os.makedirs('images/task_com', exist_ok=True)
            fig.savefig(f'images/task_com/{fname}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    #  FEET
    # ══════════════════════════════════════════════════════════════════════════

    plot_components(t, p_lsole_des,
        [fr'Des L Sole ${l}$' for l in labels_xyz],
        'Desired Left Sole Position', r'Position [$\mathrm{m}$]',
        'images/task_soles/references/desired_left_sole_position_plot.png')
    plot_components(t, p_rsole_des,
        [fr'Des R Sole ${l}$' for l in labels_xyz],
        'Desired Right Sole Position', r'Position [$\mathrm{m}$]',
        'images/task_soles/references/desired_right_sole_position_plot.png')

    plot_components(t, _sub(p_lsole_des, p_lsole),
        [fr'L Sole Pos Error ${l}$' for l in labels_xyz],
        'Error – Left Sole Position', r'Position [$\mathrm{m}$]',
        'images/task_soles/errors/error_left_sole_position_plot.png')
    plot_components(t, _sub(p_rsole_des, p_rsole),
        [fr'R Sole Pos Error ${l}$' for l in labels_xyz],
        'Error – Right Sole Position', r'Position [$\mathrm{m}$]',
        'images/task_soles/errors/error_right_sole_position_plot.png')
    plot_components(t, _sub(v_lsole_des, v_lsole),
        [fr'L Sole Vel Error ${l}$' for l in labels_xyz],
        'Error – Left Sole Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/task_soles/errors/error_left_sole_velocity_plot.png')
    plot_components(t, _sub(v_rsole_des, v_rsole),
        [fr'R Sole Vel Error ${l}$' for l in labels_xyz],
        'Error – Right Sole Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/task_soles/errors/error_right_sole_velocity_plot.png')

    plot_comparison(t, p_lsole, p_lsole_des,
        [fr'${l}$' for l in labels_xyz], 'Left Sole Position', r'[$\mathrm{m}$]',
        'images/task_soles/errors/comparison_left_sole_position_plot.png')
    plot_comparison(t, p_rsole, p_rsole_des,
        [fr'${l}$' for l in labels_xyz], 'Right Sole Position', r'[$\mathrm{m}$]',
        'images/task_soles/errors/comparison_right_sole_position_plot.png')
    plot_comparison(t, v_lsole, v_lsole_des,
        [fr'${l}$' for l in labels_xyz], 'Left Sole Velocity', r'[$\mathrm{m/s}$]',
        'images/task_soles/errors/comparison_left_sole_velocity_plot.png')
    plot_comparison(t, v_rsole, v_rsole_des,
        [fr'${l}$' for l in labels_xyz], 'Right Sole Velocity', r'[$\mathrm{m/s}$]',
        'images/task_soles/errors/comparison_right_sole_velocity_plot.png')

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

    # ── joint velocity (EKF vs measured, left-right pairs) ───────────────────
    if measured_joint_velocity is not None and filtered_joint_velocity is not None:
        _nv = min(len(measured_joint_velocity), len(filtered_joint_velocity))
        for li, ri, suffix in _lr_pairs:
            plot_components(t_full[:_nv],
                np.column_stack([filtered_joint_velocity[:_nv, li], filtered_joint_velocity[:_nv, ri]]),
                [f'Left {suffix}', f'Right {suffix}'],
                f'EKF Joint Velocity – {suffix}', r'Velocity [$\mathrm{rad/s}$]',
                f'images/ekf/joints/velocities/{suffix}_velocity_plot.png')

            plot_components(t_full[:_nv],
                np.column_stack([
                    measured_joint_velocity[:_nv, li] - filtered_joint_velocity[:_nv, li],
                    measured_joint_velocity[:_nv, ri] - filtered_joint_velocity[:_nv, ri],
                ]),
                [f'Err Left {suffix}', f'Err Right {suffix}'],
                f'EKF Velocity Error – {suffix}', r'Velocity [$\mathrm{rad/s}$]',
                f'images/ekf/joints/error/velocities/error_{suffix}_velocity_plot.png')

        if _waist_idx:
            plot_components(t_full[:_nv], filtered_joint_velocity[:_nv, _waist_idx],
                [_jnames_stripped[i] for i in _waist_idx],
                'EKF Joint Velocity – waist', r'Velocity [$\mathrm{rad/s}$]',
                'images/ekf/joints/velocities/waist_velocity_plot.png')
            plot_components(t_full[:_nv],
                _sub(measured_joint_velocity[:_nv, _waist_idx], filtered_joint_velocity[:_nv, _waist_idx]),
                [_jnames_stripped[i] for i in _waist_idx],
                'EKF Velocity Error – waist', r'Velocity [$\mathrm{rad/s}$]',
                'images/ekf/joints/error/velocities/error_waist_velocity_plot.png')

        colormap    = plt.colormaps['tab10']
        line_styles = ['-', '--', '-.', ':']
        _nj = min(len(measured_joint_velocity), len(filtered_joint_velocity))
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in range(num_joints):
            ax.plot(t_full[:_nj], filtered_joint_velocity[:_nj, i] - measured_joint_velocity[:_nj, i],
                    label=joint_names[i].strip(),
                    color=colormap(i % 10),
                    linestyle=line_styles[(i // 10) % 4],
                    linewidth=2)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Velocity [$\mathrm{rad/s}$]', fontsize=11)
        ax.set_title('Error EKF vs Measured – Joint Velocity', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', frameon=True, fontsize=4)
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        os.makedirs('images/ekf/joints/error', exist_ok=True)
        fig.savefig('images/ekf/joints/error/error_joint_velocity_plot.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    #  TASK ORIENTATION  (torso + pelvis, WBC signals, use t)
    # ══════════════════════════════════════════════════════════════════════════

    plot_comparison(t, torso_orientation, des_torso_orientation,
        ['Roll', 'Pitch', 'Yaw'], 'Torso Orientation', r'[$\mathrm{rad}$]',
        'images/task_orientation/torso/torso_orientation_comparison_plot.png')
    plot_components(t, _sub(torso_orientation, des_torso_orientation),
        ['Roll Error', 'Pitch Error', 'Yaw Error'],
        'Torso Orientation Error', r'Orientation [$\mathrm{rad}$]',
        'images/task_orientation/torso/torso_orientation_error_plot.png')
    plot_components(t, torso_angular_velocity,
        [f'AngVel {l}' for l in labels_xyz],
        'Torso Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/task_orientation/torso/torso_angular_velocity_plot.png')
    plot_components(t, _sub(torso_angular_velocity, des_torso_angular_velocity),
        [f'AngVel Error {l}' for l in labels_xyz],
        'Torso Angular Velocity Error', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/task_orientation/torso/torso_angular_velocity_error_plot.png')

    plot_comparison(t, pelvis_orientation, des_pelvis_orientation,
        ['Roll', 'Pitch', 'Yaw'], 'Pelvis Orientation', r'[$\mathrm{rad}$]',
        'images/task_orientation/pelvis/pelvis_orientation_comparison_plot.png')
    plot_components(t, _sub(pelvis_orientation, des_pelvis_orientation),
        ['Roll Error', 'Pitch Error', 'Yaw Error'],
        'Pelvis Orientation Error', r'Orientation [$\mathrm{rad}$]',
        'images/task_orientation/pelvis/pelvis_orientation_error_plot.png')
    plot_components(t, pelvis_angular_velocity,
        [f'AngVel {l}' for l in labels_xyz],
        'Pelvis Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/task_orientation/pelvis/pelvis_angular_velocity_plot.png')
    plot_components(t, _sub(pelvis_angular_velocity, des_pelvis_angular_velocity),
        [f'AngVel Error {l}' for l in labels_xyz],
        'Pelvis Angular Velocity Error', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/task_orientation/pelvis/pelvis_angular_velocity_error_plot.png')

    # ── sole roll, left vs right (diagnostic for ankle-roll asymmetry) ────────
    # lsole/rsole_orientation columns are (roll, pitch, yaw) via eulerAngles(0,1,2).
    if lsole_orientation is not None and rsole_orientation is not None:
        plot_components(t,
            np.column_stack([lsole_orientation[:, 0], rsole_orientation[:, 0]]),
            ['Left', 'Right'], 'Sole Roll – Measured (Left vs Right)', r'Roll [$\mathrm{rad}$]',
            'images/task_orientation/soles/roll_measured_left_vs_right_plot.png')
    if des_lsole_orientation is not None and des_rsole_orientation is not None:
        plot_components(t,
            np.column_stack([des_lsole_orientation[:, 0], des_rsole_orientation[:, 0]]),
            ['Left', 'Right'], 'Sole Roll – Desired (Left vs Right)', r'Roll [$\mathrm{rad}$]',
            'images/task_orientation/soles/roll_desired_left_vs_right_plot.png')
    if all(x is not None for x in (lsole_orientation, rsole_orientation,
                                    des_lsole_orientation, des_rsole_orientation)):
        roll_err_l = _sub(lsole_orientation[:, 0:1], des_lsole_orientation[:, 0:1])
        roll_err_r = _sub(rsole_orientation[:, 0:1], des_rsole_orientation[:, 0:1])
        _n_roll = min(len(roll_err_l), len(roll_err_r))
        plot_components(t,
            np.column_stack([roll_err_l[:_n_roll], roll_err_r[:_n_roll]]),
            ['Left', 'Right'], 'Sole Roll Error – Measured vs Desired (Left vs Right)',
            r'Roll Error [$\mathrm{rad}$]',
            'images/task_orientation/soles/roll_error_left_vs_right_plot.png')

    # ══════════════════════════════════════════════════════════════════════════
    #  FEEDBACK  (full sensor data – from tick 0, use t_full)
    # ══════════════════════════════════════════════════════════════════════════

    plot_components(t_full, odometry_base_position,
        [fr'Odometry Pos ${l}$' for l in labels_xyz],
        'Odometry Base Position', r'Position [$\mathrm{m}$]',
        'images/feedback/odometry/odometry_base_position_plot.png')
    plot_components(t_full, odometry_base_velocity,
        [fr'Odometry Vel ${l}$' for l in labels_xyz],
        'Odometry Base Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/feedback/odometry/odometry_base_velocity_plot.png')
    plot_components(t_full, odometry_imu_orientation,
        [fr'Odometry IMU Orient ${l}$' for l in labels_quat],
        'Odometry IMU Orientation Quat', r'Orientation [quat]',
        'images/feedback/odometry/odometry_imu_orientation_quat_plot.png')
    plot_components(t_full, odometry_imu_orientation_rpy,
        [fr'Odometry IMU Orient ${l}$' for l in labels_rpy],
        'Odometry IMU Orientation RPY', r'Orientation [$\mathrm{rad}$]',
        'images/feedback/odometry/odometry_imu_orientation_rpy_plot.png')

    plot_components(t_full, measured_imu_pelvis_angular_velocity,
        [fr'Pelvis AngVel ${l}$' for l in labels_xyz],
        'Pelvis IMU – Gyroscope', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/feedback/imu_pelvis/pelvis_imu_angular_velocity_plot.png')
    plot_components(t_full, measured_imu_pelvis_accelerometer,
        [fr'Pelvis Acc ${l}$' for l in labels_xyz],
        'Pelvis IMU – Accelerometer', r'Acceleration [$\mathrm{m/s^2}$]',
        'images/feedback/imu_pelvis/pelvis_imu_acceleration_plot.png')
    plot_components(t_full, measured_imu_pelvis_rpy,
        [fr'Pelvis RPY ${l}$' for l in labels_rpy],
        'Pelvis IMU – RPY', r'RPY [$\mathrm{rad}$]',
        'images/feedback/imu_pelvis/pelvis_imu_rpy_plot.png')
    plot_components(t_full, measured_imu_pelvis_quaternion,
        [fr'Pelvis Quat ${l}$' for l in labels_quat],
        'Pelvis IMU – Quaternion', r'Quaternion',
        'images/feedback/imu_pelvis/pelvis_imu_quaternion_plot.png')

    plot_components(t_full, measured_imu_torso_rpy,
        [fr'Torso RPY ${l}$' for l in labels_rpy],
        'Torso IMU – RPY', r'RPY [$\mathrm{rad}$]',
        'images/feedback/imu_torso/torso_imu_rpy_plot.png')
    plot_components(t_full, measured_imu_torso_quaternion,
        [fr'Torso Quat ${l}$' for l in labels_quat],
        'Torso IMU – Quaternion', r'Quaternion',
        'images/feedback/imu_torso/torso_imu_quaternion_plot.png')
    plot_components(t_full, measured_imu_torso_accelerometer,
        [fr'Torso Acc ${l}$' for l in labels_xyz],
        'Torso IMU – Accelerometer', r'Acceleration [$\mathrm{m/s^2}$]',
        'images/feedback/imu_torso/torso_imu_acceleration_plot.png')
    plot_components(t_full, measured_imu_torso_angular_velocity,
        [fr'Torso AngVel ${l}$' for l in labels_xyz],
        'Torso IMU – Gyroscope', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/feedback/imu_torso/torso_imu_angular_velocity_plot.png')

    for li, ri, suffix in _lr_pairs:
        if measured_joint_position is not None:
            plot_components(t_full,
                np.column_stack([measured_joint_position[:, li], measured_joint_position[:, ri]]),
                [f'Left {suffix}', f'Right {suffix}'],
                f'Joint Position – {suffix}', r'Position [$\mathrm{rad}$]',
                f'images/feedback/joints/positions/{suffix}_position_plot.png')
        if measured_joint_velocity is not None:
            plot_components(t_full,
                np.column_stack([measured_joint_velocity[:, li], measured_joint_velocity[:, ri]]),
                [f'Left {suffix}', f'Right {suffix}'],
                f'Joint Velocity – {suffix}', r'Velocity [$\mathrm{rad/s}$]',
                f'images/feedback/joints/velocities/{suffix}_velocity_plot.png')
    if _waist_idx:
        if measured_joint_position is not None:
            plot_components(t_full, measured_joint_position[:, _waist_idx],
                [_jnames_stripped[i] for i in _waist_idx],
                'Joint Position – waist', r'Position [$\mathrm{rad}$]',
                'images/feedback/joints/positions/waist_position_plot.png')
        if measured_joint_velocity is not None:
            plot_components(t_full, measured_joint_velocity[:, _waist_idx],
                [_jnames_stripped[i] for i in _waist_idx],
                'Joint Velocity – waist', r'Velocity [$\mathrm{rad/s}$]',
                'images/feedback/joints/velocities/waist_velocity_plot.png')

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