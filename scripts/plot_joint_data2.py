import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform
from math import ceil, floor, sqrt
from collections import defaultdict
import matplotlib.axes
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
import atexit
import glob
import os
import io
import warnings
import imageio.v2 as imageio
# import cv2

# ---------------------------------------------------------------------------
# Partial-log handling
# ---------------------------------------------------------------------------
# A run whose controller-side logging stopped early (or that predates a given
# channel) leaves some logs missing or much shorter than the rest — the usual
# situation on the real robot, where the sensor logger keeps writing after the
# controller-side channels have stopped. Every channel is therefore stretched
# onto a single time axis, as long as the longest log of the run: a shorter
# channel keeps its own samples and is padded with NaN, a missing one becomes
# all-NaN. NaN simply leaves a gap in the plot, so every figure is still drawn,
# showing whatever the run holds on a time axis shared with all other figures.
# The names of the incomplete channels travel with the arrays (see _Partial) so
# they can be reported at exit, and only a figure that ends up with no finite
# value at all is skipped rather than written empty.


def _collect_sources(values):
    """Union of the partial-channel names reachable from `values`."""
    found = frozenset()
    for v in values:
        if isinstance(v, (list, tuple)):
            found |= _collect_sources(v)
        else:
            found |= getattr(v, 'sources', frozenset())
    return found


class _Partial(np.ndarray):
    """A log channel that is missing or shorter than the common time axis."""

    def __new__(cls, data, sources):
        obj = np.asarray(data, dtype=float).view(cls)
        obj.sources = frozenset(sources)
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.sources = getattr(obj, 'sources', frozenset())

    # Ufuncs (arithmetic, matmul, ...) already keep the subclass and carry
    # `sources` over through __array_finalize__. np.stack & friends drop it,
    # so re-apply the mark to whatever they return.
    def __array_function__(self, func, types, args, kwargs):
        out = super().__array_function__(func, types, args, kwargs)
        if isinstance(out, np.ndarray):
            out = out.view(_Partial)
            out.sources = _collect_sources(list(args) + list(kwargs.values()))
        return out


def _figure_sources(fig):
    return getattr(fig, '_partial_sources', frozenset())


class _TimeAxis(np.ndarray):
    """The common time vector, so that plots against it share their x limits."""


# Longest time span drawn by the script; every time axis is stretched to it.
_time_span = [0.0]


def _time(values):
    """Tag a time vector as the common axis (see _span_time_axes)."""
    arr = np.asarray(values, dtype=float).view(_TimeAxis)
    if arr.size:
        _time_span[0] = max(_time_span[0], float(arr[-1]))
    return arr


def _is_reference_line(line):
    """True for an axhline/axvline: decoration in axis coordinates, not data."""
    x = np.asarray(line.get_xdata(), dtype=float)
    y = np.asarray(line.get_ydata(), dtype=float)
    return x.size == 2 and (np.array_equal(x, [0.0, 1.0])
                            or np.array_equal(y, [0.0, 1.0]))


def _figure_time_extent(fig):
    """Last instant at which `fig` still holds data, or None if it holds none.

    Every channel is NaN-padded up to the longest log of the run, so a figure
    drawn from a channel that stopped early carries a long tail of NaN. The
    x limit is taken from the samples that are actually there, so such a figure
    is stretched over its own data instead of over the whole run.
    """
    last = None
    for ax in fig.axes:
        if not getattr(ax, '_is_time_axis', False):
            continue
        for line in ax.lines:
            if _is_reference_line(line):
                continue
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            if x.shape != y.shape:
                continue
            finite = x[np.isfinite(x) & np.isfinite(y)]
            if finite.size:
                last = finite.max() if last is None else max(last, finite.max())
    return last


def _span_time_axes(fig):
    """Give every time-based Axes of `fig` the same, data-driven time span.

    The subplots of one figure share their x limits, so they can be read
    against each other, but the figure as a whole ends where its own data
    ends rather than at the end of the longest channel of the run: a figure
    holding few samples would otherwise be squeezed into a sliver of an axis
    spanning the whole run. A figure with no finite sample at all (skipped
    anyway) falls back to the run's full span.
    """
    last = _figure_time_extent(fig)
    if last is None or last <= 0.0:
        last = _time_span[0]
    for ax in fig.axes:
        if getattr(ax, '_is_time_axis', False):
            ax.set_xlim(0.0, last)


def _track_partial(method_name):
    """Wrap an Axes plotting method so it marks figures fed partial data."""
    original = getattr(matplotlib.axes.Axes, method_name)

    def wrapper(self, *args, **kwargs):
        values = list(args) + list(kwargs.values())
        sources = _collect_sources(values)
        if sources:
            fig = self.figure
            fig._partial_sources = _figure_sources(fig) | sources
        if any(isinstance(v, _TimeAxis) for v in values):
            self._is_time_axis = True
        return original(self, *args, **kwargs)

    return wrapper


for _method in ('plot', 'bar', 'axhline'):
    setattr(matplotlib.axes.Axes, _method, _track_partial(_method))


def _figure_has_data(fig):
    """True when at least one artist of `fig` carries a finite value.

    A figure whose channels are all missing holds nothing but NaN, and is not
    worth writing; one drawn from a channel that merely stopped early holds its
    samples up to that point and is written as usual.
    """
    for ax in fig.axes:
        for line in ax.lines:
            if _is_reference_line(line):
                continue
            y = np.asarray(line.get_ydata(), dtype=float)
            if y.size and np.isfinite(y).any():
                return True
        for patch in ax.patches:
            height = getattr(patch, 'get_height', None)
            if height is None or np.isfinite(height()):
                return True
        if ax.collections or ax.images:
            return True
    return False


_skipped_figures = []

# Mirror every PNG plot saved via fig.savefig("images/...") as a vector PDF
# under SIM_PLOTS_DIR, preserving the same sub-directory layout as
# scripts/images/. This keeps a high-quality, vectorized copy of every plot
# ready to be included in the thesis without touching each savefig() call.
SIM_PLOTS_DIR = os.path.expanduser('~/thesis/material/images/sim_plots')
_original_savefig = plt.Figure.savefig


def _savefig_and_mirror_pdf(self, fname, *args, **kwargs):
    _span_time_axes(self)
    if isinstance(fname, str) and not _figure_has_data(self):
        _skipped_figures.append((fname, _figure_sources(self)))
        return
    _original_savefig(self, fname, *args, **kwargs)
    if isinstance(fname, str) and fname.startswith('images/') and fname.endswith('.png'):
        pdf_path = os.path.join(SIM_PLOTS_DIR, fname[len('images/'):-len('.png')] + '.pdf')
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        _original_savefig(self, pdf_path, *args, **kwargs)


plt.Figure.savefig = _savefig_and_mirror_pdf


@atexit.register
def _report_skipped_figures():
    if not _skipped_figures:
        return
    print(f"\n[SKIP] {len(_skipped_figures)} figure(s) not saved: every curve they hold "
          f"is blank, so the plot would show no data at all.")
    blank = sorted({c for _, sources in _skipped_figures for c in sources
                    if c.endswith('(not found)')})
    if blank:
        print("       Channels this run never logged:")
        for channel in blank:
            print(f"         - {channel}")


if __name__ == '__main__':
    #request input from terminal
    expNumber = input("Enter 0 to plot data from the last simulation or the number of the experiment: ")
    if expNumber == '0':
        folder = '/tmp/robot_logs'
        expType = "Simulation"
    else:
        folder = 'experiments/experiment_' + expNumber
        expType = "Experiment"

    endPlot = input("Enter the time (in seconds) at which you want to end the plots (or press Enter to plot all data): ")
    if endPlot != '':
        endPlot = int(float(endPlot) * 500)  # Assuming a control frequency of 500 Hz
    else:
        endPlot = 10

    if os.path.isdir(folder + '/robot_logs'):
        folder = folder + '/robot_logs'

    joint_names = open(folder + '/joint_names.txt').readlines()

    # Joint position limits [rad] (lower, upper), from
    # robot/g1/g1_description/g1_29dof_with_hand_rev_1_0.urdf (the URDF loaded
    # by WalkingManager.cpp).
    joint_limits = {
        'left_hip_pitch_joint': (-2.5307, 2.8798),
        'left_hip_roll_joint': (-0.5236, 2.9671),
        'left_hip_yaw_joint': (-2.7576, 2.7576),
        'left_knee_joint': (-0.087267, 2.8798),
        'left_ankle_pitch_joint': (-0.87267, 0.5236),
        'left_ankle_roll_joint': (-0.2618, 0.2618),
        'right_hip_pitch_joint': (-2.5307, 2.8798),
        'right_hip_roll_joint': (-2.9671, 0.5236),
        'right_hip_yaw_joint': (-2.7576, 2.7576),
        'right_knee_joint': (-0.087267, 2.8798),
        'right_ankle_pitch_joint': (-0.87267, 0.5236),
        'right_ankle_roll_joint': (-0.2618, 0.2618),
        'waist_yaw_joint': (-2.618, 2.618),
        'waist_roll_joint': (-0.52, 0.52),
        'waist_pitch_joint': (-0.52, 0.52),
        'left_shoulder_pitch_joint': (-3.0892, 2.6704),
        'left_shoulder_roll_joint': (-1.5882, 2.2515),
        'left_shoulder_yaw_joint': (-2.618, 2.618),
        'left_elbow_joint': (-1.0472, 2.0944),
        'left_wrist_roll_joint': (-1.972222054, 1.972222054),
        'left_wrist_pitch_joint': (-1.614429558, 1.614429558),
        'left_wrist_yaw_joint': (-1.614429558, 1.614429558),
        'right_shoulder_pitch_joint': (-3.0892, 2.6704),
        'right_shoulder_roll_joint': (-2.2515, 1.5882),
        'right_shoulder_yaw_joint': (-2.618, 2.618),
        'right_elbow_joint': (-1.0472, 2.0944),
        'right_wrist_roll_joint': (-1.972222054, 1.972222054),
        'right_wrist_pitch_joint': (-1.614429558, 1.614429558),
        'right_wrist_yaw_joint': (-1.614429558, 1.614429558),
    }

    startPlot = 0

    def _count_rows(path):
        with open(path) as fh:
            return sum(1 for line in fh if line.strip())

    # The time axis of every figure spans the longest channel of the run, so
    # that all plots share it whatever each individual log holds. On the real
    # robot the controller-side channels (com_position and friends) often stop
    # well before the sensor logs, or are missing altogether; they are padded
    # up to this length instead of shrinking the axis of every other plot.
    lengths = {
        os.path.basename(p): _count_rows(p)
        for p in glob.glob(folder + '/*.txt')
        if os.path.basename(p) != 'joint_names.txt'
    }
    longest = max(lengths.values(), default=0)
    if longest - endPlot <= startPlot:
        raise SystemExit(
            f"[ERROR] no channel in {folder} holds more than "
            f"{endPlot + startPlot} sample(s) — nothing to plot."
        )
    num_samples = longest - endPlot
    longest_name = max(lengths, key=lengths.get)
    com_rows = lengths.get('com_position.txt', 0)
    if com_rows < longest:
        print(f"[WARN] com_position.txt holds {com_rows} sample(s) against the "
              f"{longest} of {longest_name}: the controller-side logs of this run\n"
              f"       stopped early (or are missing). All plots span the longest "
              f"channel and the shorter ones stop where their data ends.\n"
              f"       The time axis still assumes 500 Hz, which may not match the "
              f"logging rate of every channel.")
    sl = slice(startPlot, num_samples)

    def _fit(data, label):
        """Trim or NaN-pad a loaded channel onto the common time axis."""
        n_have = data.shape[0]
        if n_have >= num_samples:
            return data[sl]
        reason = f"{label} ({n_have} of {num_samples} samples)"
        print(f"[INFO] {reason} — plotted up to its last sample, then blank.")
        pad = np.full((num_samples - n_have,) + data.shape[1:], np.nan)
        return _Partial(np.concatenate([np.asarray(data, dtype=float), pad])[sl],
                        [reason])

    def _blank(label, shape):
        """All-NaN stand-in for a channel this run never logged."""
        reason = f"{label} (not found)"
        print(f"[INFO] {reason} — its curves are left blank.")
        return _Partial(np.full(shape, np.nan), [reason])

    def _load(fname, ncols):
        p = folder + '/' + fname
        # ndmin=2 keeps a single-sample log as (1, ncols) rather than
        # collapsing it to 1-D, so short channels are padded correctly.
        if os.path.exists(p):
            data = np.loadtxt(p, ndmin=2)
            if data.size:
                return _fit(data, fname)
        return _blank(fname, (num_samples - startPlot, ncols))

    def _load1d(fname):
        p = folder + '/' + fname
        if os.path.exists(p):
            data = np.loadtxt(p, ndmin=1)
            if data.size:
                return _fit(data, fname)
        return _blank(fname, (num_samples - startPlot,))

    def _load_own(path):
        """np.loadtxt for the sections below, which set their own time axis.

        The channel is fitted to the common axis too, so those sections end up
        with the same number of samples — and hence the same time span — as
        every other plot of the run.
        """
        return _fit(np.loadtxt(path, ndmin=2), os.path.basename(path))

    def _note_saved(message, *arrays):
        """Report a section's figures, flagging those drawn from a short log."""
        missing = _collect_sources(arrays)
        if missing:
            message += " — partial: " + ', '.join(sorted(missing))
        print(message)

    def _nan_mean(values, axis=0):
        """Mean over the samples both channels actually hold (NaN if none)."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(values, axis=axis)

    fb_com_position = _load('com_position.txt', 3)
    fb_com_velocity = _load('com_velocity.txt', 3)
    fb_zmp_position = _load('zmp_position.txt', 3)
    kf_com_position = _load('kf_com_position.txt', 3)
    kf_com_velocity = _load('kf_com_velocity.txt', 3)
    kf_zmp_position = _load('kf_zmp_position.txt', 3)
    des_com_position = _load('des_com_position.txt', 3)
    des_com_velocity = _load('des_com_velocity.txt', 3)
    des_zmp_position = _load('des_zmp_position.txt', 3)
    des_com_acceleration = _load('des_com_acceleration.txt', 3)
    current_disturbance = _load('current_disturbance.txt', 3)
    angular_momentum = _load('angular_momentum.txt', 3)
    angular_momentum_rate = _load('angular_momentum_rate.txt', 3)

    input_torque = _load('input_torque.txt', 29)
    motor_torque_filt = _load('motor_torque_filt.txt', 29)  # real-robot only (EMA-filtered motor torques)
    q_dot_des = _load('q_dot_des.txt', 29)
    q_des = _load('q_des.txt', 29)

    base_position_des = _load('base_position_des.txt', 3)
    base_orientation_des = _load('base_orientation_des.txt', 4)  # (x, y, z, w)
    base_linear_velocity_des = _load('base_linear_velocity_des.txt', 3)
    base_angular_velocity_des = _load('base_angular_velocity_des.txt', 3)

    ef_zmp_position = _load('ef_zmp_position.txt', 3)

    p_lsole_fb = _load('p_lsole.txt', 3)
    p_rsole_fb = _load('p_rsole.txt', 3)
    v_lsole_fb = _load('v_lsole.txt', 3)
    v_rsole_fb = _load('v_rsole.txt', 3)
    p_lsole_des = _load('p_lsole_des.txt', 3)
    p_rsole_des = _load('p_rsole_des.txt', 3)
    v_lsole_des = _load('v_lsole_des.txt', 3)
    v_rsole_des = _load('v_rsole_des.txt', 3)

    fb_lsole_orientation = _load('lsole_orientation.txt', 3)
    fb_rsole_orientation = _load('rsole_orientation.txt', 3)
    des_lsole_orientation = _load('des_lsole_orientation.txt', 3)
    des_rsole_orientation = _load('des_rsole_orientation.txt', 3)

    estimated_force_lsole = _load('estimated_force_lsole.txt', 3)
    estimated_force_rsole = _load('estimated_force_rsole.txt', 3)
    estimated_moment_lsole = _load('estimated_moment_lsole.txt', 3)
    estimated_moment_rsole = _load('estimated_moment_rsole.txt', 3)
    wbc_accelerations = _load('wbc_accelerations.txt', 35)
    wbc_force_lsole = _load('wbc_force_lsole.txt', 6)
    wbc_force_rsole = _load('wbc_force_rsole.txt', 6)
    wbc_corner_forces_left = _load('wbc_corner_forces_left.txt', 12)
    wbc_corner_forces_right = _load('wbc_corner_forces_right.txt', 12)
    wbc_friction_coefficient = _load1d('wbc_friction_coefficient.txt')
    # Friction cone ratios (|fx|/fz, |fy|/fz) computed and logged by the WBC itself,
    # one scalar channel per foot/axis/corner (fl, fr, bl, br).
    friction_cone_ratio_left_x = np.stack([
        _load1d(f'friction_cone_ratio_left_x_{c}.txt') for c in ('fl', 'fr', 'bl', 'br')
    ], axis=1)
    friction_cone_ratio_left_y = np.stack([
        _load1d(f'friction_cone_ratio_left_y_{c}.txt') for c in ('fl', 'fr', 'bl', 'br')
    ], axis=1)
    friction_cone_ratio_right_x = np.stack([
        _load1d(f'friction_cone_ratio_right_x_{c}.txt') for c in ('fl', 'fr', 'bl', 'br')
    ], axis=1)
    friction_cone_ratio_right_y = np.stack([
        _load1d(f'friction_cone_ratio_right_y_{c}.txt') for c in ('fl', 'fr', 'bl', 'br')
    ], axis=1)
    wbc_friction_coefficient = _load1d('wbc_friction_coefficient.txt')

    # C++ side (main.cpp / main_g1.cpp) only logs a single base-frame EKF
    # estimate ("filtered_base_*"), not a separate IMU-frame one — reuse it
    # for the ekf_imu_* channels too.
    ekf_base_position = _load('filtered_base_position.txt', 3)
    ekf_base_velocity = _load('filtered_base_velocity.txt', 3)
    ekf_base_orientation = _load('filtered_base_quat.txt', 4)
    ekf_base_orientation_rpy = _load('filtered_base_rpy.txt', 3)
    ekf_base_angular_velocity = _load('filtered_base_ang_vel.txt', 3)
    ekf_imu_orientation = _load('filtered_base_quat.txt', 4)
    ekf_imu_orientation_rpy = _load('filtered_base_rpy.txt', 3)
    ekf_imu_angular_velocity = _load('filtered_base_ang_vel.txt', 3)
    ekf_joint_position = _load('filtered_joint_position.txt', 29)
    ekf_joint_velocity = _load('filtered_joint_velocity.txt', 29)

    torso_orientation = _load('torso_orientation.txt', 3)
    torso_angular_velocity = _load('torso_angular_velocity.txt', 3)
    des_torso_orientation = _load('des_torso_orientation.txt', 3)
    des_torso_angular_velocity = _load('des_torso_angular_velocity.txt', 3)

    execution_time_ekf = _load1d('execution_time_ekf.txt')
    execution_time_kf = _load1d('execution_time_kf.txt')
    execution_time_mpc = _load1d('execution_time_mpc.txt')
    execution_time_wbc = _load1d('execution_time_wbc.txt')
    execution_time_res_obs = _load1d('execution_time_res_obs.txt')
    execution_time_hac = _load1d('execution_time_hac.txt')
    execution_time_coop_planner = _load1d('execution_time_coop_planner.txt')
    execution_time_update = _load1d('execution_time_update.txt')

    odometry_base_position = _load('odom_pos.txt', 3)
    odometry_base_velocity = _load('odom_vel.txt', 3)
    odometry_imu_orientation = _load('odom_quat.txt', 4)
    odometry_imu_orientation_rpy = _load('odom_rpy.txt', 3)
    measured_joint_position = _load('joint_pos.txt', 29)
    measured_joint_velocity = _load('joint_vel.txt', 29)
    # Not currently logged anywhere in the C++ side — stays zero until a
    # "measured_joint_torque" channel is added to sensor_logger.
    measured_joint_torque = _load('measured_joint_torque.txt', 29)
    # Only main_g1.cpp logs pelvis_quat/pelvis_rpy (main.cpp doesn't) — will
    # still read as zero when plotting logs produced by main.cpp.
    measured_imu_orientation = _load('pelvis_quat.txt', 4)
    measured_imu_orientation_rpy = _load('pelvis_rpy.txt', 3)
    measured_imu_angular_velocity = _load('pelvis_gyro.txt', 3)
    measured_imu_accelerometer = _load('pelvis_acc.txt', 3)
        
    # rotate relative positions depending on the actual yaw angle
    yaw = odometry_imu_orientation_rpy[0, 2]
    if not np.isfinite(yaw):
        # A run without odometry logs leaves the yaw NaN, which would turn
        # every channel rotated below into NaN as well: keep them unrotated.
        print("[INFO] initial yaw unavailable — relative positions left unrotated.")
        yaw = 0.0
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    for i in range(num_samples - startPlot):
        p_lsole_fb[i, :] = rotation_matrix.T @ p_lsole_fb[i, :]
        p_rsole_fb[i, :] = rotation_matrix.T @ p_rsole_fb[i, :]
        p_lsole_des[i, :] = rotation_matrix.T @ p_lsole_des[i, :]
        p_rsole_des[i, :] = rotation_matrix.T @ p_rsole_des[i, :]
        kf_com_position[i, :] = rotation_matrix.T @ kf_com_position[i, :]
        kf_zmp_position[i, :] = rotation_matrix.T @ kf_zmp_position[i, :]
        des_com_position[i, :] = rotation_matrix.T @ des_com_position[i, :]
        des_zmp_position[i, :] = rotation_matrix.T @ des_zmp_position[i, :]


    reference_positions = np.array([
        -0.44,  # l_hip_p
        0.04,  # l_hip_r
        0.0,  # l_hip_y
        0.95,  # l_knee
        -0.50,  # l_ankle_p
        0.00,  # l_ankle_r
        -0.44,  # r_hip_p
        -0.04,  # r_hip_r
        0.0,  # r_hip_y
        0.95,  # r_knee
        -0.50,  # r_ankle_p
        0.00,  # r_ankle_r
        0.0,  # waist_y
        0.07,  # l_shoulder_p
        0.25,  # l_shoulder_r
        0.0,  # l_shoulder_y
        3.14 / 2.0 - 0.44,   # l_elbow_p
        0.0, # wrist_roll
        0.0, # wrist_pitch
        0.0, # wrist_yaw
        0.07,  # r_shoulder_p
        -0.25,  # r_shoulder_r
        0.0,  # r_shoulder_y
        3.14 / 2.0 - 0.44,  # r_elbow_p
        0.0, # wrist_roll
        0.0, # wrist_pitch
        0.0 # wrist_yaw
    ])

    delta = 1 / 500  # Assuming a control frequency of 500 Hz
    t = _time(np.linspace(0.0, delta * (num_samples - startPlot), num_samples - startPlot))
    # num_joints = 27
    num_joints = 29


    if not os.path.exists('images/feedback/joints/positions'):
        os.makedirs('images/feedback/joints/positions')
    if not os.path.exists('images/feedback/joints/velocities'):
        os.makedirs('images/feedback/joints/velocities')
    if not os.path.exists('images/feedback/base'):
        os.makedirs('images/feedback/base')
    if not os.path.exists('images/ekf/joints/positions'):
        os.makedirs('images/ekf/joints/positions')
    if not os.path.exists('images/ekf/joints/velocities'):
        os.makedirs('images/ekf/joints/velocities')
    if not os.path.exists('images/ekf/joints/error/positions'):
        os.makedirs('images/ekf/joints/error/positions')
    if not os.path.exists('images/ekf/joints/error/velocities'):
        os.makedirs('images/ekf/joints/error/velocities')
    if not os.path.exists('images/ekf/base/errors'):
        os.makedirs('images/ekf/base/errors')
    if not os.path.exists('images/ekf/performance'):
        os.makedirs('images/ekf/performance')
    if not os.path.exists('images/execution_times'):
        os.makedirs('images/execution_times')
    if not os.path.exists('images/com/references'):
        os.makedirs('images/com/references')
    if not os.path.exists('images/com/errors'):
        os.makedirs('images/com/errors')
    if not os.path.exists('images/wbc_solutions/wbc_joint_accelerations'):
        os.makedirs('images/wbc_solutions/wbc_joint_accelerations')
    if not os.path.exists('images/wbc_solutions/wbc_base_accelerations'):
        os.makedirs('images/wbc_solutions/wbc_base_accelerations')
    if not os.path.exists('images/wbc_solutions/wbc_joint_torques_ffw'):
        os.makedirs('images/wbc_solutions/wbc_joint_torques_ffw')
    if not os.path.exists('images/wbc_solutions/wbc_sole_forces'):
        os.makedirs('images/wbc_solutions/wbc_sole_forces')
    if not os.path.exists('images/wbc_solutions/friction_cone'):
        os.makedirs('images/wbc_solutions/friction_cone')
    if not os.path.exists('images/wbc_solutions/online_references/joint_positions'):
        os.makedirs('images/wbc_solutions/online_references/joint_positions')
    if not os.path.exists('images/wbc_solutions/online_references/joint_velocities'):
        os.makedirs('images/wbc_solutions/online_references/joint_velocities')
    if not os.path.exists('images/wbc_solutions/online_references/floating_base_positions'):
        os.makedirs('images/wbc_solutions/online_references/floating_base_positions')
    if not os.path.exists('images/wbc_solutions/online_references/floating_base_velocities'):
        os.makedirs('images/wbc_solutions/online_references/floating_base_velocities')
    if not os.path.exists('images/wrench_estimations/sole_wrenches'):
        os.makedirs('images/wrench_estimations/sole_wrenches')
    if not os.path.exists('images/soles/references'):
        os.makedirs('images/soles/references')
    if not os.path.exists('images/soles/errors'):
        os.makedirs('images/soles/errors')
    if not os.path.exists('images/mpc'):
        os.makedirs('images/mpc')
    if not os.path.exists('images/feedback/motor_torques'):
        os.makedirs('images/feedback/motor_torques')
    if not os.path.exists('images/residuals/right_arm'):
        os.makedirs('images/residuals/right_arm')
    if not os.path.exists('images/residuals/left_arm'):
        os.makedirs('images/residuals/left_arm')
    if not os.path.exists('images/residuals/base'):
        os.makedirs('images/residuals/base')
    if not os.path.exists('images/residuals/right_leg'):
        os.makedirs('images/residuals/right_leg')
    if not os.path.exists('images/residuals/left_leg'):
        os.makedirs('images/residuals/left_leg')
    if not os.path.exists('images/residuals/waist'):
        os.makedirs('images/residuals/waist')
    if not os.path.exists('images/tau_g/right_arm'):
        os.makedirs('images/tau_g/right_arm')
    if not os.path.exists('images/tau_g/left_arm'):
        os.makedirs('images/tau_g/left_arm')

    grouped_indices = defaultdict(list)

    for idx, name in enumerate(joint_names):
        base_name = '_'.join(name.split('_')[:2])  # E.g., "left_ankle" da "left_ankle_roll_joint"
        grouped_indices[base_name].append(idx)

    #################################
    # MPC PREDICTED TRAJECTORIES
    #################################
    #disabilita per ora
    if False:
        frames = []
        mpc_horizon = 20              # number of predicted MPC points
        mpc_dt = 0.1                  # MPC sample time (0.1 s)
        prediction_window = mpc_horizon * mpc_dt   # should be 2 seconds

        num_blocks = num_samples

        for block in range(0, num_blocks, 50):
            # Index range of the predictions in mpc_pred_com_pos
            start = block * mpc_horizon
            end = start + mpc_horizon

            # Compute the real time at which this block starts
            t0 = t[block]

            # Build the MPC prediction timeline (20 points over exactly 2 seconds)
            t_pred = np.linspace(t0, t0 + prediction_window, mpc_horizon)

            fig, ax = plt.subplots()

            # Plot MPC predicted COM (using uniform 2-second time axis)
            ax.plot(t_pred, mpc_pred_com_pos[start:end, 0],
                    label='MPC Pred COM X', linestyle='--')
            ax.plot(t_pred, mpc_pred_com_pos[start:end, 1],
                    label='MPC Pred COM Y', linestyle='--')
            ax.plot(t_pred, mpc_pred_com_pos[start:end, 2],
                    label='MPC Pred COM Z', linestyle='--')

            ax.set_xlabel('Time [s]')
            ax.set_ylabel('COM Position [m]')
            ax.set_title(f'MPC Predicted COM Position (Block {block})')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()

            # Convert figure to image frame
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)

        # Save GIF
        # imageio.mimsave('images/mpc/mpc_com_prediction.gif', frames, duration=0.2)

        with imageio.get_writer('images/mpc/mpc_pred_com_pos.mp4', fps=5) as writer:
            for frame in frames:
                writer.append_data(frame)

        frames = []
        mpc_horizon = 20              # number of predicted MPC points
        mpc_dt = 0.1                  # MPC sample time (0.1 s)
        prediction_window = mpc_horizon * mpc_dt   # should be 2 seconds

        num_blocks = num_samples

        for block in range(0, num_blocks, 50):
            # Index range of the predictions in mpc_pred_com_pos
            start = block * mpc_horizon
            end = start + mpc_horizon

            # Compute the real time at which this block starts
            t0 = t[block]

            # Build the MPC prediction timeline (20 points over exactly 2 seconds)
            t_pred = np.linspace(t0, t0 + prediction_window, mpc_horizon)

            fig, ax = plt.subplots()

            # Plot MPC predicted COM (using uniform 2-second time axis)
            ax.plot(t_pred, mpc_pred_com_vel[start:end, 0],
                    label='MPC Pred COM Vel X', linestyle='--')
            ax.plot(t_pred, mpc_pred_com_vel[start:end, 1],
                    label='MPC Pred COM Vel Y', linestyle='--')
            ax.plot(t_pred, mpc_pred_com_vel[start:end, 2],
                    label='MPC Pred COM Vel Z', linestyle='--')

            ax.set_xlabel('Time [s]')
            ax.set_ylabel('COM Velocity [m]')
            ax.set_title(f'MPC Predicted COM Velocity (Block {block})')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()

            # Convert figure to image frame
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)

        # Save GIF
        # imageio.mimsave('images/mpc/mpc_com_prediction.gif', frames, duration=0.2)

        with imageio.get_writer('images/mpc/mpc_pred_com_vel.mp4', fps=5) as writer:
            for frame in frames:
                writer.append_data(frame)

        frames = []
        mpc_horizon = 20              # number of predicted MPC points
        mpc_dt = 0.1                  # MPC sample time (0.1 s)
        prediction_window = mpc_horizon * mpc_dt   # should be 2 seconds

        num_blocks = num_samples

        for block in range(0, num_blocks, 50):
            # Index range of the predictions in mpc_pred_com_pos
            start = block * mpc_horizon
            end = start + mpc_horizon

            # Compute the real time at which this block starts
            t0 = t[block]

            # Build the MPC prediction timeline (20 points over exactly 2 seconds)
            t_pred = np.linspace(t0, t0 + prediction_window, mpc_horizon)

            fig, ax = plt.subplots()

            # Plot MPC predicted ZMP (using uniform 2-second time axis)
            ax.plot(t_pred, mpc_pred_zmp_pos[start:end, 0],
                    label='MPC Pred ZMP X', linestyle='--')
            ax.plot(t_pred, mpc_pred_zmp_pos[start:end, 1],
                    label='MPC Pred ZMP Y', linestyle='--')
            ax.plot(t_pred, mpc_pred_zmp_pos[start:end, 2],
                    label='MPC Pred ZMP Z', linestyle='--')

            ax.set_xlabel('Time [s]')
            ax.set_ylabel('ZMP Position [m]')
            ax.set_title(f'MPC Predicted ZMP Position (Block {block})')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()

            # Convert figure to image frame
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)

        # Save GIF
        # imageio.mimsave('images/mpc/mpc_com_prediction.gif', frames, duration=0.2)

        with imageio.get_writer('images/mpc/mpc_pred_zmp_pos.mp4', fps=5) as writer:
            for frame in frames:
                writer.append_data(frame)



    


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
        fig.savefig(f"images/wbc_solutions/wbc_joint_torques_ffw/{group_name}_input_joint_torques.png")
        plt.close(fig)
        figs.append(fig)

    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, q_dot_des[:, i], label=joint_names[i].strip())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [rad/s]')
        ax.set_title(f'WBC Desired Joint Velocity (Online Reference) - {group_name}')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"images/wbc_solutions/online_references/joint_velocities/{group_name}_q_dot_des.png")
        plt.close(fig)

    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            jname = joint_names[i].strip()
            line, = ax.plot(t, q_des[:, i], label=jname)
            if jname in joint_limits:
                lower, upper = joint_limits[jname]
                ax.axhline(lower, color=line.get_color(), linestyle='--', linewidth=1, alpha=0.5)
                ax.axhline(upper, color=line.get_color(), linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [rad]')
        ax.set_title(f'WBC Desired Joint Position (Online Reference) - {group_name}')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"images/wbc_solutions/online_references/joint_positions/{group_name}_q_des.png")
        plt.close(fig)

    # Floating-base online references (desired trajectory used for the
    # inverse-dynamics feedforward, one control cycle ahead of the current
    # state -- see WholeBodyController::compute_inverse_dynamics()).
    fig, ax = plt.subplots()
    ax.plot(t, base_position_des[:, 0], label='Base Position X')
    ax.plot(t, base_position_des[:, 1], label='Base Position Y')
    ax.plot(t, base_position_des[:, 2], label='Base Position Z')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('WBC Desired Floating Base Position (Online Reference)')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wbc_solutions/online_references/floating_base_positions/base_position_des.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, base_orientation_des[:, 0], label='Base Orientation X')
    ax.plot(t, base_orientation_des[:, 1], label='Base Orientation Y')
    ax.plot(t, base_orientation_des[:, 2], label='Base Orientation Z')
    ax.plot(t, base_orientation_des[:, 3], label='Base Orientation W')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Orientation [quat]')
    ax.set_title('WBC Desired Floating Base Orientation (Online Reference)')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wbc_solutions/online_references/floating_base_positions/base_orientation_des.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, base_linear_velocity_des[:, 0], label='Base Linear Velocity X')
    ax.plot(t, base_linear_velocity_des[:, 1], label='Base Linear Velocity Y')
    ax.plot(t, base_linear_velocity_des[:, 2], label='Base Linear Velocity Z')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('WBC Desired Floating Base Linear Velocity (Online Reference)')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wbc_solutions/online_references/floating_base_velocities/base_linear_velocity_des.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, base_angular_velocity_des[:, 0], label='Base Angular Velocity X')
    ax.plot(t, base_angular_velocity_des[:, 1], label='Base Angular Velocity Y')
    ax.plot(t, base_angular_velocity_des[:, 2], label='Base Angular Velocity Z')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular velocity [rad/s]')
    ax.set_title('WBC Desired Floating Base Angular Velocity (Online Reference)')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wbc_solutions/online_references/floating_base_velocities/base_angular_velocity_des.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, wbc_accelerations[:, 0], label='Acceleration X', color='blue')
    ax.plot(t, wbc_accelerations[:, 1], label='Acceleration Y', color='orange')
    ax.plot(t, wbc_accelerations[:, 2], label='Acceleration Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Acceleration [N]')
    ax.set_title('Acceleration base')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wbc_solutions/wbc_base_accelerations/linear_acceleration.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, wbc_accelerations[:, 3], label='Acceleration X', color='blue')
    ax.plot(t, wbc_accelerations[:, 4], label='Acceleration Y', color='orange')
    ax.plot(t, wbc_accelerations[:, 5], label='Acceleration Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Acceleration [N]')
    ax.set_title('Acceleration base')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wbc_solutions/wbc_base_accelerations/angular_acceleration.png")
    plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, wbc_accelerations[:, i + 6], label=joint_names[i].strip())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Acceleration [rad/s^2]')
        ax.set_title(f'WBC Acceleration - {group_name}')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"images/wbc_solutions/wbc_joint_accelerations/{group_name}_acceleration.png")
        plt.close(fig)
        figs.append(fig)

    wbc_wrench_labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    fig, ax = plt.subplots()
    for i, label in enumerate(wbc_wrench_labels):
        ax.plot(t, wbc_force_lsole[:, i], label=label)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N] / Torque [Nm]')
    ax.set_title('WBC Optimal Left Foot Wrench')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wbc_solutions/wbc_sole_forces/wbc_force_left_sole.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for i, label in enumerate(wbc_wrench_labels):
        ax.plot(t, wbc_force_rsole[:, i], label=label)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N] / Torque [Nm]')
    ax.set_title('WBC Optimal Right Foot Wrench')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wbc_solutions/wbc_sole_forces/wbc_force_right_sole.png")
    plt.close(fig)

    # WBC corner (contact-point) forces, one figure per foot, one subplot per force component
    corner_labels = ['Front-Left', 'Front-Right', 'Back-Left', 'Back-Right']
    corner_components = ['Fx', 'Fy', 'Fz']
    n_corners = len(corner_labels)

    def _plot_corner_wrenches(data, foot_name, filename):
        fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
        for comp_idx, ax in enumerate(axes):
            for corner_idx, corner_label in enumerate(corner_labels):
                ax.plot(
                    t, data[:, 3 * corner_idx + comp_idx],
                    label=corner_label, linewidth=2.0
                )
            ax.set_ylabel(f'{corner_components[comp_idx]} [N]', fontsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(loc='best', frameon=True, fontsize=9)
            ax.tick_params(axis='both', labelsize=9)
        axes[-1].set_xlabel('Time [s]', fontsize=11)
        fig.suptitle(f'WBC Corner Forces - {foot_name} Sole', fontsize=12)
        fig.tight_layout()
        fig.savefig(f"images/wbc_solutions/wbc_sole_forces/{filename}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    _plot_corner_wrenches(wbc_corner_forces_left, 'Left', 'wbc_corner_forces_left')
    _plot_corner_wrenches(wbc_corner_forces_right, 'Right', 'wbc_corner_forces_right')

    # Friction cone check: |fx|/fz and |fy|/fz ratios vs +-mu for each of the 4 corners.
    # The linearized friction pyramid used in the WBC enforces |fx| <= mu*fz and
    # |fy| <= mu*fz, so the ratios must stay within the +-mu band. Ratios are computed
    # and logged directly by the WBC (WalkingManager::frictionConeRatios).
    def _plot_friction_cone(ratio_x, ratio_y, foot_name, filename):
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        for corner_idx, corner_label in enumerate(corner_labels):
            axes[0].plot(t, ratio_x[:, corner_idx], label=corner_label, linewidth=1.5)
            axes[1].plot(t, ratio_y[:, corner_idx], label=corner_label, linewidth=1.5)
        for ax, ratio_name in zip(axes, ['|fx| / fz', '|fy| / fz']):
            ax.plot(t, wbc_friction_coefficient, color='red', linestyle='--', linewidth=1.5, label=r'$\mu$')
            ax.set_ylabel(ratio_name, fontsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(loc='best', frameon=True, fontsize=9)
            ax.tick_params(axis='both', labelsize=9)
        axes[-1].set_xlabel('Time [s]', fontsize=11)
        fig.suptitle(f'Friction Cone Constraint - {foot_name} Sole', fontsize=12)
        fig.tight_layout()
        fig.savefig(f"images/wbc_solutions/friction_cone/{filename}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    _plot_friction_cone(friction_cone_ratio_left_x, friction_cone_ratio_left_y, 'Left', 'friction_cone_left')
    _plot_friction_cone(friction_cone_ratio_right_x, friction_cone_ratio_right_y, 'Right', 'friction_cone_right')

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
    fig.savefig("images/wrench_estimations/sole_wrenches/estimated_force_left_sole.png")
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
    fig.savefig("images/wrench_estimations/sole_wrenches/estimated_force_right_sole.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, estimated_moment_lsole[:, 0], label='Estimated Moment Left Sole Mx', color='blue')
    ax.plot(t, estimated_moment_lsole[:, 1], label='Estimated Moment Left Sole My', color='orange')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Estimated Moment [Nm]')
    ax.set_title('Estimated Moments on Left Sole')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wrench_estimations/sole_wrenches/estimated_moment_left_sole.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, estimated_moment_rsole[:, 0], label='Estimated Moment Right Sole Mx', color='blue')
    ax.plot(t, estimated_moment_rsole[:, 1], label='Estimated Moment Right Sole My', color='orange')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Estimated Moment [Nm]')
    ax.set_title('Estimated Moments on Right Sole')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/wrench_estimations/sole_wrenches/estimated_moment_right_sole.png")
    plt.close(fig)

    # Side-by-side overviews (left sole on the left, right sole on the right),
    # one figure for the linear forces and one for the moments, so the two feet
    # can be read against each other on a shared time and value axis.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    for ax, data, foot_name in zip(
        axes,
        (estimated_force_lsole, estimated_force_rsole),
        ('Left', 'Right'),
    ):
        ax.plot(t, data[:, 0], label=r'$f_x$', color='blue', linewidth=1.8)
        ax.plot(t, data[:, 1], label=r'$f_y$', color='orange', linewidth=1.8)
        ax.plot(t, data[:, 2], label=r'$f_z$', color='green', linewidth=1.8)
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax.set_title(f'{foot_name} Sole', fontsize=11)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.tick_params(labelsize=9)
    # sharey hides the right subplot's tick labels: one label on the left is enough
    axes[0].set_ylabel('Estimated Force [N]', fontsize=11)
    fig.suptitle('Estimated Sole Forces', fontsize=13)
    fig.tight_layout()
    fig.savefig(
        "images/wrench_estimations/sole_wrenches/estimated_force_soles_overview.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    for ax, data, foot_name in zip(
        axes,
        (estimated_moment_lsole, estimated_moment_rsole),
        ('Left', 'Right'),
    ):
        ax.plot(t, data[:, 0], label=r'$m_x$', color='blue', linewidth=1.8)
        ax.plot(t, data[:, 1], label=r'$m_y$', color='orange', linewidth=1.8)
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax.set_title(f'{foot_name} Sole', fontsize=11)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel('Estimated Moment [Nm]', fontsize=11)
    fig.suptitle('Estimated Sole Moments', fontsize=13)
    fig.tight_layout()
    fig.savefig(
        "images/wrench_estimations/sole_wrenches/estimated_moment_soles_overview.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)



    #################################
    #  COM AND ZMP PLOTS
    #################################

    # REFERENCES PLOTS

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_acceleration[:, 0],
        label=r'Desired CoM Acceleration $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_acceleration[:, 1],
        label=r'Desired CoM Acceleration $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_acceleration[:, 2],
        label=r'Desired CoM Acceleration $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Acceleration [$\mathrm{m/s^2}$]', fontsize=11)
    ax.set_title('Desired Center of Mass Acceleration', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/references/des_com_acceleration_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, current_disturbance[:, 0],
        label=r'Disturbance $x$',
        linewidth=2.0
    )
    ax.plot(
        t, current_disturbance[:, 1],
        label=r'Disturbance $y$',
        linewidth=2.0
    )
    ax.plot(
        t, current_disturbance[:, 2],
        label=r'Disturbance $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Disturbance [$\mathrm{m/s^2}$]', fontsize=11)
    ax.set_title('PLIP Disturbance Term (before integration)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/references/current_disturbance_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_position[:, 0] - des_com_position[0, 0],
        label=r'Desired CoM Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_position[:, 1] - des_com_position[0, 1],
        label=r'Desired CoM Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_position[:, 2],
        label=r'Desired CoM Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Desired Center of Mass Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/references/des_com_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_velocity[:, 0],
        label=r'Desired CoM Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_velocity[:, 1],
        label=r'Desired CoM Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_velocity[:, 2],
        label=r'Desired CoM Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Desired Center of Mass Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/references/des_com_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_zmp_position[:, 0] - des_zmp_position[0, 0],
        label=r'Desired ZMP Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_zmp_position[:, 1] - des_zmp_position[0, 1],
        label=r'Desired ZMP Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_zmp_position[:, 2],
        label=r'Desired ZMP Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Desired Zero Moment Point Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/references/des_zmp_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)



    # ERROR PLOTS

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_zmp_position[:,0] - kf_zmp_position[:, 0],
        label=r'ZMP Position Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_zmp_position[:, 1] - kf_zmp_position[:, 1],
        label=r'ZMP Position Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_zmp_position[:, 2] - kf_zmp_position[:, 2],
        label=r'ZMP Position Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Zero Moment Point Position Error', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/errors/error_zmp_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_position[:,0] - kf_com_position[:, 0],
        label=r'CoM Position Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_position[:, 1] - kf_com_position[:, 1],
        label=r'CoM Position Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_position[:, 2] - kf_com_position[:, 2],
        label=r'CoM Position Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Center of Mass Position Error', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/errors/error_com_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_velocity[:,0] - kf_com_velocity[:, 0],
        label=r'CoM Velocity Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_velocity[:, 1] - kf_com_velocity[:, 1],
        label=r'CoM Velocity Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_velocity[:, 2] - kf_com_velocity[:, 2],
        label=r'CoM Velocity Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Center of Mass Velocity Error', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/errors/error_com_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_zmp_position[:, 0] - des_zmp_position[0, 0],
        label=r'Desired ZMP Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_zmp_position[:, 0] - kf_zmp_position[0, 0],
        label=r'Actual ZMP Position $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_zmp_position[:, 1] - des_zmp_position[0, 1],
        label=r'Desired ZMP Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_zmp_position[:, 1] - kf_zmp_position[0, 1],
        label=r'Actual ZMP Position $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_zmp_position[:, 2],
        label=r'Desired ZMP Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_zmp_position[:, 2],
        label=r'Actual ZMP Position $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Comparison between reference and actual Zero Moment Point Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/errors/comparison_zmp_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_position[:,0] - des_com_position[0, 0],
        label=r'Desired CoM Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_com_position[:,0] - kf_com_position[0, 0],
        label=r'Actual CoM Position $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_com_position[:, 1] - des_com_position[0, 1],
        label=r'Desired CoM Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_com_position[:, 1] - kf_com_position[0, 1],
        label=r'Actual CoM Position $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_com_position[:, 2],
        label=r'Desired CoM Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_com_position[:, 2],
        label=r'Actual CoM Position $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Comparison between reference and actual Center of Mass Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/errors/comparison_com_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, des_com_velocity[:,0],
        label=r'Desired CoM Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_com_velocity[:,0],
        label=r'Actual CoM Velocity $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_com_velocity[:, 1],
        label=r'Desired CoM Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, des_com_velocity[:, 1],
        label=r'Actual CoM Velocity $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, des_com_velocity[:, 2],
        label=r'Desired CoM Velocity $z$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_com_velocity[:, 2],
        label=r'Actual CoM Velocity $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Comparison between reference and actual Center of Mass Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/errors/comparison_com_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, kf_com_position[:, 0] - kf_com_position[0, 0],
        label=r'CoM Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_zmp_position[:, 0] - kf_zmp_position[0, 0],
        label=r'ZMP Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_fb[:, 0] - p_lsole_fb[0, 0],
        label=r'Left Foot Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_fb[:, 0] - p_rsole_fb[0, 0],
        label=r'Right Foot Position $x$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position $x$ [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Motion in the forward direction', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/motion_x.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, kf_com_position[:, 1],
        label=r'CoM Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_zmp_position[:, 1],
        label=r'ZMP Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_fb[:, 1],
        label=r'Left Foot Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_fb[:, 1],
        label=r'Right Foot Position $y$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position $y$ [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Motion in the lateral direction', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='upper left',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/motion_y.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, kf_com_position[:, 2],
        label=r'CoM Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, kf_zmp_position[:, 2],
        label=r'ZMP Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_fb[:, 2],
        label=r'Left Foot Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_fb[:, 2],
        label=r'Right Foot Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position $z$ [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Motion in the vertical direction', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/com/motion_z.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


    fig, ax = plt.subplots()
    ax.plot(t, ef_zmp_position[:, 0], label='residual based ZMP X', color='blue')
    ax.plot(t, ef_zmp_position[:, 1], label='residual based ZMP Y', color='orange')
    ax.plot(t, ef_zmp_position[:, 2], label='residual based ZMP Z', color='green')
    ax.plot(t, fb_zmp_position[:, 0], label='plip based ZMP X', color='blue', linestyle='--')
    ax.plot(t, fb_zmp_position[:, 1], label='plip based ZMP Y', color='orange', linestyle='--')
    ax.plot(t, fb_zmp_position[:, 2], label='plip based ZMP Z', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('ZMP Position Feedback: PLIP-based vs Residual-based')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/zmp_lip_vs_residual_plot.png")
    plt.close(fig)

    # Error stats: residual-based ZMP vs PLIP-based ZMP
    def _print_zmp_error_stats(zmp_res, zmp_plip):
        err = zmp_res - zmp_plip
        if not np.isfinite(err).any():
            print("[ZMP] residual-based and PLIP-based estimates never overlap — "
                  "no stats computed.")
            return
        norm_err = np.linalg.norm(err, axis=1)
        norm_err_xy = np.linalg.norm(err[:, :2], axis=1)
        print(f"\n{'='*50}")
        print(f"  Error — ZMP residual-based vs PLIP-based")
        print(f"{'='*50}")
        print(f"  {'Axis':<8} {'Mean Error [m]':>18} {'Variance [m²]':>18}")
        print(f"  {'-'*46}")
        for i, lbl in enumerate(['x', 'y', 'z']):
            mean_i = _nan_mean(err[:, i], axis=None)
            var_i = np.nanvar(err[:, i])
            print(f"  ZMP_{lbl:<4} {mean_i:>18.6f} {var_i:>18.6f}")
        print(f"  {'-'*46}")
        print(f"  {'||err||':<8} {_nan_mean(norm_err, axis=None):>18.6f} "
              f"{np.nanvar(norm_err):>18.6f}")
        print(f"  {'||err||xy':<8} {_nan_mean(norm_err_xy, axis=None):>17.6f} "
              f"{np.nanvar(norm_err_xy):>18.6f}")
        print(f"{'='*50}")

    _print_zmp_error_stats(ef_zmp_position, fb_zmp_position)

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    axis_labels = ['x', 'y', 'z']
    for i, ax in enumerate(axes):
        ax_rate = ax.twinx()
        l1, = ax.plot(
            t, angular_momentum[:, i],
            label=r'Angular Momentum $%s$' % axis_labels[i],
            color='blue', linewidth=2.0
        )
        l2, = ax_rate.plot(
            t, angular_momentum_rate[:, i],
            label=r'Angular Momentum Rate $%s$' % axis_labels[i],
            color='orange', linewidth=2.0, linestyle='--'
        )
        ax.set_ylabel(r'$L_%s$ [$\mathrm{kg\,m^2/s}$]' % axis_labels[i], fontsize=10)
        ax_rate.set_ylabel(r'$\dot{L}_%s$ [$\mathrm{kg\,m^2/s^2}$]' % axis_labels[i], fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(handles=[l1, l2], loc='best', frameon=True, fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        ax_rate.tick_params(axis='both', labelsize=9)
    axes[-1].set_xlabel('Time [s]', fontsize=11)
    fig.suptitle('Centroidal Angular Momentum and its Rate of Change', fontsize=12)
    fig.tight_layout()
    fig.savefig(
        "images/com/angular_momentum_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


    ##########################
    #  FEET PLOT
    ##########################

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, p_lsole_des[:, 0],
        label=r'Desired Left Sole Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 1],
        label=r'Desired Left Sole Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 2],
        label=r'Desired Left Sole Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Desired Left Sole Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/references/desired_left_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, p_lsole_des[:, 0],
        label=r'Desired Right Sole Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 1],
        label=r'Desired Right Sole Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 2],
        label=r'Desired Right Sole Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Desired Right Sole Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/references/desired_right_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, p_lsole_des[:, 0] - p_lsole_fb[:, 0],
        label=r'Left Sole Position Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 1] - p_lsole_fb[:, 1],
        label=r'Left Sole Position Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 2] - p_lsole_fb[:, 2],
        label=r'Left Sole Position Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Error between Desired and Actual Left Sole Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/error_left_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, p_rsole_des[:, 0] - p_rsole_fb[:, 0],
        label=r'Right Sole Position Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_des[:, 1] - p_rsole_fb[:, 1],
        label=r'Right Sole Position Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_des[:, 2] - p_rsole_fb[:, 2],
        label=r'Right Sole Position Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Error between Desired and Actual Right Sole Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/error_right_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, p_lsole_fb[:, 0] - p_lsole_fb[0, 0],
        label=r'Actual Left Sole Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 0] - p_lsole_des[0, 0],
        label=r'Desired Left Sole Position $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, p_lsole_fb[:, 1],
        label=r'Actual Left Sole Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 1],
        label=r'Desired Left Sole Position $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, p_lsole_fb[:, 2],
        label=r'Actual Left Sole Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, p_lsole_des[:, 2],
        label=r'Desired Left Sole Position $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Comparison between Desired and Actual Left Sole Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/comparison_left_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, p_rsole_fb[:, 0] - p_rsole_fb[0, 0],
        label=r'Actual Right Sole Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_des[:, 0] - p_rsole_des[0, 0],
        label=r'Desired Right Sole Position $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, p_rsole_fb[:, 1],
        label=r'Actual Right Sole Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_des[:, 1],
        label=r'Desired Right Sole Position $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, p_rsole_fb[:, 2],
        label=r'Actual Right Sole Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, p_rsole_des[:, 2],
        label=r'Desired Right Sole Position $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Comparison between Desired and Actual Right Sole Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/comparison_right_sole_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, v_lsole_des[:, 0] - v_lsole_fb[:, 0],
        label=r'Left Sole Velocity Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, v_lsole_des[:, 1] - v_lsole_fb[:, 1],
        label=r'Left Sole Velocity Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, v_lsole_des[:, 2] - v_lsole_fb[:, 2],
        label=r'Left Sole Velocity Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Error between Desired and Actual Left Sole Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/error_left_sole_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, v_rsole_des[:, 0] - v_rsole_fb[:, 0],
        label=r'Right Sole Velocity Error $x$',
        linewidth=2.0
    )
    ax.plot(
        t, v_rsole_des[:, 1] - v_rsole_fb[:, 1],
        label=r'Right Sole Velocity Error $y$',
        linewidth=2.0
    )
    ax.plot(
        t, v_rsole_des[:, 2] - v_rsole_fb[:, 2],
        label=r'Right Sole Velocity Error $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Error between Desired and Actual Right Sole Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/error_right_sole_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, v_lsole_fb[:, 0],
        label=r'Actual Left Sole Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, v_lsole_des[:, 0],
        label=r'Desired Left Sole Velocity $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, v_lsole_fb[:, 1],
        label=r'Actual Left Sole Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, v_lsole_des[:, 1],
        label=r'Desired Left Sole Velocity $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, v_lsole_fb[:, 2],
        label=r'Actual Left Sole Velocity $z$',
        linewidth=2.0
    )
    ax.plot(
        t, v_lsole_des[:, 2],
        label=r'Desired Left Sole Velocity $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Comparison between Desired and Actual Left Sole Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/comparison_left_sole_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, v_rsole_fb[:, 0],
        label=r'Actual Right Sole Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, v_rsole_des[:, 0],
        label=r'Desired Right Sole Velocity $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, v_rsole_fb[:, 1],
        label=r'Actual Right Sole Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, v_rsole_des[:, 1],
        label=r'Desired Right Sole Velocity $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, v_rsole_fb[:, 2],
        label=r'Actual Right Sole Velocity $z$',
        linewidth=2.0
    )
    ax.plot(
        t, v_rsole_des[:, 2],
        label=r'Desired Right Sole Velocity $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Comparison between Desired and Actual Right Sole Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/soles/errors/comparison_right_sole_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)



    ##########################
    #  EKF PLOTS
    ##########################

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_position[:, i],
                label=r'Measured Position'+ f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0
            )
            ax.plot(
                t, ekf_joint_position[:, i],
                label=r'Filtered Position' + f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0,
                linestyle='--'
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=12)
        ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=7
        )
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/ekf/joints/positions/{group_name}_position_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_velocity[:, i],
                label=r'Measured Velocity'+ f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0
            )
            ax.plot(
                t, ekf_joint_velocity[:, i],
                label=r'Filtered Velocity' + f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0,
                linestyle='--'
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Velocity [$\mathrm{rad/s}$]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=12)
        ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=7
        )
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/ekf/joints/velocities/{group_name}_velocity_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_position[:, i] - ekf_joint_position[:, i],
                label=r'Error Position'+ f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=12)
        ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=7
        )
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/ekf/joints/error/positions/error_{group_name}_position_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_velocity[:, i] - ekf_joint_velocity[:, i],
                label=r'Error Velocity'+ f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Velocity [$\mathrm{rad/s}$]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=12)
        ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=7
        )
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/ekf/joints/error/velocities/error_{group_name}_velocity_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
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
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{rad}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured Joints Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=4
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/joints/error/error_joint_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
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
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{rad/s}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured Joints Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=4
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/joints/error/error_joint_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_position[:, 0],
        label=r'EKF Base Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_position[:, 1],
        label=r'EKF Base Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_position[:, 2],
        label=r'EKF Base Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('EKF Base Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/base_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_velocity[:, 0],
        label=r'EKF Base Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_velocity[:, 1],
        label=r'EKF Base Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_velocity[:, 2],
        label=r'EKF Base Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('EKF Base Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/base_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_orientation[:, 0],
        label=r'EKF Base Orientation $W$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_orientation[:, 1],
        label=r'EKF Base Orientation $X$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_orientation[:, 2],
        label=r'EKF Base Orientation $Y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_orientation[:, 3],
        label=r'EKF Base Orientation $Z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{quat}$]', fontsize=11)
    ax.set_title('EKF Base Orientation Quat', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/base_orientation_quat_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_orientation_rpy[:, 0],
        label=r'EKF Base Orientation $R$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_orientation_rpy[:, 1],
        label=r'EKF Base Orientation $P$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_orientation_rpy[:, 2],
        label=r'EKF Base Orientation $Y$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{grad}$]', fontsize=11)
    ax.set_title('EKF Base Orientation RPY', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/base_orientation_rpy_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_angular_velocity[:, 0],
        label=r'EKF Base Angular Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_angular_velocity[:, 1],
        label=r'EKF Base Angular Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_angular_velocity[:, 2],
        label=r'EKF Base Angular Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('EKF Base Angular Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/base_angular_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


    # plot mean squared error between ekf joint position and simulated joint position
    mse_position = _nan_mean((ekf_joint_position - measured_joint_position) ** 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_joints), mse_position, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(r'Mean Squared Error', fontsize=14)
    ax.set_title('Mean Squared Error between EKF Joint Position and Feedback Joint Position', fontsize=16)
    ax.set_xticks(range(num_joints))
    ax.set_xticklabels([name.strip().replace("_"," ").replace("joint", "") for name in joint_names], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig("images/ekf/performance/mse_joint_position_plot.png")
    plt.close(fig)

    #plot mean squared error between ekf joint velocity and simulated joint velocity
    mse_velocity = _nan_mean((ekf_joint_velocity - measured_joint_velocity) ** 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_joints), mse_velocity, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(r'Mean Squared Error', fontsize=14)
    ax.set_title('Mean Squared Error between EKF Joint Velocity and Feedback Joint Velocity', fontsize=16)
    ax.set_xticks(range(num_joints))
    ax.set_xticklabels([name.strip().replace("_"," ").replace("joint", "") for name in joint_names], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig("images/ekf/performance/mse_joint_velocity_plot.png")
    plt.close(fig)

    #plot mean squared error between ekf base position and simulated base position, orientation, velocity, angular velocity
    mse_base_position = _nan_mean((ekf_base_position - odometry_base_position) ** 2)
    mse_base_velocity = _nan_mean((ekf_base_velocity - odometry_base_velocity) ** 2)
    mse_base_orientation = _nan_mean((ekf_base_orientation - odometry_imu_orientation) ** 2)
    mse_base_angular_velocity = _nan_mean((ekf_base_angular_velocity - measured_imu_angular_velocity) ** 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(3), mse_base_position, label='Position MSE', color='skyblue', alpha=0.7)
    ax.bar(range(3, 6), mse_base_velocity, label='Velocity MSE', color='orange', alpha=0.7)
    ax.bar(range(6, 10), mse_base_orientation, label='Orientation MSE', color='green', alpha=0.7)
    ax.bar(range(10, 13), mse_base_angular_velocity, label='Angular Velocity MSE', color='red', alpha=0.7)
    ax.set_xlabel('Base State Index', fontsize=14)
    ax.set_ylabel('Mean Squared Error', fontsize=14)
    ax.set_title('Mean Squared Error between EKF Base States and Simulated Base States', fontsize=16)
    ax.set_xticks(range(13))
    ax.set_xticklabels(['Position X', 'Position Y', 'Position Z', 'Velocity X', 'Velocity Y', 'Velocity Z', 'Orientation W', 'Orientation X', 'Orientation Y', 'Orientation Z',
                        'Angular Velocity X', 'Angular Velocity Y', 'Angular Velocity Z'], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/performance/mse_base_states_plot.png")
    plt.close(fig)

    #plot variance between ekf joint position and simulated joint position
    variance_position = np.var(ekf_joint_position - measured_joint_position, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_joints), variance_position, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(r'Variance', fontsize=14)
    ax.set_title('Variance between EKF Joint Position and Feedback Joint Position', fontsize=16)
    ax.set_xticks(range(num_joints))
    ax.set_xticklabels([name.strip().replace("_"," ").replace("joint", "") for name in joint_names], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig("images/ekf/performance/var_joint_position_plot.png")
    plt.close(fig)

    #plot variance between ekf joint position and simulated joint position
    variance_velocity = np.var(ekf_joint_velocity - measured_joint_velocity, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_joints), variance_velocity, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(r'Variance', fontsize=14)
    ax.set_title('Variance between EKF Joint Velocity and Feedback Joint Velocity', fontsize=16)
    ax.set_xticks(range(num_joints))
    ax.set_xticklabels([name.strip().replace("_"," ").replace("joint", "") for name in joint_names], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig("images/ekf/performance/var_joint_velocity_plot.png")
    plt.close(fig)

    #plot variance between ekf base position and simulated base position, orientation, velocity, angular velocity
    variance_base_position = np.var(ekf_base_position - odometry_base_position, axis=0)
    variance_base_velocity = np.var(ekf_base_velocity - odometry_base_velocity, axis=0)
    variance_base_orientation = np.var(ekf_base_orientation - odometry_imu_orientation, axis=0)
    variance_base_angular_velocity = np.var(ekf_base_angular_velocity - measured_imu_angular_velocity, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(3), variance_base_position, label='Position Variance', color='skyblue', alpha=0.7)
    ax.bar(range(3, 6), variance_base_velocity, label='Velocity Variance', color='orange', alpha=0.7)
    ax.bar(range(6, 10), variance_base_orientation, label='Orientation Variance', color='green', alpha=0.7)
    ax.bar(range(10, 13), variance_base_angular_velocity, label='Angular Velocity Variance', color='red', alpha=0.7)
    ax.set_xlabel('Base State Index', fontsize=14)
    ax.set_ylabel('Variance', fontsize=14)
    ax.set_title('Variance between EKF Base States and Simulated Base States', fontsize=16)
    ax.set_xticks(range(13))
    ax.set_xticklabels(['Position X', 'Position Y', 'Position Z', 'Velocity X', 'Velocity Y', 'Velocity Z', 'Orientation W', 'Orientation X', 'Orientation Y', 'Orientation Z',
                        'Angular Velocity X', 'Angular Velocity Y', 'Angular Velocity Z'], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/performance/var_base_states_plot.png")
    plt.close(fig)

    #plot variance of measured joint velocity
    variance_measured_velocity = np.var(measured_joint_velocity, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_joints), variance_measured_velocity, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(r'Variance', fontsize=14)
    ax.set_title('Variance of Feedback Joint Velocity', fontsize=16)
    ax.set_xticks(range(num_joints))
    ax.set_xticklabels([name.strip().replace("_"," ").replace("joint", "") for name in joint_names], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig("images/ekf/performance/var_joint_velocity_measured_plot.png")
    plt.close(fig)

    #plot variance of ekf joint velocity
    variance_ekf_velocity = np.var(ekf_joint_velocity, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_joints), variance_ekf_velocity, color='skyblue')
    ax.set_xlabel('Joint Index', fontsize=14)
    ax.set_ylabel(r'Variance', fontsize=14)
    ax.set_title('Variance of Feedback Joint Velocity', fontsize=16)
    ax.set_xticks(range(num_joints))
    ax.set_xticklabels([name.strip().replace("_"," ").replace("joint", "") for name in joint_names], rotation=45, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig("images/ekf/performance/var_joint_velocity_filtered_plot.png")
    plt.close(fig)

    #plot torso orientation error
    fig, ax = plt.subplots()
    ax.plot(t, torso_orientation[:, 0] - des_torso_orientation[:, 0], label='Torso Orientation Roll Error', color='blue')
    ax.plot(t, torso_orientation[:, 1] - des_torso_orientation[:, 1], label='Torso Orientation Pitch Error', color='orange')
    ax.plot(t, torso_orientation[:, 2] - des_torso_orientation[:, 2], label='Torso Orientation Yaw Error', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Orientation [rad]')
    ax.set_title('Torso Orientation Error between feedback and desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/torso_orientation_error_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, torso_angular_velocity[:, 0], label='Torso Angular Velocity X', color='blue')
    ax.plot(t, torso_angular_velocity[:, 1], label='Torso Angular Velocity Y', color='orange')
    ax.plot(t, torso_angular_velocity[:, 2], label='Torso Angular Velocity Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular Velocity [rad/s]')
    ax.set_title('Torso Angular Velocity from feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/torso_angular_velocity_plot.png")
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_position[:, 0],
        label=r'EKF Base Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_position[:, 1],
        label=r'EKF Base Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_position[:, 2],
        label=r'EKF Base Position $z$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_base_position[:, 0],
        label=r'Measured Base Position $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_base_position[:, 1],
        label=r'Measured Base Position $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_base_position[:, 2],
        label=r'Measured Base Position $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Comparison between EKF and Measured Base Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_base_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_velocity[:, 0],
        label=r'EKF Base Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_velocity[:, 1],
        label=r'EKF Base Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_velocity[:, 2],
        label=r'EKF Base Velocity $z$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_base_velocity[:, 0],
        label=r'Measured Base Velocity $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_base_velocity[:, 1],
        label=r'Measured Base Velocity $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_base_velocity[:, 2],
        label=r'Measured Base Velocity $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Comparison between EKF and Measured Base Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_base_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_imu_orientation[:, 0],
        label=r'EKF IMU Orientation $W$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation[:, 1],
        label=r'EKF IMU Orientation $X$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation[:, 2],
        label=r'EKF IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation[:, 3],
        label=r'EKF IMU Orientation $Z$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation[:, 0],
        label=r'Measured IMU Orientation $W$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_imu_orientation[:, 1],
        label=r'Measured IMU Orientation $X$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_imu_orientation[:, 2],
        label=r'Measured IMU Orientation $Y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_imu_orientation[:, 3],
        label=r'Measured IMU Orientation $Z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{quat}$]', fontsize=11)
    ax.set_title('Comparison between EKF and Measured IMU Orientation Quat', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_imu_orientation_quat_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_imu_orientation_rpy[:, 0],
        label=r'EKF IMU Orientation $R$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation_rpy[:, 1],
        label=r'EKF IMU Orientation $P$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation_rpy[:, 2],
        label=r'EKF IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation_rpy[:, 0],
        label=r'Measured IMU Orientation $R$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_imu_orientation_rpy[:, 1],
        label=r'Measured IMU Orientation $P$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, odometry_imu_orientation_rpy[:, 2],
        label=r'Measured IMU Orientation $Y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{rad}$]', fontsize=11)
    ax.set_title('Comparison between EKF and Measured IMU Orientation RPY', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_imu_orientation_rpy_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_imu_angular_velocity[:, 0],
        label=r'EKF IMU Angular Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_angular_velocity[:, 1],
        label=r'EKF IMU Angular Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_angular_velocity[:, 2],
        label=r'EKF IMU Angular Velocity $z$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_angular_velocity[:, 0],
        label=r'Measured IMU Angular Velocity $x$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, measured_imu_angular_velocity[:, 1],
        label=r'Measured IMU Angular Velocity $y$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.plot(
        t, measured_imu_angular_velocity[:, 2],
        label=r'Measured IMU Angular Velocity $z$',
        linewidth=2.0,
        linestyle='--'
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Angular Velocity [$\mathrm{rad/s}$]', fontsize=11)
    ax.set_title('Comparison between EKF and Measured IMU Angular Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/comparison_imu_angular_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_position[:, 0] - odometry_base_position[:, 0],
        label=r'Error Base Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_position[:, 1] - odometry_base_position[:, 1],
        label=r'Error Base Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_position[:, 2] - odometry_base_position[:, 2],
        label=r'Error Base Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured Base Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/error_base_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_base_velocity[:, 0] - odometry_base_velocity[:, 0],
        label=r'EKF Base Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_velocity[:, 1] - odometry_base_velocity[:, 1],
        label=r'EKF Base Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_base_velocity[:, 2] - odometry_base_velocity[:, 2],
        label=r'EKF Base Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured Base Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/error_base_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_imu_orientation[:, 0] - odometry_imu_orientation[:, 0],
        label=r'Error IMU Orientation $W$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation[:, 1] - odometry_imu_orientation[:, 1],
        label=r'Error IMU Orientation $X$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation[:, 2] - odometry_imu_orientation[:, 2],
        label=r'Error IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation[:, 3] - odometry_imu_orientation[:, 3],
        label=r'Error IMU Orientation $Z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{quat}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured IMU Orientation Quat', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/error_imu_orientation_quat_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_imu_orientation_rpy[:, 0] - odometry_imu_orientation_rpy[:, 0],
        label=r'Error IMU Orientation $R$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation_rpy[:, 1] - odometry_imu_orientation_rpy[:, 1],
        label=r'Error IMU Orientation $P$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_orientation_rpy[:, 2] - odometry_imu_orientation_rpy[:, 2],
        label=r'Error IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{rad}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured IMU Orientation RPY', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/error_imu_orientation_rpy_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, ekf_imu_angular_velocity[:, 0] - measured_imu_angular_velocity[:, 0],
        label=r'Error IMU Angular Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_angular_velocity[:, 1] - measured_imu_angular_velocity[:, 1],
        label=r'Error IMU Angular Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, ekf_imu_angular_velocity[:, 2] - measured_imu_angular_velocity[:, 2],
        label=r'Error IMU Angular Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Angular Velocity [$\mathrm{rad/s}$]', fontsize=11)
    ax.set_title('Error between EKF and Measured IMU Angular Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/ekf/base/errors/error_imu_angular_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)


    ##########################
    #  FEEDBACK PLOTS
    ##########################

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, odometry_base_position[:, 0],
        label=r'Odometry Base Position $x$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_base_position[:, 1],
        label=r'Odometry Base Position $y$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_base_position[:, 2],
        label=r'Odometry Base Position $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Position [$\mathrm{m}$]', fontsize=11)
    ax.set_title('Odometry Base Position', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/odometry_base_position_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, odometry_base_velocity[:, 0],
        label=r'Odometry Base Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_base_velocity[:, 1],
        label=r'Odometry Base Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_base_velocity[:, 2],
        label=r'Odometry Base Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Velocity [$\mathrm{m/s}$]', fontsize=11)
    ax.set_title('Odometry Base Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/odometry_base_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, odometry_imu_orientation[:, 0],
        label=r'Odometry IMU Orientation $W$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation[:, 1],
        label=r'Odometry IMU Orientation $X$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation[:, 2],
        label=r'Odometry IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation[:, 3],
        label=r'Odometry IMU Orientation $Z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{quat}$]', fontsize=11)
    ax.set_title('Odometry IMU Orientation Quat', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/odometry_imu_orientation_quat_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, odometry_imu_orientation_rpy[:, 0],
        label=r'Odometry IMU Orientation $R$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation_rpy[:, 1],
        label=r'Odometry IMU Orientation $P$',
        linewidth=2.0
    )
    ax.plot(
        t, odometry_imu_orientation_rpy[:, 2],
        label=r'Odometry IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{rad}$]', fontsize=11)
    ax.set_title('Odometry IMU Orientation RPY', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/odometry_imu_orientation_rpy_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, measured_imu_orientation[:, 0],
        label=r'Measured IMU Orientation $W$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_orientation[:, 1],
        label=r'Measured IMU Orientation $X$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_orientation[:, 2],
        label=r'Measured IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_orientation[:, 3],
        label=r'Measured IMU Orientation $Z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{quat}$]', fontsize=11)
    ax.set_title('Measured IMU Orientation Quat', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/measured_imu_orientation_quat_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, measured_imu_orientation_rpy[:, 0],
        label=r'Measured IMU Orientation $R$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_orientation_rpy[:, 1],
        label=r'Measured IMU Orientation $P$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_orientation_rpy[:, 2],
        label=r'Measured IMU Orientation $Y$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Orientation [$\mathrm{rad}$]', fontsize=11)
    ax.set_title('Measured IMU Orientation RPY', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/measured_imu_orientation_rpy_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, measured_imu_angular_velocity[:, 0],
        label=r'Measured IMU Angular Velocity $x$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_angular_velocity[:, 1],
        label=r'Measured IMU Angular Velocity $y$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_angular_velocity[:, 2],
        label=r'Measured IMU Angular Velocity $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Angular Velocity [$\mathrm{rad/s}$]', fontsize=11)
    ax.set_title('Measured Base Angular Velocity', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/measured_imu_angular_velocity_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t, measured_imu_accelerometer[:, 0],
        label=r'Measured IMU Acceleration $x$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_accelerometer[:, 1],
        label=r'Measured IMU Acceleration $y$',
        linewidth=2.0
    )
    ax.plot(
        t, measured_imu_accelerometer[:, 2],
        label=r'Measured IMU Acceleration $z$',
        linewidth=2.0
    )
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'Acceleration [$\mathrm{m/s^2}$]', fontsize=11)
    ax.set_title('Measured IMU Acceleration', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(
        loc='best',
        frameon=True,
        fontsize=9
    )
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/feedback/base/measured_imu_acceleration_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
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
    fig.savefig("images/feedback/joints/velocities/overall_joint_velocity_plot.png")
    plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_position[:, i],
                label=r'Measured Position'+ f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Position [$\mathrm{rad}$]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=12)
        ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=7
        )
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/feedback/joints/positions/{group_name}_position_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_velocity[:, i],
                label=r'Measured Velocity'+ f' {joint_names[i].replace(group_name, "").replace("_", "").replace("joint", "")}',
                linewidth=2.0
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Velocity [$\mathrm{rad/s}$]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=12)
        ax.legend(
            loc='upper left',
            frameon=True,
            fontsize=7
        )
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/feedback/joints/velocities/{group_name}_velocity_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)




    ##########################
    #  MEASURED MOTOR TORQUES (robot experiment)
    ##########################

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in indices:
            ax.plot(
                t, measured_joint_torque[:, i],
                label=joint_names[i].replace(group_name, '').replace('_', '').replace('joint', '').strip(),
                linewidth=2.0
            )
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'Torque [$\mathrm{Nm}$]', fontsize=11)
        ax.set_title(f'Measured Motor Torques — {group_name.replace("_", " ").title()}', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', frameon=True, fontsize=7)
        ax.tick_params(axis='both', labelsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/feedback/motor_torques/{group_name}_measured_torque.png",
            dpi=300,
            bbox_inches='tight'
        )
        figs.append(fig)
        plt.close(fig)

    ##########################
    #  EXECUTION TIME PLOTS
    ##########################

    exec_times = {
        'EKF': execution_time_ekf,
        'KF': execution_time_kf,
        'MPC': execution_time_mpc,
        'WBC': execution_time_wbc,
        'RB-WO': execution_time_res_obs,
        'HAC': execution_time_hac,
        'COOP_PLANNER': execution_time_coop_planner,
        'Update': execution_time_update
    }

    for name, times in exec_times.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            times,
            linewidth=2.0,
            label=f'{name}'
        )
        if name == 'Update':
            ax.axhline(
                y=2000,
                linestyle='--',
                linewidth=1.5,
                label='Real-time threshold (2000 µs)'
            )
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(r'Execution Time [$\mu s$]', fontsize=11)
        ax.set_title(f'{name} Execution Time per Iteration', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(frameon=True, fontsize=10)
        fig.tight_layout()
        fig.savefig(
            f"images/execution_times/{name}_execution_time_plot.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig)

    total_execution_time = (
        execution_time_ekf +
        execution_time_kf +
        execution_time_mpc +
        execution_time_wbc +
        execution_time_res_obs +
        execution_time_hac +
        execution_time_coop_planner
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        total_execution_time,
        linewidth=2.0,
        label='Total Execution Time'
    )
    ax.axhline(
        y=2000,
        linestyle='--',
        linewidth=1.5,
        label='Real-time threshold (2000 µs)',
        color='red'
    )
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel(r'Total Execution Time [$\mu s$]', fontsize=11)
    ax.set_title('Total Execution Time per Iteration', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(
        "images/execution_times/total_execution_time_plot.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)



    # -----------------------------------------------------------------------
    # Hand Admittance Controller (HAC) — e_h and e_h_dot
    # Replicates the plots produced by plot_hac.py.
    # Data: hac_eh.txt (N,2), hac_eh_dot.txt (N,2) from the selected folder.
    # Output: images/hac/
    # -----------------------------------------------------------------------
    hac_eh = _load('hac_eh.txt', 2)
    hac_eh_dot = _load('hac_eh_dot.txt', 2)

    if not os.path.exists('images/hac'):
        os.makedirs('images/hac')

    N_hac = hac_eh.shape[0]
    t_hac = _time(np.arange(N_hac) / 500.0)  # control frequency 500 Hz

    # Plot 1 — e_h (average hand position error, F frame xy)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(t_hac, hac_eh[:, 0], linewidth=1.8, label=r'$e_{h,x}$')
    axes[0].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel(r'$e_{h,x}$ [m]', fontsize=11)
    axes[0].set_title(r'Average Hand Position Error $e_h$ (F frame)', fontsize=12)
    axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0].legend(fontsize=10)

    axes[1].plot(t_hac, hac_eh[:, 1], linewidth=1.8, color='tab:orange', label=r'$e_{h,y}$')
    axes[1].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel(r'$e_{h,y}$ [m]', fontsize=11)
    axes[1].set_xlabel('Time [s]', fontsize=11)
    axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1].legend(fontsize=10)

    fig.tight_layout()
    fig.savefig('images/hac/eh.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot 2 — e_h_dot (derivative of average hand error, F frame xy)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(t_hac, hac_eh_dot[:, 0], linewidth=1.8, color='tab:green', label=r'$\dot{e}_{h,x}$')
    axes[0].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel(r'$\dot{e}_{h,x}$ [m/s]', fontsize=11)
    axes[0].set_title(r'Average Hand Error Derivative $\dot{e}_h$ (F frame)', fontsize=12)
    axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0].legend(fontsize=10)

    axes[1].plot(t_hac, hac_eh_dot[:, 1], linewidth=1.8, color='tab:red', label=r'$\dot{e}_{h,y}$')
    axes[1].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel(r'$\dot{e}_{h,y}$ [m/s]', fontsize=11)
    axes[1].set_xlabel('Time [s]', fontsize=11)
    axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1].legend(fontsize=10)

    fig.tight_layout()
    fig.savefig('images/hac/ehdot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot 3 — combined overview (4 subplots in one figure)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    axes[0, 0].plot(t_hac, hac_eh[:, 0], linewidth=1.8, label=r'$e_{h,x}$')
    axes[0, 1].plot(t_hac, hac_eh[:, 1], linewidth=1.8, color='tab:orange', label=r'$e_{h,y}$')
    axes[1, 0].plot(t_hac, hac_eh_dot[:, 0], linewidth=1.8, color='tab:green', label=r'$\dot{e}_{h,x}$')
    axes[1, 1].plot(t_hac, hac_eh_dot[:, 1], linewidth=1.8, color='tab:red', label=r'$\dot{e}_{h,y}$')

    hac_labels = [r'$e_{h,x}$ [m]', r'$e_{h,y}$ [m]',
                  r'$\dot{e}_{h,x}$ [m/s]', r'$\dot{e}_{h,y}$ [m/s]']

    for ax, lbl in zip(axes.flat, hac_labels):
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax.set_ylabel(lbl, fontsize=11)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=9)

    fig.suptitle('Hand Admittance Controller — AHE', fontsize=13)
    fig.tight_layout()
    fig.savefig('images/hac/hac_overview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Wrist force estimation — replicates plot_wrist_force.py
    # (the initial-transient skipping is removed: the whole signal is plotted)
    # Data read from `folder`; output saved in images/wrench_estimations/wrist_force/
    # and images/residuals/, consistently with the other plots of this script.
    # -----------------------------------------------------------------------
    if not os.path.exists('images/wrench_estimations/wrist_force'):
        os.makedirs('images/wrench_estimations/wrist_force')
    if not os.path.exists('images/residuals'):
        os.makedirs('images/residuals')

    WF_FREQ = 500
    WF_LABELS = ['x', 'y', 'z']
    WF_EST_COLORS = ['tab:blue', 'tab:orange', 'tab:green']
    WF_GT_COLORS = ['tab:cyan', 'tab:red', 'tab:olive']

    def _robust_loadtxt(path):
        # If the simulation is interrupted (Ctrl+C/crash) while a row is being
        # written, the log's last line ends up with fewer columns than the rest.
        # Fall back to dropping any malformed trailing row(s) instead of failing.
        try:
            return np.loadtxt(path)
        except ValueError:
            with open(path) as f:
                lines = f.readlines()
            ncols = len(lines[0].split())
            good_lines = [ln for ln in lines if len(ln.split()) == ncols]
            dropped = len(lines) - len(good_lines)
            if dropped:
                print(f"[WARN] {os.path.basename(path)}: dropped {dropped} malformed "
                      f"row(s) (likely truncated by an interrupted simulation).")
            return np.loadtxt(io.StringIO(''.join(good_lines)))

    def _wf_reshape(data, ncols):
        # A log holding a single row loads as 1-D, and so does a one-column
        # channel such as residual_norm.txt: the expected width tells them apart.
        if data.ndim == 1:
            return data.reshape(1, -1) if data.shape[0] == ncols else data.reshape(-1, 1)
        return data

    def _wf_load(filename, ncols, required=True):
        path = folder + '/' + filename
        if not os.path.exists(path):
            if required:
                # Left blank rather than fatal, so that the rest of the section
                # (and of the script) is still plotted.
                return _blank(filename, (num_samples - startPlot, ncols))
            print(f"[INFO] {filename} non trovato — ground truth non visualizzato.")
            return None
        return _fit(_wf_reshape(_robust_loadtxt(path), ncols), filename)

    def _wf_trim(a, b):
        n = min(len(a), len(b))
        return a[:n], b[:n]

    def _wf_save(fig, name, outdir='images/wrench_estimations/wrist_force'):
        out = os.path.join(outdir, name)
        fig.savefig(out, dpi=300, bbox_inches='tight')
        saved = _figure_has_data(fig)
        plt.close(fig)
        if saved:
            print(f"Saved: {out}")

    f_right = _wf_load('estimated_force_rwrist.txt', 3)
    f_left = _wf_load('estimated_force_lwrist.txt', 3)
    # The ground truth MuJoCo makes sense only for the simulation (expType == "Simulation",
    # i.e., folder == '/tmp'). For a real experiment, only the estimates are plotted.
    if expType == "Simulation":
        _gt_base = os.path.dirname(folder)
        def _load_gt(name):
            p = _gt_base + '/' + name
            if os.path.exists(p):
                return _fit(_wf_reshape(_robust_loadtxt(p), 3), name)
            print(f"[INFO] {name} non trovato in {_gt_base} — ground truth non visualizzato.")
            return None
        gt_right = _load_gt('gt_right_wrist.txt')
        gt_left = _load_gt('gt_left_wrist.txt')
    else:
        gt_right = None
        gt_left = None
        print("[INFO] Real experiment — ground truth not plotted, only estimates.")
    residual = _wf_load('residual_norm.txt', 1, required=False)

    # Allinea stima destra e sinistra
    Nwf = min(len(f_right), len(f_left))
    f_right, f_left = f_right[:Nwf], f_left[:Nwf]

    # Allinea ground truth con la rispettiva stima
    if gt_right is not None:
        f_right, gt_right = _wf_trim(f_right, gt_right)
        Nwf = len(f_right)
    if gt_left is not None:
        f_left, gt_left = _wf_trim(f_left, gt_left)
        Nwf = min(Nwf, len(f_left))

    f_right = f_right[:Nwf]
    f_left = f_left[:Nwf]

    # Nessun transitorio escluso: si plotta tutto il segnale
    sl_wf = slice(0, Nwf)
    t_wf = _time(np.arange(Nwf)[sl_wf] / WF_FREQ)

    print(f"Total samples (wrist force): {Nwf}  ({Nwf/WF_FREQ:.2f} s)")

    def _plot_wrist(f_est, gt, side_label, side_tag):
        has_gt = gt is not None

        # Componenti Fx / Fy / Fz
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        fig.suptitle(f'Force estimate — Wrist {side_label}', fontsize=13, fontweight='bold')

        for i, (ax, lbl, ec, gc) in enumerate(zip(axes, WF_LABELS, WF_EST_COLORS, WF_GT_COLORS)):
            ax.plot(t_wf, f_est[sl_wf, i], linewidth=1.5, color=ec,
                    label=rf'Estimated $F_{{{lbl}}}$')
            if has_gt:
                ax.plot(t_wf, gt[sl_wf, i], linewidth=1.5, color=gc, linestyle='--',
                        label=rf'Ground truth $F_{{{lbl}}}$')
            ax.axhline(0, color='k', linewidth=0.7, linestyle=':', alpha=0.5)
            ax.set_ylabel(rf'$F_{{{lbl}}}$ [N]', fontsize=11)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

        axes[-1].set_xlabel('Time [s]', fontsize=11)
        fig.tight_layout()
        _wf_save(fig, f'{side_tag}_components.png')

        # Norma ||F||
        norm_est = np.linalg.norm(f_est[sl_wf], axis=1)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t_wf, norm_est, linewidth=1.5, color=WF_EST_COLORS[0],
                label=r'$\|\hat{F}\|$ estimated')
        if has_gt:
            norm_gt = np.linalg.norm(gt[sl_wf], axis=1)
            ax.plot(t_wf, norm_gt, linewidth=1.5, color=WF_GT_COLORS[0], linestyle='--',
                    label=r'$\|F\|$ ground truth')
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'$\|F\|$ [N]', fontsize=11)
        ax.set_title(f'Force norm — Wrist {side_label}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        _wf_save(fig, f'{side_tag}_norm.png')

    _plot_wrist(f_right, gt_right, 'RIGHT', 'right_wrist')
    _plot_wrist(f_left, gt_left, 'LEFT', 'left_wrist')

    # Overview of both wrists: the two *_components figures side by side, left
    # wrist on the left column and right wrist on the right one, one row per
    # component. Each row shares its force axis, so the same component can be
    # read across the two arms.
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True, sharey='row')
    for col, (f_est, gt, side_label) in enumerate(zip(
        (f_left, f_right),
        (gt_left, gt_right),
        ('Left', 'Right'),
    )):
        for i, (lbl, ec, gc) in enumerate(zip(WF_LABELS, WF_EST_COLORS, WF_GT_COLORS)):
            ax = axes[i, col]
            ax.plot(t_wf, f_est[sl_wf, i], linewidth=1.5, color=ec,
                    label=rf'Estimated $F_{{{lbl}}}$')
            if gt is not None:
                ax.plot(t_wf, gt[sl_wf, i], linewidth=1.5, color=gc, linestyle='--',
                        label=rf'Ground truth $F_{{{lbl}}}$')
            ax.axhline(0, color='k', linewidth=0.7, linestyle=':', alpha=0.5)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            ax.tick_params(labelsize=9)
        axes[0, col].set_title(f'{side_label} Wrist', fontsize=12)
        axes[-1, col].set_xlabel('Time [s]', fontsize=11)
    # sharey='row' hides the right column's tick labels: label the left one only.
    for i, lbl in enumerate(WF_LABELS):
        axes[i, 0].set_ylabel(rf'$F_{{{lbl}}}$ [N]', fontsize=11)
    fig.suptitle('Force estimate — both wrists', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _wf_save(fig, 'wrist_forces_overview.png')

    # Confronto norma destro vs sinistro
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_wf, np.linalg.norm(f_right[sl_wf], axis=1), linewidth=1.5,
            color='tab:blue', label='Wrist right — estimated')
    ax.plot(t_wf, np.linalg.norm(f_left[sl_wf], axis=1), linewidth=1.5,
            color='tab:orange', linestyle='--', label='Wrist left — estimated')

    if gt_right is not None:
        ax.plot(t_wf, np.linalg.norm(gt_right[sl_wf], axis=1), linewidth=1.2,
                color='tab:cyan', linestyle=':', label='GT right')
    if gt_left is not None:
        ax.plot(t_wf, np.linalg.norm(gt_left[sl_wf], axis=1), linewidth=1.2,
                color='tab:red', linestyle=':', label='GT left')

    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(r'$\|F\|$', fontsize=11)
    ax.set_title('Comparison of norms: both wrists', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    _wf_save(fig, 'comparison_norm.png')

    # Norma del vettore residuo -> images/residuals
    if residual is not None:
        r = np.ravel(residual)
        Nr = len(r)
        sl_r = slice(0, Nr)
        t_r = _time(np.arange(Nr)[sl_r] / WF_FREQ)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t_r, r[sl_r], linewidth=1.5, color='tab:purple',
                label=r'$\|r\|$ residual vector norm')
        ax.axhline(0, color='k', linewidth=0.7, linestyle=':', alpha=0.5)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'$\|r\|$', fontsize=11)
        ax.set_title('Residual vector norm', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        _wf_save(fig, 'residual_norm.png', outdir='images/residuals')

        if np.isfinite(r[sl_r]).any():
            print(f"\n[RESIDUAL] samples: {int(np.isfinite(r[sl_r]).sum())} of {Nr}  "
                  f"mean: {_nan_mean(r[sl_r], axis=None):.4f}  "
                  f"max: {np.nanmax(r[sl_r]):.4f}")
    else:
        print("[INFO] residual_norm.txt not found — skipped residual plots.")

    # Error stats
    def _print_error_stats(f_est, gt, side_label):
        if gt is None:
            print(f"[{side_label}] Ground not available — no stats computed.")
            return
        err = f_est[sl_wf] - gt[sl_wf]
        if not np.isfinite(err).any():
            print(f"[{side_label}] estimate and ground truth never overlap — "
                  f"no stats computed.")
            return
        norm_err = np.linalg.norm(err, axis=1)
        print(f"\n{'='*50}")
        print(f"  Error — Wrist {side_label}")
        print(f"{'='*50}")
        print(f"  {'Axis':<6} {'Mean Error [N]':>20} {'Variance [N²]':>18}")
        print(f"  {'-'*46}")
        for i, lbl in enumerate(WF_LABELS):
            mean_i = _nan_mean(err[:, i], axis=None)
            var_i = np.nanvar(err[:, i])
            print(f"  F_{lbl:<4}  {mean_i:>20.4f} {var_i:>18.4f}")
        print(f"  {'-'*46}")
        print(f"  {'||err||':<6} {'Mean Error [N]':>20} {'Variance [N²]':>18}")
        print(f"  {'':6}  {_nan_mean(norm_err, axis=None):>20.4f} "
              f"{np.nanvar(norm_err):>18.4f}")
        print(f"{'='*50}")

    _print_error_stats(f_right, gt_right, 'RIGHT')
    _print_error_stats(f_left, gt_left, 'LEFT')

    ##########################
    #  ARM RESIDUALS PLOTS
    ##########################

    right_arm_joint_labels = [
        'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw',
        'r_elbow', 'r_wrist_roll', 'r_wrist_pitch', 'r_wrist_yaw'
    ]
    left_arm_joint_labels = [
        'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw',
        'l_elbow', 'l_wrist_roll', 'l_wrist_pitch', 'l_wrist_yaw'
    ]

    right_arm_residuals_path = folder + '/right_arm_residual.txt'
    left_arm_residuals_path  = folder + '/left_arm_residual.txt'

    if os.path.exists(right_arm_residuals_path):
        right_arm_res = _load_own(right_arm_residuals_path)
        Nr_arm = right_arm_res.shape[0]
        t_arm = _time(np.linspace(0.0, delta * Nr_arm, Nr_arm))
        n_joints_right = min(right_arm_res.shape[1], len(right_arm_joint_labels))
        for i in range(n_joints_right):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t_arm, right_arm_res[:, i], linewidth=1.5)
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel('Residual [Nm]', fontsize=11)
            ax.set_title(f'Right Arm Residual — {right_arm_joint_labels[i]}', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            fig.tight_layout()
            fig.savefig(f'images/residuals/right_arm/{right_arm_joint_labels[i]}_residual.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(n_joints_right):
            ax.plot(t_arm, right_arm_res[:, i], label=right_arm_joint_labels[i], linewidth=1.5)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel('Residual [Nm]', fontsize=11)
        ax.set_title('Right Arm — All Joint Residuals', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        fig.savefig('images/residuals/right_arm/all_joints_residual.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        _note_saved(f"[INFO] Right arm residual plots saved ({Nr_arm} samples).", right_arm_res)
    else:
        print("[INFO] right_arm_residuals.txt not found — skipped right arm residual plots.")

    if os.path.exists(left_arm_residuals_path):
        left_arm_res = _load_own(left_arm_residuals_path)
        Nl_arm = left_arm_res.shape[0]
        t_arm_l = _time(np.linspace(0.0, delta * Nl_arm, Nl_arm))
        n_joints_left = min(left_arm_res.shape[1], len(left_arm_joint_labels))
        for i in range(n_joints_left):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t_arm_l, left_arm_res[:, i], linewidth=1.5)
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel('Residual [Nm]', fontsize=11)
            ax.set_title(f'Left Arm Residual — {left_arm_joint_labels[i]}', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            fig.tight_layout()
            fig.savefig(f'images/residuals/left_arm/{left_arm_joint_labels[i]}_residual.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(n_joints_left):
            ax.plot(t_arm_l, left_arm_res[:, i], label=left_arm_joint_labels[i], linewidth=1.5)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel('Residual [Nm]', fontsize=11)
        ax.set_title('Left Arm — All Joint Residuals', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        fig.savefig('images/residuals/left_arm/all_joints_residual.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        _note_saved(f"[INFO] Left arm residual plots saved ({Nl_arm} samples).", left_arm_res)
    else:
        print("[INFO] left_arm_residuals.txt not found — skipped left arm residual plots.")

    ##########################
    #  BASE / LEGS / WAIST RESIDUALS PLOTS
    ##########################

    def _plot_residual_group(path, joint_labels, outdir, group_title, ylabel='Residual [Nm]'):
        if not os.path.exists(path):
            print(f"[INFO] {os.path.basename(path)} not found — skipped {group_title} residual plots.")
            return
        res = _load_own(path)
        N = res.shape[0]
        t = _time(np.linspace(0.0, delta * N, N))
        n_joints = min(res.shape[1], len(joint_labels))
        for i in range(n_joints):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t, res[:, i], linewidth=1.5)
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{group_title} Residual — {joint_labels[i]}', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            fig.tight_layout()
            fig.savefig(f'{outdir}/{joint_labels[i]}_residual.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(n_joints):
            ax.plot(t, res[:, i], label=joint_labels[i], linewidth=1.5)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{group_title} — All DOF Residuals', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        fig.savefig(f'{outdir}/all_joints_residual.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        _note_saved(f"[INFO] {group_title} residual plots saved ({N} samples).", res)

    _plot_residual_group(
        folder + '/base_residual.txt',
        ['base_lin_x', 'base_lin_y', 'base_lin_z', 'base_ang_x', 'base_ang_y', 'base_ang_z'],
        'images/residuals/base', 'Base', ylabel='Residual [N, Nm]')

    _plot_residual_group(
        folder + '/right_leg_residual.txt',
        ['r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'],
        'images/residuals/right_leg', 'Right Leg')

    _plot_residual_group(
        folder + '/left_leg_residual.txt',
        ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll'],
        'images/residuals/left_leg', 'Left Leg')

    _plot_residual_group(
        folder + '/waist_residual.txt',
        ['waist_yaw', 'waist_roll', 'waist_pitch'],
        'images/residuals/waist', 'Waist')

    ##########################
    #  GENERALIZED MOMENTUM
    ##########################

    gm_path  = folder + '/generalized_momentum.txt'
    gm0_path = folder + '/initialized_generalized_momentum.txt'

    if os.path.exists(gm_path) and os.path.exists(gm0_path):
        p_data  = _load_own(gm_path)
        p0_data = _load_own(gm0_path)

        N_gm  = p_data.shape[0]
        n_dof = p_data.shape[1]
        t_gm  = _time(np.linspace(0.0, delta * N_gm, N_gm))

        # DOF labels: first 6 are floating base, rest are joints
        base_labels = ['base_vx', 'base_vy', 'base_vz', 'base_wx', 'base_wy', 'base_wz']
        joint_labels_gm = [jn.strip() for jn in joint_names]  # from joint_names.txt loaded earlier
        dof_labels = base_labels + joint_labels_gm
        dof_labels = dof_labels[:n_dof]  # trim if needed

        if not os.path.exists('images/generalized_momentum'):
            os.makedirs('images/generalized_momentum')
        if not os.path.exists('images/generalized_momentum/dofs'):
            os.makedirs('images/generalized_momentum/dofs')

        cmap_gm = plt.colormaps['tab20']

        # -- Separate: p_ overview (all DOFs) --
        n_cols_gm = 5
        n_rows_gm = int(np.ceil(n_dof / n_cols_gm))
        fig_p, axes_p = plt.subplots(n_rows_gm, n_cols_gm,
                                      figsize=(4 * n_cols_gm, 3 * n_rows_gm))
        axes_p = np.array(axes_p).flatten()
        for i in range(n_dof):
            ax = axes_p[i]
            ax.plot(t_gm, p_data[:, i], color=cmap_gm(i % 20), linewidth=1.2)
            ax.set_title(dof_labels[i] if i < len(dof_labels) else f'DOF {i}', fontsize=7)
            ax.set_xlabel('t [s]', fontsize=6)
            ax.set_ylabel('p [kg·m²/s]', fontsize=6)
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        for j in range(n_dof, len(axes_p)):
            axes_p[j].set_visible(False)
        fig_p.suptitle('Generalized Momentum p(t) — all DOFs', fontsize=13)
        fig_p.tight_layout()
        fig_p.savefig('images/generalized_momentum/p_all_dofs.png', dpi=150, bbox_inches='tight')
        plt.close(fig_p)

        # -- Separate: p0_ overview (all DOFs) --
        fig_p0, axes_p0 = plt.subplots(n_rows_gm, n_cols_gm,
                                        figsize=(4 * n_cols_gm, 3 * n_rows_gm))
        axes_p0 = np.array(axes_p0).flatten()
        for i in range(n_dof):
            ax = axes_p0[i]
            ax.plot(t_gm, p0_data[:, i], color=cmap_gm(i % 20), linewidth=1.2, linestyle='--')
            ax.set_title(dof_labels[i] if i < len(dof_labels) else f'DOF {i}', fontsize=7)
            ax.set_xlabel('t [s]', fontsize=6)
            ax.set_ylabel('p₀ [kg·m²/s]', fontsize=6)
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        for j in range(n_dof, len(axes_p0)):
            axes_p0[j].set_visible(False)
        fig_p0.suptitle('Initial Generalized Momentum p₀ — all DOFs', fontsize=13)
        fig_p0.tight_layout()
        fig_p0.savefig('images/generalized_momentum/p0_all_dofs.png', dpi=150, bbox_inches='tight')
        plt.close(fig_p0)

        # -- Insieme: p_ and p0_ overlaid per DOF (subplots) --
        fig_cmp, axes_cmp = plt.subplots(n_rows_gm, n_cols_gm,
                                          figsize=(4 * n_cols_gm, 3 * n_rows_gm))
        axes_cmp = np.array(axes_cmp).flatten()
        for i in range(n_dof):
            ax = axes_cmp[i]
            color = cmap_gm(i % 20)
            ax.plot(t_gm, p_data[:, i],  color=color, linewidth=1.2, label='p')
            ax.plot(t_gm, p0_data[:, i], color=color, linewidth=1.2, linestyle='--', label='p₀')
            ax.set_title(dof_labels[i] if i < len(dof_labels) else f'DOF {i}', fontsize=7)
            ax.set_xlabel('t [s]', fontsize=6)
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
            if i == 0:
                ax.legend(fontsize=6)
        for j in range(n_dof, len(axes_cmp)):
            axes_cmp[j].set_visible(False)
        fig_cmp.suptitle('Generalized Momentum p(t) vs p₀ — all DOFs', fontsize=13)
        fig_cmp.tight_layout()
        fig_cmp.savefig('images/generalized_momentum/p_vs_p0_all_dofs.png', dpi=150, bbox_inches='tight')
        plt.close(fig_cmp)

        # -- Per-DOF separate PNGs (p_ and p0_ together per DOF) --
        for i in range(n_dof):
            label = dof_labels[i] if i < len(dof_labels) else f'dof_{i}'
            safe_label = label.replace('/', '_').replace(' ', '_')
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(t_gm, p_data[:, i],  linewidth=1.5, label='p(t)')
            ax.plot(t_gm, p0_data[:, i], linewidth=1.5, linestyle='--', label='p₀')
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel('[kg·m²/s]', fontsize=11)
            ax.set_title(f'Generalized Momentum — {label}', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
            fig.tight_layout()
            fig.savefig(f'images/generalized_momentum/dofs/{i:02d}_{safe_label}.png',
                        dpi=120, bbox_inches='tight')
            plt.close(fig)

        _note_saved(f"[INFO] Generalized momentum plots saved ({N_gm} samples, {n_dof} DOFs).", p_data, p0_data)
    else:
        print("[INFO] generalized_momentum.txt or initial_generalized_momentum.txt not found — skipped.")

    ##########################
    #  TAU_M - G  ARM PLOTS
    ##########################

    right_arm_joint_labels = [
        'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw',
        'r_elbow', 'r_wrist_roll', 'r_wrist_pitch', 'r_wrist_yaw'
    ]
    left_arm_joint_labels = [
        'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw',
        'l_elbow', 'l_wrist_roll', 'l_wrist_pitch', 'l_wrist_yaw'
    ]

    right_arm_tau_g_path = folder + '/right_arm_tau_g.txt'
    left_arm_tau_g_path  = folder + '/left_arm_tau_g.txt'

    if os.path.exists(right_arm_tau_g_path):
        right_arm_tg = _load_own(right_arm_tau_g_path)
        Nr_tg = right_arm_tg.shape[0]
        t_tg = _time(np.linspace(0.0, delta * Nr_tg, Nr_tg))
        n_joints_right_tg = min(right_arm_tg.shape[1], len(right_arm_joint_labels))
        for i in range(n_joints_right_tg):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t_tg, right_arm_tg[:, i], linewidth=1.5)
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel(r'$\tau_m - g$ [Nm]', fontsize=11)
            ax.set_title(fr'Right Arm $\tau_m - g$ — {right_arm_joint_labels[i]}', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            fig.tight_layout()
            fig.savefig(f'images/tau_g/right_arm/{right_arm_joint_labels[i]}_tau_g.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(n_joints_right_tg):
            ax.plot(t_tg, right_arm_tg[:, i], label=right_arm_joint_labels[i], linewidth=1.5)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'$\tau_m - g$ [Nm]', fontsize=11)
        ax.set_title(r'Right Arm — $\tau_m - g$ All Joints', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        fig.savefig('images/tau_g/right_arm/all_joints_tau_g.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        _note_saved(f"[INFO] Right arm tau_m-g plots saved ({Nr_tg} samples).", right_arm_tg)
    else:
        print("[INFO] right_arm_tau_g.txt not found — skipped right arm tau_m-g plots.")

    if os.path.exists(left_arm_tau_g_path):
        left_arm_tg = _load_own(left_arm_tau_g_path)
        Nl_tg = left_arm_tg.shape[0]
        t_tg_l = _time(np.linspace(0.0, delta * Nl_tg, Nl_tg))
        n_joints_left_tg = min(left_arm_tg.shape[1], len(left_arm_joint_labels))
        for i in range(n_joints_left_tg):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t_tg_l, left_arm_tg[:, i], linewidth=1.5)
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel(r'$\tau_m - g$ [Nm]', fontsize=11)
            ax.set_title(fr'Left Arm $\tau_m - g$ — {left_arm_joint_labels[i]}', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            fig.tight_layout()
            fig.savefig(f'images/tau_g/left_arm/{left_arm_joint_labels[i]}_tau_g.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i in range(n_joints_left_tg):
            ax.plot(t_tg_l, left_arm_tg[:, i], label=left_arm_joint_labels[i], linewidth=1.5)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(r'$\tau_m - g$ [Nm]', fontsize=11)
        ax.set_title(r'Left Arm — $\tau_m - g$ All Joints', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        fig.savefig('images/tau_g/left_arm/all_joints_tau_g.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        _note_saved(f"[INFO] Left arm tau_m-g plots saved ({Nl_tg} samples).", left_arm_tg)
    else:
        print("[INFO] left_arm_tau_g.txt not found — skipped left arm tau_m-g plots.")

