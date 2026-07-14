import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict
for _np_alias, _np_type in {
    'bool': bool,
    'int': int,
    'float': float,
    'complex': complex,
    'object': object,
    'str': str,
}.items():
    if _np_alias not in np.__dict__:
        setattr(np, _np_alias, _np_type)
import scipy.spatial.transform
from math import ceil, floor, sqrt
from collections import defaultdict
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
import argparse
import os
from pathlib import Path

try:
    from .crg_wbc_plotting import (
        folder_has_compliance_logs,
        generate_compliance_batch,
        generate_compliance_evidence,
        parse_formats,
    )
except ImportError:
    from crg_wbc_plotting import (
        folder_has_compliance_logs,
        generate_compliance_batch,
        generate_compliance_evidence,
        parse_formats,
    )


def ensure_2d(data: np.ndarray, expected_cols: int) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        if expected_cols == 1:
            return arr.reshape(-1, 1)
        return arr.reshape(1, -1)
    return arr


def is_nonempty_file(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def load_optional_matrix(path: str):
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) == 0:
        return None
    try:
        data = np.loadtxt(path)
    except (ValueError, OSError):
        return None
    if np.size(data) == 0:
        return None
    return np.atleast_2d(data)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot walking-controller logs and generate a timestamp-aligned "
            "CRG/WBC compliance evidence suite."
        )
    )
    parser.add_argument(
        "--folder",
        help=(
            "Log directory. Without this option the original interactive "
            "experiment-number prompt is retained."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "CRG/WBC figure directory. With --folder the default is "
            "<folder>/plots/crg_wbc; legacy plots keep their existing images/ paths."
        ),
    )
    parser.add_argument(
        "--crg-wbc-only",
        "--compliance-only",
        dest="compliance_only",
        action="store_true",
        help="Generate only the CRG/WBC evidence suite; no legacy logs are required.",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="LOG_DIR",
        help=(
            "Generate one evidence suite per directory plus a quantitative "
            "multi-run comparison. Unquoted shell globs are supported."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels for --compare directories, in the same order.",
    )
    parser.add_argument(
        "--case-label",
        help="Optional title label for a single --folder run.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated CRG/WBC figure formats: png,pdf,svg (default: png,pdf).",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="Optional displayed time range in seconds; metrics still use full logs.",
    )
    parser.add_argument(
        "--force-window",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="Override the force-window inferred from applied-wrench enable flags.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help=(
            "Maximum displayed samples per line; 0 keeps every sample. Full data "
            "are always used for metrics (default: 0)."
        ),
    )
    parser.add_argument(
        "--strict-compliance",
        action="store_true",
        help="Fail if any required CRG/WBC evidence log or named column is missing.",
    )
    return parser

if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.compare and args.folder:
        parser.error('--compare and --folder are mutually exclusive')
    if args.compare and args.case_label:
        parser.error('--case-label is only valid for a single run')
    if args.compare and args.compliance_only:
        parser.error('--compare already selects CRG/WBC-only batch mode')
    if args.labels and not args.compare:
        parser.error('--labels requires --compare')
    try:
        compliance_formats = parse_formats(args.formats)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_points < 0:
        parser.error('--max-points must be non-negative')
    if args.time_range is not None and (
        not np.all(np.isfinite(args.time_range))
        or args.time_range[1] <= args.time_range[0]
    ):
        parser.error('--time-range must be finite and END must be greater than START')
    if args.force_window is not None and (
        not np.all(np.isfinite(args.force_window))
        or args.force_window[1] <= args.force_window[0]
    ):
        parser.error('--force-window must be finite and END must be greater than START')

    if args.compare:
        comparison_output = args.output_dir or 'images/crg_wbc_comparison'
        try:
            generate_compliance_batch(
                args.compare,
                comparison_output,
                labels=args.labels,
                formats=compliance_formats,
                max_points=args.max_points,
                time_range=args.time_range,
                force_window=args.force_window,
                strict=args.strict_compliance,
            )
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
        raise SystemExit(0)

    explicit_folder = args.folder is not None
    if explicit_folder:
        folder = str(Path(args.folder).expanduser().resolve())
    else:
        # Preserve the original interactive entry point.
        number = input("Enter 0 to plot data from the last simulation or the number of the experiment: ")
        if number == '0':
            folder = '/tmp'
        else:
            folder = 'experiments/experiment_' + number

    has_crg_wbc_logs = folder_has_compliance_logs(folder)
    if has_crg_wbc_logs:
        if args.output_dir:
            compliance_output = args.output_dir
        elif explicit_folder:
            compliance_output = str(Path(folder) / 'plots' / 'crg_wbc')
        else:
            compliance_output = 'images/crg_wbc'
        try:
            generate_compliance_evidence(
                folder,
                compliance_output,
                label=args.case_label,
                formats=compliance_formats,
                max_points=args.max_points,
                time_range=args.time_range,
                force_window=args.force_window,
                strict=args.strict_compliance,
            )
        except (OSError, ValueError, KeyError) as exc:
            if args.compliance_only or args.strict_compliance:
                parser.error(str(exc))
            print(f'[WARN] Skip CRG/WBC evidence suite: {exc}')
            has_crg_wbc_logs = False
    elif args.compliance_only or args.strict_compliance:
        parser.error(
            f"{folder} does not contain a readable compliance_torso_state.txt"
        )

    if args.compliance_only:
        raise SystemExit(0)

    # Archived CRG/WBC snapshots intentionally contain only evidence logs.
    # In that case --folder is useful without requiring an extra mode flag.
    legacy_required = (
        'joint_names.txt', 'input_torque.txt',
        'sim_com_position.txt', 'sim_com_velocity.txt', 'sim_zmp_position.txt',
        'fb_com_position.txt', 'fb_com_velocity.txt', 'fb_zmp_position.txt',
        'kf_com_position.txt', 'kf_com_velocity.txt', 'kf_zmp_position.txt',
        'des_com_position.txt', 'des_com_velocity.txt', 'des_zmp_position.txt',
        'ef_zmp_position.txt',
        'p_lsole_sim.txt', 'p_rsole_sim.txt', 'v_lsole_sim.txt', 'v_rsole_sim.txt',
        'p_lsole_fb.txt', 'p_rsole_fb.txt', 'v_lsole_fb.txt', 'v_rsole_fb.txt',
        'p_lsole_des.txt', 'p_rsole_des.txt', 'v_lsole_des.txt', 'v_rsole_des.txt',
        'estimated_force_lsole.txt', 'estimated_force_rsole.txt',
        'ekf_base_position.txt', 'ekf_base_velocity.txt',
        'ekf_base_orientation.txt', 'ekf_base_angular_velocity.txt',
        'ekf_joint_position.txt', 'ekf_joint_velocity.txt',
        'sim_base_position.txt', 'sim_base_velocity.txt',
        'sim_base_orientation.txt', 'sim_base_angular_velocity.txt',
        'sim_joint_position.txt', 'sim_joint_velocity.txt',
        'execution_time_ekf.txt', 'execution_time_kf.txt',
        'execution_time_mpc.txt', 'execution_time_wbc.txt',
        'execution_time_update.txt',
        'measured_joint_position.txt', 'measured_joint_velocity.txt',
        'measured_imu_orientation.txt', 'measured_imu_angular_velocity.txt',
        'measured_imu_accelerometer.txt',
    )
    if explicit_folder:
        missing_legacy = [
            name for name in legacy_required
            if not is_nonempty_file(os.path.join(folder, name))
        ]
        if missing_legacy and has_crg_wbc_logs:
            print(
                '[INFO] Legacy walking logs are incomplete; all available '
                'CRG/WBC evidence has been generated.'
            )
            raise SystemExit(0)
        if missing_legacy:
            preview = ', '.join(missing_legacy[:6])
            suffix = ' ...' if len(missing_legacy) > 6 else ''
            parser.error(f'Legacy log set is incomplete; missing: {preview}{suffix}')
    
    joint_names = open(folder + '/joint_names.txt').readlines()
    input_torque: np.ndarray = np.loadtxt(folder +'/input_torque.txt')
    sim_com_position =  np.loadtxt(folder + '/sim_com_position.txt')
    sim_com_velocity =  np.loadtxt(folder + '/sim_com_velocity.txt')
    sim_zmp_position =  np.loadtxt(folder + '/sim_zmp_position.txt')
    fb_com_position = np.loadtxt(folder + '/fb_com_position.txt')
    fb_com_velocity = np.loadtxt(folder + '/fb_com_velocity.txt')
    fb_zmp_position = np.loadtxt(folder + '/fb_zmp_position.txt')
    kf_com_position =  np.loadtxt(folder + '/kf_com_position.txt')
    kf_com_velocity =  np.loadtxt(folder + '/kf_com_velocity.txt')
    kf_zmp_position =  np.loadtxt(folder + '/kf_zmp_position.txt')
    des_com_position = np.loadtxt(folder + '/des_com_position.txt')
    des_com_velocity = np.loadtxt(folder + '/des_com_velocity.txt')
    des_zmp_position = np.loadtxt(folder + '/des_zmp_position.txt')

    ef_zmp_position = np.loadtxt(folder + '/ef_zmp_position.txt')

    # base_estimate = np.loadtxt(folder + '/base_estimate.txt')
    # orientation_estimate = np.loadtxt(folder + '/orientation_estimate.txt')
    # left_foot_position_base_estimation = np.loadtxt(folder + '/left_foot_position_base_estimation.txt')
    # right_foot_position_base_estimation = np.loadtxt(folder + '/right_foot_position_base_estimation.txt')
    # left_foot_position_with_zero_base = np.loadtxt(folder + '/left_foot_position_with_zero_base.txt')
    # right_foot_position_with_zero_base = np.loadtxt(folder + '/right_foot_position_with_zero_base.txt')

    p_lsole_sim = np.loadtxt(folder + '/p_lsole_sim.txt')
    p_rsole_sim = np.loadtxt(folder + '/p_rsole_sim.txt')
    v_lsole_sim = np.loadtxt(folder + '/v_lsole_sim.txt')
    v_rsole_sim = np.loadtxt(folder + '/v_rsole_sim.txt')
    p_lsole_fb = np.loadtxt(folder + '/p_lsole_fb.txt')
    p_rsole_fb = np.loadtxt(folder + '/p_rsole_fb.txt')
    v_lsole_fb = np.loadtxt(folder + '/v_lsole_fb.txt')
    v_rsole_fb = np.loadtxt(folder + '/v_rsole_fb.txt')
    p_lsole_des = np.loadtxt(folder + '/p_lsole_des.txt')
    p_rsole_des = np.loadtxt(folder + '/p_rsole_des.txt')
    v_lsole_des = np.loadtxt(folder + '/v_lsole_des.txt')
    v_rsole_des = np.loadtxt(folder + '/v_rsole_des.txt')

    estimated_force_lsole = np.loadtxt(folder + '/estimated_force_lsole.txt')
    estimated_force_rsole = np.loadtxt(folder + '/estimated_force_rsole.txt')
    has_wrist_force_logs = (
        os.path.exists(folder + '/estimated_force_left_wrist.txt') and
        os.path.exists(folder + '/estimated_force_right_wrist.txt') and
        os.path.exists(folder + '/estimated_force_left_wrist_filtered.txt') and
        os.path.exists(folder + '/estimated_force_right_wrist_filtered.txt')
    )
    if has_wrist_force_logs:
        estimated_force_left_wrist = np.loadtxt(folder + '/estimated_force_left_wrist.txt')
        estimated_force_right_wrist = np.loadtxt(folder + '/estimated_force_right_wrist.txt')
        estimated_force_left_wrist_filtered = np.loadtxt(folder + '/estimated_force_left_wrist_filtered.txt')
        estimated_force_right_wrist_filtered = np.loadtxt(folder + '/estimated_force_right_wrist_filtered.txt')

    applied_external_force_file = folder + '/applied_external_wrist_force.txt'
    has_applied_wrist_force_logs = os.path.exists(applied_external_force_file)
    if has_applied_wrist_force_logs:
        applied_wrist_force = np.loadtxt(applied_external_force_file, skiprows=1)
        applied_wrist_force = np.atleast_2d(applied_wrist_force)

    compliance_hand_state_file = folder + '/compliance_hand_state.txt'
    has_compliance_hand_logs = os.path.exists(compliance_hand_state_file)
    if has_compliance_hand_logs:
        try:
            compliance_hand_state = np.loadtxt(compliance_hand_state_file, skiprows=1)
            compliance_hand_state = np.atleast_2d(compliance_hand_state)
            if compliance_hand_state.shape[1] < 37:
                has_compliance_hand_logs = False
                print('[INFO] compliance_hand_state.txt has fewer than 37 columns, skip compliance plots.')
        except (ValueError, OSError):
            has_compliance_hand_logs = False
            print('[INFO] compliance_hand_state.txt is unreadable, skip compliance plots.')

    compliance_torso_state_file = folder + '/compliance_torso_state.txt'
    has_compliance_torso_logs = os.path.exists(compliance_torso_state_file)
    if has_compliance_torso_logs:
        try:
            compliance_torso_state = np.loadtxt(compliance_torso_state_file, skiprows=1)
            compliance_torso_state = np.atleast_2d(compliance_torso_state)
            if compliance_torso_state.shape[1] < 68:
                has_compliance_torso_logs = False
                print('[INFO] compliance_torso_state.txt has fewer than 68 columns, skip torso compliance plots.')
        except (ValueError, OSError):
            has_compliance_torso_logs = False
            print('[INFO] compliance_torso_state.txt is unreadable, skip torso compliance plots.')

    wbc_torso_tracking_file = folder + '/wbc_torso_orientation_tracking.txt'
    has_wbc_torso_tracking_logs = os.path.exists(wbc_torso_tracking_file)
    if has_wbc_torso_tracking_logs:
        try:
            wbc_torso_tracking = np.loadtxt(wbc_torso_tracking_file, skiprows=1)
            wbc_torso_tracking = np.atleast_2d(wbc_torso_tracking)
            if wbc_torso_tracking.shape[1] < 16:
                has_wbc_torso_tracking_logs = False
                print('[INFO] wbc_torso_orientation_tracking.txt has fewer than 16 columns, skip WBC torso tracking plots.')
        except (ValueError, OSError):
            has_wbc_torso_tracking_logs = False
            print('[INFO] wbc_torso_orientation_tracking.txt is unreadable, skip WBC torso tracking plots.')

    all_joint_motor_names_file = folder + '/all_joint_motor_names.txt'
    all_joint_motor_time_file = folder + '/all_joint_motor_time.txt'
    all_joint_motor_q_file = folder + '/all_joint_motor_q.txt'
    all_joint_motor_dq_file = folder + '/all_joint_motor_dq.txt'
    all_joint_motor_ddq_file = folder + '/all_joint_motor_ddq.txt'
    all_joint_motor_tau_est_file = folder + '/all_joint_motor_tau_est.txt'
    all_joint_motor_tau_applied_file = folder + '/all_joint_motor_tau_applied.txt'
    has_all_joint_motor_tau_applied = os.path.exists(all_joint_motor_tau_applied_file)
    has_all_joint_motor_logs = (
        os.path.exists(all_joint_motor_names_file) and
        os.path.exists(all_joint_motor_time_file) and
        os.path.exists(all_joint_motor_q_file) and
        os.path.exists(all_joint_motor_dq_file) and
        os.path.exists(all_joint_motor_ddq_file) and
        os.path.exists(all_joint_motor_tau_est_file)
    )
    motor_plot_joint_count = 0
    motor_plot_sample_count = 0
    if has_all_joint_motor_logs:
        all_joint_motor_names = [line.strip() for line in open(all_joint_motor_names_file).readlines() if line.strip()]
        n_motor_joints = len(all_joint_motor_names)
        motor_files_have_data = (
            n_motor_joints > 0 and
            is_nonempty_file(all_joint_motor_time_file) and
            is_nonempty_file(all_joint_motor_q_file) and
            is_nonempty_file(all_joint_motor_dq_file) and
            is_nonempty_file(all_joint_motor_ddq_file) and
            is_nonempty_file(all_joint_motor_tau_est_file)
        )

        if not motor_files_have_data:
            has_all_joint_motor_logs = False
            print('[INFO] all_joint_motor_* logs are present but empty/incomplete, skip motor-state plots.')
        else:
            all_joint_motor_time = np.atleast_1d(np.loadtxt(all_joint_motor_time_file))

            all_joint_motor_q = ensure_2d(np.loadtxt(all_joint_motor_q_file), n_motor_joints)
            all_joint_motor_dq = ensure_2d(np.loadtxt(all_joint_motor_dq_file), n_motor_joints)
            all_joint_motor_ddq = ensure_2d(np.loadtxt(all_joint_motor_ddq_file), n_motor_joints)
            all_joint_motor_tau_est = ensure_2d(np.loadtxt(all_joint_motor_tau_est_file), n_motor_joints)
            if has_all_joint_motor_tau_applied and is_nonempty_file(all_joint_motor_tau_applied_file):
                all_joint_motor_tau_applied = ensure_2d(np.loadtxt(all_joint_motor_tau_applied_file), n_motor_joints)
            else:
                has_all_joint_motor_tau_applied = False

            num_motor_samples = min(
                all_joint_motor_time.shape[0],
                all_joint_motor_q.shape[0],
                all_joint_motor_dq.shape[0],
                all_joint_motor_ddq.shape[0],
                all_joint_motor_tau_est.shape[0],
            )
            if has_all_joint_motor_tau_applied:
                num_motor_samples = min(num_motor_samples, all_joint_motor_tau_applied.shape[0])
            motor_plot_joint_count = min(
                n_motor_joints,
                all_joint_motor_q.shape[1],
                all_joint_motor_dq.shape[1],
                all_joint_motor_ddq.shape[1],
                all_joint_motor_tau_est.shape[1],
            )
            if has_all_joint_motor_tau_applied:
                motor_plot_joint_count = min(motor_plot_joint_count, all_joint_motor_tau_applied.shape[1])
            motor_plot_sample_count = num_motor_samples

            if num_motor_samples <= 0 or motor_plot_joint_count <= 0:
                has_all_joint_motor_logs = False
                print('[INFO] all_joint_motor_* logs have zero valid samples/columns, skip motor-state plots.')
            else:
                all_joint_motor_time = all_joint_motor_time[:num_motor_samples]
                all_joint_motor_q = all_joint_motor_q[:num_motor_samples, :motor_plot_joint_count]
                all_joint_motor_dq = all_joint_motor_dq[:num_motor_samples, :motor_plot_joint_count]
                all_joint_motor_ddq = all_joint_motor_ddq[:num_motor_samples, :motor_plot_joint_count]
                all_joint_motor_tau_est = all_joint_motor_tau_est[:num_motor_samples, :motor_plot_joint_count]
                if has_all_joint_motor_tau_applied:
                    all_joint_motor_tau_applied = all_joint_motor_tau_applied[:num_motor_samples, :motor_plot_joint_count]
                all_joint_motor_names = all_joint_motor_names[:motor_plot_joint_count]

    ekf_base_position = np.loadtxt(folder + '/ekf_base_position.txt')
    ekf_base_velocity = np.loadtxt(folder + '/ekf_base_velocity.txt')
    ekf_base_orientation = np.loadtxt(folder + '/ekf_base_orientation.txt')
    ekf_base_angular_velocity = np.loadtxt(folder + '/ekf_base_angular_velocity.txt')
    ekf_joint_position = np.loadtxt(folder + '/ekf_joint_position.txt')
    ekf_joint_velocity = np.loadtxt(folder + '/ekf_joint_velocity.txt')
    sim_base_position = np.loadtxt(folder + '/sim_base_position.txt')
    sim_base_velocity = np.loadtxt(folder + '/sim_base_velocity.txt')
    sim_base_orientation = np.loadtxt(folder + '/sim_base_orientation.txt')
    sim_base_angular_velocity = np.loadtxt(folder + '/sim_base_angular_velocity.txt')
    sim_joint_position: np.ndarray = np.loadtxt(folder + '/sim_joint_position.txt')
    sim_joint_velocity: np.ndarray = np.loadtxt(folder + '/sim_joint_velocity.txt')

    execution_time_ekf = np.loadtxt(folder + '/execution_time_ekf.txt')
    execution_time_kf = np.loadtxt(folder + '/execution_time_kf.txt')
    execution_time_mpc = np.loadtxt(folder + '/execution_time_mpc.txt')
    execution_time_wbc = np.loadtxt(folder + '/execution_time_wbc.txt')
    execution_time_update = np.loadtxt(folder + '/execution_time_update.txt')

    measured_joint_position: np.ndarray = np.loadtxt(folder +'/measured_joint_position.txt')
    measured_joint_velocity: np.ndarray = np.loadtxt(folder +'/measured_joint_velocity.txt')
    measured_imu_orientation: np.ndarray = np.loadtxt(folder + '/measured_imu_orientation.txt')
    measured_imu_angular_velocity: np.ndarray = np.loadtxt(folder + '/measured_imu_angular_velocity.txt')
    measured_imu_accelerometer: np.ndarray = np.loadtxt(folder + '/measured_imu_accelerometer.txt')
    # estimated_imu_accelerometer: np.ndarray = np.loadtxt(folder + '/estimated_imu_accelerometer.txt')
    # estimated_imu_angular_velocity: np.ndarray = np.loadtxt(folder + '/estimated_imu_angular_velocity.txt')
    # estimated_imu_orientation: np.ndarray = np.loadtxt(folder + '/estimated_imu_orientation.txt')


    num_samples = sim_joint_position.shape[0] - 10
    input_torque = input_torque[:num_samples, :]

    sim_com_position = sim_com_position[:num_samples, :]
    sim_com_velocity = sim_com_velocity[:num_samples, :]
    sim_zmp_position = sim_zmp_position[:num_samples, :]
    fb_com_position = fb_com_position[:num_samples, :]
    fb_com_velocity = fb_com_velocity[:num_samples, :]
    fb_zmp_position = fb_zmp_position[:num_samples, :]
    kf_com_position = kf_com_position[:num_samples, :]
    kf_com_velocity = kf_com_velocity[:num_samples, :]
    kf_zmp_position = kf_zmp_position[:num_samples, :]
    des_com_position = des_com_position[:num_samples, :]
    des_com_velocity = des_com_velocity[:num_samples, :]
    des_zmp_position = des_zmp_position[:num_samples, :]

    ef_zmp_position = ef_zmp_position[:num_samples, :]

    # base_estimate = base_estimate[:num_samples, :]
    # orientation_estimate = orientation_estimate[:num_samples, :]
    # left_foot_position_base_estimation = left_foot_position_base_estimation[:num_samples, :]
    # right_foot_position_base_estimation = right_foot_position_base_estimation[:num_samples, :]
    # left_foot_position_with_zero_base = left_foot_position_with_zero_base[:num_samples, :]
    # right_foot_position_with_zero_base = right_foot_position_with_zero_base[:num_samples, :]

    p_lsole_sim = p_lsole_sim[:num_samples, :]
    p_rsole_sim = p_rsole_sim[:num_samples, :]
    v_lsole_sim = v_lsole_sim[:num_samples, :]
    v_rsole_sim = v_rsole_sim[:num_samples, :]
    p_lsole_fb = p_lsole_fb[:num_samples, :]
    p_rsole_fb = p_rsole_fb[:num_samples, :]
    v_lsole_fb = v_lsole_fb[:num_samples, :]
    v_rsole_fb = v_rsole_fb[:num_samples, :]
    p_lsole_des = p_lsole_des[:num_samples, :]
    p_rsole_des = p_rsole_des[:num_samples, :]
    v_lsole_des = v_lsole_des[:num_samples, :]
    v_rsole_des = v_rsole_des[:num_samples, :]

    estimated_force_lsole = estimated_force_lsole[:num_samples, :]
    estimated_force_rsole = estimated_force_rsole[:num_samples, :]
    if has_wrist_force_logs:
        estimated_force_left_wrist = estimated_force_left_wrist[:num_samples, :]
        estimated_force_right_wrist = estimated_force_right_wrist[:num_samples, :]
        estimated_force_left_wrist_filtered = estimated_force_left_wrist_filtered[:num_samples, :]
        estimated_force_right_wrist_filtered = estimated_force_right_wrist_filtered[:num_samples, :]
    if has_applied_wrist_force_logs:
        applied_wrist_force = applied_wrist_force[:num_samples, :]
    if has_compliance_hand_logs:
        compliance_hand_state = compliance_hand_state[:num_samples, :]
    if has_compliance_torso_logs:
        compliance_torso_state = compliance_torso_state[:min(num_samples, compliance_torso_state.shape[0]), :]
    if has_wbc_torso_tracking_logs:
        wbc_torso_tracking = wbc_torso_tracking[:min(num_samples, wbc_torso_tracking.shape[0]), :]
    
    ekf_base_position = ekf_base_position[:num_samples, :]
    ekf_base_velocity = ekf_base_velocity[:num_samples, :]
    ekf_base_orientation = ekf_base_orientation[:num_samples, :]
    ekf_base_angular_velocity = ekf_base_angular_velocity[:num_samples, :]
    ekf_joint_position = ekf_joint_position[:num_samples, :]
    ekf_joint_velocity = ekf_joint_velocity[:num_samples, :]
    sim_base_position = sim_base_position[:num_samples, :]
    sim_base_velocity = sim_base_velocity[:num_samples, :]
    sim_base_orientation = sim_base_orientation[:num_samples, :]
    sim_base_angular_velocity = sim_base_angular_velocity[:num_samples, :]
    sim_joint_position = sim_joint_position[:num_samples, :]
    sim_joint_velocity = sim_joint_velocity[:num_samples, :]
    measured_joint_position = measured_joint_position[:num_samples, :]
    measured_joint_velocity = measured_joint_velocity[:num_samples, :]
    measured_imu_orientation = measured_imu_orientation[:num_samples, :]
    measured_imu_angular_velocity = measured_imu_angular_velocity[:num_samples, :]
    measured_imu_accelerometer = measured_imu_accelerometer[:num_samples, :]
    execution_time_ekf = execution_time_ekf[:num_samples]
    execution_time_kf = execution_time_kf[:num_samples]
    execution_time_mpc = execution_time_mpc[:num_samples]
    execution_time_wbc = execution_time_wbc[:num_samples]
    execution_time_update = execution_time_update[:num_samples]
    
    #add 1000 zeros to estimated imu
    # estimated_imu_accelerometer = np.vstack((np.zeros((2000, 3)), estimated_imu_accelerometer))
    # estimated_imu_angular_velocity = np.vstack((np.zeros((2000, 3)), estimated_imu_angular_velocity))
    # estimated_imu_orientation = np.vstack((np.zeros((2000, 4)), estimated_imu_orientation))
    # estimated_imu_accelerometer = estimated_imu_accelerometer[:num_samples, :]
    # estimated_imu_angular_velocity = estimated_imu_angular_velocity[:num_samples, :]
    # estimated_imu_orientation = estimated_imu_orientation[:num_samples, :]
    
        
    delta = 1 / 500  # Assuming a control frequency of 500 Hz
    t = np.linspace(0.0, delta * num_samples, num_samples)

    if not os.path.exists('images/simulation/positions'):
        os.makedirs('images/simulation/positions')
    if not os.path.exists('images/simulation/velocities'):
        os.makedirs('images/simulation/velocities')
    if not os.path.exists('images/feedback/positions'):
        os.makedirs('images/feedback/positions')
    if not os.path.exists('images/feedback/velocities'):
        os.makedirs('images/feedback/velocities')
    if not os.path.exists('images/ekf'):
        os.makedirs('images/ekf')
    if not os.path.exists('images/execution_times'):
        os.makedirs('images/execution_times')
    if not os.path.exists('images/com'):
        os.makedirs('images/com')
    if not os.path.exists('images/forces_torques'):
        os.makedirs('images/forces_torques')
    if not os.path.exists('images/forces_torques/joints'):
        os.makedirs('images/forces_torques/joints')
    if not os.path.exists('images/motor_states'):
        os.makedirs('images/motor_states')
    if not os.path.exists('images/soles'):
        os.makedirs('images/soles')

    grouped_indices = defaultdict(list)

    for idx, name in enumerate(joint_names):
        base_name = '_'.join(name.split('_')[:2])  # E.g., "left_ankle" da "left_ankle_roll_joint"
        grouped_indices[base_name].append(idx)

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
        fig.savefig(f"images/forces_torques/joints/{group_name}_input_joint_torques.png")
        plt.close(fig)
        figs.append(fig)

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
    fig.savefig("images/forces_torques/estimated_force_left_sole.png")
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
    fig.savefig("images/forces_torques/estimated_force_right_sole.png")
    plt.close(fig)

    if has_wrist_force_logs:
        has_wrist_torque_logs = (
            estimated_force_left_wrist.shape[1] >= 6 and
            estimated_force_right_wrist.shape[1] >= 6 and
            estimated_force_left_wrist_filtered.shape[1] >= 6 and
            estimated_force_right_wrist_filtered.shape[1] >= 6
        )

        fig, ax = plt.subplots()
        ax.plot(t, estimated_force_left_wrist[:, 0], label='Left Wrist Fx', color='blue')
        ax.plot(t, estimated_force_left_wrist[:, 1], label='Left Wrist Fy', color='orange')
        ax.plot(t, estimated_force_left_wrist[:, 2], label='Left Wrist Fz', color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Estimated Force [N]')
        ax.set_title('Estimated Forces on Left Wrist (Raw)')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/forces_torques/estimated_force_left_wrist.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(t, estimated_force_left_wrist_filtered[:, 0], label='Left Wrist Fx Filtered', color='blue')
        ax.plot(t, estimated_force_left_wrist_filtered[:, 1], label='Left Wrist Fy Filtered', color='orange')
        ax.plot(t, estimated_force_left_wrist_filtered[:, 2], label='Left Wrist Fz Filtered', color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Estimated Force [N]')
        ax.set_title('Estimated Forces on Left Wrist (Filtered)')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/forces_torques/estimated_force_left_wrist_filtered.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(t, estimated_force_right_wrist[:, 0], label='Right Wrist Fx', color='blue')
        ax.plot(t, estimated_force_right_wrist[:, 1], label='Right Wrist Fy', color='orange')
        ax.plot(t, estimated_force_right_wrist[:, 2], label='Right Wrist Fz', color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Estimated Force [N]')
        ax.set_title('Estimated Forces on Right Wrist (Raw)')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/forces_torques/estimated_force_right_wrist.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(t, estimated_force_right_wrist_filtered[:, 0], label='Right Wrist Fx Filtered', color='blue')
        ax.plot(t, estimated_force_right_wrist_filtered[:, 1], label='Right Wrist Fy Filtered', color='orange')
        ax.plot(t, estimated_force_right_wrist_filtered[:, 2], label='Right Wrist Fz Filtered', color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Estimated Force [N]')
        ax.set_title('Estimated Forces on Right Wrist (Filtered)')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("images/forces_torques/estimated_force_right_wrist_filtered.png")
        plt.close(fig)

        if has_wrist_torque_logs:
            fig, ax = plt.subplots()
            ax.plot(t, estimated_force_left_wrist[:, 3], label='Left Wrist Tx', color='blue')
            ax.plot(t, estimated_force_left_wrist[:, 4], label='Left Wrist Ty', color='orange')
            ax.plot(t, estimated_force_left_wrist[:, 5], label='Left Wrist Tz', color='green')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Estimated Torque [N*m]')
            ax.set_title('Estimated Torques on Left Wrist (Raw)')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig("images/forces_torques/estimated_torque_left_wrist.png")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(t, estimated_force_left_wrist_filtered[:, 3], label='Left Wrist Tx Filtered', color='blue')
            ax.plot(t, estimated_force_left_wrist_filtered[:, 4], label='Left Wrist Ty Filtered', color='orange')
            ax.plot(t, estimated_force_left_wrist_filtered[:, 5], label='Left Wrist Tz Filtered', color='green')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Estimated Torque [N*m]')
            ax.set_title('Estimated Torques on Left Wrist (Filtered)')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig("images/forces_torques/estimated_torque_left_wrist_filtered.png")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(t, estimated_force_right_wrist[:, 3], label='Right Wrist Tx', color='blue')
            ax.plot(t, estimated_force_right_wrist[:, 4], label='Right Wrist Ty', color='orange')
            ax.plot(t, estimated_force_right_wrist[:, 5], label='Right Wrist Tz', color='green')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Estimated Torque [N*m]')
            ax.set_title('Estimated Torques on Right Wrist (Raw)')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig("images/forces_torques/estimated_torque_right_wrist.png")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(t, estimated_force_right_wrist_filtered[:, 3], label='Right Wrist Tx Filtered', color='blue')
            ax.plot(t, estimated_force_right_wrist_filtered[:, 4], label='Right Wrist Ty Filtered', color='orange')
            ax.plot(t, estimated_force_right_wrist_filtered[:, 5], label='Right Wrist Tz Filtered', color='green')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Estimated Torque [N*m]')
            ax.set_title('Estimated Torques on Right Wrist (Filtered)')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig("images/forces_torques/estimated_torque_right_wrist_filtered.png")
            plt.close(fig)

    if has_applied_wrist_force_logs:
        applied_t = applied_wrist_force[:, 0]

        fig, ax = plt.subplots()
        ax.plot(applied_t, applied_wrist_force[:, 5], label='Applied Left Wrist Fx', color='tab:blue')
        ax.plot(applied_t, applied_wrist_force[:, 6], label='Applied Left Wrist Fy', color='tab:orange')
        ax.plot(applied_t, applied_wrist_force[:, 7], label='Applied Left Wrist Fz', color='tab:green')
        ax.plot(applied_t, applied_wrist_force[:, 15], '--', label='Applied Right Wrist Fx', color='tab:blue')
        ax.plot(applied_t, applied_wrist_force[:, 16], '--', label='Applied Right Wrist Fy', color='tab:orange')
        ax.plot(applied_t, applied_wrist_force[:, 17], '--', label='Applied Right Wrist Fz', color='tab:green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Applied Force [N]')
        ax.set_title('Applied External Wrist Forces')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig('images/forces_torques/applied_external_wrist_force.png')
        plt.close(fig)

        if applied_wrist_force.shape[1] >= 21:
            fig, ax = plt.subplots()
            ax.plot(applied_t, applied_wrist_force[:, 8], label='Applied Left Wrist Tx', color='tab:blue')
            ax.plot(applied_t, applied_wrist_force[:, 9], label='Applied Left Wrist Ty', color='tab:orange')
            ax.plot(applied_t, applied_wrist_force[:, 10], label='Applied Left Wrist Tz', color='tab:green')
            ax.plot(applied_t, applied_wrist_force[:, 18], '--', label='Applied Right Wrist Tx', color='tab:blue')
            ax.plot(applied_t, applied_wrist_force[:, 19], '--', label='Applied Right Wrist Ty', color='tab:orange')
            ax.plot(applied_t, applied_wrist_force[:, 20], '--', label='Applied Right Wrist Tz', color='tab:green')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Applied Torque [N*m]')
            ax.set_title('Applied External Wrist Torques')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig('images/forces_torques/applied_external_wrist_torque.png')
            plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(applied_t, applied_wrist_force[:, 1], label='Left Enabled', color='tab:purple')
        ax.plot(applied_t, applied_wrist_force[:, 11], label='Right Enabled', color='tab:brown')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Enabled Flag')
        ax.set_title('External Wrist Force Enable Flags')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig('images/forces_torques/applied_external_wrist_force_enable.png')
        plt.close(fig)

    if has_compliance_hand_logs:
        compliance_t = compliance_hand_state[:, 0]
        dof_labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

        left_delta_x = compliance_hand_state[:, 1:7]
        left_delta_dx = compliance_hand_state[:, 7:13]
        left_delta_ddx = compliance_hand_state[:, 13:19]
        right_delta_x = compliance_hand_state[:, 19:25]
        right_delta_dx = compliance_hand_state[:, 25:31]
        right_delta_ddx = compliance_hand_state[:, 31:37]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for i, label in enumerate(dof_labels):
            axes[0].plot(compliance_t, left_delta_x[:, i], label=label)
            axes[1].plot(compliance_t, left_delta_dx[:, i], label=label)
            axes[2].plot(compliance_t, left_delta_ddx[:, i], label=label)
        axes[0].set_ylabel('delta_x')
        axes[1].set_ylabel('delta_dx')
        axes[2].set_ylabel('delta_ddx')
        axes[2].set_xlabel('Time [s]')
        axes[0].set_title('Left Hand Compliance State')
        for axis in axes:
            axis.grid(True)
            axis.legend(loc='upper right', ncol=3)
        fig.tight_layout()
        fig.savefig('images/forces_torques/left_hand_compliance_state.png')
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for i, label in enumerate(dof_labels):
            axes[0].plot(compliance_t, right_delta_x[:, i], label=label)
            axes[1].plot(compliance_t, right_delta_dx[:, i], label=label)
            axes[2].plot(compliance_t, right_delta_ddx[:, i], label=label)
        axes[0].set_ylabel('delta_x')
        axes[1].set_ylabel('delta_dx')
        axes[2].set_ylabel('delta_ddx')
        axes[2].set_xlabel('Time [s]')
        axes[0].set_title('Right Hand Compliance State')
        for axis in axes:
            axis.grid(True)
            axis.legend(loc='upper right', ncol=3)
        fig.tight_layout()
        fig.savefig('images/forces_torques/right_hand_compliance_state.png')
        plt.close(fig)

    if has_compliance_torso_logs:
        torso_t = compliance_torso_state[:, 0]
        dof_labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        force_labels = ['Fx', 'Fy', 'Fz']
        torque_labels = ['Tx', 'Ty', 'Tz']

        left_wrench = compliance_torso_state[:, 1:7]
        right_wrench = compliance_torso_state[:, 7:13]
        left_manual_delta = compliance_torso_state[:, 13:19]
        right_manual_delta = compliance_torso_state[:, 19:25]
        left_delta_filtered = compliance_torso_state[:, 37:43]
        right_delta_filtered = compliance_torso_state[:, 43:49]
        delta_xb = compliance_torso_state[:, 49:55]
        delta_xb_final = compliance_torso_state[:, 61:67]
        qp_solved = compliance_torso_state[:, -1]
        angular_xb_rad = np.hstack((delta_xb[:, 3:6], delta_xb_final[:, 3:6]))
        angular_xb_limit_rad = max(1.0e-4, 1.05 * np.nanmax(np.abs(angular_xb_rad)))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for i, label in enumerate(force_labels):
            axes[0].plot(torso_t, left_wrench[:, i], label=f'L {label}')
            axes[0].plot(torso_t, right_wrench[:, i], '--', label=f'R {label}')
        for i, label in enumerate(torque_labels):
            axes[1].plot(torso_t, left_wrench[:, i + 3], label=f'L {label}')
            axes[1].plot(torso_t, right_wrench[:, i + 3], '--', label=f'R {label}')
        axes[0].set_ylabel('Force [N]')
        axes[1].set_ylabel('Torque [N*m]')
        axes[1].set_xlabel('Time [s]')
        axes[0].set_title('Torso Compliance Numeric Test Input Wrenches')
        for axis in axes:
            axis.grid(True)
            axis.legend(loc='upper right', ncol=3)
        fig.tight_layout()
        fig.savefig('images/forces_torques/torso_compliance_input_wrenches.png')
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for i, label in enumerate(dof_labels[:3]):
            axes[0].plot(torso_t, left_manual_delta[:, i], label=f'L manual {label}')
            axes[0].plot(torso_t, right_manual_delta[:, i], '--', label=f'R manual {label}')
            axes[0].plot(torso_t, left_delta_filtered[:, i], ':', label=f'L filtered {label}')
            axes[0].plot(torso_t, right_delta_filtered[:, i], '-.', label=f'R filtered {label}')
        for i, label in enumerate(dof_labels[3:]):
            idx = i + 3
            axes[1].plot(torso_t, left_manual_delta[:, idx], label=f'L manual {label}')
            axes[1].plot(torso_t, right_manual_delta[:, idx], '--', label=f'R manual {label}')
            axes[1].plot(torso_t, left_delta_filtered[:, idx], ':', label=f'L filtered {label}')
            axes[1].plot(torso_t, right_delta_filtered[:, idx], '-.', label=f'R filtered {label}')
        axes[0].set_ylabel('Linear delta_xc [m]')
        axes[1].set_ylabel('Angular delta_xc [rad]')
        axes[1].set_xlabel('Time [s]')
        axes[0].set_title('Torso Compliance Numeric Test Hand Displacement Inputs')
        for axis in axes:
            axis.grid(True)
            axis.legend(loc='upper right', ncol=3)
        fig.tight_layout()
        fig.savefig('images/forces_torques/torso_compliance_hand_displacement_inputs.png')
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for i, label in enumerate(dof_labels[:3]):
            axes[0].plot(torso_t, delta_xb[:, i], '--', label=f'raw {label}')
            axes[0].plot(torso_t, delta_xb_final[:, i], label=f'final {label}')
        for i, label in enumerate(dof_labels[3:]):
            idx = i + 3
            axes[1].plot(torso_t, delta_xb[:, idx], '--', label=f'raw {label}')
            axes[1].plot(torso_t, delta_xb_final[:, idx], label=f'final {label}')
        axes[0].set_ylabel('Linear delta_xb [m]')
        axes[1].set_ylabel('Angular delta_xb [rad]')
        axes[1].set_ylim(-angular_xb_limit_rad, angular_xb_limit_rad)
        axes[1].set_xlabel('Time [s]')
        axes[0].set_title('Torso Compliance QP Output')
        for axis in axes:
            axis.grid(True)
            axis.legend(loc='upper right', ncol=3)
        fig.tight_layout()
        fig.savefig('images/forces_torques/torso_compliance_qp_output.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        delta_xb_ang_deg = np.rad2deg(delta_xb[:, 3:6])
        delta_xb_final_ang_deg = np.rad2deg(delta_xb_final[:, 3:6])
        angular_xb_limit_deg = np.rad2deg(angular_xb_limit_rad)
        for i, label in enumerate(dof_labels[3:]):
            ax.plot(torso_t, delta_xb_ang_deg[:, i], '--', label=f'raw {label}')
            ax.plot(torso_t, delta_xb_final_ang_deg[:, i], label=f'final {label}')
        ax.set_ylabel('Angular delta_xb [deg]')
        ax.set_xlabel('Time [s]')
        ax.set_ylim(-angular_xb_limit_deg, angular_xb_limit_deg)
        ax.set_title('Torso Compliance QP Angular Output')
        ax.grid(True)
        ax.legend(loc='upper right', ncol=3)
        fig.tight_layout()
        fig.savefig('images/forces_torques/torso_compliance_qp_output_angular_deg.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(torso_t, qp_solved, label='qp_solved')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Solved Flag')
        ax.set_title('Torso Compliance QP Solve Status')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig('images/forces_torques/torso_compliance_qp_solved.png')
        plt.close(fig)

    if has_wbc_torso_tracking_logs:
        torso_track_t = wbc_torso_tracking[:, 0]
        torso_offset_deg = np.rad2deg(wbc_torso_tracking[:, 1:4])
        torso_nom_rad = wbc_torso_tracking[:, 4:7]
        torso_des_rad = wbc_torso_tracking[:, 7:10]
        torso_cur_rad = wbc_torso_tracking[:, 10:13]

        def relative_rpy_change_deg(
            nominal_rpy_rad: np.ndarray,
            target_rpy_rad: np.ndarray,
        ) -> np.ndarray:
            nominal_rotations = R.from_euler(
                'ZYX', nominal_rpy_rad[:, [2, 1, 0]]
            )
            target_rotations = R.from_euler(
                'ZYX', target_rpy_rad[:, [2, 1, 0]]
            )
            relative_rotations = target_rotations * nominal_rotations.inv()
            relative_ypr_rad = relative_rotations.as_euler('ZYX')
            return np.rad2deg(relative_ypr_rad[:, [2, 1, 0]])

        torso_des_delta_deg = relative_rpy_change_deg(
            torso_nom_rad, torso_des_rad
        )
        torso_cur_delta_deg = relative_rpy_change_deg(
            torso_nom_rad, torso_cur_rad
        )
        torso_err_deg = np.rad2deg(wbc_torso_tracking[:, 13:16])
        angular_labels = ['roll', 'pitch', 'yaw']

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for i, label in enumerate(angular_labels):
            axes[0].plot(torso_track_t, torso_offset_deg[:, i], label=f'offset {label}')
            axes[1].plot(torso_track_t, torso_des_delta_deg[:, i], '--', label=f'des {label}')
            axes[1].plot(torso_track_t, torso_cur_delta_deg[:, i], label=f'cur {label}')
            axes[2].plot(torso_track_t, torso_err_deg[:, i], label=f'err {label}')
        axes[0].set_ylabel('QP offset [deg]')
        axes[1].set_ylabel('Torso RPY change [deg]')
        axes[1].set_ylim(-20, 20)
        axes[2].set_ylabel('Error [deg]')
        axes[2].set_xlabel('Time [s]')
        axes[0].set_title('WBC Torso Orientation Tracking')
        for axis in axes:
            axis.grid(True)
            axis.legend(loc='upper right', ncol=3)
        fig.tight_layout()
        fig.savefig('images/forces_torques/wbc_torso_orientation_tracking.png')
        plt.close(fig)

    if has_all_joint_motor_logs and motor_plot_joint_count > 0 and motor_plot_sample_count > 0:
        motor_plot_len = min(
            t.shape[0],
            all_joint_motor_q.shape[0],
            all_joint_motor_dq.shape[0],
            all_joint_motor_ddq.shape[0],
            all_joint_motor_tau_est.shape[0],
        )
        if has_all_joint_motor_tau_applied:
            motor_plot_len = min(motor_plot_len, all_joint_motor_tau_applied.shape[0])
        motor_t = t[:motor_plot_len]

        for i, joint_name in enumerate(all_joint_motor_names):
            n_motor_subplots = 5 if has_all_joint_motor_tau_applied else 4
            fig, axs = plt.subplots(n_motor_subplots, 1, figsize=(10, 10 if has_all_joint_motor_tau_applied else 8), sharex=True)

            axs[0].plot(motor_t, all_joint_motor_q[:motor_plot_len, i], color='tab:blue')
            axs[0].set_ylabel('q [rad]')
            axs[0].grid(True)
            axs[0].set_title(f'Motor State - {joint_name}')

            axs[1].plot(motor_t, all_joint_motor_dq[:motor_plot_len, i], color='tab:orange')
            axs[1].set_ylabel('dq [rad/s]')
            axs[1].grid(True)

            axs[2].plot(motor_t, all_joint_motor_ddq[:motor_plot_len, i], color='tab:green')
            axs[2].set_ylabel('ddq [rad/s²]')
            axs[2].grid(True)

            axs[3].plot(motor_t, all_joint_motor_tau_est[:motor_plot_len, i], color='tab:red')
            axs[3].set_ylabel('tau_est [Nm]')
            axs[3].grid(True)

            if has_all_joint_motor_tau_applied:
                axs[4].plot(motor_t, all_joint_motor_tau_applied[:motor_plot_len, i], color='tab:purple')
                axs[4].set_ylabel('tau_appl [Nm]')
                axs[4].grid(True)
                axs[4].set_xlabel('Time [s]')
            else:
                axs[3].set_xlabel('Time [s]')

            fig.tight_layout()
            fig.savefig(f'images/motor_states/{joint_name}_motor_state.png')
            plt.close(fig)

    #################################
    #  COM AND ZMP PLOTS
    #################################
    fig, ax = plt.subplots()
    ax.plot(t, fb_zmp_position[:, 0], label='used ZMP X', color='blue', linestyle=':')
    ax.plot(t, fb_zmp_position[:, 1], label='used ZMP Y', color='orange', linestyle=':')
    ax.plot(t, ef_zmp_position[:, 0], label='not used ZMP X', color='blue', linestyle='--')
    ax.plot(t, ef_zmp_position[:, 1], label='not used ZMP Y', color='orange', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('ZMP X & Y Position Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/fb_used_and_not_used_zmp_plot.png")

    fig, ax = plt.subplots()
    ax.plot(t, des_zmp_position[:, 0], label='des ZMP X', color='blue', linestyle=':')
    ax.plot(t, des_zmp_position[:, 1], label='des ZMP Y', color='orange', linestyle=':')
    ax.plot(t, des_com_position[:, 0], label='des COM X', color='blue', linestyle='-.')
    ax.plot(t, des_com_position[:, 1], label='des COM Y', color='orange', linestyle='-.')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('ZMP and COM Position Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/des_zmp_and_com_plot.png")

    #plot x position of com, left and right sole
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 0], label='fb left sole X', color='blue')
    ax.plot(t, p_rsole_fb[:, 0], label='fb right sole X', color='orange')
    ax.plot(t, kf_com_position[:, 0], label='kf COM X', color='green')
    ax.plot(t, kf_zmp_position[:, 0], label='kf ZMP X', color='red')
    ax.plot(t, p_lsole_des[:, 0], label='des left sole X', color='blue', linestyle='--')
    ax.plot(t, p_rsole_des[:, 0], label='des right sole X', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 0], label='des COM X', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position X [m]')
    ax.set_title('Left Sole, Right Sole and COM X Position Filtered vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_left_right_sole_and_com_x_plot.png")

    #plot y position of com, left and right sole
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 1], label='fb left sole Y', color='blue')
    ax.plot(t, p_rsole_fb[:, 1], label='fb right sole Y', color='orange')
    ax.plot(t, kf_com_position[:, 1], label='kf COM Y', color='green')
    ax.plot(t, kf_zmp_position[:, 1], label='kf ZMP Y', color='red')
    # ax.plot(t, sim_com_position[:, 1], label='sim COM Y', color='green')
    # ax.plot(t, sim_zmp_position[:, 1], label='sim ZMP Y', color='red')
    ax.plot(t, p_lsole_des[:, 1], label='des left sole Y', color='blue', linestyle='--')
    ax.plot(t, p_rsole_des[:, 1], label='des right sole Y', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 1], label='des COM Y', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Y [m]')
    ax.set_title('Left Sole, Right Sole and COM Y Position Filtered vs Desired')
    ax.grid(True)
    #lim between 24 and 26
    # ax.set_xlim([27.99, 28.02])
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_left_right_sole_and_com_y_plot.png")

    #plot x position of com, left and right sole
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 2], label='fb left sole z', color='blue')
    ax.plot(t, p_rsole_fb[:, 2], label='fb right sole z', color='orange')
    ax.plot(t, kf_com_position[:, 2], label='kf COM z', color='green')
    ax.plot(t, kf_zmp_position[:, 2], label='kf ZMP z', color='red')
    ax.plot(t, p_lsole_des[:, 2], label='des left sole z', color='blue', linestyle='--')
    ax.plot(t, p_rsole_des[:, 2], label='des right sole z', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 2], label='des COM z', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position X [m]')
    ax.set_title('Left Sole, Right Sole and COM X Position Filtered vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_left_right_sole_and_com_z_plot.png")

    #plot y position of com, left and right sole
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 2], label='fb left sole Z', color='blue')
    ax.plot(t, p_rsole_fb[:, 2], label='fb right sole Z', color='orange')
    ax.plot(t, fb_com_position[:, 2], label='fb COM Z', color='green')
    ax.plot(t, fb_zmp_position[:, 2], label='fb ZMP Z', color='red')
    # ax.plot(t, sim_com_position[:, 2], label='sim COM Z', color='green')
    # ax.plot(t, sim_zmp_position[:, 2], label='sim ZMP Z', color='red')
    ax.plot(t, p_lsole_des[:, 2], label='des left sole Z', color='blue', linestyle='--')
    ax.plot(t, p_rsole_des[:, 2], label='des right sole Z', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 2], label='des COM Z', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Y [m]')
    ax.set_title('Left Sole, Right Sole and COM Y Position Filtered vs Desired')
    ax.grid(True)
    #lim between 24 and 26
    # ax.set_xlim([27.99, 28.02])
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/fb_left_right_sole_and_com_z_plot.png")

    #plot x position of com, left and right sole
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 0], label='fb left sole X', color='blue')
    ax.plot(t, p_rsole_fb[:, 0], label='fb right sole X', color='orange')
    ax.plot(t, fb_com_position[:, 0], label='fb COM X', color='green')
    ax.plot(t, fb_zmp_position[:, 0], label='fb ZMP X', color='red')
    ax.plot(t, p_lsole_des[:, 0], label='des left sole X', color='blue', linestyle='--')
    ax.plot(t, p_rsole_des[:, 0], label='des right sole X', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 0], label='des COM X', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position X [m]')
    ax.set_title('Left Sole, Right Sole and COM X Position Feedback vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/fb_left_right_sole_and_com_x_plot.png")

    #plot y position of com, left and right sole
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_fb[:, 1], label='fb left sole Y', color='blue')
    ax.plot(t, p_rsole_fb[:, 1], label='fb right sole Y', color='orange')
    ax.plot(t, fb_com_position[:, 1], label='fb COM Y', color='green')
    ax.plot(t, fb_zmp_position[:, 1], label='fb ZMP Y', color='red')
    ax.plot(t, p_lsole_des[:, 1], label='des left sole Y', color='blue', linestyle='--')
    ax.plot(t, p_rsole_des[:, 1], label='des right sole Y', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 1], label='des COM Y', color='green', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Y [m]')
    ax.set_title('Left Sole, Right Sole and COM Y Position Feedback vs Desired')
    ax.grid(True)
    #lim between 24 and 26
    # ax.set_xlim([27.99, 28.02])
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/fb_left_right_sole_and_com_y_plot.png")

    fig, ax = plt.subplots()
    ax.plot(t, kf_zmp_position[:, 0], label='kf ZMP X', color='blue', linestyle=':')
    ax.plot(t, kf_zmp_position[:, 1], label='kf ZMP Y', color='orange', linestyle=':')
    ax.plot(t, kf_com_position[:, 0], label='kf COM X', color='blue', linestyle='-.')
    ax.plot(t, kf_com_position[:, 1], label='kf COM Y', color='orange', linestyle='-.')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('ZMP and COM X & Y Position Filtered')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_zmp_and_com_plot.png")


    fig, ax = plt.subplots()
    ax.plot(t, sim_zmp_position[:, 0], label='Sim ZMP X', color='blue')
    ax.plot(t, sim_zmp_position[:, 1], label='Sim ZMP Y', color='orange')
    ax.plot(t, sim_zmp_position[:, 2], label='Sim ZMP Z', color='green')
    ax.plot(t, fb_zmp_position[:, 0], label='FB ZMP X', color='blue', linestyle='--')
    ax.plot(t, fb_zmp_position[:, 1], label='FB ZMP Y', color='orange', linestyle='--')
    ax.plot(t, fb_zmp_position[:, 2], label='FB ZMP Z', color='green', linestyle='--')
    ax.plot(t, des_zmp_position[:, 0], label='Des ZMP X', color='blue', linestyle=':')
    ax.plot(t, des_zmp_position[:, 1], label='Des ZMP Y', color='orange', linestyle=':')
    ax.plot(t, des_zmp_position[:, 2], label='Des ZMP Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('ZMP Position [m]')
    ax.set_title('ZMP Position Simulation vs Feedback vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/sim_vs_fb_vs_des_zmp_plot.png")
    plt.close(fig)

    #plot filtered zmp sim and fb
    fig, ax = plt.subplots()
    ax.plot(t, kf_zmp_position[:, 0], label='kf ZMP X', color='blue')
    ax.plot(t, kf_zmp_position[:, 1], label='kf ZMP Y', color='orange')
    ax.plot(t, kf_zmp_position[:, 2], label='kf ZMP Z', color='green')
    ax.plot(t, des_zmp_position[:, 0], label='Des ZMP X', color='blue', linestyle=':')
    ax.plot(t, des_zmp_position[:, 1], label='Des ZMP Y', color='orange', linestyle=':')
    ax.plot(t, des_zmp_position[:, 2], label='Des ZMP Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Filtered ZMP Position [m]')
    ax.set_title('Filtered vs Desired ZMP Position')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_vs_des_zmp_plot.png")
    plt.close(fig)

    #plot com sim and fb
    fig, ax = plt.subplots()
    ax.plot(t, sim_com_position[:, 0], label='Sim COM X', color='blue')
    ax.plot(t, sim_com_position[:, 1], label='Sim COM Y', color='orange')
    ax.plot(t, fb_com_position[:, 0], label='FB COM X', color='blue', linestyle='--')
    ax.plot(t, fb_com_position[:, 1], label='FB COM Y', color='orange', linestyle='--')
    ax.plot(t, des_com_position[:, 0], label='Des COM X', color='blue', linestyle=':')
    ax.plot(t, des_com_position[:, 1], label='Des COM Y', color='orange', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('COM Position [m]')
    ax.set_title('COM Position Simulation vs Feedback vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/sim_vs_fb_vs_des_com_plot.png")
    plt.close(fig)

    #plot filtered com sim and fb
    fig, ax = plt.subplots()
    ax.plot(t, kf_com_position[:, 0], label='Kf COM X', color='blue')
    ax.plot(t, kf_com_position[:, 1], label='Kf COM Y', color='orange')
    ax.plot(t, des_com_position[:, 0], label='Des COM X', color='blue', linestyle=':')
    ax.plot(t, des_com_position[:, 1], label='Des COM Y', color='orange', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.set_title('Filtered vs Desired X & Y COM Position')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_vs_des_com_plot.png")
    plt.close(fig)

    # plot com velocities sim and fb
    fig, ax = plt.subplots()
    ax.plot(t, sim_com_velocity[:, 0], label='Sim COM Vel X', color='blue')
    ax.plot(t, sim_com_velocity[:, 1], label='Sim COM Vel Y', color='orange')
    ax.plot(t, fb_com_velocity[:, 0], label='FB COM Vel X', color='blue', linestyle='--')
    ax.plot(t, fb_com_velocity[:, 1], label='FB COM Vel Y', color='orange', linestyle='--')
    ax.plot(t, des_com_velocity[:, 0], label='Des COM Vel X', color='blue', linestyle=':')
    ax.plot(t, des_com_velocity[:, 1], label='Des COM Vel Y', color='orange', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('COM Velocity Simulation vs Feedback vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/sim_vs_fb_vs_des_com_velocity_plot.png")
    plt.close(fig)

    # plot filtered com velocities sim and fb
    fig, ax = plt.subplots()
    ax.plot(t, kf_com_velocity[:, 0], label='Kf COM Vel X', color='blue')
    ax.plot(t, kf_com_velocity[:, 1], label='Kf COM Vel Y', color='orange')
    ax.plot(t, kf_com_velocity[:, 2], label='Kf COM Vel Z', color='green')
    ax.plot(t, des_com_velocity[:, 0], label='Des COM Vel X', color='blue', linestyle=':')
    ax.plot(t, des_com_velocity[:, 1], label='Des COM Vel Y', color='orange', linestyle=':')
    ax.plot(t, des_com_velocity[:, 2], label='Des COM Vel Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Filtered vs Desired COM Velocity')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/com/kf_vs_des_com_velocity_plot.png")
    plt.close(fig)


    ##########################
    #  FEET PLOT
    ##########################
    #plot des, sim and fb lsole position
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_sim[:, 0], label='Sim Left Sole X', color='blue')
    ax.plot(t, p_lsole_sim[:, 1], label='Sim Left Sole Y', color='orange')
    ax.plot(t, p_lsole_sim[:, 2], label='Sim Left Sole Z', color='green')
    ax.plot(t, p_lsole_fb[:, 0], label='FB Left Sole X', color='blue', linestyle='--')
    ax.plot(t, p_lsole_fb[:, 1], label='FB Left Sole Y', color='orange', linestyle='--')
    ax.plot(t, p_lsole_fb[:, 2], label='FB Left Sole Z', color='green', linestyle='--')
    ax.plot(t, p_lsole_des[:, 0], label='Des Left Sole X', color='blue', linestyle=':')
    ax.plot(t, p_lsole_des[:, 1], label='Des Left Sole Y', color='orange', linestyle=':')
    ax.plot(t, p_lsole_des[:, 2], label='Des Left Sole Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Left Sole Position [m]')
    ax.set_title('Left Sole Position Simulation vs Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/soles/sim_vs_fb_left_sole_position_plot.png")
    plt.close(fig)

    #plot des, sim and fb rsole position
    fig, ax = plt.subplots()
    ax.plot(t, p_rsole_sim[:, 0], label='Sim Right Sole X', color='blue')
    ax.plot(t, p_rsole_sim[:, 1], label='Sim Right Sole Y', color='orange')
    ax.plot(t, p_rsole_sim[:, 2], label='Sim Right Sole Z', color='green')
    ax.plot(t, p_rsole_fb[:, 0], label='FB Right Sole X', color='blue', linestyle='--')
    ax.plot(t, p_rsole_fb[:, 1], label='FB Right Sole Y', color='orange', linestyle='--')
    ax.plot(t, p_rsole_fb[:, 2], label='FB Right Sole Z', color='green', linestyle='--')
    ax.plot(t, p_rsole_des[:, 0], label='Des Right Sole X', color='blue', linestyle=':')
    ax.plot(t, p_rsole_des[:, 1], label='Des Right Sole Y', color='orange', linestyle=':')
    ax.plot(t, p_rsole_des[:, 2], label='Des Right Sole Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Right Sole Position [m]')
    ax.set_title('Right Sole Position Simulation vs Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/soles/sim_vs_fb_right_sole_position_plot.png")
    plt.close(fig)

    #plot des, sim and fb lsole velocity
    fig, ax = plt.subplots()
    ax.plot(t, v_lsole_sim[:, 0], label='Sim Left Sole Vel X', color='blue')
    ax.plot(t, v_lsole_sim[:, 1], label='Sim Left Sole Vel Y', color='orange')
    ax.plot(t, v_lsole_sim[:, 2], label='Sim Left Sole Vel Z', color='green')
    ax.plot(t, v_lsole_fb[:, 0], label='FB Left Sole Vel X', color='blue', linestyle='--')
    ax.plot(t, v_lsole_fb[:, 1], label='FB Left Sole Vel Y', color='orange', linestyle='--')
    ax.plot(t, v_lsole_fb[:, 2], label='FB Left Sole Vel Z', color='green', linestyle='--')
    ax.plot(t, v_lsole_des[:, 0], label='Des Left Sole Vel X', color='blue', linestyle=':')
    ax.plot(t, v_lsole_des[:, 1], label='Des Left Sole Vel Y', color='orange', linestyle=':')
    ax.plot(t, v_lsole_des[:, 2], label='Des Left Sole Vel Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Left Sole Velocity [m/s]')
    ax.set_title('Left Sole Velocity Simulation vs Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/soles/sim_vs_fb_left_sole_velocity_plot.png")
    plt.close(fig)

    #plot des, sim and fb rsole velocity
    fig, ax = plt.subplots()
    ax.plot(t, v_rsole_sim[:, 0], label='Sim Right Sole Vel X', color='blue')
    ax.plot(t, v_rsole_sim[:, 1], label='Sim Right Sole Vel Y', color='orange')
    ax.plot(t, v_rsole_sim[:, 2], label='Sim Right Sole Vel Z', color='green')
    ax.plot(t, v_rsole_fb[:, 0], label='FB Right Sole Vel X', color='blue', linestyle='--')
    ax.plot(t, v_rsole_fb[:, 1], label='FB Right Sole Vel Y', color='orange', linestyle='--')
    ax.plot(t, v_rsole_fb[:, 2], label='FB Right Sole Vel Z', color='green', linestyle='--')
    ax.plot(t, v_rsole_des[:, 0], label='Des Right Sole Vel X', color='blue', linestyle=':')
    ax.plot(t, v_rsole_des[:, 1], label='Des Right Sole Vel Y', color='orange', linestyle=':')
    ax.plot(t, v_rsole_des[:, 2], label='Des Right Sole Vel Z', color='green', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Right Sole Velocity [m/s]')
    ax.set_title('Right Sole Velocity Simulation vs Feedback')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/soles/sim_vs_fb_right_sole_velocity_plot.png")
    plt.close(fig)


    ##########################
    #  SIMULATION JOINTS PLOTS
    ##########################
    

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, sim_joint_position[:, i], label=joint_names[i])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [rad]')
        ax.set_title(group_name.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        fig.savefig(f"images/simulation/positions/{group_name}_position_plot.png")
        plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, sim_joint_velocity[:, i], label=joint_names[i])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angular velocities [rad/s]')
        ax.set_title(group_name.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        fig.savefig(f"images/simulation/velocities/{group_name}_velocities_plot.png")
        plt.close(fig)

    #plot simulation joint velocities
    fig, ax = plt.subplots(figsize=(18, 12))
    for i in range(sim_joint_velocity.shape[1]):
        ax.plot(t, sim_joint_velocity[:, i], label=joint_names[i].strip())
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [rad/s]')
    ax.set_title('Simulation Joint Velocities')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/simulation/velocities/simulation_joint_velocities_plot.png")
    plt.close(fig)




    ##########################
    #  EKF PLOTS
    ##########################
    
    # Plot EKF base position
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_position[:, 0] - sim_base_position[:, 0], label='EKF Base Position X', color='blue')
    ax.plot(t, ekf_base_position[:, 1] - sim_base_position[:, 1], label='EKF Base Position Y', color='orange')
    ax.plot(t, ekf_base_position[:, 2] - sim_base_position[:, 2], label='EKF Base Position Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Position [m]')
    ax.set_title('Base Position Error between EKF estimation and Simulation')
    ax.grid(True)
    ax.legend()
    # fig.tight_layout()
    fig.savefig("images/ekf/base_position_error_plot.png")
    plt.close(fig)

    # Plot EKF base velocity
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_velocity[:, 0] - sim_base_velocity[:, 0], label='EKF Base Velocity X', color='blue')
    ax.plot(t, ekf_base_velocity[:, 1] - sim_base_velocity[:, 1], label='EKF Base Velocity Y', color='orange')
    ax.plot(t, ekf_base_velocity[:, 2] - sim_base_velocity[:, 2], label='EKF Base Velocity Z', color='green')
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
    ax.plot(t, sim_base_position[:, 0], label='Base position X', color='blue', linestyle = "--")
    ax.plot(t, ekf_base_position[:, 1], label='EKF Base position Y', color='orange')
    ax.plot(t, sim_base_position[:, 1], label='Base position Y', color='orange', linestyle = "--")
    ax.plot(t, ekf_base_position[:, 2] - sim_base_position[:, 2], label='EKF Base position Z', color='green')
    ax.plot(t, sim_base_position[:, 2] - sim_base_position[:, 2], label='Base position Z', color='green', linestyle = "--")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Position [m/s]')
    ax.set_title('Base Position of EKF estimation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_position_plot.png")
    plt.close(fig)

    # Plot EKF base orientation in quaternion format
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_orientation[:, 0] - sim_base_orientation[:, 0], label='EKF Base Orientation W', color='blue')
    ax.plot(t, ekf_base_orientation[:, 1] - sim_base_orientation[:, 1], label='EKF Base Orientation X', color='orange')
    ax.plot(t, ekf_base_orientation[:, 2] - sim_base_orientation[:, 2], label='EKF Base Orientation Y', color='green')
    ax.plot(t, ekf_base_orientation[:, 3] - sim_base_orientation[:, 3], label='EKF Base Orientation Z', color='red')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Orientation [Quaternion]')
    ax.set_title('Base Orientation Error between EKF estimation and Simulation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_orientation_error_plot.png")
    plt.close(fig)

    #plot ekf orientation in euler angles
    ekf_base_orientation_euler = R.from_quat(ekf_base_orientation).as_euler('xyz', degrees=False)
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_orientation_euler[:, 0], label='EKF Base Orientation Roll', color='blue')
    ax.plot(t, ekf_base_orientation_euler[:, 1], label='EKF Base Orientation Pitch', color='orange')
    ax.plot(t, ekf_base_orientation_euler[:, 2], label='EKF Base Orientation Yaw', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Orientation [Euler angles]')
    ax.set_title('Base Orientation EKF estimation in Euler angles')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_orientation_euler_plot.png")
    plt.close(fig)

    #plot error between orientation and simulation in euler angles 
    fig, ax = plt.subplots()
    sim_base_orientation_euler = R.from_quat(sim_base_orientation).as_euler('xyz', degrees=False)
    #check every value of the error with a for and if a value is greater than 5 radiants, remove 6.28
    for i in range(len(ekf_base_orientation_euler)):
        for j in range(3):
            error = ekf_base_orientation_euler[i, j] - sim_base_orientation_euler[i, j]
            if abs(error) > 3.14:
                if error > 0:
                    ekf_base_orientation_euler[i, j] -= 2 * 3.14
                else:
                    ekf_base_orientation_euler[i, j] += 2 * 3.14
    ax.plot(t, ekf_base_orientation_euler[:, 0] - sim_base_orientation_euler[:, 0], label='EKF Base Orientation Roll', color='blue')
    ax.plot(t, ekf_base_orientation_euler[:, 1] - sim_base_orientation_euler[:, 1], label='EKF Base Orientation Pitch', color='orange')
    ax.plot(t, ekf_base_orientation_euler[:, 2] - sim_base_orientation_euler[:, 2], label='EKF Base Orientation Yaw', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Orientation Error [Euler angles]')
    ax.set_title('Base Orientation Error between EKF estimation and Simulation in Euler angles')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_orientation_euler_error_plot.png")
    plt.close(fig)

    # Plot EKF base angular velocity
    fig, ax = plt.subplots()
    ax.plot(t, ekf_base_angular_velocity[:, 0] - sim_base_angular_velocity[:, 0], label='EKF Base Angular Velocity X', color='blue')
    ax.plot(t, ekf_base_angular_velocity[:, 1] - sim_base_angular_velocity[:, 1], label='EKF Base Angular Velocity Y', color='orange')
    ax.plot(t, ekf_base_angular_velocity[:, 2] - sim_base_angular_velocity[:, 2], label='EKF Base Angular Velocity Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('EKF Base Angular Velocity [rad/s]')
    ax.set_title('Angular Velocity Error between EKF estimation and Simulation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/ekf/base_angular_velocity_error_plot.png")
    plt.close(fig)

    # Plot position error between ekf joint position and simulated joint position
    fig, ax = plt.subplots(figsize=(18, 12))
    num_joints = sim_joint_position.shape[1]
    colormap = plt.get_cmap('tab10')
    line_styles = ['-', '--', '-.', ':']
    for i in range(num_joints):
        color = colormap(i % 10)
        linestyle = line_styles[(i // 10) % len(line_styles)]  # cambia stile ogni 10 joint
        error = ekf_joint_position[:, i] - sim_joint_position[:, i]
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


    # Plot velocity error between ekf joint velocity and simulated joint velocity
    fig, ax = plt.subplots(figsize=(18, 12))
    colormap = plt.get_cmap('tab10')
    line_styles = ['-', '--', '-.', ':']
    for i in range(num_joints):
        color = colormap(i % 10)
        linestyle = line_styles[(i // 10) % len(line_styles)]  # cambia stile ogni 10 joint
        error = ekf_joint_velocity[:, i] - sim_joint_velocity[:, i]
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

    #plot position error between ekf joint position and fb joint position
    fig, ax = plt.subplots(figsize=(18, 12))
    colormap = plt.get_cmap('tab10')
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
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Position [rad]', fontsize=14)
    ax.set_title('Joint Position Error between EKF estimation and Feedback', fontsize=16)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=12, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.savefig("images/ekf/joint_position_error_vs_feedback_plot.png")
    plt.close(fig)

    #plot velocity error between ekf joint velocity and fb joint velocity
    fig, ax = plt.subplots(figsize=(18, 12))
    colormap = plt.get_cmap('tab10')
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
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Velocity [rad/s]', fontsize=14)
    ax.set_title('Joint Velocity Error between EKF estimation and Feedback', fontsize=16)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=12, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.savefig("images/ekf/joint_velocities_error_vs_feedback_plot.png")
    plt.close(fig)

    # # plot imu orientation estimated
    # fig, ax = plt.subplots()
    # ax.plot(t, estimated_imu_orientation[:, 0], label='estimated IMU Orientation W', color='blue')
    # ax.plot(t, estimated_imu_orientation[:, 1], label='estimated IMU Orientation X', color='orange')
    # ax.plot(t, estimated_imu_orientation[:, 2], label='estimated IMU Orientation Y', color='green')
    # ax.plot(t, estimated_imu_orientation[:, 3], label='estimated IMU Orientation Z', color='red')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('estimated IMU Orientation [Quaternion]')
    # ax.set_title('estimated IMU Orientation')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig("images/ekf/estimated_imu_orientation_plot.png")
    # plt.close(fig)

    # # plot imu accelerometer estimated
    # # fig, ax = plt.subplots()
    # # ax.plot(t, estimated_imu_accelerometer[:, 0], label='estimated IMU Accelerometer X', color='blue')
    # # ax.plot(t, estimated_imu_accelerometer[:, 1], label='estimated IMU Accelerometer Y', color='orange')
    # # ax.plot(t, estimated_imu_accelerometer[:, 2], label='estimated IMU Accelerometer Z', color='green')
    # # ax.set_xlabel('Time [s]')
    # # ax.set_ylabel('estimated IMU Accelerometer [m/s^2]')
    # # ax.set_title('estimated IMU Accelerometer')
    # # ax.grid(True)
    # # ax.legend()
    # # fig.tight_layout()
    # # fig.savefig("images/ekf/estimated_imu_accelerometer_plot.png")
    # # plt.close(fig)

    # # plot imu angular velocity estimated
    # fig, ax = plt.subplots()
    # ax.plot(t, estimated_imu_angular_velocity[:, 0], label='estimated IMU Angular Velocity X', color='blue')
    # ax.plot(t, estimated_imu_angular_velocity[:, 1], label='estimated IMU Angular Velocity Y', color='orange')
    # ax.plot(t, estimated_imu_angular_velocity[:, 2], label='estimated IMU Angular Velocity Z', color='green')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('estimated IMU Angular Velocity [rad/s]')
    # ax.set_title('estimated IMU Angular Velocity')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig("images/ekf/estimated_imu_angular_velocity_plot.png")
    # plt.close(fig)

    # plot error between real and estimated imu accelerometer
    # fig, ax = plt.subplots()
    # ax.plot(t, measured_imu_accelerometer[:, 0] - estimated_imu_accelerometer[:, 0], label='IMU Accelerometer X Error', color='blue')
    # ax.plot(t, measured_imu_accelerometer[:, 1] - estimated_imu_accelerometer[:, 1], label='IMU Accelerometer Y Error', color='orange')
    # ax.plot(t, measured_imu_accelerometer[:, 2] - estimated_imu_accelerometer[:, 2], label='IMU Accelerometer Z Error', color='green')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('IMU Accelerometer Error [m/s^2]')
    # ax.set_title('IMU Accelerometer Error: Real - estimated')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig("images/ekf/imu_accelerometer_error_plot.png")
    # plt.close(fig)

    # plot error between real and estimated imu angular velocity
    # fig, ax = plt.subplots()
    # ax.plot(t, measured_imu_angular_velocity[:, 0] - estimated_imu_angular_velocity[:, 0], label='IMU Angular Velocity X Error', color='blue')
    # ax.plot(t, measured_imu_angular_velocity[:, 1] - estimated_imu_angular_velocity[:, 1], label='IMU Angular Velocity Y Error', color='orange')
    # ax.plot(t, measured_imu_angular_velocity[:, 2] - estimated_imu_angular_velocity[:, 2], label='IMU Angular Velocity Z Error', color='green')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('IMU Angular Velocity Error [rad/s]')
    # ax.set_title('IMU Angular Velocity Error: Real - estimated')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig("images/ekf/imu_angular_velocity_error_plot.png")
    # plt.close(fig)

    # # plot error between real and estimated imu orientation
    # fig, ax = plt.subplots()
    # ax.plot(t, measured_imu_orientation[:, 0] - estimated_imu_orientation[:, 0], label='IMU Orientation W Error', color='blue')
    # ax.plot(t, measured_imu_orientation[:, 1] - estimated_imu_orientation[:, 1], label='IMU Orientation X Error', color='orange')
    # ax.plot(t, measured_imu_orientation[:, 2] - estimated_imu_orientation[:, 2], label='IMU Orientation Y Error', color='green')
    # ax.plot(t, measured_imu_orientation[:, 3] - estimated_imu_orientation[:, 3], label='IMU Orientation Z Error', color='red')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('IMU Orientation Error [Quaternion]')
    # ax.set_title('IMU Orientation Error: Real - estimated')
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig("images/ekf/imu_orientation_error_plot.png")
    # plt.close(fig)

    ##########################
    #  FEEDBACK PLOTS
    ##########################

    # plot imu orientation
    fig, ax = plt.subplots()
    ax.plot(t, measured_imu_orientation[:, 0], label='IMU Orientation W', color='blue')
    ax.plot(t, measured_imu_orientation[:, 1], label='IMU Orientation X', color='orange')
    ax.plot(t, measured_imu_orientation[:, 2], label='IMU Orientation Y', color='green')
    ax.plot(t, measured_imu_orientation[:, 3], label='IMU Orientation Z', color='red')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('IMU Orientation [Quaternion]')
    ax.set_title('IMU Orientation')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/measured_imu_orientation_quaternion_plot.png")
    plt.close(fig)

    #convert quaternion w x y z to euler angles and plot them not using R
    measured_imu_orientation_euler = R.from_quat(measured_imu_orientation[:, 0:4].copy()).as_euler('xyz', degrees=True)
    fig, ax = plt.subplots()
    ax.plot(t, measured_imu_orientation_euler[:, 0], label='IMU Orientation Roll', color='blue')
    ax.plot(t, measured_imu_orientation_euler[:, 1], label='IMU Orientation Pitch', color='orange')
    ax.plot(t, measured_imu_orientation_euler[:, 2], label='IMU Orientation Yaw', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('IMU Orientation [Degrees]')
    ax.set_title('IMU Orientation in Euler Angles')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/measured_imu_orientation_euler_plot.png")
    plt.close(fig)

    # plot imu angular velocity
    fig, ax = plt.subplots()
    ax.plot(t, measured_imu_angular_velocity[:, 0], label='IMU Angular Velocity X', color='blue')
    ax.plot(t, measured_imu_angular_velocity[:, 1], label='IMU Angular Velocity Y', color='orange')
    ax.plot(t, measured_imu_angular_velocity[:, 2], label='IMU Angular Velocity Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('IMU Angular Velocity [rad/s]')
    ax.set_title('IMU Angular Velocity')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/measured_imu_angular_velocity_plot.png")
    plt.close(fig)
    
    # plot imu accelerometer
    fig, ax = plt.subplots()
    ax.plot(t, measured_imu_accelerometer[:, 0], label='IMU Accelerometer X', color='blue')
    ax.plot(t, measured_imu_accelerometer[:, 1], label='IMU Accelerometer Y', color='orange')
    ax.plot(t, measured_imu_accelerometer[:, 2], label='IMU Accelerometer Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('IMU Accelerometer [m/s^2]')
    ax.set_title('IMU Accelerometer')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/feedback/measured_imu_accelerometer_plot.png")
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
    fig.savefig("images/feedback/measured_joint_velocity_plot.png")
    plt.close(fig)

    figs = []
    for group_name, indices in grouped_indices.items():
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(t, measured_joint_position[:, i], label=joint_names[i])
            ax.plot(t, sim_joint_position[:, i], label=f"{joint_names[i].strip()} Simulation", linestyle='--')
            ax.plot(t, ekf_joint_position[:, i], label=f"{joint_names[i].strip()} EKF", linestyle=':')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [rad]')
        ax.set_title(group_name.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        fig.savefig(f"images/feedback/positions/{group_name}_fb_position_plot.png")
        plt.close(fig)


    # plot velocity error between input command and feedback joint velocity, everything in one single plot
    fig, ax = plt.subplots(figsize=(18, 12))
    for i in range(measured_joint_velocity.shape[1]):
        error = measured_joint_velocity[:, i] - sim_joint_velocity[:, i]
        ax.plot(t, error, label=joint_names[i].strip())
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity Error [rad/s]')
    ax.set_title('Velocity Error between Simulation and Feedback Joint Velocity')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"images/feedback/velocity_error_plot.png")
    plt.close(fig)


    ##########################
    #  EXECUTION TIME PLOTS
    ##########################

    #plot execution times over the itarations, first in different plots, then summed up in a single plot with a line at 2000
    figs = []
    exec_times = {
        'EKF': execution_time_ekf,
        'KF': execution_time_kf,
        'MPC': execution_time_mpc,
        'WBC': execution_time_wbc,
        'Update': execution_time_update
    }
    for name, times in exec_times.items():
        fig, ax = plt.subplots()
        exec_t = t[:times.shape[0]]
        ax.plot(exec_t, times, label=f'{name} Execution Time', color='blue')
        if name == 'Update':
            ax.axhline(y=2000, color='r', linestyle='--', label='2000 microseconds')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Execution Time [microseconds]')
        ax.set_title(f'{name} Execution Time')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        fig.savefig(f"images/execution_times/{name}_execution_time_plot.png")
        plt.close(fig)

    #plot the sum of each execution time
    total_execution_time = (execution_time_ekf + execution_time_kf + execution_time_mpc + execution_time_wbc)
    fig, ax = plt.subplots(figsize=(12, 8))
    total_exec_t = t[:total_execution_time.shape[0]]
    ax.plot(total_exec_t, total_execution_time, label='Total Execution Time', color='blue')
    ax.axhline(y=2000, color='r', linestyle='--', label='2000 microseconds')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Total Execution Time [microseconds]')
    ax.set_title('Total Execution Time')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/execution_times/total_execution_time_plot.png")
    plt.close(fig)
