"""Builds a fully-populated PlotContext from a log folder.

This is a straight extraction of the setup block that used to sit at the
top of plot_joint_data.py's `if __name__ == '__main__':` (folder/window
determination, the L()/_full() loading closures, and the joint-name/group
bookkeeping) — same behavior, just packaged as a callable that returns a
dataclass instead of leaving ~85 loose variables in module scope.
"""

import sys
from collections import defaultdict

import numpy as np

from .common import DT, try_load
from .context import PlotContext


def load_context(folder: str, wbc_only: bool) -> PlotContext:
    joint_names = open(folder + '/joint_names.txt').readlines()
    num_joints = len(joint_names)

    # ── load all sensor / EKF logs ────────────────────────────────────────
    sens_raw = {}
    t_sensor = None
    for nm in ('pelvis_acc', 'pelvis_gyro', 'pelvis_rpy', 'pelvis_quat',
               'torso_acc', 'torso_gyro', 'torso_rpy', 'torso_quat',
               'odom_pos', 'odom_vel', 'odom_quat', 'odom_rpy',
               'joint_pos', 'joint_vel',
               'filtered_base_position', 'filtered_base_velocity',
               'filtered_base_quat', 'filtered_base_rpy', 'filtered_base_ang_vel',
               'filtered_joint_velocity'):
        d = try_load(folder, nm)
        if d is not None:
            sens_raw[nm] = d
            if t_sensor is None:
                t_sensor = d.shape[0]

    # ── determine WBC window length ───────────────────────────────────────
    ctrl_anchor = None
    for try_name in ('com_position', 'input_torque', 'wbc_accelerations', 'execution_time_wbc'):
        ctrl_anchor = try_load(folder, try_name)
        if ctrl_anchor is not None:
            break
    if ctrl_anchor is not None:
        total = ctrl_anchor.shape[0] if ctrl_anchor.ndim > 1 else len(ctrl_anchor)
    elif t_sensor is not None:
        total = t_sensor
    else:
        print("No data found in", folder)
        sys.exit(1)

    n_wbc = total
    t = np.linspace(0.0, DT * n_wbc, n_wbc)
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

    # ── sensor / EKF time axis ────────────────────────────────────────────
    if t_sensor is not None:
        ae = min(n_wbc, t_sensor) if wbc_only else t_sensor
        t_full = np.linspace(0.0, DT * ae, ae)

        def _full(nm):
            d = sens_raw.get(nm)
            if d is None:
                return None
            return d[len(d) - ae:] if wbc_only else d[:ae]
    else:
        t_full = t

        def _full(nm):
            return None

    # ── labels / joint groups ─────────────────────────────────────────────
    labels_xyz = ['x', 'y', 'z']
    labels_quat = ['w', 'x', 'y', 'z']
    labels_rpy = ['r', 'p', 'y']

    grouped_indices = defaultdict(list)
    kw_groups = {
        'hips': 'hip',
        'knees': 'knee',
        'ankles': 'ankle',
        'waist': 'waist',
        'shoulders': 'shoulder',
        'elbows': 'elbow',
        'wrists': 'wrist',
    }
    for i, jname in enumerate(joint_names):
        jname_lower = jname.strip().lower()
        for gname, kw in kw_groups.items():
            if kw in jname_lower:
                grouped_indices[gname].append(i)
                break

    # Build left-right joint pairs, skip waist
    jnames_stripped = [jn.strip() for jn in joint_names]
    lr_pairs = []
    for li, lname in enumerate(jnames_stripped):
        if not lname.startswith('left_') or 'waist' in lname:
            continue
        suffix = lname[len('left_'):]
        rname = 'right_' + suffix
        if rname in jnames_stripped:
            lr_pairs.append((li, jnames_stripped.index(rname), suffix))

    waist_idx = grouped_indices.get('waist', [])

    return PlotContext(
        folder=folder,
        wbc_only=wbc_only,
        n_wbc=n_wbc,
        t=t,
        t_full=t_full,
        iter_t=iter_t,
        joint_names=joint_names,
        num_joints=num_joints,
        jnames_stripped=jnames_stripped,
        lr_pairs=lr_pairs,
        waist_idx=waist_idx,
        labels_xyz=labels_xyz,
        labels_quat=labels_quat,
        labels_rpy=labels_rpy,
        grouped_indices=dict(grouped_indices),

        # WBC/controller-aligned
        com_position=L('com_position'),
        com_velocity=L('com_velocity'),
        zmp_position=L('zmp_position'),
        kf_com_position=L('kf_com_position'),
        kf_com_velocity=L('kf_com_velocity'),
        kf_zmp_position=L('kf_zmp_position'),
        des_com_position=L('des_com_position'),
        des_com_velocity=L('des_com_velocity'),
        des_zmp_position=L('des_zmp_position'),
        des_com_acceleration=L('des_com_acceleration'),
        q_ref_joints=L('q_ref_joints'),
        dq_ref_joints=L('dq_ref_joints'),
        joint_pos_wbc=L('joint_pos'),
        joint_vel_wbc=L('joint_vel'),
        input_torque=L('input_torque'),
        wbc_accelerations=L('wbc_accelerations'),
        estimated_force_lsole=L('estimated_force_lsole'),
        estimated_force_rsole=L('estimated_force_rsole'),
        wbc_force_lsole=L('wbc_force_lsole'),
        wbc_force_rsole=L('wbc_force_rsole'),
        wbc_contact_forces=L('wbc_contact_forces'),
        wbc_mu=L('wbc_mu'),
        wbc_joint_vel_limit_margin=L('wbc_joint_vel_limit_margin'),
        wbc_joint_pos_limit_margin=L('wbc_joint_pos_limit_margin'),
        wbc_worst_vel_limit_joint=L('wbc_worst_vel_limit_joint'),
        wbc_worst_vel_limit_is_upper=L('wbc_worst_vel_limit_is_upper'),
        wbc_worst_pos_limit_joint=L('wbc_worst_pos_limit_joint'),
        wbc_worst_pos_limit_is_upper=L('wbc_worst_pos_limit_is_upper'),
        mpc_zmp_box_center=L('mpc_zmp_box_center'),
        mpc_foot_constraint_size=L('mpc_foot_constraint_size'),
        p_lsole=L('p_lsole'),
        p_rsole=L('p_rsole'),
        v_lsole=L('v_lsole'),
        v_rsole=L('v_rsole'),
        p_lsole_des=L('p_lsole_des'),
        p_rsole_des=L('p_rsole_des'),
        v_lsole_des=L('v_lsole_des'),
        v_rsole_des=L('v_rsole_des'),
        lsole_orientation=L('lsole_orientation'),
        rsole_orientation=L('rsole_orientation'),
        des_lsole_orientation=L('des_lsole_orientation'),
        des_rsole_orientation=L('des_rsole_orientation'),
        torso_orientation=L('torso_orientation'),
        torso_angular_velocity=L('torso_angular_velocity'),
        des_torso_orientation=L('des_torso_orientation'),
        des_torso_angular_velocity=L('des_torso_angular_velocity'),
        pelvis_orientation=L('pelvis_orientation'),
        pelvis_angular_velocity=L('pelvis_angular_velocity'),
        des_pelvis_orientation=L('des_pelvis_orientation'),
        des_pelvis_angular_velocity=L('des_pelvis_angular_velocity'),
        execution_time_ekf=L('execution_time_ekf'),
        execution_time_kf=L('execution_time_kf'),
        execution_time_mpc=L('execution_time_mpc'),
        execution_time_wbc=L('execution_time_wbc'),
        execution_time_update=L('execution_time_update'),

        # sensor/EKF-aligned
        odometry_base_position=_full('odom_pos'),
        odometry_base_velocity=_full('odom_vel'),
        odometry_imu_orientation=_full('odom_quat'),
        odometry_imu_orientation_rpy=_full('odom_rpy'),
        measured_joint_position=_full('joint_pos'),
        measured_joint_velocity=_full('joint_vel'),
        measured_imu_pelvis_angular_velocity=_full('pelvis_gyro'),
        measured_imu_pelvis_accelerometer=_full('pelvis_acc'),
        measured_imu_pelvis_rpy=_full('pelvis_rpy'),
        measured_imu_pelvis_quaternion=_full('pelvis_quat'),
        measured_imu_torso_rpy=_full('torso_rpy'),
        measured_imu_torso_quaternion=_full('torso_quat'),
        measured_imu_torso_accelerometer=_full('torso_acc'),
        measured_imu_torso_angular_velocity=_full('torso_gyro'),
        filtered_base_position=_full('filtered_base_position'),
        filtered_base_velocity=_full('filtered_base_velocity'),
        filtered_base_orientation=_full('filtered_base_quat'),
        filtered_base_orientation_rpy=_full('filtered_base_rpy'),
        filtered_base_angular_velocity=_full('filtered_base_ang_vel'),
        filtered_joint_velocity=_full('filtered_joint_velocity'),
    )
