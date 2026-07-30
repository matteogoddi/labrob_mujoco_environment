"""PlotContext: the single flat data container passed to every section.

All fields are populated once by plotting.loaders.load_context(). Kept flat
(not nested by subsystem) because sections freely mix data from different
sources (e.g. the EKF section uses both EKF signals and the "meta" joint
grouping fields) — nesting would add friction without real encapsulation
benefit. A dataclass (not a plain dict) so a typo'd field name is an
immediate AttributeError instead of a silently-None dict .get() (which,
in this codebase, is already a legitimate value for "log not present").
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PlotContext:
    # ── meta ──────────────────────────────────────────────────────────────
    folder: str
    wbc_only: bool
    n_wbc: int
    t: np.ndarray
    t_full: np.ndarray
    iter_t: np.ndarray
    joint_names: List[str]
    num_joints: int
    jnames_stripped: List[str]
    lr_pairs: List[Tuple[int, int, str]]
    waist_idx: List[int]
    labels_xyz: List[str] = field(default_factory=lambda: ['x', 'y', 'z'])
    labels_quat: List[str] = field(default_factory=lambda: ['w', 'x', 'y', 'z'])
    labels_rpy: List[str] = field(default_factory=lambda: ['r', 'p', 'y'])
    grouped_indices: Dict[str, List[int]] = field(default_factory=dict)

    # ── WBC/controller-aligned signals (loaded via L(), window = n_wbc) ────
    com_position: Optional[np.ndarray] = None
    com_velocity: Optional[np.ndarray] = None
    zmp_position: Optional[np.ndarray] = None
    kf_com_position: Optional[np.ndarray] = None
    kf_com_velocity: Optional[np.ndarray] = None
    kf_zmp_position: Optional[np.ndarray] = None
    des_com_position: Optional[np.ndarray] = None
    des_com_velocity: Optional[np.ndarray] = None
    des_zmp_position: Optional[np.ndarray] = None
    des_com_acceleration: Optional[np.ndarray] = None
    q_ref_joints: Optional[np.ndarray] = None
    dq_ref_joints: Optional[np.ndarray] = None
    joint_pos_wbc: Optional[np.ndarray] = None
    joint_vel_wbc: Optional[np.ndarray] = None
    input_torque: Optional[np.ndarray] = None
    wbc_accelerations: Optional[np.ndarray] = None
    estimated_force_lsole: Optional[np.ndarray] = None
    estimated_force_rsole: Optional[np.ndarray] = None
    wbc_force_lsole: Optional[np.ndarray] = None
    wbc_force_rsole: Optional[np.ndarray] = None
    wbc_contact_forces: Optional[np.ndarray] = None
    wbc_mu: Optional[np.ndarray] = None
    wbc_joint_vel_limit_margin: Optional[np.ndarray] = None
    wbc_joint_pos_limit_margin: Optional[np.ndarray] = None
    wbc_worst_vel_limit_joint: Optional[np.ndarray] = None
    wbc_worst_vel_limit_is_upper: Optional[np.ndarray] = None
    wbc_worst_pos_limit_joint: Optional[np.ndarray] = None
    wbc_worst_pos_limit_is_upper: Optional[np.ndarray] = None
    mpc_zmp_box_center: Optional[np.ndarray] = None
    mpc_foot_constraint_size: Optional[np.ndarray] = None
    p_lsole: Optional[np.ndarray] = None
    p_rsole: Optional[np.ndarray] = None
    v_lsole: Optional[np.ndarray] = None
    v_rsole: Optional[np.ndarray] = None
    p_lsole_des: Optional[np.ndarray] = None
    p_rsole_des: Optional[np.ndarray] = None
    v_lsole_des: Optional[np.ndarray] = None
    v_rsole_des: Optional[np.ndarray] = None
    lsole_orientation: Optional[np.ndarray] = None
    rsole_orientation: Optional[np.ndarray] = None
    des_lsole_orientation: Optional[np.ndarray] = None
    des_rsole_orientation: Optional[np.ndarray] = None
    torso_orientation: Optional[np.ndarray] = None
    torso_angular_velocity: Optional[np.ndarray] = None
    des_torso_orientation: Optional[np.ndarray] = None
    des_torso_angular_velocity: Optional[np.ndarray] = None
    pelvis_orientation: Optional[np.ndarray] = None
    pelvis_angular_velocity: Optional[np.ndarray] = None
    des_pelvis_orientation: Optional[np.ndarray] = None
    des_pelvis_angular_velocity: Optional[np.ndarray] = None
    execution_time_ekf: Optional[np.ndarray] = None
    execution_time_kf: Optional[np.ndarray] = None
    execution_time_mpc: Optional[np.ndarray] = None
    execution_time_wbc: Optional[np.ndarray] = None
    execution_time_update: Optional[np.ndarray] = None

    # ── sensor/EKF-aligned signals (loaded via _full(), window = t_full) ───
    odometry_base_position: Optional[np.ndarray] = None
    odometry_base_velocity: Optional[np.ndarray] = None
    odometry_imu_orientation: Optional[np.ndarray] = None
    odometry_imu_orientation_rpy: Optional[np.ndarray] = None
    measured_joint_position: Optional[np.ndarray] = None
    measured_joint_velocity: Optional[np.ndarray] = None
    measured_imu_pelvis_angular_velocity: Optional[np.ndarray] = None
    measured_imu_pelvis_accelerometer: Optional[np.ndarray] = None
    measured_imu_pelvis_rpy: Optional[np.ndarray] = None
    measured_imu_pelvis_quaternion: Optional[np.ndarray] = None
    measured_imu_torso_rpy: Optional[np.ndarray] = None
    measured_imu_torso_quaternion: Optional[np.ndarray] = None
    measured_imu_torso_accelerometer: Optional[np.ndarray] = None
    measured_imu_torso_angular_velocity: Optional[np.ndarray] = None
    filtered_base_position: Optional[np.ndarray] = None
    filtered_base_velocity: Optional[np.ndarray] = None
    filtered_base_orientation: Optional[np.ndarray] = None
    filtered_base_orientation_rpy: Optional[np.ndarray] = None
    filtered_base_angular_velocity: Optional[np.ndarray] = None
    filtered_joint_velocity: Optional[np.ndarray] = None
