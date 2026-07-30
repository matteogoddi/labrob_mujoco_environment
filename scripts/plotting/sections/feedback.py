"""FEEDBACK: full sensor data, from tick 0, use t_full.

Extracted from the original plot_joint_data.py section H (was lines 966-1043).
"""

import numpy as np

from ..common import plot_components
from ..context import PlotContext


def run(ctx: PlotContext) -> None:
    plot_components(ctx.t_full, ctx.odometry_base_position,
        [fr'Odometry Pos ${l}$' for l in ctx.labels_xyz],
        'Odometry Base Position', r'Position [$\mathrm{m}$]',
        'images/feedback/odometry/odometry_base_position_plot.png')
    plot_components(ctx.t_full, ctx.odometry_base_velocity,
        [fr'Odometry Vel ${l}$' for l in ctx.labels_xyz],
        'Odometry Base Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/feedback/odometry/odometry_base_velocity_plot.png')
    plot_components(ctx.t_full, ctx.odometry_imu_orientation,
        [fr'Odometry IMU Orient ${l}$' for l in ctx.labels_quat],
        'Odometry IMU Orientation Quat', r'Orientation [quat]',
        'images/feedback/odometry/odometry_imu_orientation_quat_plot.png')
    plot_components(ctx.t_full, ctx.odometry_imu_orientation_rpy,
        [fr'Odometry IMU Orient ${l}$' for l in ctx.labels_rpy],
        'Odometry IMU Orientation RPY', r'Orientation [$\mathrm{rad}$]',
        'images/feedback/odometry/odometry_imu_orientation_rpy_plot.png')

    plot_components(ctx.t_full, ctx.measured_imu_pelvis_angular_velocity,
        [fr'Pelvis AngVel ${l}$' for l in ctx.labels_xyz],
        'Pelvis IMU – Gyroscope', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/feedback/imu_pelvis/pelvis_imu_angular_velocity_plot.png')
    plot_components(ctx.t_full, ctx.measured_imu_pelvis_accelerometer,
        [fr'Pelvis Acc ${l}$' for l in ctx.labels_xyz],
        'Pelvis IMU – Accelerometer', r'Acceleration [$\mathrm{m/s^2}$]',
        'images/feedback/imu_pelvis/pelvis_imu_acceleration_plot.png')
    plot_components(ctx.t_full, ctx.measured_imu_pelvis_rpy,
        [fr'Pelvis RPY ${l}$' for l in ctx.labels_rpy],
        'Pelvis IMU – RPY', r'RPY [$\mathrm{rad}$]',
        'images/feedback/imu_pelvis/pelvis_imu_rpy_plot.png')
    plot_components(ctx.t_full, ctx.measured_imu_pelvis_quaternion,
        [fr'Pelvis Quat ${l}$' for l in ctx.labels_quat],
        'Pelvis IMU – Quaternion', r'Quaternion',
        'images/feedback/imu_pelvis/pelvis_imu_quaternion_plot.png')

    plot_components(ctx.t_full, ctx.measured_imu_torso_rpy,
        [fr'Torso RPY ${l}$' for l in ctx.labels_rpy],
        'Torso IMU – RPY', r'RPY [$\mathrm{rad}$]',
        'images/feedback/imu_torso/torso_imu_rpy_plot.png')
    plot_components(ctx.t_full, ctx.measured_imu_torso_quaternion,
        [fr'Torso Quat ${l}$' for l in ctx.labels_quat],
        'Torso IMU – Quaternion', r'Quaternion',
        'images/feedback/imu_torso/torso_imu_quaternion_plot.png')
    plot_components(ctx.t_full, ctx.measured_imu_torso_accelerometer,
        [fr'Torso Acc ${l}$' for l in ctx.labels_xyz],
        'Torso IMU – Accelerometer', r'Acceleration [$\mathrm{m/s^2}$]',
        'images/feedback/imu_torso/torso_imu_acceleration_plot.png')
    plot_components(ctx.t_full, ctx.measured_imu_torso_angular_velocity,
        [fr'Torso AngVel ${l}$' for l in ctx.labels_xyz],
        'Torso IMU – Gyroscope', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/feedback/imu_torso/torso_imu_angular_velocity_plot.png')

    for li, ri, suffix in ctx.lr_pairs:
        if ctx.measured_joint_position is not None:
            plot_components(ctx.t_full,
                np.column_stack([ctx.measured_joint_position[:, li], ctx.measured_joint_position[:, ri]]),
                [f'Left {suffix}', f'Right {suffix}'],
                f'Joint Position – {suffix}', r'Position [$\mathrm{rad}$]',
                f'images/feedback/joints/positions/{suffix}_position_plot.png')
        if ctx.measured_joint_velocity is not None:
            plot_components(ctx.t_full,
                np.column_stack([ctx.measured_joint_velocity[:, li], ctx.measured_joint_velocity[:, ri]]),
                [f'Left {suffix}', f'Right {suffix}'],
                f'Joint Velocity – {suffix}', r'Velocity [$\mathrm{rad/s}$]',
                f'images/feedback/joints/velocities/{suffix}_velocity_plot.png')
    if ctx.waist_idx:
        if ctx.measured_joint_position is not None:
            plot_components(ctx.t_full, ctx.measured_joint_position[:, ctx.waist_idx],
                [ctx.jnames_stripped[i] for i in ctx.waist_idx],
                'Joint Position – waist', r'Position [$\mathrm{rad}$]',
                'images/feedback/joints/positions/waist_position_plot.png')
        if ctx.measured_joint_velocity is not None:
            plot_components(ctx.t_full, ctx.measured_joint_velocity[:, ctx.waist_idx],
                [ctx.jnames_stripped[i] for i in ctx.waist_idx],
                'Joint Velocity – waist', r'Velocity [$\mathrm{rad/s}$]',
                'images/feedback/joints/velocities/waist_velocity_plot.png')
