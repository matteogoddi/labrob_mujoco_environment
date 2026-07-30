"""EKF: all plots use t_full (aligned with feedback, not the WBC window).

Extracted from the original plot_joint_data.py section F (was lines 791-904).
"""

import numpy as np

from ..common import _sub, plot_aggregate_joint_error, plot_comparison, plot_components
from ..context import PlotContext


def run(ctx: PlotContext) -> None:
    # ── base state ────────────────────────────────────────────────────────
    plot_components(ctx.t_full, ctx.filtered_base_position,
        [fr'Base Pos ${l}$' for l in ctx.labels_xyz],
        'Filtered Base Position', r'Position [$\mathrm{m}$]',
        'images/ekf/base/base_position_plot.png')
    plot_components(ctx.t_full, ctx.filtered_base_velocity,
        [fr'Base Vel ${l}$' for l in ctx.labels_xyz],
        'Filtered Base Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/ekf/base/base_velocity_plot.png')
    plot_components(ctx.t_full, ctx.filtered_base_orientation,
        [fr'Base Orient ${l}$' for l in ctx.labels_quat],
        'Filtered Base Orientation Quat', r'Orientation [quat]',
        'images/ekf/base/base_orientation_quat_plot.png')
    plot_components(ctx.t_full, ctx.filtered_base_orientation_rpy,
        [fr'Base Orient ${l}$' for l in ctx.labels_rpy],
        'Filtered Base Orientation RPY', r'Orientation [$\mathrm{rad}$]',
        'images/ekf/base/base_orientation_rpy_plot.png')
    plot_components(ctx.t_full, ctx.filtered_base_angular_velocity,
        [fr'Base AngVel ${l}$' for l in ctx.labels_xyz],
        'Filtered Base Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/ekf/base/base_angular_velocity_plot.png')

    # ── base vs odometry comparison ───────────────────────────────────────────
    plot_comparison(ctx.t_full, ctx.filtered_base_position, ctx.odometry_base_position,
        [fr'${l}$' for l in ctx.labels_xyz], 'Base Position', r'[$\mathrm{m}$]',
        'images/ekf/base/errors/comparison_base_position_plot.png')
    plot_comparison(ctx.t_full, ctx.filtered_base_velocity, ctx.odometry_base_velocity,
        [fr'${l}$' for l in ctx.labels_xyz], 'Base Velocity', r'[$\mathrm{m/s}$]',
        'images/ekf/base/errors/comparison_base_velocity_plot.png')
    plot_comparison(ctx.t_full, ctx.filtered_base_orientation, ctx.odometry_imu_orientation,
        [fr'${l}$' for l in ctx.labels_quat], 'Base Orientation Quat', r'[quat]',
        'images/ekf/base/errors/comparison_base_orientation_plot.png')
    plot_comparison(ctx.t_full, ctx.filtered_base_orientation_rpy, ctx.odometry_imu_orientation_rpy,
        [fr'${l}$' for l in ctx.labels_rpy], 'Base Orientation RPY', r'[$\mathrm{rad}$]',
        'images/ekf/base/errors/comparison_base_orientation_rpy_plot.png')
    plot_comparison(ctx.t_full, ctx.filtered_base_angular_velocity, ctx.measured_imu_pelvis_angular_velocity,
        [fr'${l}$' for l in ctx.labels_xyz], 'Base Angular Velocity', r'[$\mathrm{rad/s}$]',
        'images/ekf/base/errors/comparison_base_angular_velocity_plot.png')

    plot_components(ctx.t_full, _sub(ctx.filtered_base_position, ctx.odometry_base_position),
        [fr'Error Pos ${l}$' for l in ctx.labels_xyz],
        'Error – Filtered vs Odometry Base Position', r'Position [$\mathrm{m}$]',
        'images/ekf/base/errors/error_base_position_plot.png')
    plot_components(ctx.t_full, _sub(ctx.filtered_base_velocity, ctx.odometry_base_velocity),
        [fr'Error Vel ${l}$' for l in ctx.labels_xyz],
        'Error – Filtered vs Odometry Base Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/ekf/base/errors/error_base_velocity_plot.png')
    plot_components(ctx.t_full, _sub(ctx.filtered_base_orientation, ctx.odometry_imu_orientation),
        [fr'Error Orient ${l}$' for l in ctx.labels_quat],
        'Error – Filtered vs Odometry Orientation Quat', r'Orientation [quat]',
        'images/ekf/base/errors/error_base_orientation_quat_plot.png')
    plot_components(ctx.t_full, _sub(ctx.filtered_base_orientation_rpy, ctx.odometry_imu_orientation_rpy),
        [fr'Error Orient ${l}$' for l in ctx.labels_rpy],
        'Error – Filtered vs Odometry Orientation RPY', r'Orientation [$\mathrm{rad}$]',
        'images/ekf/base/errors/error_base_orientation_rpy_plot.png')
    plot_components(ctx.t_full, _sub(ctx.filtered_base_angular_velocity, ctx.measured_imu_pelvis_angular_velocity),
        [fr'Error AngVel ${l}$' for l in ctx.labels_xyz],
        'Error – Filtered vs Measured Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/ekf/base/errors/error_base_angular_velocity_plot.png')

    # ── joint velocity (EKF vs measured, left-right pairs) ───────────────────
    if ctx.measured_joint_velocity is not None and ctx.filtered_joint_velocity is not None:
        nv = min(len(ctx.measured_joint_velocity), len(ctx.filtered_joint_velocity))
        for li, ri, suffix in ctx.lr_pairs:
            plot_components(ctx.t_full[:nv],
                np.column_stack([ctx.filtered_joint_velocity[:nv, li], ctx.filtered_joint_velocity[:nv, ri]]),
                [f'Left {suffix}', f'Right {suffix}'],
                f'EKF Joint Velocity – {suffix}', r'Velocity [$\mathrm{rad/s}$]',
                f'images/ekf/joints/velocities/{suffix}_velocity_plot.png')

            plot_components(ctx.t_full[:nv],
                np.column_stack([
                    ctx.measured_joint_velocity[:nv, li] - ctx.filtered_joint_velocity[:nv, li],
                    ctx.measured_joint_velocity[:nv, ri] - ctx.filtered_joint_velocity[:nv, ri],
                ]),
                [f'Err Left {suffix}', f'Err Right {suffix}'],
                f'EKF Velocity Error – {suffix}', r'Velocity [$\mathrm{rad/s}$]',
                f'images/ekf/joints/error/velocities/error_{suffix}_velocity_plot.png')

        if ctx.waist_idx:
            plot_components(ctx.t_full[:nv], ctx.filtered_joint_velocity[:nv, ctx.waist_idx],
                [ctx.jnames_stripped[i] for i in ctx.waist_idx],
                'EKF Joint Velocity – waist', r'Velocity [$\mathrm{rad/s}$]',
                'images/ekf/joints/velocities/waist_velocity_plot.png')
            plot_components(ctx.t_full[:nv],
                _sub(ctx.measured_joint_velocity[:nv, ctx.waist_idx], ctx.filtered_joint_velocity[:nv, ctx.waist_idx]),
                [ctx.jnames_stripped[i] for i in ctx.waist_idx],
                'EKF Velocity Error – waist', r'Velocity [$\mathrm{rad/s}$]',
                'images/ekf/joints/error/velocities/error_waist_velocity_plot.png')

        plot_aggregate_joint_error(
            ctx.t_full, ctx.filtered_joint_velocity, ctx.measured_joint_velocity,
            [jn.strip() for jn in ctx.joint_names],
            'Error EKF vs Measured – Joint Velocity', r'Velocity [$\mathrm{rad/s}$]',
            'images/ekf/joints/error/error_joint_velocity_plot.png')
