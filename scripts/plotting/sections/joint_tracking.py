"""JOINT REFERENCE TRACKING: q_ref_joints / dq_ref_joints vs measured feedback.

Extracted from the original plot_joint_data.py section A (was lines 283-365).
"""

from ..common import _sub, plot_aggregate_joint_error, plot_comparison, plot_components
from ..context import PlotContext


def run(ctx: PlotContext) -> None:
    if ctx.q_ref_joints is not None and ctx.joint_pos_wbc is not None:
        for li, ri, suffix in ctx.lr_pairs:
            plot_comparison(ctx.t,
                ctx.joint_pos_wbc[:, [li, ri]], ctx.q_ref_joints[:, [li, ri]],
                [f'Left {suffix}', f'Right {suffix}'],
                f'Joint Position Tracking – {suffix}', r'Position [rad]',
                f'images/wbc_solutions/joint_tracking/{suffix}_position_comparison.png')
        if ctx.waist_idx:
            plot_comparison(ctx.t,
                ctx.joint_pos_wbc[:, ctx.waist_idx], ctx.q_ref_joints[:, ctx.waist_idx],
                [ctx.jnames_stripped[i] for i in ctx.waist_idx],
                'Joint Position Tracking – waist', r'Position [rad]',
                'images/wbc_solutions/joint_tracking/waist_position_comparison.png')

        plot_aggregate_joint_error(
            ctx.t, ctx.q_ref_joints, ctx.joint_pos_wbc, ctx.jnames_stripped,
            'Joint Position Error (q_ref − measured), all joints', r'Position Error [rad]',
            'images/wbc_solutions/joint_tracking/error_joint_position_plot.png')

        for i in range(ctx.num_joints):
            plot_components(ctx.t, _sub(ctx.q_ref_joints[:, i:i + 1], ctx.joint_pos_wbc[:, i:i + 1]),
                [f'{ctx.jnames_stripped[i]} Pos Error'],
                f'Position Error – {ctx.jnames_stripped[i]}', r'Error [rad]',
                f'images/wbc_solutions/joint_tracking/errors/{ctx.jnames_stripped[i]}_position_error.png')

    if ctx.dq_ref_joints is not None and ctx.joint_vel_wbc is not None:
        for li, ri, suffix in ctx.lr_pairs:
            plot_comparison(ctx.t,
                ctx.joint_vel_wbc[:, [li, ri]], ctx.dq_ref_joints[:, [li, ri]],
                [f'Left {suffix}', f'Right {suffix}'],
                f'Joint Velocity Tracking – {suffix}', r'Velocity [rad/s]',
                f'images/wbc_solutions/joint_tracking/{suffix}_velocity_comparison.png')
        if ctx.waist_idx:
            plot_comparison(ctx.t,
                ctx.joint_vel_wbc[:, ctx.waist_idx], ctx.dq_ref_joints[:, ctx.waist_idx],
                [ctx.jnames_stripped[i] for i in ctx.waist_idx],
                'Joint Velocity Tracking – waist', r'Velocity [rad/s]',
                'images/wbc_solutions/joint_tracking/waist_velocity_comparison.png')

        plot_aggregate_joint_error(
            ctx.t, ctx.dq_ref_joints, ctx.joint_vel_wbc, ctx.jnames_stripped,
            'Joint Velocity Error (dq_ref − measured), all joints', r'Velocity Error [rad/s]',
            'images/wbc_solutions/joint_tracking/error_joint_velocity_plot.png')

        for i in range(ctx.num_joints):
            plot_components(ctx.t, _sub(ctx.dq_ref_joints[:, i:i + 1], ctx.joint_vel_wbc[:, i:i + 1]),
                [f'{ctx.jnames_stripped[i]} Vel Error'],
                f'Velocity Error – {ctx.jnames_stripped[i]}', r'Error [rad/s]',
                f'images/wbc_solutions/joint_tracking/errors/{ctx.jnames_stripped[i]}_velocity_error.png')
