"""CoM AND ZMP.

Extracted from the original plot_joint_data.py section D (was lines 645-743).
"""

import os

import matplotlib.pyplot as plt

from ..common import _sub, plot_comparison, plot_components
from ..context import PlotContext


def run(ctx: PlotContext) -> None:
    plot_components(ctx.t, ctx.des_com_acceleration,
        [fr'Des CoM Acc ${l}$' for l in ctx.labels_xyz],
        'Desired CoM Acceleration', r'Acceleration [$\mathrm{m/s^2}$]',
        'images/task_com/references/des_com_acceleration_plot.png')
    plot_components(ctx.t, ctx.des_com_position,
        [fr'Des CoM Pos ${l}$' for l in ctx.labels_xyz],
        'Desired CoM Position', r'Position [$\mathrm{m}$]',
        'images/task_com/references/des_com_position_plot.png')
    plot_components(ctx.t, ctx.des_com_velocity,
        [fr'Des CoM Vel ${l}$' for l in ctx.labels_xyz],
        'Desired CoM Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/task_com/references/des_com_velocity_plot.png')
    plot_components(ctx.t, ctx.des_zmp_position,
        [fr'Des ZMP Pos ${l}$' for l in ctx.labels_xyz],
        'Desired ZMP Position', r'Position [$\mathrm{m}$]',
        'images/task_com/references/des_zmp_position_plot.png')

    # ── MPC ZMP-in-foot-box feasibility (x, y) ────────────────────────────────
    # mpc_zmp_box_center/size describe the box the ISMPC ZMP constraint keeps
    # the ZMP inside: [center - size/2, center + size/2] per axis. Compare
    # against the measured/estimated ZMP to see if/where the MPC is being
    # asked to place the ZMP outside the reachable foot polygon.
    if ctx.mpc_zmp_box_center is not None and ctx.mpc_foot_constraint_size is not None:
        n_box = min(len(ctx.mpc_zmp_box_center), len(ctx.mpc_foot_constraint_size))
        t_box = ctx.t[:n_box]
        center = ctx.mpc_zmp_box_center[:n_box]
        size = ctx.mpc_foot_constraint_size[:n_box]
        axis_labels = ['x', 'y']
        fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        for i, lbl in enumerate(axis_labels):
            half = size[:, i] / 2.0
            box_min = center[:, i] - half
            box_max = center[:, i] + half
            axs[i].fill_between(t_box, box_min, box_max, color='wheat', alpha=0.5,
                                 label='ZMP box (feasible)')
            if ctx.zmp_position is not None:
                n = min(n_box, len(ctx.zmp_position))
                axs[i].plot(t_box[:n], ctx.zmp_position[:n, i], linewidth=1.8,
                            label='Actual ZMP')
            if ctx.kf_zmp_position is not None:
                n = min(n_box, len(ctx.kf_zmp_position))
                axs[i].plot(t_box[:n], ctx.kf_zmp_position[:n, i], linewidth=1.5,
                            linestyle='--', label='Estimated ZMP (MPC input)')
            axs[i].set_ylabel(f'ZMP {lbl} [m]', fontsize=10)
            axs[i].set_title(f'ZMP vs Foot Box — {lbl}', fontsize=11)
            axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            axs[i].legend(fontsize=9, loc='best')
        axs[-1].set_xlabel('Time [s]', fontsize=11)
        fig.tight_layout()
        os.makedirs('images/task_com', exist_ok=True)
        fig.savefig('images/task_com/mpc_zmp_box_feasibility_plot.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    plot_components(ctx.t, _sub(ctx.des_zmp_position, ctx.kf_zmp_position),
        [fr'ZMP Error ${l}$' for l in ctx.labels_xyz],
        'ZMP Position Error', r'Position [$\mathrm{m}$]',
        'images/task_com/errors/error_zmp_position_plot.png')
    plot_components(ctx.t, _sub(ctx.des_com_position, ctx.kf_com_position),
        [fr'CoM Pos Error ${l}$' for l in ctx.labels_xyz],
        'CoM Position Error', r'Position [$\mathrm{m}$]',
        'images/task_com/errors/error_com_position_plot.png')
    plot_components(ctx.t, _sub(ctx.des_com_velocity, ctx.kf_com_velocity),
        [fr'CoM Vel Error ${l}$' for l in ctx.labels_xyz],
        'CoM Velocity Error', r'Velocity [$\mathrm{m/s}$]',
        'images/task_com/errors/error_com_velocity_plot.png')

    plot_comparison(ctx.t, ctx.kf_zmp_position, ctx.des_zmp_position,
        [fr'${l}$' for l in ctx.labels_xyz], 'ZMP Position', r'[$\mathrm{m}$]',
        'images/task_com/errors/comparison_zmp_position_plot.png')
    plot_comparison(ctx.t, ctx.kf_com_position, ctx.des_com_position,
        [fr'${l}$' for l in ctx.labels_xyz], 'CoM Position', r'[$\mathrm{m}$]',
        'images/task_com/errors/comparison_com_position_plot.png')
    plot_comparison(ctx.t, ctx.kf_com_velocity, ctx.des_com_velocity,
        [fr'${l}$' for l in ctx.labels_xyz], 'CoM Velocity', r'[$\mathrm{m/s}$]',
        'images/task_com/errors/comparison_com_velocity_plot.png')

    if ctx.kf_com_position is not None and ctx.kf_zmp_position is not None and \
            ctx.p_lsole is not None and ctx.p_rsole is not None:
        for axis, direction, fname in [(0, 'forward', 'motion_x'), (1, 'lateral', 'motion_y')]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(ctx.t, ctx.kf_com_position[:, axis], label=fr'CoM ${ctx.labels_xyz[axis]}$', linewidth=2.0)
            ax.plot(ctx.t, ctx.kf_zmp_position[:, axis], label=fr'ZMP ${ctx.labels_xyz[axis]}$', linewidth=2.0)
            ax.plot(ctx.t, ctx.p_lsole[:, axis], label=fr'Left foot ${ctx.labels_xyz[axis]}$', linewidth=2.0)
            ax.plot(ctx.t, ctx.p_rsole[:, axis], label=fr'Right foot ${ctx.labels_xyz[axis]}$', linewidth=2.0)
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel(fr'Position ${ctx.labels_xyz[axis]}$ [$\mathrm{{m}}$]', fontsize=11)
            ax.set_title(f'Motion in the {direction} direction', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(loc='best' if axis == 0 else 'upper left', frameon=True, fontsize=9)
            ax.tick_params(axis='both', labelsize=10)
            fig.tight_layout()
            os.makedirs('images/task_com', exist_ok=True)
            fig.savefig(f'images/task_com/{fname}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
