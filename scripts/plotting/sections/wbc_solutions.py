"""WBC SOLUTIONS: torques, base accelerations, foot wrenches.

Extracted from the original plot_joint_data.py section B (was lines 367-419).

Note: the foot-wrench plots below write into images/contact_forces/, not
images/wbc_solutions/ — a pre-existing naming inconsistency from the
original file, intentionally left as-is (see refactor plan, not fixed in
this pass to keep the output tree diffable against the pre-refactor script).
"""

import numpy as np

from ..common import plot_components
from ..context import PlotContext


def run(ctx: PlotContext) -> None:
    if ctx.input_torque is not None:
        for li, ri, suffix in ctx.lr_pairs:
            plot_components(ctx.t,
                np.column_stack([ctx.input_torque[:, li], ctx.input_torque[:, ri]]),
                [f'Left {suffix}', f'Right {suffix}'],
                f'WBC Joint Torque – {suffix}', 'Torque [Nm]',
                f'images/wbc_solutions/torques/{suffix}_torque.png')
        if ctx.waist_idx:
            plot_components(ctx.t, ctx.input_torque[:, ctx.waist_idx],
                [ctx.jnames_stripped[i] for i in ctx.waist_idx],
                'WBC Joint Torque – waist', 'Torque [Nm]',
                'images/wbc_solutions/torques/waist_torque.png')

    if ctx.wbc_accelerations is not None:
        plot_components(ctx.t, ctx.wbc_accelerations[:, :3],
            [f'Linear acc {l}' for l in ctx.labels_xyz],
            'WBC Base Linear Acceleration', r'Acceleration [m/s²]',
            'images/wbc_solutions/accelerations/base_linear_acceleration.png')
        plot_components(ctx.t, ctx.wbc_accelerations[:, 3:6],
            [f'Angular acc {l}' for l in ctx.labels_xyz],
            'WBC Base Angular Acceleration', r'Acceleration [rad/s²]',
            'images/wbc_solutions/accelerations/base_angular_acceleration.png')
        for li, ri, suffix in ctx.lr_pairs:
            plot_components(ctx.t,
                np.column_stack([ctx.wbc_accelerations[:, li + 6], ctx.wbc_accelerations[:, ri + 6]]),
                [f'Left {suffix}', f'Right {suffix}'],
                f'WBC Joint Acceleration – {suffix}', r'Acceleration [rad/s²]',
                f'images/wbc_solutions/accelerations/{suffix}_acceleration.png')
        if ctx.waist_idx:
            plot_components(ctx.t, ctx.wbc_accelerations[:, [i + 6 for i in ctx.waist_idx]],
                [ctx.jnames_stripped[i] for i in ctx.waist_idx],
                'WBC Joint Acceleration – waist', r'Acceleration [rad/s²]',
                'images/wbc_solutions/accelerations/waist_acceleration.png')

    labels_wrench = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    plot_components(ctx.t, ctx.wbc_force_lsole, labels_wrench,
        'WBC Optimal Left Foot Wrench', r'Force [N] / Torque [Nm]',
        'images/contact_forces/wbc_force_left_sole.png')
    plot_components(ctx.t, ctx.wbc_force_rsole, labels_wrench,
        'WBC Optimal Right Foot Wrench', r'Force [N] / Torque [Nm]',
        'images/contact_forces/wbc_force_right_sole.png')
    plot_components(ctx.t, ctx.estimated_force_lsole,
        [f'Left sole force {l}' for l in ctx.labels_xyz],
        'Estimated Forces on Left Sole', 'Force [N]',
        'images/contact_forces/estimated_force_left_sole.png')
    plot_components(ctx.t, ctx.estimated_force_rsole,
        [f'Right sole force {l}' for l in ctx.labels_xyz],
        'Estimated Forces on Right Sole', 'Force [N]',
        'images/contact_forces/estimated_force_right_sole.png')
