"""FEET: sole position/velocity tracking, and sole orientation tracking.

Position/velocity part extracted from the original plot_joint_data.py
section E (was lines 745-789).

Orientation part: the original file only had a left-vs-right *roll*
diagnostic for the soles (inside section G "TASK ORIENTATION", writing into
images/task_soles/orientation/ despite that section's own header). This is
now folded into feet.py (which owns task_soles end to end) and extended to
match how torso/pelvis orientation is treated elsewhere (task_orientation.py):
a proper actual-vs-desired comparison + error plot for all three Euler
components (roll, pitch, yaw), not just roll. The original left-vs-right
roll-only diagnostic is kept alongside it (still useful specifically for
spotting left/right ankle-roll asymmetry), not removed.
"""

import numpy as np

from ..common import _sub, plot_comparison, plot_components
from ..context import PlotContext


def run(ctx: PlotContext) -> None:
    # ── position / velocity ───────────────────────────────────────────────
    plot_components(ctx.t, ctx.p_lsole_des,
        [fr'Des L Sole ${l}$' for l in ctx.labels_xyz],
        'Desired Left Sole Position', r'Position [$\mathrm{m}$]',
        'images/task_soles/position/desired_left_sole_position_plot.png')
    plot_components(ctx.t, ctx.p_rsole_des,
        [fr'Des R Sole ${l}$' for l in ctx.labels_xyz],
        'Desired Right Sole Position', r'Position [$\mathrm{m}$]',
        'images/task_soles/position/desired_right_sole_position_plot.png')

    plot_components(ctx.t, _sub(ctx.p_lsole_des, ctx.p_lsole),
        [fr'L Sole Pos Error ${l}$' for l in ctx.labels_xyz],
        'Error – Left Sole Position', r'Position [$\mathrm{m}$]',
        'images/task_soles/position/error_left_sole_position_plot.png')
    plot_components(ctx.t, _sub(ctx.p_rsole_des, ctx.p_rsole),
        [fr'R Sole Pos Error ${l}$' for l in ctx.labels_xyz],
        'Error – Right Sole Position', r'Position [$\mathrm{m}$]',
        'images/task_soles/position/error_right_sole_position_plot.png')
    plot_components(ctx.t, _sub(ctx.v_lsole_des, ctx.v_lsole),
        [fr'L Sole Vel Error ${l}$' for l in ctx.labels_xyz],
        'Error – Left Sole Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/task_soles/position/error_left_sole_velocity_plot.png')
    plot_components(ctx.t, _sub(ctx.v_rsole_des, ctx.v_rsole),
        [fr'R Sole Vel Error ${l}$' for l in ctx.labels_xyz],
        'Error – Right Sole Velocity', r'Velocity [$\mathrm{m/s}$]',
        'images/task_soles/position/error_right_sole_velocity_plot.png')

    plot_comparison(ctx.t, ctx.p_lsole, ctx.p_lsole_des,
        [fr'${l}$' for l in ctx.labels_xyz], 'Left Sole Position', r'[$\mathrm{m}$]',
        'images/task_soles/position/comparison_left_sole_position_plot.png')
    plot_comparison(ctx.t, ctx.p_rsole, ctx.p_rsole_des,
        [fr'${l}$' for l in ctx.labels_xyz], 'Right Sole Position', r'[$\mathrm{m}$]',
        'images/task_soles/position/comparison_right_sole_position_plot.png')
    plot_comparison(ctx.t, ctx.v_lsole, ctx.v_lsole_des,
        [fr'${l}$' for l in ctx.labels_xyz], 'Left Sole Velocity', r'[$\mathrm{m/s}$]',
        'images/task_soles/position/comparison_left_sole_velocity_plot.png')
    plot_comparison(ctx.t, ctx.v_rsole, ctx.v_rsole_des,
        [fr'${l}$' for l in ctx.labels_xyz], 'Right Sole Velocity', r'[$\mathrm{m/s}$]',
        'images/task_soles/position/comparison_right_sole_velocity_plot.png')

    # ── orientation: full actual-vs-desired, roll/pitch/yaw (like torso/pelvis) ──
    # lsole/rsole_orientation columns are (roll, pitch, yaw) via eulerAngles(0,1,2).
    rpy_labels = ['Roll', 'Pitch', 'Yaw']
    if ctx.lsole_orientation is not None and ctx.des_lsole_orientation is not None:
        plot_comparison(ctx.t, ctx.lsole_orientation, ctx.des_lsole_orientation,
            rpy_labels, 'Left Sole Orientation', r'[$\mathrm{rad}$]',
            'images/task_soles/orientation/left_sole_orientation_comparison_plot.png')
        plot_components(ctx.t, _sub(ctx.lsole_orientation, ctx.des_lsole_orientation),
            ['Roll Error', 'Pitch Error', 'Yaw Error'],
            'Left Sole Orientation Error', r'Orientation [$\mathrm{rad}$]',
            'images/task_soles/orientation/left_sole_orientation_error_plot.png')
    if ctx.rsole_orientation is not None and ctx.des_rsole_orientation is not None:
        plot_comparison(ctx.t, ctx.rsole_orientation, ctx.des_rsole_orientation,
            rpy_labels, 'Right Sole Orientation', r'[$\mathrm{rad}$]',
            'images/task_soles/orientation/right_sole_orientation_comparison_plot.png')
        plot_components(ctx.t, _sub(ctx.rsole_orientation, ctx.des_rsole_orientation),
            ['Roll Error', 'Pitch Error', 'Yaw Error'],
            'Right Sole Orientation Error', r'Orientation [$\mathrm{rad}$]',
            'images/task_soles/orientation/right_sole_orientation_error_plot.png')

    # ── sole roll, left vs right (diagnostic for ankle-roll asymmetry) ────────
    if ctx.lsole_orientation is not None and ctx.rsole_orientation is not None:
        plot_components(ctx.t,
            np.column_stack([ctx.lsole_orientation[:, 0], ctx.rsole_orientation[:, 0]]),
            ['Left', 'Right'], 'Sole Roll – Measured (Left vs Right)', r'Roll [$\mathrm{rad}$]',
            'images/task_soles/orientation/roll_measured_left_vs_right_plot.png')
    if ctx.des_lsole_orientation is not None and ctx.des_rsole_orientation is not None:
        plot_components(ctx.t,
            np.column_stack([ctx.des_lsole_orientation[:, 0], ctx.des_rsole_orientation[:, 0]]),
            ['Left', 'Right'], 'Sole Roll – Desired (Left vs Right)', r'Roll [$\mathrm{rad}$]',
            'images/task_soles/orientation/roll_desired_left_vs_right_plot.png')
    if all(x is not None for x in (ctx.lsole_orientation, ctx.rsole_orientation,
                                    ctx.des_lsole_orientation, ctx.des_rsole_orientation)):
        roll_err_l = _sub(ctx.lsole_orientation[:, 0:1], ctx.des_lsole_orientation[:, 0:1])
        roll_err_r = _sub(ctx.rsole_orientation[:, 0:1], ctx.des_rsole_orientation[:, 0:1])
        n_roll = min(len(roll_err_l), len(roll_err_r))
        plot_components(ctx.t,
            np.column_stack([roll_err_l[:n_roll], roll_err_r[:n_roll]]),
            ['Left', 'Right'], 'Sole Roll Error – Measured vs Desired (Left vs Right)',
            r'Roll Error [$\mathrm{rad}$]',
            'images/task_soles/orientation/roll_error_left_vs_right_plot.png')
