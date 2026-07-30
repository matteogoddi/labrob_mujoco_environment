"""TASK ORIENTATION: torso + pelvis (WBC signals, use t).

Extracted from the original plot_joint_data.py section G (was lines 906-964).
The sole-roll diagnostics that used to live at the end of this section
(writing into images/task_soles/orientation/ despite this section's own
header) have moved to sections/feet.py, which now owns task_soles end to
end (position + orientation) — see feet.py's module docstring.
"""

from ..common import _sub, plot_comparison, plot_components
from ..context import PlotContext


def run(ctx: PlotContext) -> None:
    plot_comparison(ctx.t, ctx.torso_orientation, ctx.des_torso_orientation,
        ['Roll', 'Pitch', 'Yaw'], 'Torso Orientation', r'[$\mathrm{rad}$]',
        'images/task_orientation/torso/torso_orientation_comparison_plot.png')
    plot_components(ctx.t, _sub(ctx.torso_orientation, ctx.des_torso_orientation),
        ['Roll Error', 'Pitch Error', 'Yaw Error'],
        'Torso Orientation Error', r'Orientation [$\mathrm{rad}$]',
        'images/task_orientation/torso/torso_orientation_error_plot.png')
    plot_components(ctx.t, ctx.torso_angular_velocity,
        [f'AngVel {l}' for l in ctx.labels_xyz],
        'Torso Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/task_orientation/torso/torso_angular_velocity_plot.png')
    plot_components(ctx.t, _sub(ctx.torso_angular_velocity, ctx.des_torso_angular_velocity),
        [f'AngVel Error {l}' for l in ctx.labels_xyz],
        'Torso Angular Velocity Error', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/task_orientation/torso/torso_angular_velocity_error_plot.png')

    plot_comparison(ctx.t, ctx.pelvis_orientation, ctx.des_pelvis_orientation,
        ['Roll', 'Pitch', 'Yaw'], 'Pelvis Orientation', r'[$\mathrm{rad}$]',
        'images/task_orientation/pelvis/pelvis_orientation_comparison_plot.png')
    plot_components(ctx.t, _sub(ctx.pelvis_orientation, ctx.des_pelvis_orientation),
        ['Roll Error', 'Pitch Error', 'Yaw Error'],
        'Pelvis Orientation Error', r'Orientation [$\mathrm{rad}$]',
        'images/task_orientation/pelvis/pelvis_orientation_error_plot.png')
    plot_components(ctx.t, ctx.pelvis_angular_velocity,
        [f'AngVel {l}' for l in ctx.labels_xyz],
        'Pelvis Angular Velocity', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/task_orientation/pelvis/pelvis_angular_velocity_plot.png')
    plot_components(ctx.t, _sub(ctx.pelvis_angular_velocity, ctx.des_pelvis_angular_velocity),
        [f'AngVel Error {l}' for l in ctx.labels_xyz],
        'Pelvis Angular Velocity Error', r'Angular Velocity [$\mathrm{rad/s}$]',
        'images/task_orientation/pelvis/pelvis_angular_velocity_error_plot.png')
