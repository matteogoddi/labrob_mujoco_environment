"""EXECUTION TIMES: WBC-aligned, use iter_t.

Extracted from the original plot_joint_data.py section I (was lines 1045-1073).
"""

import os

import matplotlib.pyplot as plt

from ..context import PlotContext


def run(ctx: PlotContext) -> None:
    exec_times = {k: v for k, v in {
        'EKF': ctx.execution_time_ekf,
        'KF': ctx.execution_time_kf,
        'MPC': ctx.execution_time_mpc,
        'WBC': ctx.execution_time_wbc,
        'Update': ctx.execution_time_update,
    }.items() if v is not None}

    for name, times in exec_times.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ctx.iter_t, times, linewidth=2.0, label=name)
        if name == 'Update':
            ax.axhline(y=2000, linestyle='--', linewidth=1.5,
                       label='Real-time threshold (2000 µs)')
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(r'Execution Time [$\mu s$]', fontsize=11)
        ax.set_title(f'{name} Execution Time per Iteration', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(frameon=True, fontsize=10)
        fig.tight_layout()
        os.makedirs('images/execution_times', exist_ok=True)
        fig.savefig(f'images/execution_times/{name}_execution_time_plot.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
