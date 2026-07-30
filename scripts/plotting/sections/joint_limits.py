"""WBC joint rate-limit margins (velocity + position).

Extracted from the second half of the original plot_joint_data.py section C
(was lines 534-643). Reuses the swing/stance transitions detected by
contact_forces.run() (passed in explicitly) to zoom in on the same instants
— margin >= 0 means the joint vel/pos rate limits (C_acc_) are satisfied;
negative means the WBC solution is violating them. Kept as a separate
diagnosis from friction-cone margins since these are a different constraint
block, checked to see whether *these* (rather than the cone) are what makes
the hard QP infeasible at transitions.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ..common import CTRL_HZ, DT
from ..context import PlotContext
from ..transitions import Transition

_ZOOM_HALF_S = 0.2


def _jname(ctx: PlotContext, idx) -> str:
    idx = int(idx)
    return ctx.jnames_stripped[idx] if 0 <= idx < len(ctx.jnames_stripped) else f'joint_{idx}'


def run(ctx: PlotContext, transitions_by_foot: Dict[str, List[Transition]]) -> None:
    zoom_half_n = max(1, int(round(_ZOOM_HALF_S * CTRL_HZ)))

    jl_series = [
        ('vel', ctx.wbc_joint_vel_limit_margin, 'Worst joint velocity-limit margin [rad/s]',
         ctx.wbc_worst_vel_limit_joint, ctx.wbc_worst_vel_limit_is_upper),
        ('pos', ctx.wbc_joint_pos_limit_margin, 'Worst joint position-limit margin [rad]',
         ctx.wbc_worst_pos_limit_joint, ctx.wbc_worst_pos_limit_is_upper),
    ]
    for kind_name, series, ylabel, worst_joint_s, worst_upper_s in jl_series:
        if series is None:
            continue
        jlm = series.reshape(-1)
        n_jlm = len(jlm)
        t_jlm = ctx.t[:n_jlm]
        worst_joint = (worst_joint_s.reshape(-1).astype(int) if worst_joint_s is not None else None)
        worst_upper = (worst_upper_s.reshape(-1).astype(bool) if worst_upper_s is not None else None)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t_jlm, jlm, linewidth=1.2, color='tab:purple')
        ax.axhline(y=0.0, color='k', linestyle='--', linewidth=1.5,
                   label='Constraint boundary (margin = 0)')
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'WBC Joint {kind_name.capitalize()}-Limit Margin '
                     '(min over all joints)', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', frameon=True, fontsize=9)
        fig.tight_layout()
        os.makedirs('images/joint_limits', exist_ok=True)
        fig.savefig(f'images/joint_limits/{kind_name}_margin_plot.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Which joint is the bottleneck most often, over the whole run?
        if worst_joint is not None:
            n_wj = min(len(worst_joint), n_jlm)
            uniq, counts = np.unique(worst_joint[:n_wj], return_counts=True)
            order = np.argsort(counts)[::-1]
            fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(uniq))))
            ax.barh([_jname(ctx, uniq[i]) for i in order],
                    [counts[i] / n_wj * 100.0 for i in order],
                    color='tab:purple')
            ax.invert_yaxis()
            ax.set_xlabel('Fraction of ticks as worst joint [%]', fontsize=10)
            ax.set_title(f'Which Joint Drives the {kind_name.capitalize()}-Limit '
                         'Margin, Overall', fontsize=12)
            ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
            fig.tight_layout()
            fig.savefig(f'images/joint_limits/{kind_name}_worst_joint_histogram.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

        zoom_dir = 'images/joint_limits/transitions'
        os.makedirs(zoom_dir, exist_ok=True)
        for foot_name, transitions in transitions_by_foot.items():
            summary_lines = [
                f'{"#":>3}  {"type":<10}  {"t [s]":>8}  {"min margin":>12}  '
                f'{"joint":<20}  {"bound":<5}  {"violation dur [ms]":>18}'
            ]
            for k, tr in enumerate(transitions):
                idx, kw = tr.idx, tr.kind
                if idx >= n_jlm:
                    continue
                i0 = max(0, idx - zoom_half_n)
                i1 = min(n_jlm, idx + zoom_half_n)
                window = jlm[i0:i1]
                local_argmin = int(np.argmin(window))
                min_margin = float(window[local_argmin])
                violation_dur_ms = float((window < 0).sum() * DT * 1000.0)

                if worst_joint is not None:
                    wj = int(worst_joint[i0 + local_argmin])
                    jname = _jname(ctx, wj)
                    bound = ('upper' if worst_upper is not None
                             and worst_upper[i0 + local_argmin] else 'lower')
                else:
                    jname, bound = 'n/a', 'n/a'

                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(t_jlm[i0:i1], window, linewidth=1.8, color='tab:purple')
                ax.axhline(y=0.0, color='k', linestyle='--', linewidth=1.5)
                ax.axvline(t_jlm[idx], color='gray', linestyle=':', linewidth=1.5,
                           label=kw)
                ax.set_xlabel('Time [s]', fontsize=11)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_title(
                    f'{foot_name.capitalize()} foot — {kw} @ t={t_jlm[idx]:.2f}s  '
                    f'(min margin {min_margin:+.3f}, {jname} [{bound}])', fontsize=9)
                ax.legend(loc='best', frameon=True, fontsize=8)
                fig.tight_layout()
                fig.savefig(f'{zoom_dir}/{foot_name}_{kind_name}_{k:02d}_{kw}.png',
                            dpi=300, bbox_inches='tight')
                plt.close(fig)

                summary_lines.append(
                    f'{k:>3}  {kw:<10}  {t_jlm[idx]:>8.2f}  {min_margin:>12.3f}  '
                    f'{jname:<20}  {bound:<5}  {violation_dur_ms:>18.1f}'
                )

            with open(
                f'images/joint_limits/{foot_name}_{kind_name}_limit_violations_summary.txt',
                'w') as f:
                f.write('\n'.join(summary_lines) + '\n')
