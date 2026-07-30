"""WBC PER-CORNER CONTACT FORCES + FRICTION CONE.

Extracted from the first half of the original plot_joint_data.py section C
(was lines 421-533). Returns transitions_by_foot so joint_limits.run() can
zoom in on the same swing/stance instants without recomputing them.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ..common import CTRL_HZ, DT, plot_components
from ..context import PlotContext
from ..transitions import Transition, detect_stance_transitions

# wbc_contact_forces layout: [left(4 corners x Fx,Fy,Fz); right(4 corners x Fx,Fy,Fz)]
# corner order per WholeBodyController.cpp pcis_[0..3]:
CORNER_LABELS = ['front-left', 'front-right', 'back-left', 'back-right']

# Zoom window half-width around each swing/stance transition.
_ZOOM_HALF_S = 0.2
# A corner-force sum below this is treated as "swing" (no contact).
_STANCE_THRESHOLD_N = 2.0
# Debounce: ignore re-crossings within this long of an accepted transition
# (avoids counting contact-bounce chatter as separate liftoff/touchdown events).
_MIN_TRANSITION_GAP_S = 0.1


def run(ctx: PlotContext) -> Dict[str, List[Transition]]:
    transitions_by_foot: Dict[str, List[Transition]] = {}

    if ctx.wbc_contact_forces is None:
        return transitions_by_foot

    n_rows = len(ctx.wbc_contact_forces)
    t_wbc = ctx.t[:n_rows]
    mu = float(np.mean(ctx.wbc_mu)) if ctx.wbc_mu is not None else 0.6

    zoom_half_n = max(1, int(round(_ZOOM_HALF_S * CTRL_HZ)))
    min_transition_gap_n = max(1, int(round(_MIN_TRANSITION_GAP_S * CTRL_HZ)))

    for foot_idx, foot_name in enumerate(('left', 'right')):
        base = foot_idx * 12
        fz_cols = np.column_stack([
            ctx.wbc_contact_forces[:, base + 3 * c + 2] for c in range(4)
        ])
        plot_components(t_wbc, fz_cols,
            [f'{lbl} Fz' for lbl in CORNER_LABELS],
            f'WBC {foot_name.capitalize()} Foot — Per-Corner Normal Force',
            'Force [N]',
            f'images/contact_forces/{foot_name}_corner_fz.png')

        # Friction-cone margin per corner: mu*Fz - sqrt(Fx^2+Fy^2).
        # Negative => the corner is outside the friction pyramid.
        margins = np.zeros((n_rows, 4))
        for c in range(4):
            fx = ctx.wbc_contact_forces[:, base + 3 * c + 0]
            fy = ctx.wbc_contact_forces[:, base + 3 * c + 1]
            fz = ctx.wbc_contact_forces[:, base + 3 * c + 2]
            margins[:, c] = mu * fz - np.sqrt(fx**2 + fy**2)

        def _plot_margin(ax, i0, i1):
            for c in range(4):
                ax.plot(t_wbc[i0:i1], margins[i0:i1, c], linewidth=1.8,
                        label=CORNER_LABELS[c])
            ax.axhline(y=0.0, color='k', linestyle='--', linewidth=1.5,
                       label='Friction cone boundary (margin = 0)')
            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel(r'$\mu F_z - \sqrt{F_x^2+F_y^2}$  [N]', fontsize=11)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        fig, ax = plt.subplots(figsize=(7, 4))
        _plot_margin(ax, 0, n_rows)
        ax.set_title(f'WBC {foot_name.capitalize()} Foot — Friction Cone Margin '
                     f'(mu={mu:.2f})', fontsize=12)
        ax.legend(loc='best', frameon=True, fontsize=9)
        fig.tight_layout()
        os.makedirs('images/contact_forces', exist_ok=True)
        fig.savefig(f'images/contact_forces/{foot_name}_friction_cone_margin.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        # ── Zoom on every swing/stance transition ─────────────────────────
        transitions = detect_stance_transitions(fz_cols, _STANCE_THRESHOLD_N, min_transition_gap_n)
        transitions_by_foot[foot_name] = transitions

        zoom_dir = 'images/contact_forces/transitions'
        os.makedirs(zoom_dir, exist_ok=True)
        summary_lines = [
            f'{"#":>3}  {"type":<10}  {"t [s]":>8}  {"min margin [N]":>15}  '
            f'{"worst corner":<12}  {"violation dur [ms]":>18}'
        ]
        for k, tr in enumerate(transitions):
            idx, kind = tr.idx, tr.kind
            i0 = max(0, idx - zoom_half_n)
            i1 = min(n_rows, idx + zoom_half_n)
            window = margins[i0:i1]
            worst_c = int(np.argmin(window.min(axis=0)))
            min_margin = float(window[:, worst_c].min())
            violation_dur_ms = float((window < 0).any(axis=1).sum() * DT * 1000.0)

            fig, ax = plt.subplots(figsize=(6, 3.5))
            _plot_margin(ax, i0, i1)
            ax.axvline(t_wbc[idx], color='gray', linestyle=':', linewidth=1.5,
                       label=f'{kind}')
            ax.set_title(
                f'{foot_name.capitalize()} foot — {kind} @ t={t_wbc[idx]:.2f}s  '
                f'(min margin {min_margin:+.1f} N, {CORNER_LABELS[worst_c]})',
                fontsize=10)
            ax.legend(loc='best', frameon=True, fontsize=8)
            fig.tight_layout()
            fig.savefig(f'{zoom_dir}/{foot_name}_{k:02d}_{kind}.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

            summary_lines.append(
                f'{k:>3}  {kind:<10}  {t_wbc[idx]:>8.2f}  {min_margin:>15.2f}  '
                f'{CORNER_LABELS[worst_c]:<12}  {violation_dur_ms:>18.1f}'
            )

        with open(f'images/contact_forces/{foot_name}_friction_violations_summary.txt',
                  'w') as f:
            f.write('\n'.join(summary_lines) + '\n')

    return transitions_by_foot
