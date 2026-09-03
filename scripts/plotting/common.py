"""Generic plotting/loading helpers shared by every diagnostic section.

Moved verbatim out of the original monolithic plot_joint_data.py, plus a new
consolidated helper (plot_aggregate_joint_error) that replaces three
near-identical "one colored line per joint" blocks that used to be
duplicated across sections (joint_tracking position/velocity error, EKF
velocity error).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

CTRL_HZ = 500
DT = 1.0 / CTRL_HZ

CMAP = plt.colormaps['tab10']
LSTYLES = ['-', '--', '-.', ':']


def _sub(a, b):
    """a - b clipped to min length, or None if either operand is None."""
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    return a[:n] - b[:n]


def plot_components(t, data, labels, title, ylabel, path,
                     figsize=(7, 4), loc='best', mean_last_s=5.0):
    if data is None:
        return
    if data.ndim == 1:
        data = data[:, np.newaxis]
    t = t[:len(data)]
    fig, ax = plt.subplots(figsize=figsize)
    for col, lbl in zip(data.T, labels):
        ax.plot(t, col, label=lbl, linewidth=2.0)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc=loc, frameon=True, fontsize=9)
    ax.tick_params(axis='both', labelsize=10)
    if mean_last_s > 0 and len(data) > 0:
        n_win = min(int(mean_last_s * CTRL_HZ), len(data))
        lines = '\n'.join(
            f'{lbl}: {np.mean(col[-n_win:]):.3f}'
            for col, lbl in zip(data.T, labels)
        )
        ax.text(0.98, 0.02, f'Mean last {mean_last_s:.0f}s\n{lines}',
                transform=ax.transAxes, fontsize=7, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.6))
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(t, actual, desired, coord_labels, title_prefix, ylabel, path):
    if actual is None or desired is None:
        return
    _nr = min(len(actual), len(desired))
    actual, desired, t = actual[:_nr], desired[:_nr], t[:_nr]
    n = len(coord_labels)
    fig, axs = plt.subplots(n, 1, figsize=(7, 3 * n), sharex=True)
    if n == 1:
        axs = [axs]
    for i, lbl in enumerate(coord_labels):
        axs[i].plot(t, actual[:, i],  label=f'Actual {lbl}',  linewidth=2.0)
        axs[i].plot(t, desired[:, i], label=f'Desired {lbl}', linewidth=2.0, linestyle='--')
        axs[i].set_ylabel(ylabel, fontsize=10)
        axs[i].set_title(f'{title_prefix} {lbl}', fontsize=11)
        axs[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].legend(fontsize=9)
    axs[-1].set_xlabel('Time [s]', fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def build_lr_pairs(names):
    """Left/right joint index pairs (li, ri, suffix) by name-matching within
    `names`, skipping waist joints. Factored out so callers can rebuild this
    against a DIFFERENT joint-name list than the full DDS/model one (e.g. a
    WBC-controlled-only subset — see sections/wbc_solutions.py) instead of
    assuming every per-joint array shares the same column layout.
    """
    pairs = []
    for li, lname in enumerate(names):
        if not lname.startswith('left_') or 'waist' in lname:
            continue
        suffix = lname[len('left_'):]
        rname = 'right_' + suffix
        if rname in names:
            pairs.append((li, names.index(rname), suffix))
    return pairs


def try_load(folder, name):
    """Load name.txt if it exists, else return None.

    Guarantees 2-D output for multi-column files even with a single data row.
    """
    path = f"{folder}/{name}.txt"
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path)
    if data.ndim == 1:
        with open(path) as f:
            first_line = f.readline().strip()
        if len(first_line.split()) > 1:
            data = data.reshape(1, -1)
    return data


def plot_aggregate_joint_error(t, minuend, subtrahend, joint_labels, title, ylabel, path,
                                legend_fontsize=4):
    """One figure, one colored line per joint, for (minuend - subtrahend).

    Consolidates the "aggregate per-joint colored-line error plot" pattern
    that used to be duplicated three times (joint reference tracking
    position/velocity error, EKF velocity error) with slightly different
    minuend/subtrahend conventions each time — the caller decides which
    signal is which, no "desired/actual" assumption is baked in here.
    """
    if minuend is None or subtrahend is None:
        return
    n = min(len(minuend), len(subtrahend))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, lbl in enumerate(joint_labels):
        ax.plot(t[:n], minuend[:n, i] - subtrahend[:n, i],
                label=lbl, color=CMAP(i % 10),
                linestyle=LSTYLES[(i // 10) % 4], linewidth=2)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc='best', frameon=True, fontsize=legend_fontsize)
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
