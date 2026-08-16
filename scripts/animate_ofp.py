#!/usr/bin/env python3
"""Animate online footstep planner (coop) solutions saved by WalkingManager.

Each invocation of FootstepPlannerCoop::computeNextSteps() writes one
subdirectory inside  ofp_data/<t_ms>/:
  solution.txt : F rows "x y"  — world-frame swing-foot targets of the QP solution
  meta.txt     : F, delta_theta, delta_p, support_foot_start, p0, yaw0
plus a single ofp_data/params.txt with the planner's constants (da_x, da_y,
ell). See WalkingManager::saveLogs() / logOfpSnapshot() in
src/WalkingManager.cpp, and FootstepPlannerCoop::updateQPMatrices() in
src/FootstepPlannerCoop.cpp for the underlying geometry reproduced below:

  - Reachability box for step j: centered at
      prev_pos_j + Rz(box_yaw_j) @ [0, s_j * ell]
    (prev_pos_0 = p0/support foot, prev_pos_j = solution[j-1] for j>0;
     box_yaw_0 = yaw0, box_yaw_j = yaw0 + delta_theta for j>0;
     s_j = +1 if the swing foot at step j is the left foot, else -1),
    size da_x x da_y, oriented like box_yaw_j — one box per planned step,
    chained from the support foot through the whole plan.
  - Delta_p contribution for step j: this planner has no separate nominal
    stride term (n_j = Rz(box_yaw_j) @ (delta_p + (0, s_j*ell))), so
    Delta_p is (most of) the actual foot-to-foot step vector — the arrow
    is drawn from the real previous foot to box_center_j + Rz(box_yaw_j)
    @ delta_p, i.e. from one foot towards the next.
  - Foot heading: short arrow per foot along its local +x (forward) axis —
    yaw0 for the support foot, yaw0 + delta_theta (constant) for every
    planned step — with a small arc at the first swing foot (the foot
    opposite the current support foot) spanning the gap between the two
    headings, i.e. Delta_theta itself.

Feet are drawn as rectangles sized like WholeBodyController::Params
(foot_length = 0.20 m, foot_width = 0.07 m, src/WholeBodyController.cpp).
Only the latest solution is shown each frame — old ones are not overlaid.

Run from the build/run directory that contains the ofp_data/ folder:
  python3 scripts/animate_ofp.py
"""
import numpy as np
import importlib
import matplotlib
for _backend in ('TkAgg', 'Qt5Agg', 'QtAgg', 'GTK3Agg', 'MacOSX'):
    try:
        importlib.import_module(f'matplotlib.backends.backend_{_backend.lower()}')
        matplotlib.use(_backend)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from pathlib import Path

# Foot rectangle size — must match WholeBodyController::Params
# (src/WholeBodyController.cpp: foot_length = 0.20, foot_width = 0.07).
FOOT_LENGTH = 0.20
FOOT_WIDTH = 0.07

HEADING_LEN = 0.10     # length of the foot-heading arrows [m]
DTHETA_ARC_RADIUS = 0.045  # radius of the Delta_theta arc drawn at the first swing foot [m]

MIN_HOLD_MS, MAX_HOLD_MS = 400, 3000


def parse_timestep(name: str):
    try:
        return int(name)
    except ValueError:
        return None


def rot2(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def rect_corners(center_xy, yaw, dx, dy):
    """4 corners of a dx x dy rectangle centered at center_xy, rotated by yaw."""
    hl, hw = dx / 2.0, dy / 2.0
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    return local @ rot2(yaw).T + np.asarray(center_xy)


def load_snapshots(base_dir: Path):
    da_x, da_y, ell = None, None, None
    params_path = base_dir / 'params.txt'
    if params_path.exists():
        for line in params_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == 'da_x':
                da_x = float(parts[1])
            elif len(parts) == 2 and parts[0] == 'da_y':
                da_y = float(parts[1])
            elif len(parts) == 2 and parts[0] == 'ell':
                ell = float(parts[1])

    entries = sorted(
        [(parse_timestep(p.name), p) for p in base_dir.iterdir()
         if p.is_dir() and parse_timestep(p.name) is not None],
        key=lambda tp: tp[0]
    )

    snapshots = []
    for t_ms, p in entries:
        sol_path = p / 'solution.txt'
        meta_path = p / 'meta.txt'
        if not (sol_path.exists() and meta_path.exists()):
            continue
        solution = np.loadtxt(sol_path, ndmin=2)  # F x 2, world-frame [x, y] per planned step
        meta = {}
        for line in meta_path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                meta[parts[0]] = parts[1:]
        if 'F' not in meta or solution.shape[0] == 0:
            continue
        snapshots.append(dict(
            t_ms=t_ms,
            solution=solution,
            F=int(meta['F'][0]),
            delta_theta=float(meta['delta_theta'][0]),
            delta_p=np.array(meta['delta_p'], dtype=float),
            support_foot_start=meta['support_foot_start'][0],  # 'L' or 'R'
            p0=np.array(meta['p0'], dtype=float),
            yaw0=float(meta['yaw0'][0]),
        ))
    return snapshots, da_x, da_y, ell


def swing_is_left_sequence(sf_is_left, F):
    """Reproduces FootstepPlannerCoop::updateQPMatrices' swing-foot alternation."""
    out = []
    swing_is_left = not sf_is_left
    for _ in range(F):
        out.append(swing_is_left)
        swing_is_left = not swing_is_left
    return out


def main() -> None:
    exp_number = input("Enter 0 to animate data from the last simulation or the number of the experiment: ").strip()
    if exp_number == '0':
        base_dir = Path('/tmp/ofp_data')
    else:
        base_dir = Path('experiments/experiment_' + exp_number) / 'ofp_logs'
    if not base_dir.exists():
        print(f'[error] Directory not found: {base_dir}')
        return

    snapshots, da_x, da_y, ell = load_snapshots(base_dir)
    if not snapshots:
        print(f'[skip] No OFP snapshots found in {base_dir} '
              '(the coop planner never triggered during that run).')
        return
    have_box = (da_x is not None and da_y is not None and ell is not None)

    F_max = max(s['F'] for s in snapshots)

    # Fixed viewport spanning every solution ever plotted, so the camera doesn't jump around.
    all_xy = [s['p0'] for s in snapshots]
    for s in snapshots:
        all_xy.extend(list(s['solution']))
    all_xy = np.array(all_xy)
    margin = 0.3
    xlim = (all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ylim = (all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    # ------------------------------------------------------------------
    # Pre-allocate every artist once (same style as animate_mpc.py's
    # make_state_lines/make_input_lines): per-frame updates only move data,
    # never recreate or clear the axes — keeps blit fast and never resets
    # whatever zoom/pan you apply via the toolbar.
    # ------------------------------------------------------------------
    support_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor='tab:green',
                                      alpha=0.6, edgecolor='black', linewidth=1.5)
    ax.add_patch(support_patch)

    support_heading = mpatches.FancyArrowPatch((0, 0), (0, 0), arrowstyle='-|>',
                                                mutation_scale=10, color='black', linewidth=1.5, zorder=5)
    ax.add_patch(support_heading)

    # Delta_theta, drawn at the first swing foot like the reference figure: a dashed ray showing
    # where the support foot's own heading would point (yaw0, "no rotation"), the foot's already-
    # existing solid heading arrow showing where it actually points (step_yaw), and a small arc
    # + label between the two right at that vertex.
    (dtheta_ref_line,) = ax.plot([], [], color='0.4', linestyle='--', linewidth=1.3, zorder=4)
    (dtheta_arc,) = ax.plot([], [], color='tab:purple', linewidth=1.8, zorder=4)
    dtheta_label = ax.text(0, 0, '', color='tab:purple', fontsize=9, fontweight='bold',
                            ha='center', va='center', visible=False)

    step_patches, step_labels, step_headings, box_lines, dp_arrows = [], [], [], [], []
    for _ in range(F_max):
        patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor='tab:blue',
                                  alpha=0.35, edgecolor='black', linewidth=1.0, visible=False)
        ax.add_patch(patch)
        label = ax.text(0, 0, '', ha='center', va='center', fontsize=8, visible=False)
        heading = mpatches.FancyArrowPatch((0, 0), (0, 0), arrowstyle='-|>', mutation_scale=8,
                                            color='black', linewidth=1.2, visible=False, zorder=5)
        ax.add_patch(heading)
        (box_line,) = ax.plot([], [], color='tab:red', linestyle='--', linewidth=1.2, visible=False)
        dp_arrow = mpatches.FancyArrowPatch((0, 0), (0, 0), arrowstyle='-|>', mutation_scale=10,
                                             color='deeppink', linewidth=1.5, visible=False, zorder=5)
        ax.add_patch(dp_arrow)
        step_patches.append(patch)
        step_labels.append(label)
        step_headings.append(heading)
        box_lines.append(box_line)
        dp_arrows.append(dp_arrow)

    info_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )
    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center', va='bottom', fontsize=12)

    legend_handles = [
        mpatches.Patch(facecolor='tab:green', edgecolor='black', alpha=0.6, label='Support foot'),
        mpatches.Patch(facecolor='tab:blue', edgecolor='black', alpha=0.35, label='Left foot (planned)'),
        mpatches.Patch(facecolor='tab:orange', edgecolor='black', alpha=0.35, label='Right foot (planned)'),
        Line2D([0], [0], color='black', marker='>', markersize=5, label='Foot heading (yaw)'),
        Line2D([0], [0], color='0.4', linestyle='--', linewidth=1.3, label='Support ref. heading (no Δθ)'),
        Line2D([0], [0], color='tab:purple', linewidth=1.8, label='Δθ (at first swing foot)'),
    ]
    if have_box:
        legend_handles.append(Line2D([0], [0], color='tab:red', linestyle='--', label='Reachability box'))
        legend_handles.append(Line2D([0], [0], color='deeppink', marker='>', markersize=5, label='Δp (foot → foot)'))
    ax.legend(handles=legend_handles, loc='lower right', fontsize=7.5)

    all_artists = ([support_patch, support_heading, dtheta_ref_line, dtheta_arc, dtheta_label,
                    info_text, title_text]
                   + step_patches + step_labels + step_headings + box_lines + dp_arrows)

    def draw_frame(idx):
        snap = snapshots[idx]
        sf_is_left = (snap['support_foot_start'] == 'L')
        p0 = snap['p0']
        yaw0 = snap['yaw0']
        delta_theta = snap['delta_theta']
        delta_p = snap['delta_p']
        step_yaw = yaw0 + delta_theta  # constant orientation for all F planned steps

        support_patch.set_xy(rect_corners(p0, yaw0, FOOT_LENGTH, FOOT_WIDTH))
        support_heading.set_positions(p0, p0 + HEADING_LEN * (rot2(yaw0) @ np.array([1.0, 0.0])))

        # Delta_theta: drawn at the first swing foot — the foot opposite the current support foot,
        # the one this rotation is actually applied to (every later planned step shares the same
        # step_yaw, so showing it once here is enough). Mirrors the reference figure: a dashed ray
        # at yaw0 (where the support foot's own heading would point, i.e. "no Delta_theta"), the
        # foot's already-drawn solid heading arrow at step_yaw (its actual heading), and a small
        # arc + label tucked right between the two at that shared vertex.
        swing0_pos = snap['solution'][0] if snap['F'] > 0 else p0
        dtheta_ref_line.set_data(
            [swing0_pos[0], swing0_pos[0] + HEADING_LEN * 1.3 * np.cos(yaw0)],
            [swing0_pos[1], swing0_pos[1] + HEADING_LEN * 1.3 * np.sin(yaw0)]
        )
        n_arc = max(2, int(abs(np.degrees(delta_theta))) + 2)
        thetas = np.linspace(yaw0, step_yaw, n_arc)
        arc_xy = swing0_pos + DTHETA_ARC_RADIUS * np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
        dtheta_arc.set_data(arc_xy[:, 0], arc_xy[:, 1])
        if abs(delta_theta) > 1e-4:
            # Tucked close to the vertex along the arc's own bisector, not a large fixed offset,
            # so it can never overshoot onto a neighboring foot regardless of delta_theta's size.
            mid = yaw0 + 0.5 * delta_theta
            dtheta_label.set_position(swing0_pos + 1.7 * DTHETA_ARC_RADIUS * np.array([np.cos(mid), np.sin(mid)]))
            dtheta_label.set_text('Δθ')
            dtheta_label.set_visible(True)
        else:
            dtheta_label.set_visible(False)

        # Planned footsteps: swing foot alternates starting from the opposite of the support foot.
        # Reachability box j is chained from the previous foot (support foot for j=0), and box_yaw_0
        # is the support foot's own yaw (no Delta_theta yet) while box_yaw_j>0 already includes it —
        # this mirrors FootstepPlannerCoop::updateQPMatrices exactly (R_1 vs R_j_rest).
        swing_flags = swing_is_left_sequence(sf_is_left, F_max)
        prev_pos = p0
        for j in range(F_max):
            patch, label, heading = step_patches[j], step_labels[j], step_headings[j]
            box_line, dp_arrow = box_lines[j], dp_arrows[j]
            if j < snap['F']:
                pos = snap['solution'][j]
                patch.set_xy(rect_corners(pos, step_yaw, FOOT_LENGTH, FOOT_WIDTH))
                patch.set_facecolor('tab:blue' if swing_flags[j] else 'tab:orange')
                patch.set_visible(True)
                # Offset the number label sideways so it doesn't sit right on top of the
                # heading arrow, which is drawn from the same center point.
                label_pos = pos + rot2(step_yaw) @ np.array([0.0, -FOOT_WIDTH * 0.65])
                label.set_position((label_pos[0], label_pos[1]))
                label.set_text(str(j + 1))
                label.set_visible(True)

                # Heading arrow always matches the foot's actual (rectangle) orientation,
                # i.e. step_yaw for every planned step — unlike the reachability box below,
                # whose own frame differs for the first step (see box_yaw_j).
                heading.set_positions(pos, pos + HEADING_LEN * (rot2(step_yaw) @ np.array([1.0, 0.0])))
                heading.set_visible(True)

                box_yaw_j = yaw0 if j == 0 else step_yaw
                if have_box:
                    s_j = 1.0 if swing_flags[j] else -1.0
                    box_center = prev_pos + rot2(box_yaw_j) @ np.array([0.0, s_j * ell])
                    box = rect_corners(box_center, box_yaw_j, da_x, da_y)
                    box = np.vstack([box, box[0]])  # close the loop
                    box_line.set_data(box[:, 0], box[:, 1])
                    box_line.set_visible(True)

                    # n_j = Rj @ (delta_p + (0, s_j*ell)) is the whole nominal step from the
                    # previous foot — i.e. Delta_p is (most of) the actual foot-to-foot step
                    # vector here (there's no separate nominal stride term in this planner),
                    # so root the arrow at the real previous foot, not at the box center.
                    dp_end = box_center + rot2(box_yaw_j) @ delta_p
                    dp_arrow.set_positions(prev_pos, dp_end)
                    dp_arrow.set_visible(True)
                else:
                    box_line.set_visible(False)
                    dp_arrow.set_visible(False)

                prev_pos = pos
            else:
                patch.set_visible(False)
                label.set_visible(False)
                heading.set_visible(False)
                box_line.set_visible(False)
                dp_arrow.set_visible(False)

        info_lines = [
            f"t = {snap['t_ms'] / 1000.0:.3f} s",
            f"Δθ = {np.degrees(delta_theta):.2f} deg",
            f"Δp = ({delta_p[0]:.3f}, {delta_p[1]:.3f}) m",
            f"support foot = {'LEFT' if sf_is_left else 'RIGHT'}",
        ]
        if have_box:
            info_lines.append(f"reach. box = {da_x:.2f} x {da_y:.2f} m  (ell={ell:.3f} m)")
        info_text.set_text("\n".join(info_lines))
        title_text.set_text(f"Online footstep planner (coop) — solution {idx + 1}/{len(snapshots)}")

        return all_artists

    state = {'idx': 0}

    def update(_frame):
        idx = state['idx']
        artists = draw_frame(idx)
        # Hold each frame roughly as long as the real time elapsed until the next solve,
        # clamped to keep the animation watchable regardless of the actual replanning cadence.
        if idx + 1 < len(snapshots):
            hold_ms = snapshots[idx + 1]['t_ms'] - snapshots[idx]['t_ms']
        else:
            hold_ms = MIN_HOLD_MS
        ani.event_source.interval = min(max(hold_ms, MIN_HOLD_MS), MAX_HOLD_MS)
        state['idx'] = (idx + 1) % len(snapshots)
        return artists

    ani = FuncAnimation(fig, update, frames=len(snapshots), init_func=lambda: draw_frame(0),
                        interval=MIN_HOLD_MS, blit=True, repeat=True)

    running = [True]

    def step(delta):
        """Manually move to the previous/next solve; pauses playback like a video scrubber."""
        if running[0]:
            ani.event_source.stop()
            running[0] = False
        state['idx'] = (state['idx'] + delta) % len(snapshots)
        draw_frame(state['idx'])
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == ' ':
            if running[0]: ani.event_source.stop()
            else:          ani.event_source.start()
            running[0] = not running[0]
        elif event.key in ('right', 'up'):
            step(+1)
        elif event.key in ('left', 'down'):
            step(-1)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print(f"[INFO] {len(snapshots)} solves loaded from {base_dir} — "
          "space: pause/resume, left/right arrows: step, "
          "use the toolbar below the plot to zoom/pan.")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
