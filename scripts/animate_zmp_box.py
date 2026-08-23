#!/usr/bin/env python3
"""Animate the ZMP admissible box (ISMPC "moving box") against the desired
ZMP produced by the PLIP integration step, at every control-loop instant t_k.

Every control loop tick (500 Hz, WalkingManager::controller_frequency_,
src/WalkingManager.cpp:254) does, in this order, inside the isMPCLoopClosed
branch of WalkingManager.cpp:
  1. ismpc_ptr_->solve(...) builds the QP, including the per-horizon-step ZMP
     admissible box (mc_x_, mc_y_, mc_z_, mc_theta_, src/ISMPC.cpp:81-158).
     Index 0 of that horizon is "now" (t_k); its center/yaw are exposed via
     ISMPC::getZmpConstraintBoxCenter() / getZmpConstraintBoxYaw() and logged
     as zmp_box_center / zmp_box_yaw right after the solve() call.
  2. discrete_plip_dynamics_ptr_->integrate(...) advances the PLIP state using
     ismpc_ptr_->getInput() (that same solve's ZMP-velocity decision),
     producing des_LipState.zmp_pos_, logged as des_zmp_position — the "ZMP
     output from the PLIP integration step".

So at every t_k the box (logged from the same solve() call the ZMP-velocity
decision came from) is exactly the constraint the following PLIP-integrated
ZMP is supposed to satisfy — this script overlays the two.

Box half-extents are foot_constraint_square_length/width / 2 (ISMPC
constructor parameters, defaults 0.22 x 0.08 m, include/WalkingManager.hpp:
90-91) — hardcoded below like animate_ofp.py hardcodes the foot rectangle
size.

The support foot/feet are also drawn, from the desired (not feedback) foot
poses p_lsole_des/p_rsole_des + des_lsole_orientation/des_rsole_orientation
(logged in the same LOGS block as des_zmp_position, WalkingManager.cpp
~1537-1545, so they're all "what the controller intends at t_k" together),
gated by the per-tick contact_flags channel (logged right next to them —
WalkingManager.cpp: `logger_.log("contact_flags", ...)` built from
get_contact(), true/true outside SingleSupport). In single support only the
in-contact foot's rectangle is drawn; in double support both are drawn plus
the line delimiting their support polygon (convex hull of both footprints).

Requires a run recorded *after* the zmp_box_center / zmp_box_yaw / walking_state /
contact_flags logging was added (WalkingManager.cpp, right after
ismpc_ptr_->solve(...) and in the per-tick LOGS block) — older logs won't
have these files.

Run from the build/run directory that contains the robot_logs folder:
  python3 scripts/animate_zmp_box.py
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

# ZMP admissible box size — must match ISMPC's foot_constraint_square_length/
# width (include/WalkingManager.hpp:94-95).
BOX_LENGTH = 0.14
BOX_WIDTH = 0.04

# Foot rectangle size — must match WholeBodyController::Params
# (src/WholeBodyController.cpp: foot_length = 0.17, foot_width = 0.05).
FOOT_LENGTH = 0.17
FOOT_WIDTH = 0.05

CONTROL_FREQUENCY_HZ = 500.0  # WalkingManager::controller_frequency_, src/WalkingManager.cpp:254
DT = 1.0 / CONTROL_FREQUENCY_HZ

TRAIL_LEN = 50        # past des_zmp samples to trail behind the current point, in control ticks
PLAYBACK_STRIDE = 10  # animate every Nth control tick (10 -> 50 Hz update rate, real-time playback)

WALKING_STATE_NAMES = {  # include/WalkingState.hpp enum order
    0: 'Init', 1: 'PostureRegulation', 2: 'Standing', 3: 'Starting',
    4: 'SingleSupport', 5: 'DoubleSupport', 6: 'Stopping',
    7: 'AbortStarting', 8: 'AbortWalking',
}


def rot2(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def rect_corners(center_xy, yaw, dx, dy):
    """4 corners of a dx x dy rectangle centered at center_xy, rotated by yaw."""
    hl, hw = dx / 2.0, dy / 2.0
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    return local @ rot2(yaw).T + np.asarray(center_xy)


def convex_hull(points):
    """Monotone-chain convex hull. points: (N, 2) array. Returns CCW hull
    vertices (no repeated closing point); caller closes the loop for plotting."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def load_data(folder: Path, end_s):
    def _load(fname, ncols):
        p = folder / fname
        if not p.exists():
            return None
        return np.loadtxt(p, ndmin=2)[:, :ncols]

    def _load1d(fname):
        p = folder / fname
        if not p.exists():
            return None
        return np.atleast_1d(np.loadtxt(p))

    des_zmp = _load('des_zmp_position.txt', 3)
    box_center = _load('zmp_box_center.txt', 3)
    box_yaw = _load1d('zmp_box_yaw.txt')
    p_lsole_des = _load('p_lsole_des.txt', 3)
    p_rsole_des = _load('p_rsole_des.txt', 3)
    lsole_yaw = _load('des_lsole_orientation.txt', 3)
    rsole_yaw = _load('des_rsole_orientation.txt', 3)
    contact_flags = _load('contact_flags.txt', 2)
    walking_state = _load1d('walking_state.txt')

    fields = (
        ('des_zmp_position.txt', des_zmp),
        ('zmp_box_center.txt', box_center),
        ('zmp_box_yaw.txt', box_yaw),
        ('p_lsole_des.txt', p_lsole_des),
        ('p_rsole_des.txt', p_rsole_des),
        ('des_lsole_orientation.txt', lsole_yaw),
        ('des_rsole_orientation.txt', rsole_yaw),
        ('contact_flags.txt', contact_flags),
        ('walking_state.txt', walking_state),
    )
    missing = [name for name, v in fields if v is None]
    if missing:
        return None, missing

    n = min(len(v) for _, v in fields)
    if end_s is not None:
        n = min(n, int(end_s * CONTROL_FREQUENCY_HZ))
    return dict(
        des_zmp=des_zmp[:n], box_center=box_center[:n], box_yaw=box_yaw[:n],
        p_lsole_des=p_lsole_des[:n], p_rsole_des=p_rsole_des[:n],
        lsole_yaw=lsole_yaw[:n, 2], rsole_yaw=rsole_yaw[:n, 2],
        contact_flags=contact_flags[:n], walking_state=walking_state[:n],
        n=n,
    ), []


def main() -> None:
    exp_number = input("Enter 0 to animate data from the last simulation or the number of the experiment: ").strip()
    if exp_number == '0':
        folder = Path('/tmp/robot_logs')
    else:
        folder = Path('experiments/experiment_' + exp_number)
        if (folder / 'robot_logs').is_dir():
            folder = folder / 'robot_logs'
    if not folder.exists():
        print(f'[error] Directory not found: {folder}')
        return

    end_s_raw = input(
        "Enter the time (in seconds) at which you want to end the animation "
        "(or press Enter for all data): "
    ).strip()
    end_s = float(end_s_raw) if end_s_raw else None

    data, missing = load_data(folder, end_s)
    if data is None:
        print(f"[error] Missing log file(s) in {folder}: {', '.join(missing)}")
        print("        zmp_box_center.txt / zmp_box_yaw.txt / contact_flags.txt / walking_state.txt "
              "require a run recorded after this logging was added in WalkingManager.cpp "
              "(right after ismpc_ptr_->solve(...) and in the per-tick LOGS block).")
        return

    n = data['n']
    des_zmp, box_center, box_yaw = data['des_zmp'], data['box_center'], data['box_yaw']
    p_lsole_des, p_rsole_des = data['p_lsole_des'], data['p_rsole_des']
    lsole_yaw, rsole_yaw = data['lsole_yaw'], data['rsole_yaw']
    contact_flags, walking_state = data['contact_flags'], data['walking_state']

    frame_indices = list(range(0, n, PLAYBACK_STRIDE))
    if not frame_indices:
        print(f'[skip] No samples to animate in {folder}.')
        return

    # Fixed viewport spanning the whole ZMP trajectory, every box corner, and every foot corner.
    box_corners_all = np.array([
        rect_corners(box_center[k, :2], box_yaw[k], BOX_LENGTH, BOX_WIDTH)
        for k in frame_indices
    ]).reshape(-1, 2)
    foot_corners_all = np.array([
        rect_corners(pos[k, :2], yaw[k], FOOT_LENGTH, FOOT_WIDTH)
        for pos, yaw in ((p_lsole_des, lsole_yaw), (p_rsole_des, rsole_yaw))
        for k in frame_indices
    ]).reshape(-1, 2)
    all_xy = np.vstack([des_zmp[:, :2], box_corners_all, foot_corners_all])
    margin = 0.05
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
    # Pre-allocate every artist once (same style as animate_ofp.py):
    # per-frame updates only move data, never recreate or clear the axes.
    # ------------------------------------------------------------------
    box_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor='tab:red',
                                  alpha=0.2, edgecolor='tab:red', linewidth=1.5)
    ax.add_patch(box_patch)

    left_foot_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor='tab:blue',
                                        alpha=0.5, edgecolor='black', linewidth=1.2, visible=False)
    ax.add_patch(left_foot_patch)
    right_foot_patch = mpatches.Polygon(np.zeros((4, 2)), closed=True, facecolor='tab:orange',
                                         alpha=0.5, edgecolor='black', linewidth=1.2, visible=False)
    ax.add_patch(right_foot_patch)

    (support_polygon_line,) = ax.plot([], [], color='black', linestyle='-', linewidth=1.8,
                                       visible=False, zorder=4)

    (trail_line,) = ax.plot([], [], color='tab:blue', linewidth=1.0, alpha=0.6)
    (zmp_point,) = ax.plot([], [], marker='o', markersize=8, linestyle='')

    info_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )
    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center', va='bottom', fontsize=12)

    legend_handles = [
        mpatches.Patch(facecolor='tab:red', edgecolor='tab:red', alpha=0.2, label='ZMP admissible box (moving box)'),
        Line2D([0], [0], color='tab:blue', linewidth=1.0, alpha=0.6, label=f'des. ZMP trail ({TRAIL_LEN} ticks)'),
        Line2D([0], [0], marker='o', markersize=8, linestyle='', color='tab:green', label='des. ZMP (inside box)'),
        Line2D([0], [0], marker='o', markersize=8, linestyle='', color='tab:red', label='des. ZMP (outside box)'),
        mpatches.Patch(facecolor='tab:blue', edgecolor='black', alpha=0.5, label='Left foot (support, des.)'),
        mpatches.Patch(facecolor='tab:orange', edgecolor='black', alpha=0.5, label='Right foot (support, des.)'),
        Line2D([0], [0], color='black', linewidth=1.8, label='Support polygon (double support)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)

    all_artists = [box_patch, left_foot_patch, right_foot_patch, support_polygon_line,
                   trail_line, zmp_point, info_text, title_text]

    def draw_frame(frame_idx):
        k = frame_indices[frame_idx]
        center = box_center[k, :2]
        yaw = box_yaw[k]
        zmp = des_zmp[k, :2]

        box_patch.set_xy(rect_corners(center, yaw, BOX_LENGTH, BOX_WIDTH))

        trail_start = max(0, k - TRAIL_LEN)
        trail_line.set_data(des_zmp[trail_start:k + 1, 0], des_zmp[trail_start:k + 1, 1])

        # Inside/outside check in the box's own (foot) frame — same convention
        # as the sagittal/lateral ZMP box constraint built in ISMPC::solve().
        d = rot2(-yaw) @ (zmp - center)
        inside = (abs(d[0]) <= BOX_LENGTH / 2.0) and (abs(d[1]) <= BOX_WIDTH / 2.0)
        zmp_point.set_data([zmp[0]], [zmp[1]])
        zmp_point.set_color('tab:green' if inside else 'tab:red')

        # Support foot/feet: contact_flags[k] = (left_in_contact, right_in_contact),
        # true/true outside SingleSupport (WalkingManager::get_contact()).
        left_c = contact_flags[k, 0] > 0.5
        right_c = contact_flags[k, 1] > 0.5
        left_corners = rect_corners(p_lsole_des[k, :2], lsole_yaw[k], FOOT_LENGTH, FOOT_WIDTH)
        right_corners = rect_corners(p_rsole_des[k, :2], rsole_yaw[k], FOOT_LENGTH, FOOT_WIDTH)

        left_foot_patch.set_xy(left_corners)
        left_foot_patch.set_visible(bool(left_c))
        right_foot_patch.set_xy(right_corners)
        right_foot_patch.set_visible(bool(right_c))

        if left_c and right_c:
            hull = convex_hull(np.vstack([left_corners, right_corners]))
            hull_closed = np.vstack([hull, hull[0]])
            support_polygon_line.set_data(hull_closed[:, 0], hull_closed[:, 1])
            support_polygon_line.set_visible(True)
        else:
            support_polygon_line.set_visible(False)

        state_name = WALKING_STATE_NAMES.get(int(walking_state[k]), 'Unknown')
        info_text.set_text(
            f"t = {k * DT:.3f} s (tick {k}/{n - 1})\n"
            f"walking state = {state_name}\n"
            f"des. ZMP = ({zmp[0]:.3f}, {zmp[1]:.3f}) m\n"
            f"box center = ({center[0]:.3f}, {center[1]:.3f}) m, yaw = {np.degrees(yaw):.1f} deg\n"
            f"box size = {BOX_LENGTH:.2f} x {BOX_WIDTH:.2f} m\n"
            f"{'INSIDE' if inside else 'OUTSIDE'} box"
        )
        title_text.set_text("Desired ZMP (PLIP integration) vs. ZMP admissible box and support foot/feet")

        return all_artists

    state = {'idx': 0}

    def update(_frame):
        idx = state['idx']
        artists = draw_frame(idx)
        state['idx'] = (idx + 1) % len(frame_indices)
        return artists

    ani = FuncAnimation(fig, update, frames=len(frame_indices), init_func=lambda: draw_frame(0),
                        interval=PLAYBACK_STRIDE * DT * 1000.0, blit=True, repeat=True)

    running = [True]

    def step(delta):
        """Manually move to the previous/next frame; pauses playback like a video scrubber."""
        if running[0]:
            ani.event_source.stop()
            running[0] = False
        state['idx'] = (state['idx'] + delta) % len(frame_indices)
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

    print(f"[INFO] {len(frame_indices)} frames (stride={PLAYBACK_STRIDE}, "
          f"{n} total ticks @ {CONTROL_FREQUENCY_HZ:.0f} Hz) loaded from {folder} — "
          "space: pause/resume, left/right arrows: step, "
          "use the toolbar below the plot to zoom/pan.")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
