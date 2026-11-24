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

The estimated ZMP (ef_zmp_position, WalkingManager.cpp ~706-739) is overlaid
too: it is reconstructed from the RB-WO foot wrenches (getLeft/RightFootWrench),
falling back to the PLIP-reconstructed zmp_position when the total vertical
force is ~0 or during the first 2 s of observer activity.

The current CoM (kf_com_position, i.e. kf_LipState.com_pos_ output by
com_kf_step, WalkingManager.cpp ~793) is overlaid as well, with its own trail.

The desired CoM (des_com_position, i.e. des_LipState.com_pos_) is overlaid too:
it comes out of the very same PLIP integration step that produces
des_zmp_position, driven by the ZMP velocity that solves the MPC QP
(ismpc_ptr_->getInput()), so des. CoM and des. ZMP are the two halves of one
PLIP state at t_k.

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

Everything that is not the scene itself — flag badges, numeric readout, legend —
sits in a side panel to the right of the plot, so that nothing covers the feet,
the box or the support polygon. The legend is anchored under the readout at
build time, after measuring how tall it actually is.

Three badges at the top of that panel track the gamepad-activated modes, read
from the control_flags channel: each stays grey with a ✗ until the tick at which its
button was pressed (A -> EKF, X -> closed loop, B -> wrench observer, main.cpp
handle_gamepad) and turns green with a ✓ from then on.

That channel is logged by the *main loop* (main.cpp, end of the per-tick block),
not by WalkingManager, because the WalkingManager logs only start once X has
closed the loop — by which point A and X are already on. So the animation opens
with a pre-roll over those earlier main-loop ticks, where the plot is empty (no
ZMP/CoM/foot logs exist yet) and only the flags advance: it starts
PREROLL_LEAD_S before the first button press, runs at PREROLL_SPEEDUP x real
time, and ends exactly where the closed-loop part below begins. In simulation
main.cpp turns all three on at the first tick, so there is no pre-roll at all.

The two timelines are lined up through column 3 of control_flags, which marks the
main-loop ticks that have a WalkingManager log row; the time shown is always the
main-loop one, so it runs continuously across the pre-roll. The panel is skipped
for older runs whose control_flags.txt is missing or has 3 columns.

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

# Pre-roll: the ticks before X closed the loop, where only the flag panel has
# anything to show (no ZMP/CoM/foot logs exist yet). Played sped up, since it is
# just the operator taking their time between button presses.
PREROLL_SPEEDUP = 10  # pre-roll playback speed relative to the closed-loop part
PREROLL_LEAD_S = 1.0  # start the pre-roll this long before the first button press

# Gamepad-activated modes, in the column order of control_flags.txt
# (WalkingManager.cpp LOGS block; buttons handled in main.cpp handle_gamepad).
CONTROL_FLAG_LABELS = ('EKF (A)', 'closed loop (X)', 'observer (B)')

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
    est_zmp = _load('ef_zmp_position.txt', 3)  # ZMP from RB-WO foot wrenches
    kf_com = _load('kf_com_position.txt', 3)   # CoM from the LIP Kalman filter (com_kf_step)
    des_com = _load('des_com_position.txt', 3)  # CoM from the PLIP integration (des_LipState.com_pos_)

    # Optional, and on a timeline of its own: control_flags is logged by the main
    # loop (main.cpp) from the very first tick, whereas every channel above only
    # starts once X closed the loop.  Column 3 marks the main-loop ticks that do
    # have a row in those logs, which is what lines the two timelines up below —
    # so it must NOT be truncated to their length here.
    control_flags = _load('control_flags.txt', 4)

    fields = (
        ('des_zmp_position.txt', des_zmp),
        ('ef_zmp_position.txt', est_zmp),
        ('kf_com_position.txt', kf_com),
        ('des_com_position.txt', des_com),
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

    if control_flags is not None and control_flags.shape[1] < 4:
        control_flags = None  # pre-4-column format: no way to align the timelines

    n = min(len(v) for _, v in fields)
    if end_s is not None:
        n = min(n, int(end_s * CONTROL_FREQUENCY_HZ))
    return dict(
        control_flags=control_flags,
        des_zmp=des_zmp[:n], est_zmp=est_zmp[:n], kf_com=kf_com[:n], des_com=des_com[:n],
        box_center=box_center[:n], box_yaw=box_yaw[:n],
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
    est_zmp, kf_com, des_com = data['est_zmp'], data['kf_com'], data['des_com']
    control_flags = data['control_flags']
    if control_flags is None:
        print("[warn] control_flags.txt not found (or in the old 3-column format): the EKF / "
              "closed-loop / observer flag panel is disabled (it needs a run recorded after "
              "that logging was added to the main loop in main.cpp).")

    # control_flags is indexed by main-loop tick, everything else by WalkingManager
    # tick. wbc_ticks[k] is the main-loop tick that produced log row k, so the two
    # are read through it. Ticks before wbc_ticks[0] are the pre-roll: the loop is
    # still open, so only the flag panel has data.
    wbc_ticks, preroll_ticks = None, []
    if control_flags is not None:
        wbc_ticks = np.flatnonzero(control_flags[:, 3] > 0.5)
        if len(wbc_ticks) > 0 and n - len(wbc_ticks) == 1:
            # The run was stopped in the middle of its last tick: WalkingManager had
            # already logged that tick, but the main loop never reached the
            # control_flags log at the end of it. Both logs append in tick order,
            # so the extra row can only be that final one — drop it.
            print(f"[info] the last tick was interrupted mid-way (WalkingManager logs have "
                  f"{n} rows, control_flags marks {len(wbc_ticks)}): dropping that tick.")
            n = len(wbc_ticks)
        if len(wbc_ticks) < n:
            print(f"[warn] control_flags.txt marks {len(wbc_ticks)} closed-loop ticks but the "
                  f"WalkingManager logs have {n} rows — the flag panel is disabled, since the "
                  "two logs are not from the same run.")
            control_flags, wbc_ticks = None, None
        else:
            wbc_ticks = wbc_ticks[:n]
            pre_end = int(wbc_ticks[0])
            pressed = np.any(control_flags[:pre_end, :3] > 0.5, axis=1)
            first_press = int(np.argmax(pressed)) if pressed.any() else pre_end
            lead = int(PREROLL_LEAD_S * CONTROL_FREQUENCY_HZ)
            preroll_ticks = list(range(max(0, first_press - lead), pre_end,
                                       PLAYBACK_STRIDE * PREROLL_SPEEDUP))

    frame_indices = list(range(0, n, PLAYBACK_STRIDE))
    if not frame_indices:
        print(f'[skip] No samples to animate in {folder}.')
        return

    # One flat playlist: the open-loop pre-roll (main-loop ticks) followed by the
    # closed-loop part (WalkingManager ticks).
    frames = [('pre', i) for i in preroll_ticks] + [('wbc', k) for k in frame_indices]

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
    all_xy = np.vstack([des_zmp[:, :2], est_zmp[:, :2], kf_com[:, :2], des_com[:, :2],
                        box_corners_all, foot_corners_all])
    margin = 0.05
    xlim = (all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ylim = (all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)

    # The plot area keeps the whole axes to itself: flags, readout and legend all
    # live in a side panel (an axes with no frame), so that nothing is drawn on
    # top of the feet, the box or the support polygon.
    fig = plt.figure(figsize=(12.5, 8.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.25], wspace=0.04,
                            left=0.06, right=0.985, top=0.93, bottom=0.07)
    ax = fig.add_subplot(grid[0, 0])
    side = fig.add_subplot(grid[0, 1])
    side.axis('off')
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

    (est_trail_line,) = ax.plot([], [], color='tab:purple', linewidth=1.0, linestyle='--',
                                alpha=0.6, zorder=5)
    (est_zmp_point,) = ax.plot([], [], marker='D', markersize=7, linestyle='', color='tab:purple',
                               markeredgecolor='black', zorder=6)

    (com_trail_line,) = ax.plot([], [], color='tab:brown', linewidth=1.0, alpha=0.6, zorder=5)
    (com_point,) = ax.plot([], [], marker='X', markersize=10, linestyle='', color='tab:brown',
                           markeredgecolor='black', zorder=7)

    (des_com_trail_line,) = ax.plot([], [], color='tab:cyan', linewidth=1.0, linestyle='--',
                                    alpha=0.6, zorder=5)
    (des_com_point,) = ax.plot([], [], marker='P', markersize=10, linestyle='', color='tab:cyan',
                               markeredgecolor='black', zorder=7)

    # Side panel, top to bottom: flag badges, numeric readout, legend.
    # These sit slightly inside the panel and draw unclipped: Axes.text() clips
    # its artists to the axes rectangle, which would otherwise cut the rounded
    # box of anything anchored flush against the left edge.
    # PANEL_TOP leaves room above the first badge for its rounded box, which is
    # drawn outside the text's own anchor point.
    PANEL_LEFT = 0.02
    PANEL_TOP = 0.96

    # Gamepad-activated modes: one badge per flag, greyed out until the tick at
    # which the corresponding button was pressed, green with a check mark after.
    flag_texts = []
    if control_flags is not None:
        for i in range(len(CONTROL_FLAG_LABELS)):
            flag_texts.append(side.text(
                PANEL_LEFT, PANEL_TOP - 0.055 * i, '', transform=side.transAxes,
                ha='left', va='top', fontsize=11, fontweight='bold', family='DejaVu Sans',
                bbox=dict(boxstyle='round,pad=0.45', facecolor='0.92', edgecolor='0.6', alpha=0.9)
            ))

    # Monospaced, so that the numbers stay in column as they change.
    info_text = side.text(
        PANEL_LEFT, PANEL_TOP - 0.055 * len(flag_texts) - 0.03, '', transform=side.transAxes,
        ha='left', va='top', fontsize=9, family='DejaVu Sans Mono', linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='0.8', alpha=0.95)
    )

    for text_artist in flag_texts + [info_text]:
        text_artist.set_clip_on(False)
    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center', va='bottom',
                         fontsize=12)

    legend_handles = [
        mpatches.Patch(facecolor='tab:red', edgecolor='tab:red', alpha=0.2, label='ZMP admissible box (moving box)'),
        Line2D([0], [0], color='tab:blue', linewidth=1.0, alpha=0.6, label=f'des. ZMP trail ({TRAIL_LEN} ticks)'),
        Line2D([0], [0], marker='o', markersize=8, linestyle='', color='tab:green', label='des. ZMP (inside box)'),
        Line2D([0], [0], marker='o', markersize=8, linestyle='', color='tab:red', label='des. ZMP (outside box)'),
        Line2D([0], [0], color='tab:purple', linewidth=1.0, linestyle='--',
               marker='D', markersize=7, markerfacecolor='tab:purple', markeredgecolor='black',
               label='est. ZMP (RB-WO) + trail'),
        Line2D([0], [0], color='tab:brown', linewidth=1.0,
               marker='X', markersize=10, markerfacecolor='tab:brown', markeredgecolor='black',
               label='CoM (LIP KF) + trail'),
        Line2D([0], [0], color='tab:cyan', linewidth=1.0, linestyle='--',
               marker='P', markersize=10, markerfacecolor='tab:cyan', markeredgecolor='black',
               label='des. CoM (PLIP integration) + trail'),
        mpatches.Patch(facecolor='tab:blue', edgecolor='black', alpha=0.5, label='Left foot (support, des.)'),
        mpatches.Patch(facecolor='tab:orange', edgecolor='black', alpha=0.5, label='Right foot (support, des.)'),
        Line2D([0], [0], color='black', linewidth=1.8, label='Support polygon (double support)'),
    ]
    all_artists = [box_patch, left_foot_patch, right_foot_patch, support_polygon_line,
                   trail_line, zmp_point, est_trail_line, est_zmp_point, com_trail_line, com_point,
                   des_com_trail_line, des_com_point, info_text, title_text] + flag_texts

    # Artists with nothing to show before the loop is closed (their logs start at
    # wbc_ticks[0]); the foot patches and the support polygon are left out because
    # their visibility is already decided per frame by the contact flags.
    trajectory_artists = [box_patch, trail_line, zmp_point, est_trail_line, est_zmp_point,
                          com_trail_line, com_point, des_com_trail_line, des_com_point]

    def update_flag_panel(main_tick):
        for i, flag_text in enumerate(flag_texts):
            active = control_flags[main_tick, i] > 0.5
            flag_text.set_text(f"{'✓' if active else '✗'}  {CONTROL_FLAG_LABELS[i]}")
            flag_text.set_color('tab:green' if active else '0.45')
            patch = flag_text.get_bbox_patch()
            patch.set_facecolor('#dff2df' if active else '0.92')
            patch.set_edgecolor('tab:green' if active else '0.6')

    def draw_preroll_frame(main_tick):
        """A tick before X: the flag panel is the only thing with data."""
        for artist in trajectory_artists:
            artist.set_visible(False)
        left_foot_patch.set_visible(False)
        right_foot_patch.set_visible(False)
        support_polygon_line.set_visible(False)
        update_flag_panel(main_tick)
        info_text.set_text(
            f"t     {main_tick * DT:8.3f} s\n"
            f"tick  {main_tick} (open loop)\n"
            "\n"
            "waiting for the closed\n"
            "loop (X): no ZMP/CoM/\n"
            "foot logs yet\n"
            "\n"
            f"pre-roll at {PREROLL_SPEEDUP}x speed"
        )
        title_text.set_text("Waiting for the gamepad to close the loop")
        return all_artists

    def draw_frame(frame_idx):
        phase, k = frames[frame_idx]
        if phase == 'pre':
            return draw_preroll_frame(k)
        for artist in trajectory_artists:
            artist.set_visible(True)

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

        est = est_zmp[k, :2]
        est_trail_line.set_data(est_zmp[trail_start:k + 1, 0], est_zmp[trail_start:k + 1, 1])
        est_zmp_point.set_data([est[0]], [est[1]])
        d_est = rot2(-yaw) @ (est - center)
        est_inside = (abs(d_est[0]) <= BOX_LENGTH / 2.0) and (abs(d_est[1]) <= BOX_WIDTH / 2.0)

        com = kf_com[k, :2]
        com_trail_line.set_data(kf_com[trail_start:k + 1, 0], kf_com[trail_start:k + 1, 1])
        com_point.set_data([com[0]], [com[1]])

        dcom = des_com[k, :2]
        des_com_trail_line.set_data(des_com[trail_start:k + 1, 0], des_com[trail_start:k + 1, 1])
        des_com_point.set_data([dcom[0]], [dcom[1]])

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

        if control_flags is not None:
            update_flag_panel(int(wbc_ticks[k]))

        # Main-loop clock, so that time runs continuously across the pre-roll.
        t_now = (wbc_ticks[k] if wbc_ticks is not None else k) * DT
        state_name = WALKING_STATE_NAMES.get(int(walking_state[k]), 'Unknown')
        # Short lines, one quantity each: the side panel is narrow, and the
        # numbers are easier to compare stacked than run together on one line.
        info_text.set_text(
            f"t     {t_now:8.3f} s\n"
            f"tick  {k}/{n - 1}\n"
            f"state {state_name}\n"
            "\n"
            f"des. ZMP  {zmp[0]:6.3f} {zmp[1]:6.3f}  {'IN' if inside else 'OUT':>3}\n"
            f"est. ZMP  {est[0]:6.3f} {est[1]:6.3f}  {'IN' if est_inside else 'OUT':>3}\n"
            f"|des-est| {np.linalg.norm(zmp - est) * 1000:6.1f} mm\n"
            "\n"
            f"des. CoM  {dcom[0]:6.3f} {dcom[1]:6.3f}\n"
            f"KF   CoM  {com[0]:6.3f} {com[1]:6.3f}\n"
            f"|des-KF|  {np.linalg.norm(dcom - com) * 1000:6.1f} mm\n"
            "\n"
            f"box cent. {center[0]:6.3f} {center[1]:6.3f}\n"
            f"box yaw   {np.degrees(yaw):6.1f} deg\n"
            f"box size  {BOX_LENGTH:.2f} x {BOX_WIDTH:.2f} m"
        )
        title_text.set_text("Desired (PLIP) / estimated (RB-WO) ZMP and desired (PLIP) / KF CoM\n"
                            "vs. ZMP box and support foot/feet")

        return all_artists

    # The legend goes right under the readout, whose height depends on how many
    # lines of text it holds — so fill it in, let the figure lay itself out, and
    # measure where it actually ends before anchoring the legend below it. The
    # closed-loop readout is the taller of the two, so measure that one, not the
    # pre-roll frame the animation happens to open on.
    draw_frame(min(len(preroll_ticks), len(frames) - 1))
    fig.canvas.draw()
    info_bottom = side.transAxes.inverted().transform(
        (0.0, info_text.get_window_extent().y0))[1]
    side.legend(handles=legend_handles, loc='upper left',
                bbox_to_anchor=(0.0, info_bottom - 0.04),
                fontsize=8.5, borderaxespad=0.0, labelspacing=0.7, handlelength=2.2,
                framealpha=0.95)

    state = {'idx': 0}

    def update(_frame):
        idx = state['idx']
        artists = draw_frame(idx)
        state['idx'] = (idx + 1) % len(frames)
        return artists

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=lambda: draw_frame(0),
                        interval=PLAYBACK_STRIDE * DT * 1000.0, blit=True, repeat=True)

    running = [True]

    def step(delta):
        """Manually move to the previous/next frame; pauses playback like a video scrubber."""
        if running[0]:
            ani.event_source.stop()
            running[0] = False
        state['idx'] = (state['idx'] + delta) % len(frames)
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

    preroll_note = (f"{len(preroll_ticks)} open-loop pre-roll frames at {PREROLL_SPEEDUP}x + "
                    if preroll_ticks else "")
    print(f"[INFO] {preroll_note}{len(frame_indices)} frames (stride={PLAYBACK_STRIDE}, "
          f"{n} total ticks @ {CONTROL_FREQUENCY_HZ:.0f} Hz) loaded from {folder} — "
          "space: pause/resume, left/right arrows: step, "
          "use the toolbar below the plot to zoom/pan.")
    plt.show()  # no tight_layout: the margins are set on the gridspec above


if __name__ == '__main__':
    main()
