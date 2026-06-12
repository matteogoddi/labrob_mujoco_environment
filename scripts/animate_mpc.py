#!/usr/bin/env python3
"""Animate ISMPC predictions saved by WalkingManager.

Each MPC solve writes two files inside  mpc_data/<t_ms>/:
  x.txt  – (N+1) x 9  rows: [com_px com_py com_pz  com_vx com_vy com_vz  zmp_x zmp_y zmp_z]
  u.txt  –  N    x 3  rows: [zdot_x zdot_y zdot_z]

Run from the build/run directory that contains the mpc_data/ folder:
  python3 scripts/animate_mpc.py [--base-dir <path>]
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
from matplotlib.animation import FuncAnimation
from pathlib import Path

DT_MS = 50.0  # mpc_timestep_msec

# Column indices in x.txt
COM_X, COM_Y, COM_Z = 0, 1, 2
ZMP_X, ZMP_Y, ZMP_Z = 6, 7, 8

# Column indices in u.txt
U_X, U_Y, U_Z = 0, 1, 2


def parse_timestep(name: str):
    try:
        return float(name)
    except ValueError:
        return None


def main() -> None:
    exp_number = input("Enter 0 to animate data from the last simulation or the number of the experiment: ").strip()
    if exp_number == '0':
        base_dir = Path('/tmp/mpc_data')
    else:
        base_dir = Path('experiments/experiment_' + exp_number) / 'mpc_data'
    if not base_dir.exists():
        print(f'[error] Directory not found: {base_dir}')
        return

    folders = sorted(
        [(parse_timestep(p.name), p) for p in base_dir.iterdir()
         if p.is_dir() and parse_timestep(p.name) is not None],
        key=lambda tp: tp[0]
    )
    if not folders:
        print(f'[skip] No numeric timestep folders found in {base_dir}')
        return

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    times = []
    hist_com_x, hist_com_y, hist_com_z = [], [], []
    hist_zmp_x, hist_zmp_y, hist_zmp_z = [], [], []
    hist_u_x,   hist_u_y,   hist_u_z   = [], [], []

    pred_xt_list = []
    pred_com_x_list, pred_com_y_list, pred_com_z_list = [], [], []
    pred_zmp_x_list, pred_zmp_y_list, pred_zmp_z_list = [], [], []

    pred_ut_list = []
    pred_u_x_list, pred_u_y_list, pred_u_z_list = [], [], []

    empty = np.array([])
    t_max_pred = -np.inf

    for t, p in folders:
        x_path = p / 'x.txt'
        u_path = p / 'u.txt'
        if not x_path.exists() or not u_path.exists():
            continue
        x_data = np.loadtxt(x_path, ndmin=2)
        u_data = np.loadtxt(u_path, ndmin=2)
        if x_data.shape[0] == 0 or u_data.shape[0] == 0:
            continue

        x0 = x_data[0]
        times.append(t)
        hist_com_x.append(x0[COM_X]); hist_com_y.append(x0[COM_Y]); hist_com_z.append(x0[COM_Z])
        hist_zmp_x.append(x0[ZMP_X]); hist_zmp_y.append(x0[ZMP_Y]); hist_zmp_z.append(x0[ZMP_Z])
        hist_u_x.append(u_data[0, U_X]); hist_u_y.append(u_data[0, U_Y]); hist_u_z.append(u_data[0, U_Z])

        N_pred = x_data.shape[0] - 1
        if N_pred > 0:
            pred_xt = t + np.arange(1, N_pred + 1, dtype=float) * DT_MS
            pred_xt_list.append(pred_xt)
            pred_com_x_list.append(x_data[1:, COM_X]); pred_com_y_list.append(x_data[1:, COM_Y])
            pred_com_z_list.append(x_data[1:, COM_Z])
            pred_zmp_x_list.append(x_data[1:, ZMP_X]); pred_zmp_y_list.append(x_data[1:, ZMP_Y])
            pred_zmp_z_list.append(x_data[1:, ZMP_Z])
            t_max_pred = max(t_max_pred, pred_xt[-1])
        else:
            pred_xt_list.append(empty)
            for lst in (pred_com_x_list, pred_com_y_list, pred_com_z_list,
                        pred_zmp_x_list, pred_zmp_y_list, pred_zmp_z_list):
                lst.append(empty)

        N_u = u_data.shape[0]
        pred_ut = t + np.arange(0, N_u, dtype=float) * DT_MS
        pred_ut_list.append(pred_ut)
        pred_u_x_list.append(u_data[:, U_X])
        pred_u_y_list.append(u_data[:, U_Y])
        pred_u_z_list.append(u_data[:, U_Z])
        if pred_ut.size:
            t_max_pred = max(t_max_pred, pred_ut[-1])

    if not times:
        print('[skip] Found folders but no readable data')
        return

    times        = np.asarray(times, dtype=float)
    hist_com_x   = np.asarray(hist_com_x);  hist_com_y  = np.asarray(hist_com_y);  hist_com_z  = np.asarray(hist_com_z)
    hist_zmp_x   = np.asarray(hist_zmp_x);  hist_zmp_y  = np.asarray(hist_zmp_y);  hist_zmp_z  = np.asarray(hist_zmp_z)
    hist_u_x     = np.asarray(hist_u_x);    hist_u_y    = np.asarray(hist_u_y);    hist_u_z    = np.asarray(hist_u_z)

    # ------------------------------------------------------------------
    # Figure: 3 rows x 2 cols
    #   Left  col: COM x/y/z  (history blue + ZMP dashed orange + prediction red)
    #   Right col: input zdot_x/y/z  (history blue + prediction red)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=False)
    ax_cx, ax_ux = axes[0]
    ax_cy, ax_uy = axes[1]
    ax_cz, ax_uz = axes[2]

    fig.suptitle('ISMPC predictions', fontsize=12)

    def setup_ax(ax, ylabel, xlabel=''):
        ax.set_ylabel(ylabel); ax.grid(True)
        if xlabel: ax.set_xlabel(xlabel)

    setup_ax(ax_cx, 'x [m]');          setup_ax(ax_ux, r'$\dot{z}_x$ [m/s]')
    setup_ax(ax_cy, 'y [m]');          setup_ax(ax_uy, r'$\dot{z}_y$ [m/s]')
    setup_ax(ax_cz, 'z [m]', 't [ms]'); setup_ax(ax_uz, r'$\dot{z}_z$ [m/s]', 't [ms]')

    def make_state_lines(ax):
        (hc,)  = ax.plot([], [], lw=2,   color='tab:blue',   label='COM hist')
        (hz,)  = ax.plot([], [], lw=1.5, color='tab:orange', label='ZMP hist', ls='--')
        (pc,)  = ax.plot([], [], lw=2,   color='red',        label='COM pred')
        (pz,)  = ax.plot([], [], lw=1.5, color='red',        label='ZMP pred', ls='--')
        (dot,) = ax.plot([], [], 'o',    color='tab:blue')
        ax.legend(fontsize=7)
        return hc, hz, pc, pz, dot

    (hcx, hzx, pcx, pzx, dotx) = make_state_lines(ax_cx)
    (hcy, hzy, pcy, pzy, doty) = make_state_lines(ax_cy)
    (hcz, hzz, pcz, pzz, dotz) = make_state_lines(ax_cz)

    def make_input_lines(ax):
        (hi,)  = ax.plot([], [], lw=2,   color='tab:blue',  label='applied')
        (pi,)  = ax.plot([], [], lw=2,   color='red',       label='pred')
        (dot,) = ax.plot([], [], 'o',    color='tab:blue')
        ax.legend(fontsize=7)
        return hi, pi, dot

    (hux, pux, dotux) = make_input_lines(ax_ux)
    (huy, puy, dotuy) = make_input_lines(ax_uy)
    (huz, puz, dotuz) = make_input_lines(ax_uz)

    # Set axis limits once
    def set_lim(ax, t_arr, *val_groups):
        t_lo = float(np.min(t_arr))
        t_hi = max(float(np.max(t_arr)), t_max_pred if t_max_pred > -np.inf else float(np.max(t_arr)))
        ax.set_xlim(t_lo, t_hi)
        flat = []
        for g in val_groups:
            for a in (g if isinstance(g, (list, tuple)) else [g]):
                a = np.asarray(a)
                if a.size and np.any(np.isfinite(a)):
                    flat.append(a[np.isfinite(a)])
        if not flat: return
        V = np.concatenate(flat)
        vlo, vhi = float(np.min(V)), float(np.max(V))
        margin = max(1e-4, 0.15 * (vhi - vlo))
        ax.set_ylim(vlo - margin, vhi + margin)

    set_lim(ax_cx, times, hist_com_x, hist_zmp_x, pred_com_x_list, pred_zmp_x_list)
    set_lim(ax_cy, times, hist_com_y, hist_zmp_y, pred_com_y_list, pred_zmp_y_list)
    set_lim(ax_cz, times, hist_com_z, hist_zmp_z, pred_com_z_list, pred_zmp_z_list)
    set_lim(ax_ux, times, hist_u_x, pred_u_x_list)
    set_lim(ax_uy, times, hist_u_y, pred_u_y_list)
    set_lim(ax_uz, times, hist_u_z, pred_u_z_list)

    all_artists = (hcx, hzx, pcx, pzx, dotx,
                   hcy, hzy, pcy, pzy, doty,
                   hcz, hzz, pcz, pzz, dotz,
                   hux, pux, dotux, huy, puy, dotuy, huz, puz, dotuz)

    def init():
        for a in all_artists: a.set_data([], [])
        return all_artists

    def update(i):
        ts = times[:i + 1]

        # COM + ZMP history
        hcx.set_data(ts, hist_com_x[:i+1]); hzx.set_data(ts, hist_zmp_x[:i+1]); dotx.set_data([ts[-1]], [hist_com_x[i]])
        hcy.set_data(ts, hist_com_y[:i+1]); hzy.set_data(ts, hist_zmp_y[:i+1]); doty.set_data([ts[-1]], [hist_com_y[i]])
        hcz.set_data(ts, hist_com_z[:i+1]); hzz.set_data(ts, hist_zmp_z[:i+1]); dotz.set_data([ts[-1]], [hist_com_z[i]])

        # Input history
        if i > 0:
            ts_u = times[:i]
            hux.set_data(ts_u, hist_u_x[:i]); huy.set_data(ts_u, hist_u_y[:i]); huz.set_data(ts_u, hist_u_z[:i])
        else:
            hux.set_data([], []); huy.set_data([], []); huz.set_data([], [])
        dotux.set_data([ts[-1]], [hist_u_x[i]]); dotuy.set_data([ts[-1]], [hist_u_y[i]]); dotuz.set_data([ts[-1]], [hist_u_z[i]])

        # State predictions
        pxt = pred_xt_list[i]
        if pxt.size > 0:
            pcx.set_data(pxt, pred_com_x_list[i]); pzx.set_data(pxt, pred_zmp_x_list[i])
            pcy.set_data(pxt, pred_com_y_list[i]); pzy.set_data(pxt, pred_zmp_y_list[i])
            pcz.set_data(pxt, pred_com_z_list[i]); pzz.set_data(pxt, pred_zmp_z_list[i])
        else:
            for a in (pcx, pzx, pcy, pzy, pcz, pzz): a.set_data([], [])

        # Input predictions
        put = pred_ut_list[i]
        if put.size > 0:
            pux.set_data(put, pred_u_x_list[i]); puy.set_data(put, pred_u_y_list[i]); puz.set_data(put, pred_u_z_list[i])
        else:
            pux.set_data([], []); puy.set_data([], []); puz.set_data([], [])

        return all_artists

    ani = FuncAnimation(fig, update, frames=len(times), init_func=init,
                        interval=80, blit=True, repeat=False)

    running = [True]
    def toggle(event):
        if event.key == ' ':
            if running[0]: ani.event_source.stop()
            else:          ani.event_source.start()
            running[0] = not running[0]
    fig.canvas.mpl_connect('key_press_event', toggle)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()