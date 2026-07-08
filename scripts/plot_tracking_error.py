import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.spatial.transform import Rotation as R

# -------------------------
# CLI ARGUMENTS
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--show_curr", action="store_true",
                    help="Show current values plots")
args = parser.parse_args()

file_path = "/tmp/wbc_log.txt"

# Load data (skip header)
data = np.loadtxt(file_path, delimiter=",", skiprows=1)

# -------------------------
# Column mapping
# -------------------------
t_ms = data[:, 0]

com = data[:, 1:4]
com_des = data[:, 4:7]

com_vel = data[:, 7:10]
com_vel_des = data[:, 10:13]

l_sole = data[:, 13:16]
l_sole_des = data[:, 16:19]

r_sole = data[:, 19:22]
r_sole_des = data[:, 22:25]

# quaternions
q_pelvis = data[:, 25:29]
q_torso  = data[:, 29:33]

# -------------------------
# Convert quaternion -> RPY
# -------------------------
pelvis_rpy = R.from_quat(q_pelvis[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=False)
torso_rpy  = R.from_quat(q_torso[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=False)

# -------------------------
# Errors
# -------------------------
com_err = com_des - com
com_vel_err = com_vel_des - com_vel
l_sole_err = l_sole_des - l_sole
r_sole_err = r_sole_des - r_sole

# =========================================================
# FIGURE 1 — ERRORS
# =========================================================
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(t_ms, com_err[:, 0], label="x error")
axs[0, 0].plot(t_ms, com_err[:, 1], label="y error")
axs[0, 0].plot(t_ms, com_err[:, 2], label="z error")
axs[0, 0].set_title("CoM Position Error")
axs[0, 0].set_xlabel("Time [ms]")
axs[0, 0].set_ylabel("Error [m]")
axs[0, 0].grid(True)
axs[0, 0].legend()

axs[0, 1].plot(t_ms, com_vel_err[:, 0], label="vx error")
axs[0, 1].plot(t_ms, com_vel_err[:, 1], label="vy error")
axs[0, 1].plot(t_ms, com_vel_err[:, 2], label="vz error")
axs[0, 1].set_title("CoM Velocity Error")
axs[0, 1].set_xlabel("Time [ms]")
axs[0, 1].set_ylabel("Error [m/s]")
axs[0, 1].grid(True)
axs[0, 1].legend()

axs[1, 0].plot(t_ms, l_sole_err[:, 0], label="x error")
axs[1, 0].plot(t_ms, l_sole_err[:, 1], label="y error")
axs[1, 0].plot(t_ms, l_sole_err[:, 2], label="z error")
axs[1, 0].set_title("Left Sole Position Error")
axs[1, 0].set_xlabel("Time [ms]")
axs[1, 0].set_ylabel("Error [m]")
axs[1, 0].grid(True)
axs[1, 0].legend()

axs[1, 1].plot(t_ms, r_sole_err[:, 0], label="x error")
axs[1, 1].plot(t_ms, r_sole_err[:, 1], label="y error")
axs[1, 1].plot(t_ms, r_sole_err[:, 2], label="z error")
axs[1, 1].set_title("Right Sole Position Error")
axs[1, 1].set_xlabel("Time [ms]")
axs[1, 1].set_ylabel("Error [m]")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()

# =========================================================
# FIGURE 2 — CURRENT VALUES (optional)
# =========================================================
if args.show_curr:

    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))

    axs2[0, 0].plot(t_ms, com[:, 0], label="x")
    axs2[0, 0].plot(t_ms, com[:, 1], label="y")
    axs2[0, 0].plot(t_ms, com[:, 2], label="z")
    axs2[0, 0].set_title("CoM Position (Current)")
    axs2[0, 0].grid(True)
    axs2[0, 0].legend()

    axs2[0, 1].plot(t_ms, com_vel[:, 0], label="vx")
    axs2[0, 1].plot(t_ms, com_vel[:, 1], label="vy")
    axs2[0, 1].plot(t_ms, com_vel[:, 2], label="vz")
    axs2[0, 1].set_title("CoM Velocity (Current)")
    axs2[0, 1].grid(True)
    axs2[0, 1].legend()

    axs2[1, 0].plot(t_ms, l_sole[:, 0], label="x")
    axs2[1, 0].plot(t_ms, l_sole[:, 1], label="y")
    axs2[1, 0].plot(t_ms, l_sole[:, 2], label="z")
    axs2[1, 0].set_title("Left Sole Position (Current)")
    axs2[1, 0].grid(True)
    axs2[1, 0].legend()

    axs2[1, 1].plot(t_ms, r_sole[:, 0], label="x")
    axs2[1, 1].plot(t_ms, r_sole[:, 1], label="y")
    axs2[1, 1].plot(t_ms, r_sole[:, 2], label="z")
    axs2[1, 1].set_title("Right Sole Position (Current)")
    axs2[1, 1].grid(True)
    axs2[1, 1].legend()

    plt.tight_layout()

# =========================================================
# FIGURE 3 — RPY ORIENTATION (NEW)
# =========================================================
fig3, axs3 = plt.subplots(1, 2, figsize=(12, 4))

# Pelvis
axs3[0].plot(t_ms, pelvis_rpy[:, 0], label="roll")
axs3[0].plot(t_ms, pelvis_rpy[:, 1], label="pitch")
axs3[0].plot(t_ms, pelvis_rpy[:, 2], label="yaw")
axs3[0].set_title("Pelvis Orientation (RPY)")
axs3[0].set_xlabel("Time [ms]")
axs3[0].set_ylabel("Angle [rad]")
axs3[0].grid(True)
axs3[0].legend()

# Torso
axs3[1].plot(t_ms, torso_rpy[:, 0], label="roll")
axs3[1].plot(t_ms, torso_rpy[:, 1], label="pitch")
axs3[1].plot(t_ms, torso_rpy[:, 2], label="yaw")
axs3[1].set_title("Torso Orientation (RPY)")
axs3[1].set_xlabel("Time [ms]")
axs3[1].set_ylabel("Angle [rad]")
axs3[1].grid(True)
axs3[1].legend()

plt.tight_layout()

plt.show()