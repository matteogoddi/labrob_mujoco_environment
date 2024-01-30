#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

class Trajectory:
    def __init__(self, x, y, z, roll=None, pitch=None, yaw=None):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

# Read file composed of one position per line:
def read_position_file(path):
    # Init lists:
    xx = []
    yy = []
    zz = []

    # Read file:
    with open(path) as f:
        for l in f.readlines():
            p = l.rstrip('\n').split()
            xx.append(float(p[0]))
            yy.append(float(p[1]))
            zz.append(float(p[2]))

    # Setup np arrays:
    xx = np.array(xx)
    yy = np.array(yy)
    zz = np.array(zz)
    trajectory = Trajectory(xx, yy, zz)
    return trajectory

if __name__ == '__main__':
    mpc_com_file_path = '/tmp/mpc_com.txt'
    mpc_zmp_file_path = '/tmp/mpc_zmp.txt'
    com_file_path = '/tmp/com.txt'
    lsole_file_path = '/tmp/lsole.txt'
    rsole_file_path = '/tmp/rsole.txt'
    lsole_des_file_path = '/tmp/lsole_des.txt'
    rsole_des_file_path = '/tmp/rsole_des.txt'

    mpc_com_trajectory = read_position_file(mpc_com_file_path)
    mpc_zmp_trajectory = read_position_file(mpc_zmp_file_path)
    com_trajectory = read_position_file(com_file_path)
    lsole_trajectory = read_position_file(lsole_file_path)
    rsole_trajectory = read_position_file(rsole_file_path)
    lsole_des_trajectory = read_position_file(lsole_des_file_path)
    rsole_des_trajectory = read_position_file(rsole_des_file_path)

    delta_t = 0.01
    samples = mpc_com_trajectory.x.shape[0]
    tt = np.linspace(0.0, delta_t * samples, samples)

    fig = plt.figure()

    ### Subplot CoM/ZMP ###
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax1.plot(mpc_com_trajectory.x, mpc_com_trajectory.y, label='CoM ref')
    ax1.plot(mpc_zmp_trajectory.x, mpc_zmp_trajectory.y, label='ZMP ref')
    ax1.plot(com_trajectory.x, com_trajectory.y, label='CoM')

    ax1.legend()
    ax1.grid()
    ax1.axis('equal')

    ### Subplot ZMP.x ###
    ax3 = fig.add_subplot(4, 1, 2)
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')

    ax3.plot(tt, mpc_com_trajectory.x, label='CoM.x ref')
    ax3.plot(tt, mpc_zmp_trajectory.x, label='ZMP.x ref')
    ax3.plot(tt, com_trajectory.x, label='CoM.x')

    ax3.legend()
    ax3.grid()
    
    ### Subplot ZMP.y ###
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.set_xlabel('t')
    ax3.set_ylabel('y')

    ax3.plot(tt, mpc_com_trajectory.y, label='CoM.y ref')
    ax3.plot(tt, mpc_zmp_trajectory.y, label='ZMP.y ref')
    ax3.plot(tt, com_trajectory.y, label='CoM.y')

    ax3.legend()
    ax3.grid()

    ### Subplot ZMP.z ###
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.set_xlabel('t')
    ax4.set_ylabel('y')

    ax4.plot(tt, mpc_com_trajectory.z, label='CoM.z ref')
    ax4.plot(tt, mpc_zmp_trajectory.z, label='ZMP.z ref')
    ax4.plot(tt, com_trajectory.z, label='CoM.z')

    ax4.legend()
    ax4.grid()

    # Figure soles:
    fig_soles = plt.figure()
    # lsole.x:
    ax_lsole_x = fig_soles.add_subplot(3, 2, 1)
    ax_lsole_x.set_xlabel('t')
    ax_lsole_x.set_ylabel('x')
    ax_lsole_x.plot(tt, lsole_des_trajectory.x, label='lsole_des.x')
    ax_lsole_x.plot(tt, lsole_trajectory.x, label='lsole.x')
    ax_lsole_x.legend()
    ax_lsole_x.grid()
    # lsole.y:
    ax_lsole_y = fig_soles.add_subplot(3, 2, 3)
    ax_lsole_y.set_xlabel('t')
    ax_lsole_y.set_ylabel('y')
    ax_lsole_y.plot(tt, lsole_des_trajectory.y, label='lsole_des.y')
    ax_lsole_y.plot(tt, lsole_trajectory.y, label='lsole.y')
    ax_lsole_y.legend()
    ax_lsole_y.grid()
    # lsole.z:
    ax_lsole_z = fig_soles.add_subplot(3, 2, 5)
    ax_lsole_z.set_xlabel('t')
    ax_lsole_z.set_ylabel('z')
    ax_lsole_z.plot(tt, lsole_des_trajectory.z, label='lsole_des.z')
    ax_lsole_z.plot(tt, lsole_trajectory.z, label='lsole.z')
    ax_lsole_z.legend()
    ax_lsole_z.grid()
    # rsole.x:
    ax_rsole_x = fig_soles.add_subplot(3, 2, 2)
    ax_rsole_x.set_xlabel('t')
    ax_rsole_x.set_ylabel('x')
    ax_rsole_x.plot(tt, rsole_des_trajectory.x, label='rsole_des.x')
    ax_rsole_x.plot(tt, rsole_trajectory.x, label='rsole.x')
    ax_rsole_x.legend()
    ax_rsole_x.grid()
    # rsole.y:
    ax_rsole_y = fig_soles.add_subplot(3, 2, 4)
    ax_rsole_y.set_xlabel('t')
    ax_rsole_y.set_ylabel('y')
    ax_rsole_y.plot(tt, rsole_des_trajectory.y, label='rsole_des.y')
    ax_rsole_y.plot(tt, rsole_trajectory.y, label='rsole.y')
    ax_rsole_y.legend()
    ax_rsole_y.grid()
    # rsole.z:
    ax_rsole_z = fig_soles.add_subplot(3, 2, 6)
    ax_rsole_z.set_xlabel('t')
    ax_rsole_z.set_ylabel('z')
    ax_rsole_z.plot(tt, rsole_des_trajectory.z, label='rsole_des.z')
    ax_rsole_z.plot(tt, rsole_trajectory.z, label='rsole.z')
    ax_rsole_z.legend()
    ax_rsole_z.grid()

    plt.show()