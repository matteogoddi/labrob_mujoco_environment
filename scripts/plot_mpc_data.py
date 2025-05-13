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
    p_lsole_file_path = '/tmp/p_lsole.txt'
    p_rsole_file_path = '/tmp/p_rsole.txt'
    v_lsole_file_path = '/tmp/v_lsole.txt'
    v_rsole_file_path = '/tmp/v_rsole.txt'
    p_lsole_des_file_path = '/tmp/p_lsole_des.txt'
    p_rsole_des_file_path = '/tmp/p_rsole_des.txt'
    v_lsole_des_file_path = '/tmp/v_lsole_des.txt'
    v_rsole_des_file_path = '/tmp/v_rsole_des.txt'
    angular_momentum_file_path = '/tmp/angular_momentum.txt'

    mpc_com_trajectory = read_position_file(mpc_com_file_path)
    mpc_zmp_trajectory = read_position_file(mpc_zmp_file_path)
    com_trajectory = read_position_file(com_file_path)
    p_lsole_trajectory = read_position_file(p_lsole_file_path)
    p_rsole_trajectory = read_position_file(p_rsole_file_path)
    v_lsole_trajectory = read_position_file(v_lsole_file_path)
    v_rsole_trajectory = read_position_file(v_rsole_file_path)
    p_lsole_des_trajectory = read_position_file(p_lsole_des_file_path)
    p_rsole_des_trajectory = read_position_file(p_rsole_des_file_path)
    v_lsole_des_trajectory = read_position_file(v_lsole_des_file_path)
    v_rsole_des_trajectory = read_position_file(v_rsole_des_file_path)
    angular_momentum_trajectory = read_position_file(angular_momentum_file_path)

    delta_t = 1e-3
    samples = mpc_com_trajectory.x.shape[0]
    tt = np.linspace(0.0, delta_t * samples, samples)

    fig = plt.figure()

    ### Subplot CoM/ZMP ###
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax1.plot(mpc_com_trajectory.x, mpc_com_trajectory.y, label='CoM ref')
    ax1.plot(mpc_zmp_trajectory.x, mpc_zmp_trajectory.y, label='ZMP ref')
    ax1.plot(com_trajectory.x, com_trajectory.y, label='CoM')

    ax1.legend()
    ax1.grid()
    ax1.axis('equal')

    ### Subplot ZMP.x ###
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')

    ax2.plot(tt, mpc_com_trajectory.x, label='CoM.x ref')
    ax2.plot(tt, mpc_zmp_trajectory.x, label='ZMP.x ref')
    ax2.plot(tt, com_trajectory.x, label='CoM.x')

    ax2.legend()
    ax2.grid()
    
    ### Subplot ZMP.y ###
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_xlabel('t')
    ax3.set_ylabel('y')

    ax3.plot(tt, mpc_com_trajectory.y, label='CoM.y ref')
    ax3.plot(tt, mpc_zmp_trajectory.y, label='ZMP.y ref')
    ax3.plot(tt, com_trajectory.y, label='CoM.y')

    ax3.legend()
    ax3.grid()

    ### Subplot ZMP.z ###
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_xlabel('t')
    ax4.set_ylabel('z')

    ax4.plot(tt, mpc_com_trajectory.z, label='CoM.z ref')
    ax4.plot(tt, mpc_zmp_trajectory.z, label='ZMP.z ref')
    ax4.plot(tt, com_trajectory.z, label='CoM.z')

    ax4.legend()
    ax4.grid()

    fig.savefig(f"images/mpc/CoM_ZMP.png", bbox_inches='tight')

    # Figure soles:
    fig_soles = plt.figure()
    # lsole.x:
    ax_lsole_x = fig_soles.add_subplot(3, 2, 1)
    ax_lsole_x.set_xlabel('t')
    ax_lsole_x.set_ylabel('x')
    ax_lsole_x.plot(tt, p_lsole_des_trajectory.x, label='p_lsole_des.x')
    ax_lsole_x.plot(tt, p_lsole_trajectory.x, label='p_lsole.x')
    ax_lsole_x.legend()
    ax_lsole_x.grid()
    # lsole.y:
    ax_lsole_y = fig_soles.add_subplot(3, 2, 3)
    ax_lsole_y.set_xlabel('t')
    ax_lsole_y.set_ylabel('y')
    ax_lsole_y.plot(tt, p_lsole_des_trajectory.y, label='p_lsole_des.y')
    ax_lsole_y.plot(tt, p_lsole_trajectory.y, label='p_lsole.y')
    ax_lsole_y.legend()
    ax_lsole_y.grid()
    # lsole.z:
    ax_lsole_z = fig_soles.add_subplot(3, 2, 5)
    ax_lsole_z.set_xlabel('t')
    ax_lsole_z.set_ylabel('z')
    ax_lsole_z.plot(tt, p_lsole_des_trajectory.z, label='p_lsole_des.z')
    ax_lsole_z.plot(tt, p_lsole_trajectory.z, label='p_lsole.z')
    ax_lsole_z.legend()
    ax_lsole_z.grid()
    # rsole.x:
    ax_rsole_x = fig_soles.add_subplot(3, 2, 2)
    ax_rsole_x.set_xlabel('t')
    ax_rsole_x.set_ylabel('x')
    ax_rsole_x.plot(tt, p_rsole_des_trajectory.x, label='p_rsole_des.x')
    ax_rsole_x.plot(tt, p_rsole_trajectory.x, label='p_rsole.x')
    ax_rsole_x.legend()
    ax_rsole_x.grid()
    # rsole.y:
    ax_rsole_y = fig_soles.add_subplot(3, 2, 4)
    ax_rsole_y.set_xlabel('t')
    ax_rsole_y.set_ylabel('y')
    ax_rsole_y.plot(tt, p_rsole_des_trajectory.y, label='p_rsole_des.y')
    ax_rsole_y.plot(tt, p_rsole_trajectory.y, label='p_rsole.y')
    ax_rsole_y.legend()
    ax_rsole_y.grid()
    # rsole.z:
    ax_rsole_z = fig_soles.add_subplot(3, 2, 6)
    ax_rsole_z.set_xlabel('t')
    ax_rsole_z.set_ylabel('z')
    ax_rsole_z.plot(tt, p_rsole_des_trajectory.z, label='p_rsole_des.z')
    ax_rsole_z.plot(tt, p_rsole_trajectory.z, label='p_rsole.z')
    ax_rsole_z.legend()
    ax_rsole_z.grid()

    fig_soles.savefig(f"images/mpc/soles.png", bbox_inches='tight')

    # Figure soles (velocity):
    fig_v_soles = plt.figure()
    # v_lsole.x:
    ax_v_lsole_x = fig_v_soles.add_subplot(3, 2, 1)
    ax_v_lsole_x.set_xlabel('t')
    ax_v_lsole_x.set_ylabel('x')
    ax_v_lsole_x.plot(tt, v_lsole_des_trajectory.x, label='v_lsole_des.x')
    ax_v_lsole_x.plot(tt, v_lsole_trajectory.x, label='v_lsole.x')
    ax_v_lsole_x.legend()
    ax_v_lsole_x.grid()
    # v_lsole.y:
    ax_v_lsole_y = fig_v_soles.add_subplot(3, 2, 3)
    ax_v_lsole_y.set_xlabel('t')
    ax_v_lsole_y.set_ylabel('y')
    ax_v_lsole_y.plot(tt, v_lsole_des_trajectory.y, label='v_lsole_des.y')
    ax_v_lsole_y.plot(tt, v_lsole_trajectory.y, label='v_lsole.y')
    ax_v_lsole_y.legend()
    ax_v_lsole_y.grid()
    # v_lsole.z:
    ax_v_lsole_z = fig_v_soles.add_subplot(3, 2, 5)
    ax_v_lsole_z.set_xlabel('t')
    ax_v_lsole_z.set_ylabel('z')
    ax_v_lsole_z.plot(tt, v_lsole_des_trajectory.z, label='v_lsole_des.z')
    ax_v_lsole_z.plot(tt, v_lsole_trajectory.z, label='v_lsole.z')
    ax_v_lsole_z.legend()
    ax_v_lsole_z.grid()
    # v_rsole.x:
    ax_v_rsole_x = fig_v_soles.add_subplot(3, 2, 2)
    ax_v_rsole_x.set_xlabel('t')
    ax_v_rsole_x.set_ylabel('x')
    ax_v_rsole_x.plot(tt, v_rsole_des_trajectory.x, label='v_rsole_des.x')
    ax_v_rsole_x.plot(tt, v_rsole_trajectory.x, label='v_rsole.x')
    ax_v_rsole_x.legend()
    ax_v_rsole_x.grid()
    # v_rsole.y:
    ax_v_rsole_y = fig_v_soles.add_subplot(3, 2, 4)
    ax_v_rsole_y.set_xlabel('t')
    ax_v_rsole_y.set_ylabel('y')
    ax_v_rsole_y.plot(tt, v_rsole_des_trajectory.y, label='v_rsole_des.y')
    ax_v_rsole_y.plot(tt, v_rsole_trajectory.y, label='v_rsole.y')
    ax_v_rsole_y.legend()
    ax_v_rsole_y.grid()
    # v_rsole.z:
    ax_v_rsole_z = fig_v_soles.add_subplot(3, 2, 6)
    ax_v_rsole_z.set_xlabel('t')
    ax_v_rsole_z.set_ylabel('z')
    ax_v_rsole_z.plot(tt, v_rsole_des_trajectory.z, label='v_rsole_des.z')
    ax_v_rsole_z.plot(tt, v_rsole_trajectory.z, label='v_rsole.z')
    ax_v_rsole_z.legend()
    ax_v_rsole_z.grid()

    fig_v_soles.savefig(f"images/mpc/soles_velocity.png", bbox_inches='tight')

    # Angular momentum:
    fig_angular_momentum = plt.figure()
    # Angular momentum (x):
    ax_angular_momentum_x = fig_angular_momentum.add_subplot(3, 1, 1)
    ax_angular_momentum_x.set_xlabel('t')
    ax_angular_momentum_x.set_xlabel('angular momentum (x)')
    ax_angular_momentum_x.plot(tt, angular_momentum_trajectory.x, label='ang. momentum (x)')
    ax_angular_momentum_x.legend()
    ax_angular_momentum_x.grid()
    # Angular momentum (y):
    ax_angular_momentum_y = fig_angular_momentum.add_subplot(3, 1, 2)
    ax_angular_momentum_y.set_xlabel('t')
    ax_angular_momentum_y.set_xlabel('angular momentum (y)')
    ax_angular_momentum_y.plot(tt, angular_momentum_trajectory.y, label='ang. momentum (y)')
    ax_angular_momentum_y.legend()
    ax_angular_momentum_y.grid()
    # Angular momentum (z):
    ax_angular_momentum_z = fig_angular_momentum.add_subplot(3, 1, 3)
    ax_angular_momentum_z.set_xlabel('t')
    ax_angular_momentum_z.set_xlabel('angular momentum (z)')
    ax_angular_momentum_z.plot(tt, angular_momentum_trajectory.z, label='ang. momentum (z)')
    ax_angular_momentum_z.legend()
    ax_angular_momentum_z.grid()

    fig_angular_momentum.savefig(f"images/mpc/angular_momentum.png", bbox_inches='tight')

    # plt.show()