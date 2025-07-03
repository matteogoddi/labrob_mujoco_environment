import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import matplotlib.animation as animation
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':

    number = input("Enter 0 to plot data from the last simulation or the number of the experiment: ")
    if number == '0':
        folder = '/tmp'
    else:
        folder = 'experiments/experiment_' + number
    mpc_com_trajectory : np.ndarray = np.loadtxt(folder + '/mpc_com.txt')
    mpc_zmp_trajectory : np.ndarray = np.loadtxt(folder + '/mpc_zmp.txt')
    com_trajectory : np.ndarray = np.loadtxt(folder + '/com.txt')
    p_lsole_trajectory : np.ndarray = np.loadtxt(folder + '/p_lsole.txt')
    p_rsole_trajectory : np.ndarray = np.loadtxt(folder + '/p_rsole.txt')
    v_lsole_trajectory : np.ndarray = np.loadtxt(folder + '/v_lsole.txt')
    v_rsole_trajectory : np.ndarray = np.loadtxt(folder + '/v_rsole.txt')
    p_lsole_des_trajectory : np.ndarray = np.loadtxt(folder + '/p_lsole_des.txt')
    p_rsole_des_trajectory : np.ndarray = np.loadtxt(folder + '/p_rsole_des.txt')
    v_lsole_des_trajectory : np.ndarray = np.loadtxt(folder + '/v_lsole_des.txt')
    v_rsole_des_trajectory : np.ndarray = np.loadtxt(folder + '/v_rsole_des.txt')
    angular_momentum_trajectory : np.ndarray = np.loadtxt(folder + '/angular_momentum.txt')
    mpc_predictions_trajectory : np.ndarray = np.loadtxt(folder + '/mpc_predictions.txt')

    delta = 1e-3
    num_samples = mpc_com_trajectory.shape[0]
    t = np.linspace(0.0, delta * num_samples, num_samples)   

    if not os.path.exists('images/mpc'):
        os.makedirs('images/mpc') 

    # Plot CoM simulation and real data
    fig, ax = plt.subplots()
    ax.plot(t, mpc_com_trajectory[:, 0], label='Desired CoM X', color='blue')
    ax.plot(t, mpc_com_trajectory[:, 1], label='Desired CoM Y', color='orange')
    ax.plot(t, mpc_com_trajectory[:, 2], label='Desired CoM Z', color='green')
    ax.plot(t, mpc_zmp_trajectory[:, 0], label='ZMP X', color='red', linestyle='--')
    ax.plot(t, mpc_zmp_trajectory[:, 1], label='ZMP Y', color='purple', linestyle='--')
    ax.plot(t, mpc_zmp_trajectory[:, 2], label='ZMP Z', color='brown', linestyle='--')
    ax.plot(t, com_trajectory[:, 0], label='CoM X', color='cyan', linestyle=':')
    ax.plot(t, com_trajectory[:, 1], label='CoM Y', color='magenta', linestyle=':')
    ax.plot(t, com_trajectory[:, 2], label='CoM Z', color='yellow', linestyle=':')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('CoM Position [m]')
    ax.set_title('CoM Position: Simulation vs Real')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/mpc/com_plot.png")
    plt.close(fig) 

    # Plot angular momentum
    fig, ax = plt.subplots()
    ax.plot(t, angular_momentum_trajectory[:, 0], label='Angular Momentum X', color='blue')
    ax.plot(t, angular_momentum_trajectory[:, 1], label='Angular Momentum Y', color='orange')
    ax.plot(t, angular_momentum_trajectory[:, 2], label='Angular Momentum Z', color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular Momentum [kg*m^2/s]')
    ax.set_title('Angular Momentum')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/mpc/angular_momentum_plot.png")
    plt.close(fig)

    # Plot left sole position
    fig, ax = plt.subplots()
    ax.plot(t, p_lsole_trajectory[:, 0], label='Left Sole X', color='blue')
    ax.plot(t, p_lsole_trajectory[:, 1], label='Left Sole Y', color='orange')
    ax.plot(t, p_lsole_trajectory[:, 2], label='Left Sole Z', color='green')
    ax.plot(t, p_lsole_des_trajectory[:, 0], label='Desired Left Sole X', color='red', linestyle='--')
    ax.plot(t, p_lsole_des_trajectory[:, 1], label='Desired Left Sole Y', color='purple', linestyle='--')
    ax.plot(t, p_lsole_des_trajectory[:, 2], label='Desired Left Sole Z', color='brown', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Left Sole Position [m]')
    ax.set_title('Left Sole Position: Simulation vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/mpc/left_sole_position_plot.png")
    plt.close(fig)

    # Plot right sole position
    fig, ax = plt.subplots()
    ax.plot(t, p_rsole_trajectory[:, 0], label='Right Sole X', color='blue')
    ax.plot(t, p_rsole_trajectory[:, 1], label='Right Sole Y', color='orange')
    ax.plot(t, p_rsole_trajectory[:, 2], label='Right Sole Z', color='green')
    ax.plot(t, p_rsole_des_trajectory[:, 0], label='Desired Right Sole X', color='red', linestyle='--')
    ax.plot(t, p_rsole_des_trajectory[:, 1], label='Desired Right Sole Y', color='purple', linestyle='--')
    ax.plot(t, p_rsole_des_trajectory[:, 2], label='Desired Right Sole Z', color='brown', linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Right Sole Position [m]')
    ax.set_title('Right Sole Position: Simulation vs Desired')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/mpc/right_sole_position_plot.png")
    plt.close(fig)

    if number != '0':
        print(number)
        print("ciao")

        # real_com_trajectory : np.ndarray = np.loadtxt(folder + '/real_com.txt')
        # # Animazione traiettoria CoM sul piano XY
        # fig, ax = plt.subplots()
        # ax.set_xlim(np.min(com_trajectory[:, 0]) - 0.1, np.max(com_trajectory[:, 0]) + 0.1)
        # ax.set_ylim(np.min(com_trajectory[:, 1]) - 0.1, np.max(com_trajectory[:, 1]) + 0.1)
        # ax.set_xlabel('CoM X [m]')
        # ax.set_ylabel('CoM Y [m]')
        # ax.set_title('CoM Trajectory in XY Plane')
        # ax.grid(True)
        # simulation_line, = ax.plot([], [], 'b-', label='Simulated CoM Trajectory')
        # simulation_point, = ax.plot([], [], 'ro')  # Punto attuale
        # real_line, = ax.plot([], [], 'r-', label='Real CoM Trajectory')
        # real_point, = ax.plot([], [], 'ro')  # Punto attuale
        # ax.legend()

        # def init():
        #     simulation_line.set_data([], [])
        #     simulation_point.set_data([], [])
        #     real_line.set_data([], [])
        #     real_point.set_data([], [])
        #     return simulation_line, simulation_point, real_line, real_point

        # def update(frame):
        #     simulation_line.set_data(com_trajectory[:frame, 0], com_trajectory[:frame, 1])
        #     simulation_point.set_data([com_trajectory[frame, 0]], [com_trajectory[frame, 1]])
        #     real_line.set_data(real_com_trajectory[:frame, 0], real_com_trajectory[:frame, 1])
        #     real_point.set_data([real_com_trajectory[frame, 0]], [real_com_trajectory[frame, 1]])
        #     return simulation_line, simulation_point, real_line, real_point

        # skip = 100  # salva un frame ogni 10 step
        # frames_to_use = range(0, num_samples, skip)

        # ani = animation.FuncAnimation(
        #     fig, update, frames=frames_to_use, init_func=init,
        #     blit=True, interval=50, repeat=False
        # )

        # anim_path = "images/mpc/com_xy_trajectory.mp4"
        # ani.save(anim_path, writer='ffmpeg', dpi=200)
        # plt.close(fig)


        # anim_path = "images/mpc/com_trajectory_3d.mp4"

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Calcolo limiti basati su entrambe le traiettorie
        # all_x = np.concatenate([com_trajectory[:, 0], real_com_trajectory[:, 0]])
        # all_y = np.concatenate([com_trajectory[:, 1], real_com_trajectory[:, 1]])
        # all_z = np.concatenate([com_trajectory[:, 2], real_com_trajectory[:, 2]])

        # ax.set_xlim(np.min(all_x), np.max(all_x))
        # ax.set_ylim(np.min(all_y), np.max(all_y))
        # ax.set_zlim(np.min(all_z), np.max(all_z))

        # ax.set_xlabel('X [m]')
        # ax.set_ylabel('Y [m]')
        # ax.set_zlabel('Z [m]')
        # ax.set_title('3D CoM Trajectories')

        # # Linee e punti animati
        # simulation_line, = ax.plot([], [], [], lw=2, color='blue', label='CoM (Simulation)')
        # simulation_point, = ax.plot([], [], [], 'o', color='blue')

        # real_line, = ax.plot([], [], [], lw=2, color='green', linestyle='--', label='CoM (Real)')
        # real_point, = ax.plot([], [], [], 'o', color='green')

        # ax.legend()

        # def init():
        #     simulation_line.set_data([], [])
        #     simulation_line.set_3d_properties([])
        #     simulation_point.set_data([], [])
        #     simulation_point.set_3d_properties([])
        #     real_line.set_data([], [])
        #     real_line.set_3d_properties([])
        #     real_point.set_data([], [])
        #     real_point.set_3d_properties([])
        #     return simulation_line, simulation_point, real_line, real_point

        # def update(frame):
        #     # CoM simulato
        #     simulation_line.set_data(com_trajectory[:frame, 0], com_trajectory[:frame, 1])
        #     simulation_line.set_3d_properties(com_trajectory[:frame, 2])
        #     simulation_point.set_data([com_trajectory[frame, 0]], [com_trajectory[frame, 1]])
        #     simulation_point.set_3d_properties([com_trajectory[frame, 2]])

        #     # CoM reale
        #     real_line.set_data(real_com_trajectory[:frame, 0], real_com_trajectory[:frame, 1])
        #     real_line.set_3d_properties(real_com_trajectory[:frame, 2])
        #     real_point.set_data([real_com_trajectory[frame, 0]], [real_com_trajectory[frame, 1]])
        #     real_point.set_3d_properties([real_com_trajectory[frame, 2]])

        #     return simulation_line, simulation_point, real_line, real_point

        # ani = animation.FuncAnimation(
        #     fig, update, frames=frames_to_use,
        #     init_func=init, blit=True, interval=50, repeat=False
        # )

        # ani.save(anim_path, writer='ffmpeg', fps=25, dpi=150)
        # plt.close(fig)
    else:
        # Animazione traiettoria CoM sul piano XY
        fig, ax = plt.subplots()
        ax.set_xlim(np.min(com_trajectory[:, 0]) - 0.1, np.max(com_trajectory[:, 0]) + 0.1)
        ax.set_ylim(np.min(com_trajectory[:, 1]) - 0.1, np.max(com_trajectory[:, 1]) + 0.1)
        ax.set_xlabel('CoM X [m]')
        ax.set_ylabel('CoM Y [m]')
        ax.set_title('CoM Trajectory in XY Plane')
        ax.grid(True)
        simulation_line, = ax.plot([], [], 'b-', label='Simulated CoM Trajectory')
        simulation_point, = ax.plot([], [], 'ro')  # Punto attuale
        mpc_line, = ax.plot([], [], 'r-', label='mpc CoM Trajectory')
        mpc_point, = ax.plot([], [], 'ro')  # Punto attuale
        mpc_predictions_line, = ax.plot([], [], 'g-', label='mpc Predictions Trajectory')
        zmp_predictions_line, = ax.plot([], [], 'y-', label='ZMP Predictions Trajectory')
        ax.legend()

        def init():
            simulation_line.set_data([], [])
            simulation_point.set_data([], [])
            mpc_line.set_data([], [])
            mpc_point.set_data([], [])
            mpc_predictions_line.set_data([], [])
            zmp_predictions_line.set_data([], [])
            return simulation_line, simulation_point, mpc_line, mpc_point, mpc_predictions_line, zmp_predictions_line

        # fuori dalla funzione update
        last_prediction_frame = [-1]  # utilizzo una lista per mutabilità

        def update(frame):
            # Simulazione
            simulation_line.set_data(com_trajectory[:frame, 0], com_trajectory[:frame, 1])
            simulation_point.set_data([com_trajectory[frame, 0]], [com_trajectory[frame, 1]])

            # MPC reale
            mpc_line.set_data(mpc_com_trajectory[:frame, 0], mpc_com_trajectory[:frame, 1])
            mpc_point.set_data([mpc_com_trajectory[frame, 0]], [mpc_com_trajectory[frame, 1]])

            # Reset predizione visiva
            mpc_predictions_line.set_data([], [])
            zmp_predictions_line.set_data([], [])

            # Estrai predizioni per il frame corrente: righe da 20*frame a 20*(frame+1)
            start_idx = frame * 20
            end_idx = start_idx + 20
            if end_idx <= mpc_predictions_trajectory.shape[0]:
                prediction = mpc_predictions_trajectory[start_idx:end_idx, :]  # shape (20, 9)
                com_pred_x = prediction[:, 0]
                com_pred_y = prediction[:, 1]
                zmp_pred_x = prediction[:, 6]
                zmp_pred_y = prediction[:, 7]
                mpc_predictions_line.set_data(com_pred_x, com_pred_y)
                zmp_predictions_line.set_data(zmp_pred_x, zmp_pred_y)

            return simulation_line, simulation_point, mpc_line, mpc_point, mpc_predictions_line, zmp_predictions_line



        skip = 100  # salva un frame ogni 10 step
        frames_to_use = range(0, num_samples, skip)

        ani = animation.FuncAnimation(
            fig, update, frames=frames_to_use, init_func=init,
            blit=True, interval=50, repeat=False
        )

        anim_path = "images/mpc/com_xy_trajectory.mp4"
        ani.save(anim_path, writer='ffmpeg', dpi=200)
        plt.close(fig)



