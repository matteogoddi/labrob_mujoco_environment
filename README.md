### INTRODUCTION
The goal of the following code was to perform closed loop experiments on the robot UnitreeG1 (code can be readapted to account for multiple robots with some parameters tuning). The robot was tasked to walk forward for an arbitrary number of steps. 

The control scheme is composed by:
1. Offline foostep planner
2. Extended Kalman Filter for state estimation
3. IS-MPC for generation of desired Center of Mass (CoM) trajectory
4. Whole Body Controller for reference tracking (like CoM, swing foot, ecc...) and enforcing constraints (contact force, kinematics, ecc...)

Main references for this work are:
1. Scianca, N., De Simone, D., Lanari, L., & Oriolo, G. (2020). MPC for humanoid gait generation: Stability and feasibility. IEEE Transactions on Robotics, 36(4), 1171-1188.
2. Marussi, D., Cipriano, M., Scianca, N., Lanari, L., & Oriolo, G. (2025). Humanoid Motion Generation in Complex 3D Environments. Robotics, 14(6), 82.

Possible contacts for techincal help:
1. support@unitree.cc (tel +86 15776583869)
2. Sara Alimonti from EagleProjects.it 

### PREREQUISITES 
You need the following dependencies:
- glfw3
- mujoco
- pinocchio
- hpipm
- blasfeo

1. GLFW3
   ```
   sudo apt install libglfw3
   ```

2. PINOCCHIO
   ```
   git clone --recursive https://github.com/stack-of-tasks/pinocchio.git
   cd pinocchio
   mkdir build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   sudo make install
   ```

3. MUJOCO
   To install Mujoco, download the source code from: https://github.com/google-deepmind/mujoco/releases

   Then, install Mujoco using:
   ```
   mkdir build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j(nproc)
   sudo make install
   ```

4. HPIPM & BLASFEO
   To install HPIPM (together with Blasfeo), follow the instructions under the section "C" at: https://github.com/giaf/hpipm

### Installation to run python scripts

0. install (once) and activate virtual environment (must be activated everytime)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

1. install dependencies listed in `requirements.txt` (once)
   ```bash
   pip install -r requirements.txt
   ```

### How to run the code

1. run python scripts:
   ```bash
   python3 scripts/'file_name'
   ```
2. execute main in simulation mode:
   ```bash
   cd build
   make -j$(nproc)
   ./main --sim
   ```

### Visualization 

1. Install vscode extension "URDF Visualizer"

2. Open "unitreeg1.urdf" and press ```CTRL+SHIFT*P```, then select 'URDF Visualizer: Preview URDF/Xacro'


### Laboratory simulation with UNITREE G1

In order to communicate with the robot and perform a laboratory simulation with the unitree G1 robot, the official SDK from the git repo "unitreerobotics" was used.
An alternative is to install ROS2, as shown in the website. However, UnitreeSDK is way simpler. Nevertheless ROS2 may be installed to check the channels were the robot is publishing.
Note that UnitreeSDK subscripts and publishes over the same channels (lowstate, etc...).
Once done we need to establish a physical connection between the machine and the G1, which may be via Ethernet or Wi-fi.
To clear further doubts the documentation is found in the following site (https://support.unitree.com/home/en/G1_developer/about_G1).
The user manual can be found in (https://reliablerobotics.ai/wp-content/uploads/2025/03/G1-User-Manual_compressed.pdf.)

0. Unitree Explorer App:

1. Install Unitree SDK:
   ```bash
   git clone https://github.com/unitreerobotics/unitree_sdk2.git
   cd unitree_sdk2
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   sudo make install
   ```

   To install examples
   ```bash
   mkdir build && cd build
   cmake .. 
   make 
   ```

   NOTE:
   To run the example "g1_dual_arm_example", yaml-cpp is needed, however its latest version (0.8.0) may cause issue with Ubuntu 24.04.2 LTS, causing segmentation fault when trying to execute every example. For this reason it is recommended to use the version 0.7.0.

2. Net configuration:
   1. Ethernet connection
      The robot's onboard computer IP address is 192.168.123.161, so set the computer's Ethernet address to the same subnet, such as 192.168.123.222.

      ADDRESS : 192.168.123.222
      NETWORK : 255.255.255.0
      GATEWAY : 192.168.123.161

      To check if the robot is receiving you can ping 
      ```bash
      ping 192.168.123.161
      ```
      To find the name of your personal network interface, for either wi-fi (wl) or ethernet (en):
      ```bash
      ifconfig
      ```
   2. WiFi


3. Run simulation:
   0. Run examples
      ```bash
      ./'example_filename' 'network_interface'
      ```
   1. Execute main:
      ```bash
      ./main --robot 'network_interface'
      ```

### TO DO LIST
0. Make the robot walk.

1. Implement footstep correction within IS-MPC framework.

2. Add arm swing task.

3. Online and offline footstep planner both add element on the list while walking with offline footstep planner (check if true). Online footstep planner should be toggle or find better solution.

4. FIXED. Online and offline footstep planner seem to start with different foot, logic should work regardless, modify IS-MPC's mapping of mc's center to adapt.

5. HAC should be turned off if not necessary.

6. Change name of estimator to better capture its nature as a generic estimator and not wrist-focused only.

7. Button for observer missing, add if necessary (decide which one aribtrarily besides of the one already used) or leave it active at all time. If a new button is added, at the start of the experiment should be added as printed in terminal.

8. plot_joint_data.py needs now no definition of new plots, there's an inside function handling plots for 2 cases, an image with 3 plots (mainly used for comparisons) or image with 1 plot (mainly used for errors).

9. However it may make sense to decompose plot_joint_data in multiple files, based on the quantity needed (or use variables to toggle not needed plot).

9. Ideally walking manager should be light, initialization of pointers (see HAC init) should be handled by the pointer definition inside relative .cpp.

10. FIXED. main.cpp may need a function to handle forces (something like addForce(link_subject_to_force, force_vector, torque_vector, start_time))

11. FIXED. No file should write on .txt directly, it may add spikes in computation time (see main.cpp).

12. FIXED. Find a working filter and remove StateFiltering file which is redundant once at least a filter is working.

13. FIXED. It may make sense to create 2 executables, one for simulation and one for experiments.

14. It may make sense to create 2 different set of parameters for the whole body controller for simulation and experiments.

15. To decide together: use one common version for hpipm, blasfeo, mujoco and pinocchio. Then fix CMakeLists.txt to work for everyone.
