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
   make -j4
   sudo make install
   ```

3. MUJOCO
   To install Mujoco, download the source code from: https://github.com/google-deepmind/mujoco/releases

   Then, install Mujoco using:
   ```
   mkdir build
   cd build
   cmake ..
   make -j(nproc)
   sudo make install
   ```

4. HPIPM & BLASFEO
   To install HPIPM (together with Blasfeo), follow the instructions under the section "C" at: https://github.com/giaf/hpipm

### Connect to github

0. clone the repository
   ```bash
   git clone 'repository_link'
   ```
   
1. to create another branch different from 'main'
   ```bash
   git checkout -b 'new_branch_name'
   ```
   and to switch from one branch to another
   ```bash
   git checkout 'branch_name'
   ```
2. pull changes
   ```bash
   git pull 'remote_name' main
   // to reset the project and copy all 
   git fetch 'remote_name' && git reset --hard 'remote_name'/'branch_name'
   ```
   
3. push changes
   ```bash
   git add -A
   git commit -m "Explain changes"
   git push 'remote_name' 'branch_name'
   ```

4. usually 'remote_name' is origin, to add another remote for personal changes
   ```bash
   git remote add 'new_remote_name' 'new_repository_link'
   git remote -v
   ```

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

In order to communicate with the robot and perform a laboratory simulation with the unitree G1 robot it is essential to download the official SDK from the git repo "unitreerobotics".
An alternative is to install ROS2, as shown in the website. However, UnitreeSDK is way simpler. Nevertheless ROS2 may be installed to check the channels were the robot is publishing.
Note that UnitreeSDK subscripts and publishes over the same channels (sportmodestate, ecc...).
Once done we need to establish a physical connection between the machine and the G1, which may be via Ethernet or Wi-fi. //TODO find user and password to connect via Wi-fi.
To clear further doubts the documentation is found in the following site (https://support.unitree.com/home/en/G1_developer/about_G1).
The user manual can be found in (https://reliablerobotics.ai/wp-content/uploads/2025/03/G1-User-Manual_compressed.pdf.)

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
1. Closed loop on CoM doesn't work (robot falls forward). Possible causes:
   1. IMU orientation bias, implement a calibration routine to be performed once.

2. Reduce computational burden
   1. Variables are duplicated for simulation and feedback, is it necessary?
   2. Storing data in more efficient way (maybe save a value every 2) and in different forms (maybe arrays cause spikes)

3. Clean code.

4. Improve plots.

5. Implement footstep correction within IS-MPC framework.

6. Fix mujoco model (thumbs swinging for some reasons).

7. Add arm swing task.

