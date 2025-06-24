You need the following dependencies:
- glfw3
- mujoco
- pinocchio
- hpipm
- blasfeo

To install glfw3, use:
```
sudo apt install glfw3
```

To install pinocchio, follow the instructions at: https://stack-of-tasks.github.io/pinocchio/download.html

To install Mujoco, download the source code from: https://github.com/google-deepmind/mujoco/releases

Then, install Mujoco using:
```
mkdir build
cd build
cmake ..
make -j(nproc)
sudo make install
```

To install HPIPM (together with Blasfeo), follow the instructions under the section "C" at: https://github.com/giaf/hpipm

### Connect to github

0. clone the repository
    ```bash
   git clone 'repository_link'
    ```

1. pull changes
   ```bash
   git pull 'remote_name' main
   // to reset the project and copy all 
   git fetch 'remote_name' && git reset --hard 'remote_name'/main
   ```
   
2. push changes
   ```bash
   git add .
   git commit -m "Explain changes"
   git push 'remote_name' main
   ```

4. usually 'remote_name' is origin, to add another remote for personal changes
   ```bash
   git remote add 'new_remote_name' 'new_repository_link'
   git remote -v
   ```
   
### Prerequisites 

1. Dependencies listed in `requirements.txt`, needed when trying to run files in /scripts

### Installation

0. install and activate virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

1. install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
### How to run the code
1. run python scripts:
   ```bash
   python3 scripts/'file_name'
   ```
2. execute main:
   ```bash
   cd build
   make -j('nproc')
   ./main
   ```

### Visualization 

1. Install vscode extension "URDF Visualizer"

2. Open "unitreeg1.urdf" and press ```CTRL+SHIFT*P```, then press 'URDF Visualizer: Preview URDF/Xacro'


### Laboratory simulation with UNITREE G1

In order to perform a laboratory simulation with the unitree G1 robot it is essential to download the 
official SDK from the git repo "unitreerobotics" suited for the G1 humanoid. (Generally Unitree SDK2 is 
based on CycloneDDS, which will be installed in the requirements).
Once done we need to establish a connection between the machine and the G1, which may be via Ethernet
or Wi-fi.
The set-up is ready and the experiment may be perfomed (make sure to have a main suited for the task).
To clear further doubts the documentation is found in the following site (https://support.unitree.com/home/en/G1_developer/about_G1).

0. Install requirements:
   ```bash
   sudo apt update
   sudo apt install build-essential cmake git

   git clone https://github.com/eclipse-cyclonedds/cyclonedds.git
   cd cyclonedds
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   sudo make install
   echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

1. Install Unitree SDK:
   ```bash
   git clone https://github.com/unitreerobotics/unitree_sdk2.git
   cd unitree_sdk2
   mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics
   make -j$(nproc)
   sudo make install
   ```

   To install examples
   ```bash
   mkdir build && cd build
   cmake .. 
   make 
   ```

2. Net configuration:
   1. Ethernet connection
      The robot's onboard computer IP address is 192.168.123.161, so set the computer's USB Ethernet address to the same subnet, such as 192.168.123.222.
      ADDRESS : 192.168.123.161
      NETWORK : 255.255.255.0
      GATEWAY : 192.168.123.222
      To check if the robot is receiving you can ping 
      ```bash
      ping 192.168.123.161
      ```
      If any network interface is needed:
      ```bash
      ifconfig
      ```
      output: list of netwiork interfaces: generally en for ethernet, wl for wifi, connect to the needed one.


3. Run simulation:
   0. Run examples
      To run examples, inside the folder /build/bin, the executables are found, which can be run by:
       ```bash
      ./'example_filename' 'network_interface'
      ```

