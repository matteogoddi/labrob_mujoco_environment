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
make -j
sudo make install
```

To install HPIPM (together with Blasfeo), follow the instructions under the section "C" at: https://github.com/giaf/hpipm
