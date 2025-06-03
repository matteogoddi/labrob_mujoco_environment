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

### Connect to github

0. clone the repository
    ```bash
    git clone 'repository_link'
    ```

1. pull changes
   ```bash
    git pull 'remote_name' main
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

### Visualization 

1. Install vscode extension "URDF Visualizer"

2. Open "unitreeg1.urdf" and press ```CTRL+SHIFT*P```, then press 'URDF Visualizer: Preview URDF/Xacro'

