# Bash reads this file when entering the experiment with g1-env.
if [[ -r ~/.bashrc ]]; then source ~/.bashrc; fi
if [[ ${G1_DEV:-0} == 1 ]]; then
  build() {
    /workspace/g1/docker/dev build && source /workspace/g1/build/ros2/install/local_setup.bash
  }
  if [[ -r /workspace/g1/build/ros2/install/local_setup.bash ]]; then
    source /workspace/g1/build/ros2/install/local_setup.bash
  fi
fi
PS1="(g1-${G1_SOURCE:-sim}) \w \$ "
