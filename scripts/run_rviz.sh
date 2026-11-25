#!/usr/bin/env bash
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
case "${1:-}" in
  noetic)
    shift
    source /opt/ros/noetic/setup.bash
    export ROS_PACKAGE_PATH="$root/rviz_observer:${ROS_PACKAGE_PATH}"
    exec roslaunch "$root/rviz_observer/observer.launch" "$@" ;;
  foxy|sim) shift ;;
  robot) shift; set -- source:=robot "$@" ;;
esac
if [[ ${ROS_DISTRO:-} != foxy ]]; then
  source /opt/ros/foxy/setup.bash
fi
# Discover the RViz panel in the project overlay when launching directly from source.
overlay="$root/build/ros2/install"
plugin_index=share/ament_index/resource_index/rviz_common__pluginlib__plugin/g1_wbc_observer
if [[ -f "$overlay/g1_wbc_observer/$plugin_index" ]]; then
  source "$overlay/local_setup.bash"
elif [[ " $* " != *" gui:=false "* ]]; then
  prefix=$(ros2 pkg prefix g1_wbc_observer 2>/dev/null || true)
  if [[ -z "$prefix" || ! -f "$prefix/$plugin_index" ]]; then
    colcon --log-base "$root/build/ros2/log" build --base-paths "$root/rviz_observer" \
      --build-base "$root/build/ros2/build" --install-base "$overlay" --symlink-install
    source "$overlay/local_setup.bash"
  fi
fi
exec ros2 launch "$root/rviz_observer/observer_foxy.launch.py" "$@"
