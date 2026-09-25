#!/usr/bin/env bash
# Run from anywhere: bash scripts/setup_ubuntu.sh
set -eo pipefail
FR3_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f /opt/ros/lyrical/setup.bash ]]; then
  echo 'Please install ROS 2 Lyrical on Ubuntu 26.04 first (/opt/ros/lyrical).' >&2
  exit 1
fi
source /opt/ros/lyrical/setup.bash
packages=(
  python3-venv python3-pip python3-colcon-common-extensions
  python3-numpy python3-opencv python3-yaml python3-pytest
  libgl1-mesa-dri libegl1 libglfw3
  ros-lyrical-moveit ros-lyrical-xacro ros-lyrical-robot-state-publisher
  ros-lyrical-rviz2 ros-lyrical-cv-bridge ros-lyrical-message-filters
  ros-lyrical-tf2-geometry-msgs ros-lyrical-vision-msgs ros-lyrical-control-msgs
  ros-lyrical-rosgraph-msgs ros-lyrical-rqt-image-view
)
missing=()
for package in "${packages[@]}"; do
  if [[ "$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null)" != 'install ok installed' ]]; then
    missing+=("$package")
  fi
done
if ((${#missing[@]})); then
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
fi
if [[ ! -x "$FR3_ROOT/.venv-fr3/bin/python" ]]; then
  /usr/bin/python3 -m venv --system-site-packages "$FR3_ROOT/.venv-fr3"
fi
source "$FR3_ROOT/.venv-fr3/bin/activate"
# Keep Ubuntu's NumPy/OpenCV/CV bridge ABI together via --system-site-packages.
python -m pip install 'mujoco==3.13.0'
bash "$FR3_ROOT/scripts/build.sh"
echo 'Ready. Run: bash scripts/run_demo.sh'
