#!/usr/bin/env bash

# Usage: ./scripts/build.sh [DOCKER_CI_DIR]

set -e
set -o pipefail

DOCKER_CI_DIR=$1

sudo apt update

sudo apt install -y libyaml-cpp-dev \
	gcc-10 g++-10 python3-opencv \
	ros-humble-image-transport ros-humble-cv-bridge \
	ros-humble-sensor-msgs ros-humble-cv-bridge \
	ros-humble-std-srvs ros-humble-std-msgs \
	ros-humble-rclcpp

$DOCKER_CI_DIR/scripts/install_cuda.sh

source /opt/ros/humble/setup.bash

$DOCKER_CI_DIR/scripts/build.sh sonia_common_ros2

cd proc_vision_ros2

source /build/sonia_common_ros2/INSTALL_BASE/setup.sh

export CC=gcc-10
export CXX=g++-10
export CUDACXX=$(find / -name nvcc)
export CUDAHOSTCXX=g++-10

colcon build --cmake-force-configure --install INSTALL_BASE
