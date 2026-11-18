# Headless simulation image: the MuJoCo Baxter, the teleop servo following a policy's targets, a
# RethoughtAI policy server and the closed-loop evaluator, on ROS 2 Kilted with Zenoh. Simulation only:
# no robot bridge, cameras or VR. The image holds dependencies; the code and its venvs come from the
# FinalProject checkout mounted at the same path (docker/sim/README.md).
FROM ros:kilted-ros-base

ENV DEBIAN_FRONTEND=noninteractive
# The teleop's ROS dependencies as rosdep resolves them (src/baxter_teleop/package.xml and those it
# depends on), the messages the policy server and evaluator use, bag storage, and EGL for MuJoCo.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-kilted-rmw-zenoh-cpp \
        ros-kilted-pinocchio \
        ros-kilted-xacro \
        ros-kilted-robot-state-publisher \
        ros-kilted-tf2-ros \
        ros-kilted-control-msgs \
        ros-kilted-moveit-msgs \
        ros-kilted-rosbag2 \
        ros-kilted-rosbag2-storage-mcap \
        ros-kilted-rosbag2-compression-zstd \
        python3-venv python3-pip python3-scipy python3-yaml python3-matplotlib python3-colcon-common-extensions \
        libegl1 libgl1 libglvnd0 libgles2 \
        ffmpeg git curl tmux \
    && rm -rf /var/lib/apt/lists/*

# uv for the baxter_mujoco and RethoughtAI venvs
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -LsSf https://astral.sh/uv/0.9.28/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# EGL through the NVIDIA container runtime (MuJoCo renders the cameras offscreen)
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl \
    RMW_IMPLEMENTATION=rmw_zenoh_cpp
