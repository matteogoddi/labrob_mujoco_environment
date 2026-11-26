FROM ros:foxy-ros-base-focal@sha256:34fbe32f5b1cd3eb25f6becaaa1d293bad754d64c31d96e638e4b30ed82f9f8e AS base
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
RUN test "$(dpkg --print-architecture)" = amd64 && apt-get -o Acquire::Retries=3 update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential curl git ca-certificates pkg-config python3-pip python3-colcon-common-extensions \
      libboost-filesystem-dev libboost-serialization-dev libboost-system-dev liburdfdom-dev \
      libjsoncpp-dev libglfw3-dev libgl1-mesa-dev libgl1-mesa-dri libglx-mesa0 libssl-dev bison \
      qtbase5-dev libqwt-qt5-dev libqt5svg5-dev libqt5opengl5-dev python3-pykdl \
      ros-foxy-rviz2 ros-foxy-robot-state-publisher ros-foxy-rmw-cyclonedds-cpp \
      ros-foxy-rosidl-generator-dds-idl fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir cmake==3.24.3

FROM base AS native
ARG BUILD_JOBS=2
WORKDIR /tmp/deps
RUN curl -fLsS --retry 3 https://github.com/google-deepmind/mujoco/releases/download/3.4.0/mujoco-3.4.0-linux-x86_64.tar.gz -o mujoco.tgz && \
    echo '3b0e47d8880da0005b61fe5e43a4d058eb5097e6e3ba44f4a23ce2b5fab2244d  mujoco.tgz' | sha256sum -c && \
    mkdir -p /opt/mujoco && tar xzf mujoco.tgz -C /opt/mujoco --strip-components=1 mujoco-3.4.0/include mujoco-3.4.0/lib
COPY docker/mujocoConfig.cmake /opt/mujoco/lib/cmake/mujoco/mujocoConfig.cmake
RUN git clone -q --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git eigen && \
    cmake -S eigen -B eigen-build -DBUILD_TESTING=OFF && cmake --install eigen-build
RUN git init -q pin && git -C pin fetch --depth 1 https://github.com/stack-of-tasks/pinocchio.git b6f5f163e4bb2f0f378d27c1d82951f30f7150ed && \
    git -C pin checkout -q FETCH_HEAD && git -C pin -c protocol.version=2 submodule update --init --depth 1 cmake && \
    cmake -S pin -B pin-build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=OFF \
      -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_WITH_COLLISION_SUPPORT=OFF && \
    cmake --build pin-build --parallel "${BUILD_JOBS}" && cmake --install pin-build
RUN git init -q sdk && git -C sdk fetch --depth 1 https://github.com/unitreerobotics/unitree_sdk2.git f29ee9f234851e9e79f75102c0f9e83008d8fdd1 && \
    git -C sdk checkout -q FETCH_HEAD && cmake -S sdk -B sdk-build -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_LIBDIR=lib && cmake --install sdk-build

FROM base AS observer
WORKDIR /tmp/unitree
ARG BUILD_JOBS=2
RUN git init -q && git fetch --depth 1 https://github.com/unitreerobotics/unitree_ros2.git 3ff13ea08ec619496c2651fd21b172f7958dd5a5 && git checkout -q FETCH_HEAD
WORKDIR /tmp/unitree/cyclonedds_ws/src
RUN git init -q cyclonedds && git -C cyclonedds fetch --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds.git 76360fb73907ce3dba397e89090a7a4ecf4f1246 && git -C cyclonedds checkout -q FETCH_HEAD && \
    git init -q rmw_cyclonedds && git -C rmw_cyclonedds fetch --depth 1 https://github.com/ros2/rmw_cyclonedds.git c12abc56983204f1d91f2d839d394528c7b29b42 && git -C rmw_cyclonedds checkout -q FETCH_HEAD
WORKDIR /tmp/unitree/cyclonedds_ws
RUN export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" && \
    colcon build --merge-install --install-base /opt/unitree-ros --packages-select cyclonedds --cmake-args -DBUILD_TESTING=OFF -DENABLE_SHM=OFF && \
    source /opt/ros/foxy/setup.bash && source /opt/unitree-ros/local_setup.bash && \
    colcon build --merge-install --install-base /opt/unitree-ros --executor sequential \
      --packages-select rmw_cyclonedds_cpp unitree_go unitree_hg unitree_api --cmake-args -DBUILD_TESTING=OFF
WORKDIR /opt/g1
COPY rviz_observer rviz_observer
COPY g1_description g1_description
RUN source /opt/ros/foxy/setup.bash && source /opt/unitree-ros/local_setup.bash && \
    CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" colcon build --base-paths rviz_observer --merge-install --install-base /opt/g1-observer --cmake-args -DBUILD_TESTING=OFF

FROM native AS controller
WORKDIR /opt/g1
COPY CMakeLists.txt *.cpp *.hpp ./
COPY include include
COPY src src
COPY blasfeo blasfeo
COPY hpipm hpipm
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/mujoco -DTARGET=GENERIC && \
    cmake --build build --target main lip_simulator_main --parallel "${BUILD_JOBS}"

# Keep the portable binary for older x86-64 hosts. The optimized BLASFEO
# kernels substantially reduce the time spent in the 500 Hz walking solve.
RUN cmake -S . -B build-avx2 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/mujoco -DTARGET=X64_INTEL_HASWELL -DBLASFEO_CROSSCOMPILING=ON && \
    cmake --build build-avx2 --target main --parallel "${BUILD_JOBS}"

FROM base AS runtime
COPY --from=native /usr/local/ /usr/local/
COPY --from=native /opt/mujoco /opt/mujoco
COPY --from=observer /opt/unitree-ros /opt/unitree-ros
COPY --from=observer /opt/g1-observer /opt/g1-observer
WORKDIR /opt/g1
RUN apt-get -o Acquire::Retries=3 update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3-matplotlib python3-seaborn python3-scipy python3-imageio && rm -rf /var/lib/apt/lists/*
COPY . .
COPY --from=controller /opt/g1/build/main /opt/g1/build/lip_simulator_main build/
COPY --from=controller /opt/g1/build-avx2/main build/main-avx2
COPY docker/entrypoint /usr/local/bin/g1-env
RUN chmod +x /usr/local/bin/g1-env && /usr/local/bin/g1-env check
ENTRYPOINT ["/usr/local/bin/g1-env"]
CMD ["bash"]
