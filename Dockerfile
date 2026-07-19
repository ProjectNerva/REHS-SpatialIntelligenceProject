# Base: Debian Trixie -- GCC 14 + CMake 3.31 + OpenCV 4.10, comfortably above the
# ORB-SLAM3 fork's C++23 build needs (GCC 11+, OpenCV 4.6+) and depthai-core's
# CMake 3.20 floor. (Ubuntu 20.04's GCC 9/CMake 3.16 can't enable C++23 at all --
# g2o's configure fails with "CMake does not know the compile flags to use".)
FROM debian:trixie

ENV DEBIAN_FRONTEND=noninteractive
# Force Mesa software rendering (llvmpipe): there is no GPU in the container, so
# the Pangolin viewer must render on the CPU.
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV DISPLAY=:1

# --- Toolchain + ORB-SLAM3 / Pangolin build dependencies ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    pkg-config \
    curl \
    unzip \
    libeigen3-dev \
    libopencv-dev \
    libboost-all-dev \
    libssl-dev \
    libglew-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgl1-mesa-dri \
    libx11-dev \
    libxkbcommon-dev \
    mesa-utils \
    libudev-dev \
    # --- headless-viewer-over-browser (VNC) stack ---
    xvfb \
    x11vnc \
    fluxbox \
    novnc \
    websockify \
    && rm -rf /var/lib/apt/lists/*

# --- Pangolin ----------------------------------------------------------------
# v0.9.1: builds cleanly on GCC 13 (v0.6, used by the old Dockerfile, does not).
RUN git clone --branch v0.9.1 --depth 1 https://github.com/stevenlovegrove/Pangolin.git /tmp/Pangolin \
    && cd /tmp/Pangolin \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build \
    && ldconfig \
    && rm -rf /tmp/Pangolin

# --- ORB-SLAM3 (the exact fork this project targets) -------------------------
COPY patches/ /tmp/patches/
RUN git clone --depth 1 https://github.com/ProjectNerva/ORB_SLAM3 /ORB_SLAM3 \
    && cd /ORB_SLAM3 \
    # System::Shutdown() requests LocalMapping/LoopClosing to finish but the wait-until-actually-
    # finished loop is commented out upstream, so Shutdown() returns while those threads are still
    # mid-culling/mid-BA -- racing with whatever the caller does next (reading the map, exiting the
    # process). Un-comment it; see patches/orb_slam3_shutdown_wait.patch for the full rationale.
    && git apply /tmp/patches/orb_slam3_shutdown_wait.patch \
    && chmod +x build.sh \
    && ./build.sh

RUN mkdir -p /ORB_SLAM3/shared_data

# --- Browser-VNC entrypoint --------------------------------------------------
# Starts a virtual X display, a window manager, a VNC server, and noVNC's
# websocket bridge, then execs the container command. Open the SLAM viewer at
# http://localhost:8080/vnc.html or http://127.0.0.1:8080/vnc.html on the host.
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -e' \
    'export DISPLAY=:1' \
    'rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 2>/dev/null || true' \
    'Xvfb :1 -screen 0 1600x900x24 -ac +extension GLX +render -noreset >/var/log/xvfb.log 2>&1 &' \
    'sleep 2' \
    'fluxbox >/var/log/fluxbox.log 2>&1 &' \
    'x11vnc -display :1 -forever -shared -nopw -rfbport 5900 -bg -o /var/log/x11vnc.log' \
    'websockify -D --web=/usr/share/novnc 8080 localhost:5900' \
    'echo "=================================================================="' \
    'echo " noVNC viewer ready:  http://localhost:8080/vnc.html  (no password)"' \
    'echo "=================================================================="' \
    'exec "$@"' \
    > /usr/local/bin/start-vnc.sh \
    && chmod +x /usr/local/bin/start-vnc.sh

WORKDIR /users/dummyuser/workspace
ENTRYPOINT ["/usr/local/bin/start-vnc.sh"]
CMD ["/bin/bash"]
