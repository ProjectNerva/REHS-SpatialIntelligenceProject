FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Run package updates and install your required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    gfortran \
    curl \
    libgtk2.0-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libeigen3-dev \
    libopencv-dev \
    libglew-dev \
    libboost-all-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Build Pangolin (using v0.6 which is highly stable for ORB-SLAM3)
RUN git clone https://github.com/stevenlovegrove/Pangolin.git /tmp/Pangolin \
    && cd /tmp/Pangolin \
    && git checkout v0.6 \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && rm -rf /tmp/Pangolin

# Build ORB-SLAM3
RUN git clone https://github.com/devansh0703/ORB_SLAM3.git /ORB_SLAM3 \
    && cd /ORB_SLAM3 \
    && chmod +x build.sh \
    && ./build.sh

WORKDIR /ORB_SLAM3

# Mount point for data shared between the host and the container
RUN mkdir -p /ORB_SLAM3/shared_data

# Set the default command to run when the container starts
CMD ["/bin/bash"]