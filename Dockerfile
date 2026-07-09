# Use the official Ubuntu 22.04 (Jammy Jellyfish) base image
FROM ubuntu:22.04

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

# making the Pangolin library - visualization
RUN git clone --recursive https://github.com/stevenlovegrove/Pangolin.git \
    cd Pangolin \
    mkdir build && cd build \
    cmake .. \
    make -j$(nproc) \
    && sudo make install


RUN cd ~ \
    git clone https://github.com/devansh0703/ORB_SLAM3.git ORB_SLAM3 \
    cd ORB_SLAM3 \
    chmod +x build.sh \
    && ./build.sh

# Set the default command to run when the container starts
CMD ["/bin/bash"]