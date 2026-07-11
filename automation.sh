#!/bin/bash

# Exit immediately if any command fails
set -e

IMAGE_NAME="sip_orb_slam3_img"
CONTAINER_NAME="SIP_ORB_SLAM3_CONTAINER"

echo "========================================="
echo "       Building the Docker image...      "
echo "========================================="
if docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo "Docker image $IMAGE_NAME already exists. Skipping build."
else
    echo "Docker image $IMAGE_NAME does not exist. Building now..."
    docker build -t $IMAGE_NAME .
fi

echo "========================================="
echo "     Cleaning up any old container...    "
echo "========================================="
# Remove old container if it exists, suppressing errors if it doesn't
docker rm -f $CONTAINER_NAME 2>/dev/null || true

echo "========================================="
echo "     Setting up XQuartz display...       "
echo "========================================="
# Point this shell at the running XQuartz server
export DISPLAY=:0
# Authorize the Docker VM to connect to XQuartz
xhost + 127.0.0.1

echo "========================================="
echo "     Entering the container...           "
echo "========================================="
# Run the container with an interactive TTY, sharing shared_data with the host.
docker run -it -e DISPLAY=host.docker.internal:0 -v "$(pwd)/shared_data:/shared_data" --name $CONTAINER_NAME $IMAGE_NAME
