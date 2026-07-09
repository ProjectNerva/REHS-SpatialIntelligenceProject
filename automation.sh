#!/bin/bash

# Exit immediately if any command fails
set -e

IMAGE_NAME="sip_orb_slam3_img"
CONTAINER_NAME="SIP_ORB_SLAM3_CONTAINER"

echo "========================================="
echo "       Building the Docker image...      "
echo "========================================="
docker build -t $IMAGE_NAME .

echo "========================================="
echo "     Cleaning up any old container...    "
echo "========================================="
# Remove old container if it exists, suppressing errors if it doesn't
docker rm -f $CONTAINER_NAME 2>/dev/null || true

echo "========================================="
echo "     Entering the container...           "
echo "========================================="
# Run the container with an interactive TTY, sharing shared_data with the host.
docker run -it -v "$(pwd)/shared_data:/ORB_SLAM3/shared_data" --name $CONTAINER_NAME $IMAGE_NAME