#include <iostream>
#include <fstream>
#include <thread>
#include <csignal>
#include <atomic>

#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

// ORB_SLAM3 headers
#include "System.h"
#include "ImuTypes.h"
#include "MapPoint.h"

// Global atomic flag to control the main loop execution
std::atomic<bool> bContinueRunning(true);

// Signal handler function for Ctrl+C (SIGINT) -- lets the loop reach pipeline.stop() and
// SLAM.Shutdown() on exit instead of an abrupt kill, which is what left the OAK-D device
// connection stuck and caused X_LINK_DEVICE_ALREADY_IN_USE on the next run.
void SignalHandler(int signum) {
    std::cout << "\n[Signal Handler] Interrupt signal (" << signum << ") received. Shuting down gracefully..." << std::endl;
    bContinueRunning = false;
}

// Exports MapPoints to a .ply file for rendering -- same implementation as OAKD.cpp's, since
// GetTrackedMapPoints() returns a per-keypoint array (mCurrentFrame.mvpMapPoints) sized by
// ORBextractor.nFeatures, with nullptr for keypoints that never matched a 3D map point, and
// GetWorldPos() returns Eigen::Vector3f in this fork, not cv::Mat.
void SaveMapPointsToPly(const std::vector<ORB_SLAM3::MapPoint*>& vpMapPoints, const std::string& filename) {
    std::cout << "Saving " << vpMapPoints.size() << " mappoints to " << filename << "..." << std::endl;

    // Collected up front so the PLY header's vertex count matches the body exactly (a header
    // count that doesn't match the number of vertex lines written produces a malformed file
    // most PLY readers reject).
    std::vector<Eigen::Vector3f> validPositions;
    validPositions.reserve(vpMapPoints.size());
    for(size_t i = 0; i < vpMapPoints.size(); i++) {
        if(!vpMapPoints[i] || vpMapPoints[i]->isBad()) continue;
        validPositions.push_back(vpMapPoints[i]->GetWorldPos());
    }

    std::ofstream f;
    f.open(filename.c_str());
    f << "ply\nformat ascii 1.0\nelement vertex " << validPositions.size() << "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";

    for(const auto& pos : validPositions) {
        f << pos.x() << " " << pos.y() << " " << pos.z() << " 255 0 0\n";
    }
    f.close();
    std::cout << "Done saving." << std::endl;
}

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage: ./oakd_orbslam3 path/to/ORBvoc.txt path/to/RGBD_settings.yaml" << std::endl;
        return -1;
    }

    // Register the signal handler for Ctrl+C
    std::signal(SIGINT, SignalHandler);

    // params
    // the params here are for syncing the data
    constexpr float kCameraFps = 30.0f;
    constexpr float kIMUHz =300.0f; // always 10x camera fps, this makes the two streams comparable in rate

    dai::Pipeline pipeline;
    ORB_SLAM3::System SLAM("/ORB_SLAM3/Vocabulary/ORBvoc.txt", argv[1], ORB_SLAM3::System::IMU_STEREO, true);

    // creating the nodes and wiring it up
    auto camLeft  = pipeline.create<dai::node::Camera>();
    auto camRight = pipeline.create<dai::node::Camera>();

    // resolution
    // All the possible options are: <-- this is DepthAIv2, but ____
    // THE_800_P | 1280 x 800
    // THE_720_P | 1280 x 720
    // THE_400_P | 640  x 400
    // THE_480_P | 640  x 480
    camLeft->build(dai::CameraBoardSocket::CAM_B, std::make_pair(1280u, 800u), kCameraFps);
    camRight->build(dai::CameraBoardSocket::CAM_C, std::make_pair(1280u, 800u), kCameraFps);

    auto* leftOut = camLeft->requestOutput(std::make_pair(1280u, 800u));
    auto* rightOut = camRight->requestOutput(std::make_pair(1280u, 800u));

    // wire the imu
    auto imu = pipeline.create<dai::node::IMU>();

    // ({Data Types}, Hz rate)
    imu->enableIMUSensor({dai::IMUSensor::ACCELEROMETER_CALIBRATED, dai::IMUSensor::GYROSCOPE_CALIBRATED}, static_cast<uint32_t>(kIMUHz));
    // above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
    imu->setBatchReportThreshold(1);
    // maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
    // if lower or equal to batchReportThreshold then the sending is always blocking on device
    // useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
    imu->setMaxBatchReports(10);

    // syncing data
    auto stereoSync = pipeline.create<dai::node::Sync>();

    // threshold for timestamp alignment
    stereoSync->setSyncThreshold(std::chrono::milliseconds(static_cast<int64_t>(1000.0 / (2.0 * kCameraFps))));

    // configure inputs to be synchronized
    leftOut->link(stereoSync->inputs["left"]);
    rightOut->link(stereoSync->inputs["right"]);

    auto stereoQueue = stereoSync->out.createOutputQueue();
    auto imuQueue = imu->out.createOutputQueue();

    // starting the pipeline
    pipeline.start();
    
    std::cout << "Starting OAK-D stream in HEADLESS mode. Press Ctrl+C to stop tracking and save files." << std::endl;

    // main logic loop governed by the signal handler flag
    std::vector<ORB_SLAM3::IMU::Point> imuBuffer;
    while(bContinueRunning && pipeline.isRunning()) {
        // Drain every IMU packet available right now without blocking on it, so the buffer
        // never falls behind while we wait for the next stereo frame pair below.
        while(auto imuData = imuQueue->tryGet<dai::IMUData>()) {
            for(const auto& packet : imuData->packets) {
                double t = std::chrono::duration<double>(packet.acceleroMeter.getTimestamp().time_since_epoch()).count();
                imuBuffer.emplace_back(packet.acceleroMeter.x, packet.acceleroMeter.y, packet.acceleroMeter.z,
                                        packet.gyroscope.x, packet.gyroscope.y, packet.gyroscope.z, t);
            }
        }

        auto msgGroup = stereoQueue->get<dai::MessageGroup>();
        auto inLeft = msgGroup->get<dai::ImgFrame>("left");
        auto inRight = msgGroup->get<dai::ImgFrame>("right");
        if(!inLeft || !inRight) continue;

        double tframe = std::chrono::duration<double>(inLeft->getTimestamp().time_since_epoch()).count();

        // Hand ORB-SLAM3 every buffered sample up to this frame's timestamp, then drop only
        // the samples actually consumed -- anything newer stays buffered for the next frame.
        auto splitPoint = imuBuffer.begin();
        while(splitPoint != imuBuffer.end() && splitPoint->t <= tframe) ++splitPoint;
        std::vector<ORB_SLAM3::IMU::Point> vImuMeas(imuBuffer.begin(), splitPoint);
        imuBuffer.erase(imuBuffer.begin(), splitPoint);

        SLAM.TrackStereo(inLeft->getCvFrame(), inRight->getCvFrame(), tframe, vImuMeas);

        // Tracking loss (fast motion, occlusion, entering an unmapped area) is handled
        // inside TrackStereo: ORB-SLAM3 attempts relocalization first.
        // if it is lost, it creates a new Atlas
        // if(SLAM.isLost()) {
        //     std::cout << "Tracking lost -- relocalizing or starting a new sub-map." << std::endl;
        // }
    }

    // termination
    std::cout << "Finalizing SLAM state..." << std::endl;
    pipeline.stop();

    // map points exports
    std::cout << "Saving KeyFrame trajectory..." << std::endl;
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    std::vector<ORB_SLAM3::MapPoint*> vpMapPoints = SLAM.GetTrackedMapPoints();
    SaveMapPointsToPly(vpMapPoints, "MapPoints.ply");

    std:: cout << "Exitting Program" << std::endl;

    return 0;
}