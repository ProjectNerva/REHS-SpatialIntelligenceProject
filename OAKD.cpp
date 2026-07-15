#include <iostream>
#include <fstream>
#include <chrono>
#include <csignal>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <depthai/depthai.hpp>
#include <Eigen/Core>

#include "System.h"
#include "Tracking.h"
#include "MapPoint.h"

// Global atomic flag to control the main loop execution
std::atomic<bool> bContinueRunning(true);

// Signal handler function for Ctrl+C (SIGINT)
void SignalHandler(int signum) {
    std::cout << "\n[Signal Handler] Interrupt signal (" << signum << ") received. Shuting down gracefully..." << std::endl;
    bContinueRunning = false;
}

// Function to export MapPoints to .ply file for rendering
void SaveMapPointsToPly(const std::vector<ORB_SLAM3::MapPoint*>& vpMapPoints, const std::string& filename) {
    std::cout << "Saving " << vpMapPoints.size() << " mappoints to " << filename << "..." << std::endl;

    // GetTrackedMapPoints() returns a per-keypoint array (mCurrentFrame.mvpMapPoints)
    // sized by ORBextractor.nFeatures, with nullptr for keypoints that never
    // matched a 3D map point -- must null-check before dereferencing.
    // Collected up front so the PLY header's vertex count matches the body
    // exactly (a header count that doesn't match the number of vertex lines
    // written produces a malformed file most PLY readers reject).
    std::vector<Eigen::Vector3f> validPositions;
    validPositions.reserve(vpMapPoints.size());
    for(size_t i=0; i<vpMapPoints.size(); i++) {
        if(!vpMapPoints[i] || vpMapPoints[i]->isBad()) continue;
        // GetWorldPos() returns Eigen::Vector3f in this fork, not cv::Mat
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
        // NOTE: settings file must be an RGB-D settings file (Camera.type: "PinHole",
        // Camera1.fx/fy/cx/cy, DepthMapFactor, ThDepth, etc.), not a stereo/mono-inertial
        // EuRoC-format file - ORB_SLAM3::System::RGBD expects the RGBD-specific fields.
        std::cerr << "Usage: ./oakd_orbslam3 path/to/ORBvoc.txt path/to/RGBD_settings.yaml" << std::endl;
        return -1;
    }

    // Register the signal handler for Ctrl+C
    std::signal(SIGINT, SignalHandler);

    // Initialize ORB-SLAM3 System in headless mode
    ORB_SLAM3::System SLAM("/ORB_SLAM3/Vocabulary/ORBvoc.txt", argv[1], ORB_SLAM3::System::RGBD, false);

    // Initialize DepthAI Pipeline (DepthAI v3 API: nodes are built/queued directly,
    // no separate dai::Device(pipeline) construction or XLinkOut nodes are needed)
    dai::Pipeline pipeline;

    // Color Camera (unified Camera node, socket CAM_A == RGB)
    auto color = pipeline.create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_A);
    auto* colorOut = color->requestOutput(std::make_pair(960, 540));

    // Mono Cameras for Depth (unified Camera node, CAM_B == LEFT, CAM_C == RIGHT)
    auto left = pipeline.create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_B);
    auto right = pipeline.create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_C);

    auto* leftOut = left->requestOutput(std::make_pair(1280, 720));
    auto* rightOut = right->requestOutput(std::make_pair(1280, 720));

    // Stereo Depth
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    // HIGH_ACCURACY doesn't exist on this depthai version's PresetMode enum;
    // FAST_ACCURACY is compiler-confirmed to exist. Check the full enum on your
    // build machine (grep StereoDepth.hpp) if a denser/higher-quality preset is available.
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::FAST_ACCURACY);
    stereo->setRectifyEdgeFillColor(0);
    stereo->setLeftRightCheck(true);
    // Sub-pixel disparity interpolation -- smoother, more accurate depth values
    // (reduces stair-step noise in mapped point positions) at a small compute cost.
    stereo->setSubpixel(true);
    // Align depth to the color sensor so colorFrame/depthFrame pixels correspond 1:1
    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);

    // Link to Stereo
    leftOut->link(stereo->left);
    rightOut->link(stereo->right);

    // Sync color + depth by timestamp so TrackRGBD always receives a matched pair,
    // instead of pulling two independently-buffered queues in lockstep
    auto sync = pipeline.create<dai::node::Sync>();
    sync->setSyncThreshold(std::chrono::milliseconds(15));
    colorOut->link(sync->inputs["color"]);
    stereo->depth.link(sync->inputs["depth"]);

    auto qSynced = sync->out.createOutputQueue(4, false);

    pipeline.start();

    std::cout << "Starting OAK-D stream in HEADLESS mode. Press Ctrl+C to stop tracking and save files." << std::endl;

    // Main tracking loop governed by the signal handler flag
    while(bContinueRunning && pipeline.isRunning()) {
        auto msgGroup = qSynced->get<dai::MessageGroup>();
        auto inColor = msgGroup->get<dai::ImgFrame>("color");
        auto inDepth = msgGroup->get<dai::ImgFrame>("depth");

        if(inColor && inDepth) {
            cv::Mat colorFrame = inColor->getCvFrame();
            cv::Mat depthFrame = inDepth->getCvFrame();

            cv::Mat depthMeters;
            // Depth is raw uint16 millimeters; converted to float meters here, so
            // DepthMapFactor in the settings YAML must be left at 1.0.
            depthFrame.convertTo(depthMeters, CV_32FC1, 0.001);

            double tframe = inColor->getTimestamp().time_since_epoch().count() / 1e9;

            // Pass the frame to ORB-SLAM3
            SLAM.TrackRGBD(colorFrame, depthMeters, tframe);
        }
    }

    pipeline.stop();

    // 3. Shutdown SLAM and Export
    std::cout << "Finalizing SLAM state..." << std::endl;
    SLAM.Shutdown();

    // Export Camera Movement
    std::cout << "Saving KeyFrame trajectory..." << std::endl;
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    // Export MapPoints
    // NOTE: System has no public accessor for the full Atlas map (mpAtlas is
    // private, no GetMap()). GetTrackedMapPoints() only returns the points
    // tracked in the most recently processed frame, not the whole reconstructed map.
    std::vector<ORB_SLAM3::MapPoint*> vpMapPoints = SLAM.GetTrackedMapPoints();
    SaveMapPointsToPly(vpMapPoints, "MapPoints.ply");

    std::cout << "Exiting program smoothly." << std::endl;
    return 0;
}
