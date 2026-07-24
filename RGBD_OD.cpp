// RGBD_OD.cpp
//
// Live OAK-D -> ORB-SLAM3 using plain RGBD mode (System::RGBD, no IMU). Mirrors IMUS_OD.cpp's
// structure exactly, minus everything IMU-specific -- there's no bias/VIBA gating to wait on
// here at all, since RGBD mode gets scale directly from the depth map instead of an
// IMU-estimated one.
//
// Depth comes from the OAK-D's on-device StereoDepth node run on the same CAM_B/CAM_C pair
// oakd_recorder.cpp / IMUS_OD.cpp already use for stereo-inertial -- NOT a third sensor. The
// image fed alongside it must be the RECTIFIED left frame (StereoDepth's own rectifiedLeft
// output), not the raw CAM_B frame the other files record: StereoDepth's block matching runs
// on rectified inputs, so its depth map is only pixel-aligned to that rectified frame, and
// OAK_D_RGBD.yaml's intrinsics (zero distortion) assume that alignment.
//
// UNVERIFIED: this environment has no depthai-core installation to compile/check against
// (confirmed earlier in this project -- dpkg/filesystem search both came up empty, and neither
// this file's live-capture siblings, IMUS_OD.cpp/oakd_recorder.cpp, have ever been built
// either). The dai::node::StereoDepth calls below (setRectification, setLeftRightCheck, the
// left/right inputs, the depth/rectifiedLeft outputs) are written against the documented v3
// API and the same Sync/createOutputQueue pattern already proven elsewhere in this repo, but
// should be treated as a first draft to build and fix against the real SDK, not working code.

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>

#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

// ORB_SLAM3 headers
#include "System.h"
#include "MapPoint.h"

std::atomic<bool> bContinueRunning(true);

void SignalHandler(int signum) {
    std::cout << "\n[Signal Handler] Interrupt signal (" << signum << ") received. Shutting down gracefully..." << std::endl;
    bContinueRunning = false;
}

void SaveMapPointsToPly(const std::vector<ORB_SLAM3::MapPoint*>& vpMapPoints, const std::string& filename) {
    std::cout << "Saving " << vpMapPoints.size() << " mappoints to " << filename << "..." << std::endl;

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
        std::cerr << "Usage: ./RGBD_OD path/to/RGBD_settings.yaml" << std::endl;
        return -1;
    }
    std::signal(SIGINT, SignalHandler);

    constexpr float kCameraFps = 30.0f;
    constexpr uint32_t kWidth = 1280;
    constexpr uint32_t kHeight = 720;

    dai::Pipeline pipeline;
    ORB_SLAM3::System SLAM("/ORB_SLAM3/Vocabulary/ORBvoc.txt", argv[1], ORB_SLAM3::System::RGBD, false);

    auto camLeft  = pipeline.create<dai::node::Camera>();
    auto camRight = pipeline.create<dai::node::Camera>();
    camLeft->build(dai::CameraBoardSocket::CAM_B, std::make_pair(kWidth, kHeight), kCameraFps);
    camRight->build(dai::CameraBoardSocket::CAM_C, std::make_pair(kWidth, kHeight), kCameraFps);

    // Same auto-exposure cap as oakd_recorder.cpp -- confirmed fix for window/light blowout
    // starving ORB of features (run11/run12), applies here just as much as stereo-inertial.
    camLeft->initialControl.setAutoExposureLimit(20000);
    camRight->initialControl.setAutoExposureLimit(20000);

    auto* leftOut = camLeft->requestOutput(std::make_pair(kWidth, kHeight));
    auto* rightOut = camRight->requestOutput(std::make_pair(kWidth, kHeight));

    // StereoDepth: rectifies both inputs internally and block-matches them into a depth map.
    // Depth output aligns to the rectified LEFT frame by default (no setDepthAlign call --
    // that's what OAK_D_RGBD.yaml's single-camera intrinsics assume).
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    stereo->setRectification(true);
    stereo->setLeftRightCheck(true);
    leftOut->link(stereo->left);
    rightOut->link(stereo->right);

    // Sync depth + rectifiedLeft the same way every other pipeline in this repo syncs its two
    // per-frame streams (leftOut/rightOut in oakd_recorder.cpp/IMUS_OD.cpp) -- same idiom, just
    // different source streams.
    auto rgbdSync = pipeline.create<dai::node::Sync>();
    rgbdSync->setSyncThreshold(std::chrono::milliseconds(static_cast<int64_t>(1000.0 / (2.0 * kCameraFps))));
    stereo->rectifiedLeft.link(rgbdSync->inputs["rgb"]);
    stereo->depth.link(rgbdSync->inputs["depth"]);

    auto rgbdQueue = rgbdSync->out.createOutputQueue();

    pipeline.start();
    std::cout << "Starting OAK-D RGBD stream in HEADLESS mode." << std::endl;

    double lastTframe = -1.0;   // TEMP DEBUG: previous frame's capture timestamp, to spot stalls
    int lastTrackState = -100;  // sentinel != any real eTrackingState, forces first print

    while(pipeline.isRunning() && bContinueRunning) {
        auto msgGroup = rgbdQueue->get<dai::MessageGroup>();
        auto inRgb = msgGroup->get<dai::ImgFrame>("rgb");
        auto inDepth = msgGroup->get<dai::ImgFrame>("depth");
        if(!inRgb || !inDepth) continue;

        double tframe = std::chrono::duration<double>(inRgb->getTimestamp().time_since_epoch()).count();

        // TEMP DEBUG: capture-to-capture gap -- same diagnostic as IMUS_OD.cpp (expect ~33ms
        // at 30fps; anything much larger is a stall).
        double frameDeltaMs = (lastTframe >= 0.0) ? (tframe - lastTframe) * 1000.0 : 0.0;
        lastTframe = tframe;

        std::cout << std::fixed << std::setprecision(3)
                  << "[DEBUG] tframe=" << tframe
                  << " frameDelta=" << frameDeltaMs << "ms" << std::endl;

        // TEMP DEBUG: wall-clock cost of TrackRGBD itself -- confirms/denies whether the RPi5
        // is CPU-bound on tracking (expect well under 33ms/frame to keep up with 30fps).
        auto trackStart = std::chrono::steady_clock::now();
        SLAM.TrackRGBD(inRgb->getCvFrame(), inDepth->getCvFrame(), tframe);
        auto trackEnd = std::chrono::steady_clock::now();
        double trackMs = std::chrono::duration<double, std::milli>(trackEnd - trackStart).count();
        std::cout << "[DEBUG] TrackRGBD took " << trackMs << "ms" << std::endl;

        int trackState = SLAM.GetTrackingState();
        if(trackState != lastTrackState) {
            static const char* kStateNames[] = {
                "SYSTEM_NOT_READY", "NO_IMAGES_YET", "NOT_INITIALIZED",
                "OK", "RECENTLY_LOST", "LOST", "OK_KLT"
            };
            auto nameOf = [&](int s) {
                int idx = s + 1;
                return (idx >= 0 && idx < 7) ? kStateNames[idx] : "UNKNOWN";
            };
            std::cout << "[DEBUG] Tracking state: "
                      << (lastTrackState == -100 ? "<init>" : nameOf(lastTrackState))
                      << " -> " << nameOf(trackState) << std::endl;
            lastTrackState = trackState;
        }
    }

    // termination
    std::cout << "Finalizing SLAM state..." << std::endl;
    pipeline.stop();
    SLAM.Shutdown();

    std::cout << "Saving KeyFrame trajectory..." << std::endl;
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    std::vector<ORB_SLAM3::MapPoint*> vpMapPoints = SLAM.GetTrackedMapPoints();
    SaveMapPointsToPly(vpMapPoints, "MapPoints.ply");

    std::cout << "Exitting Program" << std::endl;
    return 0;
}
