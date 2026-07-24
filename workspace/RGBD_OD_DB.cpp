// RGBD_OD_DB.cpp
//
// Offline replay/debug harness for ORB-SLAM3's plain RGBD sensor mode (System::RGBD -- no
// IMU). Mirrors IMUS_OD_DB.cpp's structure and debug-print conventions exactly, minus
// everything IMU-specific: no [GATE] motion gate (RGBD initializes on the first frame with
// enough features + valid depth -- see Tracking::StereoInitialization's mSensor==System::RGBD
// path, which skips the IMU-only branches entirely), no vImuMeas bucketing, no bias/VIBA
// concerns at all. That's the whole point of this sensor mode: scale comes directly from the
// depth map, not from an IMU that has to survive 15 seconds of continuous motion to become
// trustworthy.
//
// Expected run directory layout (produced by a depth-capable recorder, not yet written):
//   <run_dir>/timestamps.txt      one <ts_ns> per line
//   <run_dir>/rgb/<ts_ns>.png     rectified left mono image (or color, if ever added)
//   <run_dir>/depth/<ts_ns>.png   16-bit depth map in millimeters, aligned to rgb/

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <csignal>
#include <atomic>
#include <set>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

// ORB_SLAM3 headers
#include "System.h"
#include "MapPoint.h"

// Global atomic flag to control both the dataset-processing loop and the post-processing
// viewer-idle loop below -- Ctrl+C is the only way to stop this program, whether that lands
// mid-playback or while just sitting on the finished map in the viewer.
std::atomic<bool> bContinueRunning(true);

void SignalHandler(int signum) {
    std::cout << "\n[Signal Handler] Interrupt signal (" << signum << ") received. Shutting down gracefully..." << std::endl;
    bContinueRunning = false;
}

// Deterministic, visually-distinct color per map id -- identical to IMUS_OD_DB.cpp's version.
// See that file's comment for why: GetAllMapPoints() aggregates every sub-map the Atlas ever
// created, including ones orphaned by a reset that never got merged via loop closure, and
// coloring by owning map makes that fragmentation visible instead of hiding it under one color.
void MapIdToColor(unsigned long mapId, unsigned char& r, unsigned char& g, unsigned char& b) {
    double hue = std::fmod(mapId * 137.508, 360.0);
    double c = 1.0, x = c * (1 - std::fabs(std::fmod(hue / 60.0, 2) - 1));
    double rf, gf, bf;
    if(hue < 60)       { rf = c; gf = x; bf = 0; }
    else if(hue < 120) { rf = x; gf = c; bf = 0; }
    else if(hue < 180) { rf = 0; gf = c; bf = x; }
    else if(hue < 240) { rf = 0; gf = x; bf = c; }
    else if(hue < 300) { rf = x; gf = 0; bf = c; }
    else               { rf = c; gf = 0; bf = x; }
    r = static_cast<unsigned char>(rf * 255);
    g = static_cast<unsigned char>(gf * 255);
    b = static_cast<unsigned char>(bf * 255);
}

// Exports MapPoints to a .ply file for rendering -- identical to IMUS_OD_DB.cpp's version.
void SaveMapPointsToPly(const std::vector<ORB_SLAM3::MapPoint*>& vpMapPoints, const std::string& filename) {
    std::cout << "Saving " << vpMapPoints.size() << " mappoints to " << filename << "..." << std::endl;

    struct ColoredPoint { Eigen::Vector3f pos; unsigned long mapId; };
    std::vector<ColoredPoint> validPoints;
    validPoints.reserve(vpMapPoints.size());
    std::set<unsigned long> distinctMapIds;
    for(size_t i = 0; i < vpMapPoints.size(); i++) {
        if(!vpMapPoints[i] || vpMapPoints[i]->isBad()) continue;
        ORB_SLAM3::Map* pMap = vpMapPoints[i]->GetMap();
        unsigned long mapId = pMap ? pMap->GetId() : 0;
        validPoints.push_back({vpMapPoints[i]->GetWorldPos(), mapId});
        distinctMapIds.insert(mapId);
    }
    if(distinctMapIds.size() > 1) {
        std::cout << "[WARN] Points span " << distinctMapIds.size()
                  << " disconnected maps (never merged by loop closure) -- colored per-map in the .ply." << std::endl;
    }

    std::ofstream f;
    f.open(filename.c_str());
    f << "ply\nformat ascii 1.0\nelement vertex " << validPoints.size() << "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";

    for(const auto& cp : validPoints) {
        unsigned char r, g, b;
        MapIdToColor(cp.mapId, r, g, b);
        f << cp.pos.x() << " " << cp.pos.y() << " " << cp.pos.z() << " "
          << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
    }
    f.close();
    std::cout << "Done saving." << std::endl;
}

// Reads <runDir>/timestamps.txt: one timestamp string per line, nanoseconds -- same convention
// oakd_recorder.cpp already writes for the stereo-inertial path. No mav0/*.csv fallback here
// since there's no IMU stream to have produced one.
bool LoadTimestamps(const std::string& runDir, std::vector<std::string>& vTimestamps) {
    std::ifstream f(runDir + "/timestamps.txt");
    if(!f.is_open()) {
        std::cerr << "[ERROR] Could not open " << runDir << "/timestamps.txt" << std::endl;
        return false;
    }
    std::string line;
    while(std::getline(f, line)) {
        if(!line.empty()) vTimestamps.push_back(line);
    }
    if(vTimestamps.empty()) {
        std::cerr << "[ERROR] " << runDir << "/timestamps.txt has no timestamps" << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if(argc < 3) {
        std::cerr << "Usage: ./RGBD_OD_DB path/to/RGBD_settings.yaml path/to/run_dir1 [path/to/run_dir2 ...]" << std::endl;
        std::cerr << "Each run_dir must contain timestamps.txt, rgb/<ts_ns>.png, depth/<ts_ns>.png." << std::endl;
        return -1;
    }

    // Register the signal handler for Ctrl+C
    std::signal(SIGINT, SignalHandler);

    std::vector<std::string> vRunDirs(argv + 2, argv + argc);

    // Viewer enabled -- rendered over the Docker image's noVNC stack, same as SLAM_DEBUG. Open
    // http://localhost:8080/vnc.html to watch it.
    ORB_SLAM3::System SLAM("/ORB_SLAM3/Vocabulary/ORBvoc.txt", argv[1], ORB_SLAM3::System::RGBD, true);

    for(size_t runIdx = 0; runIdx < vRunDirs.size() && bContinueRunning; runIdx++) {
        const std::string& runDir = vRunDirs[runIdx];
        std::cout << "=== Run " << (runIdx + 1) << "/" << vRunDirs.size() << ": " << runDir << " ===" << std::endl;

        std::vector<std::string> vTimestamps;
        if(!LoadTimestamps(runDir, vTimestamps)) {
            std::cerr << "[WARN] Skipping run (failed to load): " << runDir << std::endl;
            continue;
        }
        std::cout << "Loaded " << vTimestamps.size() << " frames." << std::endl;

        // Signals a new recording session to ORB-SLAM3 so the gap between the previous run's
        // last frame and this run's first isn't treated as continuous -- same as IMUS_OD_DB.cpp.
        if(runIdx > 0) SLAM.ChangeDataset();

        double lastTframe = -1.0;   // previous frame's timestamp, to spot stalls/gaps
        int lastTrackState = -100;  // sentinel != any real eTrackingState, forces first print

        for(const auto& ts : vTimestamps) {
            if(!bContinueRunning) break;

            cv::Mat imRGB = cv::imread(runDir + "/rgb/" + ts + ".png", cv::IMREAD_UNCHANGED);
            cv::Mat imDepth = cv::imread(runDir + "/depth/" + ts + ".png", cv::IMREAD_UNCHANGED);
            if(imRGB.empty() || imDepth.empty()) {
                std::cerr << "[WARN] Missing image for timestamp " << ts << " -- skipping frame." << std::endl;
                continue;
            }

            double tframe = std::stod(ts) * 1e-9;

            // TEMP DEBUG: frame-to-frame gap -- confirms/denies whether the recording has
            // stalls/drops, same diagnostic that caught the recorder's write-thread stall on
            // the stereo-inertial side.
            double frameDeltaMs = (lastTframe >= 0.0) ? (tframe - lastTframe) * 1000.0 : 0.0;
            lastTframe = tframe;

            std::cout << std::fixed << std::setprecision(3)
                      << "[DEBUG] tframe=" << tframe
                      << " frameDelta=" << frameDeltaMs << "ms" << std::endl;

            // TEMP DEBUG: wall-clock cost of TrackRGBD itself.
            auto trackStart = std::chrono::steady_clock::now();
            SLAM.TrackRGBD(imRGB, imDepth, tframe);
            auto trackEnd = std::chrono::steady_clock::now();
            double trackMs = std::chrono::duration<double, std::milli>(trackEnd - trackStart).count();
            std::cout << "[DEBUG] TrackRGBD took " << trackMs << "ms" << std::endl;

            // Log every tracking-state transition -- same as IMUS_OD_DB.cpp, so a LOST->OK
            // bounce (relocalized) can be told apart from LOST persisting until a fresh map
            // appears (reset).
            int trackState = SLAM.GetTrackingState();
            if(trackState != lastTrackState) {
                static const char* kStateNames[] = {
                    "SYSTEM_NOT_READY", "NO_IMAGES_YET", "NOT_INITIALIZED",
                    "OK", "RECENTLY_LOST", "LOST", "OK_KLT"
                };
                auto nameOf = [&](int s) {
                    int idx = s + 1; // SYSTEM_NOT_READY == -1, so shift to a 0-based index
                    return (idx >= 0 && idx < 7) ? kStateNames[idx] : "UNKNOWN";
                };
                std::cout << "[DEBUG] Tracking state: "
                          << (lastTrackState == -100 ? "<init>" : nameOf(lastTrackState))
                          << " -> " << nameOf(trackState) << std::endl;
                lastTrackState = trackState;
            }
        }
    }

    if(bContinueRunning) {
        std::cout << "Dataset playback complete." << std::endl;
    }

    // termination
    std::cout << "Finalizing SLAM state..." << std::endl;
    SLAM.Shutdown();

    // map points exports
    std::cout << "Saving KeyFrame trajectory..." << std::endl;
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    std::vector<ORB_SLAM3::MapPoint*> vpMapPoints = SLAM.GetAllMapPoints();
    SaveMapPointsToPly(vpMapPoints, "MapPoints.ply");

    std::cout << "Exitting Program" << std::endl;

    // Same shutdown-segfault workaround as IMUS_OD_DB.cpp (non-deterministic thread/Pangolin-
    // static race during Shutdown() teardown on this fork) -- quick_exit for a deterministic
    // exit 0 in unattended/batch runs.
    std::cout.flush();
    std::quick_exit(0);
}
