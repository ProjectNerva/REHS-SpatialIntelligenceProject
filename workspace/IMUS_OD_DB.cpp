#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
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
#include "ImuTypes.h"
#include "MapPoint.h"

// Global atomic flag to control both the dataset-processing loop and the post-processing
// viewer-idle loop below -- Ctrl+C is the only way to stop this program, whether that lands
// mid-playback or while just sitting on the finished map in the viewer.
std::atomic<bool> bContinueRunning(true);

void SignalHandler(int signum) {
    std::cout << "\n[Signal Handler] Interrupt signal (" << signum << ") received. Shutting down gracefully..." << std::endl;
    bContinueRunning = false;
}

// Deterministic, visually-distinct color per map id. GetAllMapPoints() aggregates points from
// every map the Atlas ever created during a run, including sub-maps orphaned by a tracking
// reset that never got merged back via loop closure -- each of those lives in its own
// unrelated local coordinate frame. Coloring by owning map makes that fragmentation visible
// instead of misleading (a uniform color makes disconnected fragments look like one map).
// Hues are spaced by the golden angle so any number of maps stay visually separable, unlike a
// small fixed palette that would repeat once there are more maps than colors.
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

// Exports MapPoints to a .ply file for rendering. GetWorldPos() returns Eigen::Vector3f in this
// fork, not cv::Mat. Points come from System::GetAllMapPoints() (added upstream in
// ProjectNerva/ORB_SLAM3 commit 30db3fc), which aggregates every map ever created during the
// run via Atlas::GetAllMapPoints() -- unlike GetTrackedMapPoints(), which only returns the last
// processed frame's matched keypoints.
void SaveMapPointsToPly(const std::vector<ORB_SLAM3::MapPoint*>& vpMapPoints, const std::string& filename) {
    std::cout << "Saving " << vpMapPoints.size() << " mappoints to " << filename << "..." << std::endl;

    // Collected up front so the PLY header's vertex count matches the body exactly (a header
    // count that doesn't match the number of vertex lines written produces a malformed file
    // most PLY readers reject).
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

// Reads timestamps for a run. Tries <runDir>/timestamps.txt first (one timestamp string per
// line, nanoseconds -- what this project's own recorder produces). Falls back to
// <runDir>/mav0/cam0/data.csv ("#timestamp [ns],filename" rows) if that file isn't present
bool LoadTimestamps(const std::string& runDir, std::vector<std::string>& vTimestamps) {
    std::ifstream f(runDir + "/timestamps.txt");
    if(f.is_open()) {
        std::string line;
        while(std::getline(f, line)) {
            if(!line.empty()) vTimestamps.push_back(line);
        }
    } else {
        std::ifstream csv(runDir + "/mav0/cam0/data.csv");
        if(!csv.is_open()) {
            std::cerr << "[ERROR] Could not open " << runDir << "/timestamps.txt or "
                      << runDir << "/mav0/cam0/data.csv" << std::endl;
            return false;
        }
        std::string line;
        while(std::getline(csv, line)) {
            if(line.empty() || line[0] == '#') continue;
            size_t commaPos = line.find(',');
            if(commaPos == std::string::npos) continue;
            vTimestamps.push_back(line.substr(0, commaPos));
        }
    }
    if(vTimestamps.empty()) {
        std::cerr << "[ERROR] " << runDir << ": no timestamps found in timestamps.txt or mav0/cam0/data.csv" << std::endl;
        return false;
    }
    return true;
}

// Parses <runDir>/mav0/imu0/data.csv (EuRoC format: header comment line, then
// "ts_ns,w_x,w_y,w_z,a_x,a_y,a_z" per row). ORB_SLAM3::IMU::Point's constructor takes
// (accel xyz, gyro xyz, t) -- the reverse of the CSV's gyro-then-accel column order.
bool LoadIMU(const std::string& runDir, std::vector<ORB_SLAM3::IMU::Point>& vImuData) {
    std::ifstream f(runDir + "/mav0/imu0/data.csv");
    if(!f.is_open()) {
        std::cerr << "[ERROR] Could not open " << runDir << "/mav0/imu0/data.csv" << std::endl;
        return false;
    }
    std::string line;
    while(std::getline(f, line)) {
        if(line.empty() || line[0] == '#') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);
        int64_t tsNs;
        double wx, wy, wz, ax, ay, az;
        ss >> tsNs >> wx >> wy >> wz >> ax >> ay >> az;
        double t = static_cast<double>(tsNs) * 1e-9;
        vImuData.emplace_back(ax, ay, az, wx, wy, wz, t);
    }
    if(vImuData.empty()) {
        std::cerr << "[ERROR] " << runDir << "/mav0/imu0/data.csv has no IMU rows" << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if(argc < 3) {
        std::cerr << "Usage: ./IMUS_OD_DB path/to/RGBD_settings.yaml path/to/run_dir1 [path/to/run_dir2 ...]" << std::endl;
        std::cerr << "Each run_dir must contain timestamps.txt and mav0/{cam0,cam1,imu0} (EuRoC layout)." << std::endl;
        return -1;
    }

    // Register the signal handler for Ctrl+C
    std::signal(SIGINT, SignalHandler);

    // init gate: hold-still window for bias/gravity, then require motion for scale (see TECHNICAL.md)
    constexpr double kHoldStillSec = 2.0;
    constexpr double kGyroRmsMin = 0.05;   // rad/s
    constexpr double kAccelStdMin = 0.3;   // m/s^2

    std::vector<std::string> vRunDirs(argv + 2, argv + argc);

    // Viewer enabled -- rendered over the Docker image's noVNC stack (Xvfb + fluxbox + x11vnc +
    // websockify, all started by the Dockerfile's entrypoint before this binary runs). Open
    // http://localhost:8080/vnc.html to watch it.
    ORB_SLAM3::System SLAM("/ORB_SLAM3/Vocabulary/ORBvoc.txt", argv[1], ORB_SLAM3::System::IMU_STEREO, true);

    for(size_t runIdx = 0; runIdx < vRunDirs.size() && bContinueRunning; runIdx++) {
        const std::string& runDir = vRunDirs[runIdx];
        std::cout << "=== Run " << (runIdx + 1) << "/" << vRunDirs.size() << ": " << runDir << " ===" << std::endl;

        std::vector<std::string> vTimestamps;
        std::vector<ORB_SLAM3::IMU::Point> vImuData;
        if(!LoadTimestamps(runDir, vTimestamps) || !LoadIMU(runDir, vImuData)) {
            std::cerr << "[WARN] Skipping run (failed to load): " << runDir << std::endl;
            continue;
        }
        std::cout << "Loaded " << vTimestamps.size() << " frames, " << vImuData.size() << " IMU samples." << std::endl;

        // Signals a new recording session to ORB-SLAM3 so the gap between the previous run's
        // last frame and this run's first isn't treated as continuous motion. Skipped before
        // the very first run since there's no prior session yet.
        if(runIdx > 0) SLAM.ChangeDataset();

        size_t imuCursor = 0;
        double lastTframe = -1.0;      // previous frame's timestamp, to spot stalls/gaps
        double runStartTframe = -1.0;  // this run's first frame timestamp, for the init gate below
        bool motionConfirmed = false;
        int lastTrackState = -100;     // sentinel != any real eTrackingState, forces first print

        for(const auto& ts : vTimestamps) {
            if(!bContinueRunning) break;

            cv::Mat imLeft  = cv::imread(runDir + "/mav0/cam0/data/" + ts + ".png", cv::IMREAD_GRAYSCALE);
            cv::Mat imRight = cv::imread(runDir + "/mav0/cam1/data/" + ts + ".png", cv::IMREAD_GRAYSCALE);
            if(imLeft.empty() || imRight.empty()) {
                std::cerr << "[WARN] Missing image for timestamp " << ts << " -- skipping frame." << std::endl;
                continue;
            }

            double tframe = std::stod(ts) * 1e-9;
            if(runStartTframe < 0.0) runStartTframe = tframe;

            // TEMP DEBUG: frame-to-frame gap -- confirms/denies whether the recording has
            // stalls/drops (should roughly match the capture interval used when recording).
            double frameDeltaMs = (lastTframe >= 0.0) ? (tframe - lastTframe) * 1000.0 : 0.0;
            lastTframe = tframe;

            size_t imuBufBefore = vImuData.size() - imuCursor;

            // Hand ORB-SLAM3 every sample up to this frame's timestamp, advancing the cursor
            // past only what's consumed -- same apportioning logic as the live-camera version,
            // just walking a pre-loaded vector instead of draining an incoming queue.
            size_t splitPoint = imuCursor;
            while(splitPoint < vImuData.size() && vImuData[splitPoint].t <= tframe) ++splitPoint;
            std::vector<ORB_SLAM3::IMU::Point> vImuMeas(vImuData.begin() + imuCursor, vImuData.begin() + splitPoint);
            imuCursor = splitPoint;

            // init gate: guidance only, never blocks TrackStereo (see TECHNICAL.md). Elapsed
            // time is measured from this run's first frame timestamp, not wall-clock -- dataset
            // playback doesn't proceed in real time, so a wall-clock hold-still window wouldn't
            // correspond to anything meaningful in the recording.
            double elapsedSec = tframe - runStartTframe;
            if(!motionConfirmed && !vImuMeas.empty()) {
                double gyroSqSum = 0, accelSum = 0, accelSqSum = 0;
                for(const auto& p : vImuMeas) {
                    gyroSqSum += p.w.squaredNorm();
                    double an = p.a.norm();
                    accelSum += an; accelSqSum += an * an;
                }
                size_t n = vImuMeas.size();
                double gyroRms = std::sqrt(gyroSqSum / n);
                double accelMean = accelSum / n;
                double accelVar = accelSqSum / n - accelMean * accelMean;
                double accelStd = std::sqrt(accelVar > 0 ? accelVar : 0.0);
                if(gyroRms >= kGyroRmsMin || accelStd >= kAccelStdMin) {
                    motionConfirmed = true;
                    std::cout << "[GATE] motion confirmed" << std::endl;
                } else if(elapsedSec < kHoldStillSec) {
                    std::cout << "[GATE] hold still -- " << (kHoldStillSec - elapsedSec) << "s" << std::endl;
                } else {
                    std::cout << "[GATE] move the camera now" << std::endl;
                }
            }

            // TEMP DEBUG: one consolidated line per frame -- frame gap, buffer occupancy before
            // and after bucketing, and how many samples actually got handed to TrackStereo.
            std::cout << std::fixed << std::setprecision(3)
                      << "[DEBUG] tframe=" << tframe
                      << " frameDelta=" << frameDeltaMs << "ms"
                      << " imuBufBefore=" << imuBufBefore
                      << " vImuMeas=" << vImuMeas.size()
                      << " imuBufAfter=" << (vImuData.size() - imuCursor)
                      << std::endl;

            if(vImuMeas.empty() && SLAM.GetTrackingState() != ORB_SLAM3::Tracking::SYSTEM_NOT_READY) {
                std::cout << "[DEBUG] SKIP: TrackStereo not called (empty vImuMeas)" << std::endl;
                continue;
            }

            // TEMP DEBUG: wall-clock cost of TrackStereo itself -- confirms/denies whether the
            // RPi5 is CPU-bound on tracking.
            auto trackStart = std::chrono::steady_clock::now();
            SLAM.TrackStereo(imLeft, imRight, tframe, vImuMeas);
            auto trackEnd = std::chrono::steady_clock::now();
            double trackMs = std::chrono::duration<double, std::milli>(trackEnd - trackStart).count();
            std::cout << "[DEBUG] TrackStereo took " << trackMs << "ms" << std::endl;

            // Tracking loss (fast motion, occlusion, entering an unmapped area) is handled
            // inside TrackStereo: ORB-SLAM3 attempts relocalization first, and only creates a
            // new Atlas sub-map if that fails. Log every state transition so a LOST->OK bounce
            // (relocalized) can be told apart from LOST persisting until a fresh map appears
            // (reset) -- currently invisible from the console output.
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

    // All runs processed (or interrupted mid-playback above via Ctrl+C) -- proceed straight to
    // shutdown/export instead of blocking on further input, so batch runs (piping to a log file
    // for later analysis) terminate on their own once playback finishes.
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

    // Shutdown() intermittently segfaults during teardown on this fork (non-deterministic
    // thread/Pangolin-static race, confirmed under gdb to happen after every output above is
    // already written) -- now that this runs unattended at the end of every batch run instead
    // of only after a manual Ctrl+C, quick_exit avoids that crash turning into a nonzero exit
    // code / truncated log in a scripted run.
    std::cout.flush();
    std::quick_exit(0);
}
