#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <csignal>
#include <atomic>

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

// Reads <runDir>/timestamps.txt (one timestamp string per line, nanoseconds). Kept as strings,
// not parsed to numbers here, so the exact text can be reused to build "<ts>.png" image paths
// without floating-point round-tripping changing a digit.
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
        std::cerr << "[ERROR] " << runDir << "/timestamps.txt is empty" << std::endl;
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
            // inside TrackStereo: ORB-SLAM3 attempts relocalization first.
            // if it is lost, it creates a new Atlas
            // if(SLAM.isLost()) {
            //     std::cout << "Tracking lost -- relocalizing or starting a new sub-map." << std::endl;
            // }
        }
    }

    // All runs processed (or interrupted mid-playback above) -- keep the process alive so the
    // finished map stays viewable in the Pangolin window until Ctrl+C, rather than exiting and
    // tearing the viewer down the moment dataset playback ends.
    // NOTE: does not also check SLAM.isFinished() here -- on this fork, isFinished() reaches
    // into LocalMapping::GetCurrKFTime(), which segfaults if called before any KeyFrame has
    // ever been created (confirmed via gdb backtrace). Ctrl+C (bContinueRunning) is the only
    // supported way to stop, which already satisfies "persistent until ctrl c is pressed".
    if(bContinueRunning) {
        std::cout << "Dataset playback complete. Viewer stays open -- press Ctrl+C to stop and save." << std::endl;
        while(bContinueRunning) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
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

    return 0;
}
