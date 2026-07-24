// oakd_recorder.cpp
//
// Records stereo (CAM_B/CAM_C) + IMU data from an OAK-D into EuRoC MAV
// dataset format, for offline iteration on ORB-SLAM3 settings without
// needing the live camera each time.
//
// Output layout (matches EuRoC convention used by ORB-SLAM3 examples):
//   <out>/mav0/cam0/data/<ts_ns>.png   left images
//   <out>/mav0/cam1/data/<ts_ns>.png   right images
//   <out>/mav0/imu0/data.csv           ts_ns,gx,gy,gz,ax,ay,az
//   <out>/timestamps.txt               one frame ts_ns per line
//
// Usage:
//   ./oakd_recorder <output_dir>
//
// Ctrl+C to stop -- SIGINT is caught so files are flushed/closed cleanly
// instead of being cut off mid-write.

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// ---- fixed capture parameters (per current SLAM iteration) ----
constexpr float kCameraFps = 30.0f;
constexpr float kImuHz = 300.0f;
constexpr uint32_t kWidth = 1280;
constexpr uint32_t kHeight = 720;
constexpr int kImuBatchReportThreshold = 5;
constexpr int kImuMaxBatchReports = 10;

// ---- preview-only motion thresholds, matching TECHNICAL.md's init gate ----
// These are display heuristics for the operator, not filters -- every frame
// and IMU sample is still recorded unconditionally regardless of this readout.
constexpr double kHoldStillSec = 2.0;
constexpr double kGyroRmsMin = 0.05;   // rad/s
constexpr double kAccelStdMin = 0.3;   // m/s^2

std::atomic<bool> g_stop{false};
void onSigint(int) { g_stop = true; }

// Converts a depthai steady_clock timestamp to integer nanoseconds since
// epoch, matching the EuRoC <ts_ns> filename/CSV convention.
template <typename TimePoint>
int64_t toNs(const TimePoint& tp) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}

struct ImuSample {
    double t;              // seconds
    double gx, gy, gz;
    double ax, ay, az;
};

// One stereo pair queued for disk write. Mats are cloned before queuing (depthai may reuse
// the underlying buffer once the pipeline's message is released), so the writer thread owns
// independent copies and never touches pipeline-owned memory.
struct WriteJob {
    fs::path leftPath, rightPath;
    cv::Mat left, right;
};

// Runs on its own thread so PNG encode + disk I/O -- the actual bottleneck behind the frame
// drops/stalls diagnosed from run15/run16/run18's frameDelta gaps -- never blocks the capture
// loop from draining the next camera/IMU packets.
class FrameWriter {
public:
    FrameWriter() : thread_(&FrameWriter::run, this) {}

    ~FrameWriter() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    void enqueue(WriteJob job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(job));
        }
        cv_.notify_one();
    }

private:
    void run() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return done_ || !queue_.empty(); });
            if (queue_.empty() && done_) return;
            WriteJob job = std::move(queue_.front());
            queue_.pop();
            lock.unlock();

            cv::imwrite(job.leftPath.string(), job.left);
            cv::imwrite(job.rightPath.string(), job.right);
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<WriteJob> queue_;
    bool done_ = false;
    std::thread thread_;  // must be declared last so it starts after the members above exist
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./oakd_recorder <output_dir>" << std::endl;
        return 1;
    }
    std::signal(SIGINT, onSigint);

    const fs::path outDir(argv[1]);
    const fs::path cam0Dir = outDir / "mav0" / "cam0" / "data";
    const fs::path cam1Dir = outDir / "mav0" / "cam1" / "data";
    const fs::path imu0Dir = outDir / "mav0" / "imu0";
    fs::create_directories(cam0Dir);
    fs::create_directories(cam1Dir);
    fs::create_directories(imu0Dir);

    std::ofstream imuCsv((imu0Dir / "data.csv").string());
    imuCsv << "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
              "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n";

    std::ofstream tsFile((outDir / "timestamps.txt").string());

    if (!imuCsv || !tsFile) {
        std::cerr << "Failed to open output files under " << outDir << std::endl;
        return 1;
    }

    // ---- pipeline setup (mirrors SIMU_OAKD.cpp) ----
    dai::Pipeline pipeline;

    auto camLeft = pipeline.create<dai::node::Camera>();
    auto camRight = pipeline.create<dai::node::Camera>();
    camLeft->build(dai::CameraBoardSocket::CAM_B, std::make_pair(kWidth, kHeight), kCameraFps);
    camRight->build(dai::CameraBoardSocket::CAM_C, std::make_pair(kWidth, kHeight), kCameraFps);

    // Cap max auto-exposure time so a bright window/light source in frame can't force a long
    // exposure that clips the rest of the scene to pure white -- confirmed the actual failure
    // mode in run11/run12 (sampled Fail-to-track frames showed near-total window blowout).
    // This doesn't fully solve backlit scenes (that needs real per-region metering or HDR, not
    // available here), but a lower ceiling measurably reduces how badly a window clips at the
    // cost of slightly darker room content -- a real partial mitigation, not a complete fix.
    camLeft->initialControl.setAutoExposureLimit(20000);   // 20ms ceiling
    camRight->initialControl.setAutoExposureLimit(20000);

    auto* leftOut = camLeft->requestOutput(std::make_pair(kWidth, kHeight));
    auto* rightOut = camRight->requestOutput(std::make_pair(kWidth, kHeight));

    auto imu = pipeline.create<dai::node::IMU>();
    imu->enableIMUSensor({dai::IMUSensor::ACCELEROMETER_CALIBRATED, dai::IMUSensor::GYROSCOPE_CALIBRATED},
                          static_cast<uint32_t>(kImuHz));
    imu->setBatchReportThreshold(kImuBatchReportThreshold);
    imu->setMaxBatchReports(kImuMaxBatchReports);

    auto stereoSync = pipeline.create<dai::node::Sync>();
    stereoSync->setSyncThreshold(
        std::chrono::milliseconds(static_cast<int64_t>(1000.0 / (2.0 * kCameraFps))));

    leftOut->link(stereoSync->inputs["left"]);
    rightOut->link(stereoSync->inputs["right"]);

    auto stereoQueue = stereoSync->out.createOutputQueue();
    auto imuQueue = imu->out.createOutputQueue();

    pipeline.start();
    std::cout << "Recording to " << outDir << " -- hold still for " << kHoldStillSec
              << "s, then move. Press Ctrl+C to stop, or 'q' in the preview window." << std::endl;

    cv::namedWindow("oakd_recorder preview", cv::WINDOW_NORMAL);

    // Owns the background PNG-encode/write thread for this recording's duration -- reset()
    // just before exit blocks until every queued frame is flushed, so nothing is lost.
    auto frameWriter = std::make_unique<FrameWriter>();

    int64_t frameCount = 0, imuCount = 0;
    std::vector<ImuSample> statBuffer;  // mirrors CSV content, used only for the on-screen readout
    double t0 = -1.0;                   // seconds, set on first frame

    while (pipeline.isRunning() && !g_stop) {
        // Drain all buffered IMU packets first, same non-blocking pattern as SIMU_OAKD.cpp,
        // so the IMU stream never falls behind while we wait on the next stereo pair. Every
        // sample is written to the CSV unconditionally -- the stat buffer below is a separate,
        // non-destructive copy used only to drive the on-screen readout.
        while (auto imuData = imuQueue->tryGet<dai::IMUData>()) {
            for (const auto& packet : imuData->packets) {
                double t = std::chrono::duration<double>(packet.acceleroMeter.getTimestamp().time_since_epoch()).count();
                int64_t tNs = toNs(packet.acceleroMeter.getTimestamp());
                imuCsv << tNs << ','
                       << packet.gyroscope.x << ',' << packet.gyroscope.y << ',' << packet.gyroscope.z << ','
                       << packet.acceleroMeter.x << ',' << packet.acceleroMeter.y << ',' << packet.acceleroMeter.z
                       << '\n';
                ++imuCount;
                statBuffer.push_back({t, packet.gyroscope.x, packet.gyroscope.y, packet.gyroscope.z,
                                       packet.acceleroMeter.x, packet.acceleroMeter.y, packet.acceleroMeter.z});
            }
        }

        auto msgGroup = stereoQueue->tryGet<dai::MessageGroup>();
        if (!msgGroup) continue;
        auto inLeft = msgGroup->get<dai::ImgFrame>("left");
        auto inRight = msgGroup->get<dai::ImgFrame>("right");
        if (!inLeft || !inRight) continue;

        int64_t tNs = toNs(inLeft->getTimestamp());
        double tframe = std::chrono::duration<double>(inLeft->getTimestamp().time_since_epoch()).count();
        if (t0 < 0.0) t0 = tframe;
        double elapsed = tframe - t0;

        cv::Mat left = inLeft->getCvFrame();
        cv::Mat right = inRight->getCvFrame();

        frameWriter->enqueue(WriteJob{cam0Dir / (std::to_string(tNs) + ".png"),
                                      cam1Dir / (std::to_string(tNs) + ".png"),
                                      left.clone(), right.clone()});
        tsFile << tNs << '\n';
        ++frameCount;

        // Same gyro_rms / accel_std excitation check as TECHNICAL.md's init gate, computed
        // over just this frame's slice of IMU samples -- display only, never affects capture.
        auto splitPoint = statBuffer.begin();
        while (splitPoint != statBuffer.end() && splitPoint->t <= tframe) ++splitPoint;
        std::vector<ImuSample> frameSamples(statBuffer.begin(), splitPoint);
        statBuffer.erase(statBuffer.begin(), splitPoint);

        double gyroRms = 0.0, accelStd = 0.0;
        bool excited = false;
        if (!frameSamples.empty()) {
            double gyroSqSum = 0.0, accelSum = 0.0;
            for (const auto& s : frameSamples) {
                gyroSqSum += s.gx * s.gx + s.gy * s.gy + s.gz * s.gz;
                accelSum += std::sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az);
            }
            gyroRms = std::sqrt(gyroSqSum / frameSamples.size());
            double accelMean = accelSum / frameSamples.size();
            double accelSqDiffSum = 0.0;
            for (const auto& s : frameSamples) {
                double mag = std::sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az);
                accelSqDiffSum += (mag - accelMean) * (mag - accelMean);
            }
            accelStd = std::sqrt(accelSqDiffSum / frameSamples.size());
            excited = (gyroRms >= kGyroRmsMin) || (accelStd >= kAccelStdMin);
        }

        // ---- preview window with status overlay ----
        cv::Mat preview;
        cv::cvtColor(left, preview, cv::COLOR_GRAY2BGR);

        std::string statusText;
        cv::Scalar statusColor;
        if (elapsed < kHoldStillSec) {
            statusText = "HOLD STILL -- " + std::to_string(int(kHoldStillSec - elapsed) + 1) + "s";
            statusColor = cv::Scalar(0, 165, 255);  // orange
        } else if (excited) {
            statusText = "EXCITED";
            statusColor = cv::Scalar(0, 220, 0);    // green
        } else {
            statusText = "QUIET -- move now";
            statusColor = cv::Scalar(0, 0, 255);    // red
        }

        cv::putText(preview, statusText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, statusColor, 2);
        char statsLine[128];
        std::snprintf(statsLine, sizeof(statsLine), "gyro_rms=%.3f accel_std=%.3f", gyroRms, accelStd);
        cv::putText(preview, statsLine, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 1);
        char countLine[64];
        std::snprintf(countLine, sizeof(countLine), "frames=%lld  imu=%lld", (long long)frameCount, (long long)imuCount);
        cv::putText(preview, countLine, cv::Point(10, 385), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(200, 200, 200), 1);

        cv::imshow("oakd_recorder preview", preview);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') g_stop = true;

        if (frameCount % 30 == 0) {
            std::cout << "\rFrames: " << frameCount << "  IMU samples: " << imuCount << std::flush;
        }
    }

    cv::destroyAllWindows();

    std::cout << "\nStopping..." << std::endl;
    pipeline.stop();

    // Blocks until every queued frame has actually been written -- without this, stopping
    // right after capture ends could truncate whatever's still sitting in the write queue.
    std::cout << "Flushing remaining queued frames..." << std::endl;
    frameWriter.reset();

    imuCsv.close();
    tsFile.close();

    std::cout << "Done. Wrote " << frameCount << " stereo frame pairs and " << imuCount
              << " IMU samples to " << outDir << std::endl;
    return 0;
}
