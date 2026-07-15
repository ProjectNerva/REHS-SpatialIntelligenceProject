#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <fstream>
#include <atomic>
#include <termios.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <depthai/depthai.hpp>
#include <System.h>
#include <ImuTypes.h>

// Extended helper function to save both map points and keyframe positions to a single PLY
void SaveCoherentMapToPLY(ORB_SLAM3::System& SLAM, const std::string& filename) {
    std::vector<ORB_SLAM3::MapPoint*> vpMPs;
    std::vector<ORB_SLAM3::KeyFrame*> vpKFs;
    
    // NOTE: GetAtlas()/Map::GetAllMapPoints()/Map::GetAllKeyFrames()/
    // KeyFrame::GetCameraCenter() aren't part of stock ORB_SLAM3's public
    // System.h -- unconfirmed whether this fork exposes them. If this
    // doesn't compile, fall back to SLAM.GetTrackedMapPoints() (confirmed
    // public, used in SI_OAKD.cpp) for the point cloud, though that only
    // returns currently-tracked points rather than the full map + keyframe
    // path.
    ORB_SLAM3::Atlas* pAtlas = SLAM.GetAtlas();
    if(pAtlas) {
        ORB_SLAM3::Map* pActiveMap = pAtlas->GetCurrentMap();
        if(pActiveMap) {
            vpMPs = pActiveMap->GetAllMapPoints();
            vpKFs = pActiveMap->GetAllKeyFrames();
        }
    }

    // Filter valid map points (Colored White: 255, 255, 255)
    std::vector<std::pair<Eigen::Vector3f, std::array<uint8_t, 3>>> plyElements;
    for (auto pMP : vpMPs) {
        if (pMP && !pMP->isBad()) {
            plyElements.push_back({pMP->GetWorldPos(), {255, 255, 255}});
        }
    }

    // Append Keyframe camera positions (Colored Red: 255, 0, 0)
    for (auto pKF : vpKFs) {
        if (pKF && !pKF->isBad()) {
            // Get camera center in world coordinates
            Eigen::Vector3f cameraCenter = pKF->GetCameraCenter();
            plyElements.push_back({cameraCenter, {255, 0, 0}});
        }
    }

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open PLY file: " << filename << std::endl;
        return;
    }

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << plyElements.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";

    for (const auto& el : plyElements) {
        out << el.first.x() << " " << el.first.y() << " " << el.first.z() << " "
            << (int)el.second[0] << " " << (int)el.second[1] << " " << (int)el.second[2] << "\n";
    }

    out.close();
    std::cout << "Exported " << plyElements.size() << " elements (Points + Path) to " << filename << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " path_to_settings\n";
        return 1;
    }

    ORB_SLAM3::System SLAM("/ORB_SLAM3/Vocabulary/ORBvoc.txt", argv[1], ORB_SLAM3::System::IMU_STEREO, false);

    dai::Pipeline pipeline;
    auto cameraNode = pipeline.create<dai::node::Camera>();

    // THE_720_P (1280x720), not THE_800_P -- OAK_D.yaml's intrinsics were
    // pulled via getCameraIntrinsics(socket, 1280, 720), so streaming at a
    // different resolution misaligns cx/cy against that calibration.
    cameraNode->configureCamera(dai::CameraBoardSocket::CAM_B, [](auto& config) {
        config.setResolution(dai::CameraProperties::SensorResolution::THE_720_P);
        config.setFps(30.0);
    });

    cameraNode->configureCamera(dai::CameraBoardSocket::CAM_C, [](auto& config) {
        config.setResolution(dai::CameraProperties::SensorResolution::THE_720_P);
        config.setFps(30.0);
    });

    auto imu = pipeline.create<dai::node::IMU>();
    imu->enableIMUSensor({dai::IMUSensor::ACCELEROMETER_CALIBRATED, dai::IMUSensor::GYROSCOPE_CALIBRATED}, 200);
    imu->setBatchReportThreshold(1);
    imu->setMaxBatchReports(10);
    imu->setTimestampSyncMode(dai::IMU_TIMESTAMP_SYNC_MODE_DEVICE_CLOCK);

    // Node routing via Luxonis API v3
    auto qLeft = cameraNode->outputs[dai::CameraBoardSocket::CAM_B].createOutputQueue();
    auto qRight = cameraNode->outputs[dai::CameraBoardSocket::CAM_C].createOutputQueue();
    auto qImu = imu->out.createOutputQueue();

    pipeline.start();

    // Kill switch so the loop can actually exit and reach Shutdown()/the
    // trajectory saves below -- while(true) with no break made those
    // permanently dead code. Raw terminal mode + background input thread,
    // same pattern as SI_OAKD.cpp.
    termios oldTermios{};
    tcgetattr(STDIN_FILENO, &oldTermios);
    termios rawTermios = oldTermios;
    rawTermios.c_lflag &= ~(ICANON | ECHO);
    rawTermios.c_cc[VMIN] = 1;
    rawTermios.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &rawTermios);

    std::atomic<bool> quitRequested{false};
    std::thread inputThread([&quitRequested]() {
        std::cout << "Press 'q' at any time to stop and save.\n";
        char ch;
        while (std::cin.get(ch)) {
            if (ch == 'q' || ch == 'Q') {
                quitRequested = true;
                break;
            }
        }
    });
    inputThread.detach();

    std::vector<ORB_SLAM3::IMU::Point> imuBuffer;
    auto lastSaveTime = std::chrono::steady_clock::now();

    while (pipeline.isRunning() && !quitRequested) {
        // 1. Buffer incoming IMU packets converting to ORB-SLAM3 format
        if (auto imuData = qImu->tryGet<dai::IMUData>()) {
            for (const auto& packet : imuData->packets) {
                // Time conversion to double (seconds) matching Device Clock
                double imuTime = std::chrono::duration<double>(packet.acceleroMeter.timestamp.getTimestamp().time_since_epoch()).count();
                
                cv::Point3f acc(packet.acceleroMeter.x, packet.acceleroMeter.y, packet.acceleroMeter.z);
                cv::Point3f gyr(packet.gyroscope.x, packet.gyroscope.y, packet.gyroscope.z);
                
                imuBuffer.push_back(ORB_SLAM3::IMU::Point(acc, gyr, imuTime));
            }
        }

        auto leftData = qLeft->tryGet<dai::ImgFrame>();
        auto rightData = qRight->tryGet<dai::ImgFrame>();

        if (leftData && rightData) {
            // Device-clock timestamps to line up with the IMU samples, which
            // are also stamped via IMU_TIMESTAMP_SYNC_MODE_DEVICE_CLOCK
            // above. getTimestamp(TimestampSource::DEVICE_CLOCK) isn't a
            // real depthai API -- getTimestampDevice() is the actual
            // accessor for this, same as SI_OAKD.cpp. NOTE: still
            // unconfirmed against this depthai-core version's ImgFrame.hpp
            // on device.
            auto leftTime = leftData->getTimestampDevice();
            auto rightTime = rightData->getTimestampDevice();

            auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(leftTime - rightTime).count();
            
            // Stereo frame sync confirmation (<= 2ms window)
            if (std::abs(timeDiff) <= 2) {
                double frameTimeSec = std::chrono::duration<double>(leftTime.time_since_epoch()).count();
                
                // Extract IMU points occurring before/up to current frame time
                std::vector<ORB_SLAM3::IMU::Point> currentFrameImu;
                auto it = imuBuffer.begin();
                while (it != imuBuffer.end()) {
                    if (it->t <= frameTimeSec) {
                        currentFrameImu.push_back(*it);
                        it = imuBuffer.erase(it); // Correctly advances iterator
                    } else {
                        ++it;
                    }
                }

                // Convert DepthAI Frame to OpenCV Mat
                cv::Mat leftImg = leftData->getCvFrame();
                cv::Mat rightImg = rightData->getCvFrame();

                // Pass successfully parsed datasets to SLAM engine
                SLAM.TrackStereo(leftImg, rightImg, frameTimeSec, currentFrameImu);
            }
        }

        // Periodically export unified map data every 5 seconds
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastSaveTime).count() >= 5) {
            SaveCoherentMapToPLY(SLAM, "active_map.ply");
            lastSaveTime = now;
        }

        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }

    // Restore the terminal before touching stdout further, and stop the
    // device explicitly (dai::Pipeline destructor racing ORB-SLAM3's
    // background threads caused a device crash in SI_OAKD.cpp -- see that
    // file's history).
    tcsetattr(STDIN_FILENO, TCSANOW, &oldTermios);
    pipeline.stop();

    SLAM.Shutdown();
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");

    return 0;
}