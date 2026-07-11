#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include <unordered_set>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

// Include the primary ORB-SLAM3 System header
#include "System.h"
#include "MapPoint.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " path_to_vocabulary path_to_settings path_to_video\n";
        return 1;
    }

    std::string strVocFile = argv[1];
    std::string strSettingsFile = argv[2];
    std::string strVideoFile = argv[3];

    cv::VideoCapture cap(strVideoFile);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << strVideoFile << '\n';
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0; // fall back if the container doesn't report it

    double totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT); // for progress (may be 0 if unknown)

    try {
        // MONOCULAR, viewer DISABLED (last arg = false): the Pangolin OpenGL
        // window can't get a GL context through XQuartz, so we run headless.
        // Tracking/mapping/loop-closure all still run; we just don't draw the map.
        ORB_SLAM3::System SLAM(strVocFile, strSettingsFile, ORB_SLAM3::System::MONOCULAR, false);

        std::cout << "ORB-SLAM3 System Initialized Successfully! (headless, no viewer)" << std::endl;
        std::cout << "Processing video: " << strVideoFile
                   << " (" << fps << " fps)" << std::endl;

        // Accumulate every map point the tracker ever locked onto. GetTrackedMapPoints()
        // only returns the CURRENT frame's points, so we union them across all frames
        // (keyed by pointer) to reconstruct the full sparse map of the surroundings.
        std::unordered_set<ORB_SLAM3::MapPoint*> mapPoints;

        cv::Mat frame;
        int frameIdx = 0;
        while (cap.read(frame)) {
            double timestamp = frameIdx / fps; // seconds since video start

            // Headless: process as fast as possible (no need to pace to real time
            // when there's no live view). This is much faster than realtime playback.
            SLAM.TrackMonocular(frame, timestamp);

            for (ORB_SLAM3::MapPoint* mp : SLAM.GetTrackedMapPoints())
                if (mp) mapPoints.insert(mp);

            ++frameIdx;

            // Print progress every ~30 frames so you can see it's alive.
            if (frameIdx % 30 == 0) {
                if (totalFrames > 0)
                    std::cout << "  processed " << frameIdx << " / "
                               << (long)totalFrames << " frames\r" << std::flush;
                else
                    std::cout << "  processed " << frameIdx << " frames\r" << std::flush;
            }
        }

        std::cout << "\nFinished processing " << frameIdx << " frames." << std::endl;

        // Write the sparse map as an ASCII .ply point cloud (openable in MeshLab/
        // CloudCompare, or our web viewer). Filter culled points via isBad().
        {
            std::vector<Eigen::Vector3f> cloud;
            cloud.reserve(mapPoints.size());
            for (ORB_SLAM3::MapPoint* mp : mapPoints)
                if (mp && !mp->isBad())
                    cloud.push_back(mp->GetWorldPos());

            std::ofstream ply("/shared_data/MapPoints.ply");
            ply << "ply\nformat ascii 1.0\n";
            ply << "element vertex " << cloud.size() << "\n";
            ply << "property float x\nproperty float y\nproperty float z\n";
            ply << "end_header\n";
            for (const auto& p : cloud)
                ply << p.x() << " " << p.y() << " " << p.z() << "\n";
            ply.close();
            std::cout << "Map cloud saved: " << cloud.size()
                       << " points -> /shared_data/MapPoints.ply" << std::endl;
        }

        SLAM.Shutdown();

        // With the viewer off, this is how you "see" the result: the estimated
        // camera trajectory is written to a TUM-format text file (timestamp tx ty
        // tz qx qy qz qw per line). Open it to confirm SLAM produced poses.
        SLAM.SaveKeyFrameTrajectoryTUM("/shared_data/KeyFrameTrajectory.txt");
        std::cout << "Trajectory saved to /shared_data/KeyFrameTrajectory.txt" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error running ORB-SLAM3: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
