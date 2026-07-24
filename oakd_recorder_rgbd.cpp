// oakd_recorder_rgbd.cpp
//
// Records OAK-D on-device stereo depth + the rectified left image into this project's plain
// RGBD layout (see RGBD_OD_DB.cpp), for offline iteration on RGBD_OD_DB.cpp/OAK_D_RGBD.yaml
// without needing the live camera each time. Sibling to oakd_recorder.cpp (which records raw
// stereo + IMU in EuRoC format for the stereo-inertial pipeline) -- this one is simpler because
// plain RGBD mode has no IMU at all: no hold-still gate, no bias/motion stats, nothing to wait
// on before it's safe to move.
//
// Output layout (matches RGBD_OD_DB.cpp's LoadTimestamps()/main() expectations):
//   <out>/rgb/<ts_ns>.png     rectified left mono image
//   <out>/depth/<ts_ns>.png   16-bit depth map in millimeters, aligned to rgb/
//   <out>/timestamps.txt      one frame ts_ns per line
//
// Usage:
//   ./oakd_recorder_rgbd <output_dir>
//
// Ctrl+C to stop -- SIGINT is caught so files are flushed/closed cleanly instead of being cut
// off mid-write.
//
// UNVERIFIED: same caveat as RGBD_OD.cpp -- this environment has no depthai-core installation
// to compile/check against (confirmed when RGBD_OD.cpp was written: dpkg/filesystem search
// both came up empty, and the repo root isn't even mounted into the running container). The
// dai::node::StereoDepth calls below are written against the documented v3 API and the same
// Sync/createOutputQueue pattern already proven (compiled + run) in oakd_recorder.cpp and
// RGBD_OD_DB.cpp, but this file itself hasn't been built -- treat it as a first draft to fix
// against the real SDK on the Pi, not working code.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// ---- fixed capture parameters (matches OAK_D_RGBD.yaml's Camera.width/height/fps) ----
constexpr float kCameraFps = 30.0f;
constexpr uint32_t kWidth = 1280;
constexpr uint32_t kHeight = 720;

std::atomic<bool> g_stop{false};
void onSigint(int) { g_stop = true; }

// Converts a depthai steady_clock timestamp to integer nanoseconds since epoch, matching the
// <ts_ns> filename/timestamps.txt convention (same as oakd_recorder.cpp).
template <typename TimePoint>
int64_t toNs(const TimePoint& tp) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}

// One rgb+depth pair queued for disk write. Mats are cloned before queuing (depthai may reuse
// the underlying buffer once the pipeline's message is released), so the writer thread owns
// independent copies and never touches pipeline-owned memory. Identical pattern to
// oakd_recorder.cpp's WriteJob/FrameWriter -- copied rather than shared to keep both recorders
// standalone and independently buildable.
struct WriteJob {
    fs::path rgbPath, depthPath;
    cv::Mat rgb, depth;
};

// Runs on its own thread so PNG encode + disk I/O never blocks the capture loop from draining
// the next frame -- same fix applied to oakd_recorder.cpp after run15/16/18's frameDelta gaps
// traced back to the blocking imwrite calls it used to make inline.
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

            cv::imwrite(job.rgbPath.string(), job.rgb);
            cv::imwrite(job.depthPath.string(), job.depth);
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
        std::cerr << "Usage: ./oakd_recorder_rgbd <output_dir>" << std::endl;
        return 1;
    }
    std::signal(SIGINT, onSigint);

    const fs::path outDir(argv[1]);
    const fs::path rgbDir = outDir / "rgb";
    const fs::path depthDir = outDir / "depth";
    fs::create_directories(rgbDir);
    fs::create_directories(depthDir);

    std::ofstream tsFile((outDir / "timestamps.txt").string());
    if (!tsFile) {
        std::cerr << "Failed to open output files under " << outDir << std::endl;
        return 1;
    }

    // ---- pipeline setup (mirrors RGBD_OD.cpp's live-tracking pipeline) ----
    dai::Pipeline pipeline;

    auto camLeft = pipeline.create<dai::node::Camera>();
    auto camRight = pipeline.create<dai::node::Camera>();
    camLeft->build(dai::CameraBoardSocket::CAM_B, std::make_pair(kWidth, kHeight), kCameraFps);
    camRight->build(dai::CameraBoardSocket::CAM_C, std::make_pair(kWidth, kHeight), kCameraFps);

    // Same auto-exposure cap as oakd_recorder.cpp / RGBD_OD.cpp -- confirmed fix for
    // window/light blowout starving ORB of features (run11/run12), applies here just as much.
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

    auto rgbdSync = pipeline.create<dai::node::Sync>();
    rgbdSync->setSyncThreshold(std::chrono::milliseconds(static_cast<int64_t>(1000.0 / (2.0 * kCameraFps))));
    stereo->rectifiedLeft.link(rgbdSync->inputs["rgb"]);
    stereo->depth.link(rgbdSync->inputs["depth"]);

    auto rgbdQueue = rgbdSync->out.createOutputQueue();

    pipeline.start();
    std::cout << "Recording RGBD to " << outDir << " -- no hold-still gate needed (no IMU)."
              << " Press Ctrl+C to stop, or 'q' in the preview window." << std::endl;

    cv::namedWindow("oakd_recorder_rgbd preview", cv::WINDOW_NORMAL);

    // Owns the background PNG-encode/write thread for this recording's duration -- reset()
    // just before exit blocks until every queued frame is flushed, so nothing is lost.
    auto frameWriter = std::make_unique<FrameWriter>();

    int64_t frameCount = 0;

    while (pipeline.isRunning() && !g_stop) {
        auto msgGroup = rgbdQueue->tryGet<dai::MessageGroup>();
        if (!msgGroup) continue;
        auto inRgb = msgGroup->get<dai::ImgFrame>("rgb");
        auto inDepth = msgGroup->get<dai::ImgFrame>("depth");
        if (!inRgb || !inDepth) continue;

        int64_t tNs = toNs(inRgb->getTimestamp());
        cv::Mat rgb = inRgb->getCvFrame();
        cv::Mat depth = inDepth->getCvFrame();

        frameWriter->enqueue(WriteJob{rgbDir / (std::to_string(tNs) + ".png"),
                                      depthDir / (std::to_string(tNs) + ".png"),
                                      rgb.clone(), depth.clone()});
        tsFile << tNs << '\n';
        ++frameCount;

        // ---- preview window ----
        cv::Mat preview;
        cv::cvtColor(rgb, preview, cv::COLOR_GRAY2BGR);
        char countLine[64];
        std::snprintf(countLine, sizeof(countLine), "frames=%lld", (long long)frameCount);
        cv::putText(preview, countLine, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 220, 0), 2);

        cv::imshow("oakd_recorder_rgbd preview", preview);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') g_stop = true;

        if (frameCount % 30 == 0) {
            std::cout << "\rFrames: " << frameCount << std::flush;
        }
    }

    cv::destroyAllWindows();

    std::cout << "\nStopping..." << std::endl;
    pipeline.stop();

    // Blocks until every queued frame has actually been written -- without this, stopping
    // right after capture ends could truncate whatever's still sitting in the write queue.
    std::cout << "Flushing remaining queued frames..." << std::endl;
    frameWriter.reset();

    tsFile.close();

    std::cout << "Done. Wrote " << frameCount << " RGBD frame pairs to " << outDir << std::endl;
    return 0;
}
