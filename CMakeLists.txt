cmake_minimum_required(VERSION 3.16)
project(ORB_SLAM3_LIB)

#  ORB-SLAM3 requires C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# RPi5 is Cortex-A76 (BCM2712) -- target it explicitly so GCC emits NEON/SIMD
# codegen instead of a generic armv8-a baseline.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -mcpu=cortex-a76 -mtune=cortex-a76)
endif()

# ORB-SLAM3 was cloned to /ORB_SLAM3 at container root (see Dockerfile),
# NOT under this shared_data mount.
set(ORB_SLAM3_DIR /ORB_SLAM3)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(depthai CONFIG REQUIRED)

include_directories(
    ${ORB_SLAM3_DIR}
    ${ORB_SLAM3_DIR}/include
    ${ORB_SLAM3_DIR}/include/CameraModels
    ${ORB_SLAM3_DIR}/Thirdparty/Sophus
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
)

# Common ORB-SLAM3 link set shared by every executable in this project.
set(ORB_SLAM3_LIBS
    ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so
    ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib/libg2o.so
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    -lboost_system
    -lboost_serialization
    -lcrypto
)

add_executable(SLAM SIMU_OAKD.cpp)
target_link_libraries(SLAM ${ORB_SLAM3_LIBS} depthai::core)