import depthai as dai
import numpy as np
import sys
from pathlib import Path

INPUT_W = int(sys.argv[1])
INPUT_H = int(sys.argv[2])

output_file = Path(f"Extrinsics/oak_d_eeprom_{INPUT_W}x{INPUT_H}.txt")
output_file.parent.mkdir(parents=True, exist_ok=True)

with dai.Device() as device, open(output_file, 'w') as f:
    def emit(*args):
        print(*args)
        print(*args, file=f)

    calib_data = device.readCalibration()

    # Define your target resolution (intrinsics change based on width/height)
    W, H = INPUT_W, INPUT_H

    # intrinsics (fx, fy, cx, cy) for Left (CAM_B) and Right (CAM_C)
    # returns a 3x3 matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    left_intrinsics = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, W, H)
    right_intrinsics = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, W, H)

    emit("Left Camera Intrinsics (Matrix):\n", np.array(left_intrinsics))
    emit(f"Left fx: {left_intrinsics[0][0]}, cx: {left_intrinsics[0][2]}")

    # stereo extrinsics (T_c1_c2: Transformation Right -> Left)
    # returns a 4x4 Transformation Matrix [R | T]
    T_c1_c2 = calib_data.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_B)
    emit("\nStereo T_c1_c2 Extrinsics:\n", np.array(T_c1_c2))

    # IMU Extrinsics (IMU_T_b_c1: Transformation IMU -> Left Camera)
    try:
        # returns 4x4 matrix from IMU to Left Camera (CAM_B)
        IMU_T_b_c1 = calib_data.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_B)
        emit("\nIMU IMU_T_b_c1 Extrinsics:\n", np.array(IMU_T_b_c1))
    except Exception as e:
        emit("\nCould not read IMU Extrinsics. Ensure your OAK-D model has an onboard IMU.")

    # Distortion coefficients for Left (CAM_B) and Right (CAM_C)
    # Perspective model order: [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, taux, tauy]
    left_distortion = calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B)
    right_distortion = calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C)
    emit("\nLeft Distortion Coefficients [k1,k2,p1,p2,k3,...]:\n", np.array(left_distortion))
    emit("Right Distortion Coefficients [k1,k2,p1,p2,k3,...]:\n", np.array(right_distortion))

    # IMU noise parameters (Allan-variance-derived). ORB-SLAM3's IMU noise model is scalar,
    # not per-axis, so average x/y/z here the same way the working OAK_D.yaml already does.
    try:
        imu_noise = calib_data.getImuNoiseParameters()
        accel, gyro = imu_noise.accelerometer, imu_noise.gyroscope

        noise_acc = (accel.x.noiseDensity + accel.y.noiseDensity + accel.z.noiseDensity) / 3.0
        noise_gyro = (gyro.x.noiseDensity + gyro.y.noiseDensity + gyro.z.noiseDensity) / 3.0
        acc_walk = (accel.x.randomWalk + accel.y.randomWalk + accel.z.randomWalk) / 3.0
        gyro_walk = (gyro.x.randomWalk + gyro.y.randomWalk + gyro.z.randomWalk) / 3.0

        emit(f"\nIMU Noise Parameters (IMU: {imu_noise.name}, averaged across x/y/z):")
        emit(f"  IMU.NoiseAcc:  {noise_acc}")
        emit(f"  IMU.NoiseGyro: {noise_gyro}")
        emit(f"  IMU.AccWalk:   {acc_walk}")
        emit(f"  IMU.GyroWalk:  {gyro_walk}")
    except Exception as e:
        emit(f"\nCould not read IMU noise parameters: {e}")

    emit(f"\nSaved to: {output_file.absolute()}")
