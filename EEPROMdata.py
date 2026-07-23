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

    emit("\nRight Camera Intrinsics (Matrix):\n", np.array(right_intrinsics))
    emit(f"Right fx: {right_intrinsics[0][0]}, cx: {right_intrinsics[0][2]}")

    # Stereo extrinsics: getCameraExtrinsics(src, dst) per DepthAI's own docstring returns the
    # transform with src as origin -- so (CAM_B, CAM_C) is left->right, matching Stereo.T_c1_c2's
    # "CAM_B -> CAM_C" label directly (no direction ambiguity, confirmed against the installed
    # depthai package's docstring). Requested directly in METERs so it can be pasted into the
    # yaml with no manual cm->m conversion (a past transcription step that had no check on it).
    #
    # useSpecTranslation compares board-spec/CAD translation (True) against this unit's actual
    # loaded calibration (False, what the yaml should use). If they match, this unit never got
    # an individually-measured translation and you're relying on the nominal design value.
    T_spec = calib_data.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C, True, dai.LengthUnit.METER)
    T_calib = calib_data.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C, False, dai.LengthUnit.METER)
    stereo_matches = np.allclose(T_spec, T_calib)
    emit("\nStereo Extrinsics CAM_B->CAM_C, useSpecTranslation=False [USE THIS for Stereo.T_c1_c2, meters]:\n", np.array(T_calib))
    emit("\nStereo Extrinsics CAM_B->CAM_C, useSpecTranslation=True [board-spec/CAD reference, meters]:\n", np.array(T_spec))
    emit(f"\nStereo: spec vs per-unit calibration {'MATCH -- no individual stereo calibration on this unit' if stereo_matches else 'DIFFER -- per-unit stereo calibration confirmed'}")

    # IMU extrinsics: getCameraToImuExtrinsics(CAM_B) is documented by DepthAI as "rotation and
    # translation from the camera to IMU" -- exactly ORB-SLAM3's IMU.T_b_c1 convention (transform
    # a point from the camera frame into the body/IMU frame). Not a guess between two candidates;
    # this is the correct call by definition. Same spec-vs-calibrated comparison as above.
    try:
        imu_spec = calib_data.getCameraToImuExtrinsics(dai.CameraBoardSocket.CAM_B, True, dai.LengthUnit.METER)
        imu_calib = calib_data.getCameraToImuExtrinsics(dai.CameraBoardSocket.CAM_B, False, dai.LengthUnit.METER)
        imu_matches = np.allclose(imu_spec, imu_calib)
        emit("\nIMU Extrinsics getCameraToImuExtrinsics(CAM_B), useSpecTranslation=False [USE THIS for IMU.T_b_c1, meters]:\n", np.array(imu_calib))
        emit("\nIMU Extrinsics getCameraToImuExtrinsics(CAM_B), useSpecTranslation=True [board-spec/CAD reference, meters]:\n", np.array(imu_spec))
        emit(f"\nIMU: spec vs per-unit calibration {'MATCH -- IMU translation is the nominal board-design value, not individually measured' if imu_matches else 'DIFFER -- per-unit IMU calibration confirmed'}")
    except Exception as e:
        emit(f"\nCould not read getCameraToImuExtrinsics: {e}")

    # FOV as stored in EEPROM -- the actual ground truth for the lens's field of view, so a
    # >180 degree claim can be checked against the device itself instead of a spec sheet.
    try:
        left_fov = calib_data.getFov(dai.CameraBoardSocket.CAM_B)
        right_fov = calib_data.getFov(dai.CameraBoardSocket.CAM_C)
        emit(f"\nLeft FOV (diagonal, from EEPROM): {left_fov}")
        emit(f"Right FOV (diagonal, from EEPROM): {right_fov}")
    except Exception as e:
        emit(f"\nCould not read getFov: {e}")

    # Query the actual distortion model rather than assuming it -- Perspective and Fisheye
    # coefficient arrays use different orders/lengths, so this must be checked, not inferred.
    # There is no separate call to fetch "the other model's" coefficients: the device was
    # calibrated with exactly one of these, and that's the only one with real data behind it.
    left_model = calib_data.getDistortionModel(dai.CameraBoardSocket.CAM_B)
    right_model = calib_data.getDistortionModel(dai.CameraBoardSocket.CAM_C)
    emit(f"\nLeft Distortion Model: {left_model}")
    emit(f"Right Distortion Model: {right_model}")

    # Perspective model order: [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, taux, tauy]
    # Fisheye (ORB-SLAM3 KannalaBrandt8) model order: [k1, k2, k3, k4]
    left_distortion = calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B)
    right_distortion = calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C)
    emit("\nLeft Distortion Coefficients:\n", np.array(left_distortion))
    emit("Right Distortion Coefficients:\n", np.array(right_distortion))

    def emit_kb8_yaml(label, model, fx, fy, cx, cy, coeffs):
        if "FISHEYE" not in str(model).upper():
            emit(f"\n{label}: distortion model is {model}, not Fisheye -- no KannalaBrandt8 "
                 f"data exists on this unit (recalibrate with a fisheye routine to get real k1-k4).")
            return
        k1, k2, k3, k4 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        emit(f"\n{label} -- ready-to-paste KannalaBrandt8 yaml block:")
        emit(f"  Camera.fx: {fx}")
        emit(f"  Camera.fy: {fy}")
        emit(f"  Camera.cx: {cx}")
        emit(f"  Camera.cy: {cy}")
        emit(f"  Camera.k1: {k1}")
        emit(f"  Camera.k2: {k2}")
        emit(f"  Camera.k3: {k3}")
        emit(f"  Camera.k4: {k4}")

    emit_kb8_yaml("Camera1 (CAM_B)", left_model, left_intrinsics[0][0], left_intrinsics[1][1],
                  left_intrinsics[0][2], left_intrinsics[1][2], left_distortion)
    emit_kb8_yaml("Camera2 (CAM_C)", right_model, right_intrinsics[0][0], right_intrinsics[1][1],
                  right_intrinsics[0][2], right_intrinsics[1][2], right_distortion)

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
