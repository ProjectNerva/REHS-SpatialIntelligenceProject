# Preventing Local Map Resets(Draft)
1. Smooth, continuous motion
  - Moderate speed relative to the set FPS capture rate(basically avoid excessive frame-to-frame displacement)
2. Include translational motion, not just pure rotation in order to have the IMU observe scale/gravity
3. Even lighting while avoiding glare and underexposure
4. Avoid blank, featureless surfaces
5. Correct IMU noise parameters
6. Higher *ORBextractor.nFeatures* value
7. Holding still 2s before moving at the start for clean bias estimates(integrated within the code)
8. Have accurate camera-IMU extrinsics
9. Have accurate stereo/camera intrinsics(EEPROM data)
10. Higher capture FPS
11. Revisiting mapped areas, let tracking relocalize instead of resetting