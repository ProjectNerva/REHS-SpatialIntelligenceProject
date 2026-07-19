# IMU Initialization: Motion Requirements and the Init Gate

## The problem

ORB-SLAM3's stereo-inertial mode needs to solve for three things before it can produce
a metrically-correct map: **gravity direction**, **accelerometer/gyro bias**, and
**metric scale** (pure vision alone is scale-ambiguous — it can't tell "small object
close" from "big object far"). All three come from the IMU stream, and how the operator
moves the camera while recording directly determines whether they're solvable at all.

The raw accelerometer model is:

```
accel_measured = R^T * (a_true - g) + bias + noise
```

- `g` (gravity) is a large, constant vector.
- `bias` is a slowly-drifting offset.
- `a_true` (real linear acceleration) is what we actually want to observe.
- `noise` is sensor noise, sized by `IMU.NoiseAcc` / `IMU.NoiseGyro` in `OAK_D.yaml`.

### Too static (not enough acceleration)

If the camera barely moves, `a_true ≈ 0`, so `accel_measured ≈ -R^T*g + bias + noise`.
Gravity and bias are now entangled in a single near-constant reading, and there's no
independent signal to separate them — more importantly, **scale is not just noisy, it's
mathematically undetermined**: nothing in the data can tell the optimizer what one unit
of visual displacement corresponds to in meters. This is why ORB-SLAM3's initializer
gates on acceleration variance and prints `"not enough acceleration"` until it clears a
threshold — averaging longer doesn't help scale, because there's no scale information
present to average.

Images are affected too: near-zero motion means near-zero parallax between consecutive
frames, so triangulated depth is highly uncertain even where feature matching succeeds.

### Too fast

- **Motion blur**: blur ≈ velocity × exposure time. At 15 fps the exposure window can be
  up to ~66 ms; fast motion smears the image enough that FAST corners (which ORB relies
  on) disappear entirely in the blurred region — not noisy features, *no* features.
- **Frame-to-frame baseline too large**: ORB's inter-frame matching assumes a small
  search window. A whip pan can leave frame N and N+1 barely overlapping.
- **IMU saturation**: the accelerometer/gyro have a fixed measurable range. A sharp
  enough motion clips the reading — a wrong-but-plausible value gets written, which is
  worse than a gap because it isn't obviously invalid.
- **Preintegration validity**: ORB-SLAM3 preintegrates IMU samples across each camera
  interval assuming roughly constant bias/dynamics within that window; violent motion
  inside one interval breaks that assumption for that frame.

## What's fixable vs. what isn't

**Gravity direction and bias** are recoverable by holding still and averaging: noise
shrinks with `1/√N` over N static samples, so a deliberate stillness period at the start
gives the initializer a clean reference before any motion noise mixes in. This is a
standard VIO convention (ARKit/ARCore/VINS-Mono use the same idea).

**Scale is not fixable by processing at all.** It requires an actual observed
translational-acceleration event to correlate against observed visual displacement.
No filter, no averaging window, and no amount of waiting substitutes for real motion —
it's a structural observability gap, not a noise problem.

**Net requirement**: the recording needs *one* adequately-excited segment, not
continuous motion throughout. Once gravity/bias/scale are solved, steady-state tracking
tolerates brief stillness fine (preintegration correctly predicts near-zero motion).

## Design: the init gate in `IMUS_OD.cpp`

A live operator-guidance gate, not a data filter:

1. **0–2 s (hold-still window, `kHoldStillSec`)**: prints `[GATE] hold still -- Ns`,
   giving the averaging described above a clean static segment to work with.
2. **After 2 s, until excitation is seen**: prints `[GATE] move the camera now`.
3. **Once excited**: prints `[GATE] motion confirmed` once, then stops checking for
   the rest of the run (no per-frame overhead after that).

Excitation is computed per-frame from the same `vImuMeas` already bucketed for
`TrackStereo` — no separate buffering:

```
gyro_rms  = sqrt(mean(|w|^2))   over the frame's IMU samples
accel_std = std(|a|)            over the frame's IMU samples
excited   = gyro_rms >= kGyroRmsMin (0.05 rad/s) OR accel_std >= kAccelStdMin (0.3 m/s^2)
```

**The gate never blocks or alters `TrackStereo`.** Every frame and IMU sample is still
processed and fed to ORB-SLAM3 exactly as before — the gate only prints guidance text.
ORB-SLAM3's own internal initializer is the real gatekeeper for when tracking actually
starts; this just tells the operator *why* it might be waiting.

`oakd_recorder.cpp` intentionally does **not** have this gate — it's a raw capture tool
and must record unconditionally so a replay is a faithful copy of whatever happened,
including a bad take (so it can be diagnosed after the fact rather than papered over
during capture).

## Tuning

`kHoldStillSec`, `kGyroRmsMin`, `kAccelStdMin` are heuristics, not measured constants —
adjust them against this specific IMU/rig if the gate proves too eager or too lax in
practice.
