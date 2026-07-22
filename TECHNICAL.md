# SLAM Filming Guide

Grew out of diagnosing `run8`'s map fragmentation (2026-07-21): 50 disconnected map resets in a
~68s recording, traced initially to `not enough acceleration` / `Not enough motion for
initializing. Reseting...` messages, then confirmed against this fork's actual source
(`/ORB_SLAM3/src/Tracking.cc`, `/ORB_SLAM3/src/LocalMapping.cc`, read 2026-07-22) rather than
inferred from log messages alone. See [[euroc-test-harness]] / [[orbslam3-docker-env]] for the
tooling this was diagnosed with.

## The real mechanism (confirmed from source, not inferred)

ORB-SLAM3 tracks a per-map clock called `mTinit` in `LocalMapping.cc`. **It is not wall-clock
time** -- it only advances when the map is actually moving:

```cpp
float dist = (KF->mPrevKF->GetCameraCenter() - KF->GetCameraCenter()).norm()
           + (KF->mPrevKF->mPrevKF->GetCameraCenter() - KF->mPrevKF->GetCameraCenter()).norm();
if (dist > 0.05)
    mTinit += KF->mTimeStamp - KF->mPrevKF->mTimeStamp;
```
`dist` is the real, metric camera-center displacement summed across the last two keyframe-to-
keyframe hops. Below 5cm of combined displacement, `mTinit` simply doesn't advance that step --
it pauses, it doesn't reset.

**The map-killing condition** is separate and much sharper:
```cpp
if (!GetIniertialBA2() && (mTinit < 10.f) && (dist < 0.02))
    // "Not enough motion for initializing. Reseting..." -- map destroyed
```
While `mTinit` is still under 10 seconds, if keyframe-to-keyframe displacement drops below **2cm**,
the entire map is destroyed outright -- not paused, killed. Once `mTinit` crosses 10s, this check
is skipped forever on that map (the guard is `mTinit<10.f`), so a map that survives its first 10
motion-seconds can no longer die this particular way.

Bias/gravity/scale refinement then happens in two stages, both gated on this same motion-time
`mTinit`, not on wall-clock survival:
- **VIBA 1** at `mTinit > 5.0f` -- matches what was empirically measured from log timestamps
  before this source was available (~5.1-5.9s), now explained: those maps happened to be moving
  consistently enough that mTinit tracked close to wall-clock time.
- **VIBA 2** at `mTinit > 15.0f` -- the map becomes "mature": from this point on, losing tracking
  triggers *relocalization* instead of a reset. No map filmed so far has reached this.

**The actionable rule this produces:** the first 10 seconds of accumulated *motion* time after
any reset are the fragile window. A single near-stop (under 2cm of keyframe-to-keyframe travel)
inside that window kills the map outright, resetting progress to zero. Survive it, and the
specific failure mode that has caused most resets so far can't happen again on that map -- the
remaining climb to 15s (VIBA 2) is real but no longer has this landmine in it.

This also explains `run10`'s best result: a map that survived 10.8 *wall-clock* seconds still
died via this exact message -- because wall-clock time isn't what was being measured. Some of
that lifespan was spent on keyframe hops under the 5cm bar (not accumulating `mTinit` at all), and
it eventually hit one under 2cm and died before `mTinit` itself ever reached 10.

### A secondary, minor mechanism: `not enough acceleration` / `not IMU meas`

These come from a *different* function (`Tracking::StereoInitialization()`), and only matter
while a map is trying to be (re-)born after a reset, not while one is alive and growing:
```cpp
if (mCurrentFrame.N > 500) {                              // needs 500+ ORB keypoints this frame
    if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
        // "not IMU meas" -- one of the two frames has no IMU preintegration yet, retry next frame
    if (!mFastInit && (curFrame.avgA - lastFrame.avgA).norm() < 0.5)
        // "not enough acceleration" -- consecutive frames' average accel vector barely changed
}
```
This retries every single frame while state is `NOT_INITIALIZED`, waiting for one frame pair
~33ms apart with enough of a jerk between them (a footstep, a small turn) and enough features
(>500). Seeing this print 40-70 times in a run sounds alarming but is really only ~1.5-2.5 seconds
of retries before it succeeds -- a minor, self-resolving delay, not the actual bottleneck. The
real bottleneck is the 10-second fragile window above.

## Before you start recording

1. Even, diffuse lighting -- avoid glare and underexposure. No exposure control is implemented
   in `oakd_recorder.cpp` yet (open item, not a filming technique), so auto-exposure is at the
   mercy of whatever's in frame -- avoid pointing at bright windows/lights while starting.
2. Frame the shot to avoid blank, featureless surfaces (bare walls, ceilings, floors) filling
   the view -- ORB needs texture to detect and match features at all. Confirmed directly: 3 of 5
   sampled `Fail to track local map!` frames from `run8` showed a blank wall, glare, or an
   extreme close-range flat surface filling most of the image.
3. Confirm the OAK-D didn't crash on the previous run (`Device crashed, but no crash dump could
   be extracted` in the console) before starting a new one -- an unstable USB power supply causes
   this and will corrupt the recording regardless of how well it's filmed.

## While filming

4. **Hold still for the first 2 seconds.** This is enforced by the code's own init gate
   (`kHoldStillSec`) for a clean bias estimate -- console prints `[GATE] hold still` until this
   clears.
5. **The moment `[GATE] motion confirmed` prints, start continuous, deliberate translational
   motion -- walking or side-stepping, not panning/rotating in place.** Pure rotation moves the
   camera center almost not at all, so it doesn't count as `dist` above.
6. **Whenever a reset happens (or at the very start), commit to at least 10 seconds of
   uninterrupted motion with no near-stop.** This is the actual fragile window, not a soft
   suggestion -- a single pause under ~2cm of travel between keyframes inside this window kills
   the map instantly and the 10-second countdown restarts from zero on the next map. Watch the
   `[DEBUG] Tracking state:` log for `NOT_INITIALIZED -> OK` transitions -- that's when the clock
   starts.
7. **After that first 10 seconds, keep going for a total of ~15-20s** to give VIBA 2 a chance --
   this part is real but comparatively forgiving, since the hard kill-switch is already behind you.
8. Keep motion smooth, not jerky -- avoid whip-fast rotations or sudden stops, especially in the
   first few seconds after any tracking loss, when the map is freshly reset and has very few
   points to match against.
9. Moderate speed relative to capture FPS -- avoid large frame-to-frame displacement. At 30fps,
   a good rule of thumb: don't let anything in frame move more than roughly a third of the image
   width between consecutive frames.
10. Periodically revisit already-mapped areas rather than only pushing into novel space -- gives
    relocalization something to match against if tracking does drop, and is what actually
    exercises loop closure (still untested -- no map has reached VIBA 2 yet).

## Already handled in code (not filming technique, listed for context)

- Camera-IMU extrinsics: proven correct via DepthAI's per-unit calibration, not a guess --
  see `OAK_D.yaml` / `OAK_D_1280x720.yaml` comments and `Extrinsics/`.
- Stereo/camera intrinsics: same, pulled directly at each pipeline's native capture resolution.
- IMU noise parameters: from the factory Allan-variance calibration, not generic defaults.
- 2s hold-still-then-motion init gate: implemented in `IMUS_OD.cpp` / `IMUS_OD_DB.cpp`.
- `Stereo.ThDepth: 40.0`: checked against the actual baseline (0.075m) and rectified focal
  length -- gives a 3.0m trusted-depth cutoff at ~7% relative error, consistent with how the
  stock EuRoC defaults were tuned for their own baseline. Confirmed reasonable, not changed.
- `ORBextractor.nFeatures: 2000` -- reasonable already; raising it further trades CPU headroom
  (relevant live on the RPi5, less so replaying recordings in the dev container) for a small
  robustness gain, not yet tested against whether it moves the needle here.

## Still open

- Exposure control (`oakd_recorder.cpp` doesn't cap auto-exposure) -- motion blur under indoor
  lighting during fast motion is a plausible remaining contributor, untested in isolation.
- Whether relocalization actually works at all once a map reaches VIBA 2 -- completely untested,
  since no map has gotten there yet.

## On pre-flight-checking a recording before running it through SLAM

Earlier draft of this guide proposed checking raw accelerometer std-dev against a threshold as a
cheap pre-check. That's now known to be the wrong signal -- the real criterion (`dist` between
estimated keyframe camera centers) is a property of the *tracked trajectory*, not the raw IMU
stream, and isn't computable without actually running the tracker. There isn't a cheap proxy for
this; the only real test is replaying the recording through `SLAM_DEBUG` and checking whether
`mTinit`-driven messages (`start VIBA 1/2`, `Not enough motion for initializing`) show what you'd
expect.
