# SLAM Filming Guide

Grew out of diagnosing `run8`'s map fragmentation (2026-07-21): 50 disconnected map resets in a
~68s recording, traced to the IMU initializer repeatedly failing with `not enough acceleration` /
`Not enough motion for initializing. Reseting...` (60 occurrences combined) far more often than
plain visual tracking failure (`Fail to track local map!`, 20 occurrences). Root cause: the
recording's motion didn't sustain enough translational acceleration for long enough. The numbers
below are measured from that investigation, not generic advice -- see [[euroc-test-harness]] /
[[orbslam3-docker-env]] for the tooling this was diagnosed with.

## Why this matters: the 15-second target

ORB-SLAM3's IMU initialization happens in two stages, gated by continuous elapsed time on a
single map (measured empirically from this fork's own log output, matches stock defaults):
- **VIBA 1** (~5s of continuous good tracking after map creation): first bias/gravity refinement.
- **VIBA 2** (~15s of continuous good tracking): the map is now "mature" -- from this point on,
  losing tracking triggers *relocalization* against the existing map instead of a full reset.

Until a map reaches VIBA 2, every tracking loss is treated as fatal and the system starts a brand
new map from scratch. Across every run filmed so far, no map has survived past ~8.5s -- meaning
relocalization has never once had the chance to prove it works. **The single goal of a good
recording is sustaining a map past 15 continuous seconds at least once.**

## Before you start recording

1. Even, diffuse lighting -- avoid glare and underexposure. No exposure control is implemented
   in `oakd_recorder.cpp` yet (open item, not a filming technique), so auto-exposure is at the
   mercy of whatever's in frame -- avoid pointing at bright windows/lights while starting.
2. Frame the shot to avoid blank, featureless surfaces (bare walls, ceilings, floors) filling
   the view -- ORB needs texture to detect and match features at all.
3. Confirm the OAK-D didn't crash on the previous run (`Device crashed, but no crash dump could
   be extracted` in the console) before starting a new one -- an unstable USB power supply causes
   this and will corrupt the recording regardless of how well it's filmed.

## While filming

4. **Hold still for the first 2 seconds.** This is enforced by the code's own init gate
   (`kHoldStillSec`) for a clean bias estimate -- console prints `[GATE] hold still` until this
   clears.
5. **The moment `[GATE] motion confirmed` prints, start continuous, deliberate translational
   motion -- walking or side-stepping, not panning/rotating in place.** Pure rotation lets the
   IMU observe orientation change but not scale/gravity, which is what stalls initialization.
6. **Sustain that motion for at least 15-20 uninterrupted seconds** right at the start, before
   doing anything else -- this is the window that needs to clear VIBA 2. Treat it like a
   dedicated "calibration walk," not just a warm-up.
7. **Never hold still or pure-pan for more than ~2-3 seconds at a stretch, for the entire
   recording, not just the start.** Measured from `run8`'s actual IMU data: the code's own motion
   gate checks a 0.3 m/s² acceleration std-dev threshold, and 28% of that recording's 1-second
   windows fell below it -- including two sustained ~5-7 second dead patches mid-recording. A
   patch that long is enough to stall or kill an in-progress IMU initialization even if visual
   tracking is going fine.
8. Keep motion smooth, not jerky -- avoid whip-fast rotations or sudden stops, especially in the
   first few seconds after any tracking loss, when the map is freshly reset and has very few
   points to match against.
9. Moderate speed relative to capture FPS -- avoid large frame-to-frame displacement. At 30fps,
   a good rule of thumb: don't let anything in frame move more than roughly a third of the image
   width between consecutive frames.
10. Periodically revisit already-mapped areas rather than only pushing into novel space -- gives
    relocalization something to match against if tracking does drop, and is what actually
    exercises loop closure.

## Already handled in code (not filming technique, listed for context)

- Camera-IMU extrinsics: proven correct via DepthAI's per-unit calibration, not a guess --
  see `OAK_D.yaml` / `OAK_D_1280x720.yaml` comments and `Extrinsics/`.
- Stereo/camera intrinsics: same, pulled directly at each pipeline's native capture resolution.
- IMU noise parameters: from the factory Allan-variance calibration, not generic defaults.
- 2s hold-still-then-motion init gate: implemented in `IMUS_OD.cpp` / `IMUS_OD_DB.cpp`.
- `ORBextractor.nFeatures: 2000` -- reasonable already; raising it further trades CPU headroom
  (relevant live on the RPi5, less so replaying recordings in the dev container) for a small
  robustness gain, not yet tested against whether it moves the needle here.

## Still open

- Exposure control (`oakd_recorder.cpp` doesn't cap auto-exposure) -- motion blur under indoor
  lighting during fast motion is a plausible remaining contributor, untested in isolation.
- `Stereo.ThDepth: 40.0` is still the generic ORB-SLAM3 example default, not a measured value
  for this rig.

## Validating a recording before running it through SLAM

Cheaper than a full SLAM run: check whether the raw IMU data actually clears the motion bar
before spending time on tracking/mapping analysis. From `mav0/imu0/data.csv`, compute the
acceleration-magnitude std-dev in 1-second windows across the clip and look for any 15+ second
stretch where every window stays above ~0.3 m/s². If no such stretch exists, the recording won't
produce a mature map regardless of how good visual tracking is -- re-film instead of debugging
further down the pipeline.
