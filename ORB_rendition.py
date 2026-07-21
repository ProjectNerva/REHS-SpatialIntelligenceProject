import sys
import os
import colorsys
import open3d as o3d
import numpy as np

mp = sys.argv[1]
kft = sys.argv[2]

# load the MapPoints PLY. IMUS_OD_DB.cpp now writes per-vertex color keyed on which of the
# Atlas's (possibly many, disconnected) maps each point belongs to -- do NOT overwrite that
# with a uniform color here, or the fragmentation the .ply is encoding becomes invisible.
map_points = o3d.io.read_point_cloud(mp)

# load and parse the KeyFrameTrajectory.txt
# extract x, y, z columns (assuming columns 1,2,3)
traj_points = np.empty((0, 3))
if os.path.exists(kft) and os.path.getsize(kft) > 0:
    # ndmin=2 keeps a single-keyframe file 2-D instead of collapsing to 1-D
    traj_data = np.loadtxt(kft, ndmin=2)
    traj_points = traj_data[:, 1:4]

# create a line set for the trajectory
lines = [[i, i+1] for i in range(len(traj_points) - 1)]

# SaveKeyFrameTrajectoryTUM() (in the ORB-SLAM3 fork itself) also aggregates keyframes from
# every map the Atlas ever created, same as the .ply above -- but it has no per-keyframe map-id
# to color by. Approximate it instead: keyframes from unrelated, disconnected maps show up as
# an implausibly large jump versus normal frame-to-frame motion, so segment on outlier jumps
# and color each segment separately. This has no access to the true map-id, so segment
# boundaries are inferred, not exact -- but it makes fragmentation visible instead of hiding it
# behind one uniform trajectory color.
def segment_colors(points):
    if len(points) < 2:
        return []
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    median = np.median(dists)
    threshold = max(median * 8.0, 0.5)  # meters; floor guards against a near-zero median
    segment_id = 0
    seg_ids = []
    for d in dists:
        if d > threshold:
            segment_id += 1
        seg_ids.append(segment_id)
    n_segments = segment_id + 1
    palette = [colorsys.hsv_to_rgb((i * 137.508 % 360) / 360.0, 1.0, 1.0) for i in range(n_segments)]
    if n_segments > 1:
        print(f"Trajectory: {n_segments} disconnected segments detected (map resets that never merged back together)")
    return [list(palette[s]) for s in seg_ids]

colors = segment_colors(traj_points)

traj_line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(traj_points),
    lines=o3d.utility.Vector2iVector(lines)
)

traj_line_set.colors = o3d.utility.Vector3dVector(colors)

# draw_geometries()'s scroll-wheel zoom is hard-clamped inside Open3D's C++
# ViewControl (zoom factor bounded to roughly [0.02, 2.0]), so it stops
# responding past a point in either direction. camera_local_translate()
# instead moves the camera itself along its view direction with no such
# clamp, so bind zoom keys to it for effectively unlimited zoom; widen the
# near/far clipping planes too so extreme-close/far views don't get clipped.
ZOOM_STEP = 0.5

def zoom_in(vis):
    ctr = vis.get_view_control()
    ctr.set_constant_z_near(0.0001)
    ctr.set_constant_z_far(1000000.0)
    ctr.camera_local_translate(ZOOM_STEP, 0, 0)
    return False

def zoom_out(vis):
    ctr = vis.get_view_control()
    ctr.set_constant_z_near(0.0001)
    ctr.set_constant_z_far(1000000.0)
    ctr.camera_local_translate(-ZOOM_STEP, 0, 0)
    return False

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="ORB-SLAM3 Map and Trajectory", width=800, height=600)
vis.add_geometry(map_points)
vis.add_geometry(traj_line_set)

# '=' / '-' zoom in/out with no limit (hold or repeat-press to keep going);
# the normal mouse scroll wheel still works too, just within Open3D's
# built-in bounded range.
vis.register_key_callback(ord("="), zoom_in)
vis.register_key_callback(ord("-"), zoom_out)

# Default to a top-down (floorplan-style) view instead of Open3D's generic default.
# ORB-SLAM3 exports points in camera-frame convention (X-right, Y-down, Z-forward), so
# "looking from above" means positioning the camera along -Y and mapping the trajectory's
# forward axis (Z) to on-screen "up". lookat/zoom were already set by add_geometry()'s
# automatic bounding-box framing above, so only front/up need overriding here.
ctr = vis.get_view_control()
ctr.set_front([0, -1, 0])
ctr.set_up([0, 0, 1])

vis.run()
vis.destroy_window()