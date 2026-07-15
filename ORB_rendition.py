import sys
import os
import open3d as o3d
import numpy as np

mp = sys.argv[1]
kft = sys.argv[2]

# load the MapPoints PLY
map_points = o3d.io.read_point_cloud(mp)
if os.path.exists(mp) and os.path.getsize(mp) > 0:
    map_points.paint_uniform_color([0.5, 0.5, 0.5]) # gray for the map

# load and parse the KeyFrameTrajectory.txt
# extract x, y, z columns (assuming columns 1,2,3)
traj_points = np.empty((0, 3))
if os.path.exists(kft) and os.path.getsize(kft) > 0:
    # ndmin=2 keeps a single-keyframe file 2-D instead of collapsing to 1-D
    traj_data = np.loadtxt(kft, ndmin=2)
    traj_points = traj_data[:, 1:4]

# create a line set for the trajectory
lines = [[i, i+1] for i in range(len(traj_points) - 1)]
colors = [[1, 0, 0] for _ in range(len(lines))] # red lines for the trajectory

traj_line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(traj_points),
    lines=o3d.utility.Vector2iVector(lines)
)

traj_line_set.colors = o3d.utility.Vector3dVector(colors)

# visualize both together
o3d.visualization.draw_geometries([map_points, traj_line_set], window_name="ORB-SLAM3 Map and Trajectory", width=800, height=600)