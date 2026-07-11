import open3d as o3d
import numpy as np

# load the MapPoints PLY
map_points = o3d.io.read_point_cloud("shared_data/MapPoints.ply")
map_points.paint_uniform_color([0.5, 0.5, 0.5]) # gray for the map

# load and parse the KeyFrameTrajectory.txt
traj_data = np.loadtxt("shared_data/KeyFrameTrajectory.txt")
# extract x, y, z columns (assuming columns 1,2,3)
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