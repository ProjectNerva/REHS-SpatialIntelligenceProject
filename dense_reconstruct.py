import sys
import os
import re
import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

kft_path = sys.argv[1] if len(sys.argv) > 1 else "workspace/fr1out/KeyFrameTrajectory.txt"
data_dir = sys.argv[2] if len(sys.argv) > 2 else "workspace/fr1room_rgbd"
yaml_path = sys.argv[3] if len(sys.argv) > 3 else "workspace/TUM1_RGBD.yaml"
out_stem = sys.argv[4] if len(sys.argv) > 4 else "dense_room"

fx, fy, cx, cy, factor = 517.306408, 516.469215, 318.643040, 255.313989, 5000.0
if yaml_path and os.path.exists(yaml_path):
    txt = open(yaml_path).read()
    def g(key, default):
        m = re.search(rf"{re.escape(key)}\s*:\s*([-\d.eE]+)", txt)
        return float(m.group(1)) if m else default
    fx = g("Camera1.fx", fx); fy = g("Camera1.fy", fy)
    cx = g("Camera1.cx", cx); cy = g("Camera1.cy", cy)
    factor = g("RGBD.DepthMapFactor", factor)

MAX_DEPTH  = 4.0
KF_STRIDE  = 2
PIX_STRIDE = 3
CLOUD_MAX  = 700000
RENDER_MAX = 120000


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


ns_list = sorted(int(os.path.splitext(os.path.basename(p))[0])
                 for p in glob.glob(os.path.join(data_dir, "rgb", "*.png")))
ns_arr = np.array(ns_list)


def nearest_ns(t_sec):
    target = t_sec * 1e9
    i = np.searchsorted(ns_arr, target)
    cands = [j for j in (i-1, i) if 0 <= j < len(ns_arr)]
    return ns_list[min(cands, key=lambda j: abs(ns_arr[j]-target))]


kf = np.loadtxt(kft_path, ndmin=2)[::KF_STRIDE]

us = np.arange(0, 640, PIX_STRIDE)
vs = np.arange(0, 480, PIX_STRIDE)
uu, vv = np.meshgrid(us, vs)
uu = uu.ravel(); vv = vv.ravel()

all_pts, all_col = [], []
used = 0
for row in kf:
    t = row[0]; tx, ty, tz = row[1:4]; qx, qy, qz, qw = row[4:8]
    ns = nearest_ns(t)
    dpath = os.path.join(data_dir, "depth", f"{ns}.png")
    rpath = os.path.join(data_dir, "rgb", f"{ns}.png")
    if not (os.path.exists(dpath) and os.path.exists(rpath)):
        continue
    depth = np.asarray(Image.open(dpath)).astype(np.float32)
    rgb = np.asarray(Image.open(rpath).convert("RGB"))
    z = depth[vv, uu] / factor
    valid = (z > 0.1) & (z < MAX_DEPTH)
    if not valid.any():
        continue
    uq, vq, zq = uu[valid], vv[valid], z[valid]
    pc = np.stack([(uq-cx)*zq/fx, (vq-cy)*zq/fy, zq], axis=1)
    pw = pc @ quat_to_R(qx, qy, qz, qw).T + np.array([tx, ty, tz])
    all_pts.append(pw)
    all_col.append(rgb[vq, uq, :] / 255.0)
    used += 1

pts = np.concatenate(all_pts)
cols = np.concatenate(all_col)
print(f"Dense cloud: {len(pts):,} points from {used} keyframes")

rng = np.random.default_rng(0)
order = rng.permutation(len(pts))

pi = order[:CLOUD_MAX]
arr = np.column_stack([pts[pi], (cols[pi]*255).astype(int)])
header = ("ply\nformat ascii 1.0\nelement vertex %d\n"
          "property float x\nproperty float y\nproperty float z\n"
          "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header" % len(pi))
np.savetxt(out_stem + ".ply", arr, fmt="%.4f %.4f %.4f %d %d %d", header=header, comments="")
print(f"Saved {out_stem}.ply ({len(pi):,} pts)")

ri = order[:RENDER_MAX]
P, C = pts[ri], cols[ri]
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(P[:, 0], P[:, 2], -P[:, 1], c=C, s=0.6, linewidths=0, depthshade=False)
ax1.set_title("dense reconstruction — perspective"); ax1.set_axis_off()
ax1.view_init(elev=18, azim=-72)
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(P[:, 0], P[:, 2], c=C, s=0.6, linewidths=0)
ax2.set_aspect("equal"); ax2.set_axis_off(); ax2.set_title("dense reconstruction — top-down")
fig.suptitle(f"freiburg1_room RGBD dense map  |  {len(pts):,} points from {used} keyframes", fontsize=13)
fig.tight_layout()
fig.savefig(out_stem + ".png", dpi=200, facecolor="white")
print(f"Saved {out_stem}.png")
