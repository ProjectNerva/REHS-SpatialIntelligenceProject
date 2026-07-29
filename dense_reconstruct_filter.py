import sys
import os
import re
import glob
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

kft_path = sys.argv[1] if len(sys.argv) > 1 else "workspace/fr1out/KeyFrameTrajectory.txt"
data_dir = sys.argv[2] if len(sys.argv) > 2 else "workspace/fr1room_rgbd"
yaml_path = sys.argv[3] if len(sys.argv) > 3 else "workspace/TUM1_RGBD.yaml"
out_stem = sys.argv[4] if len(sys.argv) > 4 else "dense_room"
yolo_weights = sys.argv[5] if len(sys.argv) > 5 else "movingObjects.pt"

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

# --- dynamic-object masking + detection visualization ---
# Any pixel whose (u,v) falls inside a detected person/dog/cat/suitcase box is dropped
# before back-projection, so those points never enter the reconstructed cloud. Loaded once,
# outside the per-keyframe loop, since model loading is the expensive part.
DETECT_CONF = 0.5
yolo_model = YOLO(yolo_weights)

# Every keyframe's RGB image gets saved here with detection boxes drawn on it, so you can
# visually confirm what the detector actually caught rather than only trusting the point count.
detections_dir = out_stem + "_detections"
os.makedirs(detections_dir, exist_ok=True)

# Running per-class detection count across the whole sequence, for the summary printed at the end.
class_counts = {}


def dynamic_object_mask_and_annotate(rgb_img, img_h, img_w, ns):
    """
    Returns a boolean array of shape (img_h, img_w) -- True where a pixel falls inside a
    detected dynamic-object box, False elsewhere. Also saves an annotated copy of this frame
    (boxes + class labels drawn) to detections_dir, and updates the running class_counts tally.
    rgb_img must be a HxWx3 uint8 array (as read from the keyframe's rgb PNG).
    """
    mask = np.zeros((img_h, img_w), dtype=bool)
    results = yolo_model.predict(rgb_img, conf=DETECT_CONF, verbose=False)

    any_detections = False
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        any_detections = True

        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            mask[y1:y2, x1:x2] = True

        # Tally detections per class name for the end-of-run summary.
        for cls_idx in r.boxes.cls.cpu().numpy().astype(int):
            cls_name = r.names[cls_idx]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        # Ultralytics' own .plot() draws boxes + labels + confidence directly onto the frame,
        # returned as a BGR array ready for cv2.imwrite.
        annotated = r.plot()
        cv2.imwrite(os.path.join(detections_dir, f"{ns}.png"), annotated)

    if not any_detections:
        # Still save the plain frame so the detections folder has one image per keyframe,
        # making it easy to confirm "no detections" vs. "keyframe was skipped".
        cv2.imwrite(os.path.join(detections_dir, f"{ns}.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

    return mask


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

# Read actual image dimensions from the first available frame instead of assuming TUM's
# 640x480 -- needed so this also works correctly on the 1280x720 OAK-D recordings.
if len(ns_list) == 0:
    raise RuntimeError(f"No rgb/*.png files found under {data_dir}")
_sample_img = np.asarray(Image.open(os.path.join(data_dir, "rgb", f"{ns_list[0]}.png")))
IMG_H, IMG_W = _sample_img.shape[0], _sample_img.shape[1]
print(f"Detected image size: {IMG_W}x{IMG_H}")

us = np.arange(0, IMG_W, PIX_STRIDE)
vs = np.arange(0, IMG_H, PIX_STRIDE)
uu, vv = np.meshgrid(us, vs)
uu = uu.ravel(); vv = vv.ravel()

all_pts, all_col = [], []
used = 0
dropped_dynamic_points = 0
for row in kf:
    t = row[0]; tx, ty, tz = row[1:4]; qx, qy, qz, qw = row[4:8]
    ns = nearest_ns(t)
    dpath = os.path.join(data_dir, "depth", f"{ns}.png")
    rpath = os.path.join(data_dir, "rgb", f"{ns}.png")
    if not (os.path.exists(dpath) and os.path.exists(rpath)):
        continue
    depth = np.asarray(Image.open(dpath)).astype(np.float32)
    rgb = np.asarray(Image.open(rpath).convert("RGB"))

    dyn_mask = dynamic_object_mask_and_annotate(rgb, IMG_H, IMG_W, ns)
    dyn_mask_flat = dyn_mask[vv, uu]  # same sampled-grid indexing as z below

    z = depth[vv, uu] / factor
    valid = (z > 0.1) & (z < MAX_DEPTH) & (~dyn_mask_flat)
    dropped_dynamic_points += int(dyn_mask_flat.sum())
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
print(f"Dropped {dropped_dynamic_points:,} sampled points that fell inside detected dynamic objects")

print("\nDetection summary (count of detections per class across all keyframes):")
if class_counts:
    for cls_name, count in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cls_name}: {count}")
else:
    print("  No detections in any keyframe.")
print(f"\nAnnotated keyframe images (with boxes drawn) saved to: {detections_dir}/")

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
ax1.set_title("dense reconstruction — perspective (dynamic objects masked)"); ax1.set_axis_off()
ax1.view_init(elev=18, azim=-72)
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(P[:, 0], P[:, 2], c=C, s=0.6, linewidths=0)
ax2.set_aspect("equal"); ax2.set_axis_off(); ax2.set_title("dense reconstruction — top-down")
fig.suptitle(f"RGBD dense map (dynamic objects masked)  |  {len(pts):,} points from {used} keyframes", fontsize=13)
fig.tight_layout()
fig.savefig(out_stem + ".png", dpi=200, facecolor="white")
print(f"Saved {out_stem}.png")