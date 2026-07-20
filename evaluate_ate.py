#!/usr/bin/env python3
"""
Absolute Trajectory Error (ATE) between an ORB-SLAM3 KeyFrameTrajectory.txt
(TUM format: "timestamp_sec x y z qx qy qz qw") and a TUM-VI mocap0/data.csv
ground truth ("timestamp_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z").

Usage: python3 evaluate_ate.py <KeyFrameTrajectory.txt> <mocap0/data.csv> [--max-diff SEC]
"""
import sys
import argparse
import numpy as np


def load_estimate(path):
    """TUM format, seconds, qx qy qz qw (scalar-last)."""
    poses = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            t = float(parts[0])
            xyz = np.array([float(p) for p in parts[1:4]])
            poses[t] = xyz
    return poses


def load_groundtruth(path):
    """TUM-VI mocap0/data.csv: comma-separated, ns timestamps, qw qx qy qz (scalar-first)."""
    poses = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            t = float(parts[0]) * 1e-9  # ns -> sec, to match estimate's units
            xyz = np.array([float(p) for p in parts[1:4]])
            poses[t] = xyz
    return poses


def associate(est, gt, max_diff):
    """For each estimate timestamp, find the closest ground-truth timestamp within max_diff."""
    gt_times = sorted(gt.keys())
    gt_arr = np.array(gt_times)
    matches = []
    for t_est in sorted(est.keys()):
        idx = np.searchsorted(gt_arr, t_est)
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(gt_arr)]
        if not candidates:
            continue
        best = min(candidates, key=lambda i: abs(gt_arr[i] - t_est))
        if abs(gt_arr[best] - t_est) <= max_diff:
            matches.append((t_est, gt_times[best]))
    return matches


def umeyama_alignment(src, dst):
    """Rigid SE3 alignment (rotation + translation, no scaling -- stereo-inertial has true
    metric scale already, so we're only correcting for the arbitrary world-frame origin/
    orientation VIO starts from, not rescaling it). src/dst: Nx3 arrays, src aligned onto dst."""
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    S = np.diag([1, 1, d])
    R = Vt.T @ S @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('estimate', help='KeyFrameTrajectory.txt')
    ap.add_argument('groundtruth', help='mocap0/data.csv')
    ap.add_argument('--max-diff', type=float, default=0.02,
                     help='max timestamp difference (sec) for association (default: 0.02)')
    args = ap.parse_args()

    est = load_estimate(args.estimate)
    gt = load_groundtruth(args.groundtruth)

    matches = associate(est, gt, args.max_diff)
    if len(matches) < 3:
        print(f"Only {len(matches)} matched pairs (need >= 3 for alignment) -- "
              f"check timestamp units/overlap, or loosen --max-diff.")
        sys.exit(1)

    est_pts = np.array([est[t_e] for t_e, _ in matches])
    gt_pts = np.array([gt[t_g] for _, t_g in matches])

    R, t = umeyama_alignment(est_pts, gt_pts)
    aligned = (R @ est_pts.T).T + t

    errors = np.linalg.norm(aligned - gt_pts, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))

    print(f"Matched poses:   {len(matches)} / {len(est)} estimated keyframes "
          f"(within {args.max_diff}s of a ground-truth sample)")
    print(f"ATE RMSE:        {rmse:.4f} m")
    print(f"ATE mean:        {errors.mean():.4f} m")
    print(f"ATE median:      {np.median(errors):.4f} m")
    print(f"ATE std:         {errors.std():.4f} m")
    print(f"ATE min/max:     {errors.min():.4f} m / {errors.max():.4f} m")


if __name__ == '__main__':
    main()
