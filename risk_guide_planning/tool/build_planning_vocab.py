import argparse
import json
import pickle
from pathlib import Path

import numpy as np


YAW_DISTANCE_WEIGHT = 0.25


def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def get_ego_future_xy(data_infos, idx, sample_rate, frames):
    adj_idx_list = range(idx + sample_rate, idx + (frames + 1) * sample_rate, sample_rate)
    cur_frame = data_infos[idx]
    adj_track = np.zeros((frames, 3), dtype=np.float32)
    adj_mask = np.zeros((frames, 2), dtype=np.float32)
    world2lidar_cur = np.asarray(cur_frame['sensors']['LIDAR_TOP']['world2lidar'], dtype=np.float32)
    for j, adj_idx in enumerate(adj_idx_list):
        if adj_idx < 0 or adj_idx >= len(data_infos):
            break
        adj_frame = data_infos[adj_idx]
        if adj_frame['folder'] != cur_frame['folder']:
            break
        world2lidar_adj = np.asarray(adj_frame['sensors']['LIDAR_TOP']['world2lidar'], dtype=np.float32)
        adj2cur_lidar = world2lidar_cur @ np.linalg.inv(world2lidar_adj)
        xy = adj2cur_lidar[0:2, 3]
        yaw = np.arctan2(adj2cur_lidar[1, 0], adj2cur_lidar[0, 0])
        yaw = -yaw - np.pi
        yaw = normalize_angle(yaw)
        adj_track[j, 0:2] = xy.astype(np.float32)
        adj_track[j, 2] = np.float32(yaw)
        adj_mask[j, :] = 1.0
    return adj_track, adj_mask


def prepare_distance_repr(trajs):
    trajs = trajs.astype(np.float32).copy()
    trajs[..., 2] *= YAW_DISTANCE_WEIGHT
    return trajs.reshape(trajs.shape[0], -1)


def farthest_point_sampling(trajs, target_count, chunk_size=5000):
    if len(trajs) == 0:
        raise ValueError('no trajectories available for vocabulary construction')
    if len(trajs) <= target_count:
        if len(trajs) == target_count:
            return trajs
        repeat_indices = np.linspace(0, len(trajs) - 1, target_count).round().astype(np.int64)
        return trajs[repeat_indices]
    features = prepare_distance_repr(trajs)  # [N, D]
    selected = np.empty(target_count, dtype=np.int64)
    norms = np.linalg.norm(features[:, :2], axis=1)
    selected[0] = int(np.argmax(norms))
    min_dist = np.full(len(trajs), np.inf, dtype=features.dtype)
    for i in range(target_count):
        last = selected[i]
        # Chunked distance update to avoid O(M×N) matrices
        for c_start in range(0, len(trajs), chunk_size):
            c_end = min(c_start + chunk_size, len(trajs))
            chunk = features[c_start:c_end]
            dist = np.sum((chunk - features[last]) ** 2, axis=1)
            min_dist[c_start:c_end] = np.minimum(min_dist[c_start:c_end], dist)
        if i + 1 < target_count:
            next_idx = int(np.argmax(min_dist))
            selected[i + 1] = next_idx
    return trajs[selected]


def batched_distance(query_trajs, candidate_trajs):
    xy_error = (query_trajs[:, None, :, :2] - candidate_trajs[None, :, :, :2]) ** 2
    yaw_error = np.abs(query_trajs[:, None, :, 2] - candidate_trajs[None, :, :, 2])
    yaw_error = np.vectorize(normalize_angle)(yaw_error)
    return xy_error.sum(axis=(2, 3)) + YAW_DISTANCE_WEIGHT * np.abs(yaw_error).sum(axis=2)


def residual_trajs(target_trajs, anchor_trajs):
    residual = target_trajs.astype(np.float32).copy()
    residual[..., :2] -= anchor_trajs[..., :2]
    residual[..., 2] = np.vectorize(normalize_angle)(residual[..., 2] - anchor_trajs[..., 2])
    return residual


def compute_residual_diff_stats(all_trajs, anchor_bank):
    """Compute diff-space statistics from all trajectories relative to anchors."""
    trajs = np.asarray(all_trajs, dtype=np.float32)
    # Chunked nearest-anchor assignment to avoid O(M×N) matrices
    chunk_size = 10000
    residuals = []
    for start in range(0, len(trajs), chunk_size):
        end = min(start + chunk_size, len(trajs))
        chunk = trajs[start:end]
        distances = batched_distance(chunk, anchor_bank)
        nearest_idx = distances.argmin(axis=1)
        residuals.append(residual_trajs(chunk, anchor_bank[nearest_idx]))
    residuals = np.concatenate(residuals, axis=0)
    zero_pad = np.zeros((residuals.shape[0], 1, 2), dtype=np.float32)
    xy_diff = residuals[..., :2] - np.concatenate([zero_pad, residuals[:, :-1, :2]], axis=1)
    x_diff = xy_diff[..., 0]
    y_diff = xy_diff[..., 1]
    x_mean = float(x_diff.mean())
    y_mean = float(y_diff.mean())
    x_scale = float(max(abs(x_diff.max() - x_mean), abs(x_diff.min() - x_mean)))
    y_scale = float(max(abs(y_diff.max() - y_mean), abs(y_diff.min() - y_mean)))
    return {
        'x_diff_mean': x_mean,
        'x_diff_scale': x_scale,
        'y_diff_mean': y_mean,
        'y_diff_scale': y_scale,
    }


def build_unified_vocab(data_infos, sample_interval, planning_frames,
                         vocab_size, anchor_size):
    """Build a unified vocabulary and anchor bank (no command separation)."""
    all_trajs = []
    total_seen = 0
    for idx in range(len(data_infos)):
        total_seen += 1
        traj, mask = get_ego_future_xy(data_infos, idx, sample_interval, planning_frames)
        if not mask.all():
            continue
        all_trajs.append(traj)

    all_trajs = np.asarray(all_trajs, dtype=np.float32)
    n_valid = len(all_trajs)
    if n_valid == 0:
        raise ValueError('no valid trajectories found')

    # Subsample to manageable size for FPS (avoid OOM on 200k trajectories)
    rng = np.random.RandomState(42)
    max_fps_candidates = 50000
    if n_valid > max_fps_candidates:
        keep = rng.choice(n_valid, max_fps_candidates, replace=False)
        pool = all_trajs[keep]
        print(f'Subsampled {n_valid} → {max_fps_candidates} trajectories for FPS')
    else:
        pool = all_trajs
        print(f'Using all {n_valid} trajectories for FPS')

    vocab = farthest_point_sampling(pool, vocab_size)
    anchors = farthest_point_sampling(pool, anchor_size)
    # Compute diff stats from full dataset against unified anchors
    diff_stats = compute_residual_diff_stats(all_trajs, anchors)

    # Per-command stats for reference (not used by the model)
    cmd_counts = [0] * 6
    for idx in range(len(data_infos)):
        cmd = data_infos[idx].get('command_near', 4)
        cmd = int(cmd) if cmd is not None else 4
        if cmd <= 0 or cmd > 6:
            cmd = 4
        cmd_counts[cmd - 1] += 1

    meta = {
        'sample_interval': int(sample_interval),
        'planning_frames': int(planning_frames),
        'vocab_size': int(vocab_size),
        'anchor_size': int(anchor_size),
        'total_seen_samples': int(total_seen),
        'valid_samples_total': int(len(all_trajs)),
        'vocab_shape': list(vocab.shape),
        'anchor_shape': list(anchors.shape),
        'residual_diff_stats': diff_stats,
        'command_counts': {str(i): c for i, c in enumerate(cmd_counts)},
    }
    return vocab, anchors, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--anchor-output', required=True)
    parser.add_argument('--sample-interval', type=int, default=5)
    parser.add_argument('--planning-frames', type=int, default=6)
    parser.add_argument('--vocab-size', type=int, default=16384)
    parser.add_argument('--anchor-size', type=int, default=3072)
    args = parser.parse_args()

    with open(args.ann_file, 'rb') as f:
        data_infos = pickle.load(f)

    vocab, anchors, meta = build_unified_vocab(
        data_infos=data_infos,
        sample_interval=args.sample_interval,
        planning_frames=args.planning_frames,
        vocab_size=args.vocab_size,
        anchor_size=args.anchor_size,
    )

    output_path = Path(args.output)
    anchor_output_path = Path(args.anchor_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, vocab)
    np.save(anchor_output_path, anchors)
    meta_path = output_path.with_suffix('.json')
    meta.update({
        'ann_file': args.ann_file,
        'vocab_path': str(output_path),
        'anchor_path': str(anchor_output_path),
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'saved unified vocab  ({vocab.shape}) to {output_path}')
    print(f'saved unified anchors ({anchors.shape}) to {anchor_output_path}')
    print(f'saved metadata to {meta_path}')


if __name__ == '__main__':
    main()
