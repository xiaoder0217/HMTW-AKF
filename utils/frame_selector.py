# utils/frame_selector.py

import numpy as np
from tqdm import tqdm
import cv2
from config import (
    ENABLE_GLOBAL_DIFF, GLOBAL_DIFF_ALPHA, GLOBAL_DIFF_PERCENTILE,
    SCORE_THRESHOLD
)
from utils.feature_extractor import (
    compute_water_index, compute_texture_features, compute_adaptive_threshold
)

def compute_global_diff(foregrounds, total_pixels):
    """
    Compute global difference between consecutive foreground masks.
    """
    diffs = [
        np.sum(cv2.absdiff(foregrounds[i + 1], foregrounds[i])) / 255 / total_pixels
        for i in range(len(foregrounds) - 1)
    ]
    if not diffs:
        return set(), {"avg_diff": 0, "threshold": 0}

    avg_diff = np.mean(diffs)
    perc_val = np.percentile(diffs, GLOBAL_DIFF_PERCENTILE)
    dynamic_alpha = (perc_val / (avg_diff + 1e-6)) * GLOBAL_DIFF_ALPHA
    threshold = dynamic_alpha * avg_diff

    key_indices = {i + 1 for i, d in enumerate(diffs) if d >= threshold}
    return key_indices, {"avg_diff": avg_diff, "threshold": threshold}

def compute_motion_pixel_count(foregrounds):
    """
    Count average number of non-zero motion pixels in the segment.
    """
    return np.mean([np.count_nonzero(fg) for fg in foregrounds]) if foregrounds else 1

def select_keyframes_from_segments(frames, gray_frames, foregrounds, segments):
    """
    Select keyframes from each segment using global diff and feature filtering.
    """
    key_indices = set()
    segment_stats = []

    for seg_idx, seg in enumerate(tqdm(segments, desc="Processing segments")):
        fg_seg = [foregrounds[i] for i in seg]
        frame_seg = [frames[i] for i in seg]
        gray_seg = [gray_frames[i] for i in seg]

        total_pixels = fg_seg[0].size if fg_seg else 1
        P = compute_motion_pixel_count(fg_seg)

        global_keys = set()
        if ENABLE_GLOBAL_DIFF:
            global_keys, _ = compute_global_diff(fg_seg, P)
            global_keys = {seg[i] for i in global_keys if i < len(seg)}

        candidates = list(global_keys)
        candidate_frames = [frames[i] for i in candidates]

        selected = set()
        if candidate_frames:
            water_values, texture_values = [], []

            for frame in candidate_frames:
                roi = frame[frame.shape[0] // 2:, :]
