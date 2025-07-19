# main.py

import os
from config import *
from utils.video_loader import sample_frames_with_foreground
from utils.hash_segmenter import segment_by_hash, estimate_frame_interval_by_hash_variation
from utils.frame_selector import select_keyframes_from_segments
from utils.io_utils import save_keyframes_and_stats

def main():
    print("[Step 1] Sampling video frames and extracting foreground masks...")
    frames, gray_frames, foregrounds = sample_frames_with_foreground(VIDEO_PATH, FRAME_INTERVAL)

    print("[Step 2] Segmenting video using hash-based clustering...")
    segments, hashes = segment_by_hash(gray_frames)

    global FRAME_INTERVAL
    if USE_AUTO_INTERVAL:
        FRAME_INTERVAL = estimate_frame_interval_by_hash_variation(hashes)
        print(f"✅ Auto-estimated optimal frame interval: {FRAME_INTERVAL}")

    print("[Step 3] Selecting keyframes from each segment based on motion and features...")
    key_indices, segment_stats = select_keyframes_from_segments(
        frames, gray_frames, foregrounds, segments
    )

    print(f"[Step 4] Saving {len(key_indices)} keyframes and metadata to output directory...")
    save_dir = save_keyframes_and_stats(
        frames, key_indices, segment_stats, OUTPUT_DIR, VIDEO_PATH, FRAME_INTERVAL
    )

    print("✅ Process completed. Results saved to:", save_dir)

if __name__ == "__main__":
    main()
