# utils/io_utils.py

import os
import cv2
import pandas as pd

def save_keyframes_and_stats(frames, key_indices, segment_stats, out_dir, video_path, frame_interval):
    """
    Save selected keyframes and segment statistics to output directory.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(out_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    saved_info = []
    for idx in sorted(key_indices):
        if idx >= len(frames):
            continue
        filename = f"keyframe_{idx:04d}.jpg"
        out_path = os.path.join(save_dir, filename)
        cv2.imwrite(out_path, frames[idx])

        timestamp = round(idx * frame_interval / fps, 2)
        saved_info.append({"index": idx, "filename": filename, "timestamp (s)": timestamp})

    df_info = pd.DataFrame(saved_info)
    df_info.to_excel(os.path.join(save_dir, f"{video_name}_keyframes.xlsx"), index=False)

    if segment_stats:
        df_stats = pd.DataFrame(segment_stats)
        df_stats.to_excel(os.path.join(save_dir, f"{video_name}_segment_stats.xlsx"), index=False)

    return save_dir
