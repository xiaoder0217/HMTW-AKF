# utils/hash_segmenter.py

import cv2
import imagehash
from PIL import Image

def segment_by_hash(gray_frames, hamming_thresh=20):
    """
    Cluster grayscale frames based on average hash similarity.
    """
    def average_hash(image):
        resized = cv2.resize(image, (8, 8))
        pil_img = Image.fromarray(resized)
        return imagehash.average_hash(pil_img)

    hashes = [average_hash(f) for f in gray_frames]
    assigned = [False] * len(hashes)
    segments = []

    for i, h in enumerate(hashes):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, len(hashes)):
            if not assigned[j] and h - hashes[j] <= hamming_thresh:
                group.append(j)
                assigned[j] = True
        segments.append(group)

    return segments, hashes

def estimate_frame_interval_by_hash_variation(hashes):
    """
    Estimate an appropriate frame interval based on average hash differences.
    """
    if len(hashes) < 2:
        return 1

    diffs = [hashes[i] - hashes[i - 1] for i in range(1, len(hashes))]
    avg_diff = sum(diffs) / len(diffs)

    if avg_diff > 15:
        return 1
    elif avg_diff > 10:
        return 2
    elif avg_diff > 6:
        return 3
    elif avg_diff > 3:
        return 5
    else:
        return 8
