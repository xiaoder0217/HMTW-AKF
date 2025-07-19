# utils/video_loader.py

import cv2
from config import USE_OTSU_FOR_MOTION, MOTION_THRESHOLD

def sample_frames_with_foreground(video_path, interval):
    """
    Sample frames from a video and extract foreground masks using MOG2 background subtractor.
    """
    cap = cv2.VideoCapture(video_path)
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    frames, gray_frames, foregrounds = [], [], []
    idx = 0
    success, frame = cap.read()

    while success:
        if idx % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = back_sub.apply(frame)

            if USE_OTSU_FOR_MOTION:
                _, fg_bin = cv2.threshold(fg_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, fg_bin = cv2.threshold(fg_mask, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)

            frames.append(frame.copy())
            gray_frames.append(gray)
            foregrounds.append(fg_bin)

        success, frame = cap.read()
        idx += 1

    cap.release()
    return frames, gray_frames, foregrounds
