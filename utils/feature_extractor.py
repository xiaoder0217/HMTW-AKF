# utils/feature_extractor.py

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import MeanShift

def compute_water_index(frame):
    """
    Compute a simple water index based on HSV saturation and blue channel.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    blue = frame[:, :, 0].astype(np.float32)
    return (saturation - blue) / (saturation + blue + 1e-6)

def compute_texture_features(gray_roi):
    """
    Compute texture features using Laplacian variance and GLCM.
    """
    laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
    lap_var = np.var(laplacian)

    quantized = (gray_roi // 16).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    return np.array([lap_var, contrast, entropy])

def compute_adaptive_threshold(values, quantile=0.85):
    """
    Compute threshold for feature filtering using MeanShift on dominant cluster.
    """
    if len(values) < 2:
        return np.median(values) if values else 0.0

    data = np.array(values).reshape(-1, 1)
    bandwidth = max(0.05, 0.2 * (np.max(data) - np.min(data)))
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)

    labels = ms.labels_
    counts = np.bincount(labels)
    dominant_label = np.argmax(counts)
    dominant_cluster = data[labels == dominant_label].flatten()

    return np.quantile(dominant_cluster, quantile)
