# HMTW-AKF: Automated Keyframe Extraction from Video Using Hashing and Feature-Based Analysis

This project extracts representative keyframes from videos using a combination of frame hashing, motion detection, and domain-specific feature extraction (e.g., water and texture characteristics).

### 🚀 Features

- Frame sampling with adaptive interval estimation
- Foreground extraction using MOG2 and optional OTSU thresholding
- Hash-based frame segmentation and clustering
- Texture and water index feature extraction from selected regions
- Keyframe selection based on global motion and domain relevance
- Output includes both images and Excel-based metadata

---

### 🛠 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/xiaoder0217/HMTW-AKF.git
cd HMTW-AKF
pip install -r requirements.txt
