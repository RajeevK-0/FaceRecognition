# Real-Time Face Recognition System 👁️

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-green.svg)
![NumPy](https://img.shields.io/badge/Library-NumPy-yellow.svg)

## 📋 Project Overview
This project is an end-to-end computer vision application capable of detecting and recognizing faces in real-time. Unlike standard library implementations, this project implements the **K-Nearest Neighbors (KNN)** classification algorithm from scratch to demonstrate the mathematical foundations of supervised learning.

The system is broken down into three modular components:
1.  **Detection:** Identifying faces in a video stream.
2.  **Data Gathering:** Automated extraction and normalization of face data.
3.  **Recognition:** Real-time classification using distance-based matching.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `faceDetection.py` | A proof-of-concept script for validating camera inputs and Haar Cascade detection. |
| `faceData.py` | **The Data Pipeline.** Captures video frames, detects faces, crops them to a standard ROI ($100 \times 100$), and serializes them into NumPy files (`.npy`) for training. |
| `faceRecognition.py` | **The Inference Engine.** Loads the training data, implements the KNN algorithm (with Euclidean distance), and predicts the identity of the person in the frame. |
| `haarcascade_frontalface_alt.xml` | Pre-trained model required for face detection (Viola-Jones algorithm). |

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.x
* A working webcam

### Step 1: Clone the Repository
```bash
git clone [https://github.com/your-username/face-recognition-knn.git](https://github.com/your-username/face-recognition-knn.git)
cd face-recognition-knn
