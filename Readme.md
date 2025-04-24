# Warning Zone Detection System

## Overview
This project implements a warning zone detection system that uses computer vision to track people's feet positions and alerts when feet enter predefined warning zones. The system can predict foot positions even when they are not directly visible by using upper body keypoints.

## Features
- Human pose detection using YOLOv8
- Feet position prediction when feet are occluded
- Warning zone definition via mouse clicks
- Real-time overlap detection between feet and warning zones

## Technical Details
The project consists of two main components:

### 1. Feet Prediction Model (FeetPredict.py)
- Uses the COCO 2017 keypoint dataset for training
- Implements both a PyTorch neural network and a scikit-learn linear regression model
- Predicts foot (ankle) positions based on shoulder and hip keypoints
- Trained and evaluated on COCO's person keypoints annotations

### 2. Warning Zone Detector (main.py)
- Uses YOLOv8 for real-time human pose estimation
- Allows users to define warning zones by clicking points on the screen
- Predicts foot positions for people whose feet are not visible
- Highlights feet in red when they enter a warning zone
- Supports both webcam input and video files

## Requirements
- PyTorch
- OpenCV
- Ultralytics
- NumPy
- scikit-learn
- tqdm
- seaborn
- asyncio

```bash
pip install torch torchvision torchaudio opencv-python ultralytics numpy scikit-learn tqdm seaborn asyncio
```

## Usage
1. Run the code within the `main.ipynb` notebook.

2. Define a warning zone by clicking points on the screen (minimum 3 points to form a polygon)

3. The system will detect people and track their feet:
   - Green circles: Left feet
   - Blue circles: Right feet
   - Red circles: Feet inside warning zones
   - Purple circles: Hip keypoints (used for prediction)
   - Red circles (upper body): Shoulder keypoints (used for feet prediction)

4. Press 'q' to quit

## Model Training
To train the feet prediction model, run the code within the `FeetPredict.ipynb` notebook.

This will:
1. Load the COCO keypoint annotations
2. Extract shoulder, hip, and foot keypoints
3. Train both linear regression and PyTorch models
4. Save the models to the `./Models/` directory
