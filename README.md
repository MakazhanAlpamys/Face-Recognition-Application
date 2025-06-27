# Face Recognition Application

[Смотреть видео (MP4)](demo.mp4)

A simple MVP for facial recognition that can detect faces, display facial landmarks, and recognize different users.

## Features

- Real-time face detection using webcam
- Display of facial landmarks (key points of the face)
- Recognition of different users based on facial features
- Anti-spoofing through facial geometry analysis

## Requirements

- Python 3.10
- Webcam

## Setup

1. Create a Python 3.10 virtual environment:

```
python -m venv venv
```

2. Activate the virtual environment:

- Windows:
```
venv\Scripts\activate
```

- Linux/Mac:
```
source venv/bin/activate
```

3. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

Run the application:

```
python face_recognition_app.py
```

### Controls

- Press 'q' to quit the application
- Press 'r' to reset the face database

## How it works

The application uses:
- OpenCV for handling video streams and basic image processing
- MediaPipe for detecting facial landmarks
- Custom algorithm for face recognition based on facial geometry

When a face is detected, the application:
1. Extracts key facial points
2. Compares these points with registered users
3. Identifies the user or registers a new one
4. Displays the face mesh and recognition information 