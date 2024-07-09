# Gaze and Mobile Detection System

This project is a computer vision-based system designed to detect gaze direction, mobile phone usage, and multiple people in a video feed. It utilizes TensorFlow, OpenCV, and the FER (Facial Emotion Recognition) library to analyze frames from a camera feed and identify potential cheating or distractions.

## Features

- **Face Detection**: Uses OpenCV's Haar Cascades to detect faces in the video feed.
- **Gaze Detection**: Analyzes the position of the detected face to determine if the person is looking away.
- **Emotion Detection**: Identifies the dominant emotion of the detected face using the FER library.
- **Mobile Phone Detection**: Utilizes a pre-trained TensorFlow object detection model to identify mobile phones in the video feed.
- **Multiple People Detection**: Counts the number of faces in the frame to detect the presence of multiple people.

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/gaze-mobile-detection.git
cd gaze-mobile-detection
