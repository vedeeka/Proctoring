# Proctoring

Enhance proctoring system: Improve face detection, gaze tracking with deep learning models (e.g., TensorFlow), integrate robust mobile phone detection, and optimize multi-person detection using OpenCV for real-time monitoring and terminal output.

## Features

- **Face Detection**: Uses OpenCV's Haar Cascades to detect faces in the video feed.
- **Gaze Detection**: Analyzes the position of the detected face to determine if the person is looking away.
- **Emotion Detection**: Identifies the dominant emotion of the detected face using the FER library.
- **Mobile Phone Detection**: Utilizes a pre-trained TensorFlow object detection model to identify mobile phones in the video feed.
- **Multiple People Detection**: Counts the number of faces in the frame to detect the presence of multiple people.

## Installation

Clone the repository:
```bash
git clone https://github.com/vedeeka/gaze-mobile-detection.git](https://github.com/vedeeka/Proctoring.git)
pip install -r requirements.txt

curl -O http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

python face_detector.py 
```
## Usage
Run the main script to start monitoring:

```bash
python face_detector.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
