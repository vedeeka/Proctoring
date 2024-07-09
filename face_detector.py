import os
import cv2
import numpy as np
import tensorflow as tf
from fer import FER
# Suppress oneDNN custom operations logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow log verbosity
emotion_detector = FER()
def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def detect_mobile(model, frame, detection_threshold=0.05):
    # Prepare the input tensor
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform inference using the specific function or method for detection
    detections = model.signatures['serving_default'](input_tensor)

    # Extract bounding boxes, classes, and scores
    bboxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()

    # Filter detections based on the detection threshold
    filtered_bboxes = []
    filtered_classes = []
    filtered_scores = []
    for bbox, cls, score in zip(bboxes, classes, scores):
        if score >= detection_threshold:
            filtered_bboxes.append(bbox)
            filtered_classes.append(cls)
            filtered_scores.append(score)

    return filtered_bboxes, filtered_classes, filtered_scores

def detect_gaze(threshold=0.3, skip_frames=30, model_path='ssd_mobilenet_v2_coco_2018_03_29/saved_model'):
    # Load the pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load the TensorFlow model
    detection_model = load_model(model_path)
    
    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Variables to count gaze directions and mobile detection
    total_frames = 0
    looking_away_frames = 0
    mobile_detected_frames = 0
    multiple_people_frames = 0
    frame_count = 0
    previous_face_center = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames to speed up processing
        if frame_count % skip_frames != 0:
            continue

        total_frames += 1

        # Convert the frame to grayscale (face detectors expect grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if there are multiple faces in the frame
        if len(faces) > 1:
            multiple_people_frames += 1
            print("Multiple people detected")

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_center = (x + w // 2, y + h // 2)
            emotion = emotion_detector.detect_emotions(face_roi)
            if emotion:
                emotions_dict = emotion[0]['emotions']
                dominant_emotion, confidence = max(emotions_dict.items(), key=lambda x: x[1])
                # Further logic based on dominant_emotion and confidence
            else:
                # Handle case where emotion detection fails or returns empty
                print("No emotions detected")

            

            print(f"Dominant Emotion: {dominant_emotion} ({confidence})")

    # Example logic to check for specific emotions
            if dominant_emotion == 'happy' and confidence > 0.5:
                print("Person is happy")
            elif dominant_emotion == 'sad' and confidence > 0.5:
                print("Person is sad")
            if previous_face_center:
                if face_center[0] < previous_face_center[0] - w * 0.4:
                    gaze_direction = "Left"
                    print("Face left")
                elif face_center[0] > previous_face_center[0] + w * 0.4:
                    gaze_direction = "Right"
                    print("Face right")
                else:
                    gaze_direction = "Forward"
                    print("Face forward")

                # Count looking away frames
                if gaze_direction == "Left" or gaze_direction == "Right":
                    looking_away_frames += 1

            previous_face_center = face_center

        # Detect mobile phones
        bboxes, classes, scores = detect_mobile(detection_model, frame, detection_threshold=0.05)

        for bbox, cls, score in zip(bboxes, classes, scores):
            if cls == 77:  # Assuming class 77 is 'cell phone' in the COCO dataset
                mobile_detected_frames += 1
                y_min, x_min, y_max, x_max = bbox
                start_point = (int(x_min * frame.shape[1]), int(y_min * frame.shape[0]))
                end_point = (int(x_max * frame.shape[1]), int(y_max * frame.shape[0]))
                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
                cv2.putText(frame, 'Mobile Phone', start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                print("Mobile phone detected")

        # For debugging: print progress
        if total_frames % 10 == 0:
            print(f"Processed {total_frames} frames...")

        # Display the frame
        cv2.imshow('Gaze Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate the frequency of looking away, mobile detection, and multiple people detection
    if total_frames == 0:
        return False, False, False  # No frames processed, assume no cheating

    looking_away_frequency = looking_away_frames / total_frames
    mobile_detection_frequency = mobile_detected_frames / total_frames
    multiple_people_frequency = multiple_people_frames / total_frames

    # Determine if the person is cheating based on the threshold
    cheating_gaze = looking_away_frequency > threshold
    cheating_mobile = mobile_detection_frequency > threshold
    multiple_people_detected = multiple_people_frequency > threshold

    return cheating_gaze, cheating_mobile, multiple_people_detected


cheating_gaze, cheating_mobile, multiple_people_detected = detect_gaze(model_path='ssd_mobilenet_v2_coco_2018_03_29/saved_model')
print(f"Cheating detected (gaze): {cheating_gaze}")
print(f"Cheating detected (mobile): {cheating_mobile}")
print(f"Multiple people detected: {multiple_people_detected}")