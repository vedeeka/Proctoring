import os
import cv2
import numpy as np
import tensorflow as tf
from fer import FER
import time
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


def detect_gaze(threshold=0.3,display_duration=2, model_path='ssd_mobilenet_v2_coco_2018_03_29/saved_model'):
    face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    last_update_time = time.time()
    detection_model = load_model(model_path)
    cap = cv2.VideoCapture(0)

    total_frames = 0
    looking_away_frames = 0
    mobile_detected_frames = 0
    multiple_people_frames = 0
    previous_face_center = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame = cv2.resize(frame, (740, 790))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(200, 200))
        if len(faces) == 0:
            faces = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(300, 300))

        if len(faces) > 1:
            multiple_people_frames += 1
            print("Multiple people detected")

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_center = (x + w // 2, y + h // 2)

            # Emotion Detection
            dominant_emotion = "Unknown"
            confidence = 0.0
            emotion = emotion_detector.detect_emotions(frame[y:y+h, x:x+w])
            text = "Human Detected"
            if emotion:
                emotions_dict = emotion[0]['emotions']
                dominant_emotion, confidence = max(emotions_dict.items(), key=lambda x: x[1])
                text = f"{text}, {dominant_emotion} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Gaze Direction Detection
            if previous_face_center:
                x_diff = face_center[0] - previous_face_center[0]
                dynamic_threshold = w * 0.008  # Adjust this multiplier based on sensitivity preference
                if x_diff < -dynamic_threshold:
                    gaze_direction = "Left"
                    print("Face left")
                elif x_diff > dynamic_threshold:
                    gaze_direction = "Right"
                    print("Face right")
                else:
                    gaze_direction = "Forward"
                    print("Face forward")
                
                if time.time() - last_update_time > display_duration:
                    current_gaze_direction = gaze_direction
                    last_update_time = time.time()

                cv2.putText(frame, current_gaze_direction, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if gaze_direction in ["Left", "Right"]:
                    looking_away_frames += 1

            previous_face_center = face_center

        # Mobile Detection
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

        

        cv2.imshow('Gaze Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    looking_away_frequency = looking_away_frames / total_frames
    mobile_detection_frequency = mobile_detected_frames / total_frames
    multiple_people_frequency = multiple_people_frames / total_frames

    cheating_gaze = looking_away_frequency > threshold
    cheating_mobile = mobile_detection_frequency > threshold
    multiple_people_detected = multiple_people_frequency > threshold

    return cheating_gaze, cheating_mobile, multiple_people_detected


# Run the detection
cheating_gaze, cheating_mobile, multiple_people_detected = detect_gaze(model_path='ssd_mobilenet_v2_coco_2018_03_29/saved_model')
print(f"Cheating detected (gaze): {cheating_gaze}")
print(f"Cheating detected (mobile): {cheating_mobile}")
print(f"Multiple people detected: {multiple_people_detected}")
