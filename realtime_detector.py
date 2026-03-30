from mtcnn import MTCNN
import cv2
import os
import numpy as np
import tensorflow as tf
from fer import FER


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  


emotion_detector = FER()

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def detect_mobile(model, frame_rgb, detection_threshold=0.50):
  
    input_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]


    detections = model.signatures['serving_default'](input_tensor)

    
    bboxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()


    filtered_bboxes = []
    filtered_classes = []
    filtered_scores = []
    for bbox, cls, score in zip(bboxes, classes, scores):
        if score >= detection_threshold:
            filtered_bboxes.append(bbox)
            filtered_classes.append(cls)
            filtered_scores.append(score)

    return filtered_bboxes, filtered_classes, filtered_scores

def detect_gaze(threshold=0.3, model_path='ssd_mobilenet_v2_coco_2018_03_29/saved_model'):
    # Load the MTCNN face detector
    mtcnn_detector = MTCNN()
    
    # Load the TensorFlow model
    detection_model = load_model(model_path)
    
    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Variables to count gaze directions and mobile detection
    total_frames = 0
    looking_away_frames = 0
    mobile_detected_frames = 0
    multiple_people_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame = cv2.resize(frame, (740, 790))
        
        # Accuracy Fix: Convert BGR to RGB for Neural Networks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using MTCNN
        try:
            detections = mtcnn_detector.detect_faces(rgb_frame)
        except ValueError:
            detections = [] # Catch MTCNN empty array bug

        # Check if there are multiple faces in the frame
        if len(detections) > 1:
            multiple_people_frames += 1
            cv2.putText(frame, "WARNING: Multiple People Detected!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            print("Multiple people detected")

        for detection in detections:
            x, y, w, h = detection['box']
            keypoints = detection['keypoints']

            # Keep core coordinates within frame boundaries
            x, y = max(0, x), max(0, y)
            if x + w > frame.shape[1]: w = frame.shape[1] - x
            if y + h > frame.shape[0]: h = frame.shape[0] - y
            if w <= 0 or h <= 0: continue # Skip invalid boxes

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # --- EMOTION DETECTION FIX ---
            # The fer library fails if the face is cropped too tightly without background.
            # We add a 40% margin around the face so its internal detector can "see" the head outlines.
            pad_x = int(w * 0.4)
            pad_y = int(h * 0.4)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)
            
            face_roi = frame[y1:y2, x1:x2]
            
            dominant_emotion = "Unknown"
            confidence = 0.0
            
            # Pass the padded face image to the emotion detector
            emotion = emotion_detector.detect_emotions(face_roi)
            
            text = "Human Detected"
            if emotion:
                # Extract the highest scoring emotion
                emotions_dict = emotion[0]['emotions']
                dominant_emotion, confidence = max(emotions_dict.items(), key=lambda e: e[1])
                text = f"{text}, {dominant_emotion} ({confidence:.2f})"
                print(f"Dominant Emotion: {dominant_emotion} ({confidence:.2f})")
            else:
                print("No emotions detected")

            # Display emotion text above the bounding box
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


            # --- GAZE DETECTION (Accuracy Fix: Facial Landmarks) ---
            # Using distances between eyes and nose to calculate accurate 3D head pose angle
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            nose = keypoints['nose']

            dist_left_eye_nose = nose[0] - left_eye[0]
            dist_right_eye_nose = right_eye[0] - nose[0]
            
            # Prevent division by zero
            if dist_right_eye_nose == 0: dist_right_eye_nose = 0.001 
            
            gaze_ratio = dist_left_eye_nose / dist_right_eye_nose

            if gaze_ratio > 1.6:
                gaze_direction = "Looking Right"
                looking_away_frames += 1
                print("Face right")
            elif gaze_ratio < 0.6:
                gaze_direction = "Looking Left"
                looking_away_frames += 1
                print("Face left")
            else:
                gaze_direction = "Looking Forward"

            # Display the gaze direction below the face rectangle
            color = (0, 255, 0) if gaze_direction == "Looking Forward" else (0, 0, 255)
            cv2.putText(frame, gaze_direction, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


        # --- MOBILE PHONE DETECTION ---
        # Pass the converted RGB frame to the object detection model
        bboxes, classes, scores = detect_mobile(detection_model, rgb_frame, detection_threshold=0.50)

        for bbox, cls, score in zip(bboxes, classes, scores):
            if cls == 77:  # Class 77 is 'cell phone' in the COCO dataset
                mobile_detected_frames += 1
                y_min, x_min, y_max, x_max = bbox
                start_point = (int(x_min * frame.shape[1]), int(y_min * frame.shape[0]))
                end_point = (int(x_max * frame.shape[1]), int(y_max * frame.shape[0]))
                
                cv2.rectangle(frame, start_point, end_point, (0, 0, 255), 3)
                cv2.putText(frame, f'Mobile Phone {score:.2f}', (start_point[0], start_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"Mobile phone detected with {score:.2f} confidence")

        # For debugging: print progress
        if total_frames % 30 == 0:
            print(f"Processed {total_frames} frames...")

        # Display the frame
        cv2.imshow('Proctoring Gaze & Mobile Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate frequencies
    if total_frames == 0:
        return False, False, False 

    looking_away_frequency = looking_away_frames / total_frames
    mobile_detection_frequency = mobile_detected_frames / total_frames
    multiple_people_frequency = multiple_people_frames / total_frames

    # Determine if the person is cheating based on the threshold
    cheating_gaze = looking_away_frequency > threshold
    cheating_mobile = mobile_detection_frequency > threshold
    multiple_people_detected = multiple_people_frequency > threshold

    return cheating_gaze, cheating_mobile, multiple_people_detected

# Run the detection
cheating_gaze, cheating_mobile, multiple_people_detected = detect_gaze(threshold=0.3, model_path='ssd_mobilenet_v2_coco_2018_03_29/saved_model')

print("\n=== PROCTORING RESULTS ===")
print(f"Cheating detected (gaze): {cheating_gaze}")
print(f"Cheating detected (mobile): {cheating_mobile}")
print(f"Multiple people detected: {multiple_people_detected}")
print("==========================")