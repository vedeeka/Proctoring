from mtcnn import MTCNN
import cv2
import os
import numpy as np
import tensorflow as tf
from fer import FER


# UNIT 2: IMAGE RESTORATION AND NOISE CONTROL

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# UNIT 2: FILTERING AND PROCESS OPTIMIZATION

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# UNIT 3: COLOR IMAGE PROCESSING

emotion_detector = FER()


# UNIT 2: FILTERING IN FREQUENCY DOMAIN

def load_model(model_path):

    model = tf.saved_model.load(model_path)

    return model


# UNIT 4: IMAGE SEGMENTATION

def detect_mobile(model, frame_rgb, detection_threshold=0.50):

    # UNIT 1: IMAGE SAMPLING AND QUANTIZATION
 
    input_tensor = tf.convert_to_tensor(
        frame_rgb,
        dtype=tf.uint8
    )

    # UNIT 1: DIGITAL IMAGE REPRESENTATION

    input_tensor = input_tensor[tf.newaxis, ...]

    # UNIT 2: CNN FILTERING OPERATIONS
 
    detections = model.signatures['serving_default'](
        input_tensor
    )

    # UNIT 4: IMAGE REPRESENTATION

    bboxes = detections['detection_boxes'][0].numpy()

    classes = detections['detection_classes'][0].numpy().astype(int)

    scores = detections['detection_scores'][0].numpy()

    filtered_bboxes = []
    filtered_classes = []
    filtered_scores = []

    # UNIT 2: THRESHOLD-BASED FILTERING
 
    for bbox, cls, score in zip(
            bboxes,
            classes,
            scores):

        if score >= detection_threshold:

            filtered_bboxes.append(bbox)

            filtered_classes.append(cls)

            filtered_scores.append(score)

    return (
        filtered_bboxes,
        filtered_classes,
        filtered_scores
    )


# UNIT 4: IMAGE SEGMENTATION AND REPRESENTATION

def detect_gaze(
        threshold=0.3,
        model_path='ssd_mobilenet_v2_coco_2018_03_29/saved_model'):

    # UNIT 4: FACE SEGMENTATION USING MTCNN

    mtcnn_detector = MTCNN()

    # UNIT 2: MODEL-BASED IMAGE ANALYSIS

    detection_model = load_model(model_path)

    # UNIT 1: IMAGE ACQUISITION
 
    cap = cv2.VideoCapture(0)


    total_frames = 0
    looking_away_frames = 0
    mobile_detected_frames = 0
    multiple_people_frames = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        total_frames += 1

        # UNIT 1: SPATIAL DOMAIN PROCESSING
    
        frame = cv2.resize(frame, (740, 790))

        # UNIT 3: RGB COLOR MODEL PROCESSING
       
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # UNIT 4: IMAGE SEGMENTATION
    
        try:

            detections = mtcnn_detector.detect_faces(
                rgb_frame
            )

        except ValueError:

            detections = []

        # UNIT 4: REGION-BASED ANALYSIS

        if len(detections) > 1:

            multiple_people_frames += 1

            # UNIT 1: SPATIAL DOMAIN ENHANCEMENT
  
            cv2.putText(
                frame,
                "WARNING: Multiple People Detected!",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            print("Multiple people detected")


        for detection in detections:

            # UNIT 4: BOUNDARY REPRESENTATION
  
            x, y, w, h = detection['box']

            # UNIT 4: FACIAL LANDMARK REPRESENTATION

            keypoints = detection['keypoints']

            # UNIT 1: IMAGE BOUNDARY PROCESSING
          
            x, y = max(0, x), max(0, y)

            if x + w > frame.shape[1]:
                w = frame.shape[1] - x

            if y + h > frame.shape[0]:
                h = frame.shape[0] - y

            if w <= 0 or h <= 0:
                continue

            # UNIT 1: SPATIAL DOMAIN PROCESSING

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            # UNIT 3: MORPHOLOGICAL IMAGE PROCESSING(padding)
     
            pad_x = int(w * 0.4)
            pad_y = int(h * 0.4)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)

            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            # UNIT 4: REGION EXTRACTION
   
            face_roi = frame[y1:y2, x1:x2]

            dominant_emotion = "Unknown"
            confidence = 0.0

            # UNIT 3: COLOR IMAGE PROCESSING
       
            emotion = emotion_detector.detect_emotions(
                face_roi
            )

            text = "Human Detected"

            if emotion:

                # UNIT 4: FEATURE REPRESENTATION
       
                emotions_dict = emotion[0]['emotions']

                dominant_emotion, confidence = max(
                    emotions_dict.items(),
                    key=lambda e: e[1]
                )

                text = (
                    f"{text}, "
                    f"{dominant_emotion} "
                    f"({confidence:.2f})"
                )

                print(
                    f"Dominant Emotion: "
                    f"{dominant_emotion} "
                    f"({confidence:.2f})"
                )

            else:

                print("No emotions detected")

            # UNIT 1: IMAGE ENHANCEMENT IN SPATIAL DOMAIN

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # UNIT 4: FACIAL LANDMARK ANALYSIS
          
            left_eye = keypoints['left_eye']

            right_eye = keypoints['right_eye']

            nose = keypoints['nose']

            # UNIT 4: FEATURE EXTRACTION

            dist_left_eye_nose = (
                nose[0] - left_eye[0]
            )

            dist_right_eye_nose = (
                right_eye[0] - nose[0]
            )

  
            if dist_right_eye_nose == 0:
                dist_right_eye_nose = 0.001

            # UNIT 4: REGION-BASED GAZE ESTIMATION
          
            gaze_ratio = (
                dist_left_eye_nose /
                dist_right_eye_nose
            )

            # UNIT 4: IMAGE ANALYSIS
        
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

            # UNIT 1: SPATIAL DOMAIN DISPLAY
     
            color = (
                (0, 255, 0)
                if gaze_direction == "Looking Forward"
                else (0, 0, 255)
            )

            # UNIT 1: IMAGE DISPLAY PROCESSING
          
            cv2.putText(
                frame,
                gaze_direction,
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # UNIT 4: OBJECT SEGMENTATION

        bboxes, classes, scores = detect_mobile(
            detection_model,
            rgb_frame,
            detection_threshold=0.50
        )


        for bbox, cls, score in zip(
                bboxes,
                classes,
                scores):

            # UNIT 4: OBJECT CLASSIFICATION
            # Class 77 = Mobile phone
            if cls == 77:

                mobile_detected_frames += 1

                y_min, x_min, y_max, x_max = bbox

                # UNIT 4: BOUNDARY DETECTION
        
                start_point = (
                    int(x_min * frame.shape[1]),
                    int(y_min * frame.shape[0])
                )

                end_point = (
                    int(x_max * frame.shape[1]),
                    int(y_max * frame.shape[0])
                )

                # UNIT 1: SPATIAL DOMAIN PROCESSING
               
                cv2.rectangle(
                    frame,
                    start_point,
                    end_point,
                    (0, 0, 255),
                    3
                )

                # UNIT 1: IMAGE ENHANCEMENT
      
                cv2.putText(
                    frame,
                    f'Mobile Phone {score:.2f}',
                    (
                        start_point[0],
                        start_point[1] - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                print(
                    f"Mobile phone detected "
                    f"with {score:.2f} confidence"
                )

        # UNIT 1: IMAGE PROCESSING DEBUGGING
   
        if total_frames % 30 == 0:

            print(
                f"Processed "
                f"{total_frames} frames..."
            )

        # UNIT 1: IMAGE DISPLAY
  
        cv2.imshow(
            'Proctoring Gaze & Mobile Detection',
            frame
        )

     
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # UNIT 1: IMAGE ACQUISITION TERMINATION

    cap.release()


    cv2.destroyAllWindows()

    # UNIT 4: IMAGE ANALYSIS

    if total_frames == 0:

        return False, False, False

    # UNIT 4: STATISTICAL ANALYSIS

    looking_away_frequency = (
        looking_away_frames / total_frames
    )

    mobile_detection_frequency = (
        mobile_detected_frames / total_frames
    )

    multiple_people_frequency = (
        multiple_people_frames / total_frames
    )

    # UNIT 4: DECISION-BASED IMAGE ANALYSIS

    cheating_gaze = (
        looking_away_frequency > threshold
    )

    cheating_mobile = (
        mobile_detection_frequency > threshold
    )

    multiple_people_detected = (
        multiple_people_frequency > threshold
    )

    return (
        cheating_gaze,
        cheating_mobile,
        multiple_people_detected
    )


# UNIT 4: AUTOMATED IMAGE ANALYSIS SYSTEM

cheating_gaze, cheating_mobile, multiple_people_detected = detect_gaze(
    threshold=0.3,
    model_path='ssd_mobilenet_v2_coco_2018_03_29/saved_model'
)

# UNIT 4: RESULT REPRESENTATION



print(f"Cheating detected (gaze): {cheating_gaze}")

print(f"Cheating detected (mobile): {cheating_mobile}")

print(
    f"Multiple people detected: "
    f"{multiple_people_detected}"
)

print("==========================")