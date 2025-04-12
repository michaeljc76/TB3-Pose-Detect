import cv2
import mediapipe as mp
import numpy as np
from os import listdir
from os.path import isfile, join

# Initialize MediaPipe Face Detection (lightweight)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize LBPH Face Recognizer (lightweight)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_recognizer.yml')  # Load trained model

# Face database (name: id mapping)
face_names = {
    0: "Steve",
    1: "Alice",
    # Add more
}

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Reduce resolution for Pi
cap.set(4, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- Face Detection & Identification ---
    face_results = face_detection.process(rgb_frame)
    
    if face_results.detections:
        for detection in face_results.detections:
            # Extract face bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Face ROI for recognition
            face_roi = frame[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Resize for recognizer
            gray_face = cv2.resize(gray_face, (100, 100))
            
            # Perform recognition
            id_, confidence = recognizer.predict(gray_face)
            
            # Draw results
            if confidence < 70:  # Lower is better match
                name = face_names.get(id_, "Unknown")
                color = (0, 255, 0)  # Green for recognized
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} {confidence:.1f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # --- Pose Estimation ---
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Face ID + Pose Tracking", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()