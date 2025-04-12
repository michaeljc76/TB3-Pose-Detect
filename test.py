import cv2
import numpy as np
import mediapipe as mp
from roboflow import Roboflow

def main():
    # Initialize RoboFlow face detection
    rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
    project = rf.workspace().project("your-project-name")
    face_model = project.version(1).model
    
    # Initialize MediaPipe pose estimation
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    
    # System state
    authorized = False
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB (needed for both models)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- FACE DETECTION ---
        try:
            face_prediction = face_model.predict(rgb_frame, confidence=70).json()
            
            # Check for authorized faces
            authorized = False
            for obj in face_prediction['predictions']:
                x = int(obj['x'] - obj['width']/2)
                y = int(obj['y'] - obj['height']/2)
                w = int(obj['width'])
                h = int(obj['height'])
                
                # Draw face bounding box (green if authorized, red otherwise)
                color = (0, 255, 0) if authorized else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Simple authorization logic (replace with your actual check)
                if obj['confidence'] > 0.7:  # High confidence face
                    authorized = True
        
        except Exception as e:
            print(f"Face detection error: {e}")
        
        # --- POSE ESTIMATION (only if authorized) ---
        if authorized:
            pose_results = pose.process(rgb_frame)
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    pose_results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0))
                    ))
                # Add your pose control logic here
                # Example: Check for specific gestures
                
        # Display authorization status
        status = "Authorized" if authorized else "Unauthorized"
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 0) if authorized else (0, 0, 255), 2)
        
        cv2.imshow('Face Auth + Pose Control', frame)
        
        # Exit on 'q' or ESC
        if cv2.waitKey(5) in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()