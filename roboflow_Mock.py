#pip install opencv-python numpy roboflow

import cv2
import numpy as np
from roboflow import Roboflow

def main():
    # Initialize RoboFlow (replace with your actual API key and project details)
    rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
    project = rf.workspace().project("your-project-name")
    model = project.version(1).model  # Update version number as needed

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device")
        return

    print("Face detection running. Press 'q' to quit...")
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert frame to RGB (RoboFlow models typically expect RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Perform prediction with confidence threshold of 50%
            prediction = model.predict(rgb_frame, confidence=50, overlap=30).json()
            
            # Draw bounding boxes on detected faces
            for obj in prediction['predictions']:
                # Convert normalized coordinates to pixel values
                x = int(obj['x'] - obj['width']/2)
                y = int(obj['y'] - obj['height']/2)
                w = int(obj['width'])
                h = int(obj['height'])
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add label and confidence score
                label = f"{obj['class']} {obj['confidence']:.2f}" if 'class' in obj else f"Face {obj['confidence']:.2f}"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        except Exception as e:
            print(f"Prediction error: {e}")
        
        # Display the resulting frame
        cv2.imshow('RoboFlow Face Detection', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped.")

if __name__ == "__main__":
    main()