import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get right wrist and right shoulder
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Check if wrist is above shoulder
        if right_wrist.y < right_shoulder.y:
            print("Right arm is up!")
        else:
            print("No unique pose detected.")

        # Optional: draw landmarks (even if not displaying frame)
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Comment out if there's no display, else script will error
    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
