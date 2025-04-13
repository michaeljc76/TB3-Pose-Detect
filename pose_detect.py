import sys
import cv2
import mediapipe as mp
import asyncio
import rclpy
from move_agent import MoveAgent

async def main():
    rclpy.init()
    agent = MoveAgent()

    cap = cv2.VideoCapture(0)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

                shoulder_width = abs(right_shoulder.x - left_shoulder.x)

                left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
                right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

                # Forward/Backward
                if right_wrist.y < right_eye.y:
                    print("Right arm is up!")
                    await agent.move('backward', 0.2)
                elif left_wrist.y < left_eye.y:
                    print("Left arm is up!")
                    await agent.move('forward', 0.2)
                # Rotation
                elif abs(left_wrist.y - left_shoulder.y) < 0.1 and abs(left_shoulder.x - left_wrist.x) > shoulder_width * 0.6:
                    print("Left arm extended horizontally!")
                    await agent.move('right', 0.2)
                elif abs(right_wrist.y - right_shoulder.y) < 0.1 and abs(right_shoulder.x - right_wrist.x) > shoulder_width * 0.6:
                    print("Right arm extended horizontally!")
                    await agent.move('left', 0.2)
                else:
                    print("No unique pose detected.")

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()

# Run everything
if __name__ == "__main__":
    asyncio.run(main())
