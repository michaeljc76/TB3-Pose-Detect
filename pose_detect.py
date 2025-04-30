import sys
import cv2
import mediapipe as mp
import asyncio
import rclpy
import time
from move_agent import MoveAgent

async def main():
    rclpy.init()
    agent = MoveAgent()

    cap = cv2.VideoCapture(0)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # 360 movement variables
    last360 = 0
    timer360 = 10

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

                print("right wrist: ", right_wrist.x)
                print("left wrist: ", left_wrist.x)
                print("right shoulder: ", right_shoulder.x)
                print("left shoulder: ", left_shoulder.x)

                # Forward/Backward
                if right_wrist.y < right_eye.y:
                    print("Right arm is up!")
                    await agent.move('backward', 0.2)
                elif left_wrist.y < left_eye.y:
                    print("Left arm is up!")
                    await agent.move('forward', 0.2)
                # 360
                elif left_wrist.x < right_shoulder.x:
                    if time.time() - last360 >= timer360:
                        print("Left arm cross detected — 360!")
                        await agent.move('left', 6.7)
                        last360 = time.time()
                elif right_wrist.x > left_shoulder.x:
                    if time.time() - last360 >= timer360:
                        print("Right arm cross detected — 360!")
                        await agent.move('right', 6.7)
                        last360 = time.time()
                # Rotation
                elif left_wrist.x > left_shoulder.x and abs(left_wrist.y - left_shoulder.y) < 0.1 and abs(left_wrist.x - left_shoulder.x) > shoulder_width * 0.6:
                    print("Left arm extended outward!")
                    await agent.move('right', 0.2)
                elif right_wrist.x < right_shoulder.x and abs(right_wrist.y - right_shoulder.y) < 0.1 and abs(right_shoulder.x - right_wrist.x) > shoulder_width * 0.6:
                    print("Right arm extended outward!")
                    await agent.move('left', 0.2)
                else:
                    print("No unique pose detected.")

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Comment out if no X11 forwarding
            cv2.imshow("Pose Detection", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()

# Run everything
if __name__ == "__main__":
    asyncio.run(main())
