import cv2
import socket
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 5060

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,   # 🔥 changed from LIVE_STREAM
    num_hands=1
)

with HandLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp)

        height_value = 0.5

        if result.hand_landmarks:
            wrist = result.hand_landmarks[0][0]
            height_value = 1 - wrist.y

        sock.sendto(str(height_value).encode(), (UDP_IP, UDP_PORT))

        cv2.imshow("MediaPipe", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
