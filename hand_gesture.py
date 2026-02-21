import os
import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np

# Use MediaPipe Tasks HandLandmarker (requires a model file)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import custom modules
from gesture_classifier import GestureClassifier
from command_mapper import get_gesture_command, is_critical_command

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_NAME = "hand_landmarker.task"


def ensure_model(path: str):
    if not os.path.exists(path):
        print(f"Downloading model to {path}...")
        urllib.request.urlretrieve(MODEL_URL, path)


def main():
    root = os.path.dirname(__file__)
    model_path = os.path.join(root, MODEL_NAME)
    ensure_model(model_path)

    BaseOptions = python.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(1)
    detector = HandLandmarker.create_from_options(options)

    mp_hands = mp.tasks.vision.HandLandmarksConnections
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles

    # Initialize gesture classifier
    gesture_classifier = GestureClassifier()
    
    # Command feedback with frame-based validation
    last_gesture = "unknown"
    last_command_time = 0
    gesture_display_duration = 1.0  # seconds
    frame_count = 0
    
    # Gesture validation tracking (10-12 frame window)
    gesture_frame_history = []  # Stores (gesture, confidence) tuples
    confidence_threshold = 0.7
    min_frames_required = 10  # Minimum consistent frames needed
    max_frames = 12  # Window size

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)

            annotated = frame.copy()
            current_time = time.time()
            
            # Draw info panel background
            cv2.rectangle(annotated, (0, 0), (w, 100), (30, 30, 30), -1)

            if detection_result and detection_result.hand_landmarks:
                for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
                    
                    # Get handedness
                    handedness = detection_result.handedness[idx][0].category_name
                    
                    # Convert landmarks to numpy format for classifier
                    lm_array = np.array([
                        [lm.x, lm.y, lm.z] for lm in hand_landmarks
                    ])
                    
                    # Classify gesture and get confidence
                    raw_gesture, raw_confidence = gesture_classifier.classify(lm_array, handedness)
                    gesture, smoothed_confidence = gesture_classifier.smooth_gesture(raw_gesture, raw_confidence)
                    
                    # Track gesture in frame history for validation
                    if gesture != "unknown":
                        gesture_frame_history.append((gesture, smoothed_confidence))
                    
                    # Keep only last 12 frames
                    if len(gesture_frame_history) > max_frames:
                        gesture_frame_history.pop(0)
                    
                    # Check if we have a validated gesture
                    validated_gesture = None
                    if len(gesture_frame_history) >= min_frames_required:
                        # Check if majority of frames have same gesture
                        gesture_names = [g for g, _ in gesture_frame_history]
                        gesture_confidences = [c for _, c in gesture_frame_history]
                        
                        most_common = max(set(gesture_names), key=gesture_names.count)
                        count = gesture_names.count(most_common)
                        avg_confidence = sum(gesture_confidences) / len(gesture_confidences)
                        
                        # Validate if: same gesture for 10+ frames AND confidence > 0.7
                        if count >= min_frames_required and avg_confidence > confidence_threshold:
                            validated_gesture = most_common
                            gesture_frame_history = []  # Reset after validation
                    
                    # Get command only if validated
                    if validated_gesture:
                        cmd_text, cmd_color = get_gesture_command(validated_gesture)
                        display_text = f"✓ {validated_gesture.upper()}: {cmd_text}"
                        last_gesture = validated_gesture
                        last_command_time = current_time
                        
                        # Highlight critical commands
                        if is_critical_command(validated_gesture):
                            cv2.rectangle(annotated, (0, 0), (w, h), cmd_color, 3)
                    elif gesture != "unknown":
                        # Show unvalidated gesture with frame count
                        display_text = f"🔄 {gesture.upper()} ({len(gesture_frame_history)}/{min_frames_required})"
                    else:
                        display_text = None
                    
                    # Draw gesture label near wrist
                    wrist = hand_landmarks[0]
                    x = int(wrist.x * w)
                    y = int(wrist.y * h)
                    cv2.putText(annotated, f"{handedness}", (x - 20, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display command in panel
            if current_time - last_command_time < gesture_display_duration:
                cmd_text, cmd_color = (get_gesture_command(last_gesture) if last_gesture in ["left_point", "right_point", "up_point", "down_point", "victory", "rock", "open_palm", "closed_palm"]
                                      else ("", (128, 128, 128)))
                if cmd_text:
                    text = f"COMMAND: {cmd_text}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                    x = (w - text_size[0]) // 2
                    cv2.putText(annotated, text, (x, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, cmd_color, 2)
            
            # Draw FPS and instructions
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(annotated, f"Buffer: {len(gesture_frame_history)}/{max_frames} | Threshold: {confidence_threshold:.1%}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(annotated, "Point your fingers to control drone", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(annotated, "Press 'q' to quit", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow('Hand Gesture Recognition - Drone Control', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Demo ended. Gesture recognition complete!")


if __name__ == '__main__':
    main()
