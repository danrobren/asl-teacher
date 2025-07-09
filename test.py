# This file tests live video and drawing functionality
# It opens a window with the webcam and draws any hand lanmarks detected

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize Hand Landmarker (Hands)
hands = mp_hands.Hands(
    model_complexity=1,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.6)

# Start webcam capture (index 0 = default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    result = hands.process(rgb)

    # Draw landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmark dots
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), thickness=-1)

            # Optionally draw hand connections
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255,0,0), thickness=2))

    cv2.imshow('MediaPipe Hands', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
