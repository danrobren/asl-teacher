# Custom defined functions to help main.py and train.py

import cv2
import mediapipe as mp
import string
import sys

def drawLandmarks(hand, image, title):
    
    if image is None:
        print("null image passed to drawLandmarks")
        return
    elif hand is None:
        print("null hand passed to drawLandmarks")
        return
        
    # initialize mediapipe drawing tool
    mp_draw = mp.solutions.drawing_utils
    
    # initialize and configure mediapipe hands
    mp_hands = mp.solutions.hands
    
    hands = mp_hands.Hands(
        model_complexity=1,
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.6)
    
    # draw the joints and bones on the hand image
    if hand.multi_hand_landmarks:
        for hand_landmarks in hand.multi_hand_landmarks:
            # Draw landmark dots
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (cx, cy), 5, (0, 0, 255), thickness=-1)

            # Optionally draw hand connections
            mp_draw.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255,0,0), thickness=2))
    else:
        print("No hands to draw in drawLandmarks")

    cv2.imshow(title, image)
    return 
