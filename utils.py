# Custom defined functions to help main.py and train.py

import cv2
import mediapipe as mp
import string
import sys

# pass the draw and hands objects so we don't have to re-instantiate them here
# how hands are drawn is controlled by the caller
def drawLandmarks(hand, image, title, mp_draw, mp_hands):
    
    if image is None:
        print("null image passed to drawLandmarks")
        return
    elif hand is None:
        print("null hand passed to drawLandmarks")
        return
    
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

def flipHand(image, debug, title, mp_draw, mp_hands, Hands):
    print("enter flipHand")
    image = cv2.flip(image, 1) # code 1 for horizontal flip
    print("flip image")
    hand = Hands.process(image) # reprocess hands for flipped image
    print("reprocess image")
    if debug:
        print(title)
        drawLandmarks(hand, image, title, mp_draw, mp_hands)
    return hand