# Custom defined functions to help main.py and train.py

import cv2
import string
import sys
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# pass the draw and hands objects so we don't have to re-instantiate them here
# how hands are drawn is controlled by the caller
def drawLandmarks(hand, image, title, mp_draw, mp_hands):
    if image is None:
        print("null image passed to drawLandmarks")
        return
    elif hand is None:
        print("null hand passed to drawLandmarks")
        return

    h, w, _ = image.shape

    # Case 1: MediaPipe hand object
    if hasattr(hand, 'multi_hand_landmarks') and hand.multi_hand_landmarks:
        for hand_landmarks in hand.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            for idx, landmark in enumerate(hand_landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                hue = int(255 * idx / 21)
                color = tuple(int(c) for c in cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
                cv2.circle(image, (cx, cy), 5, color, thickness=-1)
                cv2.putText(image, str(idx), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    # Case 2: Dict of arrays — convert to MediaPipe LandmarkList
    elif isinstance(hand, dict) and all(k in hand for k in ['x', 'y', 'z']):
        landmark_list = landmark_pb2.NormalizedLandmarkList()

        for i in range(21):
            lm = landmark_pb2.NormalizedLandmark()
            lm.x = float(hand['x'][i])
            lm.y = float(hand['y'][i])
            lm.z = float(hand['z'][i])
            landmark_list.landmark.append(lm)

        mp_draw.draw_landmarks(
            image, landmark_list, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        for idx, landmark in enumerate(landmark_list.landmark):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            hue = int(255 * idx / 21)
            color = tuple(int(c) for c in cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
            cv2.circle(image, (cx, cy), 5, color, thickness=-1)
            cv2.putText(image, str(idx), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    else:
        print("Unrecognized hand format or no hand landmarks found.")

    cv2.imshow(title, image)
    return


def flipHand(image, debug, title, mp_draw, mp_hands, Hands):
    image = cv2.flip(image, 1) # code 1 for horizontal flip
    hand = Hands.process(image) # reprocess hands for flipped image
    if debug:
        print(title)
        drawLandmarks(hand, image, title, mp_draw, mp_hands)
    return hand