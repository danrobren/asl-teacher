# Main program with GUI and decision maker for letter detection

import cv2
import mediapipe as mp
import string
import sys
import pickle
import numpy as np
import utils # utils.py custom defines functions

# constants and settings
# TODO:  have settings and constants in their own file that we load in main, train, and utils
# so we can modify settings globally
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# initialize hand landmarker (hands)
hands = mp_hands.Hands(
    model_complexity=1,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.6)

# all SignAlphaSet images are 296x296
width = 296

# create a black canvas on which to draw ideals (for testing)
blackCanvas = np.zeros((296, 296, 3), dtype=np.uint8)

# load the ideal hand landmarks
ideals = []
with open('ideals.pkl', 'wb') as f:
    pickle.dump(ideals, f)
    
# test print ideals on a black canvas
for hand_landmarks in ideals:
    utils.drawLandmarks(hand_landmarks, blackCanvas, "ideal")
    k = cv2.waitKey(0)
    if k == 27 or k == ord('q'):
        cv2.destroyAllWindows()
        sys.exit()
    else: 
        cv2.destroyAllWindows()
    
    
# TODO: find a way to associate the hands in ideals with a letter or index 
    
# TODO: clever algorithm to implement a decision tree for hands 