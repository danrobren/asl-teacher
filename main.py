# Main program with GUI and decision maker for letter detection

import cv2
import mediapipe as mp
import string
import sys
import pickle
import numpy as np

import utils # utils.py custom defines functions
import settings # settings.py constants and Hands configurations

print("Welcome to ASL Teacher")

# constants and settings
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# initialize hand landmarker (hands)
Hands = mp_hands.Hands(**settings.Hands_config_main)

# create a black canvas on which to draw ideals (for testing)
# blackCanvas = np.zeros((settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT, 3), dtype=np.uint8)

# load the ideal hand landmarks
ideals = []
with open('ideals.pkl', 'rb') as f:
    if not f:
        print("Ideal hands dataset (ideals.pkl) not found. Please run train.py.")
    ideals = pickle.load(f)
    
# open webcam with opencv
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    sys.exit()
    
# user input
# TODO: make this more robust, handle uppercase letters, etc.
# TODO: image as the start screen 
selection = input("Select a letter of the alphabet to train (capital, no J or Z)")  
print(selection)
print(type(selection))
print(ideals)

match_ideal = []
# search ideals and gram the matching letter
# atch_ideal = next((entry for entry in ideals if entry["letter"] == selection), None)
for entry in ideals:
    print(entry["letter"])
    if entry["letter"] == selection:
        match_ideal = entry
        print("found it!")
    

if not match_ideal:
    print(f"Letter '{selection}' not found.")
    sys.exit()

# arduino-style superloop
while True:
    
    # load a new frame as the first action in the loop
    ret, frame = cap.read()
    
    # if no new frame is detected, skip the whole loop and try again
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    hand = Hands.process(frame)
    
    if hand.multi_hand_landmarks:
        utils.drawLandmarks(hand, frame, "main", mp_draw, mp_hands)
        utils.drawLandmarks(match_ideal, frame, "main", mp_draw, mp_hands)
    else:
        utils.drawLandmarks(match_ideal, frame, "main", mp_draw, mp_hands)
    
    # user input and CLI
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    
# TODO: clever algorithm to implement a decision tree for hands 