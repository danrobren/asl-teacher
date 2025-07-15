# Main program with GUI and decision maker for letter detection

import cv2
import mediapipe as mp
import string
import time
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
    
# scale all the ideal letters to the webcam frame aspect ratio
ret = []
while not ret:
    ret, frame = cap.read()

# squish the letter along the horizontal axis so that it displays properly in the rectangular frame
f_height, f_width = frame.shape[:2]
for entry in ideals:
    entry['points']['x'] = entry['points']['x']*(f_height/f_width)
    
# TODO: add "return to menu" capability
# TODO: wrap the whole CLI menu and program in a state machine
mode = []
while mode not in [1, 2]:
    print("Choose Program Mode:")
    print("1: Letter Select")
    print("2: Minimum RMS Distance")
    mode = int(input(""))
    if mode not in [1, 2]:
        print("Please only enter a valid mode")
        print("")
    
match mode:
    case 1:
        # user input
        # TODO: make this more robust, handle uppercase letters, etc.
        selection = input("Select a letter of the alphabet to train (capital, no J or Z)")  

        match_ideal = []
        # search ideals and gram the matching letter
        # atch_ideal = next((entry for entry in ideals if entry["letter"] == selection), None)
        for entry in ideals:
            if entry["letter"] == selection:
                match_ideal = entry["points"]

        if not match_ideal:
            print(f"Letter '{selection}' not found.")
            sys.exit()
            
        prev_time = 1
        # arduino-style superloop
        while True:
            
            # load a new frame as the first action in the loop
            ret, frame = cap.read()
            
            # if no new frame is detected, skip the whole loop and try again
            if not ret:
                continue
            
            # get the hand sign that the user is making in the current video frame
            frame = cv2.flip(frame, 1)
            hand = Hands.process(frame)
            
            # drawConnections has to be first so the hands will be drawn over the conencting lines
            if hand.multi_hand_landmarks:
                utils.drawConnections(hand, match_ideal, frame)
                utils.drawLandmarks(hand, frame, 0, mp_draw, mp_hands)
                
            # always the ideal hand in every frame
            utils.drawLandmarks(match_ideal, frame, 0, mp_draw, mp_hands)
            
            # draw stats text on last
            cv2.rectangle(frame, (int(0.7*f_width), int(0)), (int(f_width), int(0.3*f_height)), (0, 0, 0), thickness=-1)
            
            # display score
            cv2.putText(frame, "RMS Dist = " + str(utils.rmsDist(hand, match_ideal)), 
                (int(0.7*f_width), int(0.03*f_height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # display frames per second
            current_time = time.time()
            cv2.putText(frame, "FPS = " + str(1/(current_time - prev_time)), 
                (int(0.7*f_width), int(0.08*f_height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
            prev_time = current_time
            
            # final display the frame for this loop
            cv2.imshow("ASL Teacher - Selected Letter" + selection, frame)

            # user input and CLI
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
    # minimum RMS distance method
    case 2:
        
        # initialize some things
        prev_time = 1
        match_letter = []
        while True: 
            # load a new frame as the first action in the loop
            ret, frame = cap.read()
            
            # if no new frame is detected, skip the whole loop and try again
            if not ret:
                continue
            
            # get the hand sign that the user is making in the current video frame
            frame = cv2.flip(frame, 1)
            hand = Hands.process(frame)
            f_height, f_width = frame.shape[:2]
        
            # setup for matching loop, make minDist start out huge
            match_ideal = []
            match_letter = []
            minDist = 100
            
            # search ideals and grab the letter with smallest RMS distance from letter in frame
            for entry in ideals:
                dist = utils.rmsDist(entry["points"], hand)
                if dist < minDist:
                    match_ideal = entry["points"]
                    match_letter = entry["letter"]
                    minDist = dist  
                    
            # drawConnections has to be first so the hands will be drawn over the conencting lines
            if hand.multi_hand_landmarks:
                utils.drawConnections(hand, match_ideal, frame)
                utils.drawLandmarks(hand, frame, 0, mp_draw, mp_hands)
                
            # always the ideal hand in every frame
            utils.drawLandmarks(match_ideal, frame, 0, mp_draw, mp_hands)
            
            # draw stats text on last
            cv2.rectangle(frame, (int(0.7*f_width), int(0)), (int(f_width), int(0.3*f_height)), (0, 0, 0), thickness=-1)
            
            cv2.putText(frame, "RMS Dist = " + str(utils.rmsDist(hand, match_ideal)), 
                        (int(0.7*f_width), int(0.03*f_height)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # display frames per second
            current_time = time.time()
            cv2.putText(frame, "FPS = " + str(1/(current_time - prev_time)), 
                (int(0.7*f_width), int(0.08*f_height)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
            prev_time = current_time
            
            # final display the frame for this loop
            cv2.imshow("ASL Teacher - Minimum RMS Distance", frame)

            # user input and CLI
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                

    
# TODO: clever algorithm to implement a decision tree for hands 