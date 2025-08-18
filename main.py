# Main program with GUI and decision maker for letter detection

import cv2
import mediapipe as mp
import string
import time
import sys
import pickle
import numpy as np
import json
import socket
import subprocess
import os

import utils  # utils.py custom defines functions
import settings  # settings.py constants and Hands configurations

print("Welcome to ASL Teacher")

# constants and settings
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# initialize socket connection for unity
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# initialize hand landmarker (hands)
Hands = mp_hands.Hands(**settings.Hands_config_main)

# load the ideal hand landmarks
ideals = []
with open('ideals.pkl', 'rb') as f:
    if not f:
        print("Ideal hands dataset (ideals.pkl) not found. Please run train.py.")
    ideals = pickle.load(f)

# handle detecting left or right handed symbols by appending a flipped version of every hand to ideals
ideals_original = ideals.copy()
for entry in ideals_original:
    print("Flipping ideal " + entry["letter"])

    # Flip x-values as a NumPy array
    x_vals = entry["points"]["x"]
    x_flipped = 1.0 - x_vals  # NumPy vectorized subtraction

    # Build the new flipped 'points' dict
    points_flipped = {
        "x": x_flipped,
        "y": entry["points"]["y"],
        "z": entry["points"]["z"]
    }

    # Append the flipped version with the same letter
    ideals.append({
        "letter": entry["letter"],
        "points": points_flipped
    })

# open webcam with opencv
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    sock.close()
    sys.exit()

# scale all the ideal letters to the webcam frame aspect ratio
ret = False
frame = None
while not ret:
    ret, frame = cap.read()

# squish the letter along the horizontal axis so that it displays properly in the rectangular frame
f_height, f_width = frame.shape[:2]
for entry in ideals:
    entry['points']['x'] *= (f_height / f_width)

# TODO: add "return to menu" capability
# TODO: wrap the whole CLI menu and program in a state machine
mode = 0
while mode not in [1, 2, 3]:
    print("Choose Program Mode:")
    print("1: Letter Select")
    print("2: Minimum RMS Distance")
    print("3: Curl Decision Tree")
    try:
        mode = int(input(""))
    except:
        mode = 0
    if mode not in [1, 2, 3]:
        print("Please only enter a valid mode\n")

try:
    #launches the .exe window afte decision is made
    utils.launch_unity_windowed(settings.EXE_NAME, settings.UNITY_WIDTH, settings.UNITY_HEIGHT)
    
    match mode:
# ─────────────────────────────────────────────────────────────
# letter select case
        case 1:
            
            # initialize time for FPS calculation
            prev_time = time.time()
            
            # capture user input for selected ASL letter
            selection = input("Select a letter of the alphabet to train (capital, no J or Z)")

            # initialize empty ideal matched letter
            match_ideal = []
            
            # search ideals and grab the matching letter
            for entry in ideals:
                if entry["letter"] == selection:
                    match_ideal = entry["points"]

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

                # get the hand sign that the user is making in the current video frame
                frame = cv2.flip(frame, 1)
                hand = Hands.process(frame)

                detected_letter = None  # We add a quick matching loop to get current letter for unity visualization
                if hand and hand.multi_hand_landmarks:
                    minDist_tmp = 1e9
                    for entry in ideals:
                        d = utils.rmsDist(entry["points"], hand)
                        if d < minDist_tmp:
                            minDist_tmp = d
                            detected_letter = entry["letter"]

                # UDP send to Unity for 3D display
                utils.send_udp_hand(hand, sock, detected_letter)  

                # only draw the user's hand if it is detected in the frame 
                if hand.multi_hand_landmarks:
                    utils.drawConnections(hand, match_ideal, frame)
                    utils.drawLandmarks(hand, frame, 0, mp_draw, mp_hands)
                    
                # draw dots representing the hand landmarks of the ideal hand onto the openCV frame
                # always draw the ideal hand landmarks
                utils.drawLandmarks(match_ideal, frame, 0, mp_draw, mp_hands)

                # calculate score
                if hand.multi_hand_landmarks:
                    score = 100 - 100 * utils.rmsDist(hand, match_ideal)
                else:
                    score = 0
                    
                # calculate instantaneous frames per second in this frame
                current_time = time.time()
                fps = 1 / (current_time - prev_time + 1e-9)
                prev_time = current_time

                utils.drawStats(
                    [f"Score = {score:.2f}",
                     f"FPS = {fps:.2f}",
                     "Selected Letter = " + selection,
                     "Detected Letter = " + (detected_letter or "?")],  # NEW: show detected letter
                    frame
                )

                # final display the frame for this loop
                cv2.imshow("ASL Teacher - Selected Letter " + selection, frame)

                # user input and CLI
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
# ─────────────────────────────────────────────────────────────
# minimum RMS distance case
        case 2:
            print("Minimum RMS Distance Algorithm Selected")
            
            
# ─────────────────────────────────────────────────────────────
# augmented curl tree case
        case 3:
            print("Augmented Curl Tree Algorithm Selected")
            prev_time = time.time()
            while True:
                # load a new frame as the first action in the loop
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                hand = Hands.process(frame)
                f_height, f_width = frame.shape[:2]

                # setup for matching loop, make minDist start out huge
                match_ideal = []
                match_letter = "A"
                minDist = 1002

                # search ideals and grab the letter with smallest RMS distance from letter in frame
                for entry in ideals:
                    dist = utils.rmsDist(entry["points"], hand)
                    if dist < minDist:
                        match_ideal = entry["points"]
                        match_letter = entry["letter"]
                        minDist = dist

                # integrate curl-based heuristics for high-curl letters
                if hand and hand.multi_hand_landmarks:
                    curls = utils.all_curls(hand)
                    thumb, index, middle, ring, pinky = curls
                    closed_thr = 1.4
                    if (thumb > closed_thr and index > closed_thr and
                        middle > closed_thr and ring > closed_thr and
                        pinky > closed_thr):
                        lm = hand.multi_hand_landmarks[0].landmark
                        thumb_tip = lm[4]
                        index_tip = lm[8]
                        middle_tip = lm[12]
                        ring_tip = lm[16]
                        pinky_tip = lm[20]
                        right_hand = index_tip.x < pinky_tip.x
                        if right_hand:
                            if thumb_tip.x < index_tip.x:
                                match_letter = "S"
                            elif thumb_tip.x < middle_tip.x:
                                match_letter = "T"
                            elif thumb_tip.x < ring_tip.x:
                                match_letter = "N"
                            elif thumb_tip.x < pinky_tip.x:
                                match_letter = "M"
                            else:
                                match_letter = "E"
                        else:
                            if thumb_tip.x > index_tip.x:
                                match_letter = "S"
                            elif thumb_tip.x > middle_tip.x:
                                match_letter = "T"
                            elif thumb_tip.x > ring_tip.x:
                                match_letter = "N"
                            elif thumb_tip.x > pinky_tip.x:
                                match_letter = "M"
                            else:
                                match_letter = "E"
                        for entry in ideals:
                            if entry["letter"] == match_letter:
                                match_ideal = entry["points"]
                                break


                utils.send_udp_hand(hand, sock, match_letter)  # UDP

                # drawConnections has to be first so the hands will be drawn over the connecting lines
                if hand.multi_hand_landmarks:
                    utils.drawConnections(hand, match_ideal, frame)
                    utils.drawLandmarks(match_ideal, frame, 0, mp_draw, mp_hands)
                    utils.drawLandmarks(hand, frame, 0, mp_draw, mp_hands)

                # calculate score
                if hand.multi_hand_landmarks:
                    score = 100 - 100 * utils.rmsDist(hand, match_ideal)
                else:
                    score = 0

                # calculate frames per second
                current_time = time.time()
                fps = 1 / (current_time - prev_time + 1e-9)
                prev_time = current_time

                utils.drawStats([f"Score = {score:.2f}",
                                f"FPS = {fps:.2f}",
                                "Detected Letter = " + match_letter],
                                frame)

                cv2.imshow("ASL Teacher - Minimum RMS Distance", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        

finally:
    try:
        cap.release()
    except:
        pass
    try:
        cv2.destroyAllWindows()
    except:
        pass
    try:
        sock.close()
    except:
        pass

