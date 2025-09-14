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

import utils      # utils.py custom helper functions
import settings   # settings.py constants and Hands configurations

print("Welcome to ASL Teacher")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
Hands    = mp_hands.Hands(**settings.Hands_config_main)

# Load ideal poses
ideals = []
with open('ideals.pkl', 'rb') as f:
    if not f:
        print("Ideal hands dataset (ideals.pkl) not found. Please run train.py.")
    ideals = pickle.load(f)

# Duplicate & flip ideals to support left/right hands
ideals_original = ideals.copy()
for entry in ideals_original:
    x_vals = entry["points"]["x"]           # numpy array
    x_flipped = 1.0 - x_vals                # horizontal flip in normalized space
    ideals.append({
        "letter": entry["letter"],
        "points": {
            "x": x_flipped,
            "y": entry["points"]["y"],
            "z": entry["points"]["z"]
        }
    })

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    sock.close()
    sys.exit(1)

# Grab a frame to learn aspect ratio
ret, frame = cap.read()
while not ret:
    ret, frame = cap.read()

# Scale all ideal X to match the camera aspect (so drawings overlay correctly)
f_height, f_width = frame.shape[:2]
for entry in ideals:
    entry['points']['x'] *= (f_height / f_width)

# Mode select
mode = 0
while mode not in [1, 2]:
    print("Choose Program Mode:")
    print("1: Letter Select")
    print("2: Detect Letter Sign")
    try:
        mode = int(input(""))
    except:
        mode = 0
    if mode not in [1, 2]:
        print("Please only enter a valid mode\n")

# Optionally launch Unity (if EXE present next to script)
utils.launch_unity_windowed(settings.EXE_NAME, settings.Unity_WIDTH, settings.Unity_HEIGHT)

try:
    if mode == 1:
        # LETTER SELECT
        selection = input("Select a letter of the alphabet to train (capital, no J or Z): ").strip()[:1]
        match_ideal = None
        for entry in ideals:
            if entry["letter"] == selection:
                match_ideal = entry["points"]
                break

        if match_ideal is None:
            print(f"Letter '{selection}' not found.")
            sys.exit(1)

        prev_time = time.time()
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            hand  = Hands.process(frame)

            # Send live + ideal
            utils.send_udp_hand(sock, hand, selection, match_ideal)

            # Draw overlays
            if hand and hand.multi_hand_landmarks:
                utils.drawConnections(hand, match_ideal, frame)
                utils.drawLandmarks(hand, frame, 0, mp_draw, mp_hands)

            utils.drawLandmarks(match_ideal, frame, 0, mp_draw, mp_hands)

            score = 100 - 100 * utils.rmsDist(hand, match_ideal) if (hand and hand.multi_hand_landmarks) else 0

            # FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time + 1e-9)
            prev_time = current_time

            utils.drawStats(
                [f"Score = {score:.2f}",
                 f"FPS = {fps:.2f}",
                 "Selected Letter = " + selection,
                 "Press 'Q' to Quit"], 
                frame
            )

            cv2.imshow("ASL Teacher - Selected Letter " + selection, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # CURL DECISION TREE MODE
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            hand  = Hands.process(frame)

            match_ideal  = None
            match_letter = "?"

            if hand and hand.multi_hand_landmarks:
                # Your curl metrics
                curls = utils.all_curls(hand)
                thumb_c, index_c, middle_c, ring_c, pinky_c = curls

                # Simple examples – expand as you like
                open_thr   = 0.6
                closed_thr = 1.4

                lm = hand.multi_hand_landmarks[0].landmark
                index_tip  = lm[8]
                thumb_tip  = lm[4]
                pinky_tip  = lm[20]
                right_hand = index_tip.x < pinky_tip.x

                # L: index extended, thumb out, others curled
                if (index_c < open_thr and
                    thumb_c  < open_thr and
                    middle_c > closed_thr and ring_c > closed_thr and pinky_c > closed_thr):
                    match_letter = "L"

                # Fist family via same heuristic as mode 2
                elif (thumb_c > closed_thr and index_c > closed_thr and
                      middle_c > closed_thr and ring_c > closed_thr and pinky_c > closed_thr):
                    if right_hand:
                        if   thumb_tip.x < index_tip.x:   match_letter = "S"
                        elif thumb_tip.x < lm[12].x:      match_letter = "T"
                        elif thumb_tip.x < lm[16].x:      match_letter = "N"
                        elif thumb_tip.x < lm[20].x:      match_letter = "M"
                        else:                              match_letter = "E"
                    else:
                        if   thumb_tip.x > index_tip.x:   match_letter = "S"
                        elif thumb_tip.x > lm[12].x:      match_letter = "T"
                        elif thumb_tip.x > lm[16].x:      match_letter = "N"
                        elif thumb_tip.x > lm[20].x:      match_letter = "M"
                        else:                              match_letter = "E"
                else:
                    # fallback to RMS best match
                    minDist = 1e9
                    for entry in ideals:
                        dist = utils.rmsDist(entry["points"], hand)
                        if dist < minDist:
                            match_ideal  = entry["points"]
                            match_letter = entry["letter"]
                            minDist      = dist

            # If decision tree picked a letter, ensure its ideal is set
            if match_letter != "?" and match_ideal is None:
                for entry in ideals:
                    if entry["letter"] == match_letter:
                        match_ideal = entry["points"]
                        break

            # Send live + ideal
            utils.send_udp_hand(sock, hand, match_letter if match_letter != "?" else None, match_ideal)

            # Draw overlays
            if hand and hand.multi_hand_landmarks and match_ideal is not None:
                utils.drawConnections(hand, match_ideal, frame)
                utils.drawLandmarks(match_ideal, frame, 0, mp_draw, mp_hands)
                utils.drawLandmarks(hand, frame, 0, mp_draw, mp_hands)

            score = 100 - 100 * utils.rmsDist(hand, match_ideal) if (hand and hand.multi_hand_landmarks and match_ideal is not None) else 0

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time + 1e-9)
            prev_time = current_time

            utils.drawStats(
                [f"Score = {score:.2f}",
                 f"FPS = {fps:.2f}",
                 "Detected Letter = " + (match_letter or "?"),
                 "Press 'Q' to Quit"],
                 
                frame
            )

            cv2.imshow("ASL Teacher - Curl Decision Tree", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # Cleanup
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