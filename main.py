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

# ─────────────────────────────────────────────────────────────
# UDP → Unity
UDP_IP   = "127.0.0.1"     # intended to run on same computer as Unity
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Optional: auto-launch a Unity build (place EXE next to this script)
EXE_NAME = "ASLUnityBridge.exe"
WIDTH    = 1600
HEIGHT   = 800

def launch_unity_windowed(exe_name=EXE_NAME, width=WIDTH, height=HEIGHT):
    exe_path = os.path.join(os.path.dirname(__file__), exe_name)
    if not os.path.isfile(exe_path):
        return
    args = [
        exe_path,
        "-screen-fullscreen", "0",
        "-screen-width",  str(width),
        "-screen-height", str(height),
    ]
    try:
        subprocess.Popen(args, shell=False)
        print("Launched Unity:", " ".join(args))
    except Exception as e:
        print("Failed to launch Unity exe:", e)

def _points_to_udp_list(points_dict):
    """
    Convert an 'ideal' points dict into a JSON-friendly list.
    points_dict: {"x": np.array(21), "y": np.array(21), "z": np.array(21)}
    returns: [{"x":..,"y":..,"z":..} x21]  or None on error
    """
    try:
        x, y, z = points_dict["x"], points_dict["y"], points_dict["z"]
        n = min(len(x), len(y), len(z), 21)
        return [{"x": float(x[i]), "y": float(y[i]), "z": float(z[i])} for i in range(n)]
    except Exception:
        return None

def send_udp_hand(hand_result, letter=None, ideal_points=None):
    """
    Sends JSON over UDP:
      {
        "present":   bool,
        "letter":    str|None,
        "landmarks": [{"x":..,"y":..,"z":..} x21],   # live frame (0..1 coords from MediaPipe)
        "ideal":     [{"x":..,"y":..,"z":..} x21]|null
      }
    """
    try:
        if hand_result and hand_result.multi_hand_landmarks:
            lm_list = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in hand_result.multi_hand_landmarks[0].landmark
            ]
            payload = {"present": True, "letter": letter, "landmarks": lm_list}
        else:
            payload = {"present": False, "letter": None, "landmarks": []}

        payload["ideal"] = _points_to_udp_list(ideal_points) if ideal_points is not None else None

        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        sock.sendto(data, (UDP_IP, UDP_PORT))
    except Exception:
        # UDP is fire-and-forget; swallow transient errors
        pass
# ─────────────────────────────────────────────────────────────

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
    print("Flipping ideal " + entry["letter"])
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

# Optionally launch Unity (if EXE present next to script)
launch_unity_windowed(EXE_NAME, WIDTH, HEIGHT)

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

            detected_letter = None
            if hand and hand.multi_hand_landmarks:
                minDist_tmp = 1e9
                for entry in ideals:
                    d = utils.rmsDist(entry["points"], hand)
                    if d < minDist_tmp:
                        minDist_tmp = d
                        detected_letter = entry["letter"]

            # Send live + ideal
            send_udp_hand(hand, detected_letter, match_ideal)

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
                 "Detected Letter = " + (detected_letter or "?")],
                frame
            )

            cv2.imshow("ASL Teacher - Selected Letter " + selection, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif mode == 2:
        # MINIMUM RMS with curl heuristics blended in
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            hand  = Hands.process(frame)

            # default via RMS
            match_ideal  = None
            match_letter = "?"
            minDist      = 1e9

            for entry in ideals:
                dist = utils.rmsDist(entry["points"], hand)
                if dist < minDist:
                    match_ideal  = entry["points"]
                    match_letter = entry["letter"]
                    minDist      = dist

            # curl-based override for closed-fist family (E, M, N, S, T)
            if hand and hand.multi_hand_landmarks:
                curls = utils.all_curls(hand)
                thumb_c, index_c, middle_c, ring_c, pinky_c = curls
                closed_thr = 1.4
                if (thumb_c > closed_thr and index_c > closed_thr and
                    middle_c > closed_thr and ring_c > closed_thr and
                    pinky_c > closed_thr):

                    lm = hand.multi_hand_landmarks[0].landmark
                    thumb_tip  = lm[4]
                    index_tip  = lm[8]
                    middle_tip = lm[12]
                    ring_tip   = lm[16]
                    pinky_tip  = lm[20]

                    # Determine handedness by index vs pinky
                    right_hand = index_tip.x < pinky_tip.x
                    # rank thumb across the curled stack (left-to-right changes with handedness)
                    if right_hand:
                        if   thumb_tip.x < index_tip.x:   match_letter = "S"
                        elif thumb_tip.x < middle_tip.x:  match_letter = "T"
                        elif thumb_tip.x < ring_tip.x:    match_letter = "N"
                        elif thumb_tip.x < pinky_tip.x:   match_letter = "M"
                        else:                              match_letter = "E"
                    else:
                        if   thumb_tip.x > index_tip.x:   match_letter = "S"
                        elif thumb_tip.x > middle_tip.x:  match_letter = "T"
                        elif thumb_tip.x > ring_tip.x:    match_letter = "N"
                        elif thumb_tip.x > pinky_tip.x:   match_letter = "M"
                        else:                              match_letter = "E"

                    # swap in the corresponding ideal
                    for entry in ideals:
                        if entry["letter"] == match_letter:
                            match_ideal = entry["points"]
                            break

            # Send live + ideal
            send_udp_hand(hand, match_letter, match_ideal)

            # Draw overlays
            if hand and hand.multi_hand_landmarks:
                utils.drawConnections(hand, match_ideal, frame)
                utils.drawLandmarks(match_ideal, frame, 0, mp_draw, mp_hands)
                utils.drawLandmarks(hand, frame, 0, mp_draw, mp_hands)

            score = 100 - 100 * utils.rmsDist(hand, match_ideal) if (hand and hand.multi_hand_landmarks) else 0

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time + 1e-9)
            prev_time = current_time

            utils.drawStats(
                [f"Score = {score:.2f}",
                 f"FPS = {fps:.2f}",
                 "Detected Letter = " + match_letter],
                frame
            )

            cv2.imshow("ASL Teacher - Minimum RMS Distance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        # CURL DECISION TREE MODE (falls back to RMS if not decisive)
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
            send_udp_hand(hand, match_letter if match_letter != "?" else None, match_ideal)

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
                 "Detected Letter = " + (match_letter or "?")],
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
