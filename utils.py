# Custom defined functions to help main.py and train.py
import cv2
import string
import sys
from types import SimpleNamespace
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# TODO: proper documentation with inputs, outputs, and behavior for each function
# TODO: length error checking for 21 landmarks in each obejct in all functions

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

    # convert the dict from of hand into a mediapipe object so we can use mp functions for printing
    if isinstance(hand, dict) and all(k in hand for k in ['x', 'y', 'z']):
        
        # dummy mediapipe object
        convert_hand = SimpleNamespace()
        convert_hand.multi_hand_landmarks = []
        convert_hand_landmarks = landmark_pb2.NormalizedLandmarkList()

        # store all the points from the dict into the dummy mediapipe object
        for i in range(21):
            lm = landmark_pb2.NormalizedLandmark()
            lm.x = float(hand['x'][i])
            lm.y = float(hand['y'][i])
            lm.z = float(hand['z'][i])
            convert_hand_landmarks.landmark.append(lm) 

        # overwrite the original hand with the mediapipe object
        convert_hand.multi_hand_landmarks.append(convert_hand_landmarks)
        hand = convert_hand

    # draw for MediaPipe hand object
    if hasattr(hand, 'multi_hand_landmarks') and hand.multi_hand_landmarks:
        for hand_landmarks in hand.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            # get the min and max Z values to show geryscaled "closeness" dots
            z_vals = [lm.z for lm in hand_landmarks.landmark]
            min_z, max_z = min(z_vals), max(z_vals)

            for idx, landmark in enumerate(hand_landmarks.landmark):
                
                norm_z = (landmark.z - min_z) / (max_z - min_z + 1e-6)  # +epsilon to avoid divide by zero
                brightness = int(255 * (1 - norm_z))  # Closer = brighter
                zcolor = (brightness, brightness, brightness)
                
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                hue = int(255 * idx / 21)
                color = tuple(int(c) for c in cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
                cv2.circle(image, (cx, cy), 5, color, thickness=-1)
                cv2.circle(image, (cx, cy), 3, zcolor, thickness=-1)
                cv2.putText(image, str(idx), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    else:
        print("drawLandmarks: unrecognized hand format or no hand landmarks found.")

    # only display the image if a title is given
    # this way the function can be used to add landmarks to a frame without displaying
    # this is because we want text drawn on last to overlay the stats on top of everything
    if title:
        cv2.imshow(title, image)
        return
    else:
        return image

# flips a hand about the vertical axis
# used to normalize all the ieal hands in the training to either left or right
# takes the image and and returns the flipped hand
# has display capability if debug
def flipHand(image, debug, title, mp_draw, mp_hands, Hands):
    image = cv2.flip(image, 1) # code 1 for horizontal flip
    hand = Hands.process(image) # reprocess hands for flipped image
    if debug:
        print(title)
        drawLandmarks(hand, image, title, mp_draw, mp_hands)
    return hand

# computes the root mean square distance between all the points in two hands
# this can take inputs as mediaPipe hands or tuple set format hands
# tuple sets are dicts of the form { 'x': np.zeros(21), 'y': np.zeros(21), 'z': np.zeros(21)} but with filled x, y, z
# only uses one hand if multiple hands are detected
def rmsDist(handA, handB):
    
    # error checking for null hands and no hands detected
    if not handA:
        # print("rmsDist HandA Null; returning")
        return -1
    elif not handB:
        # print("rmsDist HandB Null; returning")
        return -2
    elif hasattr(handA, 'multi_hand_landmarks') and not handA.multi_hand_landmarks:
        # print("rmsDist HandA is medapipe object wtih no hands detected; returning")
        return -3
    elif hasattr(handB, 'multi_hand_landmarks') and not handB.multi_hand_landmarks:
        # print("rmsDist HandB is medapipe object with no hands detected; returning")
        return -4
    # elif hasattr(handA, 'multi_hand_landmarks') and handA.multi_hand_landmarks[1]:
    #     print("Warning: rmsDist HandA is medapipe object wtih multiple hands detected")
    # elif hasattr(handB, 'multi_hand_landmarks') and handB.multi_hand_landmarks[1]:
    #     print("Warning: rmsDist HandB is medapipe object with multiple hands detected")
    
    # extract all the (x, y, z) from handA and handB if they are mediapipe objects
    if hasattr(handA, 'multi_hand_landmarks'):
        handA_convert =  { 'x': np.zeros(21), 'y': np.zeros(21), 'z': np.zeros(21)}
        for hand_landmarks in handA.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                handA_convert['x'][idx] = landmark.x
                handA_convert['y'][idx] = landmark.y
                handA_convert['z'][idx] = landmark.z
            handA = handA_convert
            
    if hasattr(handB, 'multi_hand_landmarks'):
        handB_convert =  { 'x': np.zeros(21), 'y': np.zeros(21), 'z': np.zeros(21)}
        for hand_landmarks in handB.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                handB_convert['x'][idx] = landmark.x
                handB_convert['y'][idx] = landmark.y
                handB_convert['z'][idx] = landmark.z
            handB = handB_convert
    
    distPts = []
    
    # iterate over each landmark
    for i in range(0, 21):
        # calculate the length of the difference vector between point i of handA and handB
        distPts.append(np.sqrt(
            pow(handA['x'][i] - handB['x'][i], 2) +
            pow(handA['y'][i] - handB['y'][i], 2) +
            pow(handA['z'][i] - handB['z'][i], 2)
        ))
        
    # square then mean then root to get RMS distance
    return np.sqrt(np.mean(np.square(distPts)))

def drawConnections(handA, handB, image):

    # convert everything to tuple form first
    # extract all the (x, y, z) from handA and handB if they are mediapipe objects
    if hasattr(handA, 'multi_hand_landmarks'):
        handA_convert =  { 'x': np.zeros(21), 'y': np.zeros(21), 'z': np.zeros(21)}
        for hand_landmarks in handA.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                handA_convert['x'][idx] = landmark.x
                handA_convert['y'][idx] = landmark.y
                handA_convert['z'][idx] = landmark.z
            handA = handA_convert
            
    if hasattr(handB, 'multi_hand_landmarks'):
        handB_convert =  { 'x': np.zeros(21), 'y': np.zeros(21), 'z': np.zeros(21)}
        for hand_landmarks in handA.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                handB_convert['x'][idx] = landmark.x
                handB_convert['y'][idx] = landmark.y
                handB_convert['z'][idx] = landmark.z
            handB = handB_convert

    h, w, _ = image.shape

    for i in range(21):
        x1, y1 = int(handA['x'][i] * w), int(handA['y'][i] * h)
        x2, y2 = int(handB['x'][i] * w), int(handB['y'][i] * h)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
        
# TODO: put display params (font, size, colors, etc) in settings.py
def drawStats(stats, image):
    f_height, f_width = image.shape[:2]
    
    cv2.rectangle(image, (int(0.7*f_width), int(0)), (int(f_width), int(len(stats)*0.055*f_height)), (0, 0, 0), thickness=-1)

    count = 1
    for text in stats:
        cv2.putText(image, text, (int(0.72*f_width), int(0.03+0.05*count*f_height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        count = count +1
    return image