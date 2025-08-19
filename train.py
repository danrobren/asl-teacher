# This file creates the ideals.pkl file with an averaged hand landmark dataset for every letter
# Assumption is that SignAlphaSet and its folder convention are used as training data
import string
import sys
import pickle
import cv2
import numpy as np
import mediapipe as mp

import utils    # utils.py custom defines functions
import settings # settings.py constants and Hands configurations

# debug macro, displays each image as it is processed 
debug = 0

# number of images per letter to process
nimpl = 100

# initialize empty set to store averaged ideal hands
ideals = []

# do not include Z or J because they have movement in the sign; out of scope for now
letters = [ch for ch in string.ascii_uppercase if ch not in ('J', 'Z')]

# instantiate mediapipe drawing object
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# initialize hands processor by calling the constructor directly 
Hands = mp_hands.Hands(**settings.Hands_config_train)


for letter in letters:
    
    # initialize the number of images that had letters detected in them
    detections = 0
    
    # initialize a structure to hold the each of the sums of all the reltive x, y, z 
    # such that they may be averaged (divided by the total number of pictures processed) at the end
    # we instantiate this at the start of the letters loop because we want to clear it after each letter
    totals = { 'x': np.zeros(21), 'y': np.zeros(21), 'z': np.zeros(21)}

    for i in range(0, nimpl):
        
        # load the image and process it 
        path = "dataset/SignAlphaSet/" + letter + "/" + letter + "_" + str(i) + ".jpg" # path to each letter image
        if debug: 
            print(path)
        #image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) # open image and convert it to RGB
        image = cv2.imread(path)
        if image is None:
            print("No image found at ./" + path)
            print("Download SignAlphaSet at: https://data.mendeley.com/datasets/8fmvr9m98w/2")
            continue
        
        hand = Hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # skip to next image if there are no hands in this one
        if not (hasattr(hand, 'multi_hand_landmarks') and hand.multi_hand_landmarks):
            print("No hand detected, continuing")
            continue
        
        if letter == 'F':
            print(hand.multi_hand_landmarks[0])
        
        # if the above if statement returned false and we're still in the loop that means there's a hand in the frame
        # increment the number of detections
        detections += 1
        
        # print(f"Detected {len(hand.multi_hand_landmarks)} hand(s) for letter {letter}, index {detections}" )
        
        # some of the hands are mirrored across the veritcal axis; dataset contains left and right handed symbols
        # we need to have the ideal case be averaging across only right signs
        # image flip criteria is hand sign dependant, so we need tailored behavior for each sign
        match letter:
            case 'A':
                # we flip the image about verical axis if thumb tip left of pinky 1st joint 
                # thumb tip is the 5th element (index 4) of the landmarks, x coordinate is normalized from [0 1], 0= left edge
                if hand.multi_hand_landmarks[0].landmark[4].x < hand.multi_hand_landmarks[0].landmark[18].x:
                    hand = utils.flipHand(hand)
                
            case 'B':
                # flip if index root left of pinky root
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    hand = utils.flipHand(hand) 
     
            case 'C':
                # flip if thumb is pointing right
                if hand.multi_hand_landmarks[0].landmark[4].x < hand.multi_hand_landmarks[0].landmark[3].x:
                    hand = utils.flipHand(hand) 
                
            case 'D':
                # flip if pointer finger tip left of pinky tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[20].x:
                    hand = utils.flipHand(hand) 
                
            case 'E':
            # flip if pointer finger tip left of pinky tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[20].x:
                    hand = utils.flipHand(hand) 
                
            case 'F':
            # flip if pointer finger tip left of pinky tip
                if hasattr(hand, 'multi_hand_landmarks') and hand.multi_hand_landmarks:
                    print("hand contains a mediapipe hand")
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[20].x:
                    hand = utils.flipHand(hand) 
                
            case 'G':
            # flip if pointer knuckle left of pointer tip
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[8].x:
                    hand = utils.flipHand(hand) 
                
            case 'H':
            # flip if pointer knuckle left of pointer tip
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[8].x:
                    hand = utils.flipHand(hand) 
                
            case 'I':
            # flip if pointer first joint left of pinky tip
                if hand.multi_hand_landmarks[0].landmark[6].x < hand.multi_hand_landmarks[0].landmark[20].x:
                    hand = utils.flipHand(hand)
                    
            case 'K':
            # flip if pointer tip left of middle finger tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[12].x:
                    hand = utils.flipHand(hand)
                
            case 'L':
            # flip if thumb tip left of thumb knuckle
                if hand.multi_hand_landmarks[0].landmark[4].x < hand.multi_hand_landmarks[0].landmark[2].x:
                    hand = utils.flipHand(hand)
            
            case 'M':
            # flip if pointer first joint left of pinky first joint
                if hand.multi_hand_landmarks[0].landmark[6].x < hand.multi_hand_landmarks[0].landmark[18].x:
                    hand = utils.flipHand(hand)
                
            case 'N':
            # flip if pointer first joint left of pinky first joint
                if hand.multi_hand_landmarks[0].landmark[6].x < hand.multi_hand_landmarks[0].landmark[18].x:
                    hand = utils.flipHand(hand)
                
            case 'O':
            # flip if pointer knuckle left of thumb knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[3].x:
                    hand = utils.flipHand(hand)
                
            case 'P':
            # flip if pointer knuckle left of pointer tip
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[8].x:
                    hand = utils.flipHand(hand) 
                
            case 'Q':
            # flip if pointer knuckle left of pointer tip
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[8].x:
                    hand = utils.flipHand(hand)
                       
            case 'R':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    hand = utils.flipHand(hand)
                
            case 'S':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    hand = utils.flipHand(hand)
                
            case 'T':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    hand = utils.flipHand(hand)
                
            case 'U':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    hand = utils.flipHand(hand)
                
            case 'V':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    hand = utils.flipHand(hand)
                    
            case 'W':
            # flip if pointer tip left of ring finger tip
                if hand.multi_hand_landmarks[0].landmark[9].x < hand.multi_hand_landmarks[0].landmark[5].x:
                    hand = utils.flipHand(hand)
                
            case 'X':
            # flip if thumb knuckle left of thumb tip
                if hand.multi_hand_landmarks[0].landmark[2].x < hand.multi_hand_landmarks[0].landmark[4].x:
                    hand = utils.flipHand(hand)
                
            case 'Y':
            # flip if pinky tip left of thumb tip
                if hand.multi_hand_landmarks[0].landmark[20].x < hand.multi_hand_landmarks[0].landmark[4].x:
                    hand = utils.flipHand(hand)
                
            case _:
                print('default case of letter matching; error!')
        # end match letter
        
        # what really matters for hand detection is not the absolute position of the hand relative to the image frame
        # we don't care whether the symbol is at the edge of the frame or the center
        # we want the **relative** positions of the landmarks relative to the wrist
        # this way in the main program we can extract the locations of each point relative to the wrist and compare those to the ideals
        # we do this by subtracting each node's position from the root node
        # iterate over all 21 points in mediapipe hands objects
        if hand.multi_hand_landmarks:
            for hand_landmarks in hand.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    totals['x'][idx] += landmark.x
                    totals['y'][idx] += landmark.y
                    totals['z'][idx] += landmark.z
        
        #print(totals)
                
        # key press to clear images or quit in debug mode
        if debug:
            k = cv2.waitKey(0)
            if k == 27 or k == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
            else: 
                cv2.destroyAllWindows()
    
    
    # store the hand in ideals so we can pickle it later  
    # this is just storing the last hand read
    # we can't store the hand objects directly because they contain all sorts of C++ and tensorflow nonsense; not pickleable
    # instead we just need the x, y, z values along with a letter ID, so store hand 0 (there should only be one)
    # export = []
    # for p in range(0, 21):
    #     export.append((totals['y'][p]/nimpl, totals ['y'][p]/nimpl, totals['z'][p]/nimpl))
    
    # divide each total by the number of hands detected over the image set
    print("number of detections for letter" + str(letter) + " = " + str(detections))
    totals['x'] = totals['x']/detections
    totals['y'] = totals['y']/detections
    totals['z'] = totals['z']/detections
        
    ideals.append({"letter": letter, "points": totals})
    print(f"Idea hand {letter} appended, total hands stored: {len(ideals)}")
    
     
for entry in ideals:
    # re-instantiate blackCanvas every time to clear it of the previous hands
    blackCanvas = np.zeros((settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT, 3), dtype=np.uint8)
    letter = entry["letter"]
    points = entry["points"]
    utils.drawLandmarks(points, blackCanvas, "Ideal Averaged " + letter, mp_draw, mp_hands)
    k = cv2.waitKey(0)
    if k == 27 or k == ord('q'):
        cv2.destroyAllWindows()
        sys.exit()
    else: 
        cv2.destroyAllWindows()

# save ideal hands to a file ideals.pkl
with open("ideals.pkl", "wb") as f:
    #print(ideals)
    pickle.dump(ideals, f)