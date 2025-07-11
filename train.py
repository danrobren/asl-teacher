print("start")
# This file creates the ideals.pkl file with an averaged hand landmark dataset for every letter
# Assumption is that SignAlphaSet and its folder convention are used as training data

import string
import sys
import pickle
import json
import cv2
import numpy as np
import mediapipe as mp
import utils # utils.py custom defines functions

# Load the settings file
with open("settings.json", "r") as f:
    settings = json.load(f)

# debug macro
debug = 1

# number of images per letter to process
nimpl = 2

# initialize empty set to store averaged ideal hands
ideals = []

# do not include Z or J because they have movement in the sign; out of scope for now
letters = [ch for ch in string.ascii_uppercase if ch not in ('J', 'Z')]

# instantiate mediapipe drawing object
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# initialize hands processor by calling the constructor cirectly 
Hands = mp_hands.Hands(**settings["train"])

# initialize empty hand data object to contain the loaded hands
hand = []

for letter in letters:
    print("start letters")
    
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
        hand = Hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # skip the rest of the loop if no hands detected
        #if not hand.multi_hand_landmarks:
            #continue
        
        # some of the hands are mirrored across the veritcal axis; dataset contains left and right handed symbols
        # we need to have the ideal case be averaging across only right signs
        # image flip criteria is hand sign dependant, so we need tailored behavior for each sign
        match letter:
            case 'A':
                # we flip the image about verical axis if thumb tip left of pinky 1st joint 
                # thumb tip is the 5th element (index 4) of the landmarks, x coordinate is normalized from [0 1], 0= left edge
                if hand.multi_hand_landmarks[0].landmark[4].x < hand.multi_hand_landmarks[0].landmark[18].x:
                    print("flipping hand")
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)   
                
            case 'B':
                # flip if index root left of pinky root
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
     
            case 'C':
                # flip if thumb is pointing right
                if hand.multi_hand_landmarks[0].landmark[4].x < hand.multi_hand_landmarks[0].landmark[3].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'D':
                # flip if pointer finger tip left of pinky tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[20].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'E':
            # flip if pointer finger tip left of pinky tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[20].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'F':
            # flip if pointer finger tip left of pinky tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[20].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'G':
            # flip if pointer knuckle left of pointer tip
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[8].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'H':
            # flip if pointer knuckle left of pointer tip
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[8].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'I':
            # flip if pointer first joint left of pinky tip
                if hand.multi_hand_landmarks[0].landmark[6].x < hand.multi_hand_landmarks[0].landmark[20].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                    
            case 'K':
            # flip if pointer tip left of middle finger tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[12].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)   
                
            case 'L':
            # flip if thumb tip left of thumb knuckle
                if hand.multi_hand_landmarks[0].landmark[4].x < hand.multi_hand_landmarks[0].landmark[2].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
            
            case 'M':
            # flip if pointer first joint left of pinky first joint
                if hand.multi_hand_landmarks[0].landmark[6].x < hand.multi_hand_landmarks[0].landmark[18].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 'N':
            # flip if pointer first joint left of pinky first joint
                if hand.multi_hand_landmarks[0].landmark[6].x < hand.multi_hand_landmarks[0].landmark[18].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 'O':
            # flip if pointer knuckle left of thumb knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[3].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 'P':
            # flip if pointer knuckle left of pointer tip
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[8].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'Q':
            # flip if pointer knuckle left of pointer tip
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[8].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                       
            case 'R':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 'S':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 'T':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 'U':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 'V':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)
                    
            case 'W':
            # flip if pointer tip left of ring finger tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[16].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)   
                
            case 'Z':
            # flip if thumb knuckle left of thumb tip
                if hand.multi_hand_landmarks[0].landmark[2].x < hand.multi_hand_landmarks[0].landmark[4].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)   
                
            case 'Y':
            # flip if pinky tip left of thumb tip
                if hand.multi_hand_landmarks[0].landmark[20].x < hand.multi_hand_landmarks[0].landmark[4].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)   
                
            case _:
                print('default case of letter matching; error!')
        # end match letter
        
        # what really matters for hand detection is not the absolute position of the hand relative to the image frame
        # we don't care whether the symbol is at the edge of the frame or the center
        # we want the **relative** positions of the landmarks relative to the wrist
        # this way in the main program we can extract the locations of each point relative to the wrist and compare those to the ideals
        # we do this by subtracting each node's position from the root node
        # iterate over all 21 points in mediapipe hands objects
        for p in range(0, 21):
            totals['x'][p] += hand.multi_hand_landmarks[0].landmark[p].x
            totals ['y'][p] += hand.multi_hand_landmarks[0].landmark[p].y
            totals['z'][p] += hand.multi_hand_landmarks[0].landmark[p].z
                
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
    export = []    
    for p in range(0, 21):
        export.append((totals['x'][p]/nimpl, totals ['y']/nimpl, totals['z']/nimpl))
        
    ideals.append({"letter" : letter, "landmarks" : export})
    print(f"Idea hand {letter} appended, total hands stored: {len(ideals)}")      
    
    
# print all the hands on a black canvas 
# create a black canvas on which to draw ideals (for testing)
blackCanvas = np.zeros((296, 296, 3), dtype=np.uint8)
#for i in range(0, 24):



# save ideal hands to a file ideals.pkl
with open("ideals.pkl", "wb") as f:
    #print(ideals)
    pickle.dump(ideals, f)