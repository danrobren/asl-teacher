# This file creates the ideals.pkl file with an averaged hand landmark dataset for every letter
# Assumption is that SignAlphaSet and its folder convention are used as training data

import string
import sys
import pickle
import json
import cv2
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
    for i in range(0, nimpl):
        
        # load the image and process the 
        path = "dataset/SignAlphaSet/" + letter + "/" + letter + "_" + str(i) + ".jpg" # path to each letter image
        if debug: 
            print(path)
        #image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) # open image and convert it to RGB
        image = cv2.imread(path)
        print("image read")
        hand = Hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print("hand process")
        
        # draw the non-flipped hand, we will clear this after drawing the flipped hand
        if debug: 
            utils.drawLandmarks(hand, image, path, mp_draw, mp_hands)
        
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
     
            case 'c':
                # flip if thumb is pointing right
                if hand.multi_hand_landmarks[0].landmark[4].x < hand.multi_hand_landmarks[0].landmark[3].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'd':
                # flip if pointer finger tip left of pinky tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[20].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'e':
            # flip if pointer finger tip left of pinky tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[20].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'f':
            # flip if pointer finger tip left of pinky tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[20].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'g':
            # flip if pointer knuckle left of pointer tip
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[8].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'h':
            # flip if pointer knuckle left of pointer tip
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[8].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'i':
            # flip if pointer first joint left of pinky tip
                if hand.multi_hand_landmarks[0].landmark[6].x < hand.multi_hand_landmarks[0].landmark[20].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                    
            case 'k':
            # flip if pointer tip left of middle finger tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[12].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)   
                
            case 'l':
            # flip if thumb tip left of thumb knuckle
                if hand.multi_hand_landmarks[0].landmark[4].x < hand.multi_hand_landmarks[0].landmark[2].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
            
            case 'm':
            # flip if pointer first joint left of pinky first joint
                if hand.multi_hand_landmarks[0].landmark[6].x < hand.multi_hand_landmarks[0].landmark[18].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 'n':
            # flip if pointer first joint left of pinky first joint
                if hand.multi_hand_landmarks[0].landmark[6].x < hand.multi_hand_landmarks[0].landmark[18].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 'o':
            # flip if pointer knuckle left of thumb knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[3].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 'p':
            # flip if pointer knuckle left of pointer tip
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[8].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)    
                
            case 'q':
            # flip if pointer knuckle left of pointer tip
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[8].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                       
            case 'r':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 's':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 't':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 'u':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands) 
                
            case 'v':
            # flip if pointer knuckle left of pinky knuckle
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)
                    
            case 'w':
            # flip if pointer tip left of ring finger tip
                if hand.multi_hand_landmarks[0].landmark[8].x < hand.multi_hand_landmarks[0].landmark[16].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)   
                
            case 'x':
            # flip if thumb knuckle left of thumb tip
                if hand.multi_hand_landmarks[0].landmark[2].x < hand.multi_hand_landmarks[0].landmark[4].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)   
                
            case 'y':
            # flip if pinky tip left of thumb tip
                if hand.multi_hand_landmarks[0].landmark[20].x < hand.multi_hand_landmarks[0].landmark[4].x:
                    utils.flipHand(image, debug, "Flipped " + letter + "_" + str(i), mp_draw, mp_hands, Hands)   
                
            case _:
                print('default case of letter matching; error!')
        
        # add all the x and y values to create a sum for each of the 21 points
        # the sum points will be divided by the number of points at the end to get the average
        
                
        # key press to clear images or quit in debug mode
        if debug:
            k = cv2.waitKey(0)
            if k == 27 or k == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
            else: 
                cv2.destroyAllWindows()
        
    # TODO: averaging
    
    
    # store the hand in ideals so we can pickle it later  
    # this is just storing the last hand read
    # we can't store the hand objects directly because they contain all sorts of C++ and tensorflow nonsense; not pickleable
    # instead we just need the x, y, z values along with a letter ID, so store hand 0 (there should only be one)
    export = []
    if hand.multi_hand_landmarks:
        for lm in hand.multi_hand_landmarks[0].landmark: export.append((lm.x, lm.y, lm.z)) 
        ideals.append({"letter" : letter, "landmarks" : export})
        print(f"Appended, total hands stored: {len(ideals)}")
    
# save ideal hands to a file ideals.pkl
with open("ideals.pkl", "wb") as f:
    #print(ideals)
    pickle.dump(ideals, f)
    
## OLD CODE

        # if debug:
        #     # draw the joints and bones on the hand image
        #     if hand.multi_hand_landmarks:
        #         for hand_landmarks in hand.multi_hand_landmarks:
        #             # Draw landmark dots
        #             for idx, landmark in enumerate(hand_landmarks.landmark):
        #                 h, w, _ = image.shape
        #                 cx, cy = int(landmark.x * w), int(landmark.y * h)
        #                 cv2.circle(image, (cx, cy), 5, (0, 0, 255), thickness=-1)

        #             # Optionally draw hand connections
        #             mp_draw.draw_landmarks(
        #                 image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        #                 mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        #                 mp_draw.DrawingSpec(color=(255,0,0), thickness=2))
                
        #     cv2.imshow('Loaded Image', image)



