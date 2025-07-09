import cv2
import mediapipe as mp
import string
import sys
import utils # utils.py custom defines

# debug macro
debug = 1

# number of images per letter to process
nimpl = 1

# do not include Z or J because they have movement in the sign; out of scope for now
letters = [ch for ch in string.ascii_uppercase if ch not in ('J', 'Z')]
#print(letters)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Iiitialize hand landmarker (hands)
hands = mp_hands.Hands(
    model_complexity=1,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.6)


for letter in letters:
    for i in range(0, nimpl):
        
        # load the image and process the 
        path = "dataset/SignAlphaSet/" + letter + "/" + letter + "_" + str(i) + ".jpg" # path to each letter image
        if debug: print(path)
        #image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) # open image and convert it to RGB
        image = cv2.imread(path)
        hand = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # some of the hands are mirrored across the veritcal axis; dataset contains left and right handed symbols
        # we need to have the ideal case be averaging across only right signs
        # image flip criteria is hand sign dependant, so we need tailored behavior for each sign
        #
        match letter:
            case 'A':
                # we flip the image about verical axis if thumb tip left of pinky 1st joint 
                # thumb tip is the 5th element (index 4) of the landmarks, x coordinate is normalized from [0 1], 0= left edge
                if hand.multi_hand_landmarks[0].landmark[4].x < hand.multi_hand_landmarks[0].landmark[18].x:
                    image = cv2.flip(image, 1) # code 1 for horizontal flip
                    hand = hands.process(image) # reprocess hands for flipped image
                    if debug:
                        txt = "Flipped " + letter + "_" + str(i)
                        print(txt)
                        utils.drawLandmarks(hand, image, txt)
                        
                
            case 'b':
                # flip if index root left of pinky root
                if hand.multi_hand_landmarks[0].landmark[5].x < hand.multi_hand_landmarks[0].landmark[17].x:
                    image = cv2.flip(image, 1) # code 1 for horizontal flip
                    hand = hands.process(image) # reprocess hands for flipped image
                    
                    

            
            # case 'c':
                
            # case 'd':
            
            # case 'e':
                
            # case 'f':
                
            # case 'g':
                
            # case 'h':
                
            # case 'i':
                
            # case 'k':
                
            # case 'l':
                
            # case 'm':
                
            # case 'n':
                
            # case 'o':
                
            # case 'p':
                
            # case 'q':
                
            # case 'r':
                
            # case 's':
                
            # case 't':
                
            # case 'u':
                
            # case 'v':
                
            # case 'w':
                
            # case 'x':
                
            # case 'y':
                
            case _:
                print('default case of letter matching; error!')
                    
       

        
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
                
        utils.drawLandmarks(hand, image, path)
        k = cv2.waitKey(0)
        if k == 27 or k == ord('q'):
            cv2.destroyAllWindows()
            sys.exit()
        else: 
            cv2.destroyAllWindows()




