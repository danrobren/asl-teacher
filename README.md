# asl-teacher
A MediaPipe-based tool for simple American Sign Language Education

# Instructions
The user must have a laptop with a webcam or some other camera hardware device connected. Python version 3.11.9 was used to develop and test this project. Other versions may work, but Python 3.11.9 is reccomended because it supports the required packages.

Install the dependencies using the following command:

> pip install -r requirements.txt

The main script requires ideals.pkl to be present in the same directory as main.py. The number of images used for each letter in training can be configured by editing the _nimpl_ variable at the top of train.py. A maximum of 1000 images can be used since that is how many SignAlphaSet provides. If ideals.pkl is not present, or to re-generate the training data, run the following command:

> python ./train.py

SignAlohaSet can be downloaded at:

> https://data.mendeley.com/datasets/8fmvr9m98w/2

Run the program using the following command:

> pyhton ./main.py

The program will pause for a few seconds as MediaPipe initializes. Some warnings and infomration may be printed to the terminal at this time. After initialization, the following main menu will then be printed to the terminal:

> Choose Program Mode:
> 1: Letter Select
> 2: Detect Letter Sign

If option 1 is selected, a subsequent menu will ask the user to select a letter. If option 2 is selected the display windows will open immediately. Two windows open, the first of which has a live feed from the webcam overlayed with hand landmark data and a score based on how close the user's hand is to the ideal hand. The user's objective is to match their hand as closely as possible with the ideal hand dislayed in the video. The 'Q' key can be pressed at any time to quit the program.