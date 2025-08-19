# settings file to hold all the constants and mediapipe settings
import string
import sys
import pickle
import json
import cv2
import numpy as np
import mediapipe as mp
import utils # utils.py custom defines functions

# TODO: give all the joint numbers descriptive tags

# "constants" in allcaps
IMAGE_WIDTH = 296
IMAGE_HEIGHT = 296

UDP_IP   = "127.0.0.1"     # intended to run on same computer as Unity
UDP_PORT = 5005

EXE_NAME = "ASLUnityBridge.exe"
Unity_WIDTH    = 1300
Unity_HEIGHT   = 600

Hands_config_train = {
    "model_complexity": 1,
    "static_image_mode": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.0
  }

Hands_config_main = {
    "model_complexity": 1,
    "static_image_mode": False,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
  }