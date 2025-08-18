# settings file to hold all the constants and mediapipe settings
import string
import sys
import pickle
import json
import cv2
import numpy as np
import mediapipe as mp
import utils # utils.py custom defines functions
import socket
import subprocess
import os

# TODO: give all the joint numbers descriptive tags

# ─────────────────────────────────────────────────────────────
# "constants" in allcaps
IMAGE_WIDTH = 296
IMAGE_HEIGHT = 296

# unity UDP connection settings 
UDP_IP = "127.0.0.1"  # intended to run on same computer as unity
UDP_PORT = 5005  # Create a UDP socket for sending data

EXE_NAME   = "ASLUnityBridge.exe"  # Unity launch config (same folder as this script)
UNITY_WIDTH      = 800 # unity settings
UNITY_HEIGHT     = 800

# mediapipe hands settings
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