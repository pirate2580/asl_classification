#Import Libraries
import cv2
from cvzone.HandTrackingModule import HandDetector # module to help with data collection
import numpy as np
import math
import time
import torch

'''
This file is purely for data collection and some preprocessing
'''

# TODO: idk how to fix but I have to use 3.11.4 64-bit interpreter for this file specifically for cvzone
# and 3.11.3 for the other ones

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labels = {alphabet[i]: i for i in range(len(alphabet))}

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)

offset = 10
imgSize = 224
SPLIT_SIZE = 7
N_CLASSES = 26

trainx_list = []
trainy_list = []


counter = 0

while True:
  success, img = cap.read()

  #NOTE: with our bounding box normalization, we DO NOT need to change aspect ratio right away
  hands, img = detector.findHands(img)
  if hands:
    hands = hands[0]
    x, y, w, h = hands['bbox']
    # print(x, y, w, h)

  key = cv2.waitKey(1)

  cv2.imshow("image", img)
  for i, letter in enumerate(alphabet):
    if key == ord(letter.lower()):
        counter += 1
        trainx_list.append(img)

        class_lst = [0] * 26
        class_lst[labels[letter]] = 1
        output = [labels[letter]] + [x, y, w, h] + class_lst
        trainy_list.append(np.array(output))
        print(f"letter: {letter}, counter: {counter}")
    
  if key == 27:
    break

cap.release()

image_stack = np.stack(trainx_list, axis = 0)
labels_stack = np.stack(trainy_list, axis = 0)

np.save('trainx.npy', image_stack)
np.save('trainy.npy', labels_stack)

