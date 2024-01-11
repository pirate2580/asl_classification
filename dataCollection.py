#Import Libraries
import cv2
from cvzone.HandTrackingModule import HandDetector # module to help with data collection
import numpy as np
import math
import time
import random

'''
This file is purely for data collection and some preprocessing
'''

# TODO: idk how to fix but I have to use 3.11.4 64-bit interpreter for this file specifically for cvzone
# and 3.11.3 for the other ones

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labels = {alphabet[i]: i for i in range(len(alphabet))}

cap = cv2.VideoCapture(0)           # try 0 or 1
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
  hands, img = detector.findHands(cv2.resize(img, (imgSize, imgSize)), draw = False)
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

combined_data = list(zip(trainx_list, trainy_list))

# Shuffle the combined data
random.shuffle(combined_data)

# Unzip the shuffled data back into separate lists
shuffled_x, shuffled_y = zip(*combined_data)

image_stack = np.stack(shuffled_x, axis = 0)
labels_stack = np.stack(shuffled_y, axis = 0)

np.save('testx.npy', image_stack)    #change to whatever npy file you want info to go into
np.save('testy.npy', labels_stack)

