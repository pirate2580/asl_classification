import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

import torch


# Create a VideoCapture object
cap = cv2.VideoCapture(1)  # 0 represents the default camera (usually the built-in laptop camera)

model = YOLO("yolov8s.pt")

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the window name and create a window
window_name = 'Camera Feed'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Break the loop if the video capture fails
    if not ret:
        print("Error: Could not read frame.")
        break
    
    results = model(frame, device='mps')

    # Display the resulting frame
    cv2.imshow(window_name, frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    results = model(frame)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype = "int")
    classes = np.array(result.boxes.cls.cpu(), dtype = "int")

    #print(bboxes)

    for cls, bbox in zip(classes, bboxes):
      (x, y, x2, y2) = bbox 

      #print(f"x: {x}, y: {y}")

      cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), 4)
      cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
    
    
    cv2.imshow('camera', frame)

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
