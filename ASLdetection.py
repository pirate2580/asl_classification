import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNet
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense, Reshape, Dropout


import ssl
ssl._create_default_https_context = ssl._create_unverified_context # not ideal but idk how else to load the model

import cv2
import numpy as np

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labels = {i: alphabet[i] for i in range(len(alphabet))}

B = 1 # no. of bounding boxes
N_CLASSES = 26
H, W = 224, 224
SPLIT_SIZE = H//32 # S=21
N_EPOCHS = 135
BATCH_SIZE = 32

NUM_FILTERS = 512
OUTPUT_DIM = N_CLASSES + 5*B

base_model = tf.keras.applications.MobileNet(
    weights = 'imagenet',
    input_shape = (H, W, 3),
    include_top = False
)
base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    Conv2D(NUM_FILTERS, (3,3), padding = 'same', kernel_initializer = 'he_normal'),
    BatchNormalization(),
    LeakyReLU(alpha = 0.1),

    Conv2D(NUM_FILTERS, (3,3), padding = 'same', kernel_initializer = 'he_normal'),
    BatchNormalization(),
    LeakyReLU(alpha = 0.1),

    Conv2D(NUM_FILTERS, (3,3), padding = 'same', kernel_initializer = 'he_normal'),
    BatchNormalization(),
    LeakyReLU(alpha = 0.1),

    Conv2D(NUM_FILTERS, (3,3), padding = 'same', kernel_initializer = 'he_normal'),
    LeakyReLU(alpha = 0.1),

    Flatten(),


    Dense(NUM_FILTERS, kernel_initializer = 'he_normal'),
    BatchNormalization(),
    LeakyReLU(alpha = 0.1),

    Dropout(0.4),
    Dense(SPLIT_SIZE * SPLIT_SIZE * OUTPUT_DIM, activation = 'sigmoid'),
    Dropout(0.4),
    Reshape((SPLIT_SIZE, SPLIT_SIZE, OUTPUT_DIM)),

])


model.load_weights('yolo_resnet_50.h5')

#model.summary()

offset = 10
imgSize = 224
SPLIT_SIZE = 7
N_CLASSES = 26
cap = cv2.VideoCapture(1)           # try 0 or 1

counter = 0

while True:
    success, img = cap.read()

    x = cv2.resize(img, (imgSize, imgSize))
    x = np.expand_dims(x, axis=0)
    output = tf.squeeze(model.predict(x), axis = 0).numpy()   # (1x)7x7x31
    
    x = np.squeeze(x, axis=0)
    THRESH = 0.01

    final_boxes_and_prediction=[]

    #print(output)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            pred = output[i][j]

            if pred[0] > THRESH:
                print(i,j)
                x_centre = (i + pred[1]) * 32
                y_centre = (j + pred[2]) * 32
                width, height = np.abs(pred[3]) * W, np.abs(pred[4]) * H
                x_min, y_min = int(x_centre - width / 2), int(y_centre - height / 2)
                x_max, y_max = int(x_centre + width / 2), int(y_centre + height / 2)

                if (x_min <= 0): x_min = 0
                if (y_min <= 0): y_min = 0
                if (x_max >= W): x_max = W
                if (y_max >= H): y_max = H
                final_boxes_and_prediction.append([pred[0], x_min, y_min, x_max, y_max, labels[np.argmax(pred[5:])]])

    final_boxes_and_prediction = sorted(final_boxes_and_prediction, key= lambda x: x[0], reverse = True)
    if (len(final_boxes_and_prediction) > 0):
        box = final_boxes_and_prediction[0]
        cv2.rectangle(x, (box[1], box[2]), (box[3], box[4]), (0, 0, 255), 4)
        cv2.putText(x, box[5], (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0,255),1)
        

    cv2.imshow("video:", cv2.resize(x, (672, 672)))
    cv2.waitKey(1)


