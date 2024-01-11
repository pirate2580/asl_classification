This is a real-time ASL hand sign detector and classifier made using a version of the YOLOV1 algorithm. Specifically, although the exact model architecture is different from the original
YOLO paper, the implementation of the loss function is the same, achieving the same effect as the algorithm described in the paper.

This application is still a work in progress as it is sill not that accurate in classifying different hand signals. For the future, better data and a more complex model will be 
implemented to improve the precision of the model's inference.

To test the project on your own device, please installl OpenCV, NumPy, TensorFlow and Keras and run ASLDetection.py.
A screen will open up outputting 672x672 wideo output from your camera and the YOLO algorithm will begin rnning.
