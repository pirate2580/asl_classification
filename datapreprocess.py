import cv2
import os
import numpy as np

# There are 29 classes for the hand signs that we want to classify correctly in real time from our camera

# creating a map corresponding to the labels
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labels = {alphabet[i]: i for i in range(len(alphabet))}
labels['del'] = 26
labels['nothing'] = 27
labels['space'] = 28


def folder_to_numpy(folder_path: str):
  """
  This function goes inside the images in each training set folder and converts images into 4D
  numpy array of dimensions (batch_size, height, width, channels)
  
  labels are correspondingly given
  """
  image_list = []
  labels_list = []

  cnt = 0

  for class_folder in os.listdir(folder_path):
    if class_folder != ".DS_Store":

      #counts the ith letter being preprocessed and prints which letter it is
      cnt += 1
      print(f"{class_folder} {cnt}")

      for filename in os.listdir(folder_path + '/' + class_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
              # Read each image using OpenCV
              # print(folder_path + '/' + class_folder)
              image_path = os.path.join(folder_path + '/' + class_folder, filename)
              image = cv2.imread(image_path)

              # Check if the image is loaded successfully
              if image is not None:
                  # Convert the image to a NumPy array
                  image_array = np.array(image)

                  # Append the image array to the image
                  image_list.append(image_array)
                  labels_list.append(labels[class_folder])

  # Stack the image arrays to create a 3D array
  image_stack = np.stack(image_list, axis = 0)
  labels_stack = np.stack(labels_list, axis = 0)

  return image_stack, labels_stack

folder_path = "/Users/naorojfarhan/Desktop/self_study/coding_projects/ML_projects/asl_classification/asl_alphabet_train"
images_4d_array, truth_labels = folder_to_numpy(folder_path)

# save the input and ground truth labels to npy file
np.save('train_x.npy', images_4d_array)
np.save('train_y.npy', truth_labels)

print("Shape of the 4D NumPy array:", images_4d_array.shape)
