# Importing Libraries
import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

 # creating a map corresponding to the labels
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

labels = {alphabet[i]: i for i in range(len(alphabet))}
labels['del'] = 26
labels['nothing'] = 27
labels['space'] = 28

# custom Dataset in order to include all 29 class labels and the bounding box for each of them
class CustomDataset(Dataset):
  def __init__(self, root_dir, transform=None):
      self.root_dir = root_dir
      self.transform = transform
      self.class_names = sorted(os.listdir(root_dir))     # list of strings containing the names of all folders in root_dir
      self.data = self._load_data()                       # calls own load_data method

  def _load_data(self):
      data = []                                                  # data is a list of dictionary for each image containing imagepath,

      for class_name in self.class_names:                        # self.class_names is a list of the names of all the folders containing classified hand images
          class_dir = os.path.join(self.root_dir, class_name)    # specific class folder
          for file_name in os.listdir(class_dir):                # file name in a given class folder
              image_path = os.path.join(class_dir, file_name)
              data.append({
                  'image_path': image_path,
                  'truth_labels': self.get_label(class_name),              # 29
                  'bounding_box': self.get_bounding_box_info(),  # 4 
              })
      return data
  
  def get_label(self, class_name):
      """
      Return a list for the ground truth labels
      """
      truth_label = [0] * 29
      truth_label[labels[class_name]] = 1    #TODO: check for weird float or int issue
      return truth_label
  

  def get_bounding_box_info(self):
      """
      Return bounding box information (e.g., coordinates)
      Since all the images are closely cropped, lets set bx, by, bw, bh

      Note bx and by encode the center and bw and bh are set to 1 since its closely cropped
      """
      return [0.5, 0.5, 1, 1] #TODO: check this later for random float nonsense

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      sample = self.data[idx]
      image = Image.open(sample['image_path']).convert('RGB')

      if self.transform:
          image = self.transform(image)

      return {
          'image': image,
          'label': sample['class_name'],
          'bounding_box': sample['bounding_box'],
      }

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create dataset instance
dataset = CustomDataset(root_dir='asl_alphabet_train', transform=transform)
# Create DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


train_transform = transforms.Compose([
  
  # Resize to 224x224, since images are 200x200, it just resizes larger while 
  # maintaining aspect ratio, thus, the image stays closely cropped, yolo works on images that satisfies : 32|size
  transforms.Resize(224),      
  # 50% chance of horizontal flip so that the model doesn't overfit to a single horizontal orientation
  transforms.RandomHorizontalFlip(),
  # transform image to tensor for training
  transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

def load_data(data_dir):
  """
  Takes string representing data directory and loads the training and test set from it
  """
  dataset = dataset = ImageFolder(root=data_dir)
  train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

  # transformation applied to each dataset
  train_dataset.dataset.transform = train_transform
  test_dataset.dataset.transform = test_transform
  # Defining Training and Testing DataLoaders
    # You can use batch size as a hyperparameter if needed?
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
  
  return train_loader, test_loader
