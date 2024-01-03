from ultralytics import YOLO

# Importing Libraries
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder


model = YOLO("yolov8s.pt")



class Modified_YOLOv8(nn.Module):
  def __init__(self, pretrained_YOLO):
    super(Modified_YOLOv8, self).__init__() # this is because initialization is mostly the same from superclass
    self.features = nn.Sequential(*list(pretrained_YOLO.children())[:22])
    self.fc1 = nn.Linear(1024, 1024)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(1024, 512)
    self.relu = nn.ReLU()
    self.fc3 = nn.Linear(512, 29)

  def forward(self, x):
        return self.features(x)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])



# Freeze earlier layers
for param in model.parameters():
    param.requires_grad = False

unfrozen_layers = ['fc1', 'fc2', 'fc3']  

for name, param in model.named_parameters():
  if any(layer_name in name for layer_name in unfrozen_layers): 
        # if any of the strings are in the name of the current layer being checked, unfreeze it
        param.requires_grad = True