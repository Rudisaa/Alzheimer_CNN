import torch;
import torch.nn as nn;
import torch.optim as optim;
import torchvision;
import torchvision.transforms as transforms;
from torchvision.datasets import ImageFolder;
from torch.utils.data import DataLoader;
import numpy as np;
from sklearn.metrics import confusion_matrix;
import matplotlib.pyplot as plt;
import cv2;
import os;
from dotenv import load_dotenv

load_dotenv()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# paths for the data
datasetPath = os.getenv('DATASET')

trainingSet = os.path.join(datasetPath, 'train')
testSet = os.path.join(datasetPath, 'test')


# preprocessing mri images from kaggale
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
])

training = ImageFolder(trainingSet , transform=transform)
test = ImageFolder(testSet, transform=transform)

# dataloaders
# creates an iterable dataset object
trainingLoader = DataLoader(training, batch_size=64, shuffle=True)
testLoader = DataLoader(test, batch_size=64, shuffle=False)

m = nn.Conv2d(3, 32, kernel_size=3, padding=1)
m = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#m = nn.Conv2d(32, 128, kernel_size=3, padding=1)
m = nn.MaxPool2d(2,2)
input = torch.randn(1,3,64,64)
output = m(input)
print(output.shape)
# Creating a CNN 

class CNN(nn.Module):
    def __init__(self):
        super().__init__(CNN, self.__init__()) 
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(8192, 128)
        self.fc2 = nn.Linear(128, 2)




