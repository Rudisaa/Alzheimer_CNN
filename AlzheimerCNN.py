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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# paths for the data
datasetPath = '/Users/rudisargueta/Documents/Alzheimer_CNN/AlzheimersDataset'

trainingSet = os.path.join(datasetPath, 'test')
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
trainingLoader = DataLoader(training, batch_size=64, shuffle=True)
testLoader = DataLoader(test, batch_size=64, shuffle=False)

#





