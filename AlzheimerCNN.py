import torch;
import torch.nn as nn;
import torch.optim as optim;
import torchvision;
import torchvision.transforms as transforms;
from torchvision.datasets import ImageFolder;
from torch.utils.data import dataloader;
import numpy as np;
from sklearn.metrics import confusion_matrix;
import matplotlib.pyplot as plt;
import cv2;

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# paths for the data
trainingDataset = ImageFolder(root='/Users/rudisargueta/Documents/Alzheimer_CNN/AlzheimersDataset/train')
testDataset = ImageFolder(root='/Users/rudisargueta/Documents/Alzheimer_CNN/AlzheimersDataset/test')

# preprocessing mri images from kaggale
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
])





