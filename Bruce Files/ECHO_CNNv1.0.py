# ECHO_CGANv1.0
# Author: Bruce Noble
# This algorithm is being developed for the PARSEC - ECHO system
# This algorithm will take in a 16x16 matrix of capacitance measurements and determine the cross-sectional area
# of the sample between the electrodes

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from glob import glob
from PIL import Image
from pathlib import Path

## hyperparameters
BATCH_SIZE = 16  # number of samples per batch
IMAGE_SIZE = 16  # width/Height of square images
LATENT_DIM = 100  # dimensionality of noise vector for Generator
EPOCHS = 100  # number of training epochs
LEARNING_RATE = 2e-4  # learning rate for optimizers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # training on GPU if available

## dataset from triangleSims
class FolderECTDataset(Dataset):
    def __init__(self, root_dir="C:/Users/scien/PycharmProjects/ECHO_ML/DATA/MLTriangleTrainingData"):
        self.sample_paths = []
        for folder in glob(os.path.join(root_dir, "*")):
            image_files = glob(os.path.join(folder, "*.jpg"))
            if image_files:
                area = int(os.path.basename(folder))
                for img_file in image_files:
                    self.sample_paths.append((img_file, area))

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        img_path, area = self.sample_paths[idx]
        capacitance_data = Image.open(img_path).convert("RGB")
        capacitance_data = self.transform(capacitance_data)

        area = torch.tensor(area, dtype=torch.float32)
        return capacitance_data, area

## create network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Linear(64*8*8, 512),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

## training and optimization
model = NeuralNetwork().to(DEVICE)

dataset = FolderECTDataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # set the model to training mode - important for batch normalization and dropout layers
    # unnecessary in this situation but added for best practice
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def test_loop(dataloader, model, loss_fn):
    # set the model to evaluation mode - important for batch normalization and dropout layers
    # unnecessary in this situation but added for best practice
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

