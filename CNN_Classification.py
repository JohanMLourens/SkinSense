import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the path to the training folder
training_folder_name = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinCancerData/ResizedTrainning'
model_path = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinCancerData/Model'
# Define the image size
img_size = (120, 90)

# List the classes from the training folder
classes = sorted(os.listdir(training_folder_name))
print(classes)

print("Libraries imported - ready to use PyTorch", torch.__version__)

class Model(nn.Module):
    def __init__(self, num_classes=3):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.2)
        self.flattened_size = 30 * 22 * 24  # Adjusted to fit 120x90 image size
        self.fc = nn.Linear(in_features=self.flattened_size, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = self.drop(x)
        x = x.view(-1, self.flattened_size)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7']
model = Model(num_classes=len(classes)).to(device)
print(model)

def train(model, device, train_loader, optimizer, epoch, loss_criteria):
    model.train()
    train_loss = 0
    print(f"Epoch: {epoch}")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criteria(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(f'\tTraining batch {batch_idx + 1}/{len(train_loader)} Loss: {loss.item():.6f}')
    avg_loss = train_loss / len(train_loader)
    print(f'Training set: Average loss: {avg_loss:.6f}')
    return avg_loss

def test(model, device, test_loader, loss_criteria):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_criteria(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Validation set: Average loss: {avg_loss:.6f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return avg_loss

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

# Create datasets and dataloaders
train_dataset = torchvision.datasets.ImageFolder(root=training_folder_name, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)

# Use an "Adam" optimizer to adjust weights
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Specify the loss criteria
loss_criteria = nn.CrossEntropyLoss()

# Track metrics in these lists
epoch_nums = []
training_loss = []
validation_loss = []

# Number of epochs to train
epochs = 1
print('Training on', device)

for epoch in range(1, epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, epoch, loss_criteria)
    test_loss = test(model, device, test_loader, loss_criteria)
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)

print('Training completed.')

# Save trained model
torch.save(model.state_dict(), 'Trained_model.pt')
print('Save Successful')

# Load model
new_model = Model()
new_model.load_state_dict(torch.load('Trained_model.pt'))
print('Load successful')
new_model.eval()