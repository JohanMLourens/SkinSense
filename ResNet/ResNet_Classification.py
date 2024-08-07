import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau

# The path to the training folder
training_folder_name = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinCancerData/ResizedTrainning' # file of all the training data
model_path = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinSense/Model' # Location of where the model gets saved to

# Define the image size
img_size = (300, 225)

# List the classes from the training folder
classes = ['akiec', 'bcc', 'mel'] # 3 main classes 
num_classes = len(classes)
print("Classes:", classes)

print("Libraries imported - ready to use PyTorch", torch.__version__)

# Use of ResNet50 (pretrained model) 
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet50Model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu" # checks if the computer is compatible with cuda

# Initialize model, loss function, optimizer, and learning rate scheduler
model = ResNet50Model(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.2, verbose=True)

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalizes the pixel values of an image by subtracting the mean and dividing by the standard deviation for each channel
])

# Create datasets and dataloaders
train_dataset = torchvision.datasets.ImageFolder(root=training_folder_name, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None          
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model): # Saves the best model through out the training process
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join(model_path, 'checkpoint.pt'))

# Training and validation loop
def train(model, device, train_loader, optimizer, epoch, loss_criteria, scheduler=None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    print(f"Epoch: {epoch}")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criteria(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        if scheduler:
            scheduler.step(train_loss)
        if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(f'\tTraining batch {batch_idx + 1}/{len(train_loader)} Loss: {loss.item():.6f}')
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Training set: Average loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

def test(model, device, test_loader, loss_criteria):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_criteria(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f'Validation set: Average loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%\n')
    return avg_loss, accuracy

# Track metrics in these lists
epoch_nums = []
training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []

# Number of epochs to train
epochs = 10  # Number of iterations
print('Training on', device)

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, criterion, scheduler)
    test_loss, test_acc = test(model, device, test_loader, criterion)
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    training_accuracy.append(train_acc)
    validation_loss.append(test_loss)
    validation_accuracy.append(test_acc)
    early_stopping(test_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Load the best model
model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint.pt')))

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(epoch_nums, training_loss, label='Training Loss')
plt.plot(epoch_nums, validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()

# Plotting the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(epoch_nums, training_accuracy, label='Training Accuracy')
plt.plot(epoch_nums, validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()

# Save the final model
model_save_path = os.path.join(model_path, 'Trained_model.pt')
torch.save(model.state_dict(), model_save_path)
print('Save Successful')

# Load the model
new_model = ResNet50Model(num_classes=len(classes)).to(device)
new_model.load_state_dict(torch.load(model_save_path, map_location=device))
new_model.eval()
print('Load successful')
