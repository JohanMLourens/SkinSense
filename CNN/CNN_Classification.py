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
from torch.optim.lr_scheduler import ReduceLROnPlateau

# The path to the training folder
training_folder_name = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinCancerData/ResizedTrainning'

# The path and filename for saving the model
model_dir = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinSense/Model'
model_filename = 'cnn_model.pth' # Name of the model
model_path = os.path.join(model_dir, model_filename) # joins the 2 string togther

# The image size
img_size = (300, 225)

# The classes from the training folder
classes = sorted(os.listdir(training_folder_name))
print("Classes:", classes)

print("Libraries imported - ready to use PyTorch", torch.__version__)

class Model(nn.Module):
    def __init__(self, num_classes=3):
        super(Model, self).__init__() 
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1) # 3 inputs for the RGB values 
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.2)
        # Calculate the flattened size after convolution and pooling
        self.flattened_size = self._get_flattened_size((3, 300, 225)) # Calculates the flattened size for you
        self.fc = nn.Linear(in_features=self.flattened_size, out_features=num_classes)

    def _get_flattened_size(self, shape):
        x = torch.zeros(1, *shape)          # Creates a tensor with shape (1, 3, 300, 225), where 1 is the batch size.
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return int(torch.prod(torch.tensor(x.shape[1:])))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Applies the first convolutional layer, followed by a ReLU activation function and max pooling.
        x = self.pool(F.relu(self.conv2(x))) # Applies the second convolutional layer, followed by ReLU activation and max pooling.
        x = self.drop(x)                    # Applies dropout to the resulting tensor.
        x = x.view(-1, self.flattened_size) # lattens the tensor to the shape so that it can be passed to the fully connected layer.
        x = self.fc(x)                      # Passes the flattened tensor through the fully connected layer.
        return F.log_softmax(x, dim=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model(num_classes=3).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.2, verbose=True)

def train(model, device, train_loader, optimizer, epoch, loss_criteria):
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
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f'Validation set: Average loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%\n')
    return avg_loss, accuracy

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders
dataset = torchvision.datasets.ImageFolder(root=training_folder_name, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Track metrics in these lists
epoch_nums = []
training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []

# Number of epochs to train
epochs = 25  # Adjust this number as needed
print('Training on', device)

for epoch in range(1, epochs + 5):
    train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, criterion)
    test_loss, test_acc = test(model, device, test_loader, criterion)
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    training_accuracy.append(train_acc)
    validation_loss.append(test_loss)
    validation_accuracy.append(test_acc)
    scheduler.step(test_loss)

# Ensure model save path exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save trained model
torch.save(model.state_dict(), model_path)
print('Save Successful')

# Load model
new_model = Model(num_classes=len(classes)).to(device)
new_model.load_state_dict(torch.load(model_path, map_location=device))
new_model.eval()
print('Load successful')

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