import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# Define the path to the test data folder
test_data_folder = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinCancerData/TestDataResized'
# Define the path to the saved model
saved_model_path = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinSense/Model/cnn_model.pth'
# Define the image size
img_size = (300, 225)

# List the classes (should be the same as during training)
classes = ['akiec', 'bcc', 'mel']
print("Classes:", classes)

print("Libraries imported - ready to use PyTorch", torch.__version__)

class Model(nn.Module):
    def __init__(self, num_classes=3):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.2)
        self.flattened_size = self._get_flattened_size((3, 300, 225))
        self.fc = nn.Linear(in_features=self.flattened_size, out_features=num_classes)

    def _get_flattened_size(self, shape):
        x = torch.zeros(1, *shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return int(torch.prod(torch.tensor(x.shape[1:])))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.view(-1, self.flattened_size)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = Model(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.eval()
print('Model loaded successfully from', saved_model_path)

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create test dataset and dataloader
test_dataset = torchvision.datasets.ImageFolder(root=test_data_folder, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the loss criteria
loss_criteria = nn.CrossEntropyLoss()

# Function to test the model
def test(model, device, test_loader, loss_criteria):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_criteria(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {avg_loss:.6f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return avg_loss, accuracy, all_preds, all_targets

# Run the test
test_loss, test_accuracy, all_preds, all_targets = test(model, device, test_loader, loss_criteria)

# Print confusion matrix
conf_matrix = confusion_matrix(all_targets, all_preds)
print('Confusion Matrix:')
print(conf_matrix)

# Calculate percentages for the confusion matrix
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Plot the confusion matrix
df_cm = pd.DataFrame(conf_matrix, index=classes, columns=classes)
df_cm_percent = pd.DataFrame(conf_matrix_percent, index=classes, columns=classes)

plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=df_cm_percent, fmt='.2f', cmap='Blues', annot_kws={"size": 16})
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.title("Confusion Matrix with Percentages", fontsize=16)
plt.show()
