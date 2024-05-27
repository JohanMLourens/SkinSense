import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import seaborn as sns
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Path to the testing folder
testing_folder_name = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinCancerData/TestDataResized'
model_path = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinCancerData/Model'

# The image size
img_size = (300, 225)  
model_save_path = os.path.join(model_path, '3_Main_82%_Model_Resnet.pt')

# Classes
classes = ['akiec', 'bcc', 'mel']
num_classes = len(classes)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define transforms for testing (resize and normalize)
test_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ResNetModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Create dataset and dataloader for testing
test_dataset = ImageFolder(root=testing_folder_name, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the trained model
model = ResNetModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# Function to test the model and generate metrics
def evaluate_model(model, test_loader, device):
    y_true = []
    y_pred = []
    test_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=classes)

    return test_loss, accuracy, conf_matrix, class_report

# Evaluate the model
test_loss, accuracy, conf_matrix, class_report = evaluate_model(model, test_loader, device)

# Print the results
print(f'Test Loss: {test_loss:.6f}')
print(f'Test Accuracy: {accuracy*100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
