import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import os
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F

# Directories
image_path = input('Please enter the path to the image: ') # promts user to give the directory of their image
model_path = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinCancerData/Model/3_Main_82%_Model_Resnet.pt'  # model filename

# Define the image size
img_size = (300, 225) 

# Classes
classes = ['akiec', 'bcc', 'mel'] # 3 main types
num_classes = len(classes)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu" # checks if the computer is compatible with cuda

# Transforms the input image 
input_transform = transforms.Compose([
    transforms.Resize(img_size), # rezises the given to 300,225
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ResNetModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetModel, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) # calls the resnet model through torchvision
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load trained model
model = ResNetModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Function to predict the class of an input image
def predict_image(image_path, model, device, transform, classes):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(img_size)
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        predicted_class = classes[predicted.item()]
        confidence = probabilities[0][predicted.item()].item()
    return predicted_class, confidence, probabilities

# Predict the class of the input image
predicted_class, confidence, probabilities = predict_image(image_path, model, device, input_transform, classes)

# Print the result
print(f'The predicted class for the input image is: {predicted_class} with confidence {confidence*100:.2f}%')

# Print the probabilities for all classes
for i, prob in enumerate(probabilities[0]):
    print(f'Class {classes[i]}: {prob.item()*100:.2f}%')
