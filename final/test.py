import torch
import torch.nn as nn
import cv2
import numpy as np

class SymbolCNN(nn.Module):
    def __init__(self):
        super(SymbolCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load the model architecture
model = SymbolCNN()

# Load the saved weights from the .pth file
model.load_state_dict(torch.load('isa_symbol_cnn.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode


def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Resize the image to the input size expected by the model (224x224 in this case)
    image_resized = cv2.resize(image, (224, 224))

    # Convert the image to a tensor and normalize the pixel values to [0, 1]
    image_normalized = image_resized / 255.0

    # Convert to a tensor and add batch dimension (1, 3, 224, 224)
    image_tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0).float()

    return image_tensor


def test_model_on_image(model, image_path):
    # Preprocess the image
    image_tensor = preprocess_image(image_path)
    
    # Make sure the model doesn't compute gradients (not needed for inference)
    with torch.no_grad():
        output = model(image_tensor)
    
    # Since the output uses a sigmoid activation, it will be between 0 and 1
    probability = torch.sigmoid(output).item()
    
    # Threshold the output to determine if the symbol is present (assuming binary classification)
    if probability > 0.5:
        print(f"Symbol detected in {image_path}. Probability: {probability:.4f}")
    else:
        print(f"No symbol detected in {image_path}. Probability: {probability:.4f}")

# Test the model on a few images
test_model_on_image(model, 'img1.jpg')
test_model_on_image(model, 'img2.jpg')






