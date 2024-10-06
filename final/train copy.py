import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os

# Dataset class to load images and labels
class SymbolDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.image_names = list(labels.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_path).convert('RGB')
        
        label = self.labels[img_name]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the CNN architecture
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

# Load and preprocess the dataset
image_dir = 'ImageGood'  # Directory where your images are stored
labels = {
     'img1.jpg': 1,
    'img2.jpg': 1,
    'img3.jpg': 1,
    'img4.jpg': 1,
    'img5.jpg': 1,
    'img6.jpg': 1,
    'img7.jpg': 1,
    'img8.jpg': 1,
    'img9.jpg': 1,
    'img10.jpg': 1,
    'img11.jpg': 1,
    'img12.jpg': 1,
    'img13.jpg': 1,
    'img14.jpg': 1,
    'img15.jpg': 1,
    'img16.jpg': 1,
    'img17.jpg': 1,
    'img18.jpg': 1,
    'img19.jpg': 1,
    'img20.jpg': 1,
    'img21.jpg': 1,
    'img22.jpg': 1,
    'img23.jpg': 1,
    'img24.jpg': 1,
    'img25.jpg': 1,
    'img26.jpg': 1,
    'img27.jpg': 1,
    'img28.jpg': 1,
    'img29.jpg': 1,
    'img30.jpg': 1,
    'img31.jpg': 1,
    'img32.jpg': 1,
    'img33.jpg': 1,
    'img34.jpg': 1,
    'img35.jpg': 1,
    'img36.jpg': 1,
    'img37.jpg': 1,
    'img38.jpg': 1,
    'img39.jpg': 1,
    'img40.jpg': 1,
    'img41.jpg': 1,
    'img42.jpg': 1,
    'image 10.jpg': 0,
    'image 11.jpg': 0,
    'image 12.jpg': 0,
    'image 13.jpg': 0,
    'image 14.jpg': 0,
    'image 15.jpg': 0,
    'image 16.jpg': 0,
    'image 17.jpg': 0,
    'image 18.jpg': 0,
    'image 19.jpg': 0,
    'image 20.jpg': 0,
    'image 21.jpg': 0,
    'image 22.jpg': 0,
    'image 23.jpg': 0,
    'image 24.jpg': 0,
    'image 25.jpg': 0,
    'image 26.jpg': 0,
    'image 27.jpg': 0,
    'image 28.jpg': 0,
    'image 29.jpg': 0,
    'image 30.jpg': 0,
    'image 31.jpg': 0,
    'image 32.jpg': 0,
    'image 33.jpg': 0,
    'image 34.jpg': 0,
    'image 35.jpg': 0,
    'image 36.jpg': 0,
    'image 37.jpg': 0,
    'image 38.jpg': 0,
    'image 39.jpg': 0,
    'image 40.jpg': 0,
    'image 41.jpg': 0,
    'image 42.jpg': 0,
    'image 43.jpg': 0,
    'image 44.jpg': 0,
    'image 45.jpg': 0,
    'image 46.jpg': 0,
    'image 47.jpg': 0,
    'image 48.jpg': 0,
    'image 49.jpg': 0,
    'image 55.jpg': 0,
    'image 56.jpg': 0,
    'image 57.jpg': 0,
    'image 58.jpg': 0,
    'image 59.jpg': 0,
    'image 60.jpg': 0,
}

# Data transformations (resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset and dataloader
dataset = SymbolDataset(image_dir, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SymbolCNN()
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 12
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        labels = labels.unsqueeze(1).float()  # Reshape labels to match output
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}')

# Save the trained model weights
torch.save(model.state_dict(), 'isa_symbol_cnn.pth')

print("Training complete, model saved as 'isa_symbol_cnn.pth'")
