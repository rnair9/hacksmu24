# Import necessary libraries
import cv2
import torch
import torch.nn as nn

# Define your CNN model (assuming it's the same as you previously defined)
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

# Load the model
model = SymbolCNN()
model.load_state_dict(torch.load('isa_symbol_cnn.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Function to preprocess the video frame
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to the model's input size
    frame_normalized = frame_resized / 255.0       # Normalize pixel values to [0, 1]
    frame_tensor = torch.tensor(frame_normalized).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, 224, 224)
    return frame_tensor

# Load the video
cap = cv2.VideoCapture('cars.mp4')

# Read until the video is completed
while True:
    # Capture frame by frame
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if there are no frames left

    # Preprocess the current frame
    frame_tensor = preprocess_frame(frame)

    # Run inference on the model
    with torch.no_grad():
        output = model(frame_tensor)

    # Apply threshold to decide if the symbol is detected
    probability = torch.sigmoid(output).item()
    if probability > 0.5:
        # Draw a rectangle around the detected symbol area
        height, width, _ = frame.shape
        cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 255, 0), 2)
        label = f'Symbol detected: {probability:.2f}'
    else:
        # Draw a rectangle indicating no detection
        height, width, _ = frame.shape
        cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 0, 255), 2)
        label = f'No symbol: {probability:.2f}'

    # Add label text on the frame
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('video', frame)

    # Press Q on the keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video-capture object
cap.release()
# Close all the frames
cv2.destroyAllWindows()
