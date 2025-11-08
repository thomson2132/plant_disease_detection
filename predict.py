import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import sys

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, len(class_names))
)
model.load_state_dict(torch.load("efficientnet_b0_best.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        print(f"✅ Predicted Class: {predicted_class}")

# Usage: Pass image path as command line argument
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict(sys.argv[1])

'''import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os
import sys

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, len(class_names))
)
model.load_state_dict(torch.load("efficientnet_b0_best1.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        print(f" {os.path.basename(image_path)} --> Predicted Class: {predicted_class}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_folder.py <path_to_image_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]

    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        sys.exit(1)

    # Supported image extensions
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Iterate and predict on each image file
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(img_extensions):
            image_path = os.path.join(folder_path, filename)
            predict(image_path)
'''
