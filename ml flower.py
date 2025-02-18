import shutil
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models

class Flower:
    def __init__(self, datapath, labels_path):
        self.labels_path = labels_path
        self.datapath = os.path.join(datapath, "organized")
        if not self.is_already_organized():
            self.create_sub_directories(datapath)
        else:
            print("Dataset is already organized.")
        self.data = self.convert_dataset(self.datapath)


    def convert_dataset(self, datapath):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        ])
        dataset = ImageFolder(datapath, transform)
        # Split dataset into training and validation
        train_size = int(0.8 * len(dataset))  # 80% for training
        valid_size = len(dataset) - train_size  # 20% for validation
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

        return train_loader, valid_loader

    def create_sub_directories(self, original_datapath):
        labels_data = loadmat(self.labels_path)
        labels = labels_data["labels"][0] - 1

        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        num_classes = len(set(labels))
        for i in range(num_classes):
            class_folder = os.path.join(self.datapath, f"class_{i}")
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

        for idx, label in enumerate(labels):
            image_filename = f"image_{idx+1:05d}.jpg"
            source_path = os.path.join(original_datapath, image_filename)
            destination_folder = os.path.join(self.datapath, f"class_{label}")
            destination_path = os.path.join(destination_folder, image_filename)

            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
            else:
                print(f" Warning: {source_path} not found.")

    def is_already_organized(self):
        if not os.path.exists(self.datapath):
            return False

        for folder in os.listdir(self.datapath):
            folder_path = os.path.join(self.datapath, folder)
            if os.path.isdir(folder_path):
                if any(fname.endswith(('.jpg', '.jpeg', '.png')) for fname in os.listdir(folder_path)):
                    continue
                else:
                    return False
        return True


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

if __name__ == "__main__":
    labels_path = "C:/Users/vbacho/OneDrive - UW/ml flower/imagelabels.mat"
    datapath = "C:/Users/vbacho/OneDrive - UW/ml flower/jpg"
    flower = Flower(datapath, labels_path)

    train_loader, valid_loader = flower.data

    num_classes = 102
    model = CustomCNN(num_classes)
    model_path = "flower_model.pth"

    if False:#os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Model loaded from saved file.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        epochs = 30
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}')

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Accuracy rate: {accuracy:.2f}%')

        torch.save(model.state_dict(), model_path)
        print("Model trained and saved.")
    # image_path = "C:/Users/vbacho/OneDrive - UW/ml flower/jpg/organized/class_7/image_03286.jpg"
    # predicted_class = predict_image(image_path, model)
    # print(f"Predicted class: {predicted_class}")
# Dataset is already organized.
# Epoch 1/30, Loss: 4.147857492726024
# Accuracy rate: 19.54%
# Epoch 2/30, Loss: 3.426781446177785
# Accuracy rate: 28.82%
# Epoch 3/30, Loss: 2.937955616741646
# Accuracy rate: 36.20%
# Epoch 4/30, Loss: 2.557198732655223
# Accuracy rate: 41.15%
# Epoch 5/30, Loss: 2.2828362796364763
# Accuracy rate: 44.38%
# Epoch 6/30, Loss: 2.020633293942707
# Accuracy rate: 47.31%
# Epoch 7/30, Loss: 1.7855786591041378
# Accuracy rate: 49.33%
# Epoch 8/30, Loss: 1.6208327712082282
# Accuracy rate: 51.77%
# Epoch 9/30, Loss: 1.4384829012359062
# Accuracy rate: 54.58%
# Epoch 10/30, Loss: 1.2610429757978858
# Accuracy rate: 53.48%
# Epoch 11/30, Loss: 1.1178159079900603
# Accuracy rate: 56.04%
# Epoch 12/30, Loss: 1.0176351765306983
# Accuracy rate: 56.78%
# Epoch 13/30, Loss: 0.9209188669193081
# Accuracy rate: 54.64%
# Epoch 14/30, Loss: 0.8216246253106653
# Accuracy rate: 55.68%
# Epoch 15/30, Loss: 0.7295350612663641
# Accuracy rate: 58.79%
# Epoch 16/30, Loss: 0.6582389056682587
# Accuracy rate: 59.10%
# Epoch 17/30, Loss: 0.5940092789690669
# Accuracy rate: 58.79%
# Epoch 18/30, Loss: 0.5336900437750467
# Accuracy rate: 56.78%
# Epoch 19/30, Loss: 0.5041012452142994
# Accuracy rate: 56.23%
# Epoch 20/30, Loss: 0.43941344022750856
# Accuracy rate: 59.10%
# Epoch 21/30, Loss: 0.4400984351227923
# Accuracy rate: 56.59%
# Epoch 22/30, Loss: 0.3733898963870072
# Accuracy rate: 58.24%
# Epoch 23/30, Loss: 0.35450238585472105
# Accuracy rate: 59.40%
# Epoch 24/30, Loss: 0.3448162566597869
# Accuracy rate: 57.45%
# Epoch 25/30, Loss: 0.3221204587235683
# Accuracy rate: 58.12%
# Epoch 26/30, Loss: 0.28335086943172827
# Accuracy rate: 58.85%
# Epoch 27/30, Loss: 0.2633045602498985
# Accuracy rate: 59.16%
# Epoch 28/30, Loss: 0.24268730151944043
# Accuracy rate: 58.49%
# Epoch 29/30, Loss: 0.22828393763885266
# Accuracy rate: 57.94%
# Epoch 30/30, Loss: 0.245116925239563
# Accuracy rate: 59.34%
# Model trained and saved.
#
# Process finished with exit code 0
