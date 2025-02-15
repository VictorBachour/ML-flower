import shutil
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from scipy.io import loadmat

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
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader

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



# import torch
# import torch.nn as nn
# import torch.optim as optim
# class FlowerClassifier(nn.Module):
#     def __init__(self):
#         super(FlowerClassifier, self).__init__()

if __name__ == "__main__":
    labels_path = "C:/Users/vbacho/OneDrive - UW/ml flower/imagelabels.mat"
    datapath = "C:/Users/vbacho/OneDrive - UW/ml flower/jpg"
    flower = Flower(datapath, labels_path)