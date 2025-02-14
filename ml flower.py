import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import tensorflow_datasets as tfds
import os
from PIL import Image
import numpy as np

class flower:
    def __init__(self, datapath):
        self.data = self.convert_dataset(datapath)

    def convert_dataset(self, datapath):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        ])
        dataset = ImageFolder(datapath, transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        data_iter = iter(dataloader)
        images, labels = next(data_iter)

        print(images.shape)
        print(labels)


if __name__ == "__main__":
    #datapath = input("Where is your flower data set stored ")
    datapath = "C:/Users/vbacho/OneDrive - UW/ml flower/jpg"
    flower(datapath)