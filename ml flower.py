import shutil
from keras.src.utils import image_dataset_from_directory
from scipy.io import loadmat
import os
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.data import AUTOTUNE


class Flower:
    def __init__(self, datapath, labels_path, img_size=(224, 224), batch_size=32):
        self.labels_path = labels_path
        self.datapath = os.path.join(datapath, "organized")
        self.img_size = img_size
        self.batch_size = batch_size
        if not self.is_already_organized():
            self.create_sub_directories(datapath)
        else:
            print("Dataset is already organized.")

        self.train_dataset, self.valid_dataset = self.load_dataset()

    def load_dataset(self):
        train_dataset = image_dataset_from_directory(
            self.datapath,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=self.img_size,
            batch_size=self.batch_size
        )

        valid_dataset = image_dataset_from_directory(
            self.datapath,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=self.img_size,
            batch_size=self.batch_size
        )

        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
        valid_dataset = valid_dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        valid_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)

        return train_dataset, valid_dataset

    def create_sub_directories(self, original_datapath):
        labels_data = loadmat(self.labels_path)
        labels = labels_data["labels"][0] - 1

        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        num_classes = len(set(labels))
        for i in range(num_classes):
            class_folder = os.path.join(self.datapath, f"class_{i}")
            os.makedirs(class_folder, exist_ok=True)

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

# Feature Extraction + Classifier – Extracting features from a CNN
# (e.g., using intermediate layers) and training a separate classifier like SVM, Random Forest, or a simple MLP on top.
class CustomCNN():
    def __init__(self, num_classes):
        None




    def forward(self, x):
        return x

def predict_image(image_path, model):
    None

if __name__ == "__main__":
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    labels_path = "C:/Users/vbacho/OneDrive - UW/ml flower/imagelabels.mat"
    datapath = "C:/Users/vbacho/OneDrive - UW/ml flower/jpg"
    flower = Flower(datapath, labels_path)



    num_classes = 102



    epochs = 30
    for epoch in range(epochs):
        None

