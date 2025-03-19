import shutil

from PIL.ImageStat import Global
from keras import Input, Model
from keras.src.layers import ReLU, BatchNormalization, Conv2D, Dense, Dropout
from keras.src.utils import image_dataset_from_directory
from scipy.io import loadmat
import os
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.layers import GlobalAveragePooling2D



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


class CustomCNN():
    def __init__(self, input_shape=(224,224,3), num_classes=102, dropout_rate=0.5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def conv_block(self, x, filters, kernel_size=3, strides=1):
        x = Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def build_model(self):
        inputs = Input(shape=self.input_shape, dtype=tf.float32)  # Explicit dtype
        x = self.conv_block(inputs, 32)
        x = self.conv_block(x, 64, strides=2)
        x = self.conv_block(x, 128)
        x = self.conv_block(x, 256, strides=2)
        x = self.conv_block(x, 512)

        print(f"Type before GlobalAveragePooling2D: {type(x)}")  # Debugging line

        x = GlobalAveragePooling2D()(x)  # Ensure x is a tensor

        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs, x)
        return model
    def compile(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])


if __name__ == "__main__":
    labels_path = "C:/Users/vbacho/OneDrive - UW/ml flower/imagelabels.mat"
    datapath = "C:/Users/vbacho/OneDrive - UW/ml flower/jpg"
    flower = Flower(datapath, labels_path)
    cnn_model = CustomCNN()
    cnn_model.compile()

    history = cnn_model.model.fit(
        flower.train_dataset,
        validation_data=flower.valid_dataset,
        epochs=25,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
        ]
    )

    # Evaluate model
    test_loss, test_acc = cnn_model.model.evaluate(flower.valid_dataset)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

