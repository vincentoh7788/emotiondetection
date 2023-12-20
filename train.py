import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import os
import cv2
import math

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Flatten, Dropout,BatchNormalization,Activation,GaussianNoise,GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory,to_categorical
from tensorflow.keras import regularizers


def build_model(bottom_model,classes):
    model = bottom_model.layers[-2].output
    model = GlobalAveragePooling2D()(model)
    model = Dense(classes, activation='softmax', name='out_layer')(model)
    return model

imagefolder = "D:/UKM Y3 S1/TTTC 3413 Aplikasi Robot/archive/"

total_images = 0
for dir_ in os.listdir(imagefolder):
    count = 0
    for f in os.listdir(imagefolder + dir_ + "/"):
        count += 1
        total_images += 1
    print(f"{dir_} has {count} number of images")

print(f"\ntotal images are {total_images}")

IMAGE_SIZE = (48,48)
IMAGE_SHAPE = IMAGE_SIZE + (3,)

BS = 64
EPOCHS = 20

classes = ["fear", "Happy", "Neutral", "Angry"]
img_arr = np.empty(shape=(total_images,48,48,3))
img_label = np.empty(shape=(total_images))
label_to_text = {}

i = 0
e = 0
for dir_ in os.listdir(imagefolder):
    if dir_ in classes:
        label_to_text[e] = dir_
        for f in os.listdir(imagefolder + dir_ + "/"):
            img_arr[i] = cv2.imread(imagefolder + dir_ + "/" + f)
            img_label[i] = e
            i += 1
        print(f"loaded all {dir_} images to numpy arrays")
        e += 1

img_label = to_categorical(img_label)

img_arr = img_arr / 255.
X_train, X_test, y_train, y_test = train_test_split(img_arr, img_label,
                                                    shuffle=True, stratify=img_label,
                                                    train_size=0.8, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#pretrained weight
vgg = tf.keras.applications.VGG19(weights = 'imagenet',
                                  include_top = False,
                                  input_shape = (48, 48, 3))
head = build_model(vgg, 4)

model = Model(inputs = vgg.input, outputs = head)

print(model.summary())


early_stopping = EarlyStopping(monitor = 'val_accuracy',
                               min_delta = 0.00005,
                               patience = 11,
                               verbose = 1,
                               restore_best_weights = True)

lr_scheduler = ReduceLROnPlateau(monitor = 'val_accuracy',
                                 factor = 0.5,
                                 patience = 7,
                                 min_lr = 1e-7,
                                 verbose = 1)

callbacks = [early_stopping,lr_scheduler]
optims = [Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999)]
model.compile(loss = 'categorical_crossentropy',
              optimizer = optims[0],
              metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    zca_whitening=False,
)
train_datagen.fit(X_train)
history = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=BS),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) / BS,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save("model.h5", save_format="h5")
