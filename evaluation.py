import itertools
import os
import cv2
from keras.models import load_model
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory,to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

model = load_model('model.h5')
imagefolder = "D:/UKM Y3 S1/TTTC 3413 Aplikasi Robot/archive/"
classes = ["Angry", "fear", "Happy", "Neutral"]


total_images = 0
for dir_ in os.listdir(imagefolder):
    count = 0
    for f in os.listdir(imagefolder + dir_ + "/"):
        count += 1
        total_images += 1
    print(f"{dir_} has {count} number of images")

print(f"\ntotal images are {total_images}")


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


y_pred = np.argmax(model.predict(X_test), axis=1)
ytest_ = np.argmax(y_test, axis=1)


test_accu = np.sum(ytest_ == y_pred) / len(ytest_) * 100
print(f"test accuracy: {round(test_accu, 4)} %\n\n")
print(classification_report(ytest_, y_pred,target_names=classes))

cm = confusion_matrix(ytest_, y_pred)
print(cm)
