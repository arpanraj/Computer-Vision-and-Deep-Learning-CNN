#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 20:37:49 2022

@author: arpanrajpurohit
"""


from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix
import numpy as np

#STEP - 4
pre_model = InceptionResNetV2(weights="imagenet", include_top =False, input_shape=(150,150, 3))

FOLDER_PATH = "/Users/arpanrajpurohit/Desktop/courses/Computer_Vision_and_Deep_Learning_510/Programs/Programming3/cats_dogs_dataset/dataset/"
TEST_PATH  = FOLDER_PATH + "test_set/"
TRAINING_PATH = FOLDER_PATH + "training_set/"
DOGS_PATH = "dogs/"
CATS_PATH = "cats/"

def pre_process(img,label):
    img = tf.cast(img/255. ,tf.float32)
    return img,label

train_set = image_dataset_from_directory(TRAINING_PATH, shuffle=True, batch_size=32, image_size=(150, 150))
test_set = image_dataset_from_directory(TEST_PATH, shuffle=True, batch_size=32, image_size=(150, 150))
train_set = train_set.map(pre_process)
test_set = test_set.map(pre_process)


model = models.Sequential()
for layer in pre_model.layers[0:7]:
    model.add(layer)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
pre_model.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5),  
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=keras.metrics.BinaryAccuracy())
model.fit(train_set, epochs=5,validation_data=test_set)
true = np.concatenate([y for x, y in test_set], axis=0)
predicted = model.predict(test_set)
conf_mat2 = confusion_matrix(true,np.int_(predicted))