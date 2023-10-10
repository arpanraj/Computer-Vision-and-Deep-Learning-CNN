#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 15:22:27 2022

@author: arpanrajpurohit
"""

from tensorflow.keras.applications import InceptionResNetV2
from matplotlib import pyplot as plt
from matplotlib.image import imread
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import random

#STEP - 1
pre_model = InceptionResNetV2(weights="imagenet", include_top =False, input_shape=(150,150, 3))
#pre_model.summary()
layer = pre_model.layers[1]
filters = layer.get_weights()[0]
print(layer.name, filters.shape)

#plot figure
figure=plt.figure(figsize=(8, 4))
columns = 8
rows = 4
filter_count = 32
for i in range(1, filter_count +1):
    m_filter = filters[:, :, :, i-1]
    #normalise
    filter_min = m_filter.min()
    filter_max = m_filter.max()
    m_filter = (m_filter - filter_min) / (filter_max - filter_min)
    figure =plt.subplot(rows, columns, i)
    figure.set_xticks([])  
    figure.set_yticks([])
    plt.imshow(m_filter[:, :, :], cmap='hsv') 
plt.show()

#STEP - 2

FOLDER_PATH = "/Users/arpanrajpurohit/Desktop/courses/Computer_Vision_and_Deep_Learning_510/Programs/Programming3/cats_dogs_dataset/dataset/"
TEST_PATH  = FOLDER_PATH + "test_set/"
TRAINING_PATH = FOLDER_PATH + "training_set/"
DOGS_PATH = "dogs/"
CATS_PATH = "cats/"

def standard_channel(channel):
    mean = np.mean(channel)
    std = np.std(channel)
    channel = (channel - mean) / std
    return channel

def standardise_image(image):
    r_channel = standard_channel(image[:,:,0])
    g_channel = standard_channel(image[:,:,1])
    b_channel = standard_channel(image[:,:,2])
    return np.stack([r_channel, g_channel, b_channel], axis=-1)

#load train images
train_arr = []
for i in range(1,4001):
    dog_name = TRAINING_PATH + DOGS_PATH + 'dog.' + str(i) + '.jpg'
    dog_image = load_img(dog_name, target_size=(150, 150))
    cat_name = TRAINING_PATH + CATS_PATH + 'cat.' + str(i) + '.jpg'
    cat_image = load_img(cat_name, target_size=(150, 150))
    dog_img_arr = standardise_image(img_to_array(dog_image))
    cat_img_arr = standardise_image(img_to_array(cat_image))
    train_arr.append([dog_img_arr,1])
    train_arr.append([cat_img_arr,0])

train_arr = np.array(train_arr)
np.random.shuffle(train_arr)

#load test images
test_arr = []
for i in range(4001,5001):
    dog_name = TEST_PATH + DOGS_PATH + 'dog.' + str(i) + '.jpg'
    dog_image = load_img(dog_name, target_size=(150, 150))
    cat_name = TEST_PATH + CATS_PATH + 'cat.' + str(i) + '.jpg'
    cat_image = load_img(cat_name, target_size=(150, 150))
    dog_img_arr = standardise_image(img_to_array(dog_image))
    cat_img_arr = standardise_image(img_to_array(cat_image))
    test_arr.append([dog_img_arr,1])
    test_arr.append([cat_img_arr,0])
test_arr =  np.array(test_arr)
np.random.shuffle(test_arr)

pre_model.trainable = False
model = models.Sequential()
model.add(pre_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

#image = test_dog_image_arrs[0].reshape((1, test_dog_image_arrs[0].shape[0], test_dog_image_arrs[0].shape[1], test_dog_image_arrs[0].shape[2]))
#image = preprocess_input(image)

#(1)
#model.predict(image)
data = preprocess_input(test_arr)
y = model.predict(preprocess_input(test_arr))