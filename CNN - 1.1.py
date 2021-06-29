#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:09:09 2021

@author: emilymassey
"""

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.applications.vgg16 import VGG16


# define all directories of images
train_path = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/Sorting Images/Data/Training/'
validate_path = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/Sorting Images/Data/Validation/'
test_path = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/Sorting Images/Data/Testing/'
train_sers_dir = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/Sorting Images/Data/Training/SERS/'
train_non_dir = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/Sorting Images/Data/Training/Non-SERS/'
valid_sers_dir = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/Sorting Images/Data/Validation/SERs/'
valid_non_dir = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/Sorting Images/Data/Validation/Non-SERS/'
test_sers_dir = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/Sorting Images/Data/Testing/SERS/'
test_non_dir = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/Sorting Images/Data/Testing/Non-SERS/'

# sample counts
train_count = len(os.listdir(train_sers_dir)) + len(os.listdir(train_non_dir))
valid_count = len(os.listdir(valid_sers_dir)) + len(os.listdir(valid_non_dir))
test_count = len(os.listdir(test_sers_dir)) + len(os.listdir(test_non_dir))

# start VGG16 base
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# extract features using pretrained CNN
# datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

# define how to extract features
# def extract_features(directory, sample_count):
#     features = np.zeros(shape =(sample_count, 4, 4, 512))
#     labels = np.zeros(shape =(sample_count))
#     generator = datagen.flow_from_directory(directory, target_size = (150, 150), batch_size = batch_size, class_mode = 'binary')
#     i = 0
#     for inputs_batch, labels_batch in generator:
#         features_batch = conv_base.predict(inputs_batch)
#         features[i * batch_size : (i + 1) * batch_size] = features_batch
#         labels[i * batch_size : (i + 1) * batch_size] = labels_batch
#         i += 1
#         if i * batch_size >= sample_count:
#             break
#     return features, labels

# extract correct features
# train_features, train_labels = extract_features(train_path, train_count)
# validation_features, validation_labels = extract_features(validate_path, valid_count)
# test_features, test_labels = extract_features(test_path, test_count)

train_features = ImageDataGenerator(rescale=1/.255).flow_from_directory(train_path, target_size = (150, 150), batch_size = batch_size)
validation_features = ImageDataGenerator(rescale=1/.255).flow_from_directory(validate_path, target_size = (150, 150), batch_size = batch_size)
test_features = ImageDataGenerator(rescale=1/.255).flow_from_directory(test_path, target_size = (150, 150), batch_size = batch_size)

# flatten samples for addition
train_features = np.reshape(train_features, (train_count, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (valid_count, 4 * 4 * 512))
test_features = np.reshape(test_features, (test_count, 4 * 4 * 512))


# # binary convnet
model = models.Sequential()
conv_base.trainable = False
model.add(conv_base)
model.add(layers.Flatten)
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = optimizers.RMSprop(lr=2e-5), loss = 'binary_crossentropy', metrics =['acc'])

history = model.fit(train_features, train_labels, epochs = 1, batch_size = 20, validation_data = (validation_features, validation_labels))

# plot accuracy and loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# #accuracy
plt.plot(epochs, acc, 'bo', label ='Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

# #validation
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()

