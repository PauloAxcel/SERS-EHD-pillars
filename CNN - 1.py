#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:09:09 2021

@author: emilymassey
"""

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
import os
import numpy as np


# define all directories of images
train_path = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/Sorting Images/Data/Training/'
validate_path = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/Sorting Images/Data/Validation/'
test_path = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/Sorting Images/Data/Testing/'
save_output = r'/Users/emilymassey/Library/Mobile Documents/com~apple~CloudDocs/Research Project/CNN/Python output/'

#read images from directories
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(train_path, batch_size = 32, classes = ['SERS', 'NONSERS'], class_mode = 'categorical')
validation_generator =  test_datagen.flow_from_directory(validate_path, batch_size = 20, classes = ['SERS', 'NONSERS'], class_mode = 'categorical')

# binary convnet
model = models.Sequential()
model.add(layers.Conv2D(192, (3, 3), activation ='relu', input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(96, (3, 3), activation = 'relu'))
model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(2, activation = 'sigmoid'))

# configure model for training
model.compile(loss = 'categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-3), metrics = ['acc'])

# callbacks
# callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3)]
# callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', patience = 1), keras.callbacks.ModelCheckpoint(filepath = save_output, monitor = 'val_loss', save_best_only = True)]

# fitting model using a batch generator
history = model.fit(train_generator, steps_per_epoch= 10, epochs=50, validation_data = validation_generator)

model.save('CNN - 1, all callbacks, all data, no augmentation')

# plot validation and accuracy curves
acc = np.array(history.history['acc'])
val_acc = np.array(history.history['val_acc'])
loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])

epochs = range(1, len(acc) + 1)

#accuracy
plt.plot(epochs, acc, 'bo', label ='Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

#validation
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()

