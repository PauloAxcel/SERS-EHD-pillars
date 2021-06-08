# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 23:22:07 2020

@author: paulo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:58:53 2019

@author: paulo
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import cv2
from keras import layers
from keras import models

from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential

from keras.applications import VGG16

from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.input_modifiers import Jitter
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator

import cv2

from keras.applications import VGG16

from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.input_modifiers import Jitter
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator

from keras.applications import VGG16
from vis.utils import utils
#from keras import activations
from keras.preprocessing import image
import keras
from keras.layers import Dropout
from keras import backend as K
from keras import regularizers
from mpl_toolkits.axes_grid1 import make_axes_locatable

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *=0.1
    
    x +=0.5
    x = np.clip(x,0,1)
    
    x *= 255
    x = np.clip(x , 0 , 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, sizex = 128 , sizey = 141):
    
    layer_output = model.layers[0].get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:, filter_index])
    
    grads = K.gradients(loss, model.layers[0].layers[0].input)[0]
    
    grads /= (K.sqrt(K.mean(K.square(grads)))+ 1e-5)
    
    iterate = K.function([model.layers[0].layers[0].input], [loss, grads])
    
    input_img_data = np.random.random((1,sizex,sizey,3))*20+128
    
    step = 1
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    
    img = input_img_data[0]
    
    return deprocess_image(img)





#DEFINE PATHS FOR SAMPLES WITH DATA AUGMENTATION IN

train_path = 'C:\\Users\\Paulo\\Desktop\\afm different colours\\keras_imaging\\data augmentation with original data\\train\\'

#valid_path = 'C:\\Users\\paulo\\Desktop\\birmingham_02\\sers ehd thyo10mM\\extended vs roughness\\afm different colours\\keras_imaging\\validation'
#test_path = 'C:\\Users\\paulo\\Desktop\\birmingham_02\\sers ehd thyo10mM\\extended vs roughness\\afm different colours\\keras_imaging\\test'
test_path = 'C:\\Users\\Paulo\\Desktop\\afm different colours\\keras_imaging\\prediction2\\height_change\\'

valid_path ='C:\\Users\\Paulo\\Desktop\\afm different colours\\keras_imaging\\data augmentation with original data\\validation\\'


train_batch = ImageDataGenerator(rescale=1/.255).flow_from_directory(train_path, target_size=(128,141),
                                 classes=['NOENH','SLIENH','SERS'], batch_size = 32)
valid_batch = ImageDataGenerator(rescale=1/.255).flow_from_directory(valid_path, target_size=(128,141),
                                 classes=['NOENH','SLIENH','SERS'], batch_size = 32)
#test_batch = ImageDataGenerator().flow_from_directory(test_path, target_size=(128,141),classes=['NOENH','SLIENH','SERS'], batch_size = 51)

#USE PRETRAINED CNN REMOVE BOTTOM PART

conv_base = VGG16(weights='imagenet', include_top = False, input_shape=(128,141,3))

classifier = Sequential()


#FREEZE THE WEIGHTS OF THE CONVNET

conv_base.trainable = False
classifier.add(conv_base)
classifier.add(Dropout(0.2))
classifier.add(Flatten()) 
classifier.add(Dense(256, activation = 'relu'))
#classifier.add(Dense(256, activation = 'relu',kernel_regularizer=regularizers.l2(0.01),
#                activity_regularizer=regularizers.l1(0.01)))
classifier.add(Dense(3, activation = 'softmax'))

#TRAIN THE CONVNET FOR THE OUTPUT LAYER

classifier.compile(optimizer = 'rmsprop',
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])


checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", 
                               monitor = 'val_accuracy',
                               verbose=1, 
                               save_best_only=True)


history = classifier.fit_generator(train_batch,
                                   steps_per_epoch = 100,
                                   epochs = 20,
                                   callbacks=[checkpointer],
                                   validation_data = valid_batch,
                                   validation_steps = 50)

#LAYER FINE TUNE OF THE LAST CONVNET BLOCK, THE LAST BLOCK TAKES CARE OF CHARACTERIZING CLASS DIFFERENCES

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

from keras import optimizers        
        
classifier.compile(optimizer = optimizers.RMSprop(lr=1e-5),
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])
      
  
history = classifier.fit_generator(train_batch,
                                   steps_per_epoch = 100,
                                   epochs = 20,
                                   validation_data = valid_batch,
                                   validation_steps = 50)        
        
#CHECK IF THERE IS OVERFITTING BY DRAWING THE ACC AND LOSS

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

   
    
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs,val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()




model = Sequential()
for layer in classifier.layers:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

import os
from PIL import Image
from skimage.color import rgb2gray

file = os.listdir(test_path)

#PREDICT TEST IMAGES

labels = ['No Enhancement','Slightly Enhancement' , 'SERS']


for i in range(len(file)):
    img = image.load_img(test_path+file[i], target_size=(128,141))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    
    preds = model.predict(images)
    
    color = []
    
    if np.argmax(preds) == 0:
        color.append(['r','k','k'])
    if np.argmax(preds) == 1:
        color.append(['k','r','k'])
    if np.argmax(preds) == 2:
        color.append(['k','k','r'])
        
#output from the conv net and not from the pooling...        

    img_output = model.layers[0].layers[-2].output[:,np.argmax(preds)]
    last_conv_layer = model.layers[0].get_layer('block5_conv3')
    
    grads = K.gradients(img_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis= (0,1,2))
    
    iterate = K.function([ model.layers[0].layers[0].input],
                         [pooled_grads, last_conv_layer.output[0]])
    
    pooled_grads_value , conv_layer_output_value = iterate([x])
    
    for j in range(512):
        conv_layer_output_value[:,:,j] *= pooled_grads_value[j]
        
    heatmap = np.mean(conv_layer_output_value , axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    
    img = cv2.imread(test_path+file[i])
    
    heatmap = cv2.resize(heatmap, (img.shape[1],img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    
#    fig ,(ax1,ax2) = plt.subplots(1,2)
    fig, ax = plt.subplots()
    plt.axis('off')

    img *= np.uint8(255.0/img.max())

    blend = cv2.addWeighted(img,0.5, heatmap,0.5, 0)
    
    im = ax.imshow(blend,interpolation='lanczos',cmap='hot')
    
    x = np.linspace(0,blend.shape[1], blend.shape[1])
    y = np.linspace(0, blend.shape[0], blend.shape[0])
    X, Y = np.meshgrid(x, y)
    
    contour = ax.contour(X,Y,rgb2gray(heatmap),3, colors = 'black')
    ax.clabel(contour, inline=True, fontsize=18)

    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)   
    
    
    cbar = fig.colorbar(im, cax=cax, ticks=[blend.min(), np.mean([blend.max(),blend.min()]), blend.max()], orientation='vertical')
    cbar.ax.set_yticklabels(['Low', 'Medium', 'High'], fontdict={'fontsize': 18, 'fontweight': 'medium'})  # horizontal colorbar
    
    ax.text(img.shape[1]+12.5,-2,'Activation', fontdict={'fontsize': 24, 'fontweight': 'medium'})
    ax.text(0 , -2, labels[0] +  " %.1f" % (preds[0][0]*100) ,fontdict={'fontsize': 18, 'fontweight': 'medium'} , color=color[0][0])
    ax.text(img.shape[1]*2//5 , -2, labels[1] + " %.1f" % (preds[0][1]*100), fontdict={'fontsize': 18, 'fontweight': 'medium'} , color=color[0][1])
    ax.text(img.shape[1]*4//5 , -2, labels[2] + " %.1f" % (preds[0][2]*100), fontdict={'fontsize': 18, 'fontweight': 'medium'} , color=color[0][2])
    
    ax.text(img.shape[0]//2,-8,file[i],fontdict={'fontsize': 24, 'fontweight': 'medium'})
    
    fig.savefig('C:\\Users\\Paulo\\Desktop\\afm different colours\\DNN\\'+file[i]+'.png', dpi = 300,bbox_inches="tight")
