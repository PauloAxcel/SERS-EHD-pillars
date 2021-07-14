# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:05:02 2019

@author: paulo
"""

#DATA AUGMENTATION
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
from skimage.color import rgb2gray

from keras.models import load_model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras.layers import InputLayer
from keras.models import Sequential
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

from keras.applications.vgg16 import VGG16

from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.input_modifiers import Jitter
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator

import cv2

from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.input_modifiers import Jitter
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator

from vis.utils import utils
#from keras import activations
from keras.preprocessing import image
import keras
from keras.layers import Dropout
from keras import backend as K
from keras import regularizers
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

#get the folders
SERS_train_dir = r'/home/newuser/Desktop/emily try/Data/SERS/'
NOENH_train_dir = r'/home/newuser/Desktop/emily try/Data/nonSERS/'

gen_dir_tra = r'/home/newuser/Desktop/emily try/Data/Training/'
gen_dir_val = r'/home/newuser/Desktop/emily try/Data/Validation/'


#get the files inside the folders
SERS_train = os.listdir(SERS_train_dir)
NOENH_train = os.listdir(NOENH_train_dir)

all_dir = [SERS_train_dir, NOENH_train_dir]
all_data = [SERS_train, NOENH_train]

shape = []

for dire,file in zip(all_dir, all_data):
    
    folder = dire.split('/')[-2]+'/'
    for f in file:
        ima = cv2.imread(dire+f)
        shape.append([ima.shape[0],ima.shape[1]])
    

max_shape = np.max(shape,axis=0)
min_shape = np.min(shape,axis=0)


# image = cap[0]
path = r'/home/newuser/Desktop/alex/'

input_frames =  'test for big area image classification.png'

size =  cv2.imread(path+input_frames).shape[0:2]

division = np.mean(size)//np.mean([max_shape,min_shape])

from PIL import Image

height = max_shape[0]
width = max_shape[1]

im = Image.open(path+input_frames)
imgwidth, imgheight = im.size
test_image = []

for i in range(0,imgheight,height):
    for j in range(0,imgwidth,width):
        box = (j, i, j+width, i+height)
        a = im.crop(box)
        test_image.append(a)
        # try:
        #     a.save(os.path.join(path,'img'+str(i)+'_'+str(j)+'.png'))
        # except:
        #     pass



classifier = load_model(r'/home/newuser/Desktop/alex/SERS_NOSERS_pillars_v05.h5')

model = Sequential()
# model.add(InputLayer(input_shape=(150,150)))
for layer in classifier.layers:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False




labels = ['No Enhancement', 'SERS']
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)

heatmaps = []
from keras.preprocessing import image

for f in test_image:

    img = f.convert('RGB').resize((150,150), Image.ANTIALIAS)

#    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
#    x = preprocess_input(x)
    
    preds = model.predict(x)

    
    color = []
    num = np.argmax(preds)
    
    
    if num == 0:
        color.append(['r','k'])
    if num == 1:
        color.append(['k','r'])

        
#output from the conv net and not from the pooling...   
    
  
    img_output = model.layers[0].layers[-1].output[:,num]
    last_conv_layer = model.layers[0].get_layer('block5_conv3')
    
    grads = K.gradients(img_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis= (0,1,2))
#    pooled_grads = K.mean(grads, axis= (0,1,2))
    
    iterate = K.function([ model.layers[0].layers[0].input],
                          [pooled_grads, last_conv_layer.output[0]])
    
    pooled_grads_value , conv_layer_output_value = iterate([x])
    
    for j in range(512):
        conv_layer_output_value[:,:,j] *= pooled_grads_value[j]
        
 
    
    heatmap = np.mean(conv_layer_output_value , axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    
    img = f.convert('RGB').resize((150,150), Image.ANTIALIAS)

    
    heatmap = cv2.resize(heatmap, (img.size[1],img.size[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    


    img *= np.uint8(255.0/max(max(img.getextrema())))

        
    blend = cv2.addWeighted(img,0.5, heatmap,0.5, 0)
    heatmaps.append(blend)
    


for h in heatmaps:
    fig, ax = plt.subplots()
    plt.axis('off')
    im = ax.imshow(h,interpolation='lanczos',cmap='hot')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)   
    
    range_b = [blend.min(), np.mean([h.max(),h.min()]), h.max()]
    
    cbar = fig.colorbar(im, cax=cax, ticks=range_b, orientation='vertical')
    cbar.ax.set_yticklabels(['Low', 'Medium', 'High'], fontdict={'fontsize': 18, 'fontweight': 'medium'})  # horizontal colorbar

w,h,c=heatmaps[i*(int(division)+1)+j].shape
itera = int(division)+1
final_heat = np.zeros(shape=(w*itera,h*itera,c))

for i in range(itera):
    for j in range(itera): 
        final_heat[i*w:(i+1)*w,j*h:(j+1)*h,:] = heatmaps[i*(int(division)+1)+j]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        





