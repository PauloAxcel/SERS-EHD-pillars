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

#identifies what is the average size of the pillars images
def image_size(SERS,NONSERS):
    
    #get the files inside the folders
    SERS_train = os.listdir(SERS)
    NONSERS_train = os.listdir(NONSERS)
    
    all_dir = [SERS, NONSERS]
    all_data = [SERS_train, NONSERS_train]
    
    shape = []
    
    for dire,file in zip(all_dir, all_data):
        
        folder = dire.split('/')[-2]+'/'
        for f in file:
            ima = cv2.imread(dire+f)
            shape.append([ima.shape[0],ima.shape[1]])
    
    return shape

from PIL import Image
#breaks the images in parts shaped like shape vector [x,y]
def break_image(input_frames,shape):
    
    if len(shape) == 2:
        
        height = int(shape[0])
        width = int(shape[1])
        size =  cv2.imread(input_frames).shape[0:2]
    
        division = [size[0]//height,size[1]//width]
        
        
    else:
            
        max_shape = np.max(shape,axis=0)
        min_shape = np.min(shape,axis=0)
        
        size =  cv2.imread(input_frames).shape[0:2]
    
        division = np.mean(size)//np.mean([max_shape,min_shape])
        
        shape_vec = np.mean([max_shape,min_shape],axis=0)
    
        height = int(shape_vec[0])
        width = int(shape_vec[1])
    
    im = Image.open(input_frames)
    imgwidth, imgheight = im.size
    test_image = []
    
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            test_image.append(a)
            
    return test_image,division




# image = cap[0]
# ADD ALEX SECTION HERE




#load the pillar identification model


    
def generate_regionof_interest(image_identifier,test_image,label):
        
        
    model = Sequential()
    for layer in image_identifier.layers:
        model.add(layer)
    
    for layer in model.layers:
        layer.trainable = False
    
    
    labels = ['pillar', 'nopillar']
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)
    
    heatmaps = []
    from keras.preprocessing import image
    
    location = []
    #get the activations from the test image broken in pieces
    
    for cnt,f in enumerate(test_image):
        img = f.convert('RGB').resize((150,150), Image.ANTIALIAS)
 
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        
        preds = model.predict(x)
        if preds[0][1]>0.5:
            location.append(1)
            plt.figure()
            plt.imshow(img)
            plt.savefig(r'/home/newuser/Desktop/alex/cnn pillars/gen/'+label+str(cnt)+'.png', dpi = 300,bbox_inches="tight")
            plt.close()
        else:
            location.append(0)
    
        num = np.argmax(preds)
            
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
        
        
        if max(max(img.getextrema()))==0:
            img *= np.uint8(255.0/(0.000001+max(max(img.getextrema()))))
        else:
            img *= np.uint8(255.0/max(max(img.getextrema())))
        
            
        blend = cv2.addWeighted(img,0.5, heatmap,0.5, 0)
        heatmaps.append(blend)
        
    return heatmaps,location
    



def plot_results(test,heatmaps,location,division):
    
    #plot the test image with heatmaps
    
    w,h,c=heatmaps[0].shape
    iteray = int(division[0])+1
    iterax = int(division[1])+1
    final_heath = np.zeros(shape=(w*iteray,h*iterax,c))
    
    for i in range(iteray):
        for j in range(iterax): 
            final_heath[i*w:(i+1)*w,j*h:(j+1)*h,:] = heatmaps[i*iterax+j]*location[i*iterax+j]
            
    # final_heat = final_heat//final_heat.max()
    plt.figure()
    plt.imshow(final_heath/255)
    
        
    import pandas as pd
    
    #perfect the mask into a square, filling up all previous area
    
    mask = pd.DataFrame(np.array(location).reshape(iteray,iterax))
    locx = []
    locy = []
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask.iloc[i,j] == 1:
                locx.append(i)
                locy.append(j)
    
    mask_fill = pd.DataFrame(np.zeros((iteray,iterax)))
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (i>np.min(locx)-1 and i<np.max(locx)+1) and (j>np.min(locy)-1 and j<np.max(locy)+1):
                mask_fill.iloc[i,j] = 1
    
    
    #display the test image in the previous square shape
    
    w,h,c=heatmaps[0].shape
    iteray = int(division[0])+1
    iterax = int(division[1])+1
    final_heat = np.zeros(shape=(w*iteray,h*iterax,c))
    
    
    for i in range(iteray):
        for j in range(iterax): 
            
            fimg = test_image[i*iterax+j].convert('RGB').resize((150,150), Image.ANTIALIAS)
            
            if max(max(fimg.getextrema()))==0:
                fimg *= np.uint8(255.0/(0.000001+max(max(fimg.getextrema()))))
            else:
                fimg *= np.uint8(255.0/max(max(fimg.getextrema())))
            
            final_heat[i*w:(i+1)*w,j*h:(j+1)*h,:] = fimg*mask_fill.stack().tolist()[i*iterax+j]
            
    # final_heat = final_heat//final_heat.max()
    plt.figure()
    plt.imshow(final_heat/255)
    
        
    plt.figure()
    plt.imshow(Image.open(test).convert('RGB').resize((final_heat.shape[1],final_heat.shape[0]), Image.ANTIALIAS))
    

    return final_heat





def output_cropped_interest_area(final_heat):
        
    final_test = np.uint8(final_heat)
    
    gray = cv2.cvtColor(final_test,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    
    crop = final_test[y:y+h,x:x+w]
    
    plt.figure()
    plt.imshow(crop)

    return crop




#get the image coming from the wire software

test = r'/home/newuser/Desktop/alex/cnn pillars/test/test.jpg'

#get the x and y dimentions of the pillar image from the example folder, or use a 50 by 50 example.

# size = image_size(r'/home/newuser/Desktop/emily try/Data/SERS/',r'/home/newuser/Desktop/emily try/Data/nonSERS/')

size = [50,50]

#break the test image into multiple images written on the test_images list, with a division list value related to the x and y shape of the number of 
#division done

test_image,division = break_image(test,size)

#load the pretrained model that identifies pillars in the wire graphical interface.

image_identifier = load_model(r'/home/newuser/Desktop/alex/cnn pillars/pillars.h5')

#from the broken images identify pillar locations, and save the activation in a heatmap list, and the location (which tiles) have an activation
#higher than 50%

heatmaps,location = generate_regionof_interest(image_identifier,test_image,'pillar_notpillar')

#plots heatmap activation over the initial test sample, plots over test image the complete reconstructed activation region.
final_heat = plot_results(test,heatmaps,location,division)


#cropes the test image into only the activation reagion
cropped = output_cropped_interest_area(final_heat)

#the preprocessing is done, now we can load the actual CNN model that distinguishes SERS from nonSERS and break again the crop image
#and analyse the tiles into SERS and not SERS

import PIL
import math

def break_image2(input_frames,shape):
  
    if len(shape) == 2:
      
      height = int(shape[0])
      width = int(shape[1])
      size =  input_frames.shape[0:2]
  
      division = [math.ceil(size[0]/height),math.ceil(size[1]/width)]
        
        
    else:
                
        max_shape = np.max(shape,axis=0)
        min_shape = np.min(shape,axis=0)
        
        size = input_frames.shape[0:2]
    
        division = np.mean(size)//np.mean([max_shape,min_shape])
        
        shape_vec = np.mean([max_shape,min_shape],axis=0)
    
        height = int(shape_vec[0])
        width = int(shape_vec[1])
        
    im = PIL.Image.fromarray(input_frames)
    imgwidth, imgheight = im.size
    test_image = []
    
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            test_image.append(a)
            
    return test_image,division




def plot_results2(test,heatmaps,division,location):
    
  #plot the test image with heatmaps
    
    w,h,c = heatmaps[0].shape
    iteray = int(division[0])
    iterax = int(division[1])
    final_heath = np.zeros(shape=(w*iteray,h*iterax,c))
    
    for i in range(iteray):
        for j in range(iterax): 
            final_heath[i*w:(i+1)*w,j*h:(j+1)*h,:] = heatmaps[i*iterax+j]*location[i*iterax+j]
            
    # final_heat = final_heat//final_heat.max()
    plt.figure()
    plt.imshow(final_heath/255)
        
    plt.figure()
    plt.imshow(test)

    return final_heat



    
def generate_regionof_interest2(image_identifier,test_image,label):
        
        
    model = Sequential()
    for layer in image_identifier.layers:
        model.add(layer)
    
    for layer in model.layers:
        layer.trainable = False
    
    
    labels = ['pillar', 'nopillar']
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)
    
    heatmaps = []
    from keras.preprocessing import image
    
    #get the activations from the test image broken in pieces
    location = []
    for cnt,f in enumerate(test_image):
        img = f.convert('RGB').resize((150,150), Image.ANTIALIAS)
 
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        
        preds = model.predict(x)
    
        num = np.argmax(preds)
        
        preds = model.predict(x)
        if preds[0][1]>0.5:
            location.append(1)
            plt.figure()
            plt.imshow(img)
            plt.savefig(r'/home/newuser/Desktop/alex/cnn pillars/gen/'+label+str(cnt)+'.png', dpi = 300,bbox_inches="tight")
            plt.close()
        else:
            location.append(0)
    
        
            
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
        
        
        if max(max(img.getextrema()))==0:
            img *= np.uint8(255.0/(0.000001+max(max(img.getextrema()))))
        else:
            img *= np.uint8(255.0/max(max(img.getextrema())))
        
            
        blend = cv2.addWeighted(img,0.5, heatmap,0.5, 0)
        heatmaps.append(blend)
        
    return heatmaps,location
    










#actual pillar size

# size2 = image_size(r'/home/newuser/Desktop/emily try/Data3_2122points/original folder/SERS/',
                    # r'/home/newuser/Desktop/emily try/Data3_2122points/original folder/nonSERS/')

size2 = [50,50]

newcropped = cropped[:,1000:,:]

test_image2,division2 = break_image2(newcropped,size2)

sers_identifier = load_model(r'/home/newuser/Desktop/emily try/Data4last/SERS_NOSERS_pillars_v05.h5')



heatmaps2,location2 = generate_regionof_interest2(sers_identifier,test_image2,'sers_notsers')
#plots heatmap activation over the initial test sample, plots over test image the complete reconstructed activation region.
final_heat2 = plot_results2(newcropped,heatmaps2,division2,location2)


    
    # return True

# pillar_locator(image_identifier, test_image)



#load first model to identify where the pillars are situated in an image
# test = r'/home/newuser/Desktop/alex/cnn pillars/test/test.jpg'
# image_break,division = break_image(test,[150,150])


      

    





# classifier = load_model(r'/home/newuser/Desktop/alex/SERS_NOSERS_pillars_v05.h5')

# model = Sequential()
# # model.add(InputLayer(input_shape=(150,150)))
# for layer in classifier.layers:
#     model.add(layer)

# for layer in model.layers:
#     layer.trainable = False

# labels = ['pillar', 'notpillar']
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)

# heatmaps = []
# from keras.preprocessing import image

# for f in test_image:

#     img = f.convert('RGB').resize((150,150), Image.ANTIALIAS)
  
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
# #    x = preprocess_input(x)
    
#     preds = model.predict(x)
    
#     if preds[0][1]>0.5:
#         plt.figure()
#         plt.imshow(img)
        

    
#     color = []
#     num = np.argmax(preds)
    
    
#     if num == 0:
#         color.append(['r','k'])
#     if num == 1:
#         color.append(['k','r'])

        
# #output from the conv net and not from the pooling...   
    
  
#     img_output = model.layers[0].layers[-1].output[:,num]
#     last_conv_layer = model.layers[0].get_layer('block5_conv3')
    
#     grads = K.gradients(img_output, last_conv_layer.output)[0]
#     pooled_grads = K.mean(grads, axis= (0,1,2))
# #    pooled_grads = K.mean(grads, axis= (0,1,2))
    
#     iterate = K.function([ model.layers[0].layers[0].input],
#                           [pooled_grads, last_conv_layer.output[0]])
    
#     pooled_grads_value , conv_layer_output_value = iterate([x])
    
#     for j in range(512):
#         conv_layer_output_value[:,:,j] *= pooled_grads_value[j]
        
 
    
#     heatmap = np.mean(conv_layer_output_value , axis=-1)
#     heatmap = np.maximum(heatmap,0)
#     heatmap /= np.max(heatmap)
    
#     img = f.convert('RGB').resize((150,150), Image.ANTIALIAS)

    
#     heatmap = cv2.resize(heatmap, (img.size[1],img.size[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

#     img *= np.uint8(255.0/max(max(img.getextrema())))

#     blend = cv2.addWeighted(img,0.5, heatmap,0.5, 0)
#     heatmaps.append(blend)
    


# # for h in heatmaps:
# #     fig, ax = plt.subplots()
# #     plt.axis('off')
# #     im = ax.imshow(h,interpolation='lanczos',cmap='hot')
# #     divider = make_axes_locatable(ax)
# #     cax = divider.append_axes("right", size="5%", pad=0.25)   
    
# #     range_b = [blend.min(), np.mean([h.max(),h.min()]), h.max()]
    
# #     cbar = fig.colorbar(im, cax=cax, ticks=range_b, orientation='vertical')
# #     cbar.ax.set_yticklabels(['Low', 'Medium', 'High'], fontdict={'fontsize': 18, 'fontweight': 'medium'})  # horizontal colorbar

# w,h,c=heatmaps[0].shape
# itera = int(division)+1
# final_heat = np.zeros(shape=(w*itera,h*itera,c))

# for i in range(itera):
#     for j in range(itera): 
#         final_heat[i*w:(i+1)*w,j*h:(j+1)*h,:] = heatmaps[i*itera+j]
        
# # final_heat = final_heat//final_heat.max()
# plt.imshow(final_heat/final_heat.max())
        
        


