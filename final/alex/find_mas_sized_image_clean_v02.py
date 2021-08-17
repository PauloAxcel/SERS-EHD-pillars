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

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image

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
        img = f.convert('RGB').resize((model.layers[0].layers[0].output.shape[1]
                                       ,model.layers[0].layers[0].output.shape[2]),
                                      Image.ANTIALIAS)
 
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        
        preds = model.predict(x)
        if preds[0][1]>0.5:
            location.append(1)
#            plt.figure()
#            plt.imshow(img)
#            plt.savefig(r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\MEng student\3rd year\report\code\files to transfer\gen\\'+label+str(cnt)+'.png', dpi = 300,bbox_inches="tight")
#            plt.close()
        else:
            location.append(0)
    
        num = np.argmax(preds)
            
        img_output = model.layers[0].layers[-2].output[:,num]
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
        
        img = f.convert('RGB').resize((f.size[0],f.size[1]), Image.ANTIALIAS)
        
        
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
    

import pandas as pd

def plot_results(test,heatmaps,location,division):
    
    #plot the test image with heatmaps
    
    w,h,c=heatmaps[0].shape
    iteray = int(division[0])
    iterax = int(division[1])
    final_heath = np.zeros(shape=(w*iteray,h*iterax,c))
    
    for i in range(iteray):
        for j in range(iterax): 
            final_heath[i*w:(i+1)*w,j*h:(j+1)*h,:] = heatmaps[i*iterax+j]*location[i*iterax+j]
            
    # final_heat = final_heat//final_heat.max()
#    plt.figure()
#    plt.imshow(final_heath/255)
        
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
    iteray = int(division[0])
    iterax = int(division[1])
    final_heat = np.zeros(shape=(w*iteray,h*iterax,c))
    
    
    for i in range(iteray):
        for j in range(iterax): 
            
            fimg = test_image[i*iterax+j].convert('RGB').resize((w,h), Image.ANTIALIAS)
            
            if max(max(fimg.getextrema()))==0:
                fimg *= np.uint8(255.0/(0.000001+max(max(fimg.getextrema()))))
            else:
                fimg *= np.uint8(255.0/max(max(fimg.getextrema())))
            
            final_heat[i*w:(i+1)*w,j*h:(j+1)*h,:] = fimg*mask_fill.stack().tolist()[i*iterax+j]
            
    # final_heat = final_heat//final_heat.max()
#    plt.figure()
#    plt.imshow(final_heat/255)
    
        
#    plt.figure()
#    plt.imshow(test)
    

    return final_heat





def output_cropped_interest_area(final_heat):
        
    final_test = np.uint8(final_heat)
    
    gray = cv2.cvtColor(final_test,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    
    crop = final_test[y:y+h,x:x+w]
    
#    plt.figure()
#    plt.imshow(crop)

    return crop


import PIL
import math



def break_image1(input_frames,shape):
  
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
            mask = np.zeros((imgheight,imgwidth,3))
            mask[i:i+height,j:j+width,:]=1

            
    return test_image,division



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
    masked = []
    
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            test_image.append(a)
            mask = np.zeros((imgheight,imgwidth,3))
            mask[i:i+height,j:j+width,:]=1
            masked.append(mask)
            
    return test_image,division,masked

#in the break image code need to add part to get the mask for the location of the breaking 
#place where the box is is 1 otherwhise 0


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
#    plt.figure()
#    plt.imshow(final_heath/255)
        
#    plt.figure()
#    plt.imshow(test)

    return final_heat

from PIL import ImageDraw,Image
 
    
def buildup1(final,masked2,cropped):
        
    buraco = []
    for f,m in zip(final,masked2):
            
           #generate a patch to put the masks times the blend image
        patch = np.zeros((cropped.shape[0],cropped.shape[1],cropped.shape[2]))  
        
        #generate a image with all the mask holes
        if not isinstance(f,int):
            continue
        else:
            buraco.append(m)
        
    buraco2 = sum(buraco)
    
    patch = np.zeros((cropped.shape[0],cropped.shape[1],cropped.shape[2]))  
    
    for f,m in zip(final,masked2):
        if not isinstance(f,int):
    
            #find the mask first position
            xx = [np.min(np.where(m==1)[0]),np.max(np.where(m==1)[0])]
            yy = [np.min(np.where(m==1)[1]),np.max(np.where(m==1)[1])]
    
    #        buraco2[xx[0]:xx[1]+1,yy[0]:yy[1]+1,:]=fscale/255.0
            patch[xx[0]:xx[1]+1,yy[0]:yy[1]+1,:]=f[0:abs(xx[0]-xx[1]-1),0:abs(yy[0]-yy[1]-1),:]
    
    
    croppedclass = buraco2*cropped/255.0+patch
    
    return croppedclass


def buildup2(frame,final_heat,croppedclass):
            
    i_img = frame
    
    buraco = final_heat/255.0
    
    image_hole = np.array([1. if a_ >0 else 0. for a_ in buraco.ravel()]).reshape(buraco.shape[0], buraco.shape[1], buraco.shape[2])
    
    xx = [np.min(np.where(image_hole==1)[0]),np.max(np.where(image_hole==1)[0])]
    yy = [np.min(np.where(image_hole==1)[1]),np.max(np.where(image_hole==1)[1])]
    
    image_hole[xx[0]:xx[1]+1,yy[0]:yy[1]+1,:] = 1.
    
    patch = np.zeros((i_img.shape[0],i_img.shape[1],i_img.shape[2]))  
    
    
    #find the mask first position
    xx = [np.min(np.where(image_hole==1)[0]),np.max(np.where(image_hole==1)[0])]
    yy = [np.min(np.where(image_hole==1)[1]),np.max(np.where(image_hole==1)[1])]
    
    #        buraco2[xx[0]:xx[1]+1,yy[0]:yy[1]+1,:]=fscale/255.0
    patch[xx[0]:xx[1]+1,yy[0]:yy[1]+1,:]=croppedclass
    
    f_img = (-1*(image_hole[0:i_img.shape[0],0:i_img.shape[1],:]-1)*i_img)/255.0
        
    croppedimage = patch+f_img
 
    return croppedimage

 
#subsection2(test_image2,sers_identifier,'sers_notsers',image_identifier)


def subsection2(test_image2,label,image_identifier):
    
    finalimg = []
    model = Sequential()
    for layer in image_identifier.layers:
        model.add(layer)
    
    for layer in model.layers:
        layer.trainable = False
    
    
#    labels = ['No Enhancement', 'SERS']
    
    for f in test_image2:
        sectionit = []
        mask = []
        size =  list(image_identifier.layers[0].layers[0].output.shape[1:3])
        img = f.convert('RGB').resize((size[0] ,size[1]), Image.ANTIALIAS)
                
        open_cv_image = np.array(img) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        img_gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
                
        im_gauss = cv2.GaussianBlur(img_gray , (5, 5), 0)
        ret, thresh = cv2.threshold(im_gauss, 127, 255, 0)
        # get contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contours_cirles = []
        contours_area = []
        # calculate area and filter into new array
        comp = size[0]//10
        for con in contours:
            if any(con.ravel()<comp) or any(con.ravel()>size[0]-comp):
                continue
            area = cv2.contourArea(con)
#            print(area)
            if 5 < area < 1000:
                contours_area.append(con)
        
        # check if contour is of circular shape
        for con in contours_area:
            perimeter = cv2.arcLength(con, True)
            area = cv2.contourArea(con)
            if perimeter == 0:
                break
            circularity = 4*math.pi*(area/(perimeter*perimeter))
#            print(circularity)
            if 0.7 < circularity < 1.2:
                contours_cirles.append(con)
        
        
        for con in contours_cirles:
    #        cv.drawContours(open_cv_image,con,-1,(0,255,0),1) 
            xm = con.ravel()[::2].min()-comp
            xM = con.ravel()[::2].max()+comp
            ym = con.ravel()[1::2].min()-comp
            yM = con.ravel()[1::2].max()+comp
            
            sectionit.append(open_cv_image[ym:yM,xm:xM,:])
            zeromask = np.zeros((img_gray.shape[0],img_gray.shape[1],3))
            zeromask[ym:yM,xm:xM,:]=np.ones((abs(ym-yM),abs(xm-xM),3))
            mask.append(zeromask)
         
        if sectionit == []:
            finalimg.append(0)
        else:
                
#            fig, axs = plt.subplots(int(np.sqrt(len(sectionit)))+1,int(np.sqrt(len(sectionit)))+1)
#            fig2, ax = plt.subplots()
#            ax.imshow(f)
#            axes = axs.ravel()
            
            heatmaps = []
            #get the activations from the test image broken in pieces
            location = []
            
            
            for cnt,sel in enumerate(sectionit):
#                axes[cnt].imshow(sel)
                
                
                
                sel2 = cv2.resize(sel,(model.layers[0].layers[0].output.shape[1]
                               ,model.layers[0].layers[0].output.shape[2]),
                                interpolation = cv2.INTER_AREA)
            
            
                x = image.img_to_array(sel2)
                x = np.expand_dims(x, axis=0)
        
                
                preds = model.predict(x)
            
                num = np.argmax(preds)
                
#                axes[cnt].imshow(sel2)
#                axes[cnt].text(0,0,labels[num])
                
                if preds[0][1]>0.5:
                    location.append(1)
                    
                    img_output = model.layers[0].layers[-2].output[:,num]
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
                    
                    heatmap = np.array([0 if a_ < 0.8 else a_ for a_ in heatmap.ravel()]).reshape(heatmap.shape[0],
                                                                                        heatmap.shape[1])
                    
                    #blend the heatmap with the image but ignore values below 0
                    
                    img = sel2
                    
                    heatmap = cv2.resize(heatmap, (img.shape[1],img.shape[0]))
                    marker = heatmap
                    
                    heatmap[heatmap!=0] = heatmap[heatmap!=0]*0.2+0.8
                    
                    #preform a proper scalling for the heatmap colour
                    
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    
                    
                    if img.max()==0:
                        img *= np.uint8(255.0/(0.000001+(img.max())))
                    else:
                        img *= np.uint8(255.0/img.max())
                    
                    blend = np.zeros((img.shape[0],img.shape[1],3))
                    
                    for n in range(blend.shape[0]):
                        for k in range(blend.shape[1]):
                            if marker[n,k]!=0:
                                blend[n,k,:] = np.transpose(cv2.addWeighted(img[n,k,:],0.5,heatmap[n,k,:],0.5,0.0))[0]
                            else:
                                blend[n,k,:] = img[n,k,:]
                                
                    blend = np.uint8(blend)
                    
                    heatmaps.append(blend)
                    
#                    axes[cnt].imshow(blend)
                    
                else:
                    location.append(0)
                    heatmaps.append(0)
        
        
            #generate a patch to put the masks times the blend image
            patch = np.zeros((mask[0].shape[0],mask[0].shape[1],mask[0].shape[2]))  
            
            #generate a image with all the mask holes
            image_hole = np.array([0 if a_ < 1 else 1 for a_ in sum(mask).ravel()]).reshape(mask[0].shape[0],
                                                                                            mask[0].shape[1],
                                                                                            mask[0].shape[2])        
            for m,h,z in zip(mask,heatmaps,sectionit):
                if not isinstance(h,int):
                    #resize the heatmap to its real heatmap size
                    hscale = cv2.resize(h,(z.shape[1],z.shape[0]))
    
                    #find the mask first position
                    xx = [np.min(np.where(m==1)[0]),np.max(np.where(m==1)[0])]
                    yy = [np.min(np.where(m==1)[1]),np.max(np.where(m==1)[1])]
    
                    patch[xx[0]:xx[1]+1,yy[0]:yy[1]+1,:]=hscale/255.0
                else:
                    xx = [np.min(np.where(m==1)[0]),np.max(np.where(m==1)[0])]
                    yy = [np.min(np.where(m==1)[1]),np.max(np.where(m==1)[1])]
    
                    patch[xx[0]:xx[1]+1,yy[0]:yy[1]+1,:]=z/255.0
            
            #final image is the initial image opened with all the holes from the masks and added patches with the sers
            #non sers model
            final_img = open_cv_image*(-1*(image_hole-1)/255.0)+patch
            
#            plt.figure()
#            plt.imshow(final_img)
            finalimg.append(final_img)
                            
                        #fill the hole with the blend
    return finalimg
    

import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage.color import rgb2gray



from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import cv2
import os.path
from tqdm import tqdm
import matplotlib.pyplot as plt 


import numpy as np
import tensorflow as tf
from tensorflow import keras

#FINAL MANUAL CROPPED
    
video_path = "pillar image irl.mkv"

    
    
# MIX MIX MIX MIX this one plot heatmaps correctly with correct crop
    

   
counts = [26,29,53,60+14,60+42,60*2+7,60*2+30,60*2+58,60*3+27]
counts = [a*60 for a in counts]


import cv2

cap = cv2.VideoCapture(video_path)
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        for i,count in enumerate(counts):
            print(count)
            print(i)
            
            if i == len(counts)-1:
                cap.release()
                break
            
            try:
                ret, frame = cap.read()
                cap.set(1, count)
                
                final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                loculx = 64
                loculy = 230
                locdrx = 818
                locdry = 712
              
                factorx = final_frame.shape[1]/loculx 
                factorxx = final_frame.shape[1]/locdrx
                
                factory = final_frame.shape[0]/loculy
                factoryy = final_frame.shape[0]/locdry
                
                y = int(final_frame.shape[1]/factorx)
                yy = int(final_frame.shape[1]/factorxx)
                x = int(final_frame.shape[0]/factory)
                xx = int(final_frame.shape[0]/factoryy)
                
                cropped = final_frame[x:xx,y:yy,:]
                
	
                size2 = [50,50]
	            
	            # newcropped = cropped[:,1000:,:]
	            
                test_image2,division2,masked2 = break_image2(cropped,size2)
	            
                sers_identifier = load_model('SERS_NOSERS_pillars_v08.h5')
                
                #previous version wasnt using the sers not sers model, but the pillar not pillar model...
                
                final = subsection2(test_image2,'sers_notsers',sers_identifier)
	            
                croppedclass = buildup1(final,masked2,cropped)  
                
                f_frame = np.zeros((final_frame.shape[0],final_frame.shape[1],final_frame.shape[2]))
                f_frame[x:xx,y:yy,:]=1
	            
                croppedimage =  buildup2(final_frame,f_frame,croppedclass)
                plt.figure()
                plt.imshow(croppedimage)
                plt.axis('off')
                plt.savefig(r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\MEng student\3rd year\report\code\files to transfer\final images\\'+'final_image_v01_'+str(count)+'.png', dpi = 300,bbox_inches="tight")
                plt.close()

            except:
                pass
    else:
        cap.release()
        break



    