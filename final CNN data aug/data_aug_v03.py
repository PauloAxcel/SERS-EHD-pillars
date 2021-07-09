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


#get the folders
SERS_train_dir = r'/home/newuser/Desktop/emily try/Data2(part1+part3+part4+part6)/Original Data/SERS/'
NOENH_train_dir = r'/home/newuser/Desktop/emily try/Data2(part1+part3+part4+part6)/Original Data/nonSERS/'

gen_dir_tra = r'/home/newuser/Desktop/emily try/Data2(part1+part3+part4+part6)/Training/'
gen_dir_val = r'/home/newuser/Desktop/emily try/Data2(part1+part3+part4+part6)/Validation/'

#num is the total number of samples that we want to generate.
num = 5000

#get the files inside the folders
SERS_train = os.listdir(SERS_train_dir)
NOENH_train = os.listdir(NOENH_train_dir)

all_dir = [SERS_train_dir, NOENH_train_dir]
all_data = [SERS_train, NOENH_train]

for dire,file in zip(all_dir, all_data):
    

#for i in range(len(all_data)):
    for j in range(num):   
        

        
        #generate a rand to select a random file in the folder
        
#        rand = random.randint(0,len(all_data[i])-1)
        rand = random.randint(0,len(file)-1)
        
        if len(file[rand].split('_'))>1:
            continue
        else:
                
            
    #        im = cv2.imread(all_dir[i]+all_data[i][rand])
            im = cv2.imread(dire+file[rand])
            
    #        plt.imshow(im)
            
            #datagen.flow needs a rank 4 matrix, hence we use np.expand_dims to increase the dimention of the image
            
            image = np.expand_dims(im,0)
    #        word_label = all_data[i][rand].split('.')[0]
            word_label = file[rand].split('.')[0]
            
            #Generate new image process
            
            datagen = ImageDataGenerator(featurewise_center=0,
                                         samplewise_center=0,
                                         rotation_range=180,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         fill_mode='nearest')
            
    
            #label files based on the train/validation by employing a rand function
            
            lab = dire.split('/')[-2]
            
            if random.random() < 0.8:
                aug_iter = datagen.flow(image,save_to_dir = gen_dir_tra , save_prefix = lab+'_train_' + word_label +'_gen_' + str(random.randint(0,num)))
            else:
                aug_iter = datagen.flow(image,save_to_dir = gen_dir_val ,save_prefix = lab+'_val_' + word_label +'_gen_' + str(random.randint(0,num)))
            
            #next function produces the result from the datagen flow. collapses the function.
    
    #        plt.imshow(next(aug_iter)[0].astype(np.uint8))
            
            aug_images = [next(aug_iter)[0].astype(np.uint8) for m in range(1)]
    
        
        









