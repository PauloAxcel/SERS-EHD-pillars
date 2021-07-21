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

notpillars_dir = r'/home/newuser/Desktop/alex/cnn pillars/notpillars/'
pillars_dir = r'/home/newuser/Desktop/alex/cnn pillars/pillars/'

gen_dir_tra = r'/home/newuser/Desktop/alex/cnn pillars/Training/'
gen_dir_val = r'/home/newuser/Desktop/alex/cnn pillars/Validation/'

#num is the total number of samples that we want to generate.
num = 20000

#get the files inside the folders

notpillars_train = os.listdir(notpillars_dir)
pillars_train = os.listdir(pillars_dir)

all_dir = [pillars_dir, notpillars_dir]
all_data = [pillars_train, notpillars_train]

for dire,file in zip(all_dir, all_data):

    folder = dire.split('/')[-2]+'/'

    for j in range(num):   

        #generate a rand to select a random file in the folder
        
    #        rand = random.randint(0,len(all_data[i])-1)
        rand = random.randint(0,len(file)-1)
     
            
        
    #        im = cv2.imread(all_dir[i]+all_data[i][rand])
        im = cv2.imread(dire+file[rand])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
        
    #        plt.imshow(im)
        
        #datagen.flow needs a rank 4 matrix, hence we use np.expand_dims to increase the dimention of the image
        
        image = np.expand_dims(im,0)
    #        word_label = all_data[i][rand].split('.')[0]
        word_label = file[rand].split('.')[0]
        
        #Generate new image process
        
        datagen = ImageDataGenerator(width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  featurewise_center=0,
                                  samplewise_center=0,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')
        
        #label files based on the train/validation by employing a rand function
        
        lab = dire.split('/')[-2]
        
        if random.random() < 0.8:
            aug_iter = datagen.flow(image,save_to_dir = gen_dir_tra+folder , save_prefix = lab+'_train_' + word_label +'_gen_' + str(random.randint(0,num)))
        else:
            aug_iter = datagen.flow(image,save_to_dir = gen_dir_val+folder ,save_prefix = lab+'_val_' + word_label +'_gen_' + str(random.randint(0,num)))
        
        #next function produces the result from the datagen flow. collapses the function.
    
    #        plt.imshow(next(aug_iter)[0].astype(np.uint8))
        
        aug_images = [next(aug_iter)[0]/255.0 for m in range(1)]
    
    
    









