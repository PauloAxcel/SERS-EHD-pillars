#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:15:20 2021

@author: alex
# """

import cv2
import os.path
from tqdm import tqdm
import matplotlib.pyplot as plt 
file_path = r"/Users/alex/Documents/Chem Eng/Masters Project  /vid dir/"
path_dir = os.listdir(file_path)   #Return the file name in the folder
save_path = r"/Users/alex/Documents/Chem Eng/Masters Project  /pics/"
count=1
name_count=1
for allDir in tqdm(path_dir):
    video_path = file_path+allDir
    video = cv2.VideoCapture(video_path)  # Read in video files
    if video.isOpened():  # Determine whether it opens normally
        rval, frame = video.read()
    else:
        rval = False
 
    timeF = 60 # Video frame count interval frequency
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)
    
    break 
    # while rval:  # Cycle through video frames
    #     rval, frame = video.read()
    #     if (count % timeF == 0):  # Store operation every timeF frame
    #         # cv2.imshow('pic',frame)
    #         cv2.imwrite(save_path + str(name_count) + '.jpg', frame)  # imwrite cannot save Chinese path in py3
    #         # cv2.imencode('.jpg', frame)[1].tofile(save_path + str(count) +'.jpg') # Save as image
    #         # print('E:\Dataset\file\Data\image/' +'%06d'% c +'.jpg')
    #         name_count=name_count+1
    #     count = count + 1
    #     cv2.waitKey(1)

video.release()

# import cv2
# import os
# import time

# # Read the video from specified path
# cam = cv2.VideoCapture('r/Users/alex/Documents/Chem Eng/Masters Project  /2021-03-26 09-42-03.mkv')
# # another video example to check 
# # cam = cv2.VideoCapture('r/Users/alex/Library/Mobile Documents/com~apple~CloudDocs/Downloads/PPS spring ball.mp4')
# try:

#     # creating a folder named data2
#     if not os.path.exists('data2'):
#         os.makedirs('data2')

#     # if not created then raise error
# except OSError:
#     print('Error: Creating directory of data')

# # # frame
# currentframe = 1
# # findning frames per second 
# frame_per_second = cam.get(cv2.CAP_PROP_FPS) 

# print(frame_per_second)
# #number of frame to screenshot 
# step = 1

# while (True):
#     # reading from frame
#     ret, frame = cam.read()

#     if ret:
#         if currentframe > (step*frame_per_second):  
#             currentframe = 0
#             # if video is still left continue creating images
#             name = './data2/frame' + str(currentframe) + '.jpg'
#             print('Creating...' + name)
    
#             # writing the extracted images
#             cv2.imwrite(name, frame)
    
#             # increasing counter so that it will
#             # show how many frames are created
#             currentframe += 1
#     else:
#         break

# # Release all space and windows once done
# cam.release()

# cv2.destroyAllWindows()
