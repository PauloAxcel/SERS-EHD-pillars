import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import os


def break_img(image,cut):
    
    image = cv2.imread(image)
    tiles = []
    x,y,c = image.shape
    
    matrix = np.zeros((x+cut-x%cut,y+cut-y%cut,3)).astype(np.uint8)
    
    a = (cut-x%cut)//2
    b = (cut-y%cut)//2
    
    matrix[a:x+a,b:y+b,:] = image.astype(np.uint8)
    matrix.astype(np.uint8)

    for i in range(matrix.shape[0]//cut):
        for j in range(matrix.shape[1]//cut):
            tiles.append(matrix[i*cut:(i+1)*cut,j*cut:(j+1)*cut,:])
    
    return tiles
    


def plot_tiles(image,tiles):
    
    im = cv2.imread(image)
    
    plt.figure()
    plt.imshow(im)
    for t in tiles:
        plt.figure()
        plt.imshow(t)



direc = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\MEng student\3rd year\DATA\rama image test'
image = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\MEng student\3rd year\DATA\rama image test\test part1 image.png'
image_t = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\MEng student\3rd year\DATA\rama image test\part1 image.png'


cut = 500
tiles = break_img(image,cut)
tiles2 = break_img(image_t,cut)

plot_tiles(image,tiles)

tiles_num = []
for t,t2 in zip(tiles,tiles2):
    img_hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)

    ## Gen lower mask (0-5) and upper mask (175-180) of RED
    mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))


    ## Merge the mask and crop the red regions
    mask = cv2.bitwise_or(mask1, mask2 )
    croped = cv2.bitwise_and(t, t, mask=mask)

    ## Display
    plt.figure()
    plt.imshow(mask)
    if (mask>0.5).any():
        tiles_num.append([mask,t2])
#    plt.figure()
#    plt.imshow(croped)
#    cv2.waitKey()

crop = tiles_num[3][0]
crop2 = tiles_num[3][1]

image = crop2

image = imutils.resize(image, height=500)

plt.figure()
plt.imshow(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 0, 50, 255)




# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
thresh = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

plt.figure()
plt.imshow(thresh)

#
#fig, ax = plt.subplots(1,1, figsize=(9, 9/1.618))
#
#blobs_log = blob_doh(thresh, max_sigma=50, threshold=.1,overlap=1)
#
## Compute radii in the 3rd column.
#blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
#
#
##        ax[idx].imshow(ima)
##        ax.imshow(img_g)
##kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
##im = cv2.filter2D(gray, -1, kernel)
#ax.imshow(gray)
#
#for blob in blobs_log:
#    y, x, r = blob
#    c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
#    ax.add_patch(c)
#    ax.set_axis_off()
##            ax[idx].add_patch(c)
##        ax[idx].set_axis_off()
#
#plt.tight_layout()
#plt.show()





thr1 = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]





from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



boxes = pytesseract.image_to_boxes(thr1, config="--psm 6 digits")
data = pytesseract.image_to_string(thr1, config="--psm 6 digits")


h, w,_ = crop2.shape


for b in boxes.splitlines():
    b = b.split()
    cv2.rectangle(crop2, ((int(b[1]), h - int(b[2]))), ((int(b[3]), h - int(b[4]))), (0, 255, 0), 2)

plt.imshow(crop2)







#
#from skimage import feature 
#
#
#def find_blobs(tiles):
#
##    fig, axes = plt.subplots(round(np.sqrt(len(tiles))), round(np.sqrt(len(tiles)))+1, figsize=(9, 9/1.618))
#    fig, ax = plt.subplots(1,1, figsize=(9, 9/1.618))
##    ax = axes.ravel()
#    
#    for idx,t in enumerate(tiles[0:1]):
#        ima = t
#        img_g = rgb2gray(ima)
#        
##        edge = feature.canny(img_g, sigma=1.5)
##        plt.imshow(edge)
#    
#        blobs_log = blob_log(edge, max_sigma=10, num_sigma=10, threshold=0.2)
#    
#        # Compute radii in the 3rd column.
#        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
#        
#        
##        ax[idx].imshow(ima)
##        ax.imshow(img_g)
#        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#        im = cv2.filter2D(img_g, -1, kernel)
#        ax.imshow(im)
#        
#        for blob in blobs_log:
#            y, x, r = blob
#            c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
#            ax.add_patch(c)
#        ax.set_axis_off()
##            ax[idx].add_patch(c)
##        ax[idx].set_axis_off()
#    
#        plt.tight_layout()
#        plt.show()
#



#blobs_dog = blob_dog(img_g, max_sigma=30, threshold=.1)
#blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
#
#blobs_doh = blob_doh(img_g, max_sigma=30, threshold=.01)

#blobs_list = [blobs_log, blobs_dog, blobs_doh]
#colors = ['yellow', 'lime', 'red']
#titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
#          'Determinant of Hessian']
#sequence = zip(blobs_list, colors, titles)






#
#
#
#def draw_circles(img, circles):
#    # img = cv2.imread(img,0)
#    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#    for i in circles[0,:]:
#    # draw the outer circle
#        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#        # draw the center of the circle
#        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
#        cv2.putText(cimg,str(i[0])+str(',')+str(i[1]), (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
#    return cimg
#
#def detect_circles(image_path):
#    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#    gray_blur = cv2.medianBlur(gray, 13)  # Remove noise before laplacian
#    gray_lap = cv2.Laplacian(gray_blur, cv2.CV_8UC1, ksize=5)
#    dilate_lap = cv2.dilate(gray_lap, (3, 3))  # Fill in gaps from blurring. This helps to detect circles with broken edges.
#    # Furture remove noise introduced by laplacian. This removes false pos in space between the two groups of circles.
#    lap_blur = cv2.bilateralFilter(dilate_lap, 5, 9, 9)
#    # Fix the resolution to 16. This helps it find more circles. Also, set distance between circles to 55 by measuring dist in image.
#    # Minimum radius and max radius are also set by examining the image.
#    circles = cv2.HoughCircles(lap_blur, cv2.HOUGH_GRADIENT, 16, 55, param2=450, minRadius=20, maxRadius=40)
#    cimg = draw_circles(gray, circles)
#    print("{} circles detected.".format(circles[0].shape[0]))
#    # There are some false positives left in the regions containing the numbers.
#    # They can be filtered out based on their y-coordinates if your images are aligned to a canonical axis.
#    # I'll leave that to you.
#    return cimg
#
#
#
#
#
#
#
#
#cimg = detect_circles(image)
#
#


































