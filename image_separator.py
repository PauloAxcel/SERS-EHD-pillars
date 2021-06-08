
import pandas as pd




image_dir = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\MEng student\3rd year\DATA\IMAGES\FIG1\PART1'
file = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\MEng student\3rd year\DATA\raman fit test'

df = pd.read_csv(file+'\keysample1_par_filtered.csv')

values = df['SERS']
points = df['map point']



import glob
image_files = (glob.glob(image_dir+'\\*.png')) + glob.glob(image_dir+'\\*.jpg')


image_points = []

for f in image_files:
    image_points.append(int(f.split('\\')[-1].split('.')[0]))






SERS_dir = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\MEng student\3rd year\DATA\raman folder separator\SERS'
NOSERS_dir = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\MEng student\3rd year\DATA\raman folder separator\NOSERS'



import shutil  


for imp,file in zip(image_points,image_files):
    for val,poi in zip(list(values.values),list(points.values)):
        if imp==poi:
            if val == 0:
                end_fol = NOSERS_dir
            else:
                end_fol = SERS_dir
            shutil.copy(file, end_fol)  
            
        
    

















































