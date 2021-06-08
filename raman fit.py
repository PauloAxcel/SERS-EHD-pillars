import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Gets rid of some warnings in output text
import warnings
warnings.filterwarnings("ignore")
import scipy
from scipy.linalg import solveh_banded
from scipy import signal
from sklearn.decomposition import PCA
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from scipy import stats
import nestle 
import sys
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns			
import random		
from random import randint
from sklearn.metrics.pairwise import euclidean_distances
#import statsmodels.stats.api as sms     # t-test
import statistics as stat               # F-test
from scipy import stats                 # KS Test
from scipy.stats import f               # F-test
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
import scipy
from scipy.linalg import solveh_banded
from scipy import signal
import matplotlib.pyplot as plt
#%matplotlib qt5
import matplotlib.pyplot as plt 

from minisom import MiniSom   
from sklearn import preprocessing
import itertools
from scipy.interpolate import make_interp_spline, BSpline

from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from itertools import groupby

import itertools
import circlify as circ
import math
import math
from scipy import ndimage
import matplotlib.patches as mpatches  
import matplotlib.gridspec as gridspec
from itertools import chain
from sklearn.preprocessing import minmax_scale, scale

bin_colours = ['r','orange',
              'g','lime',
              'b','cyan',
              'gold','yellow',
              'indigo','violet',
              'pink','magenta',
              'maroon','chocolate',
              'grey','darkkhaki']
 
font = {'family' : 'Arial',
        'size'   : 22}

plt.rc('font', **font)

# From MatrixExp
def matrix_exp_eigen(U, s, t, x):
    exp_diag = np.diag(np.exp(s * t), 0)
    return U.dot(exp_diag.dot(U.transpose().dot(x))) 

# From LineLaplacianBuilder
def get_line_laplacian_eigen(n):
    assert n > 1
    eigen_vectors = np.zeros([n, n])
    eigen_values = np.zeros([n])

    for j in range(1, n + 1):
        theta = np.pi * (j - 1) / (2 * n)
        sin = np.sin(theta)
        eigen_values[j - 1] = 4 * sin * sin
        if j == 0:
            sqrt = 1 / np.sqrt(n)
            for i in range(1, n + 1):
                eigen_vectors[i - 1, j - 1] = sqrt
        else:
            for i in range(1, n + 1):
                theta = (np.pi * (i - 0.5) * (j - 1)) / n
                math_sqrt = np.sqrt(2.0 / n)
                eigen_vectors[i - 1, j - 1] = math_sqrt * np.cos(theta)
    return eigen_vectors, eigen_values

def smooth2(t, signal):
    dim = signal.shape[0]
    U, s = get_line_laplacian_eigen(dim)
    return matrix_exp_eigen(U, -s, t, signal)


def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e6,
                 max_iters=10, conv_thresh=1e-5, verbose=False):
  '''Computes the asymmetric least squares baseline.
  * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
  smoothness_param: Relative importance of smoothness of the predicted response.
  asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                       Setting p=1 is effectively a hinge loss.
  '''
  smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
  # Rename p for concision.
  p = asymmetry_param
  # Initialize weights.
  w = np.ones(intensities.shape[0])
  for i in range(max_iters):
    z = smoother.smooth(w)
    mask = intensities > z
    new_w = p*mask + (1-p)*(~mask)
    conv = np.linalg.norm(new_w - w)
    if verbose:
      print (i+1, conv)
    if conv < conv_thresh:
      break
    w = new_w
  else:
    print ('ALS did not converge in %d iterations' % max_iters)
  return z


class WhittakerSmoother(object):
  def __init__(self, signal, smoothness_param, deriv_order=1):
    self.y = signal
    assert deriv_order > 0, 'deriv_order must be an int > 0'
    # Compute the fixed derivative of identity (D).
    d = np.zeros(deriv_order*2 + 1, dtype=int)
    d[deriv_order] = 1
    d = np.diff(d, n=deriv_order)
    n = self.y.shape[0]
    k = len(d)
    s = float(smoothness_param)

    # Here be dragons: essentially we're faking a big banded matrix D,
    # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
    diag_sums = np.vstack([
        np.pad(s*np.cumsum(d[-i:]*d[:i]), ((k-i,0),), 'constant')
        for i in range(1, k+1)])
    upper_bands = np.tile(diag_sums[:,-1:], n)
    upper_bands[:,:k] = diag_sums
    for i,ds in enumerate(diag_sums):
      upper_bands[i,-i-1:] = ds[::-1][:i+1]
    self.upper_bands = upper_bands

  def smooth(self, w):
    foo = self.upper_bands.copy()
    foo[-1] += w  # last row is the diagonal
    return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)




def ImportData(inputFilePath):
#    print(inputFilePath)
    
    df = pd.read_csv(inputFilePath,   
                          sep = '\s+',
                          header = None)
    if df.shape[1] == 2:
    
        df = pd.read_csv(inputFilePath, 
                              sep = '\s+',
                              header = None, 
                              skiprows = 1,
                              names = ['Wavenumber', 'Intensity'])
        n_points = sum((df.iloc[0,1]==df.iloc[:,1][df.iloc[0,0]==df.iloc[:,0]]))
        n_raman = df.shape[0]//n_points
        wavenumber = df['Wavenumber']
            
        dataset = []
        label = inputFilePath.split('/')[-1][:11] 
        intensity = df['Intensity']
        flag = 0
        
    else:        
        
        df = pd.read_csv(inputFilePath, 
                              sep = '\s+',
                              header = None, 
                              skiprows = 1,
                              names = ['X','Y','Wavenumber', 'Intensity'])
        n_points = sum((df.iloc[0,1]==df.iloc[:,1][df.iloc[0,0]==df.iloc[:,0]]))
        n_raman = df.shape[0]//n_points
        wavenumber = df['Wavenumber'][0:n_points]
        
        
        dataset = []
#        label = inputFilePath.split('/')[-1][:11]
        
#        works for testing
        label = inputFilePath.split('\\')[-1][:11]

            
        for i in range(df.shape[0]//n_points):
            ind_df = df['Intensity'][i*n_points:n_points*(i+1)].reset_index(drop=True)
            ind_df = ind_df.rename('spectrum '+str(i+1))
            dataset.append(ind_df)
    
        intensity = pd.concat(dataset,axis=1)
#        print(intensity.shape)
        
#        if zscore == 0:
#            pass
#        else:
#            out = np.abs(stats.zscore(intensity.T))
#            
#            if inv == 0:
#                intensity = intensity.T[(out < zscore).all(axis=1)].T
##                print(intensity.shape)
#            else:
#                intensity = intensity.T[(out > zscore).all(axis=1)].T       
##                print(intensity.shape)
        
        if intensity.shape[1] == 1:
            flag = 0
        else:
            flag = 1
        
    return (wavenumber, intensity, n_points, n_raman, label,flag)





def IndividualPlot(x,y,label):
    if pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    plt.figure(figsize=(9,9/1.618))
    plt.plot(x, y_avg,color='k',label=label,lw=2)    
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel(' Intensity (a.u.)')
    plt.legend(loc='best', prop={'size':12},frameon=False)





def _1Lorentzian(x, amp, cen, wid):
    return amp*wid**2/((x-cen)**2+wid**2)



def find_local_max(signal):
    dim = signal.shape[0]
    ans = np.zeros(dim)
    for i in range(dim):
        dxf = signal[i - 1] - signal[i]
        dxb = signal[i - 1] - signal[i - 2]
        ans[i] = 1.0 if dxf > 0 and dxb > 0 else 0.0
    return ans > 0.5


def Baseline(y,asymmetry_param, smoothness_param, max_iters, conv_thresh):
    if pd.DataFrame(y).shape[1]==1 and isinstance(y, pd.DataFrame):
        y_avg = y.iloc[:,0]
    elif pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    return y_avg - als_baseline(y_avg ,asymmetry_param, smoothness_param, max_iters, conv_thresh)



def Normalize(y,zscore,inv):  
    if zscore == 0:
        pass
    else:
        out = np.abs(stats.zscore(y.T)) 
        
        if inv == 0:
            y = y.T[(out < zscore).all(axis=1)].T
    #                print(intensity.shape)
        else: 
            y = y.T[np.array([not c for c in (out < zscore).all(axis=1)])].T       
    #                print(intensity.shape)
    if pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    return ((y_avg-y_avg.min())/(y_avg.max()-y_avg.min()))
 

def fit_region(wavenumber,pos,tol = 10):
    min_l = min(wavenumber.index.tolist())
    max_l = max(wavenumber.index.tolist())
    index = wavenumber.index[wavenumber==pos].tolist()[0]
    index1 = index-tol
    index2 = index+tol
    if index1<min_l:
        index1 = 0
    if index2>max_l:
        index2 = max_l
        
    return wavenumber[index1:index2]

from scipy.integrate import simps
from scipy.stats import chisquare

def peakfinder(files):
    zscore = 0
    inv = 0
    smooth=7
    asymmetry_param = 0.05
    smoothness_param = 1000000
    max_iters = 10
    conv_thresh =0.00001

    



    wavenumber, intensity, n_points, n_raman, label,flag = ImportData(files)
    
    for i in range(intensity.shape[1]):
        
        peak_vector = []
        max_value = []
        spectra = []
        
        intensity_b = Baseline(intensity.iloc[:,i],asymmetry_param, smoothness_param, max_iters, conv_thresh)
        max_value.append(intensity_b.max())
    
        spectra.append(Normalize(intensity_b,zscore,inv))
    
#        smoothed_signal = Normalize(intensity_b,zscore,inv)
#        smoothed_signal = Normalize(smoothed_signal,zscore,inv)
    
        smoothed_signal = smooth2(smooth,  intensity_b)
        smoothed_signal = Normalize(smoothed_signal,zscore,inv)
       
        local_max_index = find_local_max(smoothed_signal)
    
        stem = np.zeros(wavenumber.shape)
        stem[local_max_index] = 1  
        
        lorentz = []

        fig = plt.figure(figsize=(9,9/1.618))
        plt.close(fig)
        plt.ioff()
        plt.plot(wavenumber,smoothed_signal)
        
        for count,wav in enumerate(wavenumber[local_max_index]):
    
            reg = fit_region(wavenumber,wav,10)
            
            try:
                popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                                reg, 
                                                                smoothed_signal[reg.index],
                                                                p0=[smoothed_signal[reg.index].max(), wav,  2/(np.pi*smoothed_signal[reg.index].max())])
                
                perr_1lorentz = np.sqrt(np.diag(pcov_1lorentz))
        
                pars_1 = popt_1lorentz[0:3]
                
                if (abs(pars_1[-1])>100) | (pars_1[0]<0):
                    pass
                else:
                    
                    lorentz_peak_1 = _1Lorentzian(reg, *pars_1)
        
                    lorentz.append(_1Lorentzian(wavenumber,*popt_1lorentz))
                    peak_vector.append([i+1,
                                        'Curve '+str(count),
                                        pars_1[1],
                                        pars_1[2],
                                        pars_1[0],
                                        perr_1lorentz[0],
                                        np.sqrt(simps(lorentz_peak_1[::-1].values , x=np.array(sorted(reg)))),
                                        simps(lorentz_peak_1[::-1].values , x=np.array(sorted(reg))),
                                        chisquare(lorentz_peak_1.values, f_exp=smoothed_signal[reg.index])[0]])
                    
                    plt.fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),  _1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
            
            except:
                pass
            
            
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        
        plt.savefig(r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\MEng student\3rd year\DATA\raman fit test\raman spec'+'\\'+(f"{i+1:04d}")+'.png',bbox_inches='tight')
        plt.close()
            
        peak = pd.DataFrame(peak_vector,columns=['map point','curve name','center','width','height','err','err_a','area','chisq'] )
        
        spectra.append(wavenumber)  
        
        spectra = pd.DataFrame(spectra).T 
    #    print(spectra)
        spectra.columns = peak['map point'].unique().tolist()+['wavenumber']
        
        new_peak = []
        new_err = []
        new_area = []
        new_err_a = []
        
        for j in range(spectra.shape[1]-1):
            spectra.iloc[:,j] = spectra.iloc[:,j]*max_value[j]
            index = spectra.columns[j] == peak['map point']
            new_peak.append(peak['height'].loc[index]*max_value[j])
            new_err.append(peak['err'].loc[index]*max_value[j])
            new_area.append(peak['area'].loc[index]*max_value[j])
            new_err_a.append(np.sqrt(peak['err_a'].loc[index]*max_value[j]))
            
        peak['height'] = pd.DataFrame(np.concatenate([np.stack(a) for a in new_peak]))
        peak['err'] = pd.DataFrame(np.concatenate([np.stack(a) for a in new_err]))
        peak['area'] = pd.DataFrame(np.concatenate([np.stack(a) for a in new_area]))
        peak['err_a'] = pd.DataFrame(np.concatenate([np.stack(a) for a in new_err_a])) 
    
        if i==0:
            peak.to_csv('key'+label+'.csv',sep=';', index=False) 
        else:
            peak.to_csv('key'+label+'.csv',sep=';', index=False , header=False , mode='a') 





files = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\MEng student\3rd year\DATA\Raman\sample1\sample1_part1_pillar_a_to_aha_10uM_MB_633nm_50x_10%_1s_3acc_1450cm-1_map.txt'

peakfinder(files)











