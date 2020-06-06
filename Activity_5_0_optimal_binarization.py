"""
	@author: romy bompart
	@title: Fifth Activity - Optimal Binarization
			
"""

import skimage
from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 18
import numpy as np

def Invert(image):
    grayim = rgb2gray(image)
    a, b = np.shape(grayim)
    inverted = np.empty([a, b])
    for k in range(a):
        for i in range (b):
            inverted[k,i] = 255 - grayim[k,i]
    return inverted

def binarization(image, middle):
    a, b = np.shape(image)
    binarized = np.empty([a, b])
    for k in range(a):
        for i in range (b):
            if (image[k,i]>=middle):
                binarized[k,i] = 255
            else:
                binarized[k,i]= 0
            
    return binarized

def histogram(image):
    bins = np.zeros([256])
    a, b = np.shape(image)
    
    for k in range(a):
        for i in range (b):
            bins[int(image[k,i])] =  1 + bins[int(image[k,i])]
            
    return bins

def quadratic(image, **coefficient):
    if (len(coefficient)<=4):
        a, b = np.shape(image)
        transformed = np.empty([a, b])
        coff = np.zeros(4)
        j=0
        
        for item in coefficient:
            coff[j] = coefficient[item]
            j=j+1
        
        for k in range(a):
            for i in range(b):
                transformed[k,i] =  ( (coff[0]*(image[k,i]**coff[3]) + coff[1]*(image[k,i]) + coff[2]) )           
        return transformed
    else:
        print ("too much arguments")
        return image

def varianzas(data,image):
    L = len ( data )
    a, b = np.shape(image) 
    ntp = a*b
    y = np.arange(L)
    Varianzas = np.zeros([L])
    
    for i in range (L):
        R1 = sum ( data[0:i])
        R2 = ntp - R1
        if (R1 == 0):
            m1 = 0
        else:
            m1 = sum (data[0:i] * y[0:i])/R1
    
        if (R2==0):
            m2 =0
        else:
            m2 = sum ( data[i+1:L]*y[i+1:L])/R2
        
        Varianzas[i] = (R1*R2*(m2-m1))
        
    return Varianzas

def whereMax(vm):
    max_v = max ( vm )
    n = len ( vm )
    for i in range ( n ):
        if ( vm[i] == max_v ):
            return i
    return "error"

def optimal_binarization(image):
    his = histogram(image)
    # print ( his )
    vm = varianzas(his, image)
    # print( vm )
    pos_vm = whereMax(vm)
    print(pos_vm)
    binimage = binarization( image,pos_vm)
    # print( image )
    return binimage

image = rgb2gray(data.rocket())
plt.figure()
plt.imshow(image)

bin_image  = optimal_binarization(image*255)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(bin_image, cmap='gray')
plt.show()
