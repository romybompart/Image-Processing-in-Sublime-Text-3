"""
	@author: romy bompart
	@title: Fourth Activity - Quadratic Transformation
			
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
                binarized[k,i] = 1
            else:
                binarized[k,i]=0
            
    return binarized

def histogram(image):
    maxi = image.max()
    bins = np.zeros([255])
    a, b = np.shape(image)
    
    for k in range(a):
        for i in range (b):
            bins[int(image[k,i])] =  1 + bins[int(image[k,i])]
            
    return bins

def quadratic(image, **coefficient):
    if (len(coefficient)<=4):
        a, b = np.shape(image)
        transformed = np.zeros([a, b])
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


image = data.chelsea()
plt.figure()
plt.imshow(image)
plt.show()

invertedimage = Invert(image)

plt.figure()
plt.imshow(invertedimage, cmap="gray")
plt.show()

value = 120
binimage = binarization( rgb2gray(image) * 255 ,value)

plt.figure()
plt.imshow(binimage, cmap="gray")
plt.show()

his = histogram(rgb2gray(image)*255)
plt.bar(np.arange(len(his)),his,align='center',alpha=1)
plt.show()

transf = quadratic(rgb2gray(image)*255, a=1,b=0,c=0,e=2)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(transf, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(rgb2gray(image), cmap='gray')
plt.show()
