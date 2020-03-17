"""
	@author: romy bompart
	@title: Fourth Activity - Histogram
			
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
            bins[int(image[k,i]*255)] =  1 + bins[int(image[k,i]*255)]
            
    return bins

image = data.chelsea()
plt.figure()
plt.imshow(image)
plt.show()

invertedimage = Invert(image)

plt.figure()
plt.imshow(invertedimage, cmap="gray")
plt.show()

value =0.5
binimage = binarization( rgb2gray(image),value)

plt.figure()
plt.imshow(binimage, cmap="gray")
plt.show()

his = histogram(rgb2gray(image))
plt.bar(np.arange(len(his)),his,align='center',alpha=1)
plt.show()