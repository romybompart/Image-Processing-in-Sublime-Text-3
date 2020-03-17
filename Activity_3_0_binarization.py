"""
	@author: romy bompart
	@title: Third Activity - Binarization
			
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
