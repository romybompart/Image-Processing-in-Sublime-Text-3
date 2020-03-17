"""
	@author: romy bompart
	@title: First Activity Continuation - Using Skimage
			Inverting Image
"""

import skimage
from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 18
import numpy as np

#inverting image function
def Invert(image):
    grayim = rgb2gray(image)
    a, b = np.shape(grayim)
    inverted = np.empty([a, b])
    for k in range(a):
        for i in range (b):
            inverted[k,i] = 255 - grayim[k,i]
    return inverted

image = data.logo()
plt.figure()
plt.imshow(image)
plt.show()

invertedimage = Invert(image)

plt.figure()
plt.imshow(invertedimage, cmap="gray")
plt.show()