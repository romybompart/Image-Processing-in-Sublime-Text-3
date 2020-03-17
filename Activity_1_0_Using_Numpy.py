"""
	@author: romy bompart
	@title: First Activity - Shapes using Python
"""

# import skimage import data, filters
# from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt


#drawing a square manually.
M = np.array ([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,255,255,255,255,255,0,0,0,0,0],
            [0,0,0,0,0,255,255,255,255,255,0,0,0,0,0],
            [0,0,0,0,0,255,255,255,255,255,0,0,0,0,0],
            [0,0,0,0,0,255,255,255,255,255,0,0,0,0,0],
            [0,0,0,0,0,255,255,255,255,255,0,0,0,0,0],
            [0,0,0,0,0,255,255,255,255,255,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

#plotting data
plt.figure()
plt.imshow(M, cmap="gray")
plt.show()

#looking at the shape, dimension of M
a, b = np.shape(M)

#printing
print ( "a = ", a, " b = ", b)

