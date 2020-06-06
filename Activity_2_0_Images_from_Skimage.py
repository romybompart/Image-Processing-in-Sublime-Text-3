"""
	@author: romy bompart
	@title: Second Activity - Using Skimage
			from data a member from skiamge
"""

import skimage
from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 18
import numpy as np

images = ('astronaut',
          'binary_blobs',
          )

for name in images:
    caller = getattr(data, name)
    image = caller()
    plt.figure()
    plt.title(name)
    plt.imshow(image, cmap="gray")
    plt.show()

data.