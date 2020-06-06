import skimage
from skimage.color import rgb2gray
from skimage import data, io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['font.size'] = 18
import numpy as np
import os

filename = os.path.join('images/arrow.png')
imageRGB = io.imread(filename)
image = rgb2gray(imageRGB)
row, col = np.shape(image)

print( "rows: {}, cols: {}".format(row,col))

cx = col/2
cy = row/2

ite = 4
imgs = []
rot = np.ones((row,col))
alpha = 0
fig = plt.figure()

for iteration in range(ite):
	alpha+=360/ite
	alpha_rad = np.pi * alpha / 180
	dx = cx - cx*np.cos(alpha_rad) - cy*np.sin(alpha_rad)
	dy = cy + cx*np.sin(alpha_rad) - cy*np.cos(alpha_rad)
	for x in range ( col ):
		for y in range (row ):
			rot_m = np.matrix([[np.cos(alpha_rad), np.sin(alpha_rad), dx],\
				 [-np.sin(alpha_rad), np.cos(alpha_rad), dy]])
			p = (rot_m * np.array([x,y,1]).reshape(3,1)).astype(int)
			x_ = p[0] 
			y_ = p[1]
			try:
				if( x_ >=0 and x_<col and y_>=0 and y_<row):
					rot [y_,x_] = image[y,x]
			except:
				print ("x:{}, y:{}, x_{}, y_{}".format(x,y,x_,y_))


	imgs.append([plt.imshow(rot, animated=True, cmap='gray')])
	print ( "processing iteration, angle: ", str(alpha))


ani = animation.ArtistAnimation(fig,imgs,interval=1000,blit=True,repeat_delay=1000)
plt.show()