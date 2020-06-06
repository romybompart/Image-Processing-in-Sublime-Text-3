import matplotlib.pyplot as plt
import os
# skimage
from skimage import io
from skimage.color import rgb2gray

#
def main():
	path = '../images/'

	image_name = 'romybompart.JPG'

	filename = os.path.join(path,image_name)
	
	print(filename)
	imageRGB = io.imread(filename)	
	image = rgb2gray(imageRGB)

	# plt.figure()
	# plt.imshow(image, cmap='gray')
	# plt.show()

	io.imsave(filename,image)

if __name__ == '__main__':
	main()
