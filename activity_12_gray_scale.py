import matplotlib.pyplot as plt
import os
# skimage
from skimage import io
from skimage.color import rgb2gray

#
def main():
	path = 'C:/Users/Romy1/Documents/Documentos Romy/Books/' \
	'UANL MASTER/Segundo Tetra/Analisis de Sistemas de Potencia/' \
	'TAREA/TAREA 3/Document/images/'

	image_name = 'javierbaca.JPG'

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