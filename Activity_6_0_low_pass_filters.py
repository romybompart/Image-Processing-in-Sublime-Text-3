"""
	@author: romy bompart
	@title: Sixth Activity - Low Pass Filter		
"""

import skimage
from skimage.color import rgb2gray
from skimage import data
from skimage.util import random_noise
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 18
import numpy as np

def kernel_creator(kernel_s,kernel_v, f_type):
    kernel = np.ones(kernel_s*kernel_s).reshape(kernel_s,kernel_s)
    if f_type==0:
         kernel = kernel * kernel_v
    elif f_type ==1 :
        kernel[0,0] = 0
        kernel[kernel_s-1,0] = 0
        kernel[0,kernel_s-1] = 0
        kernel[round((kernel_s-1)/2),round((kernel_s-1)/2)] = kernel_v
        kernel[(kernel_s-1),(kernel_s-1) ]=0
    else:
        kernel = 0
    
    return kernel

def mediana(matrix):
    l = np.shape(matrix)[0] * np.shape(matrix)[0]
    vector = np.sort(matrix.reshape(l))
    m_p = round(l/2)
    if ( l%2 ==0 ):
        median = (vector[m_p] + vector[m_p-1]) /2
    else:
        median = (vector[m_p])
    return median

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def filter_application(image,kernel_size=3, kernel_value=1, filter_type=0):
    if( round(kernel_size,0) <2):
        return "error: the kernel size should be higher than 3"
    
    print ("filter type: ", filter_type)
    if filter_type <=1 :
        kernel = kernel_creator (kernel_size,kernel_value,filter_type)
        print ( "...the kernel that you are using...")
        print ( kernel )
        padimage = np.pad(image,kernel_size, pad_with)
        row, col = np.shape(padimage)
        filtered_image = np.empty([row-kernel_size-1, col-kernel_size-1])
    else:
        row, col = np.shape(image)
        filtered_image = np.empty([row-kernel_size-1, col-kernel_size-1])
    
    for i in range(row-kernel_size-1):
        for j in range(col-kernel_size-1):
            
            if filter_type <=1: 
                subm_ =  padimage[ i:kernel_size+i , j:kernel_size+j]
                mult_ = np.multiply(subm_,kernel)
                filter_ = np.sum(mult_) / np.sum(kernel)
                filtered_image[i,j] = filter_
            else:
                subm_ =  image[ i:kernel_size+i , j:kernel_size+j]
                median = mediana(subm_)
                filtered_image[i,j] = median
            
    return filtered_image


image = rgb2gray(data.rocket())
plt.figure()
plt.imshow(image)

#image_ruido = image + 2.4*image.std()*np.random.random(image.shape)
image_ruido = random_noise(image,mode="s&p", amount = 0.3)

filtered = filter_application(image_ruido,kernel_size=5,filter_type=2)


plt.figure()
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.subplot(1,3,2)
plt.title("Noise")
plt.imshow(image_ruido, cmap='gray')
plt.subplot(1,3,3)
plt.title("Filtered")
plt.imshow(filtered, cmap='gray')
plt.show()
