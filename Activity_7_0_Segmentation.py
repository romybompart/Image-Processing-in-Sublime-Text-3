"""
	@author: romy bompart
	@title: Seventh Activity - Segmentation		
"""

import skimage
from skimage.color import rgb2gray
from skimage import data
from skimage.util import random_noise
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 18
import numpy as np

def kernel_creator(kernel_s,kernel_v=1, f_type=1):
    kernel = np.ones(kernel_s*kernel_s).reshape(kernel_s,kernel_s)

    if f_type ==1 : #paso bajo
        kernel = kernel * kernel_v
    elif f_type ==2: # paso bajo dando peso al medio
        kernel[0,0] = 0
        kernel[kernel_s-1,0] = 0
        kernel[0,kernel_s-1] = 0
        kernel[round((kernel_s-1)/2),round((kernel_s-1)/2)] = kernel_v
        kernel[(kernel_s-1),(kernel_s-1) ]=0
    elif f_type == 3: #paso alto dando peso en al medio
        kernel = kernel * -1
        kernel[round((kernel_s-1)/2),round((kernel_s-1)/2)] = kernel_v
    elif f_type == 4: #paso alto con variacion de peso al medio
        kernel = kernel * - 2
        kernel[0,0] = 1
        kernel[kernel_s-1,0] = 1
        kernel[0,kernel_s-1] = 1
        kernel[round((kernel_s-1)/2),round((kernel_s-1)/2)] = kernel_v
        kernel[(kernel_s-1),(kernel_s-1) ]=1
    elif f_type == 5: #paso alto con variacion de peso al medio
        kernel = kernel * -1
        kernel[0,0] = 0
        kernel[kernel_s-1,0] = 0
        kernel[0,kernel_s-1] = 0
        kernel[round((kernel_s-1)/2),round((kernel_s-1)/2)] = kernel_v
        kernel[(kernel_s-1),(kernel_s-1) ]=0
    elif f_type ==6: #for segmentation horizontal
        kernel = kernel * 0
        kernel [round((kernel_s-1)/2),round((kernel_s-1)/2):] = -1
        kernel[round((kernel_s-1)/2),round((kernel_s-1)/2)] = 1
    elif f_type ==7: #for segmentation vertical
        kernel = kernel * 0
        kernel [:round((kernel_s-1)/2),round((kernel_s-1)/2)] = -1
        kernel[round((kernel_s-1)/2),round((kernel_s-1)/2)] = 1
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
    if filter_type ==0 :
        row, col = np.shape(image)
    else:
        kernel = kernel_creator (kernel_size,kernel_value,filter_type)
        print ( "...the kernel that you are using...")
        print ( kernel )
        padimage = np.pad(image,kernel_size, pad_with)
        row, col = np.shape(padimage)
        
    filtered_image = np.empty([row-kernel_size-1, col-kernel_size-1])

    
    for i in range(row-kernel_size-1):
        for j in range(col-kernel_size-1):
            
            if filter_type ==0: 
                subm_ =  image[ i:kernel_size+i , j:kernel_size+j]
                median = mediana(subm_)
                filtered_image[i,j] = median
            elif filter_type == 3:
                subm_ =  padimage[ i:kernel_size+i , j:kernel_size+j]
                mult_ = np.multiply(subm_,kernel)
                filter_ = np.sum(mult_) / kernel_value
                filtered_image[i,j] = filter_
            else:
                subm_ =  padimage[ i:kernel_size+i , j:kernel_size+j]
                mult_ = np.multiply(subm_,kernel)
                filter_ = np.sum(mult_) / np.sum(np.absolute(kernel))
                filtered_image[i,j] = filter_

            
    return filtered_image

def segmentation (image, kernel_s = 3):
    if( round(kernel_s,0) <2):
        return "error: the kernel size should be higher than 3"
    
    kernel_A = kernel_creator(kernel_s, f_type = 6)
    kernel_B = kernel_creator(kernel_s, f_type = 7)
    print ( "...the kernel that you are using...")
    print ( kernel_A )
    print ( "...the kernel that you are using...")
    print ( kernel_B )

    padimage = np.pad(image,kernel_s, pad_with)
    row, col = np.shape(padimage)
    segmented_image = np.empty([row-kernel_s-1, col-kernel_s-1])


    for i in range(row-kernel_s-1):
        for j in range(col-kernel_s-1):
            
            subm_ =  padimage[ i:kernel_s+i , j:kernel_s+j]
            a = np.sum(np.multiply(subm_,kernel_A))
            b = np.sum(np.multiply(subm_,kernel_B))
            r = a-b
            segmented_image[i,j] = r
    
    return np.abs(segmented_image)

image = rgb2gray(data.rocket())
plt.figure()
plt.title("Original rgb")
plt.imshow(image)


segmented = segmentation(image,kernel_s = 9)


plt.figure()
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.subplot(1,2,2)
plt.title("segmented")
plt.imshow(segmented, cmap='gray')
plt.show()
