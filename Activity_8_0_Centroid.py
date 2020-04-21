"""
	@author: romy bompart
	@title: Eighth Activity - Centroid		
"""

import skimage
from skimage.color import rgb2gray
from skimage import data, io
from skimage.util import random_noise
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 18
import numpy as np
import os

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


def varianzas(data,image):
    L = len ( data )
    a, b = np.shape(image) 
    ntp = a*b
    y = np.arange(L)
    Varianzas = np.zeros([L])
    
    for i in range (L):
        R1 = sum ( data[0:i])
        R2 = ntp - R1
        if (R1 == 0):
            m1 = 0
        else:
            m1 = sum (data[0:i] * y[0:i])/R1
    
        if (R2==0):
            m2 =0
        else:
            m2 = sum ( data[i+1:L]*y[i+1:L])/R2
        
        Varianzas[i] = (R1*R2*(m2-m1))
        
    return Varianzas

def whereMax(vm):
    max_v = max ( vm )
    n = len ( vm )
    for i in range ( n ):
        if ( vm[i] == max_v ):
            return i
    return "error"

def histogram(image):
    bins = np.zeros([255])
    a, b = np.shape(image)
    
    for k in range(a):
        for i in range (b):
            bins[round(int(image[k,i]),0)] =  1 + bins[round(int(image[k,i]),0)]
    
    return bins

def binarization(image, middle):
    a, b = np.shape(image)
    binarized = np.empty([a, b])
    for k in range(a):
        for i in range (b):
            if (image[k,i]>=middle):
                binarized[k,i] = 255
            else:
                binarized[k,i]=0
            
    return binarized

def optimal_binarization(image):
    his = histogram(image)
    vm = varianzas(his, image)
    pos_vm = whereMax(vm)
    binimage = binarization( image,pos_vm)
    return binimage

def area(image):
    return np.sum(image)

def invert(image):
    return np.abs((image/255) - 1)

def centroide(image):
    image = invert(image)
    xis, yis = np.nonzero(image)  
    x = xis.mean()
    y = yis.mean()
       
    return (x,y)

def centroide2(image):
    row, col = np.shape(image)
    ci = np.zeros(row)
    cj = np.zeros(col)
    image = invert(image)
    area_ = area(image)
    for i in range(row):
        for j in range (col):
            ci[i] = i*image[i,j] + ci[i]
            cj[j] = j*image[i,j] + cj[j]

    y = np.sum(ci)/area_        
    x = np.sum(cj)/area_
            
    return (x,y)

#filename = os.path.join('images/apple_Red3.jpg')
filename = os.path.join('images/apple_Red2.jpg')
#filename = os.path.join('images/banana.jpg')
#filename = os.path.join('images/persona.jfif')
#filename = os.path.join('images/mexico_map_2.gif')
#filename = os.path.join('images/square.jpg')
#filename = os.path.join('images/pentagon.jpg')
image_c = io.imread(filename)
#image_c = image = data.rocket()
image = rgb2gray(image_c)


binimage = optimal_binarization(image*254)
filtered = filter_application(binimage,kernel_size=3, kernel_value=1, filter_type=1)
x,y = centroide2(filtered)

plt.figure()
plt.subplot(1,2,1)
plt.title("Binarized")
plt.imshow(binimage, cmap='gray')
plt.subplot(1,2,2)
plt.title("filtered")
plt.imshow(filtered, cmap='gray')
plt.show()

print ( "Centroid - > X = {} , Y= {}".format(x,y))


###
plt.imshow(image_c)
x_line = np.arange(0, np.shape(image_c)[0], 1)
y_line = np.arange(0, np.shape(image_c)[1], 1)
x_c = np.ones(np.shape(image_c)[0])*x
y_c = np.ones(np.shape(image_c)[1])*y
###
plt.plot(x_c, x_line, 'r--')
plt.plot(y_line, y_c, 'r--')
plt.scatter(x,y,c='w',s=60)
plt.show()

