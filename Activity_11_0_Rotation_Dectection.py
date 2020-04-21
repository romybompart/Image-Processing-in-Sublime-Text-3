import skimage
from skimage.color import rgb2gray
from skimage import data, io
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
    yis,xis = np.nonzero(image)  
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

def signature(image,centroide_x, centroide_y ):
    y1,x1 = np.nonzero(image)
    x = np.abs(x1-centroide_x)
    y = np.abs(y1-centroide_y)
    arg1, arg2 = np.power(x,2) , np.power(y,2)
    r = np.power((arg1+arg2),1/2)

    max_x = x[np.argmax(r)]
    max_y = y[np.argmax(r)]

    return r, max_x, max_y


filename = os.path.join('images/square_paint.png')
imageRGB = io.imread(filename)
#imageRGB = data.astronaut()
#plt.figure()
#plt.imshow(image)
#plt.show()


image = rgb2gray(imageRGB)
row, col = np.shape(image)

alpha = 45
alpha_rad = np.pi * alpha / 180

cx = col/2
cy = row/2

dx = cx - cx*np.cos(alpha_rad) - cy*np.sin(alpha_rad)
dy = cy + cx*np.sin(alpha_rad) - cy*np.cos(alpha_rad)

rot_m = np.matrix([[np.cos(alpha_rad), np.sin(alpha_rad), dx],\
				 [-np.sin(alpha_rad), np.cos(alpha_rad), dy]])

p0 = np.round(rot_m * np.array([0,0,1]).reshape(3,1),0).astype(int)			# x0,y0
p1 = np.round(rot_m * np.array([col,0,1]).reshape(3,1),0).astype(int)		# x1,y0
p2 = np.round(rot_m * np.array([0,row,1]).reshape(3,1),0).astype(int)		# x0,y1
p3 = np.round(rot_m * np.array([col,row,1]).reshape(3,1),0).astype(int) 	# x0,y0

p = [p0,p1,p2,p3]

i=0

print ("rotation ange...")
print ( str(alpha) + "degrees")

print ( "checking Rotated vertex...")
for items in p:
	print ("point : ", i)
	print ("x: {} , y: {}".format(items[0],items[1]))
	i+=1

print ( "image center...")
print ("x: {} , y: {}".format(cx,cy))

print ( "image size...")
print ("x: {} , y: {}".format(col,row))

a = np.array(p).reshape(4,2)

pmin = np.min(a,0)
pmax = np.max(a,0)

print ( "min point...")
print ( pmin )

print ( "max point...")
print ( pmax )


new_col = pmax[0]-pmin[0]
new_row = pmax[1]-pmin[1]

print ("the new image rotaged will have shape of")
print ("x: {}, y: {}".format(new_col, new_row))

rot = np.ones((new_row,new_col))
#rot = np.ones((row+1,col+1))

for x in range ( col ):
	for y in range (row):
		p = np.round(rot_m * np.array([x,y,1]).reshape(3,1),0).astype(int)
		x_ = p[0] + np.abs(pmin[0])
		y_ = p[1] + np.abs(pmin[1])
		try:
			rot[y_,x_] = image[y,x]
		except:
			pass
			#print ("x = {}, y = {}, x_ = {}, y_ = {}".format(x,y,x_,y_))

rot = filter_application(rot,kernel_size=3,filter_type=0)

x1 = int((new_col-col)/2) 
x2 = int(new_col - x1)
y1 = int((new_row-row)/2)
y2 = int(new_row - y1)

rot = rot[x1:x2,y1:y2]

plt.figure()
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.subplot(1,2,2)
plt.title("Rotated")
plt.imshow(rot, cmap='gray')
plt.show()

oseg = segmentation (image, kernel_s = 3)
rseg = segmentation (rot, kernel_s = 3)

cen_o_x, cen_o_y = centroide(image)

osig , xmax_o, ymax_o = signature(oseg,cen_o_x, cen_o_y)
rsig , xmax_r, ymax_r = signature(rseg,cen_o_x, cen_o_y)

plt.plot(osig, 'r')
y_c = np.ones(np.shape(osig)[0])*np.average(osig)
y_line = np.arange(0, np.shape(osig)[0], 1)
plt.plot(y_line, y_c, 'y--')
plt.show()

plt.figure()
plt.subplot(1,3,1)
plt.title("Segmented 1 ")
plt.imshow(oseg, cmap='gray')
plt.subplot(1,3,2)
plt.title("Segmented 2")
plt.imshow(rseg, cmap='gray')
plt.subplot(1,3,3)
plt.title("signature")
plt.plot(osig, 'r')
y_c = np.ones(np.shape(osig)[0])*np.average(osig)
y_line = np.arange(0, np.shape(osig)[0], 1)
plt.plot(y_line, y_c, 'y--')
plt.tight_layout(pad=0.4, w_pad=0.5)
plt.show()


dx1 = xmax_o -cen_o_x
dy1 = ymax_o-cen_o_y
tetha1=np.degrees(np.arctan(dy1/dx1))

dx2 = xmax_r-cen_o_x
dy2 = ymax_r-cen_o_y
tetha2=np.degrees(np.arctan(dy2/dx2))

tetha = tetha2-tetha1
print ( "rotation")
print(tetha)