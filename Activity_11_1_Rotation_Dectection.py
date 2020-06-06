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

filename = os.path.join('images/arrow.png')
imageRGB = io.imread(filename)
#imageRGB = data.astronaut()
#plt.figure()
#plt.imshow(image)
#plt.show()


image = rgb2gray(imageRGB)
row, col = np.shape(image)

alpha = 15
alpha_rad = np.pi * alpha / 180

cx = col/2
cy = row/2

dx = cx - cx*np.cos(alpha_rad) - cy*np.sin(alpha_rad)
dy = cy + cx*np.sin(alpha_rad) - cy*np.cos(alpha_rad)

rot_m = np.matrix([[np.cos(alpha_rad), np.sin(alpha_rad), dx],\
         [-np.sin(alpha_rad), np.cos(alpha_rad), dy]])

p0 = np.round(rot_m * np.array([0,0,1]).reshape(3,1),0).astype(int)     # x0,y0
p1 = np.round(rot_m * np.array([col,0,1]).reshape(3,1),0).astype(int)   # x1,y0
p2 = np.round(rot_m * np.array([0,row,1]).reshape(3,1),0).astype(int)   # x0,y1
p3 = np.round(rot_m * np.array([col,row,1]).reshape(3,1),0).astype(int)   # x0,y0

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


print( "size: ", np.shape(rot))
print ( "x1: {} , x2: {}, y1: {} , y2: {}".format(x1,x2,y1,y2))

plt.figure()
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.subplot(1,2,2)
plt.title("Rotated")
plt.imshow(rot, cmap='gray')
plt.show()

image_size=[0,0]
columns=0
rows=0
mean=0
area_=0
centroid_x=0
centroid_y=0
max_x=0
max_y=0
max_x_r=0
max_y_r=0
index = 0
radius = 0
contour_points = []
contour_points_r = []



mask_1=np.array([  [0., 0., -1.],  # definir filtro pasa-altas personalizado
          [0., 3., -1.],
          [0., 0., -1.]])

mask_2=np.array([  [0., 0., 0.], # definir filtro pasa-altas personalizado
          [0., 3., 0.],
          [-1., -1., -1.]])

corner=[0,0]
local_average_1=0
local_average_2=0

input_image = image
input_image_r = rot
image_size = np.shape(rot)
segmented_image = np.zeros(np.shape(input_image))
segmented_image_r = np.zeros(np.shape(input_image))

#area and centroid function
for rows in range(1,image_size[0]-1):
       for columns in range(1,image_size[1]-1):
              if(input_image[rows,columns]==1):
                     area_ = area_ + 1
                     centroid_x += columns
                     centroid_y += rows

centroid_x=centroid_x/area_
centroid_y=centroid_y/area_

#print area and centroid location
print("area: ",area_)
print("centroid_x: ",centroid_x)
print("centroid_y: ",centroid_y)


#segmentation function of first image
for rows in range(1,image_size[0]-1):
  for columns in range(1,image_size[1]-1):
    corner[0] = rows-1
    corner[1] = columns-1
    for i in range(0,3):
      for j in range(0,3):
        local_average_1 = local_average_1 + mask_1[i,j]*input_image[corner[0]+i,corner[1]+j]
        local_average_2 = local_average_2 + mask_2[i,j]*input_image[corner[0]+i,corner[1]+j]
    segmented_image[rows,columns]=np.abs(local_average_1 - local_average_2)
    local_average_1=0
    local_average_2=0

#segmentation function of second image
for rows in range(1,image_size[0]-1):
       for columns in range(1,image_size[1]-1):
              corner[0] = rows-1
              corner[1] = columns-1
              for i in range(0,3):
                     for j in range(0,3):
                            local_average_1 = local_average_1 + mask_1[i,j]*input_image_r[corner[0]+i,corner[1]+j]
                            local_average_2 = local_average_2 + mask_2[i,j]*input_image_r[corner[0]+i,corner[1]+j]
              segmented_image_r[rows,columns]=np.abs(local_average_1 - local_average_2)
              local_average_1=0
              local_average_2=0

#binarize first image
for rows in range(1,image_size[0]-1):
       for columns in range(1,image_size[1]-1):
              if segmented_image[rows,columns]>0.5:
                     segmented_image[rows,columns]=1
              else:
                     segmented_image[rows,columns]=0

#binarize second image
for rows in range(1,image_size[0]-1):
       for columns in range(1,image_size[1]-1):
              if segmented_image_r[rows,columns]>0.5:
                     segmented_image_r[rows,columns]=1
              else:
                     segmented_image_r[rows,columns]=0

#find signature of first image and its more distant corner
for rows in range(1,image_size[0]-1):
       for columns in range(1,image_size[1]-1):
              if(segmented_image[rows,columns]==1): 
                     radius = np.sqrt(np.abs(centroid_x-columns)**2 + np.abs(centroid_y-rows)**2)
                     if(len(contour_points)>0):
                            if(radius>np.max(contour_points)):
                                   max_x=columns
                                   max_y=rows
                     contour_points.append(radius)

#find signature of second image and its more distant corner
for rows in range(1,image_size[0]-1):
       for columns in range(1,image_size[1]-1):
              if(segmented_image_r[rows,columns]==1):
                     radius = np.sqrt(np.abs(centroid_x-columns)**2 + np.abs(centroid_y-rows)**2)
                     if(len(contour_points_r)>0):
                            if(radius>np.max(contour_points_r)):
                                   max_x_r=columns
                                   max_y_r=rows
                     contour_points_r.append(radius)

print(np.argmax(contour_points))
print(np.max(contour_points))
print("max x:", max_x, "max y:", max_y)

print("second image")
print(np.argmax(contour_points_r))
print(np.max(contour_points_r))
print("max x:", max_x_r, "max y:", max_y_r)


dx1 = max_x-centroid_x
dy1 = max_y-centroid_y
tetha1=np.degrees(np.arctan(dy1/dx1))

dx2 = max_x_r-centroid_x
dy2 = max_y_r-centroid_y
tetha2=np.degrees(np.arctan(dy2/dx2))

tetha = tetha2-tetha1
print ( "... Detecting rotation angle of: .")
print(tetha)


plt.figure(1)
plt.subplot(2,3,1)
plt.imshow(input_image, cmap='gray')
plt.subplot(2,3,2)
plt.imshow(segmented_image, cmap='gray')
plt.subplot(2,3,3)
y_pos = np.arange(len(contour_points))
plt.bar(y_pos, contour_points, align='center', alpha=0.5)
plt.xlabel('histogram 1')
plt.subplot(2,3,4)
plt.imshow(input_image_r, cmap='gray')
plt.subplot(2,3,5)
plt.imshow(segmented_image_r, cmap='gray')
plt.subplot(2,3,6)
y_pos = np.arange(len(contour_points_r))
plt.bar(y_pos, contour_points_r, align='center', alpha=0.5)
plt.xlabel('histogram 1')
plt.show()