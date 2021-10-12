
#----------------------------------
#----------------------------------
# # Image Segmentation
#----------------------------------
#----------------------------------


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
width = 18
height = 12
img = cv.imread('mri_brain.jpg',0)
plt.figure(figsize=(width, height))
plt.subplot(1,2,1),plt.imshow(img,'gray')
plt.subplot(1,2,2),plt.hist(img.ravel(),256,)
plt.show()


#----------------------------------
# # Image Binarization - Thresholding
# Grouping pixels in two classes
#----------------------------------

unit = 6
width = 3*unit
height = 2*unit
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
plt.figure(figsize=(width, height))
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

#----------------------------------
# # Thresholding - OTSU
#----------------------------------

# global thresholding
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu's thresholding [only works on grayscale]
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
#Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
#plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
plt.figure(figsize=(width, height))
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

#----------------------------------
# # Image Gradient - Sobel Operator
#----------------------------------
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
plt.figure(figsize=(3*unit,unit))
plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

#----------------------------------
# # Canny Edge detection
#----------------------------------
edges = cv.Canny(img,100,255)

plt.figure(figsize=(width, height))
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

#----------------------------------
# # Marr-Hildreth Edge Detector # Laplacian
#----------------------------------
laplacian = cv.Laplacian(img,cv.CV_64F)

plt.figure(figsize=(width, height))
plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()

#----------------------------------
# # Watershed
#----------------------------------

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
D = ndimage.distance_transform_edt(th1)
localMax = peak_local_max(D, indices=False, min_distance=10,labels=th1)
plt.figure(figsize=(width, height))
plt.subplot(1,2,1),plt.imshow(th1,cmap="gray")
plt.subplot(1,2,2),plt.imshow(D,cmap="gray")
plt.show()
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=th1)


# In[8]:

# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
        continue
 
    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(img.shape, dtype="uint8")
    mask[labels == label] = 255
    
 
    # detect contours in the mask and grab the largest one
    im2, contours, hiererchy = cv.findContours(th1, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
print("Number of pixels: ", labels.size)
print("Number of clusters: ", np.unique(labels).size)
cv.drawContours(img, contours, -1, (0,255,0), 2)

 
# show the output image
plt.figure(figsize=(width, height))
plt.imshow(img,cmap='gray')
plt.title("Watershed Segmentation")
plt.show()

#----------------------------------
# Kmeans Clustering
#----------------------------------
import numpy as np
import cv2

img = cv2.imread('hw1/mri_brain.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)
plt.figure(figsize=(unit*3, unit*4))

# define criteria, number of clusters(K) and apply kmeans()
i=1
kmeans = [2,8,20]
plt.subplot(int(np.ceil(len(kmeans)/2))+1,2,i),plt.imshow(img,cmap="gray")
plt.suptitle("Image after K-Means Clustering")
plt.title("original")
for K in kmeans:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # show the output image
    
   
    i=i+1    
    
    plt.subplot(int(np.ceil(len(kmeans)/2))+1,2,i),plt.imshow(res2,cmap="gray")
    plt.title("K = " + str(K))
    plt.imshow(res2,cmap='gray')
plt.show()

#----------------------------------
# # Hierarchical Clustering
#----------------------------------

import time as time

import numpy as np
import scipy as sp
import cv2

import matplotlib.pyplot as plt

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering



# Generate data
img = sp.misc.imread('hw1/mri_brain.jpg',flatten=True) 

# Resize it to 10% of the original size to speed up the processing
img = sp.misc.imresize(img, 0.30) / 255.

X = np.reshape(img, (-1, 1))


# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*img.shape)


# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()
n_region = 20  # number of regions
ward = AgglomerativeClustering(n_clusters=n_region, linkage='ward',
                               connectivity=connectivity)
ward.fit(X)
label = np.reshape(ward.labels_, img.shape)
print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", label.size)
print("Number of clusters: ", np.unique(label).size)


# Plot the results on an image
plt.figure(figsize=(18, 12))
plt.imshow(face, cmap=plt.cm.gray)
for l in range(n_region):
    plt.contour(label == l, contours=1,color=(0,255,0))
plt.xticks(())
plt.yticks(())
plt.suptitle("Region based on Hiererchical Clustering")
plt.show()




