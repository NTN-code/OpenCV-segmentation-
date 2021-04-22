# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:14:03 2021

@author: NTN-code
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt

origin_img = cv2.imread('C:\python\(((_OpenCV\image\corona.jpg')
origin_img = cv2.resize(origin_img,(900,900))

# преобразование черно-белое пространство
origin_gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)

# Гаусово размытие
origin_gray_img = cv2.GaussianBlur(origin_gray_img,(5,5),0)

cv2.imshow("ORIGIN",origin_img)
cv2.imshow("GRAY",origin_gray_img)

#laplacian = cv2.Laplacian(origin_gray_img,cv2.CV_64F)
#cv2.imshow("laplacian",laplacian)

#edges = cv2.Canny(origin_gray_img,100,200)
#cv2.imshow("edges",edges)

ret, thresh = cv2.threshold(origin_gray_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('ret',ret)
cv2.imshow('thresh',thresh)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

cv2.imshow('opening',opening)

sure_bg = cv2.dilate(opening,kernel,iterations=3)

cv2.imshow('sure-bg',sure_bg)

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

cv2.imshow('sure-fg',sure_fg)


sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

#cv2.imshow('unknown',unknown)

ret, markers = cv2.connectedComponents(sure_fg)

#cv2.imshow('marker',markers)
markers = markers+1

markers[unknown==255] = 0

markers = cv2.watershed(origin_img,markers)

#cv2.imshow('markers',markers)

origin_img[markers == -1] = [255,0,0]

dpi = plt.rcParams['figure.dpi']
figsize = origin_img.shape[1] / float(dpi), origin_img.shape[0] / float(dpi)

plt.figure(figsize=figsize)
plt.imshow(markers, cmap="jet")
filename = "markerswater.jpg"
plt.axis('off')
plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0)


cv2.imshow('finish',origin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.waitKey(0)


