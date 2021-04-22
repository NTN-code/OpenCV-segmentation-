# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 01:10:43 2021

@author: NTN-code
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
#from skimage.segmentation import clear_border



def main():
    times = []
    cv2.setUseOptimized(True)


    for i in range(0,2):
        t1 = cv2.getTickCount()

        img = cv2.imread('C:\python\(((_OpenCV\image\corona.jpg')

        img = cv2.resize(img,(900,600))
        cv2.imshow('ORIGIN',img)

        #img = cv2.resize(img,(500,500),0)
        #img = cv2.GaussianBlur(img,(1,1),0)
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # берем матрицу голубово цвета
        gray = img[:,:,2]
        cv2.imshow('CONVERT',gray)


        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imshow('THRESHOLD',thresh)

        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
        cv2.imshow('MORPHOLOGY',opening)

        # заливка соприкасающиеся сегменты с границей
        #opening = clear_border(opening)

        sure_bg = cv2.dilate(opening,kernel,iterations=2)
        cv2.imshow('background',sure_bg)

        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        ret, sure_fg = cv2.threshold(dist_transform,0.09*dist_transform.max(),255,0)
        cv2.imshow('frontground',sure_fg)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        cv2.imshow('UNKNOWN_AREA',unknown)

        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0

        plt.imshow(markers,cmap='jet')


        markers = cv2.watershed(img,markers)
        img[markers == -1] = [255,255 ,0]
        cv2.imshow('WATERWHED',img)
        #cv2.imshow('21',img)

        t2 = cv2.getTickCount()
        t = (t2 - t1)/cv2.getTickFrequency()
        times.append(t)
        print(t)


    print("Время исполнения")
    print(times)
    print("Используем оптимизацию " + str(cv2.useOptimized()))
    print("End")

    dpi = plt.rcParams['figure.dpi']
    figsize = img.shape[1] / float(dpi), img.shape[0] / float(dpi)
    plt.figure(figsize=figsize)
    plt.imshow(markers, cmap="jet")
    filename = "markerswater.jpg"
    plt.axis('off')
    plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()





