# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:45:27 2021

@author: NTN-code
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    # чтение изображения
    img = cv2.imread('C:\python\(((_OpenCV\image\corona.jpg')

    # конвертация в черно-белое цветовое пространство
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # бинаризация изображения
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # вывод изображения
    cv2.imshow('gray',gray)

    # Гаусово размытие
    thresh = cv2.GaussianBlur(thresh,(5,5),0)
    cv2.imshow('binary img',thresh)

    # избавление от шумов на изображении (удаление не нужных пикселей)
    kernel = np.ones((50,50),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
    cv2.imshow('noise',opening)

    # находим задние области
    sure_bg = cv2.dilate(opening,kernel,iterations=1)
    cv2.imshow('big',sure_bg)

    # поиск нужной области переднего плана
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    cv2.imshow('small',dist_transform)

    # нахождение переднии области
    ret, sure_fg = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0)



    # нахождение безымянной области
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    cv2.imshow('unknown',unknown)

    # нахождение маркеров
    ret, markers = cv2.connectedComponents(sure_fg)



    # счёт и цвет маркеров
    markers = markers+1

    # конвертируем пиксили имеющие 255 в 0
    markers[unknown==255] = 0

    # применение алгоритма водораздела
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    cv2.imshow('finish',img)



    # создание сегментов
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

