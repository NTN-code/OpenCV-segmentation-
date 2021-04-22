# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:40:02 2021

@author: NTN-code
"""
import numpy as np
import cv2

def main():

    #прочитать изображение
    img = cv2.imread('C:\python\(((_OpenCV\image\corona.jpg')

    #преобразование 3 матриц с размерами (sizeXsize) в матрицу newsizeX3
    Z = img.reshape((-1,3))

    # конвертируем в вещественный 32 битный тип
    Z = np.float32(Z)

    # критерии к алгоритму кластеризации
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # определяем количество кластреров
    K = 3

    # применение алгоритма кластеризации
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # конвертируем в целочисленный 8 битный тип
    center = np.uint8(center)

    # возращаем исходное изображение с алгоритмом кластеризации
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # показываем наше изображение
    cv2.imshow('res2',res2)

    # ожидание закрытия изображения
    cv2.waitKey(0)

    # уничтожение окон
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()