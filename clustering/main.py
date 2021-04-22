# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:04:59 2021

@author:NTN-code
"""

"""
Предтем как читать код посмотрите example.py
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt


def main():
    # список для времени
    times = []

    # включаем оптимизацию методов для модуля openCV
    cv2.setUseOptimized(True)

    print("Start")

    for x in range(0,2):
        #начало отсчёта
        time1 = cv2.getTickCount()

        oimg = cv2.imread("C:\python\(((_OpenCV\image\mill.jpg")
        # изменить размер изображения
        oimg = cv2.resize(oimg,(900,600))
        cv2.imshow('Origin',oimg)

        # методы конвертации исходного изображения в другие цветовые пространства
        #cimg = cv2.cvtColor(oimg,cv2.COLOR_BGR2Lab)
        #cimg = cv2.cvtColor(oimg,cv2.COLOR_BGR2LUV)
        #cimg = cv2.cvtColor(oimg,cv2.COLOR_BGR2GRAY)
        #cimg = cv2.cvtColor(oimg,cv2.COLOR_BGR2RGBA)
        cimg = cv2.cvtColor(oimg,cv2.COLOR_BGR2HLS)
        cv2.imshow("HSV",cimg)
        #cimg = cv2.cvtColor(oimg,cv2.COLOR_BGR2RGB)
        #cv2.imshow('convert',cimg)
        #cimg = oimg

        img_matrix = cimg.reshape((-1,3))
        img_matrix = np.float32(img_matrix)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        K = 5
        ret,label,center=cv2.kmeans(img_matrix,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((cimg.shape))
        cv2.imshow('CLASTER',res2)
        cv2.imwrite('name.png',res2)

        # конец отсчёта
        time2 = cv2.getTickCount()

        # нахождение разницы во времени
        time = (time2 - time1)/ cv2.getTickFrequency()
        times.append(time)

        print(time)

    print("Время исполнения")
    print (times)
    print("Используем оптимизацию " + str(cv2.useOptimized()))
    print("End")
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
