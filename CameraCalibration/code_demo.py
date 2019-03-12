# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:32:24 2019

@author: 小梁
"""

import cv2 as cv
import glob
import numpy as np


def code_demo(src):
    #棋盘每行内角点个数
    num_x = 9
    #棋盘每列内角点个数
    num_y = 6
    #初始化世界坐标Z为0
    objectPs = np.zeros((num_x*num_y, 3), np.float32)
    #记录每个内角点的世界坐标
    objectPs[:,:2] = np.mgrid[0:(num_x-1)*90+1:90,0:(num_y-1)*90+1:90].T.reshape(-1,2)
    #世界坐标
    objPs = []
    #图像坐标
    imgPs = []
    #文件夹内的图片全为同一个相机对同一个棋盘图片的拍摄照片
    images = glob.glob('{}/*.jpg'.format(src))
    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #寻找内角点
        ret, corners = cv.findChessboardCorners(gray, (9,6))
        #优化内角点（亚像素级）
        corners_sub = cv.cornerSubPix(gray, corners, (15,15), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objPs.append(objectPs)
        imgPs.append(corners_sub)
        #绘图
        img = cv.drawChessboardCorners(img, (9,6), corners_sub, ret)
        cv.imshow('result', img)
        cv.waitKey(1000)
    #计算相机内参矩阵，畸变矩阵，旋转矩阵，平移矩阵
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objPs, imgPs, gray.shape[::-1],None,None)
    print('cameraMatrix:\n',mtx)
    print('distCoeffs\n:',dist)
    
    
code_demo('C:/Users/userltk/Desktop/cv_Img/example')
cv.waitKey(0)
cv.destroyAllWindows()