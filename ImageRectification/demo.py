# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 19:13:52 2019

@author: 小梁
"""
import numpy as np
import cv2 as cv
def getROI(src):
    img = cv.imread(src)
    cv.namedWindow('img',cv.WINDOW_NORMAL)
    cv.resizeWindow('img',500,600)
    cv.imshow('img',img)
    h,w = img.shape[:2]
    cenx = 0.5*w
    ceny = 0.5*h
    newimg = np.zeros((h,w),np.uint8)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    bl = cv.GaussianBlur(gray,(15,15),0)
    ret, binary = cv.threshold(bl,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT,(7,7))
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT,(121,121))
    dst = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel1)
    dst = cv.morphologyEx(dst,cv.MORPH_CLOSE,kernel2)
    #获取快递单大体轮廓
    cv.namedWindow('img1',cv.WINDOW_NORMAL)
    cv.resizeWindow('img1',500,600)
    cv.imshow('img1',dst)
    cloneImage,contours,heriachy = cv.findContours(dst,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv.minAreaRect(c)
        ps = cv.boxPoints(rect)
        line1 = ((ps[0][0]-ps[1][0])**2+(ps[0][1]-ps[1][1])**2)**0.5
        line2 = ((ps[0][0]-ps[3][0])**2+(ps[0][1]-ps[3][1])**2)**0.5
        if (line1*line2)<6000:
            continue
        cv.drawContours(newimg,contours,-1,(255,255,255),cv.FILLED)
        masked = cv.bitwise_and(img,img,mask = newimg)
        if line1>line2:
            theta = rect[2]
        else:
            theta = 90+rect[2]
        R = cv.getRotationMatrix2D((cenx,ceny),theta,1)
        finimg = cv.warpAffine(masked,R,(w,h))
        cv.circle(img,(ps[0][0],ps[0][1]),55,(0,0,0),15)
        cv.circle(img,(ps[1][0],ps[1][1]),55,(255,0,0),15)
        cv.circle(img,(ps[2][0],ps[2][1]),55,(0,255,0),15)
        cv.circle(img,(ps[3][0],ps[3][1]),55,(0,255,255),15)
        cv.line(img,(ps[0][0],ps[0][1]),(ps[1][0],ps[1][1]),(0,0,0),15)
        cv.line(img,(ps[1][0],ps[1][1]),(ps[2][0],ps[2][1]),(255,0,0),15)
        cv.line(img,(ps[2][0],ps[2][1]),(ps[3][0],ps[3][1]),(0,255,0),15)
        cv.line(img,(ps[3][0],ps[3][1]),(ps[0][0],ps[0][1]),(0,255,255),15)
        gray1 = cv.cvtColor(finimg,cv.COLOR_BGR2GRAY)
        ret1, binary1 = cv.threshold(gray1,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
        kernel1 = cv.getStructuringElement(cv.MORPH_RECT,(7,7))
        kernel2 = cv.getStructuringElement(cv.MORPH_RECT,(101,101))
        dst1 = cv.morphologyEx(binary1,cv.MORPH_OPEN,kernel1)
        dst1 = cv.morphologyEx(dst1,cv.MORPH_CLOSE,kernel2)
        cloneImage1,contours1,heriachy1 = cv.findContours(dst1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        cv.namedWindow('img2',cv.WINDOW_NORMAL)
        cv.resizeWindow('img2',500,600)
        cv.imshow('img2',img)
	#提取ROI区域
        for c1 in contours1:
            rect1 = cv.minAreaRect(c1)
            ps1 = cv.boxPoints(rect1)
            line11 = ((ps1[0][0]-ps1[1][0])**2+(ps1[0][1]-ps1[1][1])**2)**0.5
            line21 = ((ps1[0][0]-ps1[3][0])**2+(ps1[0][1]-ps1[3][1])**2)**0.5
            if (line11*line21)<6000:
                continue
            roi = finimg[int(ps1[1][1]):int(ps1[0][1]),int(ps1[1][0]):int(ps1[2][0]),:]
            cv.namedWindow('roi',cv.WINDOW_NORMAL)
            cv.resizeWindow('roi',500,600)
            cv.imshow('roi',roi)
    
            
src = 'example.jpg'
getROI(src)
cv.waitKey(0)
cv.destroyAllWindows()
