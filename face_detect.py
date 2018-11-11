# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 21:25:12 2018

@author: shobhit
"""

import cv2
import numpy as np

face_cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cas=cv2.CascadeClassifier('haarcascade_eye.xml')

cap=cv2.VideoCapture(0)
l_avg=(0,0)
r_avg=(0,0)
pl_avg=(0,0)
pr_avg=(0,0)
loop_r=0
loop_l=0


while True:
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    face=face_cas.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        
        eye=eye_cas.detectMultiScale(roi_gray)
        mid=(x+(int(((x+w)-x)/2)))
        cv2.line(img,(mid,y),(mid,y+h),(0,255,0),2)
        cv2.line(img,(x,y+int(h*0.7)),(x+w,y+int(h*0.7)),(0,255,0),2)
        cv2.line(img,(x,y+int(h*0.1)),(x+w,y+int(h*0.1)),(0,255,0),2)
        
            
        
        for (ex,ey,ew,eh) in eye:
            
            if (x+ex)>mid & (y+ey+int(eh/2))>(y+int(h*0.1)) & (y+ey+int(eh/2))<(y+int(h*0.7)):
                r_eye_roi_gray=roi_gray[ey:ey+eh,ex:ex+ew]
                r_eye_roi_color=roi_color[ey:ey+eh,ex:ex+ew]
                #take avg here 
                g_gray = cv2.GaussianBlur(r_eye_roi_gray, (15,15), 0)
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g_gray)
                #avg of blue pos
                r_avg=(minLoc[0]+r_avg[0],minLoc[1]+r_avg[1])
                loop_r=loop_r+1
                
                if(loop_r==10):
                    temp_x=int(r_avg[0]/10)
                    temp_y=int(r_avg[1]/10)
                    pr_avg=(temp_x,temp_y)                    
                    r_avg=(0,0)
                    loop_r=0
                    cv2.line(img,(x,y+int(h*0.5)),(x+w,y+int(h*0.5)),(0,255,0),2)
        
                    
                cv2.circle(r_eye_roi_color, pr_avg, 5, (0, 255, 0), 5)
                
                cv2.imshow('left',r_eye_roi_color)
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
            
                
            
            if (x+ex)<mid & (y+ey+int(eh/2))>(y+int(h*0.1)) & (y+ey+int(eh/2))<(y+int(h*0.7)):
                l_eye_roi_gray=roi_gray[ey:ey+eh,ex:ex+ew]
                l_eye_roi_color=roi_color[ey:ey+eh,ex:ex+ew]
                
                g_gray = cv2.GaussianBlur(l_eye_roi_gray, (15,15), 0)
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g_gray)
                
                l_avg=(minLoc[0]+l_avg[0],minLoc[1]+l_avg[1])
                loop_l=loop_l+1
                
                if(loop_l==10):
                    temp_x=int(l_avg[0]/10)
                    temp_y=int(l_avg[1]/10)
                    pl_avg=(temp_x,temp_y)                    
                    l_avg=(0,0)
                    loop_l=0
                
                cv2.circle(l_eye_roi_color, pl_avg, 5, (0, 255, 0), 5)
                #circle
                circle = cv2.HoughCircles(l_eye_roi_gray, cv2.HOUGH_GRADIENT, 1, 20,param1=50,param2=30,minRadius=5,maxRadius=10)
                if circle is not None:
                    circle=np.round(circle[0,:]).astype("int")
                    for (cx, cy, cr) in circle:
                        cv2.circle(l_eye_roi_gray, (cx, cy), cr, (0, 255, 0), 2)
    
                #---
    
                
                cv2.imshow('right',l_eye_roi_color)
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
            
                
                    
            
    
    
            
            
    
    cv2.imshow('Face Detect',img)
    
    if cv2.waitKey(1) & 0xFF==ord('f'):
        break
    
cap.release()
cv2.destroyAllWindows()
