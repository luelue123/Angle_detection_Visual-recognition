# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import math
import cv2          
import imutils         
import numpy as np
              



num=2     
# the video path 
videopath = "D:/dl/2022-12-12/0.6 mm 40 kPa.MOV"
# the data saved path
dataname =   "D:/dl/2022-12-12/0.6 mm 40 kPa_"+str(num)+".txt"                                                

## get the color 
## "BLUE"   
def get_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lowerb=np.array([30, 60, 100]),upperb=np.array([250, 250, 255]))# 3
    return mask


# calculate the actuator angle
def get_angle(image):
    angle=None
    ret, binary = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    N=len(contours)
 
    contours_merge = contours[0]
    if N>1:
        for i in range(1,len(contours)):
            if len(contours[i])>10:
                contours_merge = np.vstack([contours_merge,contours[i]])
# find the min rectangle
    rect = cv2.minAreaRect(contours_merge)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    angle=int(rect[2])
    center=rect[0]
    return box,center,angle


fps = None
vs = cv2.VideoCapture(videopath)
FrameNumber = vs.get(7) 
angle_list=list([])
print(FrameNumber)
n=1
split=False
while vs.isOpened():
    n+=1              
    _,frame = vs.read()

    cv2.namedWindow('frame0')
    cv2.imshow("frame0", frame)
    try:
        mask=get_color(frame)
        box,center,angle=get_angle(mask)
    

        # draw the angle and the point
        frame = cv2.drawContours(frame, [box], 0, (0, 0, 255), 3)
        frame = cv2.line(frame ,box[0],box[1],(0,255,0),3)
        
        cv2.circle(frame,box[0], 3, (255,255,0), 4) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(angle), (int(center[0])+10,int(center[1])-10), font, 1, (255, 255, 255), 2) 
        angle_list.append(angle)
    except:
        angle_list.append(100)
        print("No")



    cv2.namedWindow('frame')
    cv2.imshow("frame", frame)

    cv2.namedWindow('mask')
    cv2.imshow("mask", mask)
    key = cv2.waitKey(1) & 0xFF 
    if 0xFF == ord('q') or n>=FrameNumber:
        print(FrameNumber)
        break              

print(np.shape(angle_list)) 
np.savetxt(dataname,angle_list)
cv2.destroyAllWindows()
