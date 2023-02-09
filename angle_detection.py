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



# below is the plot code
############################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

end0=np.array([1011,1490,1791,1983,2525,969,2218,1519])
split_list=np.array([804,1264,1567,1795,2335,773,1109,801])
start_list=end0-split_list

 
num=2
filepath="D:/dl/2212129new/pm221129/"
file = filepath+"angle_"+str(num)+".txt"
angle0=np.loadtxt(file )

def preprossion_angle(angle_list,num):
    angle_new=np.zeros((5,split_list[num-1]))
    N=len(angle_list)
    for i in range(1,N-1):
        if angle_list[i] > 100 or angle_list[i] <= 0 :
            angle_list[i]=angle_list[i-1]
        if abs(angle_list[i]-angle_list[i-1]) > 10 and abs(angle_list[i+1]-angle_list[i]) > 10 :
            angle_list[i]=int((angle_list[i+1]+angle_list[i-1])*0.5)
    for j in range(5):

        start=start_list[num-1]+split_list[num-1]*j
        end=start+split_list[num-1]
        print(np.shape(angle_new))
        angle_new[j,:]=angle_list[start:end]

        #Find the index of the angle suddendly changes
        lenG=len(angle_new[j,:])
        angle90_num=list([])
        for k in range(int(lenG*0.4),lenG):
            if angle_new[j,k-1]<20 and angle_new[j,k]>80:
                angle90_num.append(k)
        angle90_num.append(lenG)
        print(angle90_num)

        # Piece together the angles
        for item in range(len(angle90_num)-1):
            angle_new[j,angle90_num[item]:angle90_num[item+1]] = angle_new[j,angle90_num[item]:angle90_num[item+1]]-90*(item+1)
    return angle_new

def get_pressure(angle_list):
    # 1 kpa 283 ms
    # 3000ms start
    # fps 50
    N=len(angle_list)
    maxpressure=int((N/50-3)/0.283)
    pressure=np.hstack((np.zeros(10),np.linspace(0,maxpressure,maxpressure+1)))
    index=np.int0(pressure*N/len(pressure))
    angle=angle_list[index]
    angle=90-angle
    P=pressure
    return P,angle



def get_meam_error(x,y):
    meanx=np.mean(x,axis=0)
    meany=np.mean(y,axis=0)
    errorx=np.row_stack((meanx-np.min(x,axis=0),np.max(x,axis=0)-meanx))
    errory=np.row_stack((meany-np.min(y,axis=0),np.max(y,axis=0)-meany))
    return meanx, meany, errorx, errory

data=np.array([])
plt.plot(angle0)
a=plt.gca()
plt.legend(loc='upper left')
plt.xlabel('Pressure (KPa)')
plt.ylabel('Angle (degree)')
plt.show()




import pandas as pd

data = pd.DataFrame(data.T)
writer = pd.ExcelWriter(filepath+"angle_"+str(num)+".xlsx")		# write to excel
data.to_excel(writer, 'Sheet1', float_format='%.3f')		
writer.save()
writer.close()





