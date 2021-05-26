import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml') 

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    '''
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)     
    
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
    '''
    
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', frame)
    
    if cv.waitKey(1)== ord('s'):
        cv.imwrite('cvtest_color.jpg', frame)
    if cv.waitKey(1)== ord('q'):
        break

cap.release() 
cv.destroyAllWindows()

img = cv.imread('ai/cvtest_color.jpg')
cv.imshow("cvtest ori",img)
cv.waitKey()

HSV_Min = np.array([0, 0, 0])
HSV_Max = np.array([180, 255, 255])
#定義六個拉桿的最大最小值
def H_Lower(val):
    HSV_Min[0] = val
def H_Upper(val):
    HSV_Max[0] = val
def S_Lower(val):
    HSV_Min[1] = val
def S_Upper(val):
    HSV_Max[1] = val
def V_Lower(val):
    HSV_Min[2] = val
def V_Upper(val):
    HSV_Max[2] = val
cv.namedWindow('HSV_TrackBar')
cv.createTrackbar('H_Lower', 'HSV_TrackBar', 0, 180, H_Lower) 
cv.createTrackbar('H_Upper', 'HSV_TrackBar', 0, 180, H_Upper)
cv.createTrackbar('S_Lower', 'HSV_TrackBar', 0, 255, S_Lower)
cv.createTrackbar('S_Upper', 'HSV_TrackBar', 0, 255, S_Upper)
cv.createTrackbar('V_Lower', 'HSV_TrackBar', 0, 255, V_Lower)
cv.createTrackbar('V_Upper', 'HSV_TrackBar', 0, 255, V_Upper)
#主程式
#cnt=0
while True:
    #先將原圖檔(彩色BGR)轉成HSV色彩空間
    hsv_key = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    #套用拉桿上的數值變化到HSV圖檔和原圖擋上
    hsv_result = cv.inRange(hsv_key, HSV_Min, HSV_Max) 
    hsvMask_output = cv.bitwise_and(img, img, None, mask =  hsv_result)
    
    #將圖檔顯示在 'HSV_TrackBar' 視窗並將原圖檔一併顯示出來做比較
    cv.imshow('HSV_TrackBar', hsv_result)
    cv.imshow('HSV_mask_result',hsvMask_output)
    
    #定義一個按鍵(這邊使用'esc')結束視窗
    if cv.waitKey(1) == 27:
        break
cv.destroyAllWindows()
