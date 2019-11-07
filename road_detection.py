import numpy as np
from PIL import ImageGrab
from sklearn.cluster import KMeans
from directkeys import PressKey, ReleaseKey, W,A,S,D
import cv2
import time


# 무한루프를 돌면서 
# Region of Interest : 관심영역을 설정하는 함수
def roi(img, vertices):
    # img 크기만큼의 영행렬을 mask 변수에 저장하고
    mask = np.zeros_like(img)
 
    # vertices 영역만큼의 Polygon 형상에만 255의 값을 넣습니다
    masked = cv2.fillPoly(mask, vertices, 255)
 
    # img와 mask 변수를 and (비트연산) 해서 나온 값들을 masked에 넣고 반환합니다
    masked = cv2.bitwise_and(img, masked)
    return masked


# 이미지에 각종 영상처리를 하는 함수
def process_img_1(image):
    original_image = image
    
    # convert to gray
 
    blue_threshold = 120
    green_threshold = 120
    red_threshold = 120
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]
    thresholds = (image[:,:,0] > bgr_threshold[0]) \
                | (image[:,:,1] > bgr_threshold[1]) \
                | (image[:,:,2] > bgr_threshold[2])
    mark[thresholds] = [255,255,255]
    processed_img = cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY)
    
    blue_threshold = 50
    green_threshold = 50
    red_threshold = 50
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]
    thresholds = (image[:,:,0] < bgr_threshold[0]) \
                | (image[:,:,1] < bgr_threshold[1]) \
                | (image[:,:,2] < bgr_threshold[2])
    mark[thresholds] = [255,255,255]
    processed_img = cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY)
    # edge detection
    #processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    #processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
 
    # 원하는 영역을
    vertices =  np.array([[100,400], [100,300], [300,250],[500,250], [700,300], [700,400]], np.int32)

    
    # roi()를 사용해 그 영역만큼 영상을 자릅니다
    processed_img = roi(processed_img, [vertices])
    
    
    # BGR 제한 값보다 작으면 검은색으로
    
    
    return processed_img, original_image

def process_img_2(image):
    original_image = image
    
    # convert to gray
 
    blue_threshold = 120
    green_threshold = 120
    red_threshold = 120
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]
    thresholds = (image[:,:,0] > bgr_threshold[0]) \
                | (image[:,:,1] > bgr_threshold[1]) \
                | (image[:,:,2] > bgr_threshold[2])
    mark[thresholds] = [255,255,255]
    processed_img = cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY)
    
    blue_threshold = 50
    green_threshold = 50
    red_threshold = 50
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]
    thresholds = (image[:,:,0] < bgr_threshold[0]) \
                | (image[:,:,1] < bgr_threshold[1]) \
                | (image[:,:,2] < bgr_threshold[2])
    mark[thresholds] = [255,255,255]
    processed_img = cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY)
    # edge detection
    #processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    #processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)

    
    
    # BGR 제한 값보다 작으면 검은색으로
    
    
    return processed_img, original_image

def image_avg_color(image):
   shape = image.shape
   shape1 = shape[0]*shape[1]
   image = image.reshape(shape1)
   sum = 0
   for i in range(0,shape1):
        sum += image[i]
   avg = int(sum / shape1)
   return avg

def compare_1_2_3(image1_color,image2_color,image3_color):
    #if((image1_color > 200) & (image2_color > 200) & (image3_color > 200)):
    #    return 4
    
    
    if(image1_color < image2_color):
        if(image1_color < image3_color):
            print("직진")
            return 1
        else:
            print("우회전")
            return 3
    else:
        if(image2_color < image3_color):
            print("좌회전")
            return 2
        else:
            print("우회전")
            return 3   

def control_car(num):
    if(num==1):
        PressKey(W)
        time.sleep(3)
        ReleaseKey(W)
        
        
    elif(num==2):
        PressKey(A)
        PressKey(W)
        time.sleep(1)
        ReleaseKey(A)
        ReleaseKey(W)
        
        

    elif(num==3):
        PressKey(D)
        PressKey(W)
        time.sleep(1)
        ReleaseKey(D)
        ReleaseKey(W)
        
        
    else:
        PressKey(S)
        time.sleep(3)
        

       
while(True):
    # (0,40)부터 (800,600)좌표까지 창을 만들어서 데이터를 저장하고 screen 변수에 저장합니다
    image = np.array(ImageGrab.grab(bbox=(0,40,800,600)))
    mark = np.copy(image) # image 복사
    
    new_screen, original_image = process_img_1(image)
    
    image1 = np.array(ImageGrab.grab(bbox=(300,275,500,325)))
    mark = np.copy(image1) # image 복사
    
    new_screen1, original_image1 = process_img_2(image1)
    

    image2 = np.array(ImageGrab.grab(bbox=(200,275,300,300)))
    mark = np.copy(image2) # image 복사
    
    new_screen2, original_image2 = process_img_2(image2)
    
    image3 = np.array(ImageGrab.grab(bbox=(500,275,600,300)))
    mark = np.copy(image3) # image 복사

    new_screen3, original_image3 = process_img_2(image3)

   
    image1_color = image_avg_color(new_screen1)
    image2_color = image_avg_color(new_screen2)
    image3_color = image_avg_color(new_screen3)
        
    control_num = compare_1_2_3(image1_color,image2_color,image3_color)
    
    control_car(control_num)
    
    
    cv2.imshow('main', new_screen)
    #cv2.imshow('screen1', new_screen1)
    #cv2.imshow('screen2', new_screen2)
    #cv2.imshow('screen3', new_screen3)
 
    # 'q'키를 누르면 종료합니다
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break