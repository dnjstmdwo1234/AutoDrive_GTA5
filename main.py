import numpy as np
from PIL import ImageGrab
from sklearn.cluster import KMeans
from directkeys import PressKey, ReleaseKey, W,A,S,D
import cv2
import time
from road_detection import control_car, process_img_1, process_img_2, image_avg_color, compare_1_2_3, control_car

num = 0
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
    
    if(num==0):
        count = 0
        control_num = compare_1_2_3(image1_color,image2_color,image3_color)
        control_car(control_num)
        control_last_num = control_num
        print(control_last_num)
        print(control_num)
        num += 1
    else:
        control_num = compare_1_2_3(image1_color,image2_color,image3_color)
        print(control_last_num)
        print(control_num)
        if(control_num == control_last_num):
            count += 1
            print(count)
        else:
            count = 0
        if(count==3):
            control_car(4)
        else:
            control_car(control_num)
        control_last_num = control_num
        
    
            
    
    
    
    
    
    cv2.imshow('main', new_screen)
    #cv2.imshow('screen1', new_screen1)
    #cv2.imshow('screen2', new_screen2)
    #cv2.imshow('screen3', new_screen3)
 
    # 'q'키를 누르면 종료합니다
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break