# coding: utf-8
# # Object Detection Demo
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
# source: https://github.com/tensorflow/models
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import time
import zipfile
from PIL import ImageGrab
from directkeys import PressKey, ReleaseKey, W,A,S,D
from PIL import ImageGrab
from sklearn.cluster import KMeans

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from grabscreen import grab_screen
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util
from utils import visualization_utils as vis_util


# # Model preparation 
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

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
 
    blue_threshold = 160
    green_threshold = 160
    red_threshold = 160
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
    vertices =  np.array([[100,500], [100,200], [300,100],[500,100], [700,200], [700,500]], np.int32)

    
    # roi()를 사용해 그 영역만큼 영상을 자릅니다
    processed_img = roi(processed_img, [vertices])
    
    
    # BGR 제한 값보다 작으면 검은색으로
    
    
    return processed_img, original_image

def process_img_2(image):
    original_image = image
    
    # convert to gray
 
    blue_threshold = 160
    green_threshold = 160
    red_threshold = 160
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]
    thresholds = (image[:,:,0] > bgr_threshold[0]) \
                & (image[:,:,1] > bgr_threshold[1]) \
                & (image[:,:,2] > bgr_threshold[2])
    mark[thresholds] = [255,255,255]
    processed_img = cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY)
    
    blue_threshold = 50
    green_threshold = 50
    red_threshold = 50
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]
    thresholds = (image[:,:,0] < bgr_threshold[0]) \
                & (image[:,:,1] < bgr_threshold[1]) \
                & (image[:,:,2] < bgr_threshold[2])
    mark[thresholds] = [255,255,255]
    processed_img = cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY)
    # edge detection
    #processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    #processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)

    
    
    # BGR 제한 값보다 작으면 검은색으로
    
    
    return processed_img, original_image
def compare_1_2_3(image1_color,image2_color,image3_color):
    #if((image1_color > 200) & (image2_color > 200) & (image3_color > 200)):
    #    return 4
    
    
    if(image1_color < image2_color):
        if(image1_color > image3_color):
            return 1
        else:
            return 3
    else:
        if(image2_color < image3_color):
            
            return 3
        else:
            
            return 2
def image_avg_color(image):
   shape = image.shape
   shape1 = shape[0]*shape[1]
   image = image.reshape(shape1)
   sum = 0
   for i in range(0,shape1):
        sum += image[i]
   avg = int(sum / shape1)
   return avg
# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def roi(img, vertices):
    # img 크기만큼의 영행렬을 mask 변수에 저장하고
    mask = np.zeros_like(img)
 
    # vertices 영역만큼의 Polygon 형상에만 255의 값을 넣습니다
    masked = cv2.fillPoly(mask, vertices, 255)
 
    # img와 mask 변수를 and (비트연산) 해서 나온 값들을 masked에 넣고 반환합니다
    masked = cv2.bitwise_and(img, masked)
    return masked
  
def process_img(image):
    original_image = image
    
    # convert to gray
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
 
    # 원하는 영역을 만들고
    vertices =  np.array([[10,500], [10,300], [300,200],[500,200], [800,300], [800,500]], np.int32)
 
    # roi()를 사용해 그 영역만큼 영상을 자릅니다
    processed_img = roi(processed_img, [vertices])
    return processed_img
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def control_car(num):
    if(num==1):
        PressKey(W)
        time.sleep(1)
        ReleaseKey(W)
        print("직진")
        
    elif(num==2):
        PressKey(A)
        PressKey(W)
        time.sleep(1)
        ReleaseKey(A)
        PressKey(W)
        time.sleep(1)
        ReleaseKey(W)
        print("좌회전")
        

    elif(num==3):
        PressKey(D)
        PressKey(W)
        time.sleep(1)
        ReleaseKey(D)
        PressKey(W)
        time.sleep(1)
        ReleaseKey(W)
        print("우회전")
        
        
    else:
        PressKey(S)
        time.sleep(1)
        ReleaseKey(S)
        PressKey(W)
        time.sleep(1)
        ReleaseKey(W)
        
        print("후진")

num = 0       
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      image = np.array(ImageGrab.grab(bbox=(0,40,800,600)))
      mark = np.copy(image) # image 복사
      
      new_screen, original_image = process_img_1(image)
      
      image1 = np.array(ImageGrab.grab(bbox=(300,300,500,350)))
      mark = np.copy(image1) # image 복사
      
      new_screen1, original_image1 = process_img_2(image1)
      

      image2 = np.array(ImageGrab.grab(bbox=(200,300,300,350)))
      mark = np.copy(image2) # image 복사
      
      new_screen2, original_image2 = process_img_2(image2)
      
      image3 = np.array(ImageGrab.grab(bbox=(500,300,600,350)))
      mark = np.copy(image3) # image 복사

      new_screen3, original_image3 = process_img_2(image3)


      image1_color = image_avg_color(new_screen1)
      image2_color = image_avg_color(new_screen2)
      image3_color = image_avg_color(new_screen3)
      #screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (WIDTH,HEIGHT))
      screen = image = np.array(ImageGrab.grab(bbox=(200,200,600,350)))
      #vertices =  np.array([[100,500], [100,200], [300,100],[500,100], [700,200], [700,500]], np.int32)
      #image_np = process_img(screen)
      image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      a = []
      min_i = 0
      for i,b in enumerate(boxes[0]):
        #                 car                    bus                  truck
        if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
          if scores[0][i] >= 0.5:
            apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
            a.append(apx_distance)
            print(apx_distance)
                  
                
      for i in range(0,len(a)):
        if(i==0):
          min = a[i]
          min_i = i
        else:
          if(min > a[i]):
                
            min = a[i]
            min_i = i
            
      if scores[0][min_i] >= 0.5:
        print(len(a)+1,"대 차가 식별됨, 그 중 대표 차 1대 추출")      
        print("score : ", scores[0][min_i])
        mid_x = (boxes[0][min_i][1]+boxes[0][min_i][3])/2
        print("mid_x : ", mid_x)
        mid_y = (boxes[0][min_i][0]+boxes[0][min_i][2])/2
        print("mid_y",mid_y)
        apx_distance = round(((1 - (boxes[0][min_i][3] - boxes[0][min_i][1]))**4),1)
        a.append(apx_distance)
                
        print("apx_distance : ", apx_distance)
        #cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        if apx_distance >=0.5:
              if mid_x > 0.2 and mid_x < 0.8:
                control_car(1)
              elif mid_x <= 0.2:
                control_car(2)
              elif mid_x >= 0.8:
                control_car(3)
              else:
                print("a")
        elif apx_distance <= 0.1:
              control_car(4)
        else:
              print("b") 
      else:
        print("차량이 식별되지 않았습니다. 도로 탐색기반으로 동작합니다.")
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
                    
                  

      cv2.imshow('window',cv2.resize(image_np,(800,450)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break 
        