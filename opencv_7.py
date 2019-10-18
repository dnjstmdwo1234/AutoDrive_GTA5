import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
 
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from grabscreen import grab_screen
import cv2
 
# This is needed to display the images.
get_ipython().magic('matplotlib inline')
 
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
# ssd_mobilenet_v1_coco_11_06_2017 이라는 모델을 다운로드하는 코드
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
 
 
# 하나의 그래프(노드&엣지로 이루어진 하나의 시스템)를 생성합니다
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    
    # CKPT (저장된 가중치) 파일을 불러온 후 모델을 복원하는 코드
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
 
 
# 라벨, 카테코리 데이터를 불러오는 코드
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
 
 
# image vector를 numpy array로 변경하는 함수
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im.height, im_width, 3)).astype(np.uint8)
 
 
# 위에서 복원한 모델이 있는 그래프에서
with detection_graph.as_default():
    # 세션을 하나 실행한다
    with tf.Session(graph=detection_graph) as sess:
        while True:
            # 아래 2줄이 Object Detection 예제코드에서 수정된 부분이다
            # grab_screen을 사용해서 해당 윈도우를 캡처하는 코드 
            screen = cv2.resize(grab_screen(region=(0,40,800,600)), (800,600))
            image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
 
            # 이미지를 인식해서 box, score, classes를 그려주는 코드
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
 
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
 
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
 
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
 
            # 실제로 이 코드가 box와 label을 visualize해주는 코드인듯하다
            vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
 
            # 콘솔창을 띄운다
            cv2.imshow('pygta-p17', image_np)
 
            # q키를 누르면 종료한다
            if cv2.waitKey(25) & 0xff == ord('q'):
                cv2.destroyAllWindows()
                break


