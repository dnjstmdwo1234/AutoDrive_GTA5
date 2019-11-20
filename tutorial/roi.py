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
    if(image1_color > 50 & image1_color<160):
          return 1
    else:
          if(image2_color > 50 & image2_color<160):
                return 2
          else:
                return 3
            
def image_avg_color(image):
       shape = image.shape
   shape1 = shape[0]*shape[1]
   image = image.reshape(shape1)
   sum = 0
   for i in range(0,shape1):
        sum += image[i]
   avg = int(sum / shape1)
   return avg