
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CompressedImage
# from ultralytics import YOLO
import numpy as np
import os
import cv2
from datetime import datetime
from sonia_common_ros2.msg import DetectionArray, Detection

if os.path.exists('/home/sonia/ssd/output_ai/'):
    OUTPUT_DIR = '/home/sonia/ssd/output_ai/'
else:
    OUTPUT_DIR = '/home/sonia/output_ai/'
SAVE_OUTPUT = False

AREA_OF_SEE = 2

class NodeTosee(Node):

    def __init__(self):
        self.declare_parameter("models", Parameter.Type.STRING_ARRAY)
        self.__front_cam_sub = self.create_subscription(CompressedImage, "camera_array/front/image_raw/compressed", self.__img_front_callback, 10)
        self.__bottom_cam_sub = self.create_subscription(Image, "camera_array/bottom/image_raw", self.__img_bottom_callback, 10)
        self.__classif_front_sub = self.create_subscription(DetectionArray, "proc_vision/front/classif", self.__create_magic_front, 10)
        self.__classif_bottom_sub = self.create_subscription(DetectionArray, "proc_vision/bottom/classif", self.__create_magic_bottom, 10)
        self.__zed_depth = self.create_subscription(DetectionArray, "zed/zed_node/point_cloud/findWHAT/compressed", self.__get_depth, 10)
        self.__classif_front_pub = self.create_publisher(Image, "proc_vision/front/image", 10)
        self.__classif_bottom_pub = self.create_publisher(Image, "proc_vision/bottom/image", 10)
        self.__bottom_list = []
        self.__front_list = []
        self.__deep_list = []
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

    def __img_front_callback(self, msg: CompressedImage):
        self.__front_list.append(cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR))

    def __img_bottom_callback(self, msg: Image):
        self.__bottom_list.append(np.frombuffer(msg.data, np.uint8))

    def __create_magic_bottom(self, msg: DetectionArray):
        image = self.__bottom_list[-1]
        self.__bottom_list = [image]
        for res in msg.detected_object:
            cv2.putText(image, 
                        res.class_name, 
                        (int((res.top_left_x+5)),
                        int((res.bottom_right_y-10)/2)), 
                        cv2.FONT_HERSHEY_PLAIN, 
                        .7, (0,0,255), 1, 1)
            cv2.putText(image, 
                        "{:.1f}%".format(res.confidence), 
                        (int((res.top_left_x+5)),
                        int((res.bottom_right_y+10)/2)), 
                        cv2.FONT_HERSHEY_PLAIN, 
                        .7, (0,0,255), 1, 1)
            cv2.rectangle(image, 
                        (int(res.top_left_x),int(res.top_left_y)), 
                        (int(res.bottom_right_x),int(res.bottom_right_y)), 
                        (0,0,255), 1)
        if SAVE_OUTPUT:
            cv2.imwrite(OUTPUT_DIR+'pred_bottom_'+str(int(1000*datetime.now().time()))+'.jpg', 
                        image) 
        self.__classif_bottom_pub.publish(image)

    def __create_magic_front(self, msg: DetectionArray):
        image = self.__front_list[-1]
        self.__front_list = [image]
        for res in msg.detected_object:
            cv2.putText(image, 
                        res.class_name, 
                        (int((res.top_left_x+5)),
                        int((res.bottom_right_y-10)/2)), 
                        cv2.FONT_HERSHEY_PLAIN, 
                        .7, (0,0,255), 1, 1)
            cv2.putText(image, 
                        "{:.1f}%".format(res.confidence), 
                        (int((res.top_left_x+5)),
                        int((res.bottom_right_y+10)/2)), 
                        cv2.FONT_HERSHEY_PLAIN, 
                        .7, (0,0,255), 1, 1)
            cv2.rectangle(image, 
                        (int(res.top_left_x),int(res.top_left_y)), 
                        (int(res.bottom_right_x),int(res.bottom_right_y)), 
                        (0,0,255), 1)
        if SAVE_OUTPUT:
            cv2.imwrite(OUTPUT_DIR+'pred_front_'+str(int(1000*datetime.now().time()))+'.jpg', 
                        image) 
        self.__classif_front_pub.publish(image)

    def __get_depth(self, msg: CompressedImage):
        self.__deep_list.append(cv2.imdecode(np.frombuffer(msg.data, np.uint8),0))
        cv2.imwrite(OUTPUT_DIR+'depth_front_'+str(int(1000*datetime.now().time()))+'.jpg', 
                    self.__deep_list[-1])
        
    def get_deep(self, x,y,w,h) -> int:
        sum = 0
        number +=1
        resized_image = cv2.resize(self.__deep_list[-1], (w, h))
        if(x<AREA_OF_SEE):
            xStart = 0
        else:
            xStart = x- AREA_OF_SEE
        if(y<AREA_OF_SEE):
            yStart = 0
        else:
            yStart = y- AREA_OF_SEE
        for i in range(AREA_OF_SEE*2+1) and x+i< w:
            for j in range(AREA_OF_SEE*2+1) and y+j<h:
                sum += resized_image[yStart+j][xStart+i]
                number+=1
        return sum/number
