
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CompressedImage
# from ultralytics import YOLO
import numpy as np
import os
import cv2
from cv_bridge import CvBridge
from datetime import datetime
from sonia_common_ros2.msg import DetectionArray, Detection
from .colorDictionary import *

if os.path.exists('/home/sonia/ssd/output_ai/'):
    OUTPUT_DIR = '/home/sonia/ssd/output_ai/'
else:
    OUTPUT_DIR = '/home/sonia/output_ai/'
SAVE_OUTPUT = False

AREA_OF_SEE = 2

class NodeTosee(Node):

    def __init__(self):
        super().__init__('NodeTosee') 
        self.__front_cam_sub = self.create_subscription(CompressedImage, "zed/zed_node/left/image_rect_color/compressed", self.__img_front_callback, 10)
        self.__bottom_cam_sub = self.create_subscription(CompressedImage, "camera_array/bottom/image_raw/compressed", self.__img_bottom_callback, 10)
        self.__classif_front_sub = self.create_subscription(DetectionArray, "proc_vision/front/classif", self.__create_magic_front, 10)
        self.__classif_bottom_sub = self.create_subscription(DetectionArray, "proc_vision/bottom/classif", self.__create_magic_bottom, 10)
        self.__classif_front_pub = self.create_publisher(Image, "proc_vision/front/image", 10)
        self.__classif_bottom_pub = self.create_publisher(Image, "proc_vision/bottom/image", 10)
        self.__bridge = CvBridge()
        self.__imageFront = None
        self.__imageBottom = None
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

    def __img_front_callback(self, msg: CompressedImage):
        self.__imageFront= cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)

    def __img_bottom_callback(self, msg: Image):
        self.__imageBottom = np.frombuffer(msg.data, np.uint8)

    def __create_magic_bottom(self, msg: DetectionArray):
        if self.__imageBottom is not None:
            image = self.__imageBottom
            for res in msg.detected_object:
                cv2.putText(image, 
                            res.class_name, 
                            (int((res.top_left_x+5)),
                            int((res.bottom_right_y-10)/2)), 
                            cv2.FONT_HERSHEY_PLAIN, 
                            3, (0,0,0), 1, 1)
                cv2.putText(image, 
                            "{:.1f}%".format(res.confidence), 
                            (int((res.top_left_x+5)),
                            int((res.bottom_right_y+10)/2)), 
                            cv2.FONT_HERSHEY_PLAIN, 
                            3, (0,0,0), 1, 1)
                cv2.rectangle(image, 
                            (int(res.top_left_x),int(res.top_left_y)), 
                            (int(res.bottom_right_x),int(res.bottom_right_y)), 
                            colorDictionary[res.class_name], 1)
            if SAVE_OUTPUT:
                cv2.imwrite(OUTPUT_DIR+'pred_bottom_'+str(int(datetime.now().time().microsecond//1000))+'.jpg', 
                            image) 
            self.__classif_bottom_pub.publish(self.__bridge.cv2_to_imgmsg(image,'bgr8'))

    def __create_magic_front(self, msg: DetectionArray):
    	if self.__imageFront is not None:
            image = self.__imageFront
            for res in msg.detected_object:
                cv2.putText(image, 
                            res.class_name, 
                            (int((res.top_left_x+5)),
                            int((res.bottom_right_y-30)/2)), 
                            cv2.FONT_HERSHEY_PLAIN, 
                            1.5, (255,255,255), 3, 1)
                cv2.putText(image, 
                            "{:.1f}%".format(res.confidence), 
                            (int((res.top_left_x+5)),
                            int((res.bottom_right_y+30)/2)), 
                            cv2.FONT_HERSHEY_PLAIN, 
                            1.5, (255,255,255), 3, 1)
                cv2.rectangle(image, 
                            (int(res.top_left_x),int(res.top_left_y)), 
                            (int(res.bottom_right_x),int(res.bottom_right_y)), 
                            colorDictionary[res.class_name], 1)
            if SAVE_OUTPUT:
                cv2.imwrite(OUTPUT_DIR+'pred_front_'+str(int(datetime.now().time().microsecond//1000))+'.jpg', 
                    image)
            self.__classif_front_pub.publish(self.__bridge.cv2_to_imgmsg(image,'bgr8'))
