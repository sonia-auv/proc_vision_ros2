import sys
sys.path.append("/home/sonia/ssd/pip_pkg")

from rclpy.node import Node
from typing import Tuple
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import os
from cv_bridge import CvBridge
import cv2
from .yolov8 import YOLOv8
from sonia_common_ros2.msg import DetectionArray, Detection
from sonia_common_ros2.srv import AiActivationService
import traceback
if os.path.exists('/home/sonia/ssd/ros2_sonia_ws/src/proc_vision_ros2/models/'):
    MODEL_DIR = '/home/sonia/ssd/ros2_sonia_ws/src/proc_vision_ros2/models/'
else:
    MODEL_DIR = '/home/sonia/ros2_sonia_ws/src/proc_vision_ros2/models/'

if os.path.exists('/home/sonia/ssd/output_ai/'):
    OUTPUT_DIR = '/home/sonia/ssd/output_ai/'
else:
    OUTPUT_DIR = '/home/sonia/output_ai/'
SAVE_OUTPUT = False

NUMBER_DETECTION = 250

class VisionNode(Node):

    def __init__(self):
        super().__init__("vision_node")
        
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.camera_front = False
        self.camera_bottom = False
        self.declare_parameter("models", Parameter.Type.STRING_ARRAY) 
        self.__ai_activation_sub = self.create_service(AiActivationService, "proc_vision/ai_activation", self.__ai_activation_callback)

        self.__front_cam_sub = self.create_subscription(CompressedImage, "zed/zed_node/left/image_rect_color/compressed", self.__img_front_callback, 10)
        self.__front_cam_sim = self.create_subscription(CompressedImage, "proc_simulation/front/compressed", self.__img_front_callback, 10)
        
        self.__bottom_cam_sub = self.create_subscription(CompressedImage, "camera_array/bottom/image_raw/compressed", self.__img_bottom_callback, qos)
        self.__bottom_cam_sim = self.create_subscription(CompressedImage, "proc_simulation/bottom/compressed", self.__img_bottom_callback, 10)

        model_front_name = self.get_parameter("models").get_parameter_value().string_array_value[0]
        model_bottom_name = self.get_parameter("models").get_parameter_value().string_array_value[0]
        self.model_front = YOLOv8(os.path.join(MODEL_DIR, model_front_name), self)
        self.model_bottom = YOLOv8(os.path.join(MODEL_DIR, model_bottom_name), self)
        self.__classif_front_pub = self.create_publisher(DetectionArray, "proc_vision/front/classif", 10)
        self.__classif_bottom_pub = self.create_publisher(DetectionArray, "proc_vision/bottom/classif", 10)

        #Deep
        self.__zed_depth = self.create_subscription(Image, "zed/zed_node/depth/depth_registered", self.__get_depth, 10)
        self.__actual = None
        self.__deep_last = None

        self.br = CvBridge()

        if SAVE_OUTPUT:
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
        self.get_logger().info("Vision node initialized")
        self.get_logger().info(f"Available providers: {self.model_front.available_providers()}")


    def __ai_activation_callback(self, request, response):
        model_list = self.get_parameter("models").get_parameter_value().string_array_value
        if request.model_choice >= 0 and request.model_choice < len(model_list):
            model_name = model_list[request.model_choice]
        else:
            model_name = model_list[0]

        if request.camera_choice == AiActivationService.Request.FRONT:
            self.camera_front = True
            self.camera_bottom = False
            self.model_front = YOLOv8(os.path.join(MODEL_DIR, model_name), self)
        elif request.camera_choice == AiActivationService.Request.BOTTOM:
            self.camera_front = False
            self.camera_bottom = True
            self.model_bottom = YOLOv8(os.path.join(MODEL_DIR, model_name), self)
        elif request.camera_choice == AiActivationService.Request.BOTH:
            self.camera_front = True
            self.camera_bottom = True
            self.model_front = YOLOv8(os.path.join(MODEL_DIR, model_name), self)
            self.model_bottom = YOLOv8(os.path.join(MODEL_DIR, model_name), self)
        else:
            self.camera_front = False
            self.camera_bottom = False
        
        if self.camera_front:
            self.get_logger().info(f"Front ON -> Model : {model_name}")
        else:
            self.get_logger().info(f"Front OFF")

        if self.camera_bottom:
            self.get_logger().info(f"Bottom ON -> Model : {model_name}")
        else:
            self.get_logger().info(f"Bottom OFF")

        return response

    def __img_front_callback(self, msg: CompressedImage):
        if self.camera_front:
            self.get_logger().info(f"Image front {msg.header.frame_id} received!!")
            self.__classif_front_pub.publish(self.__img_detection(msg, self.model_front))

    def __img_bottom_callback(self, msg: Image):
        if self.camera_bottom:
            self.get_logger().info("Image Bottom received!!")
            self.__classif_bottom_pub.publish(self.__img_detection(msg, self.model_bottom))

    def __img_detection(self, msg: CompressedImage, model: YOLOv8) -> DetectionArray:
        try:
            self.get_logger().info(f"start")
            self.actualise_deep()
            detections = model.detect(cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR), msg.header.frame_id)
            if(self.camera_front):
                for detect in detections.detected_object:
                    # self.get_logger().info(str(self.get_deep_istogram(int(detect.top_left_x), int(detect.bottom_right_x), int(detect.top_left_y) ,int(detect.bottom_right_y))))
                    detect.distance = self.get_deep_istogram(int(detect.top_left_x), int(detect.bottom_right_x), int(detect.top_left_y) ,int(detect.bottom_right_y))
            return detections
        except Exception as e:
            self.get_logger().info(f"Vision node failure :{e}")
            self.get_logger().info(f"Vision node failure trace :{traceback.format_exc()}")
            detections = DetectionArray()
            detections.detected_object = []
            return detections


    def print_results(self, img, results: DetectionArray):
        res:Detection
        for res in results.detected_object:
            img_res = cv2.rectangle(img, 
                          (int(res.top_left_x), int(res.top_left_y)), 
                          (int(res.bottom_right_x), int(res.bottom_right_y)),
                          (0, 255, 0),
                          1)
            img_res = cv2.putText(img_res, f"{res.class_name} {res.confidence:.2f}", 
                                  (int(res.top_left_x), int(res.top_left_y-10)), 1, 1, (0, 255, 0), 1)
        if len(results.detected_object) > 0:
            cv2.imwrite('/home/sonia/ssd/image_window.jpg', img_res)

    def __get_depth(self, msg: Image) -> None:
        """Function to received the deep image and write

        Args:
            msg (Image): image received
        """
        try:
            self.__deep_last = self.br.imgmsg_to_cv2(msg)
            if (SAVE_OUTPUT):
                cv2.imwrite(OUTPUT_DIR+'image_deep.tif', self.__deep_last)
        except Exception as e:
            self.get_logger().info(str(e))

    def actualise_deep(self) -> None:
        """Function to define the deep to use
        """
        self.__actual = self.__deep_last
        
    def get_deep_istogram(self, x1: int,x2: int,y1: int,y2: int) -> int:
        """function to return the distance of a object on a image

        Args:
            x1 (int): cordoninante in x of top left point
            x2 (int): cordoninante in x of bottom rigth
            y1 (int): cordoninante in y of top left point
            y2 (int): cordoninante in y of bottom rigth

        Returns:
            int: the distance of object
        """
        if(self.__actual is None):
            self.actualise_deep()
            if (self.__actual is None):
                return float(60)
        # resized_image = self.__actual
        # for i in range(min(max(0,x1),1280),min(max(0,x2),1280)):
        #     for j in range(min(max(0,y1),720),min(max(0,y2),720)):
        #         if not resized_image[j,i] in dict_value.keys():
        #             dict_value[resized_image[j,i]]=0
        #         dict_value[resized_image[j,i]]+=1
        elem = self.__actual[min(max(0,x1),1280):min(max(0,x2),1280), min(max(0,y1),720):min(max(0,y2),720)]
        histo = np.histogram(elem,range = (0,65535),bins=1)[0]
        # self.get_logger().info(str(histo))
        # histogram = sorted(dict_value.items())
        # max1 = 65535
        # value_max1 = 0
        # max2 = 65535
        # value_max2 = 0
        # for keys,value in histogram:
        #     if (value) > value_max1:
        #         max2 = max1
        #         value_max2 = value_max1
        #         value_max1 = value
        #         max1 = keys
        #     elif value > value_max2:
        #         max2 = keys
        #         value_max2 = value
        return float(histo.argmax()/1000)
        # part to remove background issue of deep
        # if value_max2 > (x2-x1)*(y2-y1)*0.05 and max1>=NUMBER_DETECTION:
        #     return float(histo.argmax()/1000)
        # else:
        #     return float(max1/1000)
