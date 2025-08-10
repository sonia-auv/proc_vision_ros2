import sys
sys.path.append("/home/sonia/ssd/pip_pkg")

from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CompressedImage
# from ultralytics import YOLO
import numpy as np
import os
import cv2
from .yolov8 import YOLOv8
from sonia_common_ros2.msg import DetectionArray
from sonia_common_ros2.srv import AiActivationService

if os.path.exists('/home/sonia/ssd/ros2_sonia_ws/src/proc_vision_ros2/models/'):
    MODEL_DIR = '/home/sonia/ssd/ros2_sonia_ws/src/proc_vision_ros2/models/'
else:
    MODEL_DIR = '/home/sonia/ros2_sonia_ws/src/proc_vision_ros2/models/'

if os.path.exists('/home/sonia/ssd/output_ai/'):
    OUTPUT_DIR = '/home/sonia/ssd/output_ai/'
else:
    OUTPUT_DIR = '/home/sonia/output_ai/'
SAVE_OUTPUT = False


class VisionNode(Node):

    def __init__(self):
        super().__init__("vision_node")
        self.camera_front = False
        self.camera_bottom = False
        self.declare_parameter("models", Parameter.Type.STRING_ARRAY) 
        self.__ai_activation_sub = self.create_service(AiActivationService, "proc_vision/ai_activation", self.__ai_activation_callback)

        self.__front_cam_sub = self.create_subscription(CompressedImage, "zed/zed_node/left/image_rect_color/compressed", self.__img_front_callback, 10)
        self.__front_cam_sim = self.create_subscription(CompressedImage, "proc_simulation/front/compressed", self.__img_front_callback, 10)
        
        self.__bottom_cam_sub = self.create_subscription(CompressedImage, "camera_array/bottom/image_raw", self.__img_bottom_callback, 10)
        self.__bottom_cam_sim = self.create_subscription(CompressedImage, "proc_simulation/bottom/compressed", self.__img_bottom_callback, 10)

        model_front_name = self.get_parameter("models").get_parameter_value().string_array_value[0]
        model_bottom_name = self.get_parameter("models").get_parameter_value().string_array_value[0]
        self.model_front = YOLOv8(os.path.join(MODEL_DIR, model_front_name))
        self.model_bottom = YOLOv8(os.path.join(MODEL_DIR, model_bottom_name))
        self.__classif_front_pub = self.create_publisher(DetectionArray, "proc_vision/front/classif", 10)
        self.__classif_bottom_pub = self.create_publisher(DetectionArray, "proc_vision/bottom/classif", 10)
        if SAVE_OUTPUT:
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
        self.get_logger().info("VISION NODE INITIALIZED !!")


    def __ai_activation_callback(self, request, response):
        if request.camera_choice == AiActivationService.Request.FRONT:
            self.camera_front = True
            self.camera_bottom = False
            if request.model_choice >= 0:
                model_name = self.get_parameter("models").get_parameter_value().string_array_value[request.model_choice]
                self.model_front = YOLOv8(os.path.join(MODEL_DIR, model_name))
        elif request.camera_choice == AiActivationService.Request.BOTTOM:
            self.camera_front = False
            self.camera_bottom = True
            if request.model_choice >= 0:
                model_name = self.get_parameter("models").get_parameter_value().string_array_value[request.model_choice]
                self.model_bottom = YOLOv8(os.path.join(MODEL_DIR, model_name))
        elif request.camera_choice == AiActivationService.Request.BOTH:
            self.camera_front = True
            self.camera_bottom = True
            if request.model_choice >= 0:
                model_name = self.get_parameter("models").get_parameter_value().string_array_value[request.model_choice]
                self.model_front = YOLOv8(os.path.join(MODEL_DIR, model_name))
                self.model_bottom = YOLOv8(os.path.join(MODEL_DIR, model_name))
        else:
            self.camera_front = False
            self.camera_bottom = False

        return response

    def __img_front_callback(self, msg: Image):
        if self.camera_front:
            self.get_logger().info("IMG FRONT received!!")
            # for detected_obj in self.__img_detection(msg, self.model_front):
            self.__classif_front_pub.publish(self.__img_detection(msg, self.model_front))

    def __img_bottom_callback(self, msg: Image):
        if self.camera_bottom:
            self.get_logger().info("IMG BOTTOM received!!")
            self.__classif_bottom_pub.publish(self.__img_detection(msg, self.model_bottom))

    def __img_detection(self, msg: Image, model: YOLOv8) -> DetectionArray:
        return model.detect(cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR))
