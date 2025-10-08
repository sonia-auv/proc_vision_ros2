import sys
import os
import traceback
import cv2
from sonia_common_ros2.msg import DetectionArray, Detection
from sonia_common_ros2.srv import AiActivationService
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
from cv_bridge import CvBridge
import math

sys.path.append("/home/sonia/ssd/pip_pkg")

from .yolov8 import YOLOv8

MODEL_DIR = os.environ['SONIA_WS']+'/src/proc_vision_ros2/models/'

if os.path.exists('/home/sonia/ssd/output_ai/'):
    OUTPUT_DIR = '/home/sonia/ssd/output_ai/'
else:
    OUTPUT_DIR = '/home/sonia/output_ai/'
SAVE_OUTPUT = False

ZED_VFOV = 52/2
ZED_HFOV = 82/2
IMAGE_WIDTH = 1280
IMAGE_HEIGTH = 720

#Multiplicateur to be in meter
UNIT=100

class VisionNode(Node):

    def __init__(self):
        super().__init__("vision_node")
        
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.camera_front = False
        self.camera_bottom = False
        self.declare_parameter("models", Parameter.Type.STRING_ARRAY) 
        self.__ai_activation_sub = self.create_service(AiActivationService, "proc_vision/ai_activation", self.__ai_activation_callback)

        self.__front_cam_sub = self.create_subscription(Image, "zed/zed_node/left/image_rect_color", self.__img_front_callback, 10)
        self.__front_cam_sim = self.create_subscription(Image, "proc_simulation/front", self.__img_front_callback, 10)
        
        self.__bottom_cam_sub = self.create_subscription(Image, "camera_array/bottom/image_raw", self.__img_bottom_callback, qos)
        self.__bottom_cam_sim = self.create_subscription(Image, "proc_simulation/bottom", self.__img_bottom_callback, 10)

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
            self.get_logger().info("Front OFF")

        if self.camera_bottom:
            self.get_logger().info(f"Bottom ON -> Model : {model_name}")
        else:
            self.get_logger().info("Bottom OFF")

        return response

    def __img_front_callback(self, msg: Image):
        if self.camera_front:
            self.__classif_front_pub.publish(self.__img_detection(msg, self.model_front))

    def __img_bottom_callback(self, msg: Image):
        if self.camera_bottom:
            self.__classif_bottom_pub.publish(self.__img_detection(msg, self.model_bottom))

    def __img_detection(self, msg: Image, model: YOLOv8) -> DetectionArray:
        try:
            self.actualise_deep()
            detections = model.detect(self.br.imgmsg_to_cv2(msg), msg.header.frame_id)
            if(self.camera_front):
                #get the depth for each element see by the front camera
                for detect in detections.detected_object:
                    detect.distance = self.get_deep_istogram(int(detect.top_left_x), int(detect.bottom_right_x), int(detect.top_left_y) ,int(detect.bottom_right_y))
                    if(detect.class_name == "torpedo-poster"):
                        angle_alpha,ditance_beta,angle_teta,distance_teta = self.get_angle(int(detect.top_left_x), int(detect.bottom_right_x), int(detect.top_left_y) ,int(detect.bottom_right_y))
                        detect.angle_alpha = angle_alpha
                        detect.distance_beta = ditance_beta
                        detect.angle_teta = angle_teta
                        detect.distance_teta = distance_teta
            return detections
        except Exception as e:
            self.get_logger().info(f"Vision node failure :{e}")
            self.get_logger().info(f"Vision node failure trace :{traceback.format_exc()}")
            detections = DetectionArray()
            detections.detected_object = []
            return detections

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
        
    def get_angle(self, x1: int,x2: int,y1: int,y2: int):
        """function to return the distance of a object on a image

        Args:
            x1 (int): cordoninante in x of top left point
            x2 (int): cordoninante in x of bottom right
            y1 (int): cordoninante in y of top left point
            y2 (int): cordoninante in y of bottom right

        Returns:
            int: the distance of object
        """
        if(self.__actual is None):
            self.actualise_deep()
            if (self.__actual is None):
                return float(0),float(0),float(0),float(0)
            
        x10 = (x1+x2)//20

        # part to get all pixel in bouding box
        area_seeLeft = self.__actual[min(max(0,x1),IMAGE_WIDTH) + x10 :min(max(0,x1),IMAGE_WIDTH) + x10*2, min(max(0,y1),IMAGE_HEIGTH):min(max(0,y2),IMAGE_HEIGTH)]
        area_seeMid = self.__actual[(min(max(0,x1),IMAGE_WIDTH)+min(max(0,x2),IMAGE_WIDTH))//2 - (x10//2) :(min(max(0,x1),IMAGE_WIDTH)+min(max(0,x2),IMAGE_WIDTH))//2 + (x10//2), min(max(0,y1),IMAGE_HEIGTH):min(max(0,y2),IMAGE_HEIGTH)]

        # part to generate the histogram to get the most probable value for the depth
        distanceX1 = np.histogram(area_seeMid,range = (0,15000),bins=1500)[0].argmax()
        distanceX2 = np.histogram(area_seeLeft,range = (0,15000),bins=1500)[0].argmax()

        centreX = (x1 + x2) // 2
        centreY = (y1 + y2) // 2

        angleX = (IMAGE_WIDTH/2 - centreX)*ZED_HFOV/IMAGE_WIDTH

        angleY = (IMAGE_HEIGTH/2 - centreY)*ZED_VFOV/IMAGE_HEIGTH

        angle2 = (IMAGE_WIDTH/2 - x1)*ZED_HFOV/IMAGE_WIDTH

        pointx1 = math.cos(angleX) * distanceX1
        pointy1 = math.sin(angleX) * distanceX1

        pointx2 = math.cos(angle2) * distanceX2
        pointy2 = math.sin(angle2) * distanceX2

        pointSubY = math.sin(angleY) * distanceX1

        hypo = math.sqrt((pointx2-pointx1)*(pointx2-pointx1)+(pointy2-pointy1)*(pointy2-pointy1))

        if hypo == 0:
            self.get_logger().info("issue hypo "+ str(hypo))
            return float(angleX),float(angleY),float(0),float(0)
        if pointy1 < pointy2:
            angleTeta = - math.asin(math.abs(pointx2)/hypo)
        else:
            angleTeta = math.asin(math.abs(pointx2)/hypo)

        #y par la matrice de rotation en 2D
        newY = math.sin(angleTeta) * pointx1 + math.cos(angleTeta) * pointy1

        return float(angleX),float(pointSubY/UNIT), float(angleTeta), float(newY/UNIT)
        
    def get_deep_istogram(self, x1: int,x2: int,y1: int,y2: int) -> int:
        """function to return the distance of a object on a image

        Args:
            x1 (int): cordoninante in x of top left point
            x2 (int): cordoninante in x of bottom right
            y1 (int): cordoninante in y of top left point
            y2 (int): cordoninante in y of bottom right

        Returns:
            int: the distance of object
        """
        if(self.__actual is None):
            self.actualise_deep()
            if (self.__actual is None):
                return float(60)

        # part to get all pixel in bouding box
        area_see = self.__actual[min(max(0,x1),IMAGE_WIDTH):min(max(0,x2),IMAGE_WIDTH), min(max(0,y1),IMAGE_HEIGTH):min(max(0,y2),IMAGE_HEIGTH)]

        # part to generate the histogram to get the most probable value for the depth
        histo = np.histogram(area_see,range = (0,65535),bins=6553)[0]

        return float(histo.argmax()/UNIT)
