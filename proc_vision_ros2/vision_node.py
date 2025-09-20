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
from .yolov8 import YOLOv8
import math

sys.path.append("/home/sonia/ssd/pip_pkg")
MODEL_DIR = os.environ['SONIA_WS']+'/src/proc_vision_ros2/models/'

if os.path.exists('/home/sonia/ssd/output_ai/'):
    OUTPUT_DIR = '/home/sonia/ssd/output_ai/'
else:
    OUTPUT_DIR = '/home/sonia/output_ai/'
SAVE_OUTPUT = False

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
            self.get_logger().info("Front OFF")

        if self.camera_bottom:
            self.get_logger().info(f"Bottom ON -> Model : {model_name}")
        else:
            self.get_logger().info("Bottom OFF")

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
            self.actualise_deep()
            detections = model.detect(cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR), msg.header.frame_id)
            if(self.camera_front):
                #get the depth for each element see by the front camera
                for detect in detections.detected_object:
                    detect.distance = self.get_deep_istogram(int(detect.top_left_x), int(detect.bottom_right_x), int(detect.top_left_y) ,int(detect.bottom_right_y))
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
        
    def get_angle(self, x1: int,x2: int,y1: int,y2: int) -> int:
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
            
        x10 = (x1+x2)//20

        # part to get all pixel in bouding box
        area_seeLeft = self.__actual[min(max(0,x1),1280):min(max(0,x1),1280) + x10, min(max(0,y1),720):min(max(0,y2),720)]
        area_seeMid = self.__actual[(min(max(0,x1),1280)+min(max(0,x2),1280))//2 - (x10//2) :(min(max(0,x1),1280)+min(max(0,x2),1280))//2 + (x10//2), min(max(0,y1),720):min(max(0,y2),720)]
        # area_seeRigth = self.__actual[min(max(0,x2),1280) - x10 :min(max(0,x2),1280), min(max(0,y1),720):min(max(0,y2),720)]

        # part to generate the histogram to get the most probable value for the depth
        distanceX1 = np.histogram(area_seeMid,range = (0,5000),bins=500)[0].argmax()
        distanceX2 = np.histogram(area_seeLeft,range = (0,5000),bins=500)[0].argmax()
        # distanceX3 = np.histogram(area_seeRigth,range = (0,5000),bins=500)[0].argmax()

        centreX = (x1 + x2) // 2

        angleX = (640 - centreX)*41/1280

        angle2 = (640 - x1)*41/1280

        pointx1 = math.cos(angleX) * distanceX1
        pointy1 = math.sin(angleX) * distanceX1

        pointx2 = math.cos(angle2) * distanceX2
        pointy2 = math.sin(angle2) * distanceX2

        hypo = math.sqrt(math.pow(pointx1-pointx2,2)+math.pow(pointy1-pointy2,2))
        adja = math.sqrt(math.pow(pointx1-pointx2,2))

        if pointy1 < pointy2:
            angleTeta = 90 - math.acos(adja/hypo)
        else:
            angleTeta = -90 + math.acos(adja/hypo)

        math.cos(angleTeta)
        math.sin(angleTeta)
        
        newY = math.sin(angleTeta) * pointx1 + math.cos(angleTeta) * pointy1

        return (angleX, angleTeta,newY)
        
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
        area_see = self.__actual[min(max(0,x1),1280):min(max(0,x2),1280), min(max(0,y1),720):min(max(0,y2),720)]

        # part to generate the histogram to get the most probable value for the depth
        histo = np.histogram(area_see,range = (0,65535),bins=6553)[0]

        return float(histo.argmax()/1000)