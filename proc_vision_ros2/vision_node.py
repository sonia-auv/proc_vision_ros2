
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
import numpy as np
from sonia_common_ros2.msg import VisionClass, AiActivationService
from items_robosub import ItemsRobosub

MODEL = 'models/model-1-yolov8n.pt'

class VisionNode(Node):

    def __init__(self):
        super().__init__("vision_node")
        self.camera_front = False
        self.camera_bottom = False
        self.__ai_activation_sub = self.create_subscription(AiActivationService, "proc_vision/ai_activation", self.__ai_activation_callback, 10)
        self.__front_cam_sub = self.create_subscription(Image, "camera_array/front/image_raw", self.__img_front_callback, 10)
        self.__bottom_cam_sub = self.create_subscription(Image, "camera_array/bottom/image_raw", self.__img_bottom_callback, 10)
        self.model = YOLO(MODEL)
        self.__classification_front_pub = self.create_publisher(VisionClass, "proc_vision/front/classification", 10)
        self.__classification_bottom_pub = self.create_publisher(VisionClass, "proc_vision/bottom/classification", 10)

    def __ai_activation_callback(self, msg: AiActivationService):
        if msg.ai_activation == AiActivationService.FRONT:
            self.camera_front = True
            self.camera_bottom = False
        elif msg.ai_activation == AiActivationService.BOTTOM:
            self.camera_front = False
            self.camera_bottom = True
        elif msg.ai_activation == AiActivationService.BOTH:
            self.camera_front = True
            self.camera_bottom = True
        else:
            self.camera_front = False
            self.camera_bottom = False

    def __img_front_callback(self, msg: Image):
        self.get_logger().info("IMG received!!")
        if self.camera_front:
            self.__classification_front_pub.publish(self.__img_detection(msg))

    def __img_bottom_callback(self, msg: Image):
        self.get_logger().info("IMG received!!")
        if self.camera_bottom:
            self.__classification_bottom_pub.publish(self.__img_detection(msg))

    def __img_detection(self, msg: Image):
        img = np.array(msg.data).reshape((400,600,3))
        results = self.model(img, imgsz=[600, 400], conf=0.5, verbose=False)
        for res in results:
            detection_count = res.boxes.shape[0]
            for i in range(detection_count):
                cls = int(res.boxes.cls[i].item())
                name = res.names[cls]
                conf = float(res.boxes.conf[i].item())
                box = res.boxes.xyxy[i].cpu().numpy()
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                w = x1 - x2
                h = y1 - y2
                items = ItemsRobosub()
                classification = VisionClass()
                classification.origin.x = float(x1)
                classification.origin.y = float(y1)
                classification.endpoint.x = float(x2)
                classification.endpoint.y = float(y2)
                classification.classification = name
                classification.distance = items.get_dist(y1 - y2, name)
                classification.confidence = conf
        return classification