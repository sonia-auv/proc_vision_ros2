
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
import numpy as np
from sonia_common_ros2.msg import VisionClass

class VisionNode(Node):

    def __init__(self):
        super().__init__("vision_node")

        self.__front_cam_sub = self.create_subscription(Image, "camera_array/front/image_raw", self.__img_callback, 10)
        self. model = YOLO('yolov8n.pt')
        self.__classification_pub = self.create_publisher(VisionClass, "proc_vision/classification", 10)


    def __img_callback(self, msg: Image):
        self.get_logger().info("IMG received!!")
        img = np.array(msg.data).reshape((400,600,3))
        results = self.model(img, imgsz=320, conf=0.5, verbose=False)
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
                log2 = 'name ='+ str(name)+ '| conf ='+ str(conf)+ '| coordinates :'+ str(x1)+ ' '+ str(y1)+ ' '+ str(x2)+ ' '+ str(y2)
                self.get_logger().info(log2)
                classification = VisionClass()
                classification.origin.x = float(x1)
                classification.origin.y = float(y1)
                classification.endpoint.x = float(x2)
                classification.endpoint.y = float(y2)
                classification.classification = name
                self.__classification_pub.publish(classification)