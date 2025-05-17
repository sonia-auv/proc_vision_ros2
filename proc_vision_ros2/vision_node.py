import sys
sys.path.append("/home/sonia/ssd/pip_pkg")

from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
from time import time
from sonia_common_ros2.msg import Detection
from sonia_common_ros2.srv import AiActivationService

# MODEL_DIR = '/home/sonia/ssd/ros2_sonia_ws/src/proc_vision_ros2/models/'
# OUTPUT_DIR = '/home/sonia/ssd/output_ai/'
MODEL_DIR = '/home/sonia2/ros2_sonia_ws/src/proc_vision_ros2/models/'
OUTPUT_DIR = '/home/sonia2/output_ai/'
SAVE_OUTPUT = True

class VisionNode(Node):

    def __init__(self):
        super().__init__("vision_node")
        self.camera_front = False
        self.camera_bottom = False
        self.declare_parameter("models") 
        self.__ai_activation_sub = self.create_service(AiActivationService, "proc_vision/ai_activation", self.__ai_activation_callback)
        self.__front_cam_sub = self.create_subscription(Image, "camera_array/front/image_raw/compressed", self.__img_front_callback, 10)
        self.__bottom_cam_sub = self.create_subscription(Image, "camera_array/bottom/image_raw", self.__img_bottom_callback, 10)
        model_front_name = self.get_parameter("models").get_parameter_value().string_array_value[1]
        model_bottom_name = self.get_parameter("models").get_parameter_value().string_array_value[1]
        self.model_front = YOLO(os.path.join(MODEL_DIR, model_front_name))
        self.model_bottom = YOLO(os.path.join(MODEL_DIR, model_bottom_name))
        self.__classif_front_pub = self.create_publisher(Detection, "proc_vision/front/classif", 10)
        self.__classif_bottom_pub = self.create_publisher(Detection, "proc_vision/bottom/classif", 10)
        if SAVE_OUTPUT:
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

    def __ai_activation_callback(self, request, response):
        if request.camera_choice == AiActivationService.Request.FRONT:
            self.camera_front = True
            self.camera_bottom = False
            if request.model_choice >= 0:
                model_name = self.get_parameter("models").get_parameter_value().string_array_value[request.model_choice]
                self.model_front = YOLO(os.path.join(MODEL_DIR, model_name))
        elif request.camera_choice == AiActivationService.Request.BOTTOM:
            self.camera_front = False
            self.camera_bottom = True
            if request.model_choice >= 0:
                model_name = self.get_parameter("models").get_parameter_value().string_array_value[request.model_choice]
                self.model_bottom = YOLO(os.path.join(MODEL_DIR, model_name))
        elif request.camera_choice == AiActivationService.Request.BOTH:
            self.camera_front = True
            self.camera_bottom = True
            if request.model_choice >= 0:
                model_name = self.get_parameter("models").get_parameter_value().string_array_value[request.model_choice]
                self.model_front = YOLO(os.path.join(MODEL_DIR, model_name))
                self.model_bottom = YOLO(os.path.join(MODEL_DIR, model_name))
        else:
            self.camera_front = False
            self.camera_bottom = False

        return response

    def __img_front_callback(self, msg: Image):
        self.get_logger().info("IMG FRONT received!!")
        if self.camera_front:
            for detected_obj in self.__img_detection(msg, self.model_front):
                self.__classif_front_pub.publish(detected_obj)

    def __img_bottom_callback(self, msg: Image):
        self.get_logger().info("IMG BOTTOM received!!")
        if self.camera_bottom:
            for detected_obj in self.__img_detection(msg, self.model_bottom):
                self.__classif_bottom_pub.publish(detected_obj)

    def __manage_oriented_boxes(self, i: int, result, classif: Detection):
        box = result.boxes.xyxy[i].cpu().numpy()
        classif.top_left_x = float(box[0])
        classif.top_left_y = float(box[1])
        classif.top_right_x = float(box[2])
        classif.top_right_y = float(box[3])
        classif.bottom_right_x = float(box[4])
        classif.bottom_right_y = float(box[5])
        classif.bottom_left_x = float(box[6])
        classif.bottom_left_y = float(box[7])
        classif.confidence = float(result.boxes.conf[i].item())
        return classif

    def __manage_boxes(self, i: int, result, classif: Detection):
        box = result.obb.xyxyxyxy[i].cpu().numpy()
        classif.top_left_x = float(box[0])
        classif.top_left_y = float(box[1])
        classif.top_right_x = float(box[0])
        classif.top_right_y = float(box[3])
        classif.bottom_right_x = float(box[2])
        classif.bottom_right_y = float(box[3])
        classif.bottom_left_x = float(box[2])
        classif.bottom_left_y = float(box[1])
        classif.confidence = float(result.obb.conf[i].item())
        return classif

    def __img_detection(self, msg: Image, model: YOLO):
        # TODO check and manage case with rgbxyz images
        img = np.array(msg.data).reshape((400,600,3))
        results = model(img, imgsz=[600, 400], conf=0.5, verbose=False)
        detections = []
        for res in results:
            detection_count = res.boxes.shape[0]
            for i in range(detection_count):
                cls = int(res.boxes.cls[i].item())
                name = res.names[cls]
                # TODO
                if res.boxes is not None:
                    classif = self.__manage_boxes(i, res)
                else:
                    classif = self.__manage_oriented_boxes(i, res)
                classif = Detection()
                classif.classif = name
                # classif.distance = np.median() TODO
                detections.append(classif)

                if SAVE_OUTPUT:
                    cv2.putText(img, 
                                name, 
                                (int((x1+5)),
                                int((y2-10)/2)), 
                                cv2.FONT_HERSHEY_PLAIN, 
                                .7, (0,0,255), 1, 1)
                    cv2.putText(img, 
                                "{:.1f}%".format(classif.confidence), 
                                (int((x1+5)),
                                int((y2+10)/2)), 
                                cv2.FONT_HERSHEY_PLAIN, 
                                .7, (0,0,255), 1, 1)
                    cv2.rectangle(img, 
                                (int(x1),int(y1)), 
                                (int(x2),int(y2)), 
                                (0,0,255), 1)
                    cv2.imwrite(OUTPUT_DIR+'pred_'+str(int(1000*time()))+'.jpg', 
                                img) 
        return detections
