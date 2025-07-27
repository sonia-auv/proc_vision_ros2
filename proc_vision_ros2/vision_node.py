import sys
sys.path.append("/home/sonia/ssd/pip_pkg")

from rclpy.node import Node
from sensor_msgs.msg import Image
# from ultralytics import YOLO
import numpy as np
import os
from .yolov8 import YOLOv8
from sonia_common_ros2.msg import DetectionArray
from sonia_common_ros2.srv import AiActivationService

# MODEL_DIR = '/home/sonia/ssd/ros2_sonia_ws/src/proc_vision_ros2/models/'
# OUTPUT_DIR = '/home/sonia/ssd/output_ai/'
MODEL_DIR = '/home/sonia/ros2_sonia_ws/src/proc_vision_ros2/models/'
OUTPUT_DIR = '/home/sonia/output_ai/'
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
        model_front_name = self.get_parameter("models").get_parameter_value().string_array_value[0]
        model_bottom_name = self.get_parameter("models").get_parameter_value().string_array_value[0]
        self.model_front = YOLOv8(os.path.join(MODEL_DIR, model_front_name))
        self.model_bottom = YOLOv8(os.path.join(MODEL_DIR, model_bottom_name))
        self.__classif_front_pub = self.create_publisher(DetectionArray, "proc_vision/front/classif", 10)
        self.__classif_bottom_pub = self.create_publisher(DetectionArray, "proc_vision/bottom/classif", 10)
        if SAVE_OUTPUT:
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

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
        self.get_logger().info("IMG FRONT received!!")
        if self.camera_front:
            for detected_obj in self.__img_detection(msg, self.model_front):
                self.__classif_front_pub.publish(detected_obj)

    def __img_bottom_callback(self, msg: Image):
        self.get_logger().info("IMG BOTTOM received!!")
        if self.camera_bottom:
            for detected_obj in self.__img_detection(msg, self.model_bottom):
                self.__classif_bottom_pub.publish(detected_obj)

    def __img_detection(self, msg: Image, model: YOLOv8) -> DetectionArray:
        # TODO check and manage case with rgbxyz images
        img = np.array(msg.data).reshape((400,600,3))
        model.detect(img)
        # results = model(img, imgsz=[600, 400], conf=0.5, verbose=False)
        # detections = DetectionArray()
        # detections.detected_object = []
        # for res in results:
        #     detection_count = res.boxes.shape[0]
        #     for i in range(detection_count):
        #         cls = int(res.boxes.cls[i].item())
        #         name = res.names[cls]
        #         classif = Detection()
        #         if res.boxes is not None:
        #             classif = self.__manage_boxes(i, res, classif)
        #         else:
        #             classif = self.__manage_oriented_boxes(i, res, classif)
        #         classif.class_name = name
        #         # TODO
        #         # if image_stereo:
        #         #     classif.distance = np.median()
        #         # else:
        #         #     classif.distance = 0
        #         classif.distance = 0
        #         detections.detected_object.append(classif)

        #         if SAVE_OUTPUT:
        #             cv2.putText(img, 
        #                         name, 
        #                         (int((classif.top_left_x+5)),
        #                         int((classif.bottom_right_y-10)/2)), 
        #                         cv2.FONT_HERSHEY_PLAIN, 
        #                         .7, (0,0,255), 1, 1)
        #             cv2.putText(img, 
        #                         "{:.1f}%".format(classif.confidence), 
        #                         (int((classif.top_left_x+5)),
        #                         int((classif.bottom_right_y+10)/2)), 
        #                         cv2.FONT_HERSHEY_PLAIN, 
        #                         .7, (0,0,255), 1, 1)
        #             cv2.rectangle(img, 
        #                         (int(classif.top_left_x),int(classif.top_left_y)), 
        #                         (int(classif.bottom_right_x),int(classif.bottom_right_y)), 
        #                         (0,0,255), 1)
        #             cv2.imwrite(OUTPUT_DIR+'pred_'+str(int(1000*time()))+'.jpg', 
        #                         img) 
        return detections
