import sys
sys.path.append("/home/sonia/ssd/pip_pkg")

from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import os
import cv2
from .yolov8 import YOLOv8
from sonia_common_ros2.msg import DetectionArray, Detection
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
        # self.__front_cam_depth = self.create_subscription(Image, "zed/zed_node/depth/depth_registered", self.__depth_front_callback, 10)
        
        self.__bottom_cam_sub = self.create_subscription(Image, "camera_array/bottom/image_raw", self.__img_bottom_callback, 10)
        self.__bottom_cam_sim = self.create_subscription(CompressedImage, "proc_simulation/bottom/compressed", self.__img_bottom_callback, 10)

        model_front_name = self.get_parameter("models").get_parameter_value().string_array_value[0]
        model_bottom_name = self.get_parameter("models").get_parameter_value().string_array_value[0]
        self.model_front = YOLOv8(os.path.join(MODEL_DIR, model_front_name), self)
        self.model_bottom = YOLOv8(os.path.join(MODEL_DIR, model_bottom_name), self)
        self.__classif_front_pub = self.create_publisher(DetectionArray, "proc_vision/front/classif", 10)
        self.__classif_bottom_pub = self.create_publisher(DetectionArray, "proc_vision/bottom/classif", 10)
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

    # def __depth_front_callback(self, msg: Image):
    #     if self.camera_front:
    #         self.get_logger().info(f"Depth {msg.header.frame_id} received!!")
    #         depth_raw = np.frombuffer(msg.data, np.uint8)
    #         depth = cv2.imdecode(depth_raw, cv2.IMREAD_GRAYSCALE)
    #         depth2 = cv2.imdecode(depth_raw, cv2.IMREAD_ANYDEPTH)
    #         self.get_logger().info(f"depth_raw shape: {depth_raw.shape}, dtype: {depth_raw.dtype}")
    #         if depth is not None:
    #             self.get_logger().info(f"Depth image max: {depth.max()}, min: {depth.min()}, mean: {depth.mean()}")
    #         if depth2 is not None:
    #             self.get_logger().info(f"Depth2 image max: {depth2.max()}, min: {depth2.min()}, mean: {depth2.mean()}")

    def __img_detection(self, msg: CompressedImage, model: YOLOv8) -> DetectionArray:
        try:
            image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            detections = model.detect(image, msg.header.frame_id)
            # self.print_results(image, detections)
            return detections
        except Exception as e:
            self.get_logger().info(f"Vision node failure {e}")
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
            cv2.imwrite('./image_window.jpg', img_res)
