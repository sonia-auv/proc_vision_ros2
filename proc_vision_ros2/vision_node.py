import sys
sys.path.append("/home/sonia/ssd/pip_pkg")

from rclpy.node import Node
from typing import Tuple
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import os
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

AREA_OF_SEE = 2


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

        #Deep
        self.__zed_depth = self.create_subscription(CompressedImage, "zed/zed_node/depth/depth_registered/compressed", self.__get_depth, 10)
        self.__actual = None
        self.__deep_last = None

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
            self.actualise_deep()
            self.get_logger().info(f"Image size = {image.shape[0]}x{image.shape[1]}")
            for detect in detections.detected_object:
                self.get_logger().info(str(self.get_deep((detect.bottom_right_x+detect.top_left_x)//2,(detect.bottom_right_y+detect.top_left_y)//2,672,376)))
                self.get_logger().info(str(self.get_deep_istogram(int(detect.top_left_x), int(detect.bottom_right_x), int(detect.top_left_y) ,int(detect.bottom_right_y),672,376)))
                detect.distance = self.get_deep_istogram(int(detect.top_left_x), int(detect.bottom_right_x), int(detect.top_left_y) ,int(detect.bottom_right_y),672,376)
            # self.print_results(image, detections)
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

    def __get_depth(self, msg: CompressedImage):
        self.__deep_last = cv2.imdecode(np.frombuffer(msg.data, np.uint8),cv2.IMREAD_GRAYSCALE)
        
    def actualise_deep(self):
        self.__actual = self.__deep_last
        
    def get_deep(self, x,y,w,h) -> int:
        if(self.__actual is None):
            self.actualise_deep()
            if (self.__actual is None):
                return 15
        sumf = 0
        number = 0
        resized_image = self.__actual
        # resized_image = cv2.resize(self.__actual, (w, h))
        if(x<AREA_OF_SEE):
            xStart = 0
        else:
            xStart = x- AREA_OF_SEE
        if(y<AREA_OF_SEE):
            yStart = 0
        else:
            yStart = y- AREA_OF_SEE
        for i in range(AREA_OF_SEE*2+1):
            if x+i< w:
                for j in range(AREA_OF_SEE*2+1):
                    if y+j<h:
                        sumf += resized_image[int(yStart+j)][int(xStart+i)]
                        number+=1
        return ((sumf/number)/255)*35
        
    def get_deep_istogram(self, x1,x2,y1,y2,w,h) -> int:
        if(self.__actual is None):
            self.actualise_deep()
            if (self.__actual is None):
                return 15
        dictValue = dict()
        resized_image = self.__actual
        # resized_image = cv2.resize(self.__actual, (w, h))
        for i in range(x1,x2):
            for j in range(y1,y2):
                if( not resized_image[j,i] <=5):
                    if not resized_image[j,i] in dictValue.keys():
                        dictValue[resized_image[j,i]]=0
                    dictValue[resized_image[j,i]]+=1
        histogram = sorted(dictValue.items())
        self.get_logger().info(str(histogram))
        max1 = 255
        valueMax1 = 0
        max2 = 255
        valueMax2 = 0
        for keys,value in histogram:
            if (value) > valueMax1:
                max2 = max1
                valueMax2 = valueMax1
                valueMax1 = value
                max1 = keys
            elif value > valueMax2:
                max2 = keys
                valueMax2 = value
        if valueMax2 > (x2-x1)*(y2-y1)*0.05 and max1>=250:
            return (max2/255)*35
        else:
            return (max1/255)*35
        


    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            img (np.ndarray): Resized and padded image.
            pad (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

        return img
