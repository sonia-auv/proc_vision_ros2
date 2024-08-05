#!/usr/bin/env python
import sys
sys.path.append("/home/sonia/ssd/pip_pkg")

import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
from time import time
from sonia_common.msg import Detection, DetectionArray
from sonia_common.srv import AiActivationService
from proc_vision_ros2.items_robosub import ItemsRobosub

MODEL_DIR = 'src/proc_vision_ros2/models/'
MODELS = ['model-1-yolov8n.pt', 
          'model-2-yolov8n.pt', 
          'yolov10-1-700.pt',
          'zac_v8.pt',
          'zac_v10.pt',
          'carriere_et_zac_v8.pt',
          'carriere_et_zac_v10.pt',
          'carriere_zac_denise_yucca_sim_v8.pt',
          'carriere_zac_denise_yucca_v8.pt']
MODEL_INDEX = 7
OUTPUT_DIR = 'output_ai/'
SAVE_OUTPUT = False

class VisionNode():

    def __init__(self):

        self.camera_front = False
        self.camera_bottom = False
        self.__ai_activation_sub = rospy.Service("proc_vision/ai_activation", AiActivationService, self.__ai_activation_callback)
        self.__front_cam_sub = rospy.Subscriber("camera_array/front/image_raw", Image, self.__img_front_callback, 10)
        self.__front_cam_sub = rospy.Subscriber("proc_simulation/front", Image, self.__img_front_sim_callback, 10)
        self.__bottom_cam_sub = rospy.Subscriber("camera_array/bottom/image_raw", Image, self.__img_bottom_callback, 10)
        self.__bottom_cam_sub = rospy.Subscriber("proc_simulation/bottom", Image, self.__img_bottom_sim_callback, 10)
        self.model = YOLO(MODEL_DIR + MODELS[MODEL_INDEX])
        self.__classification_front_pub = rospy.Publisher("proc_vision/front/classification", DetectionArray, queue_size=10)
        self.__classification_bottom_pub = rospy.Publisher("proc_vision/bottom/classification", DetectionArray, queue_size=10)

        if SAVE_OUTPUT:
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)


    def __ai_activation_callback(self, request):
        if request.ai_activation  == 1:
            self.camera_front = True
            self.camera_bottom = False
        elif request.ai_activation == 2:
            self.camera_front = False
            self.camera_bottom = True
        elif request.ai_activation == 3:
            self.camera_front = True
            self.camera_bottom = True
        else:
            self.camera_front = False
            self.camera_bottom = False

        return []

    def __img_front_callback(self, msg: Image, empty):
        if self.camera_front:
            imgconv = np.frombuffer(msg.data, dtype=np.uint8).reshape(400, 600, 3)
            img = np.array(imgconv)

            detection_array = DetectionArray()
            for detected_obj in self.__img_detection(img):
                detection_array.detected_object.append(detected_obj)
            
            self.__classification_front_pub.publish(detection_array)

    def __img_front_sim_callback(self, msg: Image, empty):
        if self.camera_front:
            imgconv = np.frombuffer(msg.data, dtype=np.uint8).reshape(400, 600, 3)
            img = np.array(imgconv)

            temp = img[:,:,0].copy()
            img[:,:,0] = img[:,:,2]
            img[:,:,2] = temp

            detection_array = DetectionArray()
            for detected_obj in self.__img_detection(img):
                detection_array.detected_object.append(detected_obj)
            
            self.__classification_front_pub.publish(detection_array)

    def __img_bottom_callback(self, msg: Image, empty):
        if self.camera_bottom:
            imgconv = np.frombuffer(msg.data, dtype=np.uint8).reshape(400, 600, 3)
            img = np.array(imgconv)

            detection_array = DetectionArray()
            for detected_obj in self.__img_detection(img):
                detection_array.detected_object.append(detected_obj)
            
            self.__classification_bottom_pub.publish(detection_array)

    def __img_bottom_sim_callback(self, msg: Image, empty):
        if self.camera_bottom:
            imgconv = np.frombuffer(msg.data, dtype=np.uint8).reshape(400, 600, 3)
            img = np.array(imgconv)

            temp = img[:,:,0].copy()
            img[:,:,0] = img[:,:,2]
            img[:,:,2] = temp

            detection_array = DetectionArray()
            for detected_obj in self.__img_detection(img):
                detection_array.detected_object.append(detected_obj)
            
            self.__classification_bottom_pub.publish(detection_array)

    def __img_detection(self, img):
        
        results = self.model(img, imgsz=[600, 400], conf=0.5, verbose=False)
        detections = []
        for res in results:
            detection_count = res.boxes.shape[0]
            for i in range(detection_count):
                cls = int(res.boxes.cls[i].item())
                name = res.names[cls]
                box = res.boxes.xyxy[i].cpu().numpy()
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                w = x1 - x2
                h = y1 - y2
                items = ItemsRobosub()
                classification = Detection()
                classification.top = float(y1)
                classification.left = float(x1)
                classification.bottom = float(y2)
                classification.right = float(x2)
                classification.class_name = name
                classification.distance = float(items.get_dist(y2 - y1, name))
                classification.confidence = float(res.boxes.conf[i].item())
                detections.append(classification)

                if SAVE_OUTPUT:
                    cv2.putText(img, 
                                name, 
                                (int((x1+5)),
                                int((y2-10)/2)), 
                                cv2.FONT_HERSHEY_PLAIN, 
                                .7, (0,0,255), 1, 1)
                    cv2.putText(img, 
                                "{:.1f}%".format(res.boxes.conf[i].item()), 
                                (int((x1+5)),
                                int((y2+10)/2)), 
                                cv2.FONT_HERSHEY_PLAIN, 
                                .7, (0,0,255), 1, 1)
                    cv2.rectangle(img, 
                                (int(x1),int(y1)), 
                                (int(x2),int(y2)), 
                                (0,0,255), 1)
                     
        if SAVE_OUTPUT and detection_count != 0:
            cv2.imwrite(OUTPUT_DIR+'pred_'+str(int(1000*time()))+'.jpg', 
                        img)

        return detections

if __name__ == "__main__":
    rospy.init_node('proc_vision_ros2')

    v = VisionNode()
    rospy.spin()
