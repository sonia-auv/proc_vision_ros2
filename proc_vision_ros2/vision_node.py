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
          'carriere_zac_denise_yucca_v8.pt',
          'carriere_zac_denise_yucca_sim_v10.pt',
          'robosub_v8_1.pt',
          'robosub_most_recent.pt']
MODEL_INDEX = 11
OUTPUT_DIR = 'output_ai/'
SAVE_OUTPUT = False
PUBLISH_OUTPUT = True
INTERSECTION_TRESH = 0.10

def iou(d1, d2):
    xA = max(d1[0], d2[0])
    yA = max(d1[1], d2[1])
    xB = min(d1[2], d2[2])
    yB = min(d1[3], d2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (d1[2] - d1[0]) * (d1[3] - d1[1])
    area2 = (d2[2] - d2[0]) * (d2[3] - d2[1])
    return (inter / float(area1 + area2 - inter))


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
        self.__image_output_pub = rospy.Publisher("proc_vision/ai_output", Image, queue_size=10)
        # self.previous_front_results = [[], [], [], [], []]
        # self.previous_bottom_results = [[], [], [], [], []]

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
            for detected_obj in self.__img_detection(img, 'front'):
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
                detection_array.detected_object.append(detected_obj, 'front')
            
            self.__classification_front_pub.publish(detection_array)

    def __img_bottom_callback(self, msg: Image, empty):
        if self.camera_bottom:
            imgconv = np.frombuffer(msg.data, dtype=np.uint8).reshape(400, 600, 3)
            img = np.array(imgconv)

            detection_array = DetectionArray()
            for detected_obj in self.__img_detection(img, 'bottom'):
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
            for detected_obj in self.__img_detection(img, 'bottom'):
                detection_array.detected_object.append(detected_obj)
            
            self.__classification_bottom_pub.publish(detection_array)

    def __remove_double(self, detections):
        for d1 in detections:
            for d2 in detections:
                if d1 != d2 and d1.cls == d2.cls:
                    if iou(d1, d2) > INTERSECTION_TRESH:
                        if d1.conf < d2.conf:
                            detections.remove(d1)
                        else:
                            detections.remove(d2)

    def __img_detection(self, img, camera='front'):
        # results_now = self.model(img, imgsz=[608, 416], conf=0.5, verbose=False)
        results = self.model(img, imgsz=[608, 416], conf=0.5, verbose=False)
        
        detections = []
        # if camera == 'front':
        #     self.previous_front_results.insert(0, results_now)
        #     self.previous_front_results.pop(len(self.previous_front_results)-1)
        #     previous_results = self.previous_front_results
        # if camera == 'bottom':
        #     self.previous_bottom_results.insert(0, results_now)
        #     self.previous_bottom_results.pop(len(self.previous_bottom_results)-1)
        #     previous_results = self.previous_bottom_results
        # for results in previous_results:
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

                if SAVE_OUTPUT or PUBLISH_OUTPUT:
                    cv2.putText(img, 
                                name, 
                                (int((x1+1)),
                                int((y1-10))), 
                                cv2.FONT_HERSHEY_PLAIN, 
                                .7, (0,0,255), 1, 1)
                    cv2.putText(img, 
                                "{:.1f}%".format(res.boxes.conf[i].item()*100), 
                                (int((x1+1)),
                                int((y1-1))), 
                                cv2.FONT_HERSHEY_PLAIN, 
                                .7, (0,0,255), 1, 1)
                    cv2.rectangle(img, 
                                (int(x1),int(y1)), 
                                (int(x2),int(y2)), 
                                (0,0,255), 1)
                     
        if SAVE_OUTPUT and detection_count != 0:
            cv2.imwrite(OUTPUT_DIR+'pred_'+str(int(1000*time()))+'.jpg', 
                        img)

        if PUBLISH_OUTPUT:
            image_out = Image()
            image_out.height=400
            image_out.width=600
            image_out.encoding="bgr8"
            image_out.is_bigendian=0
            image_out.step=1800
            image_out.data = img.reshape(720000).tolist()
            self.__image_output_pub.publish(image_out)

        detections = self.__remove_double(detections)
        return detections

if __name__ == "__main__":
    rospy.init_node('proc_vision_ros2')

    v = VisionNode()
    rospy.spin()