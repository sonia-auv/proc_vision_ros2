#pragma once
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.hpp>

#include "proc_vision_ros2/Yolo.hpp"

#include "proc_vision_ros2/logging.h"

#include "sonia_common_ros2/srv/ai_activation_service.hpp"
#include "sonia_common_ros2/msg/detection_array.hpp"
#include "sonia_common_ros2/msg/node_status.hpp"
#include <vector>

namespace proc_vision_ros2
{
    class Proc_vision : public rclcpp::Node
    {
    public:
        Proc_vision();
        ~Proc_vision();

        struct AngleDetection
        {
            float angle_alpha = 0;
            float distance_beta = 0;
            float angle_teta = 0;
            float distance_teta = 0;
        };
        

    private:

        /**
         * @brief Processes a ai activation service request.
         *
         * @param request message to actiavte cameras and models
         * @param response if the message was correctly traited
         */
        void processAiActivationRequest(const std::shared_ptr<sonia_common_ros2::srv::AiActivationService::Request> request,
                                   std::shared_ptr<sonia_common_ros2::srv::AiActivationService::Response> response);

        // Function for the front camera

        /**
         * @brief Callback for message for the front camera
         * 
         * @param msg message to read
         */
        void messageFrontCameraCallBack(const sensor_msgs::msg::Image &msg);

        // Function for the bottom camera

        /**
         * @brief Callback for message for the bottom camera
         * 
         * @param msg message to read
         */
        void messageBottomCameraCallBack(const sensor_msgs::msg::Image &msg);

        // Function the IA

        /**
         * @brief function to detect image
         * 
         * @param msg message to read
         * @param model model of yolo to use
         * @return sonia_common_ros2::msg::DetectionArray 
         */
        sonia_common_ros2::msg::DetectionArray imgDetection(const sensor_msgs::msg::Image &msg, Yolo* model);

        // Function for the depth

        /**
         * @brief Callback for message for the depth map
         * 
         * @param msg The depth Image in GreyScale
         */
        void messageZedDepthCallBack(const sensor_msgs::msg::Image &msg);

        /**
         * @brief Define the last depth read to the depth to use
         * 
         */
        void actualiseDepth();

        /**
         * @brief Get the Deep Istogram object from the last depth map actualise
         * 
         * @param x1 coordinate in x for the top left point
         * @param y1 coordinate in y for the top left point
         * @param x2 coordinate in x for the bottom right point
         * @param y2 coordinate in y for the bottom right point
         * @return float the distance
         */
        float getDeepHistogram(int x1, int y1, int x2, int y2);

        /**
         * @brief Get the Angle object to alignement with it
         * 
         * @param x1 coordinate in x for the top left point
         * @param y1 coordinate in y for the top left point
         * @param x2 coordinate in x for the bottom right point
         * @param y2 coordinate in y for the bottom right point
         * @param angle the variable to return result
         */
        void getAngle(int x1, int y1, int x2, int y2, AngleDetection angle);

        /**
         * @brief Publishes node information of its state and quality.
         */
        void publishStatus();

        // all variable to work

        rclcpp::Service<sonia_common_ros2::srv::AiActivationService>::SharedPtr _aiActivationService;

        rclcpp::Publisher<sonia_common_ros2::msg::DetectionArray>::SharedPtr _publisherDetectionArrayBottom;
        rclcpp::Publisher<sonia_common_ros2::msg::DetectionArray>::SharedPtr _publisherDetectionArrayFront;
        rclcpp::Publisher<sonia_common_ros2::msg::NodeStatus>::SharedPtr _publisherNodeStatus;

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subscriberFrontCamera;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subscriberFrontCameraSim;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subscriberBottomCamera;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subscriberBottomCameraSim;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subscriberZedDepth;

        rclcpp::TimerBase::SharedPtr _timerNodeStatus;

        sonia_common_ros2::msg::DetectionArray detections;
        sonia_common_ros2::msg::NodeStatus node_status;

        bool _cameraFront = false;
        bool _cameraBottom = false;

        Yolo* _modelFront;
        Yolo* _modelBottom;

        cv_bridge::CvImagePtr _cvBridge;
        cv::Mat _actualDepth;
        cv::Mat _lastDepth;

        Logger logger;

        string MODELDIR;

        AngleDetection _angle;

        // constant Image link need to be changed
        const int ZEDVFOV = 52/2;
        const int ZEDHFOV = 82/2;
        const int IMAGEWIDTH = 1280;
        const int IMAGEHEIGTH = 720;

        const int _UNIT=100;
    };
}