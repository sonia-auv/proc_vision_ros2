#pragma once
#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.hpp>

#include "proc_vision_ros2/Yolo.hpp"

#include "proc_vision_ros2/logging.h"

#include "sonia_common_ros2/srv/ai_activation_service.hpp"
#include "sonia_common_ros2/msg/detection_array.hpp"
#include <vector>

namespace proc_vision_ros2
{
    class Proc_vision_ros2 : public rclcpp::Node
    {
    public:
        Proc_vision_ros2();
        ~Proc_vision_ros2();

    private:

        /**
         * @brief Processes a ai activation service request.
         *
         * @param request message to actiavte cameras and models
         * @param response if the message was correctly traited
         */
        void processActuatorRequest(const std::shared_ptr<sonia_common_ros2::srv::AiActivationService::Request> request,
                                   std::shared_ptr<sonia_common_ros2::srv::AiActivationService::Response> response);

        // Function for the front camera

        void messageFrontCameraCallBack(const sensor_msgs::msg::Image::SharedPtr &msg);

        // Function for the bottom camera

        void messageBottomCameraCallBack(const sensor_msgs::msg::Image::SharedPtr &msg);

        // Function the IA

        sonia_common_ros2::msg::DetectionArray imgDetection(const sensor_msgs::msg::Image::SharedPtr &msg, Yolo model);

        // Function for the depth

        void messageZedDepthCallBack(const sensor_msgs::msg::Image::SharedPtr &msg);

        void actualiseDepp();

        float getDeepIstogram(int x1, int y1, int x2, int y2);

        void getAngle(int x1, int y1, int x2, int y2, vector<float> angle);

        // all variable to work

        rclcpp::Service<sonia_common_ros2::srv::AiActivationService>::SharedPtr _aiActivationService;

        rclcpp::Publisher<sonia_common_ros2::msg::DetectionArray>::SharedPtr _publisherDetectionArrayBottom;
        rclcpp::Publisher<sonia_common_ros2::msg::DetectionArray>::SharedPtr _publisherDetectionArrayFront;

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subscriberFrontCamera;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subscriberBottomCameraSim;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subscriberBottomCamera;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subscriberBottomCameraSim;

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _subscriberZedDepth;

        sonia_common_ros2::msg::DetectionArray detections;

        bool _cameraFront = false;
        bool _cameraBottom = false;

        Yolo _modelFront;
        Yolo _modelBottom;

        cv_bridge::CvImagePtr _cvBridge;
        cv::Mat _actualDepth;
        cv::Mat _lastDepth;

        Logger logger;

        string MODELDIR;

        // constant Image link need to be changed
        const int ZEDVFOV = 52/2;
        const int ZEDHFOV = 82/2;
        const int IMAGEWIDTH = 1280;
        const int IMAGEHEIGTH = 720;

        const int _UNIT=100;
    };
}