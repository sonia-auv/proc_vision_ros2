#include "proc_vision_ros2/Proc_vision_ros2.hpp"
#include <algorithm>
#include <cstdlib>

using std::placeholders::_1;

namespace proc_vision_ros2
{
    Proc_vision_ros2::Proc_vision_ros2()
        : Node("proc_vision_ros2"){

        this->declare_parameter("models", std::vector<string>({"robosub-2025-v2"}));

        MODELDIR = (string)std::getenv("SONIA_WS")+"/src/proc_vision_ros2/models/";

        rclcpp::QoS qosBestEffort(10);
        qosBestEffort.reliability(rclcpp::ReliabilityPolicy::BestEffort).durability(rclcpp::DurabilityPolicy::Volatile).history(rclcpp::HistoryPolicy::KeepLast);

        _subscriberFrontCamera =
            this->create_subscription<sensor_msgs::msg::Image>("/zed/zed_node/left/image_rect_color", 10, std::bind(&Proc_vision_ros2::messageFrontCameraCallBack, this, _1));

        _subscriberBottomCameraSim =
            this->create_subscription<sensor_msgs::msg::Image>("/proc_simulation/front", 10, std::bind(&Proc_vision_ros2::messageFrontCameraCallBack, this, _1));

        _subscriberBottomCamera =
            this->create_subscription<sensor_msgs::msg::Image>("/camera_array/bottom/image_raw", qosBestEffort, std::bind(&Proc_vision_ros2::messageBottomCameraCallBack, this, _1));

        _subscriberBottomCameraSim =
            this->create_subscription<sensor_msgs::msg::Image>("/proc_simulation/bottom", 10, std::bind(&Proc_vision_ros2::messageBottomCameraCallBack, this, _1));


        _publisherDetectionArrayFront =
            this->create_publisher<sonia_common_ros2::msg::DetectionArray>("/proc_vision/bottom/classif", 10);

        _publisherDetectionArrayBottom =
            this->create_publisher<sonia_common_ros2::msg::DetectionArray>("/proc_vision/front/classif", 10);
        
        _subscriberZedDepth =
            this->create_subscription<sensor_msgs::msg::Image>("/zed/zed_node/depth/depth_registered", 10, std::bind(&Proc_vision_ros2::messageZedDepthCallBack, this, _1));
    }

    Proc_vision_ros2::~Proc_vision_ros2(){
        if(_cameraFront){
            _modelFront.~Yolo();
        }
        if(_cameraBottom){
            _modelBottom.~Yolo();
        }
    }

    void Proc_vision_ros2::processActuatorRequest(const std::shared_ptr<sonia_common_ros2::srv::AiActivationService::Request> request,
                                   std::shared_ptr<sonia_common_ros2::srv::AiActivationService::Response> response){
        vector<string> model_list = this->get_parameter("models").get_value<vector<string>>();
        
        string model_name;
        if(request.get()->model_choice >=0 and request.get()->model_choice <= model_list.size()){
            model_name = model_list[request.get()->model_choice];
        }else{
            model_name = model_list[0];
        }

        if(request.get()->camera_choice == request.get()->FRONT){
            _cameraBottom =false;
            _cameraFront =true;
            if(_modelFront.getInit()){
                _modelFront.~Yolo();
                _modelFront = new Yolo();
            }
            _modelFront = new Yolo(MODELDIR+model_name,logger);
        }else if(request.get()->camera_choice == request.get()->BOTTOM){
            _cameraBottom =true;
            _cameraFront =false;
            if(_modelBottom.getInit()){
                _modelBottom.~Yolo();
                _modelBottom = new Yolo();
            }
            _modelBottom = new Yolo(MODELDIR+model_name,logger);
        }else if(request.get()->camera_choice == request.get()->BOTH) {
            _cameraBottom =true;
            _cameraFront =true;
            if(_modelFront.getInit()){
                _modelFront.~Yolo();
                _modelFront = new Yolo();
            }
            if(_modelBottom.getInit()){
                _modelBottom.~Yolo();
                _modelBottom = new Yolo();
            }
            _modelFront = new Yolo(MODELDIR+model_name,logger);
            _modelBottom = new Yolo(MODELDIR+model_name,logger);
        }else{
            _cameraBottom = false;
            _cameraFront = false;
        }
    }

    void Proc_vision_ros2::messageFrontCameraCallBack(const sensor_msgs::msg::Image::SharedPtr &msg){
        if(_cameraFront){
            _publisherDetectionArrayFront->publish(imgDetection(msg,_modelFront));
        }
    }

    void Proc_vision_ros2::messageBottomCameraCallBack(const sensor_msgs::msg::Image::SharedPtr &msg){
        if(_cameraBottom){
            _publisherDetectionArrayBottom->publish(imgDetection(msg,_modelBottom));
        }
    }

    sonia_common_ros2::msg::DetectionArray Proc_vision_ros2::imgDetection(const sensor_msgs::msg::Image::SharedPtr &msg, Yolo model){
        try
        {
            actualiseDepp();
            detections.detected_object={};
            auto temp = cv_bridge::toCvCopy(msg);
            model.detect(temp->image,temp->header.frame_id,detections);
            if(_cameraFront){
                for(sonia_common_ros2::msg::Detection detection : detections.detected_object){
                    detection.distance = getDeepIstogram((int)detection.top_left_x,(int)detection.top_left_y,(int)detection.bottom_right_x,(int)detection.bottom_right_y);
                    // vector<float> angle = {0.0f,0.0f,0.0f,0.0f};
                    // getAngle((int)detection.top_left_x,(int)detection.top_left_y,(int)detection.bottom_right_x,(int)detection.bottom_right_y, angle);
                    // detection.angle_alpha = angle[0];
                    // //detection.ditance_beta = angle[1];
                    // detection.angle_teta = angle[2];
                    // detection.distance_teta = angle[3];
                }
            }
        }
        catch(const std::exception& e)
        {
            RCLCPP_INFO(this->get_logger(),  "ERROR when infer on the iamge");
            detections.detected_object = {};
            return detections;
        }
        
    }

    void Proc_vision_ros2::messageZedDepthCallBack(const sensor_msgs::msg::Image::SharedPtr &msg){
        try
        {
            _lastDepth = cv_bridge::toCvCopy(msg)->image;
        }
        catch(const std::exception& e)
        {
            RCLCPP_INFO(this->get_logger(),  "ERROR when read the message for the depth");
        }
        
    }

    void Proc_vision_ros2::actualiseDepp(){
        _actualDepth = _lastDepth;
    }

    float Proc_vision_ros2::getDeepIstogram(int x1, int y1, int x2, int y2){
        cv::Range rows(min(max(0,x1),IMAGEWIDTH), min(max(0,x2),IMAGEWIDTH));
        cv::Range cols(min(max(0,y1),IMAGEHEIGTH), min(max(0,y2),IMAGEHEIGTH));
        Mat subMatrice = _actualDepth(rows,cols);
        cv::Mat hist;
        int histSize[] = {6553};
        float range[] = {0, 65535};
        const float* ranges[] = {range};
        int channels[] = {0};

        cv::calcHist(&subMatrice, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;

        cv::minMaxLoc(hist, &minVal, &maxVal, &minLoc, &maxLoc);
        return maxVal/_UNIT;
    }

    void Proc_vision_ros2::getAngle(int x1, int y1, int x2, int y2, vector<float> angle){
    }
}