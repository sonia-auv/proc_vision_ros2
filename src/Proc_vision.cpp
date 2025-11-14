#include "proc_vision_ros2/Proc_vision.hpp"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

using std::placeholders::_1;
using std::placeholders::_2;

namespace proc_vision_ros2
{
    Proc_vision::Proc_vision()
        : Node("proc_vision"){

        this->declare_parameter("models", std::vector<string>({"yolo11l"}));

        MODELDIR = (string)std::getenv("SONIA_WS")+"/src/proc_vision_ros2/models/";

        rclcpp::QoS qosBestEffort(10);
        qosBestEffort.reliability(rclcpp::ReliabilityPolicy::BestEffort).durability(rclcpp::DurabilityPolicy::Volatile).history(rclcpp::HistoryPolicy::KeepLast);

        _subscriberFrontCamera =
            this->create_subscription<sensor_msgs::msg::Image>("/zed/zed_node/left/image_rect_color", 10, std::bind(&Proc_vision::messageFrontCameraCallBack, this, _1));

        _subscriberFrontCameraSim =
            this->create_subscription<sensor_msgs::msg::Image>("/proc_simulation/front", 10, std::bind(&Proc_vision::messageFrontCameraCallBack, this, _1));

        _subscriberBottomCamera =
            this->create_subscription<sensor_msgs::msg::Image>("/camera_array/bottom/image_raw", qosBestEffort, std::bind(&Proc_vision::messageBottomCameraCallBack, this, _1));

        _subscriberBottomCameraSim =
            this->create_subscription<sensor_msgs::msg::Image>("/proc_simulation/bottom", 10, std::bind(&Proc_vision::messageBottomCameraCallBack, this, _1));


        _publisherDetectionArrayFront =
            this->create_publisher<sonia_common_ros2::msg::DetectionArray>("/proc_vision/front/classif", 10);

        _publisherDetectionArrayBottom =
            this->create_publisher<sonia_common_ros2::msg::DetectionArray>("/proc_vision/bottom/classif", 10);
        
        _subscriberZedDepth =
            this->create_subscription<sensor_msgs::msg::Image>("/zed/zed_node/depth/depth_registered", 10, std::bind(&Proc_vision::messageZedDepthCallBack, this, _1));

        _aiActivationService = this->create_service<sonia_common_ros2::srv::AiActivationService>(
            "/proc_vision/ai_activation", std::bind(&Proc_vision::processAiActivationRequest, this, _1, _2));
        
        int driverVersion;

        // Get the CUDA driver version
        cudaError_t error = cudaDriverGetVersion(&driverVersion);

        if (error == cudaSuccess) {
            // Extract major, minor, and patch versions
            int major = driverVersion / 1000;
            int minor = (driverVersion % 1000) / 10;
            int patch = driverVersion % 10;

            RCLCPP_INFO_STREAM(this->get_logger(),  "CUDA Driver Version: " << major << "." << minor << "." << patch);
        } else {
            RCLCPP_INFO_STREAM(this->get_logger(),  "CUDA Driver Version: " << cudaGetErrorString(error));
        }
        
        RCLCPP_INFO(this->get_logger(),  "Started");
    }

    Proc_vision::~Proc_vision(){
        _modelBottom = NULL;
        _modelFront = NULL;
    }

    void Proc_vision::processAiActivationRequest(const std::shared_ptr<sonia_common_ros2::srv::AiActivationService::Request> request,
                                   std::shared_ptr<sonia_common_ros2::srv::AiActivationService::Response> response){
        vector<string> model_list = this->get_parameter("models").get_value<vector<string>>();

        try
        {
            string model_name;
            if(request.get()->model_choice >=0 and request.get()->model_choice <= model_list.size()){
                model_name = model_list[request.get()->model_choice];
            }else{
                model_name = model_list[0];
            }

            if(request.get()->camera_choice == request.get()->FRONT){
                _modelFront = new Yolo(MODELDIR+model_name,logger);
                _modelBottom = NULL;
                _cameraBottom =false;
                _cameraFront =true;
            }else if(request.get()->camera_choice == request.get()->BOTTOM){
                _modelFront = NULL;
                _modelBottom = new Yolo(MODELDIR+model_name,logger);
                _cameraBottom =true;
                _cameraFront =false;
            }else if(request.get()->camera_choice == request.get()->BOTH) {
                _modelFront = new Yolo(MODELDIR+model_name,logger);
                _modelBottom = new Yolo(MODELDIR+model_name,logger);
                _cameraBottom =true;
                _cameraFront =true;
            }else{
                _cameraBottom = false;
                _cameraFront = false;
                _modelBottom = NULL;
                _modelFront = NULL;
            }

            response->model_name = model_name;
        }
        catch(const std::exception& e)
        {
            RCLCPP_INFO_STREAM(this->get_logger(),  "ERROR on the load of the model " << e.what());
        }
    }

    void Proc_vision::messageFrontCameraCallBack(const sensor_msgs::msg::Image &msg){
        if(_cameraFront){
            _publisherDetectionArrayFront->publish(imgDetection(msg,_modelFront));
        }
    }

    void Proc_vision::messageBottomCameraCallBack(const sensor_msgs::msg::Image &msg){
        if(_cameraBottom){
            _publisherDetectionArrayBottom->publish(imgDetection(msg,_modelBottom));
        }
    }

    sonia_common_ros2::msg::DetectionArray Proc_vision::imgDetection(const sensor_msgs::msg::Image &msg, Yolo* model){
        try
        {
	        RCLCPP_INFO(this->get_logger(),  "start");
            actualiseDepth();
            detections.detected_object={};
            auto imageCV2 = cv_bridge::toCvCopy(msg);
            RCLCPP_INFO_STREAM(this->get_logger(),  "Number detection "<<std::to_string(model->detect(imageCV2->image,imageCV2->header.frame_id,detections)));
            if(_cameraFront){
                for(sonia_common_ros2::msg::Detection detection : detections.detected_object){
                    detection.distance = getDeepHistogram((int)detection.top_left_x,(int)detection.top_left_y,(int)detection.bottom_right_x,(int)detection.bottom_right_y);
                    getAngle((int)detection.top_left_x,(int)detection.top_left_y,(int)detection.bottom_right_x,(int)detection.bottom_right_y, _angle);
                    detection.angle_alpha = _angle.angle_alpha;
                    detection.distance_beta = _angle.distance_beta;
                    detection.angle_teta = _angle.angle_teta;
                    detection.distance_teta = _angle.distance_teta;
                }
            }
	        RCLCPP_INFO(this->get_logger(),  "end");
            return detections;
        }
        catch(const std::exception& e)
        {
            RCLCPP_INFO_STREAM(this->get_logger(),  "ERROR when infer on the iamge " << e.what());
            detections.detected_object = {};
            return detections;
        }
        
    }

    void Proc_vision::messageZedDepthCallBack(const sensor_msgs::msg::Image &msg){
        try
        {
            _lastDepth = cv_bridge::toCvCopy(msg)->image;
        }
        catch(const std::exception& e)
        {
            RCLCPP_INFO(this->get_logger(),  "ERROR when read the message for the depth");
        }
        
    }

    void Proc_vision::actualiseDepth(){
        _actualDepth = _lastDepth;
    }

    float Proc_vision::getDeepHistogram(int x1, int y1, int x2, int y2){
        cv::Range rows(min(max(0,x1),IMAGEWIDTH), min(max(0,x2),IMAGEWIDTH));
        cv::Range cols(min(max(0,y1),IMAGEHEIGTH), min(max(0,y2),IMAGEHEIGTH));
        Mat subMatrice = _actualDepth(rows,cols);
        cv::Mat hist;
        int histSize[] = {6553};
        float range[] = {0, 65530};
        const float* ranges[] = {range};
        int channels[] = {0};

        cv::calcHist(&subMatrice, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;

        cv::minMaxLoc(hist, &minVal, &maxVal, &minLoc, &maxLoc);
        return maxVal/_UNIT;
    }

    void Proc_vision::getAngle(int x1, int y1, int x2, int y2, AngleDetection angle){
        int x10 = (x2-x1)/10;
        cv::Range rowsLeft(min(max(0,x1),IMAGEWIDTH)+ x10, min(max(0,x2),IMAGEWIDTH)+ x10*2);
        cv::Range colsLeft(min(max(0,y1),IMAGEHEIGTH), min(max(0,y2),IMAGEHEIGTH));
        cv::Range rowsMid((min(max(0,x1),IMAGEWIDTH) + min(max(0,x2),IMAGEWIDTH))/2 - x10/2, (min(max(0,x2),IMAGEWIDTH)+ min(max(0,x2),IMAGEWIDTH))/2 + x10/2);
        cv::Range colsMid(min(max(0,y1),IMAGEHEIGTH), min(max(0,y2),IMAGEHEIGTH));

        Mat subMatriceLeft = _actualDepth(rowsLeft,colsLeft);
        Mat subMatriceMid = _actualDepth(rowsMid,colsMid);
        cv::Mat histLeft;
        cv::Mat histMid;
        int histSize[] = {2500};
        float range[] = {0, 25000};
        const float* ranges[] = {range};
        int channels[] = {0};

        cv::calcHist(&subMatriceLeft, 1, channels, cv::Mat(), histLeft, 1, histSize, ranges, true, false);
        cv::calcHist(&subMatriceLeft, 1, channels, cv::Mat(), histMid, 1, histSize, ranges, true, false);

        double minValLeft, distanceLeft, distanceMid;
        cv::Point minLocLeft, maxLocleft;

        cv::minMaxLoc(histLeft, &minValLeft, &distanceLeft, &minLocLeft, &maxLocleft);
        cv::minMaxLoc(histMid, &minValLeft, &distanceMid, &minLocLeft, &maxLocleft);

        int centreX = (x1+x2)/2;
        int centreY = (y1+y2)/2;

        double angleX = (IMAGEWIDTH/2 -centreX)*ZEDHFOV/IMAGEWIDTH;
        double angleY = (IMAGEHEIGTH/2 -centreY)*ZEDVFOV/IMAGEHEIGTH;
        double angle2 = (IMAGEWIDTH/2 -x1+x10)*ZEDHFOV/IMAGEWIDTH;

        double pointXMid = cos(angleX) * distanceMid;
        double pointYMid = sin(angleX) * distanceMid;

        double pointXLeft = cos(angle2) * distanceMid;
        double pointYLeft = sin(angle2) * distanceMid;

        double pointSubY = cos(angleY) * distanceMid;

        double hypo = sqrt((pointXLeft-pointXMid)*(pointXLeft-pointXMid)+(pointYLeft-pointYMid)*(pointYLeft-pointYMid));

        if(hypo < 0.01){
            angle.angle_alpha = angleX;
            angle.distance_beta = pointSubY/_UNIT;
            angle.angle_teta = 0.0;
            angle.distance_teta = 0.0;
        }else{
            double angleTeta = - asin((pointXMid - pointXLeft)/hypo);
            double newDistanceY = sin(angleTeta)*pointXMid+ cos(angleTeta)*pointYMid;
            angle.angle_alpha = angleX;
            angle.distance_beta = pointSubY/_UNIT;
            angle.angle_teta = angleTeta;
            angle.distance_teta = newDistanceY/_UNIT;
        }
    }
}
