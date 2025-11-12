#pragma once
#include "sonia_common_ros2/msg/detection_array.hpp"
#include "sonia_common_ros2/msg/detection.hpp"
#include <yaml-cpp/yaml.h>

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

#include "proc_vision_ros2/Proc_vision.hpp"

using namespace nvinfer1;
using namespace std;
using namespace cv;


namespace proc_vision_ros2
{
    class Yolo{
        public:

            Yolo(){};

            /**
             * @brief Constructor to initialize the YOLOv11 object.
             *
             * Loads the model and initializes TensorRT objects.
             *
             * @param model_path Path to the model engine or ONNX file.
             * @param logger Reference to a TensorRT logger for error reporting.
             */
            Yolo(string model_path, nvinfer1::ILogger& logger,proc_vision_ros2::Proc_vision proc_vision);

            /**
             * @brief Destructor to clean up resources.
             *
             * Frees the allocated memory and TensorRT resources.
             */
            ~Yolo();

            /**
             * @brief launch the process to detect on a image
             *
             */
            int detect(Mat& image,string frameID,sonia_common_ros2::msg::DetectionArray& output);

        private:
            /**
             * @brief Initialize TensorRT components from the given engine file.
             *
             * @param engine_path Path to the serialized TensorRT engine file.
             * @param logger Reference to a TensorRT logger for error reporting.
             */
            void init(std::string engine_path, nvinfer1::ILogger& logger,proc_vision_ros2::Proc_vision proc_vision);

            /**
             * @brief Preprocess the input image.
             *
             * Prepares the image for inference by resizing and normalizing it.
             *
             * @param image The input image to be preprocessed.
             */
            void preprocess(Mat& image);

            /**
             * @brief Run inference on the preprocessed image.
             *
             * load all parameter of a model.
             */
            void loadingParam();

            /**
             * @brief Run inference on the preprocessed image.
             *
             * Executes the TensorRT engine for object detection.
             */
            void infer();

            /**
             * @brief Postprocess the output from the model.
             *
             * Filters and decodes the raw output from the TensorRT engine into detection results.
             *
             * @param output A vector to store the detected objects.
             */
            int postprocess(sonia_common_ros2::msg::DetectionArray& output);

            /**
             * @brief Build the TensorRT engine from the ONNX model.
             *
             * @param onnxPath Path to the ONNX file.
             * @param logger Reference to a TensorRT logger for error reporting.
             */
            void build(std::string onnxPath, nvinfer1::ILogger& logger);

            /**
             * @brief Save the TensorRT engine to a file.
             *
             * @param filename Path to save the serialized engine.
             * @return True if the engine was saved successfully, false otherwise.
             */
            bool saveEngine(const std::string& filename);

            float* gpu_buffers[2]; //!< The vector of device buffers needed for engine execution.
            float* cpu_output_buffer; //!< Pointer to the output buffer on the host.

            cudaStream_t stream; //!< CUDA stream for asynchronous execution.
            IRuntime* runtime; //!< The TensorRT runtime used to deserialize the engine.
            ICudaEngine* engine; //!< The TensorRT engine used to run the network.
            IExecutionContext* context; //!< The context for executing inference using an ICudaEngine.

            // Model parameters
            int input_w; //!< Width of the input image.
            int input_h; //!< Height of the input image.
            int num_detections; //!< Number of detections output by the model.
            int detection_attribute_size; //!< Size of each detection attribute.
            int num_classes = 22; //!< Number of object classes that can be detected.
            const int MAX_IMAGE_SIZE = 2208 * 1242; //!< Maximum allowed input image size.
            float conf_threshold = 0.3f; //!< Confidence threshold for filtering detections.
            float nms_threshold = 0.4f; //!< Non-Maximum Suppression (NMS) threshold for filtering overlapping boxes.

            string frameId;

            YAML::Node _config;
    };
}