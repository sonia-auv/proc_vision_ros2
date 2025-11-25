#include "proc_vision_ros2/Yolo.hpp"
#include "proc_vision_ros2/logging.h"             // Logging utilities
#include "proc_vision_ros2/cuda_utils.h"          // CUDA utility functions
#include "proc_vision_ros2/macros.h"              // Common macros
#include "proc_vision_ros2/preprocess.h"          // Preprocessing functions
#include <NvOnnxParser.h>        // NVIDIA ONNX parser for TensorRT
#include <fstream>
#include <iostream>
#include <cstring>

namespace proc_vision_ros2
{
    // Initialize a static logger instance
    static Logger logger;

    // Define whether to use FP16 precision
    #define isFP16 true

    // Define whether to perform model warmup
    #define warmup false

    // Constructor for the Yolo class
    Yolo::Yolo(string model_path, nvinfer1::ILogger& logger)
    {

        std::ifstream file(model_path+"/model.engine");

        if(file.good()){
            // Initialize the engine from a serialized engine file
            init(model_path+"/model.engine", logger);
        }else{
            std::ifstream fileOnnx(model_path+"/model.onnx");
            if(fileOnnx.good()){
                string model_path_engine = model_path+"/model.onnx";
                // Build the engine from an ONNX model
                build(model_path_engine, logger);
                // Save the built engine to a file
                saveEngine(model_path_engine);
            }
        }

        _config = YAML::LoadFile(model_path+"/data.yaml")["names"];

        // Handle input dimensions based on TensorRT version
        #if NV_TENSORRT_MAJOR < 10
            // For TensorRT versions less than 10, get binding dimensions directly
            auto input_dims = engine->getBindingDimensions(0);
            input_h = input_dims.d[2];
            input_w = input_dims.d[3];
        #else
            // For TensorRT versions 10 and above, use getTensorShape
            auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
            input_h = input_dims.d[2];
            input_w = input_dims.d[3];
        #endif
    }

    // Initialize the engine from a serialized engine file
    void Yolo::init(std::string engine_path, nvinfer1::ILogger& logger)
    {
        // Open the engine file in binary mode
        ifstream engineStream(engine_path, ios::binary);
        // Move to the end to determine file size
        engineStream.seekg(0, ios::end);
        const size_t modelSize = engineStream.tellg();
        // Move back to the beginning of the file
        engineStream.seekg(0, ios::beg);
        // Allocate memory to read the engine data
        unique_ptr<char[]> engineData(new char[modelSize]);
        // Read the engine data into memory
        engineStream.read(engineData.get(), modelSize);
        engineStream.close();

        // Create a TensorRT runtime instance
        runtime = createInferRuntime(logger);
        // Deserialize the CUDA engine from the engine data
        engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
        // Create an execution context for the engine
        context = engine->createExecutionContext();

        loadingParam();
    }

    void Yolo::loadingParam(){

        // Retrieve input dimensions from the engine
        input_h = engine->getBindingDimensions(0).d[2];
        input_w = engine->getBindingDimensions(0).d[3];
        // Retrieve detection attributes and number of detections
        detection_attribute_size = engine->getBindingDimensions(1).d[1];
        num_detections = engine->getBindingDimensions(1).d[2];
        // Calculate the number of classes based on detection attributes
        num_classes = detection_attribute_size - 4;

        // Allocate CPU memory for output buffer
        cpu_output_buffer = new float[detection_attribute_size * num_detections];
        // Allocate GPU memory for input buffer (assuming 3 channels: RGB)
        CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
        // Allocate GPU memory for output buffer
        CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

        // Initialize CUDA preprocessing with maximum image size
        cuda_preprocess_init(MAX_IMAGE_SIZE);

        // Create a CUDA stream for asynchronous operations
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Perform model warmup if enabled
        if (warmup) {
            for (int i = 0; i < 10; i++) {
                this->infer(); // Run inference to warm up the model
            }
        }
    }

    // Destructor for the Yolo class
    Yolo::~Yolo()
    {
        // Synchronize and destroy the CUDA stream
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        // Free allocated GPU buffers
        for (int i = 0; i < 2; i++)
            CUDA_CHECK(cudaFree(gpu_buffers[i]));
        // Free CPU output buffer
        delete[] cpu_output_buffer;

        // Destroy CUDA preprocessing resources
        cuda_preprocess_destroy();
        // Delete TensorRT context, engine, and runtime
        delete context;
        delete engine;
        delete runtime;
    }

    // Preprocess the input image and transfer it to the GPU buffer
    void Yolo::preprocess(Mat& image) {
        // Perform CUDA-based preprocessing
        cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
        // Synchronize the CUDA stream to ensure preprocessing is complete
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Perform inference using the TensorRT execution context
    void Yolo::infer()
    {
    #if NV_TENSORRT_MAJOR < 10
        // For TensorRT versions less than 10, use enqueueV2 with GPU buffers
        context->enqueueV2((void**)gpu_buffers, stream, nullptr);
    #else
        // For TensorRT versions 10 and above, use enqueueV3 with the CUDA stream
        this->context->enqueueV3(this->stream);
    #endif
    }

    // Postprocess the inference output to extract detections
    int Yolo::postprocess(sonia_common_ros2::msg::DetectionArray& output)
    {
        int nbDetec = 0;
        // Asynchronously copy output from GPU to CPU
        CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        // Synchronize the CUDA stream to ensure copy is complete
        CUDA_CHECK(cudaStreamSynchronize(stream));

        vector<Rect> boxes;          // Bounding boxes
        vector<int> class_ids;       // Class IDs
        vector<float> confidences;   // Confidence scores

        // Create a matrix view of the detection output
        const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

        // Iterate over each detection
        for (int i = 0; i < det_output.cols; ++i) {
            // Extract class scores for the current detection
            const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
            Point class_id_point;
            double score;
            // Find the class with the maximum score
            minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

            // Check if the confidence score exceeds the threshold
            if (score > conf_threshold) {
                // Extract bounding box coordinates
                const float cx = det_output.at<float>(0, i);
                const float cy = det_output.at<float>(1, i);
                const float ow = det_output.at<float>(2, i);
                const float oh = det_output.at<float>(3, i);
                Rect box;
                // Calculate top-left corner of the bounding box
                if (imageHeightFactor > imageWidthFactor){
                    box.x = static_cast<int>((cx - 0.5 * ow)/ imageWidthFactor);
                    box.y = static_cast<int>(((cy - 0.5 * oh) - (input_h - imageHeightFactor * imageHeight) / 2) / imageWidthFactor);
                    // Set width and height of the bounding box
                    box.width = static_cast<int>(ow / imageWidthFactor);
                    box.height = static_cast<int>(oh / imageWidthFactor);
                }else{
                    box.x = static_cast<int>(((cx - 0.5 * ow) - (input_h - imageWidthFactor * imageWidth) / 2) / imageHeightFactor);
                    box.y = static_cast<int>((cy - 0.5 * oh) / imageHeightFactor);
                    // Set width and height of the bounding box
                    box.width = static_cast<int>(ow / imageHeightFactor);
                    box.height = static_cast<int>(oh / imageHeightFactor);
                }

                // Store the bounding box, class ID, and confidence
                boxes.push_back(box);
                class_ids.push_back(class_id_point.y);
                confidences.push_back(score);
            }
        }

        vector<int> nms_result; // Indices after Non-Maximum Suppression (NMS)
        // Apply NMS to remove overlapping boxes
        dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

        // Iterate over NMS results and populate the output detections
        for (int i = 0; i < nms_result.size(); i++)
        {
            ++nbDetec;
            sonia_common_ros2::msg::Detection result;
            int idx = nms_result[i];
            result.class_name = _config[class_ids[idx]].as<std::string>();//; 
            result.confidence = confidences[idx];
            result.top_left_x = boxes[idx].x;
            result.top_left_y = boxes[idx].y;
            result.top_right_x = boxes[idx].x;
            result.top_right_y = boxes[idx].y + boxes[idx].height;
            result.top_left_x = boxes[idx].x + boxes[idx].width;
            result.top_left_y = boxes[idx].y;
            result.bottom_right_x = boxes[idx].x + boxes[idx].width;
            result.bottom_right_y = boxes[idx].y + boxes[idx].height;
            result.frame_id = frameId;
            
            output.detected_object.push_back(result);
        }
        return nbDetec;
    }

    // Build the TensorRT engine from an ONNX model
    void Yolo::build(std::string onnxPath, nvinfer1::ILogger& logger)
    {
        // Create a TensorRT builder
        auto builder = createInferBuilder(logger);
        // Define network flags for explicit batch dimensions
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        // Create a network definition with explicit batch
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
        // Create builder configuration
        IBuilderConfig* config = builder->createBuilderConfig();
        // Enable FP16 precision if specified
        if (isFP16)
        {
            config->setFlag(BuilderFlag::kFP16);
        }
        // Create an ONNX parser
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
        // Parse the ONNX model file
        bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
        // Build the serialized network plan
        IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

        // Create a TensorRT runtime
        runtime = createInferRuntime(logger);

        // Deserialize the CUDA engine from the serialized plan
        engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

        // Create an execution context for the engine
        context = engine->createExecutionContext();

        // Clean up allocated resources
        delete network;
        delete config;
        delete parser;
        delete plan;
    }

    // Save the serialized TensorRT engine to a file
    bool Yolo::saveEngine(const std::string& onnxpath)
    {
        // Generate the engine file path by replacing the extension with ".engine"
        std::string engine_path;
        size_t dotIndex = onnxpath.find_last_of(".");
        if (dotIndex != std::string::npos) {
            engine_path = onnxpath.substr(0, dotIndex) + ".engine";
        }
        else
        {
            return false; // Return false if no extension is found
        }

        // Check if the engine is valid
        if (engine)
        {
            // Serialize the engine
            nvinfer1::IHostMemory* data = engine->serialize();
            std::ofstream file;
            // Open the engine file in binary write mode
            file.open(engine_path, std::ios::binary | std::ios::out);
            if (!file.is_open())
            {
                std::cout << "Create engine file " << engine_path << " failed" << std::endl;
                return false;
            }
            // Write the serialized engine data to the file
            file.write((const char*)data->data(), data->size());
            file.close();

            // Load all parameters to infer
            loadingParam();

            // Free the serialized data memory
            delete data;
        }

        // Load all parameters to infer
        loadingParam();

        return true;
    }

    // Save the serialized TensorRT engine to a file
    int Yolo::detect(Mat& image,string frameID, sonia_common_ros2::msg::DetectionArray& output)
    {
        frameId = frameID;

        imageHeightFactor = input_h / image.rows;
        imageWidth = input_w / image.cols;

        imageHeight = image.rows;
        imageWidth = image.cols;

        // Preprocess the frame
        preprocess(image);

        // Perform inference
        infer();

        // Postprocess to get detections
        return postprocess(output);
    }
}
