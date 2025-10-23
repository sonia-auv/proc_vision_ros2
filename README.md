# template-ros2

# How to use

## Start node

To start the node use the command:
ros2 launch proc_vision_ros2 launch.py

## Start the run of the IA

To start the run of your model, enter this command:

ros2 service call /proc_vision/ai_activation sonia_common_ros2/srv/AiActivationService "{model_choice: X, camera_choice: Y}"

X: the models you want to launch

    Pick the models number you want to launch from config.yaml

    By default the most recent models is 0

Y: The camera you want to launch

    0: front
    1: bottom
    2: both

The models and camera can be change at any time during the run

# How to import

## 1) Importation

You need to train you IA and add in the project in models in a folder of the form:

    folder name
        data.yaml
        model.onnx

## 2) Transformation

Go to the folder you created

use 
$trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

It optimize your model and trasform in good format

## 3) add model

Open config/config.yaml

add in list of models your name of your ia

# Choice

## why use Tensorrt

Tensorrt add one major advantage to be more linked with the GPU so it's more optimized and run faster