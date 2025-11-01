# proc_vision_ros2

The project use a model of IA to infer on iage from the camera. It's use Yolo for the model

---

## Dependencies

### ROS 2 Distro

* Humble

### ROS 2 Packages

* `ament_cmake`
* `rclcpp`
* `std_msgs`
* `std_srvs`
* `cv_bridge`
* `sensor_msgs`
* `cv_bridge`
* `image_transport`
* `CUDAToolkit`
* `yaml-cpp`
* `OpenCV`
* `CUDA`

### Sonia packages

* `sonia_common_ros2`

### External packages

* `TensorRt`

---

## Node

* Name: `proc_vision`
* models: {"robosub-2025-v2"}

---

## Registered Topics / Services / Actions

| Type                             | Name                                       | Direction             | Message/Service Type                          | Description                                   |
| -------------------------------- | ------------------------------------------ | --------------------- | --------------------------------------------- | --------------------------------------------- |
| Topic                            | `/zed/zed_node/left/image_rect_color`      | Subscribed            | `sensor_msgs/msg/Image`                       | Image from the zed front cam                  |
| Topic                            | `/zed/zed_node/depth/depth_registered`     | Subscribed            | `sensor_msgs/msg/Image`                       | Depth image from the zed front cam            |
| Topic                            | `/camera_array/bottom/image_raw`           | Subscribed            | `sensor_msgs/msg/Image`                       | Image from the bottom cam                     |
| Topic                            | `/proc_simulation/front`                   | Subscribed            | `sensor_msgs/msg/Image`                       | Image from the simulation for the front cam   |
| Topic                            | `/proc_simulation/bottom`                  | Subscribed            | `sensor_msgs/msg/Image`                       | Image from the simulation for the bottom cam  |
| Topic                            | `/proc_vision/front/classif`               | Published             | `sonia_common_ros2/msg/Image`                 | Detection for the front image                 |
| Topic                            | `/proc_vision/bottom/classif`              | Published             | `sonia_common_ros2/msg/Image`                 | Detection for the bottom image                |
| Service                          | `/proc_vision/ai_activation`               | Service Server        | `sonia_common_ros2/srv/AiActivationService`   | Select the models and cam to use              |

---
## Build Instructions
To build the project, the following commands should be run directly from your ROS2 workspace.

```bash
colcon build --packages-select proc_vision_ros2 --symlink-install
source install/setup.bash
```

---

## Launch Instructions

### Default launch

you need to have the environnemental variable SONIA_WS set to launch

```bash
ros2 launch proc_vision_ros2 launch.py
```

---

## How to import

### 1) Importation

You need to train you IA and add in the project in models in a folder of the form:

    folder name
        data.yaml
        model.onnx

### 2) Transformation

Go to the folder you created

use 
$trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

It optimize your model and trasform in good format

### 3) add model

Open config/config.yaml

add in list of models your name of your ia

---

## Useful ROS 2 Commands

```bash
ros2 node list
ros2 node info /proc_vision_ros2
ros2 param list /proc_vision_ros2
```

```start Model
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
```

---

## References

* [sonia_common_ros2](https://github.com/sonia-auv/sonia_common_ros2)

---