cmake_minimum_required(VERSION 3.24)

project(proc_vision_ros2 LANGUAGES CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sonia_common_ros2 REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please install the CUDA Toolkit.")
endif()

set(TENSORRT_PATH "/usr/local/cuda/")
        
include_directories(include ${TENSORRT_PATH}/include
  /usr/local/cuda/targets/aarch64-linux/include)

file(GLOB SOURCES src/*.cu src/*.cpp )

cuda_add_executable(${PROJECT_NAME} ${SOURCES})

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)
install(TARGETS
  ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE API_EXPORTS)

ament_target_dependencies(${PROJECT_NAME} rclcpp sonia_common_ros2 std_msgs std_srvs cv_bridge sensor_msgs image_transport OpenCV CUDA yaml-cpp)

set(TENSORRT_LIBS
        "/usr/lib/aarch64-linux-gnu/libnvinfer.so"
        "/usr/lib/aarch64-linux-gnu/libnvonnxparser.so"
        "/usr/lib/aarch64-linux-gnu/libnvparsers.so"
        "/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so")

# Link libraries to the target
target_link_libraries(
    ${PROJECT_NAME}
    ${TENSORRT_LIBS}
    yaml-cpp
)

set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# set(CMAKE_CUDA_ARCHITECTURES "72")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
