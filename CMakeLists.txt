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
find_package(sonia_common_cpp REQUIRED)
find_package(sonia_common_ros2 REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please install the CUDA Toolkit.")
endif()

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)

set(TENSORRT_PATH "/usr/local/cuda/")

set( SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 80 86 87)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
)
        
include_directories(include ${TENSORRT_PATH}/include
  /usr/local/cuda/include)

file(GLOB SOURCES src/*.cpp)

set_source_files_properties(src/preprocess.cu PROPERTIES LANGUAGE CUDA)

cuda_add_executable(${PROJECT_NAME} ${SOURCES}
  src/preprocess.cu)

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

ament_target_dependencies(${PROJECT_NAME} rclcpp sonia_common_cpp sonia_common_ros2 std_msgs std_srvs cv_bridge sensor_msgs image_transport OpenCV CUDA)

set(TENSORRT_LIBS
        "/usr/lib/aarch64-linux-gnu/libnvinfer.so"
        "/usr/lib/aarch64-linux-gnu/libnvonnxparser.so"
        "/usr/lib/aarch64-linux-gnu/libnvparsers.so"
        "/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so")

# Link libraries to the target
target_link_libraries(
    ${PROJECT_NAME}
    ${TENSORRT_LIBS}
    CUDA::cudart
)

set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "80;86;87")

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
