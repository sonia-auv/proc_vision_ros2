cmake_minimum_required(VERSION 2.8.3)
project(proc_vision_ros2)


#===============================================================================
# C A T K I N   C O N F I G U R A T I O N

# find_package(catkin REQUIRED COMPONENTS
#   rospy
#   sonia_common
# )

#===============================================================================
# G E N E R A T E   M E S S A G E S   &   S E R V I C E S

catkin_python_setup()

find_package(catkin REQUIRED COMPONENTS
    rospy
    sensor_msgs
    sonia_common
)

catkin_package(
    CATKIN_DEPENDS
    rospy
    roscpp
    message_runtime
    cv_bridge
    roslaunch
    sonia_common
)

#===============================================================================
# I N C L U D E   F I L E S

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#===============================================================================
# I N S T A L L`
catkin_install_python(PROGRAMS proc_vision_ros2/vision_node.py
 DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#===============================================================================
# U N I T   T E S T S

# Add folders to be run by python nosetests
#catkin_add_nosetests(test)