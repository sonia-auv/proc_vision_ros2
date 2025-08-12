import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory("proc_vision_ros2"), "config", "config.yaml"
    )
    return LaunchDescription(
        [
            Node(
                package="proc_vision_ros2",
                executable="proc_vision_ros2",
                parameters=[config],
                arguments=[('__log_level:=debug')]
            )
        ]
    )
