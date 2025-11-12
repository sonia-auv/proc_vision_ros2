#include "rclcpp/rclcpp.hpp"
#include "proc_vision_ros2/Proc_vision.hpp"
#include <cstdlib>

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    std::shared_ptr<proc_vision_ros2::Proc_vision> proc_vision = std::make_shared<proc_vision_ros2::Proc_vision>();

    rclcpp::spin(proc_vision);

    rclcpp::shutdown();
    return EXIT_SUCCESS;
}
