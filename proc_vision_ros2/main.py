import rclpy
from .vision_node import VisionNode


def main():
    rclpy.init()
    node = VisionNode()

    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == "__main__":
    main()