import rclpy
from .node_to_see import NodeTosee


def main():
    rclpy.init()
    node = NodeTosee()

    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == "__main__":
    main()
