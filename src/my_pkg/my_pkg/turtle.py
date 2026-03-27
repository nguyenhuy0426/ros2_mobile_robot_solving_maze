import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class Turtle2CmdPublisher(Node):
    def __init__(self):
        super().__init__('turtle2_cmd_publisher')

        self.pub = self.create_publisher(
            Twist,
            '/turtle2/cmd_vel',
            10
        )

        self.timer = self.create_timer(0.1, self.publish_cmd)

    def publish_cmd(self):
        msg = Twist()
        msg.linear.x = 2.0     # đi thẳng
        msg.angular.z = 1.0   # quay

        self.pub.publish(msg)
        self.get_logger().info(
            f'Publish: linear.x={msg.linear.x}, angular.z={msg.angular.z}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = Turtle2CmdPublisher()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()