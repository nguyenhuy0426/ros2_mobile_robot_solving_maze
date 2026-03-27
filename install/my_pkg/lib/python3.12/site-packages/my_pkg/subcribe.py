#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelSubTest(Node):
    def __init__(self):
        super().__init__('cmdvel_sub_test')
        self.create_subscription(
            Twist,
            '/PC',
            self.cb,
            10
        )
        self.get_logger().info("Listening /cmd_vel ...")

    def cb(self, msg: Twist):
        self.get_logger().info(
            f"RECV cmd_vel -> "
            f"vx={msg.linear.x:.2f}, "
            f"vy={msg.linear.y:.2f}, "
            f"wz={msg.angular.z:.2f}"
        )


def main():
    rclpy.init()
    rclpy.spin(CmdVelSubTest())
    rclpy.shutdown()


if __name__ == '__main__':
    main()