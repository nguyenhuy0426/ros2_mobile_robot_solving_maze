#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelPub1s(Node):
    def __init__(self):
        super().__init__('pc_cmdvel_pub_1s')
        self.pub = self.create_publisher(Twist, '/PC', 10)

        # vận tốc bạn muốn gửi
        self.vx = -0.1
        self.vy =   0.0
        self.wz = 0.0

        # gửi mỗi 1 giây
        self.create_timer(1.0, self.publish_cmd)

    def publish_cmd(self):
        msg = Twist()
        msg.linear.x = self.vx
        msg.linear.y = self.vy
        msg.angular.z = self.wz
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(CmdVelPub1s())
    rclpy.shutdown()

if __name__ == '__main__':
    main()