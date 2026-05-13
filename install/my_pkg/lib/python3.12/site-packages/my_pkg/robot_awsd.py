#!/usr/bin/env python3
import sys, tty, termios, select
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class Teleop(Node):
    def __init__(self):
        super().__init__('teleop')
        self.pub = self.create_publisher(Twist, '/PC', 10)

    def get_key(self, timeout=0.05):  # 20Hz
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if select.select([sys.stdin], [], [], timeout)[0]:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def run(self):
        while rclpy.ok():
            k = self.get_key()

            msg = Twist()  # mặc định = 0,0,0 (không bấm)

            if k == 'w': msg.linear.x = 0.03
            elif k == 's': msg.linear.x = -0.03
            elif k == 'a': msg.angular.z = 0.5
            elif k == 'd': msg.angular.z = -0.5
            elif k == '\x03': break  # Ctrl+C

            self.pub.publish(msg)

def main():
    rclpy.init()
    node = Teleop()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()