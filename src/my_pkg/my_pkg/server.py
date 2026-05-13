#!/usr/bin/env python3
"""
Node điều khiển vận tốc xe bằng Service
Phần 2: Yêu cầu điều khiển bằng Service
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from my_interfaces.srv import MySrv


class VelocityService(Node):
    """
    Node điều khiển vận tốc xe bằng Service
    Nhận request với vận tốc từ [-1, 1]
    """
    
    def __init__(self):
        super().__init__('velocity_service')
        
        # Tạo service
        self.srv = self.create_service(
            MySrv,
            'set_velocity',
            self.handle_set_velocity
        )
        
        # Tạo publisher để gửi velocity commands
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Tham số vận tốc tối đa
        self.declare_parameter('max_linear_vel', 1.0)
        self.declare_parameter('max_angular_vel', 1.0)
        
        self.max_linear = self.get_parameter('max_linear_vel').value
        self.max_angular = self.get_parameter('max_angular_vel').value
        
        # Lưu vận tốc hiện tại
        self.current_velocity = {'vx': 0.0, 'vy': 0.0, 'wz': 0.0}
        
        self.get_logger().info('='*50)
        self.get_logger().info('Velocity Service Node Started')
        self.get_logger().info(f'Service name: /set_velocity')
        self.get_logger().info(f'Max linear velocity: {self.max_linear} m/s')
        self.get_logger().info(f'Max angular velocity: {self.max_angular} rad/s')
        self.get_logger().info('='*50)
        self.get_logger().info('Waiting for service calls...')
    
    def handle_set_velocity(self, request, response):
        """
        Xử lý service request
        """
        try:
            # Giới hạn giá trị trong khoảng [-1, 1]
            vx = max(-1.0, min(1.0, request.vx))
            vy = max(-1.0, min(1.0, request.vy))
            wz = max(-1.0, min(1.0, request.wz))
            
            # Lưu vận tốc hiện tại
            self.current_velocity = {'vx': vx, 'vy': vy, 'wz': wz}
            
            # Tạo và publish Twist message
            msg = Twist()
            msg.linear.x = vx * self.max_linear
            msg.linear.y = vy * self.max_linear
            msg.angular.z = wz * self.max_angular
            
            self.publisher.publish(msg)
            
            # Tạo response
            response.success = True
            response.message = (
                f'Velocity set successfully: '
                f'vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}'
            )
            
            self.get_logger().info('='*50)
            self.get_logger().info('Service Call Received:')
            self.get_logger().info(f'  Request: vx={request.vx:.3f}, '
                                 f'vy={request.vy:.3f}, wz={request.wz:.3f}')
            self.get_logger().info(f'  Actual: vx={msg.linear.x:.3f} m/s, '
                                 f'vy={msg.linear.y:.3f} m/s, '
                                 f'wz={msg.angular.z:.3f} rad/s')
            self.get_logger().info(f'  Status: SUCCESS')
            self.get_logger().info('='*50)
            
        except Exception as e:
            response.success = False
            response.message = f'Error: {str(e)}'
            self.get_logger().error('='*50)
            self.get_logger().error(f'Service Call Failed: {str(e)}')
            self.get_logger().error('='*50)
        
        return response


def main(args=None):
    rclpy.init(args=args)
    
    node = VelocityService()
    
    print("\n" + "="*60)
    print("HƯỚNG DẪN SỬ DỤNG - VELOCITY SERVICE")
    print("="*60)
    print("Service đang chạy tại: /set_velocity")
    print("\nĐể gọi service từ terminal khác:")
    print("  ros2 service call /set_velocity "
          "robot_velocity_control/srv/MySrv "
          '"{vx: 0.5, vy: 0.0, wz: 0.0}"')
    print("\nVí dụ:")
    print("  Tiến thẳng:  {vx: 0.5, vy: 0.0, wz: 0.0}")
    print("  Lùi:         {vx: -0.5, vy: 0.0, wz: 0.0}")
    print("  Quay phải:   {vx: 0.0, vy: 0.0, wz: -0.5}")
    print("  Quay trái:   {vx: 0.0, vy: 0.0, wz: 0.5}")
    print("  Dừng:        {vx: 0.0, vy: 0.0, wz: 0.0}")
    print("\nHoặc dùng velocity_client.py để gọi service từ Python")
    print("="*60 + "\n")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nDừng service...")
    finally:
        # Dừng robot trước khi thoát
        msg = Twist()
        node.publisher.publish(msg)
        node.get_logger().info('Robot stopped. Shutting down...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
