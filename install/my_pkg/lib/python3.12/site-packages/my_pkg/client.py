#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from my_interfaces.srv import MySrv

class CallSrv1s(Node):
    def __init__(self):
        super().__init__('call_srv_1s')
        self.cli = self.create_client(MySrv, '/mecanum_srv')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting /mecanum_srv ...')

        self.vx, self.vy, self.wz = 0.0, 0.0, -0.5
        self.busy = False
        self.create_timer(1.0, self.tick)

    def tick(self):
        if self.busy:
            return
        req = MySrv.Request()
        req.vx, req.vy, req.wz = self.vx, self.vy, self.wz
        self.busy = True
        fut = self.cli.call_async(req)
        fut.add_done_callback(self.done)

    def done(self, fut):
        self.busy = False
        try:
            res = fut.result()
            self.get_logger().info(f"RESP ok={res.ok}, phi={list(res.phi)}")
        except Exception as e:
            self.get_logger().error(f"call failed: {e}")

def main():
    rclpy.init()
    rclpy.spin(CallSrv1s())
    rclpy.shutdown()

if __name__ == '__main__':
    main()