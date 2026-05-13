import rclpy
from rclpy.node import Node
from my_interfaces.srv import CheckStudent


class CheckMemberServer(Node):
    def __init__(self):
        super().__init__('check_member_server')

        self.srv = self.create_service(
            CheckStudent,
            'nhom2',
            self.check_callback
        )

        # Danh sách thành viên nhóm
        self.members = [
            {"name": "Nguyen Van A", "mssv": 20123456},
            {"name": "Tran Van B", "mssv": 20123457},
            {"name": "Le Van C", "mssv": 20123458},
        ]

        self.get_logger().info("Check Student Server is running...")

    def check_callback(self, request, response):
        for member in self.members:
            if request.name == member["name"] and request.mssv == member["mssv"]:
                response.result = "OK"
                return response

        response.result = "Not found"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CheckMemberServer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

