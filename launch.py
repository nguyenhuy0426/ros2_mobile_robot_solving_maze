from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_pkg',
            namespace='vd1',
            executable='talker',
            output = 'screen',
            name='sim1',
            arguments=['--ros-args', '--log-level', 'info']
        ),
        Node(
            package='my_pkg',
            namespace='vd1',
            executable='listener',
            output = 'screen',
            name='sim2',
             arguments=['--ros-args', '--log-level', 'info']
        ),
    ])
