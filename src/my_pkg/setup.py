from setuptools import find_packages, setup

package_name = 'my_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='huynn',
    maintainer_email='huypro26122004@gmail.com',
    description='TODO: example for publish and subcribe msg ',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                'talker = my_pkg.publish:main',
                'listener = my_pkg.subcribe:main',
                'client = my_pkg.client:main',
                'server = my_pkg.server:main',
                'turtle = my_pkg.turtle:main',
                'aw = my_pkg.robot_awsd:main',
        ],
    },
)
