from setuptools import setup
from glob import glob
import os

package_name = 'intelli_h1_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jack Su',
    maintainer_email='jack@example.com',
    description='ROS2 integration for IntelliH1 cognitive humanoid framework',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_node = intelli_h1_ros.sim_node:main',
            'rl_node = intelli_h1_ros.rl_node:main',
            'brain_node = intelli_h1_ros.brain_node:main',
        ],
    },
)
