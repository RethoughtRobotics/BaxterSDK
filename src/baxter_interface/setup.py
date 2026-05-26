from setuptools import find_packages, setup

package_name = 'baxter_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/param.yaml']),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andnet DeBoer',
    maintainer_email='deboerandnet@gmail.com',
    description='Convenient python interface classes for control of the Baxter Research Robot from Rethink Robotics.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gripper_action_server = baxter_interface.gripper_action_server:main',
            'head_action_server = baxter_interface.head_action_server:main',
            'joint_trajectory_action_server = baxter_interface.joint_trajectory_action_server:main',
        ],
    },
)
