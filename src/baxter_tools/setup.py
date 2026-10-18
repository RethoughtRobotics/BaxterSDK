from setuptools import find_packages, setup

package_name = 'baxter_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andnet DeBoer',
    maintainer_email='deboerandnet@gmail.com',
    description='Useful operational and maintenance tools for use with the Baxter Research Robot from Rethink Robotics',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'enable_robot = baxter_tools.enable_robot:main',
            'camera_control = baxter_tools.camera_control:main',
            'calibrate_arm  = baxter_tools.calibrate_arm:main',
            'calibrate_gripper = baxter_tools.calibrate_gripper:main',
            'tuck_arms = baxter_tools.tuck_arms:main',
        ],
    },
)
