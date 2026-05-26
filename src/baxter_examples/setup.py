from setuptools import find_packages, setup

package_name = 'baxter_examples'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/joint_position_joystick.launch.xml']),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andnet DeBoer',
    maintainer_email='deboerandnet@gmail.com',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'analog_io_rampup = baxter_examples.analog_io_rampup:main',
            'xdisplay_image = baxter_examples.xdisplay_image:main',
            'head_wobbler = baxter_examples.head_wobbler:main',
            'navigator_io = baxter_examples.navigator_io:main',
            'gripper_cuff_control = baxter_examples.gripper_cuff_control:main',
            'joint_position_joystick = baxter_examples.joint_position_joystick:main',
            'joint_recorder = baxter_examples.joint_recorder:main',
            'joint_playback = baxter_examples.joint_position_file_playback:main',
            'gripper_action_client = baxter_examples.gripper_action_client:main',
            'head_action_client = baxter_examples.head_action_client:main',
        ],
    },
)
