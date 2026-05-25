from setuptools import find_packages, setup

package_name = 'baxter_demo'

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
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'blink_observer = baxter_demo.blink_observer:main',
            'blink_facade = baxter_demo.blink_facade:main',
        ],
    },
)
