# baxter_interface migration

## Organization of executable logic

ROS 1

In ROS 1 it was a standalone script in scripts/. In ROS 2 the pattern is to put executable logic as a module and register it as a console_scripts entry point in setup.py:

```python
  entry_points={
      'console_scripts': [
          'analog_io_rampup = baxter_examples.analog_io_rampup:main',
      ],
  },

```

Now to run our examples we can use the standard ROS 2 commands:

```bash
ros2 run baxter_examples analog_io_rampup
```

## Logging

ROS 1

```python
rospy.loginfo('Created node')

```

ROS 2

```python
node.get_logger().info('Created node')
```

## Command line arguments

In ROS 1, ROS injected its own tokens (`__name:=`, `__log:=`, `_param:=`) directly into `sys.argv` alongside your program's args:

```bash
rosrun baxter_examples analog_io_rampup.py -c torso_fan __name:=my_node __log:=/tmp/foo.log
# sys.argv = ['-c', 'torso_fan', '__name:=my_node', '__log:=/tmp/foo.log']
```

Argparse didn't know what `__name:=my_node` was and would crash. `rospy.myargv()` stripped them first:

```python
args = parser.parse_args(rospy.myargv()[1:])
# after strip: ['-c', 'torso_fan']
```

In ROS 2, ROS args live after `--ros-args` and are consumed by `rclpy.init()`. Argparse never sees them:

```bash
ros2 run baxter_examples analog_io_rampup -c torso_fan --ros-args -p component_id:=torso_fan
# sys.argv seen by argparse = ['-c', 'torso_fan']
```

```python
rclpy.init(args=sys.argv)
args = parser.parse_args()  # just works
```

## Parameters

ROS 1

```python
io_component = rospy.get_param('~component_id', args.component_id)
```

The `~` prefix made the parameter private to the node (`rsdk_analog_io_rampup/component_id`). No declaration required.

ROS 2

```python
node.declare_parameter('component_id', args.component_id)
io_component = node.get_parameter('component_id').get_parameter_value().string_value
```

Parameters must be declared before use. Scope is always node-local, so no `~` prefix is needed.

## Rate

ROS 1

```python
rate = rospy.Rate(2)
rate.sleep()
```

ROS 2

```python
rate = node.create_rate(2)
rate.sleep()
```

## Node naming

In ROS 1, `anonymous=True` appended a random suffix to the node name to avoid collisions when running multiple instances:

```python
rospy.init_node('rsdk_analog_io_rampup', anonymous=True)
```

In ROS 2, one-shot example scripts don't need this. Just name the node directly:

```python
node = rclpy.create_node('rsdk_analog_io_rampup')
```

### Installing Joystick in Kilted
sudo apt-get install ros-kilted-joy

ros2 launch baxter_examples joint_position_joystick.launch.xml joystick:=xbox
