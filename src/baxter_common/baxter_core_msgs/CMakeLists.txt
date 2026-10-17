cmake_minimum_required(VERSION 3.5)
project(baxter_core_msgs)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AnalogIOState.msg"
  "msg/AnalogIOStates.msg"
  "msg/AnalogOutputCommand.msg"
  "msg/AssemblyState.msg"
  "msg/AssemblyStates.msg"
  "msg/CameraControl.msg"
  "msg/CameraSettings.msg"
  "msg/CollisionAvoidanceState.msg"
  "msg/CollisionDetectionState.msg"
  "msg/DigitalIOState.msg"
  "msg/DigitalIOStates.msg"
  "msg/DigitalOutputCommand.msg"
  "msg/EndEffectorCommand.msg"
  "msg/EndEffectorProperties.msg"
  "msg/EndEffectorState.msg"
  "msg/EndpointState.msg"
  "msg/EndpointStates.msg"
  "msg/HeadPanCommand.msg"
  "msg/HeadState.msg"
  "msg/HomingCommand.msg"
  "msg/JointCommand.msg"
  "msg/NavigatorState.msg"
  "msg/NavigatorStates.msg"
  "msg/RobustControllerStatus.msg"
  "msg/SEAJointState.msg"
  "msg/URDFConfiguration.msg"
  "srv/CloseCamera.srv"
  "srv/ListCameras.srv"
  "srv/OpenCamera.srv"
  "srv/SolvePositionIK.srv"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs

)

install(
  FILES mapping_rules.yaml
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
