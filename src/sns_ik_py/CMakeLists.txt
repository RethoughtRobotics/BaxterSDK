cmake_minimum_required(VERSION 3.16)
project(sns_ik_py)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(orocos_kdl_vendor REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(urdf REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11_vendor REQUIRED)
find_package(pybind11 REQUIRED)

# Rethink's SNS-IK checkout (ROS 1 workspace). Its sources are compiled as-is; only the
# ROS 1 headers they include are provided by compat/.
set(SNS_IK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../rethink_ws/src/sns_ik"
  CACHE PATH "Path to the RethinkRobotics-opensource/sns_ik checkout")
set(SNS_IK_LIB "${SNS_IK_DIR}/sns_ik_lib")

pybind11_add_module(_sns_ik
  src/bindings.cpp
  ${SNS_IK_LIB}/src/fosns_velocity_ik.cpp
  ${SNS_IK_LIB}/src/fsns_velocity_ik.cpp
  ${SNS_IK_LIB}/src/osns_sm_velocity_ik.cpp
  ${SNS_IK_LIB}/src/osns_velocity_ik.cpp
  ${SNS_IK_LIB}/src/sns_acc_ik_base.cpp
  ${SNS_IK_LIB}/src/sns_ik.cpp
  ${SNS_IK_LIB}/src/sns_ik_base.cpp
  ${SNS_IK_LIB}/src/sns_position_ik.cpp
  ${SNS_IK_LIB}/src/sns_vel_ik_base.cpp
  ${SNS_IK_LIB}/src/sns_velocity_ik.cpp
  ${SNS_IK_LIB}/utilities/sns_ik_math_utils.cpp
  ${SNS_IK_LIB}/utilities/sns_linear_solver.cpp)
target_compile_features(_sns_ik PRIVATE cxx_std_17)
target_include_directories(_sns_ik PRIVATE
  compat
  ${SNS_IK_LIB}/include
  ${SNS_IK_LIB}/utilities)
target_link_libraries(_sns_ik PRIVATE
  Eigen3::Eigen
  orocos-kdl
  kdl_parser::kdl_parser
  urdf::urdf)

ament_python_install_package(${PROJECT_NAME})
install(TARGETS _sns_ik LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")
# Rethink's Baxter limits, as loaded by sns_ik_examples/launch/test_baxter.launch.
install(FILES "${SNS_IK_DIR}/sns_ik_examples/config/example_baxter_joint_limits.yaml"
  DESTINATION share/${PROJECT_NAME}/config)

ament_package()
