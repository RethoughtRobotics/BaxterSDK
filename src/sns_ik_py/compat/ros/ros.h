// Stand-in for ROS 1 <ros/ros.h>: an in-process parameter server for the one place sns_ik_lib
// uses it, SNS_IK's URDF constructor (robot_description + robot_description_planning/joint_limits).
#pragma once

#include <iostream>  // ROS 1 ros.h provided it; sns_ik.cpp uses std::cout without including it
#include <map>
#include <string>

#include "ros/console.h"

namespace ros
{

inline std::map<std::string, std::string> & string_params()
{
  static std::map<std::string, std::string> params;
  return params;
}

inline std::map<std::string, double> & double_params()
{
  static std::map<std::string, double> params;
  return params;
}

class NodeHandle
{
public:
  explicit NodeHandle(const std::string & /*ns*/ = "")
  {}

  bool getParam(const std::string & key, std::string & value) const
  {
    return lookup(string_params(), key, value);
  }

  bool getParam(const std::string & key, double & value) const
  {
    return lookup(double_params(), key, value);
  }

  template <typename T>
  void param(const std::string & key, T & value, const T & default_value) const
  {
    if (!getParam(key, value)) {
      value = default_value;
    }
  }

  bool searchParam(const std::string & key, std::string & result) const
  {
    result = key;
    return string_params().count(key) || double_params().count(key);
  }

private:
  template <typename T>
  static bool lookup(const std::map<std::string, T> & params, const std::string & key, T & value)
  {
    auto it = params.find(key);
    if (it == params.end()) {
      return false;
    }
    value = it->second;
    return true;
  }
};

}  // namespace ros
