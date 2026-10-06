// Stand-in for ROS 1 <ros/console.h> so sns_ik_lib compiles unmodified under ROS 2.
#pragma once

#include <cstdio>
#include <cstdlib>

#define SNS_IK_LOG(level, ...) (std::fprintf(stderr, "[" level "] " __VA_ARGS__), std::fputc('\n', stderr))

#define ROS_DEBUG(...) \
  do {                 \
  } while (0)
#define ROS_DEBUG_NAMED(name, ...) \
  do {                             \
  } while (0)
#define ROS_DEBUG_STREAM(args) \
  do {                         \
  } while (0)
#define ROS_DEBUG_STREAM_NAMED(name, args) \
  do {                                     \
  } while (0)
#define ROS_INFO(...) SNS_IK_LOG("INFO", __VA_ARGS__)
#define ROS_WARN(...) SNS_IK_LOG("WARN", __VA_ARGS__)
#define ROS_ERROR(...) SNS_IK_LOG("ERROR", __VA_ARGS__)
#define ROS_FATAL(...) SNS_IK_LOG("FATAL", __VA_ARGS__)
#define ROS_FATAL_NAMED(name, ...) SNS_IK_LOG("FATAL", __VA_ARGS__)

// Always evaluated: sns_ik.cpp relies on the side effect inside ROS_ASSERT_MSG(setVelocitySolveType(...)).
#define ROS_ASSERT_MSG(cond, ...)       \
  do {                                  \
    if (!(cond)) {                      \
      SNS_IK_LOG("FATAL", __VA_ARGS__); \
      std::abort();                     \
    }                                   \
  } while (0)
