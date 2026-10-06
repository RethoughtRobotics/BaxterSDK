// Python bindings for sns_ik::SNS_IK, constructed exactly as Rethink's ROS 1 launch files do:
// URDF on robot_description, joint limit overrides on robot_description_planning/joint_limits.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ros/ros.h>

#include <array>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <kdl/chainfksolverpos_recursive.hpp>
#include <sns_ik/sns_ik.hpp>

namespace
{

using JointLimits = std::map<std::string, std::map<std::string, double>>;

class SnsIk
{
public:
  SnsIk(
    const std::string & urdf_xml,
    const std::string & base_link,
    const std::string & tip_link,
    const JointLimits & joint_limits,
    double loop_period)
  {
    ros::string_params().clear();
    ros::double_params().clear();
    ros::string_params()["/robot_description"] = urdf_xml;
    for (const auto & [joint, limits] : joint_limits) {
      for (const auto & [key, value] : limits) {
        ros::double_params()["/robot_description_planning/joint_limits/" + joint + "/" + key] = value;
      }
    }
    ik_ = std::make_unique<sns_ik::SNS_IK>(base_link, tip_link, "/robot_description", loop_period);
    if (!ik_->getJointNames(joint_names_)) {
      throw std::runtime_error("SNS_IK failed to initialize chain " + base_link + " -> " + tip_link);
    }
    ik_->getKDLChain(chain_);
    fk_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
  }

  std::vector<std::string> joint_names() const
  {
    return joint_names_;
  }

  void set_nullspace_gain(double gain)
  {
    ik_->setNullspaceGain(gain);
  }

  // SNS_IK::CartToJntVel: twist = [vx, vy, vz, wx, wy, wz] in the base frame, about the tip origin.
  // An empty q_bias skips the nullspace bias task. Plain lists cross the boundary: the system
  // pybind11 Eigen caster predates numpy 2. A negative primary scale factor means SNS could not
  // execute the task ("task not executed: reached sing"); it then returns zero joint velocities.
  std::vector<double> cart_to_jnt_vel(
    const std::vector<double> & q, const std::array<double, 6> & twist, const std::vector<double> & q_bias)
  {
    KDL::JntArray q_in(q.size());
    q_in.data = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
    KDL::JntArray bias(q_bias.size());
    bias.data = Eigen::Map<const Eigen::VectorXd>(q_bias.data(), q_bias.size());
    KDL::JntArray qdot(q.size());
    const KDL::Twist v(KDL::Vector(twist[0], twist[1], twist[2]), KDL::Vector(twist[3], twist[4], twist[5]));
    ik_->CartToJntVel(q_in, v, bias, qdot);
    return std::vector<double>(qdot.data.data(), qdot.data.data() + qdot.data.size());
  }

  // URDF position limits used by the solver: (lower, upper).
  std::pair<std::vector<double>, std::vector<double>> position_limits()
  {
    KDL::JntArray lower, upper, velocity, acceleration;
    ik_->getKDLLimits(lower, upper, velocity, acceleration);
    return {
      std::vector<double>(lower.data.data(), lower.data.data() + lower.data.size()),
      std::vector<double>(upper.data.data(), upper.data.data() + upper.data.size())};
  }

  // Orientation of the tip link in the base frame (row-major 3x3), from the same KDL chain.
  std::array<std::array<double, 3>, 3> tip_rotation(const std::vector<double> & q)
  {
    KDL::JntArray q_in(q.size());
    q_in.data = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
    KDL::Frame tip;
    if (fk_->JntToCart(q_in, tip) < 0) {
      throw std::runtime_error("KDL forward kinematics failed");
    }
    std::array<std::array<double, 3>, 3> rotation;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        rotation[i][j] = tip.M(i, j);
      }
    }
    return rotation;
  }

  // Per-task scale factors from the last solve: 1.0 = achieved in full, < 1.0 = scaled to fit limits.
  std::vector<double> task_scale_factors()
  {
    std::vector<double> factors;
    ik_->getTaskScaleFactors(factors);
    return factors;
  }

private:
  std::unique_ptr<sns_ik::SNS_IK> ik_;
  KDL::Chain chain_;  // the solver keeps a reference to it
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
  std::vector<std::string> joint_names_;
};

}  // namespace

PYBIND11_MODULE(_sns_ik, m)
{
  pybind11::class_<SnsIk>(m, "SnsIk")
    .def(
      pybind11::init<const std::string &, const std::string &, const std::string &, const JointLimits &, double>(),
      pybind11::arg("urdf_xml"),
      pybind11::arg("base_link"),
      pybind11::arg("tip_link"),
      pybind11::arg("joint_limits"),
      pybind11::arg("loop_period") = 0.01)
    .def("joint_names", &SnsIk::joint_names)
    .def("set_nullspace_gain", &SnsIk::set_nullspace_gain)
    .def(
      "cart_to_jnt_vel", &SnsIk::cart_to_jnt_vel, pybind11::arg("q"), pybind11::arg("twist"), pybind11::arg("q_bias"))
    .def("position_limits", &SnsIk::position_limits)
    .def("tip_rotation", &SnsIk::tip_rotation, pybind11::arg("q"))
    .def("task_scale_factors", &SnsIk::task_scale_factors);
}
