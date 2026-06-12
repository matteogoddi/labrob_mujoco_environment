#ifndef LABROB_SE3_HPP_
#define LABROB_SE3_HPP_

#include <Eigen/Core>

namespace labrob {

class SE3 {
 public:
  SE3() : R(Eigen::Matrix3d::Identity()), p(Eigen::Vector3d::Zero()) { }
  SE3(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation) : R(rotation), p(translation) { }

  const Eigen::Matrix3d& rotation() const { return R; }
  const Eigen::Vector3d& translation() const { return p; }

  Eigen::Matrix3d R;
  Eigen::Vector3d p;
}; // end class SE3

} // end namespace labrob

#endif // LABROB_SE3_HPP_