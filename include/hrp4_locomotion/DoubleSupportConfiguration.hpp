#ifndef LABROB_DOUBLE_SUPPORT_CONFIGURATION_HPP_
#define LABROB_DOUBLE_SUPPORT_CONFIGURATION_HPP_

#include <hrp4_locomotion/Foot.hpp>
#include <hrp4_locomotion/SE3.hpp>

namespace labrob {

class DoubleSupportConfiguration {
 public:
  DoubleSupportConfiguration(
      const labrob::SE3& qLeft,
      const labrob::SE3& qRight,
      const labrob::Foot& support_foot
  ) : qLeft_(qLeft),
      qRight_(qRight),
      support_foot_(support_foot) { }

  const labrob::SE3& getLeftFootConfiguration() const {return qLeft_;}
  const labrob::SE3& getRightFootConfiguration() const {return qRight_;}
  const labrob::SE3& getSupportFootConfiguration() const {
    if (support_foot_ == labrob::Foot::LEFT) return qLeft_;
    else return qRight_;
  }
  const labrob::SE3& getSwingFootConfiguration() const {
    if (support_foot_ == labrob::Foot::LEFT) return qRight_;
    else return qLeft_;
  }
  const labrob::Foot& getSupportFoot() const {return support_foot_;}

  void setFeetConfiguration(const labrob::SE3& qLeft, const labrob::SE3& qRight) {
    qLeft_ = qLeft;
    qRight_ = qRight;
  }
  void setLeftFootConfiguration(const labrob::SE3& qLeft) {qLeft_ = qLeft;}
  void setRightFootConfiguration(const labrob::SE3& qRight) {qRight_ = qRight;}
  void setSupportFootConfiguration(const labrob::SE3& qSupport) {
    if (support_foot_ == labrob::Foot::LEFT) qLeft_ = qSupport;
    else qRight_ = qSupport;
  }
  void setSwingFootConfiguration(const labrob::SE3& qSwing) {
    if (support_foot_ == labrob::Foot::LEFT) qRight_ = qSwing;
    else qLeft_ = qSwing;
  }
  void setSupportFoot(const labrob::Foot& support_foot) {support_foot_ = support_foot;}

 protected:
  labrob::SE3 qLeft_;
  labrob::SE3 qRight_;
  labrob::Foot support_foot_;

}; // end class DoubleSupportConfiguration

} // end namespace labrob

#endif // LABROB_DOUBLE_SUPPORT_CONFIGURATION_HPP_