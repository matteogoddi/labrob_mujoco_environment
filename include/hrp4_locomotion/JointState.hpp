#ifndef LABROB_JOINT_STATE_H_
#define LABROB_JOINT_STATE_H_

// STL
#include <string>
#include <unordered_map>

namespace labrob {

struct JointData {
  double pos = 0.0;
  double vel = 0.0;
  double acc = 0.0;
  double eff = 0.0;
}; // end struct JointData

class JointState {
  using JointStateMap = std::unordered_map<std::string, labrob::JointData>;
 public:
  labrob::JointData operator[](const std::string& key) const {return joint_state_.at(key);}
  labrob::JointData& operator[](const std::string& key) {return joint_state_[key];}
  JointStateMap::iterator begin(){return joint_state_.begin();}
  JointStateMap::iterator end(){return joint_state_.end();}

 protected:
  JointStateMap joint_state_;
}; // end class JointState

} // end namespace labrob

#endif // LABROB_JOINT_STATE_H_