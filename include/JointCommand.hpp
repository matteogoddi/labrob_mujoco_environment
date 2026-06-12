#ifndef LABROB_JOINT_COMMAND_H_
#define LABROB_JOINT_COMMAND_H_

// STL
#include <string>
#include <unordered_map>

namespace labrob {

class JointCommand {
  using JointCommandMap = std::unordered_map<std::string, double>;
 public:
  double operator[](const std::string& key) const {return joint_command_.at(key);};
  double& operator[](const std::string& key) {return joint_command_[key];};
  JointCommandMap::iterator begin() {return joint_command_.begin();};
  JointCommandMap::iterator end() {return joint_command_.end();};

 protected:
  JointCommandMap joint_command_;
}; // end class JointCommand

} // end namespace labrob

#endif // LABROB_JOINT_COMMAND_H_