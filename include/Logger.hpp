#ifndef LABROB_LOGGER_HPP_
#define LABROB_LOGGER_HPP_

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

namespace labrob {

// Accumulates named time-series in memory and flushes them to disk in one shot.
//
// Vector channels: logger.log("name", eigen_expr)  — any shape, stored as VectorXd
// Scalar channels: logger.log("name", double_val)
// Save all:        logger.save("/tmp")  → /tmp/name.txt per channel
class Logger {
 public:
  // Pre-allocate capacity (optional, avoids reallocations for known lengths).
  void reserve(const std::string& name, std::size_t n) {
    vec_data_[name].reserve(n);
  }
  void reserveScalar(const std::string& name, std::size_t n) {
    scalar_data_[name].reserve(n);
  }

  // Append an Eigen vector/expression. Accepts any shape — stored flat as VectorXd.
  template <typename Derived>
  void log(const std::string& name, const Eigen::MatrixBase<Derived>& v) {
    const auto tmp = v.eval();
    vec_data_[name].emplace_back(
        Eigen::Map<const Eigen::VectorXd>(tmp.data(), tmp.size()));
  }

  // Append a scalar value.
  void log(const std::string& name, double v) {
    scalar_data_[name].push_back(v);
  }

  // Write all channels to dir/name.txt (one row per timestep).
  // Overwrites existing files. Does nothing if a channel is empty.
  void save(const std::string& dir) const {
    for (const auto& [name, entries] : vec_data_) {
      if (entries.empty()) continue;
      std::ofstream f(dir + "/" + name + ".txt");
      for (const auto& v : entries) f << v.transpose() << "\n";
    }
    for (const auto& [name, entries] : scalar_data_) {
      if (entries.empty()) continue;
      std::ofstream f(dir + "/" + name + ".txt");
      for (double v : entries) f << v << "\n";
    }
  }

  // Clear all accumulated data (keeps reserved capacity).
  void clear() {
    for (auto& [_, v] : vec_data_)    v.clear();
    for (auto& [_, v] : scalar_data_) v.clear();
  }

 private:
  std::unordered_map<std::string, std::vector<Eigen::VectorXd>> vec_data_;
  std::unordered_map<std::string, std::vector<double>>          scalar_data_;
};

} // end namespace labrob

#endif // LABROB_LOGGER_HPP_