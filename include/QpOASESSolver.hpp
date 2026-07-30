#pragma once

#include <Eigen/Core>
#include <qpOASES.hpp>

#include <vector>

namespace labrob {

// Drop-in alternative to QpSolver (HPIPM-based), same public interface:
// solve(H, g, A, b, C, lg, ug) / get_solution(). Backed by qpOASES::SQProblem
// (active-set), which warm-starts well across consecutive, slightly
// perturbed solves — the typical receding-horizon MPC/WBC pattern.
//
// qpOASES has no native soft-constraint/slack mechanism like HPIPM's
// Zl/Zu/idxs_rev, so it is replicated explicitly here: for each of the first
// `num_soft_constraints` rows of C, two extra slack variables (sl>=0, su>=0)
// are added and the two-sided row `lg <= Cx <= ug` becomes two one-sided
// rows `Cx + sl >= lg` and `Cx - su <= ug`, with quadratic cost
// soft_weight*(sl^2+su^2). This doubles the row/variable count for the
// softened part of the problem — if that hurts performance, consider
// softening only a targeted subset of rows instead of all of them.
class QpOASESSolver {
 public:
  QpOASESSolver(int num_variables,
                int num_equality_constraints,
                int num_inequality_constraints,
                int num_soft_constraints = 0,
                double soft_weight = 0.0,
                int max_working_set_recalculations = 200)
      : n_vars_(num_variables),
        n_eq_(num_equality_constraints),
        n_ineq_(num_inequality_constraints),
        n_soft_(num_soft_constraints),
        soft_weight_(soft_weight),
        max_wsr_(max_working_set_recalculations),
        n_vars_aug_(num_variables + 2 * num_soft_constraints),
        n_rows_(num_equality_constraints
                + (num_inequality_constraints - num_soft_constraints)
                + 2 * num_soft_constraints),
        solution_(Eigen::VectorXd::Zero(num_variables)),
        qp_(n_vars_aug_, n_rows_) {

    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_NONE;
    qp_.setOptions(options);

    H_aug_   = RowMajorMat::Zero(n_vars_aug_, n_vars_aug_);
    A_aug_   = RowMajorMat::Zero(n_rows_, n_vars_aug_);
    g_aug_   = Eigen::VectorXd::Zero(n_vars_aug_);
    lb_aug_  = Eigen::VectorXd::Zero(n_vars_aug_);
    ub_aug_  = Eigen::VectorXd::Zero(n_vars_aug_);
    lbA_aug_ = Eigen::VectorXd::Zero(n_rows_);
    ubA_aug_ = Eigen::VectorXd::Zero(n_rows_);

    // x is unbounded (nb=0 in this codebase, everything goes through C);
    // slack pairs sl_i, su_i are constrained to [0, +inf).
    lb_aug_.head(n_vars_).setConstant(-qpOASES::INFTY);
    ub_aug_.head(n_vars_).setConstant(qpOASES::INFTY);
    lb_aug_.tail(2 * n_soft_).setConstant(0.0);
    ub_aug_.tail(2 * n_soft_).setConstant(qpOASES::INFTY);
  }

  template <typename DH, typename Dg, typename DA, typename Db,
            typename DC, typename Dlg>
  void solve(const Eigen::PlainObjectBase<DH>&  H,
             const Eigen::PlainObjectBase<Dg>&  g,
             const Eigen::PlainObjectBase<DA>&  A,
             const Eigen::PlainObjectBase<Db>&  b,
             const Eigen::PlainObjectBase<DC>&  C,
             const Eigen::PlainObjectBase<Dlg>& lg,
             const Eigen::PlainObjectBase<Dlg>& ug) {

    // H is symmetric, so embedding it directly works regardless of the
    // caller's storage order; H_aug_/A_aug_ are RowMajor so .data() is
    // already in the layout qpOASES expects (no transpose needed).
    H_aug_.setZero();
    H_aug_.block(0, 0, n_vars_, n_vars_) = H;
    for (int i = 0; i < 2 * n_soft_; ++i)
      H_aug_(n_vars_ + i, n_vars_ + i) = soft_weight_;

    g_aug_.setZero();
    g_aug_.head(n_vars_) = g;

    A_aug_.setZero();
    A_aug_.block(0, 0, n_eq_, n_vars_) = A;
    lbA_aug_.head(n_eq_) = b;
    ubA_aug_.head(n_eq_) = b;

    int row = n_eq_;
    for (int i = 0; i < n_soft_; ++i) {
      A_aug_.block(row, 0, 1, n_vars_) = C.row(i);
      A_aug_(row, n_vars_ + 2 * i) = 1.0;             // + sl_i
      lbA_aug_(row) = lg(i);
      ubA_aug_(row) = qpOASES::INFTY;
      ++row;

      A_aug_.block(row, 0, 1, n_vars_) = C.row(i);
      A_aug_(row, n_vars_ + 2 * i + 1) = -1.0;         // - su_i
      lbA_aug_(row) = -qpOASES::INFTY;
      ubA_aug_(row) = ug(i);
      ++row;
    }
    for (int i = n_soft_; i < n_ineq_; ++i) {
      A_aug_.block(row, 0, 1, n_vars_) = C.row(i);
      lbA_aug_(row) = lg(i);
      ubA_aug_(row) = ug(i);
      ++row;
    }

    int nWSR = max_wsr_;
    qpOASES::returnValue status;
    if (!initialized_) {
      status = qp_.init(H_aug_.data(), g_aug_.data(), A_aug_.data(),
                         lb_aug_.data(), ub_aug_.data(),
                         lbA_aug_.data(), ubA_aug_.data(), nWSR);
      initialized_ = true;
    } else {
      status = qp_.hotstart(H_aug_.data(), g_aug_.data(), A_aug_.data(),
                             lb_aug_.data(), ub_aug_.data(),
                             lbA_aug_.data(), ubA_aug_.data(), nWSR);
    }
    last_ok_ = (status == qpOASES::SUCCESSFUL_RETURN);

    std::vector<qpOASES::real_t> x_aug(n_vars_aug_);
    qp_.getPrimalSolution(x_aug.data());
    for (int i = 0; i < n_vars_; ++i)
      solution_(i) = x_aug[i];
  }

  const Eigen::VectorXd& get_solution() const { return solution_; }

  bool has_valid_solution() const { return last_ok_ && solution_.allFinite(); }

 private:
  using RowMajorMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  int n_vars_, n_eq_, n_ineq_, n_soft_;
  double soft_weight_;
  int max_wsr_;
  int n_vars_aug_, n_rows_;
  bool initialized_ = false;
  bool last_ok_ = false;

  Eigen::VectorXd solution_;
  qpOASES::SQProblem qp_;

  RowMajorMat H_aug_, A_aug_;
  Eigen::VectorXd g_aug_, lb_aug_, ub_aug_, lbA_aug_, ubA_aug_;
};

} // namespace labrob