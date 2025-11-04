#ifndef LABROB_HPIPM_QP_SOLVER_HPP_
#define LABROB_HPIPM_QP_SOLVER_HPP_

// HPIPM
#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_v_aux_ext_dep.h>
#include <blasfeo_d_aux_ext_dep.h>
#include <blasfeo_i_aux_ext_dep.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_blas.h>
#include <hpipm_d_dense_qp_ipm.h>
#include <hpipm_d_dense_qp_dim.h>
#include <hpipm_d_dense_qp.h>
#include <hpipm_d_dense_qp_sol.h>
#include <hpipm_timing.h>

// Eigen
#include <Eigen/Core>

// STL
#include <iostream>

namespace labrob {

class HPIPMQPSolver {
 public:
  HPIPMQPSolver(int numVariables, int numEqualityConstraints, int numInequalityConstraints) :
  num_variables_(numVariables),
  num_equality_constraints_(numEqualityConstraints),
  num_inequality_constraints_(numInequalityConstraints) {
    int dim_size = d_dense_qp_dim_memsize();
    dim_mem_ = malloc(dim_size);
    d_dense_qp_dim_create(&dim_, dim_mem_);

    d_dense_qp_dim_set_all(numVariables, numEqualityConstraints, 0, numInequalityConstraints, 0, 0, &dim_);

    int qp_size = d_dense_qp_memsize(&dim_);
    qp_mem_ = malloc(qp_size);
    d_dense_qp_create(&dim_, &qp_, qp_mem_);

    // allocate memory for the solution
    int qp_sol_size = d_dense_qp_sol_memsize(&dim_);
    qp_sol_mem_ = malloc(qp_sol_size);
    d_dense_qp_sol_create(&dim_, &qp_sol_, qp_sol_mem_);

    // allocate memory for ipm solver and its workspace
    int ipm_arg_size = d_dense_qp_ipm_arg_memsize(&dim_);
    ipm_arg_mem_ = malloc(ipm_arg_size);
    d_dense_qp_ipm_arg_create(&dim_, &arg_, ipm_arg_mem_);
    enum hpipm_mode mode = SPEED; // set mode ROBUST, SPEED, BALANCE, SPEED_ABS
    d_dense_qp_ipm_arg_set_default(mode, &arg_);

    int ipm_size = d_dense_qp_ipm_ws_memsize(&dim_, &arg_);
    ipm_mem_ = malloc(ipm_size);
    d_dense_qp_ipm_ws_create(&dim_, &arg_, &workspace_, ipm_mem_);

    u_ = (double*) malloc(numVariables * sizeof(double));
  }

  ~HPIPMQPSolver() {
    free(dim_mem_);
    free(qp_mem_);
    free(ipm_mem_);
    free(qp_sol_mem_);
    free(ipm_arg_mem_);
    free(u_);
  }

  // Raw pointer interface
  void solve(
      const double* H,
      const double* f,
      const double* A,
      const double* b,
      const double* C,
      const double* d_min,
      const double* d_max) {
    d_dense_qp_set_H((double*) H, &qp_);
    d_dense_qp_set_g((double*) f, &qp_);

    d_dense_qp_set_A((double*) A, &qp_);
    d_dense_qp_set_b((double*) b, &qp_);

    d_dense_qp_set_C((double*) C, &qp_);
    d_dense_qp_set_lg((double*) d_min, &qp_);
    d_dense_qp_set_ug((double*) d_max, &qp_);

    // solve QP
    d_dense_qp_ipm_solve(&qp_, &qp_sol_, &arg_, &workspace_);
    d_dense_qp_sol_get_v(&qp_sol_, u_);
  }

  // Eigen interface
  template <typename DerivedCostH, typename DerivedCostg,
            typename DerivedEqA, typename DerivedEqb,
            typename DerivedIneqC, typename DerivedIneqg>
  void solve(
      const Eigen::PlainObjectBase<DerivedCostH>& H,
      const Eigen::PlainObjectBase<DerivedCostg>& g,
      const Eigen::PlainObjectBase<DerivedEqA>& A,
      const Eigen::PlainObjectBase<DerivedEqb>& b,
      const Eigen::PlainObjectBase<DerivedIneqC>& C,
      const Eigen::PlainObjectBase<DerivedIneqg>& lg,
      const Eigen::PlainObjectBase<DerivedIneqg>& ug) {
    solve(
        H.data(),
        g.data(),
        A.data(),
        b.data(),
        C.data(),
        lg.data(),
        ug.data()
    );
  }

  // Get solution as raw pointer
  const double* get_solution() const {
    return u_;
  }

  // Get solution as Eigen vector
  Eigen::VectorXd get_solution_eigen() const {
    Eigen::VectorXd solution(num_variables_);
    for (int idx = 0; idx < num_variables_; ++idx) {
      solution(idx) = u_[idx];
    }
    return solution;
  }

  // Public accessors for dimensions
  int num_variables() const { return num_variables_; }
  int num_equality_constraints() const { return num_equality_constraints_; }
  int num_inequality_constraints() const { return num_inequality_constraints_; }

 private:
  const int num_variables_;
  const int num_equality_constraints_;
  const int num_inequality_constraints_;

  struct d_dense_qp qp_;
  struct d_dense_qp_dim dim_;
  struct d_dense_qp_sol qp_sol_;
  struct d_dense_qp_ipm_arg arg_;
  struct d_dense_qp_ipm_ws workspace_;

  void* dim_mem_;
  void* qp_mem_;
  void* ipm_mem_;
  void* qp_sol_mem_;
  void* ipm_arg_mem_;

  double* u_;

}; // end class HPIPMQPSolver

} // end namespace labrob

#endif // LABROB_HPIPM_QP_SOLVER_HPP_
