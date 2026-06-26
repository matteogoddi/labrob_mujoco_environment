#pragma once

#include <Eigen/Core>

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

#include <numeric>

namespace labrob {

class QpSolver {
 public:
  QpSolver(int num_variables,
           int num_equality_constraints,
           int num_inequality_constraints,
           enum hpipm_mode solver_mode = SPEED_ABS,
           int iter_max = 50)
      : n_vars_(num_variables),
        solution_(Eigen::VectorXd::Zero(num_variables)) {

    memset(&dim_,       0, sizeof(dim_));
    memset(&qp_,        0, sizeof(qp_));
    memset(&qp_sol_,    0, sizeof(qp_sol_));
    memset(&arg_,       0, sizeof(arg_));
    memset(&workspace_, 0, sizeof(workspace_));

    int dim_size = d_dense_qp_dim_memsize();
    dim_mem_ = calloc(1, dim_size);
    d_dense_qp_dim_create(&dim_, dim_mem_);
    
    d_dense_qp_dim_set_all(num_variables, num_equality_constraints, 0,
                           num_inequality_constraints, 0, num_inequality_constraints, &dim_);

    // d_dense_qp_dim_set_all(num_variables, num_equality_constraints, 0,
    //                        num_inequality_constraints, num_inequality_constraints, &dim_);


    int qp_size = d_dense_qp_memsize(&dim_);
    qp_mem_ = calloc(1, qp_size);
    d_dense_qp_create(&dim_, &qp_, qp_mem_);

    int qp_sol_size = d_dense_qp_sol_memsize(&dim_);
    qp_sol_mem_ = calloc(1, qp_sol_size);
    d_dense_qp_sol_create(&dim_, &qp_sol_, qp_sol_mem_);

    int ipm_arg_size = d_dense_qp_ipm_arg_memsize(&dim_);
    ipm_arg_mem_ = calloc(1, ipm_arg_size);
    d_dense_qp_ipm_arg_create(&dim_, &arg_, ipm_arg_mem_);
    d_dense_qp_ipm_arg_set_default(solver_mode, &arg_);
    if (iter_max > 0)
      d_dense_qp_ipm_arg_set_iter_max(&iter_max, &arg_);

    int ipm_size = d_dense_qp_ipm_ws_memsize(&dim_, &arg_);
    ipm_mem_ = calloc(1, ipm_size);
    d_dense_qp_ipm_ws_create(&dim_, &arg_, &workspace_, ipm_mem_);

    idxs_sg_.resize(num_inequality_constraints);
    std::iota(idxs_sg_.begin(), idxs_sg_.end(), 0);
    Zl_ = std::vector<double>(num_inequality_constraints, 1e-6);
    Zu_ = std::vector<double>(num_inequality_constraints, 1e-6);
    zl_ = std::vector<double>(num_inequality_constraints, 0.0);
    zu_ = std::vector<double>(num_inequality_constraints, 0.0);

    sol_buf_ = (double*) calloc(num_variables, sizeof(double));
  }

  ~QpSolver() {
    free(dim_mem_);
    free(qp_mem_);
    free(qp_sol_mem_);
    free(ipm_arg_mem_);
    free(ipm_mem_);
    free(sol_buf_);
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
    d_dense_qp_set_H(const_cast<double*>(H.derived().data()), &qp_);
    d_dense_qp_set_g(const_cast<double*>(g.derived().data()), &qp_);
    d_dense_qp_set_A(const_cast<double*>(A.derived().data()), &qp_);
    d_dense_qp_set_b(const_cast<double*>(b.derived().data()), &qp_);
    d_dense_qp_set_C(const_cast<double*>(C.derived().data()), &qp_);
    d_dense_qp_set_lg(const_cast<double*>(lg.derived().data()), &qp_);
    d_dense_qp_set_ug(const_cast<double*>(ug.derived().data()), &qp_);
    d_dense_qp_set_idxs_rev(idxs_sg_.data(), &qp_);
    d_dense_qp_set_Zl(Zl_.data(), &qp_);
    d_dense_qp_set_Zu(Zu_.data(), &qp_);
    d_dense_qp_set_zl(zl_.data(), &qp_);
    d_dense_qp_set_zu(zu_.data(), &qp_);
    d_dense_qp_ipm_solve(&qp_, &qp_sol_, &arg_, &workspace_);
    d_dense_qp_sol_get_v(&qp_sol_, sol_buf_);
    for (int i = 0; i < n_vars_; ++i)
      solution_(i) = sol_buf_[i];
  }

  const Eigen::VectorXd& get_solution() const { return solution_; }

  int get_status() const { return workspace_.status; }

  bool has_valid_solution() const {
    return workspace_.status == 0 && solution_.allFinite();
  }

 private:
  int n_vars_;
  double* sol_buf_;
  Eigen::VectorXd solution_;

  struct d_dense_qp        qp_;
  struct d_dense_qp_dim    dim_;
  struct d_dense_qp_sol    qp_sol_;
  struct d_dense_qp_ipm_arg arg_;
  struct d_dense_qp_ipm_ws  workspace_;

  void* dim_mem_;
  void* qp_mem_;
  void* qp_sol_mem_;
  void* ipm_arg_mem_;
  void* ipm_mem_;

  std::vector<int>    idxs_sg_;
  std::vector<double> Zl_, Zu_, zl_, zu_;
};

} // namespace labrob