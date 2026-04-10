#ifndef LABROB_COMPLIANCE_REFERENCE_GENERATOR_HPP_
#define LABROB_COMPLIANCE_REFERENCE_GENERATOR_HPP_

#include <vector>

#include <casadi/casadi.hpp>

#include <Eigen/Core>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <hrp4_locomotion/RobotState.hpp>

namespace labrob {

class ComplianceReferenceGenerator {
 public:
  ComplianceReferenceGenerator();
  struct Parameters{
        // arm compliance stiffness
        MatrixXd Ka_left = MatrixXd::Identity(6,6);
        MatrixXd Ka_right = MatrixXd::Identity(6,6);
        
        //torso compliance stiffness
        MatrixXd Kb = MatrixXd::Identity(6,6);

        //selection matrix for arm compliance DOFs
        MatrixXd S_left = MatrixXd::Identity(6,6);
        MatrixXd S_right = MatrixXd::Identity(6,6);

        //additional QP regularization/smoothing
        MatrixXd W_smooth = MatrixXd::Zero(6,6);
        MatrixXd W_reg = MatrixXd::Zero(6,6);

        //torso displacement bounds
        VectorXd delta_xb_min = VectorXd::Constant(6, -1e9);
        VectorXd delta_xb_max = VectorXd::Constant(6, 1e9);

        //optional arm displacement bounds
        VectorXd delta_xc_left_limit = VectorXd::Constant(6, 1e9);
        VectorXd delta_xc_right_limit = VectorXd::Constant(6, 1e9);

        //low-pass filter alpha (0-1), higher is more smoothing
        double filter_alpha = 0.99;

        //CasADi QP solver settings
        int print_level = 0;
        double cpu_time = 0.0;
        double bound_tolerance = 1e-8;
        double bound_relaxation = 1e-8;
    };

    struct Input{

        double dt = 0.001;

        //estimated external wrenches at left and right wrists
        VectorXd wrench_left = VectorXd::Zero(6);
        VectorXd wrench_right = VectorXd::Zero(6);

        //reference wrenches for compliance mapping (can be zero or nominal values)
        VectorXd wrench_left_ref = VectorXd::Zero(6);
        VectorXd wrench_right_ref = VectorXd::Zero(6);

        //torso Jacobians at left and right wrists
        MatrixXd Jb_left = MatrixXd::Zero(6,6);
        MatrixXd Jb_right = MatrixXd::Zero(6,6);

        //nominal reference for torso compliance (can be zero or nominal posture)
        VectorXd torso_nominal = VectorXd::Zero(6);
    };

    struct Output{
        //local arm compliance displacements
        VectorXd delta_xc_left = VectorXd::Zero(6);
        VectorXd delta_xc_right = VectorXd::Zero(6);

        //torso compliance displacement from QP solution
        VectorXd delta_xb = VectorXd::Zero(6);

        //first-order low-pass filtered compliance references for smoothness
        VectorXd delta_xc_left_filtered = VectorXd::Zero(6);
        VectorXd delta_xc_right_filtered = VectorXd::Zero(6);
        VectorXd delta_xb_filtered = VectorXd::Zero(6);

        //final reference sent to whole-body controller
        VectorXd delta_x_left = VectorXd::Zero(6);
        VectorXd delta_x_right = VectorXd::Zero(6);
        VectorXd delta_xb_final = VectorXd::Zero(6);

        bool valid = false;
    };

    struct DebugInfo{
        MatrixXd H= MatrixXd::Zero(6,6);
        VectorXd g= VectorXd::Zero(6);
        VectorXd lbx= VectorXd::Zero(6);
        VectorXd ubx= VectorXd::Zero(6);
        double objective_value = 0.0;
        bool qp_solved = false;
    };

  public:
    explicit ComplianceReferenceGenerator(const Parameters& params);

    void reset();
    
    void setParameters(const Parameters& params);
    const Parameters& getParameters() const;

    Output update(const Input& input);

    const DebugInfo& getDebugInfo() const;

  private:
    // build CasADi + qpOASES solver once
    void buildSolver(); //build CasADi + qpOASES solver once

    // arm local compliance mapping:
    // delta_xc = S * Ka^{-1} * (wrench - wrench_ref)
    VectorXd computeArmCompliance(
        const VectorXd& wrench,
        const VectorXd& wrench_ref,
        const MatrixXd& Ka,
        const MatrixXd& S
    ) const;

    // box limit for arm compliant displacement
    VectorXd applyVectorLimit(const VectorXd& x,const VectorXd& limit)const;

    // low-pass filtering for compliance references to avoid high-frequency noise and ensure smoothness
    VectorXd firstOrderLowpass(const VectorXd& x, const VectorXd& x_prev, double alpha) const;

    // build numeric QP terms for torso compliance
    void buildTorsoComplianceQP(
        const VectorXd& delta_xc_left,
        const VectorXd& delta_xc_right,
        const MatrixXd& Jb_left,
        const MatrixXd& Jb_right,
        double dt,
        MatrixXd& H,
        VectorXd& g,
        VectorXd& lbx,
        VectorXd& ubx
    )const;

    // solve torso QP and return delta_xb
    VectorXd solveTorsoComlianceQP(
        const VectorXd& delta_xc_left,
        const VectorXd& delta_xc_right,
        const MatrixXd& Jb_left,
        const MatrixXd& Jb_right,
        double dt,);

    // optional objective evaluation for debugging    
    double computeObjective(const VectorXd& delta_xb,
        const VectorXd& g,
        const MatrixXd& H) const;

    static casadi::DM eigenToDM(const Eigen::MatrixXd& M);
    static casadi::DM eigenToDM(const Eigen::VectorXd& V);
    static Eigen::VectorXd dmToEigen(const casadi::DM& dm);

  private:
    Parameters params_;
    DebugInfo debug_;

    casadi::Function qp_solver_;

    bool initialized = false;

    VectorXd delta_xc_left_prev_ = VectorXd::Zero(6);
    VectorXd delta_xc_right_prev_ = VectorXd::Zero(6);
    VectorXd delta_xb_prev_ = VectorXd::Zero(6);
};

} // end namespace labrob

#endif // LABROB_COMPLIANCE_REFERENCE_GENERATOR_HPP_