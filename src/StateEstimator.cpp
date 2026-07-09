#include <hrp4_locomotion/StateEstimator.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <hrp4_locomotion/utils.hpp>

namespace labrob {

static pinocchio::Model buildModelFromUrdf(const std::string& urdf_path)
{
    pinocchio::Model full_model;
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), full_model);
    return pinocchio::buildReducedModel(full_model, {}, pinocchio::neutral(full_model));
}

StateEstimator::StateEstimator(const std::string& urdf_path,
                               double dt,
                               Filter filter,
                               const NoiseParams& noise)
    : StateEstimator(buildModelFromUrdf(urdf_path), dt, filter, noise)
{}

StateEstimator::StateEstimator(const pinocchio::Model& model,
                               double dt,
                               Filter filter,
                               const NoiseParams& noise)
    : model_(model), dt_(dt), filter_(filter)
{
    njnt_ = model_.nv - 6;

    Eigen::VectorXd q0 = pinocchio::neutral(model_);

    if (filter_ == Filter::RightInvariantEKF) {
        std::array<RightInvariantEKF::FootConfig, 2> feet = {{
            {"left_foot_link",  0},
            {"right_foot_link", 1}
        }};
        ri_ekf_ = std::make_unique<RightInvariantEKF>(model_, q0, dt_, feet, noise);
    } else {
        std::array<DiligentKio::FootConfig, 2> feet = {{
            {"left_foot_link",  0},
            {"right_foot_link", 1}
        }};
        diligent_kio_ = std::make_unique<DiligentKio>(model_, q0, dt_, feet, noise);
    }
}

void StateEstimator::activate(const RobotState& robot_state,
                              const Eigen::VectorXd& /*q_joints_unused*/)
{
    if (active_) return;
    active_ = true;

    // Build joint positions in pinocchio ordering from robot_state (keyed by joint name).
    Eigen::VectorXd q_joints_pin(njnt_);
    for (int i = 0; i < njnt_; ++i)
        q_joints_pin(i) = robot_state.joint_state[model_.names[i + 2]].pos;

    if (filter_ == Filter::RightInvariantEKF) {
        // Reconstruct initial base pose from FK + known contact geometry:
        // both feet are at Z=0 with RPY=0 at t=0, so T_world_foot = I.
        // T_world_base = T_world_foot * FK(q_joints)^{-1} = FK^{-1}
        //   R_world_base = R_foot_in_base^T
        //   p_world_base = -R_world_base * p_foot_in_base   (Z: foot at 0)
        pinocchio::Data data_init(model_);
        Eigen::VectorXd q_fk = pinocchio::neutral(model_);
        q_fk.tail(njnt_) = q_joints_pin;
        pinocchio::forwardKinematics(model_, data_init, q_fk);
        pinocchio::updateFramePlacements(model_, data_init);

        auto fk_foot = [&](const std::string& frame) -> std::pair<Eigen::Matrix3d, Eigen::Vector3d> {
            const auto& T = data_init.oMf[model_.getFrameId(frame)];
            Eigen::Matrix3d R_wb = T.rotation().transpose();
            Eigen::Vector3d p_wb = -R_wb * T.translation();  // T_world_foot.t=0
            return {R_wb, p_wb};
        };

        auto [R_l, p_l] = fk_foot("left_foot_link");
        auto [R_r, p_r] = fk_foot("right_foot_link");

        // Average height from both contacts; project averaged rotation to SO(3)
        Eigen::Vector3d p_wb(0.0, 0.0, 0.5 * (p_l.z() + p_r.z()));
        Eigen::Matrix3d R_avg = 0.5 * (R_l + R_r);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_avg, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d R_wb = svd.matrixU()
                             * Eigen::DiagonalMatrix<double,3>(1, 1, (svd.matrixU()*svd.matrixV().transpose()).determinant())
                             * svd.matrixV().transpose();

        Eigen::Quaterniond q_wb(R_wb);
        Eigen::VectorXd q_init = pinocchio::neutral(model_);
        q_init.head<3>()       = p_wb;
        q_init.segment<4>(3)   = q_wb.coeffs();   // (x,y,z,w)
        q_init.tail(njnt_)     = q_joints_pin;

        ri_ekf_->initialize(q_init, q_joints_pin);
    } else {
        diligent_kio_->addContact(0, q_joints_pin);
        diligent_kio_->addContact(1, q_joints_pin);
    }
}

void StateEstimator::update(RobotState& robot_state,
                            const Eigen::Vector3d& gyro,
                            const Eigen::Vector3d& acc,
                            const std::array<bool,2>& contact)
{
    if (!active_) return;

    Eigen::VectorXd jnt_pos(njnt_);
    Eigen::VectorXd jnt_vel(njnt_);
    for (int i = 0; i < njnt_; ++i) {
        const std::string& name = model_.names[i + 2];
        jnt_pos(i) = robot_state.joint_state[name].pos;
        jnt_vel(i) = robot_state.joint_state[name].vel;
    }

    if (filter_ == Filter::RightInvariantEKF) {
        ri_ekf_->filter(gyro, acc, jnt_pos, jnt_vel, contact);
        robot_state.position         = ri_ekf_->getPosition();
        robot_state.orientation      = ri_ekf_->getQuaternion();
        // EKF tracks velocity in world frame; Pinocchio free-flyer expects body frame.
        robot_state.linear_velocity  = ri_ekf_->getQuaternion().toRotationMatrix().transpose()
                                       * ri_ekf_->getVelocity();
        robot_state.angular_velocity = ri_ekf_->getOmegaBody();
    } else {
        diligent_kio_->filter(gyro, acc, jnt_pos, jnt_vel, contact);
        robot_state.position         = diligent_kio_->getPosition();
        robot_state.orientation      = diligent_kio_->getQuaternion();
        robot_state.linear_velocity  = diligent_kio_->getVelocity();
        robot_state.angular_velocity = diligent_kio_->getOmegaBody();
    }
}

} // namespace labrob